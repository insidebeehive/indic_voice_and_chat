"""DB-backed tenant resolver — replaces the YAML-on-boot in-memory resolver.

Tenants now live in the database (`tenants` + `tenant_phone_numbers` +
`tenant_api_keys` + `tenant_secrets`). This resolver loads them all into an
in-memory cache at startup (same lookup shape as the old in-memory resolver, so
every caller is unchanged) and rebuilds a `TenantContext` per tenant:

- `TenantSettings` is reconstructed from the row + its `pipeline_config` JSON.
- The per-tenant **telephony** secrets are decrypted and attached so
  `TenantContext.secret(name)` returns them; non-telephony keys fall back to the
  shared master env.

`refresh(tenant_id)` reloads one tenant after Register Tenant / updates.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.auth import secrets as secret_crypto
from src.auth.context import TenantContext
from src.config_tenant import (
    ChatSupportConfig,
    DepositVerificationConfig,
    TenantCompliance,
    TenantCRMConfig,
    TenantPipelineConfig,
    TenantSettings,
)
from src.models.tenant import Tenant

log = logging.getLogger(__name__)


def tenant_context_from_row(
    tenant: Tenant,
    crm_prompt_pack: Optional[str] = None,
    crm_pronunciation_overrides: Optional[dict] = None,
) -> TenantContext:
    """Build a TenantContext from a fully-loaded Tenant ORM row.

    The row's relationships (`phone_numbers`, `secrets`) must be eager-loaded.
    ``crm_prompt_pack`` is the linked Crm row's ``prompt_pack`` value (looked up
    by the caller, since ``Tenant`` has no ORM relationship to `Crm`) — falls
    back to "generic" when unset/NULL or when the tenant has no linked CRM.
    ``crm_pronunciation_overrides`` is the linked Crm row's
    ``pronunciation_overrides`` value, same lookup — None when unset/NULL or
    when the tenant has no linked CRM (that CRM's TTS then uses only the
    generic DEFAULT_PRONUNCIATIONS default, no extra terms).
    """
    pc = tenant.pipeline_config or {}
    pipeline = TenantPipelineConfig(**pc)
    # compliance rides inside pipeline_config (seeded there); pull it back out so a
    # tenant's calling-hours / DND config is honored, not the defaults.
    compliance = TenantCompliance(**(pc.get("compliance") or {}))
    crm = TenantCRMConfig(**(pc.get("crm") or {}))
    chat_support = ChatSupportConfig(**(pc.get("chat_support") or {}))
    deposit_verification = DepositVerificationConfig(**(pc.get("deposit_verification") or {}))
    # events_webhook_url lives at the top level of pipeline_config (not under telephony).
    # Fall back to the old telephony location for rows written before this change.
    tel_pc = pc.get("telephony") or {}
    events_webhook_url = pc.get("events_webhook_url") or tel_pc.get("events_webhook_url")
    events_webhook_secret_env = (
        pc.get("events_webhook_secret_env") or tel_pc.get("events_webhook_secret_env")
    )
    settings = TenantSettings(
        id=tenant.id,
        slug=tenant.slug,
        name=tenant.name,
        status=tenant.status,
        default_language=tenant.default_language,
        timezone=tenant.timezone,
        max_concurrent_calls=tenant.max_concurrent_calls,
        events_webhook_url=events_webhook_url,
        events_webhook_secret_env=events_webhook_secret_env,
        pipeline=pipeline,
        compliance=compliance,
        crm=crm,
        crm_id=tenant.crm_id,
        prompt_pack=crm_prompt_pack or "generic",
        pronunciation_overrides=crm_pronunciation_overrides or None,
        chat_support=chat_support,
        deposit_verification=deposit_verification,
        phone_numbers=[p.phone_number for p in tenant.phone_numbers],
    )
    resolved: dict[str, str] = {}
    for s in tenant.secrets:
        try:
            resolved[s.name] = secret_crypto.decrypt(s.value_encrypted)
        except secret_crypto.SecretsError:
            log.exception("failed to decrypt tenant secret", extra={
                "tenant": tenant.slug, "name": s.name})
    return TenantContext(settings=settings, secrets_resolved=resolved)


class DbTenantResolver:
    """Loads tenants from the DB into an in-memory cache; resolves by token/slug/phone."""

    def __init__(self, sessionmaker) -> None:
        self._sm = sessionmaker
        self._by_token: dict[str, TenantContext] = {}
        self._by_slug: dict[str, TenantContext] = {}
        self._by_phone: dict[str, TenantContext] = {}
        self._by_id: dict[str, TenantContext] = {}
        self._by_chatwoot_inbox: dict[str, TenantContext] = {}
        self._by_stringee_webhook_token: dict[str, TenantContext] = {}
        self._by_chatwoot_webhook_id: dict[str, TenantContext] = {}
        self._by_deposit_verification_reply_token: dict[str, TenantContext] = {}
        # Optional callback fired after a (re)load so cached per-tenant provider
        # clients can be evicted (e.g. providers.evict) — otherwise a key/config
        # update leaves stale clients behind.
        self.on_reload: Optional[callable] = None

    async def reload(self) -> int:
        """(Re)load every tenant from the DB. Returns the count loaded."""
        from src.models.crm import Crm

        async with self._sm() as session:
            rows = (await session.execute(
                select(Tenant).options(
                    selectinload(Tenant.phone_numbers),
                    selectinload(Tenant.api_keys),
                    selectinload(Tenant.secrets),
                )
            )).scalars().all()
            # crm_id -> prompt_pack, denormalized onto each linked tenant's
            # settings below (mirrors the crm_id denormalization itself) —
            # one query for all CRMs rather than one per tenant.
            crm_prompt_pack_by_id = dict(
                (await session.execute(select(Crm.id, Crm.prompt_pack))).all()
            )
            # Same denormalization for pronunciation_overrides (TTS).
            crm_pronunciation_overrides_by_id = dict(
                (await session.execute(select(Crm.id, Crm.pronunciation_overrides))).all()
            )
            by_token, by_slug, by_phone, by_id, by_cw_inbox = {}, {}, {}, {}, {}
            by_stringee_webhook_token, by_cw_webhook_id = {}, {}
            by_dv_reply_token = {}
            for t in rows:
                ctx = tenant_context_from_row(
                    t,
                    crm_prompt_pack=crm_prompt_pack_by_id.get(t.crm_id),
                    crm_pronunciation_overrides=crm_pronunciation_overrides_by_id.get(t.crm_id),
                )
                by_slug[t.slug] = ctx
                by_id[t.id] = ctx
                for k in t.api_keys:
                    by_token[k.token_hash] = ctx
                for p in t.phone_numbers:
                    by_phone[p.phone_number] = ctx
                inbox_id = ctx.secrets_resolved.get("chatwoot:inbox_id")
                if inbox_id:
                    by_cw_inbox[str(inbox_id)] = ctx
                stringee_webhook_token = ctx.secrets_resolved.get("webhook:stringee_path_token")
                if stringee_webhook_token:
                    by_stringee_webhook_token[str(stringee_webhook_token)] = ctx
                chatwoot_webhook_id = ctx.secrets_resolved.get("chatwoot:webhook_id")
                if chatwoot_webhook_id:
                    by_cw_webhook_id[str(chatwoot_webhook_id)] = ctx
                dv_reply_token = ctx.secrets_resolved.get("deposit_verification:reply_token")
                if dv_reply_token:
                    by_dv_reply_token[str(dv_reply_token)] = ctx
            self._by_token, self._by_slug, self._by_phone = by_token, by_slug, by_phone
            self._by_id = by_id
            self._by_chatwoot_inbox = by_cw_inbox
            self._by_stringee_webhook_token = by_stringee_webhook_token
            self._by_chatwoot_webhook_id = by_cw_webhook_id
            self._by_deposit_verification_reply_token = by_dv_reply_token
        log.info("tenant resolver loaded from DB", extra={"count": len(self._by_slug)})
        if self.on_reload is not None:
            self.on_reload()   # drop stale per-tenant provider clients
        return len(self._by_slug)

    async def refresh(self, tenant_id: str) -> None:
        """Reload one tenant (after register/update). Simplest correct impl: full reload."""
        await self.reload()

    def loaded_settings(self) -> dict:
        """{slug -> TenantSettings} for the currently-cached tenants (for /health)."""
        return {slug: ctx.settings for slug, ctx in self._by_slug.items()}

    async def resolve_by_token(self, token_hash: str) -> Optional[TenantContext]:
        return self._by_token.get(token_hash)

    async def resolve_by_slug(self, slug: str) -> Optional[TenantContext]:
        return self._by_slug.get(slug)

    async def resolve_by_phone_number(self, phone_number: str) -> Optional[TenantContext]:
        return self._by_phone.get(phone_number)

    async def resolve_by_id(self, tenant_id: str) -> Optional[TenantContext]:
        return self._by_id.get(tenant_id)

    async def resolve_by_chatwoot_inbox(self, inbox_id: str) -> Optional[TenantContext]:
        return self._by_chatwoot_inbox.get(str(inbox_id))

    async def resolve_by_stringee_webhook_token(self, token: str) -> Optional[TenantContext]:
        return self._by_stringee_webhook_token.get(str(token))

    async def resolve_by_chatwoot_webhook_id(self, webhook_id: str) -> Optional[TenantContext]:
        return self._by_chatwoot_webhook_id.get(str(webhook_id))

    async def resolve_by_deposit_verification_reply_token(
        self, token: str
    ) -> Optional[TenantContext]:
        return self._by_deposit_verification_reply_token.get(str(token))
