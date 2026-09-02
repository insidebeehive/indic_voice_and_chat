"""Register Tenant endpoint (``POST /api/v1/tenants``) — admin-authed.

A CRM/partner self-registers a tenant through this API instead of dropping a
YAML file. The body carries **provider choices** for STT/LLM/TTS/realtime (which
use the shared master keys — no keys accepted here) and the **telephony**
credentials (the only per-tenant secrets — encrypted at rest into
``tenant_secrets``). We build the same ``TenantPipelineConfig`` the YAML path
produced, persist the tenant + phone numbers + telephony secrets, issue one API
token (returned once, stored only as a hash), and refresh the live resolver.
"""

from __future__ import annotations

import logging
import re
import secrets as pysecrets
import uuid
from types import SimpleNamespace
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.answer_paths import answer_url_for
from src.api.deps import get_db_session
from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.middleware import require_admin
from src.config_tenant import (
    TelephonyCreds,
    TenantPipelineConfig,
    TenantRealtimeConfig,
    TenantSTTConfig,
    TenantTelephonyConfig,
    TenantTTSConfig,
    platform_webhook_base_url,
)
from src.config_tenant import TenantLLMConfig as _LLM
from src.dialogue.campaign_loader import parse_campaign_yaml
from src.models.campaign import Campaign
from src.models.conversation import Conversation
from src.models.tenant import ProviderCost, Tenant, TenantApiKey, TenantPhoneNumber, TenantSecret

log = logging.getLogger(__name__)
router = APIRouter(prefix="/tenants", tags=["tenants"])

# STT/LLM/TTS/realtime always resolve their key from the platform master env
# vars (e.g. GEMINI_API_KEY, via TenantContext.secret's fallback to
# os.environ) — never stored per tenant. Telephony keys are the per-tenant
# exception (see _map_telephony_keys below).


# --- Schemas ------------------------------------------------------------


class LayerChoice(BaseModel):
    """Provider choice for one cascade layer — no keys (uses master keys)."""
    provider: str
    model: Optional[str] = None
    language: Optional[str] = None
    voice_id: Optional[str] = None  # tts only
    speed: Optional[float] = None   # tts only


class RealtimeChoice(BaseModel):
    provider: str
    model: Optional[str] = None
    voice: Optional[str] = None
    language_code: Optional[str] = None


class TelephonyConfigIn(BaseModel):
    provider: str
    from_number: Optional[str] = None
    stringee_base_url: Optional[str] = None   # regional Stringee REST host, if any
    livekit_url: Optional[str] = None         # LiveKit server/project URL (wss://...), if any
    # No per-tenant inbound webhook base URL — it's the platform WEBHOOK_BASE_URL
    # (always our app, common to every tenant).
    # Telephony credentials — the ONLY per-tenant secrets. Encrypted at rest.
    # e.g. {"account_sid": "AC...", "auth_token": "..."}. Optional for providers
    # (Stringee) whose adapter reads its keys from the platform env directly.
    keys: dict[str, str] = Field(default_factory=dict)
    phone_numbers: list[str] = Field(default_factory=list)


class RegisterTenantRequest(BaseModel):
    name: str = Field(min_length=1)
    slug: Optional[str] = None
    timezone: str = "Asia/Kolkata"
    default_language: str = "hi"
    mode: str = Field(default="s2s", pattern="^(layered|s2s)$")
    max_concurrent_calls: int = Field(default=10, ge=1)
    stt: Optional[LayerChoice] = None
    llm: Optional[LayerChoice] = None
    tts: Optional[LayerChoice] = None
    realtime: Optional[RealtimeChoice] = None
    telephony: TelephonyConfigIn = Field(default_factory=lambda: TelephonyConfigIn(provider="none"))
    # Outbound event webhook — receives call + chat handover lifecycle events,
    # signed when events_webhook_secret is provided (pass as keys["events_webhook_secret"]).
    events_webhook_url: Optional[str] = None
    # CRM operator ID — the CRM system's identifier for this operator/tenant.
    # Injected into every CRM tool call as "operator_id" so the CRM can scope
    # responses. Defaults to the platform's tenant ID if not provided.
    crm_operator_id: Optional[str] = None
    # Which Crm entity row this tenant links to (the real Tenant.crm_id FK).
    # Unrelated to crm_operator_id above — that's an opaque identifier string
    # injected into tool calls, this is which shared CRM (tools/KB) the tenant
    # is attached to. Don't conflate the two.
    crm_id: Optional[str] = None


class RegisterTenantResponse(BaseModel):
    tenant_id: str
    slug: str
    api_token: str


class RotateTokenResponse(BaseModel):
    tenant_id: str
    api_token: str


# --- Helpers ------------------------------------------------------------


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or f"tenant-{uuid.uuid4().hex[:8]}"


# Telephony credential logical-name → the TenantTelephonyConfig *_env field it sets.
# account_sid/auth_token drive server dialing; the api_key_* / twiml_app_sid drive
# the browser softphone (Twilio). Stringee reuses account_sid/auth_token.
_TEL_KEY_ENV_FIELD = {
    "account_sid": "account_sid_env", "sid": "account_sid_env",
    "auth_token": "auth_token_env", "token": "auth_token_env",
    "api_key_sid": "api_key_sid_env", "twilio_api_key_sid": "api_key_sid_env",
    "api_key_secret": "api_key_secret_env", "twilio_api_key_secret": "api_key_secret_env",
    "twiml_app_sid": "twiml_app_sid_env",
    # LiveKit reuses the generic account_sid_env/auth_token_env slots for its
    # API key/secret pair — these are purely UI-facing aliases so the backoffice
    # can label the fields with LiveKit-appropriate names; no new secret fields.
    "livekit_api_key": "account_sid_env", "livekit_api_secret": "auth_token_env",
    # Stringee outbound app-user id — a non-null userId keeps the callout an
    # app-user->phone call so the Answer URL/SCCO runs (else silent phone->phone).
    "user_id": "user_id_env", "stringee_user_id": "user_id_env",
    # The HMAC secret we sign outbound call-event webhooks with — entered as a
    # VALUE, encrypted, and referenced by events_webhook_secret_env.
    "events_webhook_secret": "events_webhook_secret_env",
}


def _map_telephony_keys(slug: str, keys: dict[str, str]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Map provider key names to encrypted secret rows + the *_env fields they set.

    Returns ``(secret_rows, env_fields)`` where ``secret_rows`` is ``[(env_name,
    value)]`` to encrypt into ``tenant_secrets`` and ``env_fields`` is
    ``{TenantTelephonyConfig field: env_name}`` (e.g. ``{"account_sid_env": "..."}``).
    """
    secret_rows: list[tuple[str, str]] = []
    env_fields: dict[str, str] = {}
    for logical, value in keys.items():
        name = f"TENANT_{slug.upper().replace('-', '_')}_{logical.upper()}"
        secret_rows.append((name, value))
        field = _TEL_KEY_ENV_FIELD.get(logical)
        if field:
            env_fields[field] = name
    return secret_rows, env_fields


# --- Route --------------------------------------------------------------


@router.post("", response_model=RegisterTenantResponse, status_code=201)
async def register_tenant(
    req: RegisterTenantRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> RegisterTenantResponse:
    slug = req.slug or _slugify(req.name)
    existing = (await session.execute(
        select(Tenant.id).where(Tenant.slug == slug)
    )).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"tenant slug {slug!r} already exists")

    tenant_id = f"t_{uuid.uuid4().hex[:16]}"

    # Telephony secrets keyed by synthetic names that pipeline_config references;
    # TenantContext.secret(<name>) finds them in the decrypted secrets dict.
    tel = req.telephony
    if tel.keys and not crypto.has_key():
        raise HTTPException(
            status_code=503,
            detail="VOX_SECRET_KEY is not set — cannot encrypt telephony keys",
        )
    secret_rows, env_fields = _map_telephony_keys(slug, tel.keys)

    pipeline = TenantPipelineConfig(
        mode=req.mode,
        stt=TenantSTTConfig(
            provider=req.stt.provider if req.stt else None,
            model=req.stt.model if req.stt else None,
            language=req.stt.language if req.stt else None,
        ),
        llm=_LLM(
            provider=req.llm.provider if req.llm else None,
            model=req.llm.model if req.llm else None,
        ),
        tts=TenantTTSConfig(
            provider=req.tts.provider if req.tts else None,
            model=req.tts.model if req.tts else None,
            language=req.tts.language if req.tts else None,
            voice_id=req.tts.voice_id if req.tts else None,
            speed=req.tts.speed if req.tts else None,
        ),
        realtime=TenantRealtimeConfig(
            provider=req.realtime.provider,
            model=req.realtime.model,
            voice=req.realtime.voice,
            language_code=req.realtime.language_code,
        ) if req.realtime else None,
        telephony=TenantTelephonyConfig(
            provider=tel.provider,
            from_number=tel.from_number,
            stringee_base_url=tel.stringee_base_url,
            livekit_url=tel.livekit_url,
            **{k: v for k, v in env_fields.items() if k != "events_webhook_secret_env"},
            # also record the creds under the provider-specific slot so a tenant
            # that later adds a second provider's keys resolves each correctly.
            creds_by_provider=(
                {tel.provider.lower(): TelephonyCreds(
                    **{k: v for k, v in env_fields.items() if k != "events_webhook_secret_env"}
                )}
                if env_fields and tel.provider else {}
            ),
        ),
    )

    pipeline_config = pipeline.model_dump()
    # events_webhook fields live at the top level of pipeline_config (not under telephony)
    # so they apply to both call events and chat handover events.
    if req.events_webhook_url:
        pipeline_config["events_webhook_url"] = req.events_webhook_url
    if env_fields.get("events_webhook_secret_env"):
        pipeline_config["events_webhook_secret_env"] = env_fields["events_webhook_secret_env"]
    if req.crm_operator_id:
        pipeline_config["crm"] = {"operator_id": req.crm_operator_id}
    session.add(Tenant(
        id=tenant_id, slug=slug, name=req.name, status="active",
        timezone=req.timezone, default_language=req.default_language,
        mode=req.mode, max_concurrent_calls=req.max_concurrent_calls,
        pipeline_config=pipeline_config,
        crm_id=req.crm_id,
    ))
    for ph in tel.phone_numbers:
        session.add(TenantPhoneNumber(
            phone_number=ph, tenant_id=tenant_id, provider=tel.provider))
    for name, value in secret_rows:
        session.add(TenantSecret(
            tenant_id=tenant_id, name=name, value_encrypted=crypto.encrypt(value)))

    api_token = f"vox_{pysecrets.token_urlsafe(32)}"
    session.add(TenantApiKey(
        token_hash=hash_api_token(api_token), tenant_id=tenant_id, label="register"))

    await session.commit()

    # Refresh the live resolver so the new tenant resolves immediately.
    resolver = getattr(request.app.state, "tenant_resolver", None)
    if resolver is not None and hasattr(resolver, "refresh"):
        await resolver.refresh(tenant_id)
        if hasattr(request.app.state, "tenants"):
            request.app.state.tenants = resolver.loaded_settings()

    log.info("registered tenant", extra={"tenant_id": tenant_id, "slug": slug})
    return RegisterTenantResponse(tenant_id=tenant_id, slug=slug, api_token=api_token)


# --- Update tenant credentials/config (admin) ---------------------------


class TelephonyUpdateIn(BaseModel):
    """Partial telephony update — only the provided fields change. ``keys`` are
    merged (upserted) into the tenant's existing encrypted secrets."""
    provider: Optional[str] = None
    from_number: Optional[str] = None
    stringee_base_url: Optional[str] = None
    livekit_url: Optional[str] = None
    keys: dict[str, str] = Field(default_factory=dict)
    phone_numbers: Optional[list[str]] = None


class ChatwootUpdateIn(BaseModel):
    """Chatwoot Agent Bot credentials — stored encrypted as TenantSecrets."""
    api_url: Optional[str] = None          # defaults to https://app.chatwoot.com
    account_id: Optional[str] = None
    api_token: Optional[str] = None        # write-only; never returned
    inbox_id: Optional[str] = None         # Chatwoot inbox ID for webhook tenant lookup (no bearer needed)


class CrmCredentialsIn(BaseModel):
    """Platform-catalog CRM credentials — stored as TenantSecrets / pipeline_config."""
    base_url: Optional[str] = None       # per-tenant override; falls back to PLATFORM_CRM_BASE_URL env
    auth_type: Optional[str] = None      # api_key | bearer  (default: api_key)
    api_token: Optional[str] = None      # write-only; never returned
    operator_id: Optional[str] = None    # stored in pipeline_config.crm.operator_id
    # Independent, additive secret: the live CRM requires BOTH an
    # Authorization header (from auth_type/api_token above) AND a separate
    # X-API-Key header — this is that second header's value, sent
    # unconditionally alongside whatever auth_type/api_token already produce.
    x_api_key: Optional[str] = None      # write-only; never returned


class DepositVerificationUpdateIn(BaseModel):
    """Deposit dispute screenshot verification webhook config — partial update."""
    enabled: Optional[bool] = None
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None   # write-only plaintext; never returned
    timeout_minutes: Optional[int] = None


class UpdateTenantRequest(BaseModel):
    status: Optional[str] = Field(default=None, pattern="^(active|suspended)$")
    events_webhook_url: Optional[str] = None
    telephony: Optional[TelephonyUpdateIn] = None
    chatwoot: Optional[ChatwootUpdateIn] = None
    crm: Optional[CrmCredentialsIn] = None
    # Links this tenant to a Crm row (real Tenant.crm_id FK column, Task 1).
    # "" clears the link (FK is nullable); omitted/None leaves it untouched.
    crm_id: Optional[str] = None
    deposit_verification: Optional[DepositVerificationUpdateIn] = None


class UpdateTenantResponse(BaseModel):
    tenant_id: str
    slug: str
    status: str
    telephony_provider: Optional[str] = None
    # Which telephony credential slots are now configured (names only, never
    # values) — lets an admin confirm e.g. the Stringee account_sid is set.
    telephony_creds_configured: list[str]
    events_webhook_url: Optional[str] = None
    events_webhook_secret_set: bool = False
    crm_id: Optional[str] = None
    deposit_verification_enabled: Optional[bool] = None
    deposit_verification_secret_set: Optional[bool] = None


async def _refresh_resolver(request: Request, tenant_id: str) -> None:
    """Reload one tenant into the live resolver + drop its cached provider clients."""
    resolver = getattr(request.app.state, "tenant_resolver", None)
    if resolver is not None and hasattr(resolver, "refresh"):
        await resolver.refresh(tenant_id)
        if hasattr(request.app.state, "tenants"):
            request.app.state.tenants = resolver.loaded_settings()
    providers = getattr(request.app.state, "providers", None)
    if providers is not None and hasattr(providers, "evict"):
        providers.evict(tenant_id)


@router.patch("/{tenant_id}", response_model=UpdateTenantResponse)
async def update_tenant(
    tenant_id: str,
    req: UpdateTenantRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> UpdateTenantResponse:
    """Update an existing tenant's telephony credentials/config or status (admin).

    Telephony ``keys`` are encrypted and **merged** into the tenant's secrets
    (existing names are overwritten, new ones added), and the matching ``*_env``
    references are written into ``pipeline_config.telephony``. The live resolver
    is refreshed so the change takes effect immediately — e.g. fixing a wrong
    Stringee API Key SID without re-registering the tenant.
    """
    t = await _require_tenant(session, tenant_id)

    if req.status is not None:
        t.status = req.status

    if req.crm_id is not None:
        # "" (the "— none —" dropdown option) clears the link; a real id sets it.
        t.crm_id = req.crm_id or None

    pc = dict(t.pipeline_config or {})

    if req.events_webhook_url is not None:
        pc["events_webhook_url"] = req.events_webhook_url

    if req.telephony is not None:
        tu = req.telephony
        tel_cfg = dict(pc.get("telephony") or {})
        if tu.provider is not None:
            tel_cfg["provider"] = tu.provider
        if tu.from_number is not None:
            tel_cfg["from_number"] = tu.from_number
        if tu.stringee_base_url is not None:
            tel_cfg["stringee_base_url"] = tu.stringee_base_url
        if tu.livekit_url is not None:
            tel_cfg["livekit_url"] = tu.livekit_url

        if tu.keys:
            if not crypto.has_key():
                raise HTTPException(
                    status_code=503,
                    detail="VOX_SECRET_KEY is not set — cannot encrypt telephony keys")
            secret_rows, env_fields = _map_telephony_keys(t.slug, tu.keys)
            for name, value in secret_rows:
                existing = (await session.execute(
                    select(TenantSecret).where(
                        TenantSecret.tenant_id == tenant_id, TenantSecret.name == name)
                )).scalar_one_or_none()
                if existing is not None:
                    existing.value_encrypted = crypto.encrypt(value)
                else:
                    session.add(TenantSecret(
                        tenant_id=tenant_id, name=name,
                        value_encrypted=crypto.encrypt(value)))
            # events_webhook_secret_env goes to top-level; telephony creds go under telephony.
            if "events_webhook_secret_env" in env_fields:
                pc["events_webhook_secret_env"] = env_fields.pop("events_webhook_secret_env")
            # write the creds into the configured provider's slot (so they're
            # selected by provider) and mirror to top-level for back-compat.
            prov = (tel_cfg.get("provider") or "").lower()
            cbp = dict(tel_cfg.get("creds_by_provider") or {})
            slot = dict(cbp.get(prov) or {})
            slot.update(env_fields)
            if prov:
                cbp[prov] = slot
                tel_cfg["creds_by_provider"] = cbp
            tel_cfg.update(env_fields)

        pc["telephony"] = tel_cfg

        if tu.phone_numbers is not None:
            for pn in (await session.execute(
                select(TenantPhoneNumber).where(TenantPhoneNumber.tenant_id == tenant_id)
            )).scalars().all():
                await session.delete(pn)
            for ph in tu.phone_numbers:
                session.add(TenantPhoneNumber(
                    phone_number=ph, tenant_id=tenant_id, provider=tel_cfg.get("provider")))

    if req.chatwoot is not None:
        cw = req.chatwoot
        if not crypto.has_key():
            raise HTTPException(
                status_code=503,
                detail="VOX_SECRET_KEY is not set — cannot encrypt Chatwoot credentials")
        cw_secrets = {}
        if cw.api_url is not None:
            cw_secrets["chatwoot:api_url"] = cw.api_url
        if cw.account_id is not None:
            cw_secrets["chatwoot:account_id"] = cw.account_id
        if cw.api_token is not None:
            cw_secrets["chatwoot:api_token"] = cw.api_token
        if cw.inbox_id is not None:
            cw_secrets["chatwoot:inbox_id"] = cw.inbox_id
        for name, value in cw_secrets.items():
            existing = (await session.execute(
                select(TenantSecret).where(
                    TenantSecret.tenant_id == tenant_id, TenantSecret.name == name)
            )).scalar_one_or_none()
            if existing is not None:
                existing.value_encrypted = crypto.encrypt(value)
            else:
                session.add(TenantSecret(
                    tenant_id=tenant_id, name=name,
                    value_encrypted=crypto.encrypt(value)))

    if req.crm is not None:
        crm = req.crm
        if not crypto.has_key():
            raise HTTPException(
                status_code=503,
                detail="VOX_SECRET_KEY is not set — cannot encrypt CRM credentials")
        crm_secrets = {}
        if crm.base_url is not None:
            crm_secrets["crm:base_url"] = crm.base_url
        if crm.auth_type is not None:
            crm_secrets["crm:auth_type"] = crm.auth_type
        if crm.api_token is not None:
            crm_secrets["crm:api_token"] = crm.api_token
        if crm.x_api_key is not None:
            crm_secrets["crm:x_api_key"] = crm.x_api_key
        for name, value in crm_secrets.items():
            existing = (await session.execute(
                select(TenantSecret).where(
                    TenantSecret.tenant_id == tenant_id, TenantSecret.name == name)
            )).scalar_one_or_none()
            if existing is not None:
                existing.value_encrypted = crypto.encrypt(value)
            else:
                session.add(TenantSecret(
                    tenant_id=tenant_id, name=name,
                    value_encrypted=crypto.encrypt(value)))
        if crm.operator_id is not None:
            crm_cfg = dict(pc.get("crm") or {})
            crm_cfg["operator_id"] = crm.operator_id
            pc["crm"] = crm_cfg

    if req.deposit_verification is not None:
        dv = req.deposit_verification
        dv_cfg = dict(pc.get("deposit_verification") or {})
        if dv.enabled is not None:
            dv_cfg["enabled"] = dv.enabled
        if dv.webhook_url is not None:
            dv_cfg["webhook_url"] = dv.webhook_url
        if dv.timeout_minutes is not None:
            dv_cfg["timeout_minutes"] = dv.timeout_minutes
        if dv.webhook_secret is not None:
            if not crypto.has_key():
                raise HTTPException(
                    status_code=503,
                    detail="VOX_SECRET_KEY is not set — cannot encrypt deposit verification webhook secret")
            name = f"TENANT_{t.slug.upper().replace('-', '_')}_DEPOSIT_VERIFICATION_WEBHOOK_SECRET"
            existing = (await session.execute(
                select(TenantSecret).where(
                    TenantSecret.tenant_id == tenant_id, TenantSecret.name == name)
            )).scalar_one_or_none()
            if existing is not None:
                existing.value_encrypted = crypto.encrypt(dv.webhook_secret)
            else:
                session.add(TenantSecret(
                    tenant_id=tenant_id, name=name,
                    value_encrypted=crypto.encrypt(dv.webhook_secret)))
            dv_cfg["webhook_secret_env"] = name
        pc["deposit_verification"] = dv_cfg

    if req.status is not None or req.events_webhook_url is not None \
            or req.telephony is not None or req.chatwoot is not None \
            or req.crm is not None or req.deposit_verification is not None:
        t.pipeline_config = pc  # reassign (new object) so the JSON column is marked dirty

    await session.commit()
    await _refresh_resolver(request, tenant_id)
    log.info("updated tenant", extra={"tenant_id": tenant_id})

    pc = t.pipeline_config or {}
    tel_cfg = pc.get("telephony") or {}
    dv_cfg = pc.get("deposit_verification") or {}
    return UpdateTenantResponse(
        tenant_id=t.id, slug=t.slug, status=t.status,
        telephony_provider=tel_cfg.get("provider"),
        telephony_creds_configured=_configured_creds(tel_cfg),
        events_webhook_url=pc.get("events_webhook_url"),
        events_webhook_secret_set=bool(pc.get("events_webhook_secret_env")),
        crm_id=t.crm_id,
        deposit_verification_enabled=dv_cfg.get("enabled"),
        deposit_verification_secret_set=bool(dv_cfg.get("webhook_secret_env")),
    )


@router.post("/{tenant_id}/rotate-token", response_model=RotateTokenResponse)
async def rotate_tenant_token(
    tenant_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> RotateTokenResponse:
    """Rotate a tenant's inbound API bearer token (admin).

    Deletes every existing ``TenantApiKey`` row for this tenant and issues one
    new ``vox_...`` token — the old token(s) stop authenticating immediately.
    The plaintext is returned once, never stored. This is the tenant's
    **inbound** token (external callers, e.g. the CRM, authenticating INTO our
    API) — unrelated to the CRM's own outbound auth secrets in
    ``TenantSecret``, which this does not touch.
    """
    t = await _require_tenant(session, tenant_id)

    existing_keys = (await session.execute(
        select(TenantApiKey).where(TenantApiKey.tenant_id == tenant_id)
    )).scalars().all()
    for key in existing_keys:
        await session.delete(key)
    # Flush the deletes before adding the new row — the new row reuses
    # label="rotated", which collides with uq_tenant_api_key_label against a
    # still-present old "rotated" row (a second rotation) unless the delete
    # has actually hit the DB first; SQLAlchemy's unit of work does not
    # guarantee DELETE-before-INSERT ordering within one flush otherwise.
    await session.flush()

    new_token = f"vox_{pysecrets.token_urlsafe(32)}"
    session.add(TenantApiKey(
        token_hash=hash_api_token(new_token), tenant_id=tenant_id, label="rotated"))

    await session.commit()
    try:
        await _refresh_resolver(request, tenant_id)
    except Exception:  # noqa: BLE001 — the DB is already rotated; the admin
        # still needs this plaintext token even if the in-memory reload
        # failed (it'll pick up the new token on the next natural reload
        # regardless). Losing the token here would mean re-doing the DB
        # surgery by hand since it's never stored anywhere.
        log.exception("resolver refresh failed after token rotation", extra={"tenant_id": tenant_id})

    log.info("rotated tenant api token", extra={"tenant_id": tenant_id})
    return RotateTokenResponse(tenant_id=t.id, api_token=new_token)


# --- Webhook credentials rotation (admin) --------------------------------


class WebhookCredentialsRotateRequest(BaseModel):
    """Mint fresh per-tenant webhook-verification credentials for one or more
    providers. Twilio needs nothing here — it already reuses the tenant's
    existing Twilio auth token secret for signature verification."""
    providers: list[Literal["stringee", "exotel", "chatwoot"]] = Field(min_length=1)


class WebhookCredentialsRotateResponse(BaseModel):
    tenant_id: str
    slug: str
    rotated: list[str]
    # Plaintext minted values, returned ONCE only — never persisted anywhere
    # except encrypted (via crypto.encrypt) in tenant_secrets.value_encrypted.
    credentials: dict[str, str]
    # Human-readable, ready-to-paste guidance per rotated provider.
    instructions: dict[str, str]


def _chatwoot_integrations_base_url(telephony_base: str) -> str:
    """Derive the ``/api/v1/integrations`` base URL from the telephony webhook
    base (``platform_webhook_base_url()``, which already includes
    ``/api/v1/telephony`` — see ``config/default.yaml``). The Chatwoot webhook
    route is registered under a sibling router prefix, ``/api/v1/integrations``
    (``external_chat.router``, mounted in ``src/api/__init__.py``), not under
    ``/api/v1/telephony`` — so it is NOT safe to reuse ``telephony_base`` as-is.
    """
    if telephony_base.endswith("/telephony"):
        return telephony_base[: -len("/telephony")] + "/integrations"
    return f"{telephony_base}/integrations"


async def _upsert_tenant_secret(
    session: AsyncSession, tenant_id: str, name: str, value: str
) -> None:
    """Encrypt + upsert one ``TenantSecret`` row — the same select-then-update-
    or-insert shape used above for telephony/Chatwoot/CRM secrets."""
    existing = (await session.execute(
        select(TenantSecret).where(
            TenantSecret.tenant_id == tenant_id, TenantSecret.name == name)
    )).scalar_one_or_none()
    if existing is not None:
        existing.value_encrypted = crypto.encrypt(value)
    else:
        session.add(TenantSecret(
            tenant_id=tenant_id, name=name, value_encrypted=crypto.encrypt(value)))


@router.post(
    "/{tenant_id}/webhook-credentials/rotate",
    response_model=WebhookCredentialsRotateResponse,
)
async def rotate_webhook_credentials(
    tenant_id: str,
    req: WebhookCredentialsRotateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> WebhookCredentialsRotateResponse:
    """Mint + rotate per-tenant inbound-webhook verification credentials (admin).

    NOT wired into any request-verification path yet — this route only mints
    and stores the values; the webhook handlers that will check them are
    separate follow-up work.

    - ``stringee``: a fresh ``webhook:stringee_path_token`` (appended as
      ``?vt=<token>`` to the Stringee Answer/Event URL). The Stringee
      *signing* secret (``webhook:stringee_signing_secret``) is never minted
      here — it comes from the tenant's own Stringee console and can only be
      set via the generic ``PATCH /tenants/{id}`` secret-update path.
    - ``exotel``: a fresh HTTP Basic Auth user/password pair
      (``webhook:exotel_basic_user`` / ``webhook:exotel_basic_password``) to
      embed in the Exotel callback URL's netloc.
    - ``chatwoot``: a fresh ``chatwoot:webhook_id`` to scope the Chatwoot
      Agent Bot webhook URL to this tenant.
    """
    t = await _require_tenant(session, tenant_id)

    if not crypto.has_key():
        raise HTTPException(
            status_code=503,
            detail="VOX_SECRET_KEY is not set — cannot encrypt webhook credentials")

    base = (platform_webhook_base_url() or "").rstrip("/") or "<your-webhook-base-url>"
    rotated: list[str] = []
    credentials: dict[str, str] = {}
    instructions: dict[str, str] = {}

    if "stringee" in req.providers:
        token = pysecrets.token_urlsafe(32)
        await _upsert_tenant_secret(session, tenant_id, "webhook:stringee_path_token", token)
        rotated.append("stringee")
        credentials["webhook:stringee_path_token"] = token
        # Build the instruction URL through the same tested helper production
        # code will use — `base` already includes /api/v1/telephony, so it
        # must be passed bare (answer_url_for appends the ANSWER_PATHS
        # segment + slug itself; re-adding the prefix here would double it up).
        stub = SimpleNamespace(
            slug=t.slug, secrets_resolved={"webhook:stringee_path_token": token})
        url = answer_url_for(stub, "stringee", base)
        instructions["stringee"] = f"Set the Stringee Answer/Event URL to {url}"

    if "exotel" in req.providers:
        # Username: token_hex(8) -> 16 hex chars ([0-9a-f]), guaranteed to
        # contain no ':'/'@'/'/' so it's always safe to embed verbatim in a
        # URL netloc. Password: token_urlsafe(24) for more entropy on the
        # half an attacker would actually need to brute-force; urlsafe's
        # alphabet (A-Za-z0-9-_) is also netloc-safe.
        user = pysecrets.token_hex(8)
        password = pysecrets.token_urlsafe(24)
        await _upsert_tenant_secret(session, tenant_id, "webhook:exotel_basic_user", user)
        await _upsert_tenant_secret(session, tenant_id, "webhook:exotel_basic_password", password)
        rotated.append("exotel")
        credentials["webhook:exotel_basic_user"] = user
        credentials["webhook:exotel_basic_password"] = password
        stub = SimpleNamespace(
            slug=t.slug,
            secrets_resolved={
                "webhook:exotel_basic_user": user,
                "webhook:exotel_basic_password": password,
            },
        )
        url = answer_url_for(stub, "exotel", base)
        instructions["exotel"] = f"Use {url} as the Exotel callback URL"

    if "chatwoot" in req.providers:
        webhook_id = pysecrets.token_urlsafe(32)
        await _upsert_tenant_secret(session, tenant_id, "chatwoot:webhook_id", webhook_id)
        rotated.append("chatwoot")
        credentials["chatwoot:webhook_id"] = webhook_id
        # The Chatwoot webhook route lives under a DIFFERENT router prefix
        # (/api/v1/integrations, registered via external_chat.router — see
        # src/api/__init__.py) than the telephony one `base` is built for
        # (/api/v1/telephony), so it can't reuse `base` directly.
        integrations_base = _chatwoot_integrations_base_url(base)
        instructions["chatwoot"] = (
            f"Point the Chatwoot Agent Bot webhook at "
            f"{integrations_base}/chatwoot/webhook/{webhook_id}"
        )

    await session.commit()
    try:
        await _refresh_resolver(request, tenant_id)
    except Exception:  # noqa: BLE001 — the DB write already succeeded and is
        # the source of truth; losing the plaintext response over a resolver
        # refresh hiccup would mean re-minting credentials for no reason.
        log.exception(
            "resolver refresh failed after webhook credential rotation",
            extra={"tenant_id": tenant_id})

    log.info(
        "rotated webhook credentials",
        extra={"tenant_id": tenant_id, "providers": rotated})
    return WebhookCredentialsRotateResponse(
        tenant_id=t.id, slug=t.slug, rotated=rotated,
        credentials=credentials, instructions=instructions,
    )


# --- Backoffice: list tenants + per-tenant analytics & billing -----------


class LayerInfo(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None


class TenantSummary(BaseModel):
    tenant_id: str
    slug: str
    name: str
    status: str
    mode: str
    max_concurrent_calls: int
    stt: LayerInfo
    llm: LayerInfo
    tts: LayerInfo
    realtime: LayerInfo
    telephony_provider: Optional[str] = None
    # Non-secret telephony config (so the backoffice can prefill it) + the NAMES
    # (never values) of the creds configured for the active provider.
    telephony_from_number: Optional[str] = None
    telephony_stringee_base_url: Optional[str] = None
    telephony_livekit_url: Optional[str] = None
    events_webhook_url: Optional[str] = None
    events_webhook_secret_set: bool = False
    telephony_creds_configured: list[str] = Field(default_factory=list)
    # Per-tenant Stringee webhook URLs (platform base + this tenant's slug) to
    # paste into the tenant's Stringee project so calls attribute correctly.
    stringee_softphone_answer_url: Optional[str] = None
    stringee_answer_url: Optional[str] = None
    # CRM this tenant is linked to (Tenant.crm_id FK, Task 1) — lets the
    # backoffice preselect the CRM dropdown.
    crm_id: Optional[str] = None
    # Whether the new per-tenant webhook-verification secrets (Task 1, minted
    # via POST .../webhook-credentials/rotate) are configured — NAMES/booleans
    # only, never values. Additive fields; default False for older tenants.
    stringee_webhook_auth_configured: bool = False
    exotel_basic_auth_configured: bool = False
    chatwoot_webhook_id_configured: bool = False


class TenantListResponse(BaseModel):
    tenants: list[TenantSummary]
    total: int


def _layer(pc: dict, key: str) -> LayerInfo:
    d = pc.get(key) or {}
    return LayerInfo(provider=d.get("provider"), model=d.get("model"))


_CRED_ENV_FIELDS = (
    "account_sid_env", "auth_token_env", "api_key_sid_env",
    "api_key_secret_env", "twiml_app_sid_env", "user_id_env",
)


def _configured_creds(tel_cfg: dict) -> list[str]:
    """Names (never values) of the cred slots set for the active provider."""
    active = TenantTelephonyConfig(**tel_cfg).active_creds()
    return [f for f in _CRED_ENV_FIELDS if getattr(active, f)]


@router.get("", response_model=TenantListResponse)
async def list_tenants(
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> TenantListResponse:
    """List every tenant with its mode + selected providers/models (admin)."""
    rows = (await session.execute(select(Tenant).order_by(Tenant.created_at))).scalars().all()
    base = (platform_webhook_base_url() or "").rstrip("/")

    # Webhook-credential secret names, per tenant — a dedicated query instead
    # of lazy-loading Tenant.secrets (the query above doesn't eager-load that
    # relationship, so touching it here would raise MissingGreenlet).
    webhook_secret_rows = (await session.execute(
        select(TenantSecret.tenant_id, TenantSecret.name).where(
            TenantSecret.name.in_([
                "webhook:stringee_path_token", "webhook:stringee_signing_secret",
                "webhook:exotel_basic_user", "webhook:exotel_basic_password",
                "chatwoot:webhook_id",
            ])
        )
    )).all()
    webhook_secret_names: dict[str, set[str]] = {}
    for secret_tenant_id, name in webhook_secret_rows:
        webhook_secret_names.setdefault(secret_tenant_id, set()).add(name)

    items = []
    for t in rows:
        pc = t.pipeline_config or {}
        tel = pc.get("telephony") or {}
        names = webhook_secret_names.get(t.id, set())
        items.append(TenantSummary(
            tenant_id=t.id, slug=t.slug, name=t.name, status=t.status,
            mode=t.mode, max_concurrent_calls=t.max_concurrent_calls,
            stt=_layer(pc, "stt"), llm=_layer(pc, "llm"), tts=_layer(pc, "tts"),
            realtime=_layer(pc, "realtime"),
            telephony_provider=tel.get("provider"),
            telephony_from_number=tel.get("from_number"),
            telephony_stringee_base_url=tel.get("stringee_base_url"),
            telephony_livekit_url=tel.get("livekit_url"),
            events_webhook_url=pc.get("events_webhook_url") or tel.get("events_webhook_url"),
            events_webhook_secret_set=bool(
                pc.get("events_webhook_secret_env") or tel.get("events_webhook_secret_env")
            ),
            telephony_creds_configured=_configured_creds(tel),
            stringee_softphone_answer_url=(
                f"{base}/stringee/softphone-answer/{t.slug}" if base else None),
            stringee_answer_url=(
                f"{base}/stringee/answer/{t.slug}" if base else None),
            crm_id=t.crm_id,
            stringee_webhook_auth_configured=bool(
                {"webhook:stringee_path_token", "webhook:stringee_signing_secret"} & names
            ),
            exotel_basic_auth_configured=(
                {"webhook:exotel_basic_user", "webhook:exotel_basic_password"} <= names
            ),
            chatwoot_webhook_id_configured="chatwoot:webhook_id" in names,
        ))
    return TenantListResponse(tenants=items, total=len(items))


async def _require_tenant(session: AsyncSession, tenant_id: str) -> Tenant:
    t = await session.get(Tenant, tenant_id)
    if t is None:
        raise HTTPException(status_code=404, detail="tenant not found")
    return t


@router.get("/{tenant_id}/chat-config")
async def get_chat_config(
    tenant_id: str,
    request: Request,
    _: None = Depends(require_admin),
) -> dict:
    """Return non-secret Chatwoot config for the tenant.  api_token is '...' if set."""
    resolver = getattr(request.app.state, "tenant_resolver", None)
    ctx = None
    if resolver and hasattr(resolver, "resolve_by_id"):
        ctx = await resolver.resolve_by_id(tenant_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="tenant not found")
    sr = ctx.secrets_resolved
    return {
        "chatwoot": {
            "inbox_id":   sr.get("chatwoot:inbox_id") or "",
            "account_id": sr.get("chatwoot:account_id") or "",
            "api_url":    sr.get("chatwoot:api_url") or "",
            "api_token":  "..." if sr.get("chatwoot:api_token") else "",
        },
        "crm": {
            "base_url":   sr.get("crm:base_url") or "",
            "auth_type":  sr.get("crm:auth_type") or "api_key",
            "api_token":  "..." if sr.get("crm:api_token") else "",
            "x_api_key": "..." if sr.get("crm:x_api_key") else "",
            "operator_id": ctx.settings.crm.operator_id or "",
        },
    }


class TenantAnalytics(BaseModel):
    tenant_id: str
    total_calls: int
    by_status: dict[str, int]
    by_outcome: dict[str, int]
    # Manual (human softphone) vs AI (voicebot) call counts, keyed by agent_type.
    by_agent_type: dict[str, int]
    by_campaign: dict[str, int]    # keyed by campaign name ("none" for ad-hoc calls)
    by_channel: dict[str, int]     # voice / softphone (webconsole remapped to voice)
    by_provider: dict[str, int]    # real telephony provider only (webconsole excluded)
    total_duration_ms: int
    avg_duration_ms: int


@router.get("/{tenant_id}/analytics", response_model=TenantAnalytics)
async def tenant_analytics(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> TenantAnalytics:
    """Call analytics for one tenant, aggregated from the conversations table."""
    await _require_tenant(session, tenant_id)
    # campaign_id → name, so the breakdown reads in campaign names not opaque ids.
    camp_names = dict((await session.execute(
        select(Campaign.id, Campaign.name).where(Campaign.tenant_id == tenant_id)
    )).all())
    rows = (await session.execute(
        select(Conversation.status, Conversation.outcome, Conversation.duration_ms,
               Conversation.agent_type, Conversation.campaign_id, Conversation.channel,
               Conversation.telephony_provider)
        .where(Conversation.tenant_id == tenant_id)
    )).all()

    def _bump(d: dict[str, int], key: str) -> None:
        d[key] = d.get(key, 0) + 1

    by_status: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    by_agent_type: dict[str, int] = {}
    by_campaign: dict[str, int] = {}
    by_channel: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    total_dur = 0
    for status, outcome, dur, agent_type, campaign_id, channel, provider in rows:
        _bump(by_status, status or "unknown")
        # Count rows with no outcome under "no_outcome" so by_outcome totals to
        # total_calls (matching by_status) — calls in progress or that ended
        # before analysis have no outcome yet.
        _bump(by_outcome, outcome or "no_outcome")
        _bump(by_agent_type, agent_type or "unknown")
        _bump(by_campaign, camp_names.get(campaign_id, campaign_id) if campaign_id else "none")
        # webconsole is a browser transport, not a channel — treat as "voice"
        effective_channel = "voice" if channel == "webconsole" else (channel or "unknown")
        _bump(by_channel, effective_channel)
        # webconsole has no telephony provider — bucket it explicitly so totals match
        effective_provider = "webconsole" if channel == "webconsole" else (provider or "none")
        _bump(by_provider, effective_provider)
        total_dur += int(dur or 0)
    n = len(rows)
    return TenantAnalytics(
        tenant_id=tenant_id, total_calls=n, by_status=by_status, by_outcome=by_outcome,
        by_agent_type=by_agent_type, by_campaign=by_campaign, by_channel=by_channel,
        by_provider=by_provider,
        total_duration_ms=total_dur, avg_duration_ms=(total_dur // n if n else 0),
    )


class ChatAnalytics(BaseModel):
    tenant_id: str
    total_sessions: int
    total_messages: int
    avg_messages_per_session: float
    escalation_rate_pct: float        # % of sessions that reached awaiting_human or human
    by_status: dict[str, int]         # active / ended
    by_mode: dict[str, int]           # ai / awaiting_human / human / closed
    by_source: dict[str, int]         # widget / external / chatwoot / unknown


@router.get("/{tenant_id}/chat-analytics", response_model=ChatAnalytics)
async def tenant_chat_analytics(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> ChatAnalytics:
    """Chat session analytics for one tenant."""
    from src.models.chat import ChatMessage, ChatSession
    await _require_tenant(session, tenant_id)

    rows = (await session.execute(
        select(ChatSession.status, ChatSession.mode,
               ChatSession.message_count, ChatSession.extra_data)
        .where(ChatSession.tenant_id == tenant_id)
    )).all()

    by_status: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    by_source: dict[str, int] = {}
    total_messages = 0
    escalated = 0

    def _bump(d: dict[str, int], k: str) -> None:
        d[k] = d.get(k, 0) + 1

    for status, mode, msg_count, extra_data in rows:
        _bump(by_status, status or "unknown")
        _bump(by_mode, mode or "unknown")
        source = ((extra_data or {}).get("source") or "widget")
        _bump(by_source, source)
        total_messages += int(msg_count or 0)
        if mode in ("awaiting_human", "human", "closed") and status != "active":
            escalated += 1
        elif mode in ("awaiting_human", "human"):
            escalated += 1

    n = len(rows)
    return ChatAnalytics(
        tenant_id=tenant_id,
        total_sessions=n,
        total_messages=total_messages,
        avg_messages_per_session=round(total_messages / n, 1) if n else 0.0,
        escalation_rate_pct=round(escalated * 100 / n, 1) if n else 0.0,
        by_status=by_status,
        by_mode=by_mode,
        by_source=by_source,
    )


class TenantBilling(BaseModel):
    tenant_id: str
    total_calls: int
    billable_minutes: float
    platform_cost: float                 # combined voice + chat: what we charge
                                          # (STT/LLM/TTS or S2S, PLUS chat LLM tokens)
    avg_cost_per_call: float
    tentative_telephony_cost: float      # tenant's own telephony — informational only
    chat_sessions: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    chat_cost: float = 0.0
    currency: str = "USD"


@router.get("/{tenant_id}/billing", response_model=TenantBilling)
async def tenant_billing(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> TenantBilling:
    """Billing summary: platform cost (voice + chat combined, telephony excluded)
    + a tentative telephony figure computed from the tenant's telephony provider rate."""
    from src.models.chat import ChatSession

    await _require_tenant(session, tenant_id)
    rows = (await session.execute(
        select(Conversation.cost, Conversation.duration_ms, Conversation.telephony_provider)
        .where(Conversation.tenant_id == tenant_id)
    )).all()
    # telephony rates (model="") for the tentative figure
    tel_rates = dict((p, c) for p, c in (await session.execute(
        select(ProviderCost.provider, ProviderCost.cost_per_min)
        .where(ProviderCost.kind == "telephony", ProviderCost.model == "")
    )).all())

    voice_cost = 0.0
    tentative_tel = 0.0
    total_ms = 0
    for cost, dur, tel in rows:
        voice_cost += float(cost or 0.0)
        total_ms += int(dur or 0)
        if tel and dur:
            tentative_tel += tel_rates.get(tel, 0.0) * (int(dur) / 60_000.0)
    n = len(rows)

    chat_row = (await session.execute(
        select(
            func.count(ChatSession.id), func.coalesce(func.sum(ChatSession.cost), 0.0),
            func.coalesce(func.sum(ChatSession.input_tokens), 0),
            func.coalesce(func.sum(ChatSession.output_tokens), 0),
        ).where(ChatSession.tenant_id == tenant_id)
    )).one()
    chat_sessions, chat_cost, chat_in_tok, chat_out_tok = chat_row

    platform = voice_cost + float(chat_cost or 0.0)
    return TenantBilling(
        tenant_id=tenant_id, total_calls=n,
        billable_minutes=round(total_ms / 60_000.0, 4),
        platform_cost=round(platform, 6),
        avg_cost_per_call=round(voice_cost / n, 6) if n else 0.0,
        tentative_telephony_cost=round(tentative_tel, 6),
        chat_sessions=int(chat_sessions or 0),
        chat_input_tokens=int(chat_in_tok or 0),
        chat_output_tokens=int(chat_out_tok or 0),
        chat_cost=round(float(chat_cost or 0.0), 6),
    )


# --- Campaign script editor (backoffice) --------------------------------
#
# The voicebot script lives in campaigns.config_yaml (a YAML blob, read live per
# call by DbCampaignResolver). These admin routes expose it as structured fields
# so the backoffice can view/edit it without hand-editing YAML. Edits merge into
# the existing YAML dict — every key we don't model (slots, id, status) is kept.


class CampaignScriptIn(BaseModel):
    """Structured campaign-script fields. All optional — only provided fields are
    overlaid onto the existing config_yaml."""
    name: Optional[str] = None
    agent_name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    personality: Optional[str] = None
    language: Optional[str] = None
    gender: Optional[str] = None
    greeting: Optional[str] = None
    objective: Optional[str] = None
    closing: Optional[str] = None
    conversation_style: Optional[str] = None
    max_turns: Optional[int] = None
    talking_points: Optional[list[str]] = None
    dos: Optional[list[str]] = None
    donts: Optional[list[str]] = None
    knowledge: Optional[dict[str, str]] = None


# CampaignScriptIn field -> (block, yaml_key); block is "agent" or "script".
_AGENT_FIELDS = {
    "agent_name": "name", "company": "company", "role": "role",
    "personality": "personality", "language": "language", "gender": "gender",
}
_SCRIPT_SCALARS = {
    "greeting": "greeting", "objective": "objective", "closing": "closing",
    "conversation_style": "conversation_style", "max_turns": "max_turns",
}
_SCRIPT_LISTS = ("talking_points", "dos", "donts")


def _campaign_script_fields(config_yaml: str) -> dict:
    """Parse config_yaml into the structured editor fields (reuses the same
    parser the runtime uses). Returns {} if the YAML can't be parsed."""
    try:
        sc = parse_campaign_yaml(config_yaml).script
    except Exception:  # noqa: BLE001 - a broken row shouldn't break the list
        return {}
    closing = sc.closing.get("default") or next(iter(sc.closing.values()), "") if sc.closing else ""
    return {
        "agent_name": sc.agent_name, "company": sc.company_name, "role": sc.agent_role,
        "personality": sc.personality, "language": sc.language_default, "gender": sc.gender,
        "greeting": sc.opening, "objective": sc.objective, "closing": closing,
        "conversation_style": sc.conversation_style, "max_turns": sc.max_turns,
        "talking_points": sc.talking_points, "dos": sc.dos, "donts": sc.donts,
        "knowledge": sc.knowledge,
    }


def _apply_campaign_script(config_yaml: str, req: "CampaignScriptIn") -> str:
    """Overlay the provided fields onto the existing campaign YAML, preserving
    every key we don't model (slots, id, status, ...). Validates the result
    parses; raises ValueError otherwise."""
    import yaml

    data = yaml.safe_load(config_yaml) or {}
    wrapped = "campaign" in data
    camp = data["campaign"] if wrapped else data
    if req.name is not None:
        camp["name"] = req.name
    agent = camp.setdefault("agent", {})
    script = camp.setdefault("script", {})
    for field_, ykey in _AGENT_FIELDS.items():
        v = getattr(req, field_)
        if v is not None:
            agent[ykey] = v
    for field_, ykey in _SCRIPT_SCALARS.items():
        v = getattr(req, field_)
        if v is not None:
            script[ykey] = v
    for field_ in _SCRIPT_LISTS:
        v = getattr(req, field_)
        if v is not None:
            script[field_] = v
    if req.knowledge is not None:
        script["knowledge"] = req.knowledge
    out = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    parse_campaign_yaml(out)  # validate it still parses into a script
    return out


@router.get("/{tenant_id}/campaigns")
async def list_tenant_campaigns(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> dict:
    """The tenant's campaigns with their script as structured fields (admin)."""
    await _require_tenant(session, tenant_id)
    rows = (await session.execute(
        select(Campaign).where(Campaign.tenant_id == tenant_id).order_by(Campaign.created_at)
    )).scalars().all()
    return {"campaigns": [
        {"id": c.id, "name": c.name, "status": c.status,
         "script": _campaign_script_fields(c.config_yaml)} for c in rows]}


@router.put("/{tenant_id}/campaigns/{campaign_id}")
async def update_campaign_script(
    tenant_id: str,
    campaign_id: str,
    req: CampaignScriptIn,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> dict:
    """Edit a campaign's script (structured fields merged into its config_yaml)."""
    row = await session.get(Campaign, campaign_id)
    if row is None or row.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="campaign not found")
    try:
        row.config_yaml = _apply_campaign_script(row.config_yaml, req)
    except Exception as e:  # noqa: BLE001 - invalid YAML/script → 400
        raise HTTPException(status_code=400, detail=f"invalid script: {e}")
    if req.name is not None:
        row.name = req.name
    await session.commit()
    return {"id": row.id, "name": row.name, "status": row.status,
            "script": _campaign_script_fields(row.config_yaml)}
