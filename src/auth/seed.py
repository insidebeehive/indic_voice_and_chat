"""One-time bridge: migrate YAML tenants into the DB.

The system used to load `config/tenants/*.yaml` at boot. We now store tenants in
the DB. To survive the cutover without losing the running `dev` tenant, this
seeds the DB **from the YAML files when the tenants table is empty** (idempotent):
each tenant becomes a row (+ phone numbers + API tokens from
`TENANT_<SLUG>_API_TOKENS`), and its **telephony** keys (only) are encrypted into
`tenant_secrets`. STT/LLM/TTS/S2S keys stay in the shared master env.

After the first boot the DB is authoritative and the YAML is ignored.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from sqlalchemy import select

from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.config_tenant import _resolve_dir, discover_tenant_slugs, load_tenant
from src.models.campaign import Campaign
from src.models.tenant import (
    ProviderCost,
    Tenant,
    TenantApiKey,
    TenantPhoneNumber,
    TenantSecret,
)

log = logging.getLogger(__name__)

_PROVIDER_COSTS_YAML = Path("config/provider_costs.yaml")


async def seed_tenants_from_yaml(session, tenant_dir=None) -> int:
    """Upsert every YAML tenant into the DB. Returns the number processed."""
    base = _resolve_dir(tenant_dir)
    have_key = bool(os.environ.get(crypto.VOX_SECRET_KEY_ENV))
    count = 0
    for slug in discover_tenant_slugs(base):
        s = load_tenant(slug, base)
        row = await session.get(Tenant, s.id)
        cfg = s.pipeline.model_dump()
        # Compliance is a top-level TenantSettings field (not in the pipeline); ride
        # it inside the pipeline_config JSON so it round-trips without a migration.
        # db_resolver pulls it back out (TenantPipelineConfig ignores the extra key).
        cfg["compliance"] = s.compliance.model_dump()
        if row is None:
            session.add(Tenant(
                id=s.id, slug=s.slug, name=s.name, status=s.status,
                timezone=s.timezone, default_language=s.default_language,
                mode=s.pipeline.mode, max_concurrent_calls=s.max_concurrent_calls,
                pipeline_config=cfg,
            ))
        else:  # refresh config from YAML
            row.name, row.status, row.timezone = s.name, s.status, s.timezone
            row.default_language, row.mode = s.default_language, s.pipeline.mode
            row.max_concurrent_calls, row.pipeline_config = s.max_concurrent_calls, cfg

        for ph in s.phone_numbers:
            if await session.get(TenantPhoneNumber, ph) is None:
                session.add(TenantPhoneNumber(
                    phone_number=ph, tenant_id=s.id,
                    provider=s.pipeline.telephony.provider or "twilio"))

        raw = os.environ.get(f"TENANT_{slug.upper()}_API_TOKENS", "")
        for tok in (t.strip() for t in raw.split(",") if t.strip()):
            h = hash_api_token(tok)
            if await session.get(TenantApiKey, h) is None:
                session.add(TenantApiKey(token_hash=h, tenant_id=s.id, label="seed"))

        # Telephony keys only → tenant_secrets (encrypted). Stringee reads its keys
        # straight from env; Twilio/Exotel go through tenant.secret(<*_env name>).
        tel = s.pipeline.telephony
        for name in (tel.account_sid_env, tel.auth_token_env):
            value = os.environ.get(name) if name else None
            if name and value and have_key:
                if await session.get(TenantSecret, (s.id, name)) is None:
                    session.add(TenantSecret(
                        tenant_id=s.id, name=name, value_encrypted=crypto.encrypt(value)))
        count += 1

    await session.commit()
    log.info("seeded tenants from YAML", extra={"count": count})
    return count


async def seed_if_empty(sessionmaker, tenant_dir=None) -> int:
    """Seed from YAML only when the tenants table is empty (boot-safe bridge)."""
    async with sessionmaker() as session:
        if (await session.execute(select(Tenant.id).limit(1))).first() is not None:
            return 0
        return await seed_tenants_from_yaml(session, tenant_dir)


async def seed_campaigns_if_empty(sessionmaker, campaigns_dir=None) -> int:
    """Give every tenant a DB campaign migrated from the global ``VOX_CAMPAIGN``
    file, when they have none. Campaigns then diverge per-tenant via the
    ``/campaigns`` API. Idempotent (skips a tenant that already has a campaign).

    This backs the no-global-fallback resolution: every tenant needs a row, so
    the live call never falls back to a shared/global script.
    """
    from src.dialogue.campaign_loader import DEFAULT_CAMPAIGNS_DIR, active_campaign_slug

    base = campaigns_dir or DEFAULT_CAMPAIGNS_DIR
    slug = active_campaign_slug()
    path = base / f"{slug}.yaml"
    if not path.exists():
        log.warning("campaign seed: %s not found; skipping", path)
        return 0
    raw = path.read_text()
    data = yaml.safe_load(raw) or {}
    camp = data.get("campaign", data)
    name = camp.get("name") or (camp.get("agent") or {}).get("company") or slug

    seeded = 0
    async with sessionmaker() as session:
        tenant_ids = (await session.execute(select(Tenant.id))).scalars().all()
        for tid in tenant_ids:
            existing = (await session.execute(
                select(Campaign.id).where(Campaign.tenant_id == tid).limit(1))).first()
            if existing is not None:
                continue
            session.add(Campaign(
                id=f"camp_{tid}_default", tenant_id=tid, name=str(name),
                status="active", config_yaml=raw))
            seeded += 1
        await session.commit()
    if seeded:
        log.info("seeded default campaigns", extra={"count": seeded})
    return seeded


async def seed_provider_costs(sessionmaker, path: Path = _PROVIDER_COSTS_YAML) -> int:
    """Insert any provider_costs rows from the YAML that don't exist yet.

    Existing rows (e.g. rates an admin updated via the API) are left untouched —
    we only add missing (kind, provider) pairs. Returns the number inserted.
    """
    if not path.exists():
        return 0
    data = yaml.safe_load(path.read_text()) or {}
    inserted = 0
    async with sessionmaker() as session:
        for kind, providers in data.items():
            for provider, val in (providers or {}).items():
                # Nested {model: cost} -> per-model rows; scalar -> provider-level
                # (model="") for telephony / a provider default.
                entries = val.items() if isinstance(val, dict) else [("", val)]
                for model, cost in entries:
                    if await session.get(ProviderCost, (kind, provider, model)) is None:
                        session.add(ProviderCost(
                            kind=kind, provider=provider, model=model,
                            cost_per_min=float(cost)))
                        inserted += 1
        await session.commit()
    if inserted:
        log.info("seeded provider costs", extra={"inserted": inserted})
    return inserted
