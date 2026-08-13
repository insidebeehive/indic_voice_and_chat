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


async def sync_telephony_from_yaml(sessionmaker, tenant_dir=None) -> int:
    """Sync telephony.from_number and outbound_from from YAML into existing tenant rows.

    Called on every boot so YAML caller-ID changes are picked up without requiring
    a DB wipe. Only writes rows that differ. Safe under rolling restart because it
    only touches pipeline_config (no secret rows or phone-number rows that could
    conflict with inbound-call lookups).
    """
    base = _resolve_dir(tenant_dir)
    patched = 0
    async with sessionmaker() as session:
        rows = (await session.execute(select(Tenant))).scalars().all()
        for row in rows:
            try:
                yaml_cfg = load_tenant(row.slug, base)
            except Exception:
                continue
            yaml_tel = yaml_cfg.pipeline.telephony
            pc = dict(row.pipeline_config or {})
            tel = dict(pc.get("telephony") or {})
            changed = False
            if yaml_tel.from_number and tel.get("from_number") != yaml_tel.from_number:
                tel["from_number"] = yaml_tel.from_number
                changed = True
            if yaml_tel.outbound_from:
                current_of = dict(tel.get("outbound_from") or {})
                merged = {**current_of, **yaml_tel.outbound_from}
                if merged != current_of:
                    tel["outbound_from"] = merged
                    changed = True
            if changed:
                pc["telephony"] = tel
                row.pipeline_config = dict(pc)
                patched += 1
        if patched:
            await session.commit()
            log.info("synced telephony caller-IDs from YAML", extra={"count": patched})
    return patched


async def patch_telephony_outbound_from(sessionmaker) -> None:
    """One-time patch: copy telephony.from_number into outbound_from[provider] when missing.

    Rows seeded before outbound_from was introduced have from_number set but
    outbound_from empty. The dev-console place-call path checks outbound_from
    first; without this the fallback to from_number only works when provider
    matches, and only if provider is stored in the row (old rows may have it
    null). This runs on every boot but is a cheap SELECT + conditional UPDATE.
    """
    async with sessionmaker() as session:
        rows = (await session.execute(select(Tenant))).scalars().all()
        patched = 0
        for row in rows:
            pc = row.pipeline_config or {}
            tel = pc.get("telephony") or {}
            provider = (tel.get("provider") or "").lower()
            from_number = tel.get("from_number") or ""
            outbound_from = tel.get("outbound_from") or {}
            if provider and from_number and provider not in outbound_from:
                outbound_from[provider] = from_number
                tel["outbound_from"] = outbound_from
                pc["telephony"] = tel
                row.pipeline_config = dict(pc)  # force SQLAlchemy to detect the change
                patched += 1
        if patched:
            await session.commit()
            log.info("patched telephony outbound_from", extra={"count": patched})


async def seed_provider_costs(sessionmaker, path: Path = _PROVIDER_COSTS_YAML) -> int:
    """Insert any provider_costs rows from the YAML that don't exist yet, and
    upsert the per-token LLM rates (``llm_token_rates``) onto their matching row.

    The per-minute rows (e.g. rates an admin updated via the API) are left
    untouched — we only add missing (kind, provider, model) rows. The
    per-token rates are always upserted from YAML (a separate, smaller rate
    catalog with no admin-edit history to protect yet — see
    ``PUT /api/v1/providers`` for the live-editable path going forward).
    Returns the number of NEW rows inserted (per-minute + per-token combined).
    """
    if not path.exists():
        return 0
    data = yaml.safe_load(path.read_text()) or {}
    # Not a `kind`, so it's excluded from the per-minute loop below and
    # handled by its own pass.
    token_rates = data.pop("llm_token_rates", None) or {}
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

        for provider, models in token_rates.items():
            for model, rates in (models or {}).items():
                in_rate = float((rates or {}).get("input_per_1k", 0.0))
                out_rate = float((rates or {}).get("output_per_1k", 0.0))
                row = await session.get(ProviderCost, ("llm", provider, model))
                if row is None:
                    session.add(ProviderCost(
                        kind="llm", provider=provider, model=model,
                        cost_per_1k_input_tokens=in_rate,
                        cost_per_1k_output_tokens=out_rate))
                    inserted += 1
                else:
                    row.cost_per_1k_input_tokens = in_rate
                    row.cost_per_1k_output_tokens = out_rate

        await session.commit()
    if inserted:
        log.info("seeded provider costs", extra={"inserted": inserted})
    return inserted
