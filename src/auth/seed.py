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


async def patch_campaign_remove_sir(sessionmaker) -> int:
    """One-time patch: remove hardcoded 'Sir' from knowledge/closing in bharat_matka
    campaigns and update the greeting to use {agent_raha_rahi}/{lead_salutation} tokens.

    Idempotent — skips rows that no longer contain 'Sir,' or 'Sir ' in the script body.
    """
    _KNOWLEDGE = {
        "safety": (
            "Main samajh sakti hun aapka concern. Par ye OFFICIAL Bharat Matka app hai. "
            "Hazaron log roz khel rahe hain. Aap befikar rahiye."
        ),
        "scam_concerns": (
            "Trust kijiye, main aapko PERSONALLY GUIDE karungi. Aap chote amount se, "
            "bas 100 rupay se check karke dekhiye. Withdrawal turant milega."
        ),
        "withdrawal": (
            "Withdrawal ki koi tension nahi. 500 se lekar jitna marzi nikal lijiye, sab instant hai."
        ),
        "deposit": (
            "Minimum Deposit bas 100 rupay hai. Aap try karne ke liye chota amount daal sakte hain."
        ),
        "transaction_speed": (
            "Deposit aur Withdrawal dono AUTOMATIC aur FAST hain. 5 minute mein paisa aa jata hai."
        ),
        "support": (
            "Main hun na. Koi bhi dikkat aayi toh hamara WhatsApp Support 24 ghante on rehta hai. "
            "Turant reply milega."
        ),
        "referral": (
            "Doston ko refer karne par 3% COMMISSION milega har Monday. "
            "Khelne ka bhi paisa, refer karne ka bhi."
        ),
    }
    _CLOSING = "Ok, Thank you time dene ke liye. Aapka din shubh rahe!"
    _GREETING = (
        "हेलो, नमस्ते{lead_salutation}! मैं {agent_name} बात कर {agent_raha_rahi} हूं "
        "{company_name} से। क्या आपका एक मिनट हो सकता है?"
    )

    patched = 0
    async with sessionmaker() as session:
        rows = (await session.execute(select(Campaign))).scalars().all()
        for row in rows:
            raw = row.config_yaml or ""
            if "Sir" not in raw and "सर" not in raw:
                continue
            try:
                data = yaml.safe_load(raw) or {}
            except Exception:
                continue
            camp = data.get("campaign", data)
            script = camp.get("script") or {}
            script["greeting"] = _GREETING
            script["knowledge"] = _KNOWLEDGE
            script["closing"] = _CLOSING
            # Strip greeting_male/greeting_female variants that hardcoded Sir/Ma'am
            script.pop("greeting_male", None)
            script.pop("greeting_female", None)
            camp["script"] = script
            if "campaign" in data:
                data["campaign"] = camp
            else:
                data = camp
            row.config_yaml = yaml.dump(data, allow_unicode=True, sort_keys=False)
            patched += 1
        if patched:
            await session.commit()
            log.info("patched campaign: removed Sir, updated greeting tokens", extra={"count": patched})
    return patched


async def patch_campaign_replace_text(sessionmaker, find: str, replace: str) -> int:
    """Generic one-time text patch across all campaign config_yaml blobs.

    Idempotent — skips rows where ``find`` is not present.
    """
    patched = 0
    async with sessionmaker() as session:
        rows = (await session.execute(select(Campaign))).scalars().all()
        for row in rows:
            if find not in (row.config_yaml or ""):
                continue
            row.config_yaml = row.config_yaml.replace(find, replace)
            patched += 1
        if patched:
            await session.commit()
            log.info("patched campaign text", extra={"find": find, "replace": replace, "count": patched})
    return patched


async def patch_campaign_parameterize_company(sessionmaker) -> int:
    """Replace hardcoded company name in script text fields with {company_name} token.

    Reads the company name from each campaign's agent.company field and replaces
    every occurrence in script text fields (greeting, knowledge, closing, objective,
    talking_points) with the {company_name} template token. The agent.company field
    itself is left unchanged — it remains the source of truth.

    Idempotent — skips rows where no script field contains the literal company name.
    """
    def _replace_in_script(script: dict, company: str) -> bool:
        changed = False
        for key in ("greeting", "opening", "objective"):
            if isinstance(script.get(key), str) and company in script[key]:
                script[key] = script[key].replace(company, "{company_name}")
                changed = True
        for key in ("knowledge", "objection_responses"):
            block = script.get(key) or {}
            if isinstance(block, dict):
                for k, v in block.items():
                    if isinstance(v, str) and company in v:
                        block[k] = v.replace(company, "{company_name}")
                        changed = True
        closing = script.get("closing")
        if isinstance(closing, str) and company in closing:
            script["closing"] = closing.replace(company, "{company_name}")
            changed = True
        elif isinstance(closing, dict):
            for k, v in closing.items():
                if isinstance(v, str) and company in v:
                    closing[k] = v.replace(company, "{company_name}")
                    changed = True
        tps = script.get("talking_points") or []
        for i, tp in enumerate(tps):
            if isinstance(tp, str) and company in tp:
                tps[i] = tp.replace(company, "{company_name}")
                changed = True
        return changed

    patched = 0
    async with sessionmaker() as session:
        rows = (await session.execute(select(Campaign))).scalars().all()
        for row in rows:
            raw = row.config_yaml or ""
            try:
                data = yaml.safe_load(raw) or {}
            except Exception:
                continue
            camp = data.get("campaign", data)
            agent = camp.get("agent") or {}
            company = (agent.get("company") or camp.get("company_name") or "").strip()
            if not company or company == "{company_name}":
                continue
            script = camp.get("script") or {}
            if not _replace_in_script(script, company):
                continue
            camp["script"] = script
            if "campaign" in data:
                data["campaign"] = camp
            else:
                data = camp
            row.config_yaml = yaml.dump(data, allow_unicode=True, sort_keys=False)
            patched += 1
        if patched:
            await session.commit()
            log.info("parameterized {company_name} in campaign scripts", extra={"count": patched})
    return patched


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
