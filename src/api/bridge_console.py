"""Bridge console — call-launcher that hands off to the AI voiceagent.

Simulates what the CRM coordination service does:
  Place an outbound call to a customer and hand it off to the AI voiceagent.

Routes (all behind dev_console_enabled gate):
  GET  /dev/bridge                   → serve bridge_console.html
  GET  /dev/bridge/tenants           → active tenants with from-numbers per provider
  POST /dev/bridge/place-call        → place outbound call (AI handles it from there)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()

_STATIC = Path(__file__).parent.parent.parent / "static"


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/dev/bridge")
async def bridge_console_page() -> FileResponse:
    return FileResponse(_STATIC / "bridge_console.html")


@router.get("/dev/bridge/tenants")
async def bridge_tenants() -> dict:
    """Return all active tenants with their from-numbers per telephony provider."""
    from sqlalchemy import select
    from src.models.tenant import Tenant
    from src.models.database import get_sessionmaker
    from src.config_tenant import load_tenant

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Tenant).where(Tenant.status == "active").order_by(Tenant.slug)
        )).scalars().all()

    tenants = []
    for row in rows:
        from_numbers: dict[str, str] = {}
        try:
            cfg = load_tenant(row.slug)
            tel = cfg.pipeline.telephony
            outbound = tel.outbound_from or {}
            for prov in ("twilio", "exotel", "stringee"):
                num = outbound.get(prov) or (tel.from_number if (tel.provider or "").lower() == prov else "")
                if num:
                    from_numbers[prov] = num
        except Exception:
            pass
        tenants.append({"slug": row.slug, "name": row.name, "from_numbers": from_numbers})

    return {"tenants": tenants}


# ── Place call ────────────────────────────────────────────────────────────────


class PlaceBridgeCallRequest(BaseModel):
    provider: str = "twilio"
    to_number: str = ""
    tenant: str = "dev"
    mode: str = "s2s"
    voice: str = ""
    caller_name: str = ""
    lead_name: str = ""
    lead_gender: str = ""


@router.post("/dev/bridge/place-call")
async def bridge_place_call(req: PlaceBridgeCallRequest) -> dict:
    """Place an outbound call; the AI voiceagent handles it from the moment it connects."""
    from src.auth.middleware import tenant_from_slug
    from src.api import dev_call_control
    from src.api.answer_paths import ANSWER_PATHS
    from src.providers import get_telephony_provider
    from src.interfaces.telephony import CallConfig
    from src.config_tenant import platform_webhook_base_url

    try:
        tenant = await tenant_from_slug(req.tenant)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")

    webhook_base = platform_webhook_base_url()
    if not webhook_base:
        raise HTTPException(status_code=400, detail="WEBHOOK_BASE_URL is not set")

    provider = req.provider.strip().lower()

    # Resolve from_number
    tel = tenant.settings.pipeline.telephony
    from_number = (tel.outbound_from or {}).get(provider)
    if not from_number and (tel.provider or "").lower() == provider:
        from_number = tel.from_number
    if not from_number:
        try:
            from src.config_tenant import load_tenant as _load_tenant
            _yaml = _load_tenant(req.tenant)
            _yaml_tel = _yaml.pipeline.telephony
            from_number = (_yaml_tel.outbound_from or {}).get(provider)
            if not from_number and (_yaml_tel.provider or "").lower() == provider:
                from_number = _yaml_tel.from_number
        except Exception:
            pass
    if not from_number:
        raise HTTPException(
            status_code=400,
            detail=f"no caller-ID configured for '{provider}'",
        )

    # Resolve credentials
    def _cred(name: str | None) -> str | None:
        try:
            return tenant.secret(name) if name else None
        except Exception:
            return None

    import os as _os
    pcreds = tel.creds_for(provider)
    acct = _cred(pcreds.account_sid_env)
    auth = _cred(pcreds.auth_token_env)
    if provider == "stringee":
        acct = (_cred(getattr(pcreds, "api_key_sid_env", None))
                or _os.environ.get("STRINGEE_API_KEY_SID") or acct)
        auth = (_cred(getattr(pcreds, "api_key_secret_env", None))
                or _os.environ.get("STRINGEE_API_KEY_SECRET") or auth)
    uid = _cred(getattr(pcreds, "user_id_env", None))

    # Build telephony adapter
    try:
        adapter = get_telephony_provider({
            "provider": provider,
            "account_sid": acct, "auth_token": auth,
            "api_key_sid": acct, "api_key_secret": auth,
            "user_id": uid,
            "base_url": getattr(tel, "stringee_base_url", None),
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"telephony adapter unavailable: {e}")

    # Set dev override — consumed by bridge factory when the call connects
    dev_call_control.set_override(
        tenant.slug,
        mode=req.mode,
        voice=req.voice.strip(),
        caller_name=req.caller_name.strip(),
        lead_name=req.lead_name.strip(),
        lead_gender=req.lead_gender.strip(),
    )

    answer_path = f"{ANSWER_PATHS[provider]}/{tenant.slug}"
    cfg = CallConfig(
        to_number=req.to_number.strip(),
        from_number=from_number,
        webhook_url=f"{webhook_base.rstrip('/')}/{answer_path}",
    )
    try:
        session = await asyncio.wait_for(adapter.initiate_call(cfg), timeout=20.0)
    except asyncio.TimeoutError:
        dev_call_control.pop_override(tenant.slug)
        raise HTTPException(status_code=502, detail="call timed out after 20s")
    except Exception as e:
        dev_call_control.pop_override(tenant.slug)
        raise HTTPException(status_code=502, detail=f"call failed: {e}")

    from src.api import dev_call_control as _dcc
    _dcc.monitor.set_status(session.session_id, "calling")

    log.info("bridge console: call placed",
             extra={"call_sid": session.session_id, "provider": provider})
    return {"call_sid": session.session_id, "provider": provider}
