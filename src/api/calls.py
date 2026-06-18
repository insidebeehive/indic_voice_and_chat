"""Call Lead (async) + call status.

- ``POST /api/v1/campaigns/{id}/calls`` — place one outbound call for a lead.
  Async: it returns a ``call_id`` immediately; the outcome lands later (the
  bridge writes it at teardown). Guards: the campaign must be ``active`` and the
  tenant must be under its ``max_concurrent_calls`` cap (else 429). On success a
  ``conversations`` row is inserted (``in_progress``, keyed by the provider Call
  SID) recording the config used — mode, stt/llm/tts/realtime providers, voice,
  telephony provider — for statistics + cost.
- ``GET  /api/v1/calls/{call_id}`` — poll the call's status/outcome/cost.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.call_store import count_active_calls, insert_call
from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.interfaces.telephony import CallConfig
from src.models.campaign import Campaign as DbCampaign
from src.models.conversation import Conversation
from src.providers import get_telephony_provider

log = logging.getLogger(__name__)
router = APIRouter(tags=["calls"])


# --- Schemas ------------------------------------------------------------


class CallLeadRequest(BaseModel):
    to_number: str = Field(min_length=1)
    from_number: str | None = None
    voice: str | None = None
    lead_id: str | None = None


class CallLeadResponse(BaseModel):
    call_id: str
    status: str
    provider_call_sid: str


class CallStatusResponse(BaseModel):
    call_id: str
    status: str
    outcome: str | None = None
    summary: str | None = None
    notes: str | None = None
    callback_at: str | None = None
    cost: float | None = None
    duration_ms: int | None = None


# --- Routes -------------------------------------------------------------


@router.post("/campaigns/{campaign_id}/calls", response_model=CallLeadResponse, status_code=202)
async def call_lead(
    campaign_id: str,
    req: CallLeadRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CallLeadResponse:
    campaign = await session.get(DbCampaign, campaign_id)
    if campaign is None or campaign.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="campaign not found")
    if campaign.status != "active":
        raise HTTPException(
            status_code=409, detail=f"campaign is {campaign.status!r}, not active")

    pipeline = tenant.settings.pipeline
    tel = pipeline.telephony
    provider = (tel.provider or "").lower()

    # webconsole tenants are tested in the browser, not dialed out.
    if provider == "webconsole":
        raise HTTPException(
            status_code=409,
            detail=("this tenant's telephony is 'webconsole' — it has no outbound "
                    "dialing. Test it from the browser console (/console or "
                    "/dev/voice); those sessions are still recorded + billed."))

    # Enforce the per-tenant concurrency cap.
    cap = tenant.settings.max_concurrent_calls
    if await count_active_calls(session, tenant.id) >= cap:
        raise HTTPException(
            status_code=429,
            detail=f"max concurrent calls reached ({cap}); retry when a call ends")

    # Per-tenant compliance gate: calling hours + DND, from the runtime registry
    # (skipped only in tests/contexts where the registry isn't wired).
    registry = getattr(request.app.state, "registry", None)
    if registry is not None:
        dnd = registry.dnd.get(tenant)
        if not dnd.hours.can_call_now():
            raise HTTPException(
                status_code=403, detail="outside this tenant's configured calling hours")
        if dnd.filter.is_blocked(req.to_number):
            raise HTTPException(
                status_code=403, detail="destination is on the tenant's DND list")

    from_number = req.from_number or (tel.outbound_from or {}).get(provider) or tel.from_number
    if not from_number:
        raise HTTPException(status_code=400, detail="no caller-ID configured for this tenant")
    if not tel.webhook_base_url:
        raise HTTPException(status_code=400, detail="tenant telephony.webhook_base_url must be set")

    # Dial with the TENANT's telephony creds (resolved for its configured
    # provider), not the platform env — otherwise every tenant's call bills/
    # identifies as the platform's account. A cred that isn't configured yet
    # resolves to None so the adapter can still fall back to env (until the
    # tenant's keys are migrated into the DB).
    def _cred(name):
        try:
            return tenant.secret(name) if name else None
        except Exception:  # noqa: BLE001 - missing env → let the adapter decide
            return None

    creds = tel.active_creds()
    acct, auth = _cred(creds.account_sid_env), _cred(creds.auth_token_env)
    # Stringee: the callout needs a non-null userId, or it goes out as a
    # phone->phone external call and the Answer URL/SCCO never runs (silent bot).
    uid = _cred(creds.user_id_env)
    try:
        adapter = get_telephony_provider({
            "provider": provider,
            "account_sid": acct, "auth_token": auth,
            # Stringee's server adapter reads api_key_sid/api_key_secret (its
            # account keys are stored as account_sid/auth_token) — pass them so it
            # dials with the tenant's creds, not STRINGEE_API_KEY_SID from env.
            "api_key_sid": acct, "api_key_secret": auth,
            "user_id": uid,
        })
    except Exception as e:  # noqa: BLE001 — e.g. missing credentials
        raise HTTPException(status_code=400, detail=f"telephony adapter unavailable: {e}")

    cfg = CallConfig(
        to_number=req.to_number.strip(),
        from_number=from_number,
        webhook_url=tel.webhook_base_url.rstrip("/"),
    )
    try:
        call_session = await adapter.initiate_call(cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("call lead failed", extra={"tenant": tenant.slug, "provider": provider})
        raise HTTPException(status_code=502, detail=f"call failed: {e}")

    call_id = f"call_{uuid.uuid4().hex[:16]}"
    await insert_call(
        session, call_id=call_id, tenant=tenant, provider_call_sid=call_session.session_id,
        channel="voice", campaign_id=campaign_id, lead_id=req.lead_id, voice=req.voice,
    )
    log.info("call lead placed", extra={
        "tenant": tenant.slug, "campaign": campaign_id, "call_id": call_id,
        "sid": call_session.session_id})
    return CallLeadResponse(
        call_id=call_id, status="in_progress", provider_call_sid=call_session.session_id)


@router.get("/calls/{call_id}", response_model=CallStatusResponse)
async def get_call(
    call_id: str,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CallStatusResponse:
    row = await session.get(Conversation, call_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="call not found")
    return CallStatusResponse(
        call_id=row.id, status=row.status, outcome=row.outcome,
        summary=row.summary, notes=row.notes,
        callback_at=row.callback_at.isoformat() if row.callback_at else None,
        cost=row.cost, duration_ms=row.duration_ms,
    )
