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

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api import telephony_hooks
from src.api.call_store import count_active_calls, insert_call, record_outcome
from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.interfaces.telephony import CallConfig
from src.models.campaign import Campaign as DbCampaign
from src.models.conversation import Conversation
from src.models.database import get_sessionmaker
from src.providers import get_telephony_provider
from src.providers.telephony.sip.transport import SipError

log = logging.getLogger(__name__)
router = APIRouter(tags=["calls"])

# Background SIP call tasks (held so they aren't garbage-collected mid-call).
_sip_tasks: set = set()


def _parse_iso(value):
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


async def _run_sip_call_task(bridge, call_id: str) -> None:
    """Run an outbound SIP call's bridge to completion, then persist its outcome
    + cost to the conversation row. Runs as a detached background task (Call Lead
    returns the call_id immediately — the call is async)."""
    try:
        await bridge.run()
    except Exception:  # noqa: BLE001
        log.exception("sip call bridge crashed", extra={"call_id": call_id})
    finally:
        payload = getattr(bridge, "_outcome_payload", None) or {}
        try:
            async with get_sessionmaker()() as s:
                await record_outcome(
                    s, call_id, status="ended",
                    outcome=payload.get("outcome"), summary=payload.get("summary"),
                    notes=payload.get("notes"),
                    callback_at=_parse_iso(payload.get("callback_datetime")))
        except Exception:  # noqa: BLE001
            log.exception("sip call finalize failed", extra={"call_id": call_id})


async def _place_sip_call(session, tenant, campaign_id, req) -> "CallLeadResponse":
    """Place an outbound call over a SIP trunk (DiDLogic) and run the agent."""
    factory = telephony_hooks.get_sip_bridge_factory()
    if factory is None:
        raise HTTPException(status_code=503, detail="SIP bridge not initialized")
    try:
        bridge = await factory(tenant, req.to_number.strip())   # sends the INVITE
    except SipError as e:
        log.exception("sip call failed", extra={"tenant": tenant.slug})
        raise HTTPException(status_code=502, detail=f"SIP call failed: {e}")
    except Exception as e:  # noqa: BLE001 — e.g. missing creds / no realtime config
        raise HTTPException(status_code=400, detail=f"SIP call setup failed: {e}")

    call_id = f"call_{uuid.uuid4().hex[:16]}"
    await insert_call(
        session, call_id=call_id, tenant=tenant, provider_call_sid=call_id,
        channel="telephony", campaign_id=campaign_id, lead_id=req.lead_id,
        voice=req.voice, mode="s2s")   # SIP outbound runs the realtime (s2s) path
    task = asyncio.create_task(_run_sip_call_task(bridge, call_id))
    _sip_tasks.add(task)
    task.add_done_callback(_sip_tasks.discard)
    log.info("sip call lead placed", extra={
        "tenant": tenant.slug, "campaign": campaign_id, "call_id": call_id})
    return CallLeadResponse(call_id=call_id, status="in_progress", provider_call_sid=call_id)


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

    # Raw SIP trunk (e.g. DiDLogic): no REST/WS — place the INVITE + run the agent
    # over RTP in-process via the SIP bridge factory.
    if provider == "didlogic":
        return await _place_sip_call(session, tenant, campaign_id, req)

    from_number = req.from_number or (tel.outbound_from or {}).get(provider) or tel.from_number
    if not from_number:
        raise HTTPException(status_code=400, detail="no caller-ID configured for this tenant")
    if not tel.webhook_base_url:
        raise HTTPException(status_code=400, detail="tenant telephony.webhook_base_url must be set")

    try:
        adapter = get_telephony_provider({"provider": provider})
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
