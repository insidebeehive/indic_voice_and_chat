"""Bridge console — UI for manually executing call transfers.

Simulates what the CRM coordination service does:
  1. Receive the transfer webhook when the AI voicebot fires the 'transfer' action.
  2. Create a Twilio conference bridge (caller + agent phone).
  3. Report success/failure back to the voicebot via transfer_store.

Routes (all behind dev_console_enabled gate):
  GET  /dev/bridge                       → serve bridge_console.html
  GET  /dev/bridge/events?token=         → SSE stream (real-time transfer events)
  POST /dev/bridge/place-call            → place outbound call with bridge webhook URL
  POST /dev/bridge/transfer-webhook      → receives transfer event from voicebot
  POST /dev/bridge/execute-transfer      → create Twilio conference + call agent phone
  POST /dev/bridge/reject-transfer       → reject the transfer (voicebot plays apology)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter()

_STATIC = Path(__file__).parent.parent.parent / "static"

# ── Per-page-session state ────────────────────────────────────────────────────


@dataclass
class _BridgeSession:
    token: str
    call_sid: str | None = None
    provider: str = "twilio"
    from_number: str = ""
    agent_phone: str = ""
    tenant_slug: str = "dev"
    account_sid: str | None = None
    auth_token: str | None = None
    sse_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


_sessions: dict[str, _BridgeSession] = {}


def _get_session(token: str) -> _BridgeSession:
    if token not in _sessions:
        _sessions[token] = _BridgeSession(token=token)
    return _sessions[token]


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/dev/bridge")
async def bridge_console_page() -> FileResponse:
    return FileResponse(_STATIC / "bridge_console.html")


@router.get("/dev/bridge/events")
async def bridge_events(token: str = Query(...)) -> StreamingResponse:
    """SSE stream — pushes JSON events to the browser as calls progress."""
    sess = _get_session(token)

    async def _generate():
        while True:
            try:
                event = await asyncio.wait_for(sess.sse_queue.get(), timeout=25.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Place call ────────────────────────────────────────────────────────────────


class PlaceBridgeCallRequest(BaseModel):
    token: str
    provider: str = "twilio"
    to_number: str
    agent_phone: str
    tenant: str = "dev"
    mode: str = "s2s"


@router.post("/dev/bridge/place-call")
async def bridge_place_call(req: PlaceBridgeCallRequest) -> dict:
    """Place an outbound call via dev_console machinery, but with the transfer
    webhook URL pointing at this bridge console so we receive the event."""
    from src.api.dev_console import PlaceCallRequest, dev_place_call
    from src.config_tenant import platform_webhook_base_url
    from src.auth.middleware import tenant_from_slug
    from src.api import dev_call_control
    from src.api.answer_paths import ANSWER_PATHS
    from src.pipeline.telephony.providers import get_telephony_provider
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

    # Resolve from_number (same logic as dev_console.dev_place_call)
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

    pcreds = tel.creds_for(provider)
    acct = _cred(pcreds.account_sid_env)
    auth = _cred(pcreds.auth_token_env)

    # Store session state so execute-transfer can reuse credentials
    sess = _get_session(req.token)
    sess.provider = provider
    sess.agent_phone = req.agent_phone.strip()
    sess.tenant_slug = req.tenant
    sess.from_number = from_number
    sess.account_sid = acct
    sess.auth_token = auth

    # Build telephony adapter
    try:
        adapter = get_telephony_provider({
            "provider": provider,
            "account_sid": acct, "auth_token": auth,
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"telephony adapter unavailable: {e}")

    # The transfer webhook URL points to this bridge console so we receive the event
    transfer_webhook_url = (
        f"{webhook_base.rstrip('/')}/dev/bridge/transfer-webhook?token={req.token}"
    )

    # Set dev override — the bridge factory will consume this when the call connects
    dev_call_control.set_override(
        tenant.slug,
        mode=req.mode,
        voice="",
        caller_name="",
        lead_name="",
        lead_gender="",
        transfer_webhook_url=transfer_webhook_url,
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

    sess.call_sid = session.session_id
    from src.api import dev_call_control as _dcc
    _dcc.monitor.set_status(session.session_id, "calling")

    log.info("bridge console: call placed",
             extra={"call_sid": session.session_id, "provider": provider,
                    "transfer_webhook": transfer_webhook_url})
    return {"call_sid": session.session_id, "provider": provider}


# ── Transfer webhook (receives call.transfer_requested from voicebot) ─────────


class TransferWebhookPayload(BaseModel):
    event: str = ""
    call_sid: str = ""
    transfer_result_url: str = ""


@router.post("/dev/bridge/transfer-webhook")
async def bridge_transfer_webhook(
    payload: TransferWebhookPayload,
    token: str = Query(...),
) -> dict:
    """Receives the 'call.transfer_requested' webhook from TelephonyLiveBridge."""
    sess = _get_session(token)
    if payload.call_sid:
        sess.call_sid = payload.call_sid

    sess.sse_queue.put_nowait({
        "type": "transfer_requested",
        "call_sid": payload.call_sid,
        "transfer_result_url": payload.transfer_result_url,
    })
    log.info("bridge console: transfer webhook received",
             extra={"token": token, "call_sid": payload.call_sid})
    return {"ok": True}


# ── Execute transfer (create conference + call agent phone) ───────────────────


class ExecuteTransferRequest(BaseModel):
    token: str
    call_sid: str


@router.post("/dev/bridge/execute-transfer")
async def bridge_execute_transfer(req: ExecuteTransferRequest) -> dict:
    """Create a Twilio conference, redirect the caller into it, and call the
    agent phone — simulating what the CRM coordination service does."""
    sess = _get_session(req.token)
    call_sid = req.call_sid or sess.call_sid
    if not call_sid:
        raise HTTPException(status_code=400, detail="call_sid missing")

    if sess.provider != "twilio":
        raise HTTPException(
            status_code=400,
            detail="conference bridge is currently supported for Twilio only",
        )
    if not sess.account_sid or not sess.auth_token:
        raise HTTPException(status_code=400, detail="Twilio credentials not in session")
    if not sess.agent_phone:
        raise HTTPException(status_code=400, detail="agent_phone not set")

    conf_name = f"bridge-{call_sid}"
    conf_twiml = (
        f"<Response><Dial>"
        f"<Conference waitUrl='' beep='false'>{conf_name}</Conference>"
        f"</Dial></Response>"
    )

    try:
        from twilio.rest import Client as TwilioClient
        client = TwilioClient(sess.account_sid, sess.auth_token)

        # Redirect the caller's existing call leg into the conference
        client.calls(call_sid).update(twiml=conf_twiml)
        log.info("bridge: caller leg redirected to conference",
                 extra={"call_sid": call_sid, "conf": conf_name})

        # Dial the agent phone into the same conference
        agent_call = client.calls.create(
            to=sess.agent_phone,
            from_=sess.from_number,
            twiml=conf_twiml,
        )
        log.info("bridge: agent leg dialled",
                 extra={"agent_call_sid": agent_call.sid, "conf": conf_name})

    except Exception as e:
        log.exception("bridge: execute-transfer failed")
        raise HTTPException(status_code=502, detail=f"bridge creation failed: {e}")

    # Resolve the in-process transfer Future directly — no HTTP round-trip needed
    from src.api.transfer_store import resolve
    resolve(call_sid, "success")

    sess.sse_queue.put_nowait({
        "type": "bridge_created",
        "call_sid": call_sid,
        "conference": conf_name,
        "agent_call_sid": agent_call.sid,
    })
    return {"ok": True, "conference": conf_name, "agent_call_sid": agent_call.sid}


# ── Reject transfer ───────────────────────────────────────────────────────────


class RejectTransferRequest(BaseModel):
    token: str
    call_sid: str


@router.post("/dev/bridge/reject-transfer")
async def bridge_reject_transfer(req: RejectTransferRequest) -> dict:
    """Reject the transfer — voicebot will play an apology and end the call."""
    call_sid = req.call_sid
    if not call_sid:
        raise HTTPException(status_code=400, detail="call_sid missing")

    from src.api.transfer_store import resolve
    resolve(call_sid, "failure")

    sess = _get_session(req.token)
    sess.sse_queue.put_nowait({"type": "transfer_rejected", "call_sid": call_sid})
    log.info("bridge: transfer rejected", extra={"call_sid": call_sid})
    return {"ok": True}
