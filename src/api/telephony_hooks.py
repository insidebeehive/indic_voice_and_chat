"""Telephony provider webhook endpoints (PRD §7.4).

Twilio voice webhook + Media Streams websocket, tenant-aware.

Inbound flow:
1. Twilio rings the called number and POSTs to ``/twilio/voice`` with the
   ``To`` form param.
2. The voice handler resolves the tenant from the ``To`` number, builds
   the TwiML response with a ``<Stream url="wss://.../stream?tenant=<slug>"/>``
   so the websocket leg can re-resolve the same tenant.
3. Twilio opens the WebSocket; the WS handler reads ``?tenant=`` from the
   query string and asks the registered bridge factory for a per-call
   bridge wired with the tenant's agent + provider stack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
from fastapi import APIRouter, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from src.api.telephony_exotel import voicebot_xml
from src.api.telephony_stringee import reprompt_scco
from src.api.telephony_stringee_bridge import StringeeIvrBridge, registry
from src.api.telephony_twilio import voice_twiml
from src.auth import TenantContext
from src.auth.middleware import tenant_from_twilio_to_number

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony"])


# Factory: takes (websocket, tenant_context) -> bridge instance with .run()
BridgeFactory = Callable[[WebSocket, TenantContext], object]
_bridge_factory: BridgeFactory | None = None
_exotel_bridge_factory: BridgeFactory | None = None


def set_bridge_factory(factory: BridgeFactory | None) -> None:
    """Register the Twilio bridge factory."""
    global _bridge_factory
    _bridge_factory = factory


def set_exotel_bridge_factory(factory: BridgeFactory | None) -> None:
    """Register the Exotel bridge factory."""
    global _exotel_bridge_factory
    _exotel_bridge_factory = factory


_stringee_bridge_factory: Callable[..., StringeeIvrBridge] | None = None


def set_stringee_bridge_factory(factory) -> None:
    """Register the Stringee IVR bridge factory."""
    global _stringee_bridge_factory
    _stringee_bridge_factory = factory


# Outbound SIP-trunk (DiDLogic) bridge factory: async (tenant, to_number) -> SipMediaBridge.
# Set during app startup; used by Call Lead for telephony provider "didlogic".
_sip_bridge_factory: Callable | None = None


def set_sip_bridge_factory(factory) -> None:
    global _sip_bridge_factory
    _sip_bridge_factory = factory


def get_sip_bridge_factory():
    return _sip_bridge_factory


@router.post("/twilio/voice", response_class=Response)
async def twilio_voice(
    request: Request,
    To: str = Form(...),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Twilio voice webhook → returns TwiML opening a tenant-aware media stream.

    Tenant resolution is direction-aware:
    - **inbound** (a customer dials our Twilio number): ``To`` is the
      owned number; look it up in ``tenant_phone_numbers``.
    - **outbound-api / outbound-dial** (we initiated the call via the
      orchestrator or place_test_call.py): ``To`` is the dialed destination
      (an end-user number we don't own); the tenant owns ``From`` instead.

    The resolved slug is embedded in the WS URL so the stream handler can
    re-resolve the same tenant on connect.
    """
    is_outbound = (Direction or "").startswith("outbound")
    lookup_number = From if (is_outbound and From) else To
    tenant = await tenant_from_twilio_to_number(lookup_number)

    base = request.headers.get("x-forwarded-host") or request.url.netloc
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = "wss" if (forwarded_proto == "https" or request.url.scheme == "https") else "ws"
    # Tenant slug goes in the URL **path** — Twilio strips query strings
    # from <Stream url=...> attributes when opening the WSS connection.
    stream_url = f"{scheme}://{base}/api/v1/telephony/twilio/stream/{tenant.slug}"
    body = voice_twiml(stream_url)
    log.info(
        "twilio voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.websocket("/twilio/stream/{tenant_slug}")
async def twilio_stream(websocket: WebSocket, tenant_slug: str) -> None:
    """Twilio Media Streams websocket → bridges audio to the tenant's agent.

    Tenant slug arrives as a path segment because Twilio strips query
    strings from ``<Stream url=...>`` attributes when establishing the
    WSS connection.
    """
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("twilio stream tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
        return

    if _bridge_factory is None:
        log.warning("twilio stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="bridge factory unset")
        return

    bridge = _bridge_factory(websocket, tenant)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("twilio stream client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("twilio stream bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# --- Exotel webhook + WS ----------------------------------------------------


@router.post("/exotel/voice", response_class=Response)
async def exotel_voice(
    request: Request,
    To: str = Form(...),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Exotel Passthru / Voicebot webhook → returns ExotelML opening a stream.

    Exotel's form params mirror Twilio's (``To``, ``From``, ``CallSid``,
    ``Direction``) so the resolution logic is identical: for outbound calls
    the tenant owns ``From``; for inbound it owns ``To``.
    """
    is_outbound = (Direction or "").startswith("outbound")
    lookup_number = From if (is_outbound and From) else To
    tenant = await tenant_from_twilio_to_number(lookup_number)

    base = request.headers.get("x-forwarded-host") or request.url.netloc
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = "wss" if (forwarded_proto == "https" or request.url.scheme == "https") else "ws"
    stream_url = f"{scheme}://{base}/api/v1/telephony/exotel/stream/{tenant.slug}"
    body = voicebot_xml(stream_url)
    log.info(
        "exotel voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.websocket("/exotel/stream/{tenant_slug}")
async def exotel_stream(websocket: WebSocket, tenant_slug: str) -> None:
    """Exotel Voicebot Streaming websocket → bridges audio to the agent."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("exotel stream tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
        return

    if _exotel_bridge_factory is None:
        log.warning("exotel stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="exotel bridge factory unset")
        return

    bridge = _exotel_bridge_factory(websocket, tenant)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("exotel stream client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("exotel stream bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# --- Stringee IVR webhook routes --------------------------------------------


def _stringee_base(request: Request) -> str:
    base = request.headers.get("x-forwarded-host") or request.url.netloc
    proto = request.headers.get("x-forwarded-proto")
    scheme = "https" if (proto == "https" or request.url.scheme == "https") else "http"
    return f"{scheme}://{base}/api/v1/telephony/stringee"


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=5.0)) as c:
        resp = await c.get(url)
        resp.raise_for_status()
        return resp.content


def _stringee_number(value: object) -> str | None:
    """Pull a phone number out of a webhook field that may be a bare string,
    an object (``{"number": ...}`` — the shape our callout uses), or a list of
    those. The exact answer-webhook shape is confirmed by the first live call.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        return value.get("number") or value.get("e164") or value.get("alias")
    return value if isinstance(value, str) else None


async def _stringee_params(request: Request) -> dict:
    """Merge a Stringee webhook's data regardless of method. Stringee fetches
    the answer_url via GET (call info in the query string); other hooks POST
    JSON. We read both so the routes work either way."""
    data = dict(request.query_params)
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - tolerate empty/non-JSON bodies
            body = None
        if isinstance(body, dict):
            data = {**data, **body}
    return data


async def _resolve_stringee_tenant(data: dict):
    """Resolve the tenant by trying every number field (both from and to). The
    caller-id is the registered number for outbound, the called number for
    inbound — so trying both is robust to whether Stringee sends a direction."""
    for key in ("from", "fromNumber", "caller", "to", "toNumber", "called"):
        num = _stringee_number(data.get(key))
        if not num:
            continue
        try:
            return await tenant_from_twilio_to_number(num)
        except Exception:  # noqa: BLE001 - not this number; try the next
            continue
    return None


@router.api_route("/stringee/answer", methods=["GET", "POST"])
async def stringee_answer(request: Request):
    """Call answered -> build the call's bridge and return the opening SCCO.

    Stringee fetches the answer_url via **GET** (call info in the query string);
    we accept POST/JSON too. The whole request is logged so the first live call
    reveals Stringee's real field names.
    """
    data = await _stringee_params(request)
    log.info("stringee answer", extra={"method": request.method, "data": data})
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    if not call_id:
        log.warning("stringee answer missing call_id; keys=%s", sorted(data.keys()))
    tenant = await _resolve_stringee_tenant(data)
    if tenant is None:
        log.warning("stringee answer: no tenant for any number; keys=%s", sorted(data.keys()))
        return Response(status_code=404)
    if _stringee_bridge_factory is None:
        return Response(status_code=503)
    bridge = _stringee_bridge_factory(
        call_id=call_id, tenant=tenant,
        base_url=_stringee_base(request), fetch=_download,
    )
    registry.put(bridge)
    scco = await bridge.start_call()
    if call_id:
        from src.api import dev_call_control
        dev_call_control.monitor.set_status(call_id, "answered")
    log.info("stringee answer registered", extra={"tenant": tenant.slug, "call_id": call_id})
    return JSONResponse(scco)


@router.api_route("/stringee/event/{tenant_slug}", methods=["GET", "POST"])
async def stringee_event(tenant_slug: str, request: Request, call_id: str | None = None):
    """Per-turn recordMessage webhook -> run a turn -> return the next SCCO.

    Note: tenant_slug is URL namespacing only; the bridge is looked up by
    call_id — tenant_slug is not validated here.
    call_id is optional so that a missing ?call_id= query param returns a
    graceful reprompt (200) instead of a FastAPI 422.
    """
    data = await _stringee_params(request)
    log.info("stringee event", extra={"tenant": tenant_slug, "call_id": call_id,
                                       "method": request.method, "data": data})
    rec_url = (
        data.get("recording_url")
        or data.get("url")
        or data.get("link")
        or data.get("fileUrl")
        or data.get("file_url")
        or data.get("recordingUrl")
    )
    bridge = registry.get(call_id) if call_id else None
    if bridge is None or not rec_url:
        base = _stringee_base(request)
        return JSONResponse(reprompt_scco(
            text="Maaf kijiye, dobara boliye?",
            event_url=f"{base}/event/{tenant_slug}?call_id={call_id or ''}",
        ))
    scco = await bridge.handle_turn(recording_url=rec_url)
    return JSONResponse(scco)


@router.get("/stringee/audio/{token}")
async def stringee_audio(token: str, call_id: str | None = None):
    """Serve a hosted reply/opening WAV for Stringee's `play` to fetch."""
    wav = None
    if call_id:
        bridge = registry.get(call_id)
        if bridge is not None:
            wav = bridge.audio.get(token)
    if wav is None:
        # Stringee may fetch without our call_id query: scan live calls.
        for b in registry.iter_bridges():
            wav = b.audio.get(token)
            if wav is not None:
                break
    log.info("stringee audio fetch", extra={"token": token, "call_id": call_id,
             "found": wav is not None, "bytes": len(wav) if wav else 0})
    if wav is not None:
        return Response(content=wav, media_type="audio/wav")
    return Response(status_code=404)


@router.api_route("/stringee/status/{tenant_slug}", methods=["GET", "POST"])
async def stringee_status(tenant_slug: str, request: Request):
    """Lifecycle webhook: on call end, record the outcome and clean up."""
    data = await _stringee_params(request)
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    status = (data.get("status") or data.get("event") or data.get("call_status") or "").upper()
    log.info("stringee status", extra={"call_id": call_id, "status": status,
                                        "method": request.method, "data": data})
    if status in ("ENDED", "FAILED", "NO_ANSWER", "BUSY"):
        await registry.end(call_id)
        if call_id:
            from src.api import dev_call_control
            dev_call_control.monitor.set_status(call_id, "ended")
    return Response(status_code=200)
