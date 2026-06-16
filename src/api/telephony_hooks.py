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

import json
import logging
from collections.abc import Callable

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.api.telephony_exotel import voicebot_xml
from src.api.telephony_stringee import reprompt_scco
from src.api.telephony_stringee_bridge import StringeeIvrBridge, registry
from src.api.telephony_twilio import softphone_dial_twiml, voice_twiml
from src.auth import TenantContext
from src.auth.middleware import tenant_from_slug, tenant_from_twilio_to_number

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony"])


# --- Browser softphone (human agent ↔ lead) deps ---------------------------
# The recording webhook needs the per-tenant provider registry (to build the
# tenant's STT + LLM for transcription + outcome analysis). main.py wires it in
# the lifespan; unset (tests without the full app) → the webhook 503s.
_softphone_providers: object | None = None


def set_softphone_providers(providers: object | None) -> None:
    """Register the per-tenant provider registry used by the recording webhook."""
    global _softphone_providers
    _softphone_providers = providers


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


# --- Browser softphone (human agent ↔ lead) ---------------------------------


def _forwarded_base(request: Request) -> str:
    """Public base URL for our telephony routes, honoring reverse-proxy headers."""
    base = request.headers.get("x-forwarded-host") or request.url.netloc
    proto = request.headers.get("x-forwarded-proto")
    scheme = "https" if (proto == "https" or request.url.scheme == "https") else "http"
    return f"{scheme}://{base}/api/v1/telephony"


async def _provider_caller_id(
    session: AsyncSession, tenant: TenantContext, provider: str
) -> str | None:
    """Outbound caller-ID for a provider's softphone leg.

    Prefers the tenant's number registered for ``provider`` in
    ``tenant_phone_numbers`` — a number that provider actually owns / dials out
    on (providers reject a caller-ID they don't own) — then falls back to the
    telephony config's ``outbound_from[provider]`` and finally ``from_number``.
    """
    from sqlalchemy import select

    from src.models.tenant import TenantPhoneNumber

    owned = (await session.execute(
        select(TenantPhoneNumber.phone_number).where(
            TenantPhoneNumber.tenant_id == tenant.id,
            TenantPhoneNumber.provider == provider,
        ).limit(1)
    )).scalars().first()
    tel = tenant.settings.pipeline.telephony
    return owned or (tel.outbound_from or {}).get(provider) or tel.from_number


@router.post("/twilio/softphone-twiml/{tenant_slug}", response_class=Response)
async def twilio_softphone_twiml(
    tenant_slug: str,
    request: Request,
    To: str = Form(...),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """TwiML App Voice URL for a tenant's browser softphone.

    Hit when a human agent's browser (Twilio Voice JS SDK) places a call: ``To``
    is the lead number the SDK passed; ``From`` is ``client:<identity>``. We log
    the call as a **manual** conversation row (so the recording webhook can
    finalize its outcome by CallSid) and return TwiML that dials the lead from
    the tenant's caller-ID with dual-channel recording.
    """
    import uuid

    from sqlalchemy import select

    from src.api.call_store import insert_call
    from src.models.conversation import Conversation

    tenant = await tenant_from_slug(tenant_slug)
    caller_id = await _provider_caller_id(session, tenant, "twilio")
    if not caller_id:
        raise HTTPException(status_code=400, detail="tenant has no Twilio caller-ID configured")

    if CallSid:
        existing = (await session.execute(
            select(Conversation).where(Conversation.provider_call_sid == CallSid)
        )).scalar_one_or_none()
        if existing is None:
            row = await insert_call(
                session, call_id=f"call_{uuid.uuid4().hex[:16]}", tenant=tenant,
                provider_call_sid=CallSid, channel="softphone", agent_type="human")
            row.notes = f"manual softphone call → {To}"
            await session.commit()

    base = _forwarded_base(request)
    cb = f"{base}/twilio/softphone-recording/{tenant_slug}"
    body = softphone_dial_twiml(to_number=To, caller_id=caller_id, recording_callback_url=cb)
    log.info("twilio softphone twiml", extra={
        "tenant": tenant.slug, "to": To, "from": From, "sid": CallSid})
    return Response(content=body, media_type="application/xml")


async def _download_twilio_recording(url: str, account_sid: str | None, auth_token: str | None) -> bytes:
    """Fetch a Twilio recording as a WAV. Twilio recordings require account auth."""
    wav_url = url if url.endswith(".wav") else f"{url}.wav"
    auth = (account_sid, auth_token) if (account_sid and auth_token) else None
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as c:
        resp = await c.get(wav_url, auth=auth)
        resp.raise_for_status()
        return resp.content


@router.post("/twilio/softphone-recording/{tenant_slug}", response_class=Response)
async def twilio_softphone_recording(
    tenant_slug: str,
    request: Request,
    CallSid: str | None = Form(None),
    RecordingUrl: str | None = Form(None),
    RecordingDuration: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Twilio recording-status callback → transcribe → analyze → outcome.

    Produces the **same** structured outcome an AI call does: the dual-channel
    recording is split (agent track + lead track), each transcribed with the
    tenant's STT, then the same ``analyze_call`` runs and the result is persisted
    via the same ``record_outcome`` keyed by CallSid.
    """
    from datetime import datetime, timezone

    from src.analysis.manual_call import finalize_manual_call
    from src.interfaces.stt import STTConfig
    from src.pipeline.audio_utils import pcm16_to_wav, wav_split_stereo

    tenant = await tenant_from_slug(tenant_slug)
    if not (CallSid and RecordingUrl):
        log.warning("twilio softphone recording missing CallSid/RecordingUrl")
        return Response(status_code=200)
    if _softphone_providers is None:
        log.warning("softphone recording webhook hit but provider registry unset")
        return Response(status_code=503)

    tel = tenant.settings.pipeline.telephony
    try:
        wav = await _download_twilio_recording(
            RecordingUrl, tenant.secret(tel.account_sid_env), tenant.secret(tel.auth_token_env))
        left, right, sr = wav_split_stereo(wav)
    except Exception:  # noqa: BLE001 — a fetch/parse failure must not 500 Twilio
        log.exception("twilio softphone recording fetch/split failed", extra={"sid": CallSid})
        return Response(status_code=200)

    # Twilio dual-channel <Dial>: channel 1 = the parent (human agent) leg,
    # channel 2 = the dialed lead. Map agent→assistant, lead→user.
    channels = [("assistant", pcm16_to_wav(left, sr)), ("user", pcm16_to_wav(right, sr))]
    stt = _softphone_providers.get_stt(tenant)
    llm = _softphone_providers.get_llm(tenant)
    stt_config = STTConfig(language=tenant.settings.pipeline.stt.language, sample_rate=sr)
    dur_ms = int(float(RecordingDuration) * 1000) if RecordingDuration else None

    row = await finalize_manual_call(
        session, provider_call_sid=CallSid, channels=channels,
        stt=stt, stt_config=stt_config, llm=llm,
        tenant_timezone=tenant.settings.timezone,
        now=datetime.now(timezone.utc), duration_ms=dur_ms, recording_url=RecordingUrl)
    log.info("twilio softphone recording finalized", extra={
        "tenant": tenant.slug, "sid": CallSid, "found": row is not None,
        "outcome": getattr(row, "outcome", None)})
    return Response(status_code=200)


# --- Stringee browser softphone --------------------------------------------
# A tenant points two project URLs at us (slug in the path → tenant routing):
#   Answer URL → /telephony/stringee/softphone-answer/<slug>
#   Event URL  → /telephony/stringee/softphone-recording/<slug>
# The agent's Web SDK call (client → PSTN) hits the Answer URL; we connect it to
# the lead with dual-channel recording, and the Event URL delivers the recording
# URL → transcribe → analyze → same outcome as AI calls.


def _stringee_recording_url(data: dict) -> str | None:
    for key in ("recordUrl", "recording_url", "recordingUrl", "url", "link",
                "fileUrl", "file_url"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    return None


@router.api_route("/stringee/softphone-answer/{tenant_slug}", methods=["GET", "POST"])
async def stringee_softphone_answer(
    tenant_slug: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    """Stringee answer webhook for a browser-softphone call → connect SCCO.

    Logs the call as a manual conversation row (keyed by the Stringee call_id)
    and returns an SCCO connecting the agent's client call to the lead from the
    tenant's DID, with dual-channel recording.
    """
    import uuid

    from sqlalchemy import select

    from src.api.call_store import insert_call
    from src.api.telephony_stringee import softphone_connect_scco
    from src.models.conversation import Conversation

    data = await _stringee_params(request)
    log.info("stringee softphone answer", extra={
        "tenant": tenant_slug, "method": request.method, "data": data})
    tenant = await tenant_from_slug(tenant_slug)
    # Caller-ID = the tenant's Stringee-owned number (Stringee rejects the connect
    # if `from` isn't a number it owns). This is the Stringee answer endpoint, so
    # the provider is always "stringee".
    from_number = await _provider_caller_id(session, tenant, "stringee")
    if not from_number:
        log.warning("stringee softphone: no caller-ID for tenant %s", tenant.slug)
        return Response(status_code=400)
    to_number = _stringee_number(data.get("to")) or _stringee_number(data.get("toNumber")) \
        or _stringee_number(data.get("called")) or _stringee_custom_destination(data)
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    if not to_number:
        log.warning("stringee softphone answer: no destination; keys=%s", sorted(data.keys()))
        return Response(status_code=404)

    if call_id:
        existing = (await session.execute(
            select(Conversation).where(Conversation.provider_call_sid == call_id)
        )).scalar_one_or_none()
        if existing is None:
            row = await insert_call(
                session, call_id=f"call_{uuid.uuid4().hex[:16]}", tenant=tenant,
                provider_call_sid=call_id, channel="softphone", agent_type="human")
            row.notes = f"manual softphone call → {to_number}"
            await session.commit()

    event_url = f"{_stringee_base(request)}/softphone-recording/{tenant_slug}"
    scco = softphone_connect_scco(
        from_number=from_number, to_number=to_number, event_url=event_url)
    return JSONResponse(scco)


@router.api_route("/stringee/softphone-recording/{tenant_slug}", methods=["GET", "POST"])
async def stringee_softphone_recording(
    tenant_slug: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    """Stringee event webhook → on a recording URL, transcribe → analyze → outcome.

    Tolerant of Stringee's field names (recording URL + call_id appear under
    several keys across versions). Non-recording events are acknowledged 200.
    """
    from datetime import datetime, timezone

    from src.analysis.manual_call import finalize_manual_call
    from src.interfaces.stt import STTConfig
    from src.pipeline.audio_utils import pcm16_to_wav, wav_split_stereo

    data = await _stringee_params(request)
    log.info("stringee softphone recording", extra={
        "tenant": tenant_slug, "method": request.method, "data": data})
    tenant = await tenant_from_slug(tenant_slug)
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    rec_url = _stringee_recording_url(data)
    if not (call_id and rec_url):
        return Response(status_code=200)   # not a recording event — ack and move on
    if _softphone_providers is None:
        log.warning("stringee softphone recording hit but provider registry unset")
        return Response(status_code=503)

    tel = tenant.settings.pipeline.telephony
    try:
        wav = await _download_stringee_recording(rec_url, tenant)
        left, right, sr = wav_split_stereo(wav)
    except Exception:  # noqa: BLE001 — a fetch/parse failure must not 500 Stringee
        log.exception("stringee softphone recording fetch/split failed", extra={"call_id": call_id})
        return Response(status_code=200)

    # Dual-channel connect recording: ch1 = caller (agent), ch2 = callee (lead).
    channels = [("assistant", pcm16_to_wav(left, sr)), ("user", pcm16_to_wav(right, sr))]
    stt = _softphone_providers.get_stt(tenant)
    llm = _softphone_providers.get_llm(tenant)
    stt_config = STTConfig(language=tenant.settings.pipeline.stt.language, sample_rate=sr)
    dur_raw = data.get("duration") or data.get("callDuration") or data.get("recordDuration")
    dur_ms = int(float(dur_raw) * 1000) if dur_raw else None

    row = await finalize_manual_call(
        session, provider_call_sid=call_id, channels=channels,
        stt=stt, stt_config=stt_config, llm=llm,
        tenant_timezone=tenant.settings.timezone,
        now=datetime.now(timezone.utc), duration_ms=dur_ms, recording_url=rec_url)
    log.info("stringee softphone recording finalized", extra={
        "tenant": tenant.slug, "call_id": call_id, "found": row is not None,
        "outcome": getattr(row, "outcome", None)})
    return Response(status_code=200)


async def _download_stringee_recording(url: str, tenant: TenantContext) -> bytes:
    """Fetch a Stringee recording, authenticating with a server token when the
    tenant's Stringee keys are available (signed/public URLs work without it)."""
    from src.providers.telephony.stringee import mint_server_token

    tel = tenant.settings.pipeline.telephony
    sid = tenant.secret(tel.account_sid_env) if tel.account_sid_env else None
    secret = tenant.secret(tel.auth_token_env) if tel.auth_token_env else None
    headers = {}
    if sid and secret:
        headers["X-STRINGEE-AUTH"] = mint_server_token(sid, secret)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as c:
        resp = await c.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


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


def _stringee_custom_destination(data: dict) -> str | None:
    """Pull the dialed number out of Stringee's ``customData``.

    For an app-to-phone softphone call Stringee does **not** forward the SDK's
    ``to`` to the answer webhook — it arrives empty with ``fromInternal=true``
    (confirmed by the live call debugger). The browser instead carries the lead
    number in ``customData``, delivered here as the ``custom`` param: usually a
    JSON string like ``{"to": "+91…"}``, occasionally a bare number.
    """
    raw = data.get("custom") or data.get("customData") \
        or data.get("customDataFromYourServer")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return _stringee_number(raw)  # a bare number string, not JSON
    if isinstance(raw, dict):
        return _stringee_number(
            raw.get("to") or raw.get("toNumber") or raw.get("number"))
    return None


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
