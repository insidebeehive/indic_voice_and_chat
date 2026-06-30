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

import asyncio
import inspect
import json
import logging
from collections.abc import Callable

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
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


# Sessionmaker for the softphone answer's background logging task (it needs its
# own session — the request's is closed once the fast response is sent). Defaults
# to the process-wide sessionmaker; tests inject an in-memory one.
_softphone_sessionmaker: object | None = None


def set_softphone_sessionmaker(sessionmaker: object | None) -> None:
    """Register the sessionmaker used by the answer webhook's background logger."""
    global _softphone_sessionmaker
    _softphone_sessionmaker = sessionmaker


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


async def prewarm_stringee_call(call_id: str, tenant) -> None:
    """Pre-synthesize the opening audio during the ringing phase so it's instant on answer.

    Called as a background task right after a Stringee outbound call is placed.
    The ringing window (~8-15s) is enough to synthesize the greeting TTS; the
    bridge is stored in the registry so _stringee_answer reuses it.
    """
    if _stringee_bridge_factory is None:
        return
    from src.config_tenant import platform_webhook_base_url as _plat_wb
    raw_base = _plat_wb() or None
    if not raw_base:
        log.warning("stringee prewarm: no platform webhook_base_url set, skipping")
        return
    stringee_base = raw_base.rstrip("/") + "/stringee"
    try:
        bridge = _stringee_bridge_factory(
            call_id=call_id, tenant=tenant, base_url=stringee_base, fetch=_download)
        if inspect.isawaitable(bridge):
            bridge = await bridge
        from src.api.telephony_stringee_bridge import registry
        registry.put(bridge)
        await bridge.prewarm()
    except Exception:
        log.exception("stringee prewarm failed", extra={"call_id": call_id})


def _ws_stream_url(request: Request, path: str) -> str:
    """Build the media-stream WSS URL for ``/api/v1/telephony/{path}``, honoring
    the reverse-proxy forwarded host/proto (Northflank terminates TLS upstream)."""
    base = request.headers.get("x-forwarded-host") or request.url.netloc
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = "wss" if (forwarded_proto == "https" or request.url.scheme == "https") else "ws"
    return f"{scheme}://{base}/api/v1/telephony/{path}"


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

    # Tenant slug goes in the URL **path** — Twilio strips query strings
    # from <Stream url=...> attributes when opening the WSS connection.
    stream_url = _ws_stream_url(request, f"twilio/stream/{tenant.slug}")
    body = voice_twiml(stream_url)
    log.info(
        "twilio voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.post("/twilio/voice/{tenant_slug}", response_class=Response)
async def twilio_voice_for_tenant(
    request: Request,
    tenant_slug: str,
    To: str | None = Form(None),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Slug-scoped answer URL for **outbound** calls we place ourselves (dev
    console / orchestrator). The placing tenant is already known, so we resolve
    by slug and skip the caller-ID lookup — the outbound caller-ID need NOT be
    registered in ``tenant_phone_numbers`` (mirrors ``/stringee/answer/{slug}``).
    Inbound calls keep using the bare ``/twilio/voice`` (resolve by number)."""
    from src.auth.middleware import tenant_from_slug

    tenant = await tenant_from_slug(tenant_slug)
    # Embed CallSid in the stream path so the bridge factory can look up
    # per-call overrides (voice/caller_name/lead_name/lead_gender). Twilio
    # strips query strings from <Stream url=...> so the SID must be in the path.
    stream_path = (f"twilio/stream/{tenant.slug}/{CallSid}"
                   if CallSid else f"twilio/stream/{tenant.slug}")
    stream_url = _ws_stream_url(request, stream_path)
    log.info(
        "twilio voice webhook (slug-scoped)",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=voice_twiml(stream_url), media_type="application/xml")


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
    if inspect.isawaitable(bridge):
        bridge = await bridge
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


@router.websocket("/twilio/stream/{tenant_slug}/{call_sid}")
async def twilio_stream_with_sid(
    websocket: WebSocket, tenant_slug: str, call_sid: str
) -> None:
    """Outbound-call variant: call_sid in path lets the factory look up
    per-call overrides (voice / caller_name / lead params) keyed by SID."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("twilio stream (sid) tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
        return

    if _bridge_factory is None:
        log.warning("twilio stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="bridge factory unset")
        return

    bridge = _bridge_factory(websocket, tenant)
    if inspect.isawaitable(bridge):
        bridge = await bridge
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

    from src.api.call_store import record_outcome
    from src.campaign.models import LeadCallOutcome

    c = tenant.settings.pipeline.telephony.active_creds()
    try:
        wav = await _download_twilio_recording(
            RecordingUrl, tenant.secret(c.account_sid_env), tenant.secret(c.auth_token_env))
        left, right, sr = wav_split_stereo(wav)
    except Exception:  # noqa: BLE001 — a fetch/parse failure must not 500 Twilio
        log.exception("twilio softphone recording fetch/split failed", extra={"sid": CallSid})
        await record_outcome(
            session, CallSid,
            status="ended",
            outcome=LeadCallOutcome.RECORDING_UNAVAILABLE.value,
            summary="Call recording could not be retrieved for analysis.",
            duration_ms=int(float(RecordingDuration) * 1000) if RecordingDuration else None,
        )
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


async def _log_softphone_call(tenant: TenantContext, provider_call_sid: str, to_number: str) -> None:
    """Insert the manual-call row for a Stringee softphone call (background).

    Runs AFTER the answer response is sent so the DB write never adds latency to
    Stringee's answer-URL fetch (a slow answer response is dropped as
    REQUEST_ANSWER_URL_ERROR). Opens its own session — the request's is closed.
    """
    import uuid

    from sqlalchemy import select

    from src.api.call_store import insert_call, mark_answered
    from src.models.conversation import Conversation
    from src.models.database import get_sessionmaker

    sessionmaker = _softphone_sessionmaker or get_sessionmaker()
    try:
        async with sessionmaker() as session:
            existing = (await session.execute(
                select(Conversation).where(Conversation.provider_call_sid == provider_call_sid)
            )).scalar_one_or_none()
            if existing is None:
                row = await insert_call(
                    session, call_id=f"call_{uuid.uuid4().hex[:16]}", tenant=tenant,
                    provider_call_sid=provider_call_sid, channel="softphone", agent_type="human")
                row.notes = f"manual softphone call → {to_number}"
                await session.commit()
            # The softphone answer IS the connect → mark answered (call.answered).
            await mark_answered(session, provider_call_sid)
    except Exception:  # noqa: BLE001 — logging must never break; the call already connected
        log.exception("stringee softphone: manual-call logging failed", extra={
            "call_sid": provider_call_sid})


@router.api_route("/stringee/softphone-answer/{tenant_slug}", methods=["GET", "POST"])
async def stringee_softphone_answer(
    tenant_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
):
    """Stringee answer webhook for a browser-softphone call → connect SCCO.

    Returns the SCCO **immediately** (Stringee's answer-URL fetch has a tight
    timeout; a slow response is dropped). The manual-call row is logged in a
    background task so the DB write doesn't block the response.
    """
    from src.api.telephony_stringee import softphone_connect_scco

    data = await _stringee_params(request)
    log.info("stringee softphone answer", extra={
        "tenant": tenant_slug, "method": request.method, "data": data})
    tenant = await tenant_from_slug(tenant_slug)
    # The connect `from` is the caller-ID NUMBER (type "internal"), bare. The
    # browser places the call FROM this number, so it arrives as the request's
    # `from`; use it (keeps the makeCall `from` and the SCCO `from` identical, as
    # in Stringee's working call). Fall back to the tenant's configured caller-ID.
    req_from = str(data.get("from") or "").lstrip("+")
    caller_id = req_from if req_from.isdigit() else \
        (await _provider_caller_id(session, tenant, "stringee") or "").lstrip("+")
    if not caller_id:
        log.warning("stringee softphone: no caller-ID for tenant %s", tenant.slug)
        return Response(status_code=400)
    to_number = _stringee_number(data.get("to")) or _stringee_number(data.get("toNumber")) \
        or _stringee_number(data.get("called")) or _stringee_custom_destination(data)
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    if not to_number:
        log.warning("stringee softphone answer: no destination; keys=%s", sorted(data.keys()))
        return Response(status_code=404)

    # Log the manual call AFTER responding — never block Stringee's answer fetch.
    if call_id:
        background_tasks.add_task(_log_softphone_call, tenant, call_id, to_number)

    event_url = f"{_stringee_base(request)}/softphone-recording/{tenant_slug}"
    scco = softphone_connect_scco(
        caller_id=caller_id, to_number=to_number, event_url=event_url)
    return JSONResponse(scco)


@router.api_route("/stringee/softphone-recording/{tenant_slug}", methods=["GET", "POST"])
async def stringee_softphone_recording(
    tenant_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Stringee event webhook → on a recording URL, transcribe → analyze → outcome.

    The fetch + transcribe + analyze runs in a BACKGROUND task: recordings lag the
    event (the download retries on 404) and STT/LLM take time, so Stringee's
    webhook gets a fast 200 instead of being held open. Tolerant of Stringee's
    field names; non-recording events are acknowledged 200.
    """
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
    dur_raw = data.get("duration") or data.get("callDuration") or data.get("recordDuration")
    dur_ms = int(float(dur_raw) * 1000) if dur_raw else None
    background_tasks.add_task(_finalize_softphone_recording, tenant, call_id, rec_url, dur_ms)
    return Response(status_code=200)


def _recording_mime(audio: bytes) -> str:
    """Sniff the recording's content type for the multimodal transcriber."""
    if audio[:4] == b"RIFF":
        return "audio/wav"
    return "audio/mpeg"   # Stringee records mono mp3


def _audio_transcriber(tenant_llm):
    """An LLM that can transcribe audio. Use the tenant's analysis LLM if it can
    (Gemini tenants); otherwise (e.g. Groq) fall back to a platform Gemini
    transcriber from GEMINI_API_KEY — Gemini handles Indian languages + the long
    mono mp3. Returns None if no audio-capable transcriber is available."""
    if getattr(tenant_llm, "transcribe_audio", None):
        return tenant_llm
    from src.providers import get_llm_provider
    try:
        return get_llm_provider({"provider": "gemini"})  # GEMINI_API_KEY from env
    except Exception:  # noqa: BLE001
        log.warning("softphone recording: no Gemini transcriber — set the platform GEMINI_API_KEY")
        return None


async def _mark_recording_unavailable(call_id: str, dur_ms: int | None) -> None:
    """Persist recording-unavailable outcome when the recording can't be fetched or
    transcribed. Opens its own session — used from background tasks."""
    from src.api.call_store import record_outcome
    from src.campaign.models import LeadCallOutcome
    from src.models.database import get_sessionmaker

    sm = _softphone_sessionmaker or get_sessionmaker()
    try:
        async with sm() as session:
            await record_outcome(
                session, call_id,
                status="ended",
                outcome=LeadCallOutcome.RECORDING_UNAVAILABLE.value,
                summary="Call recording could not be retrieved for analysis.",
                duration_ms=dur_ms,
            )
    except Exception:  # noqa: BLE001
        log.exception("failed to mark recording-unavailable", extra={"call_id": call_id})


async def _finalize_softphone_recording(
    tenant: TenantContext, call_id: str, rec_url: str, dur_ms: int | None,
) -> None:
    """Fetch the recording (retry-on-404), transcribe it with the tenant's
    multimodal LLM (Gemini — strong on Indian languages, handles the long mono mp3
    the turn-STT can't), analyze, and persist the outcome. Background task → own
    DB session."""
    from datetime import datetime, timezone

    from src.analysis.manual_call import finalize_manual_call
    from src.interfaces.llm import LLMMessage
    from src.models.database import get_sessionmaker

    try:
        audio = await _download_stringee_recording(rec_url, tenant)
    except Exception:  # noqa: BLE001 — a fetch failure must not break anything
        log.exception("stringee softphone recording fetch failed", extra={"call_id": call_id})
        await _mark_recording_unavailable(call_id, dur_ms)
        return

    llm = _softphone_providers.get_llm(tenant)
    transcriber = _audio_transcriber(llm)
    if transcriber is None:
        log.warning("softphone recording: no audio transcriber; marking recording-unavailable",
                    extra={"call_id": call_id})
        await _mark_recording_unavailable(call_id, dur_ms)
        return
    try:
        text = await transcriber.transcribe_audio(audio, _recording_mime(audio))
    except Exception:  # noqa: BLE001
        log.exception("stringee softphone recording transcription failed", extra={"call_id": call_id})
        await _mark_recording_unavailable(call_id, dur_ms)
        return
    transcript = [LLMMessage(role="user", content=text)] if (text or "").strip() else []

    sm = _softphone_sessionmaker or get_sessionmaker()
    try:
        async with sm() as session:
            row = await finalize_manual_call(
                session, provider_call_sid=call_id, transcript=transcript, llm=llm,
                tenant_timezone=tenant.settings.timezone,
                now=datetime.now(timezone.utc), duration_ms=dur_ms, recording_url=rec_url)
        log.info("stringee softphone recording finalized", extra={
            "tenant": tenant.slug, "call_id": call_id, "found": row is not None,
            "outcome": getattr(row, "outcome", None), "transcript_chars": len(text or "")})
    except Exception:  # noqa: BLE001 — never raise from a background task
        log.exception("stringee softphone recording finalize failed", extra={"call_id": call_id})


async def _download_stringee_recording(
    url: str, tenant: TenantContext, *, attempts: int = 6, _sleep=asyncio.sleep,
) -> bytes:
    """Fetch a Stringee recording, authenticating with a server token when the
    tenant's Stringee keys are available (signed/public URLs work without it).

    The recording lags the event — Stringee briefly returns 404 right after the
    call ends — so retry on 404 with backoff until it's ready."""
    from src.providers.telephony.stringee import mint_server_token

    tel = tenant.settings.pipeline.telephony
    creds = tel.active_creds()
    sid = tenant.secret(creds.account_sid_env) if creds.account_sid_env else None
    secret = tenant.secret(creds.auth_token_env) if creds.auth_token_env else None
    # Go straight to https (Stringee 301-redirects http→https), and rewrite the
    # host to the tenant's regional Stringee REST base when set — the URL arrives
    # pointing at api.stringee.com, but a regional project's keySid is only valid
    # on its regional host (e.g. asia-2.api.stringee.com), else r:5 "keySid
    # invalid" (see scripts/stringee_recording_probe.py). Auth is the same
    # X-STRINGEE-AUTH server token the callout uses.
    if tel.stringee_base_url:
        from urllib.parse import urlsplit
        base = tel.stringee_base_url.rstrip("/")
        if "://" not in base:
            base = "https://" + base
        parts = urlsplit(url if "://" in url else "https://" + url)
        url = f"{base}{parts.path}" + (f"?{parts.query}" if parts.query else "")
    else:
        url = url.replace("http://", "https://", 1)
    headers = {}
    if sid and secret:
        headers["X-STRINGEE-AUTH"] = mint_server_token(sid, secret)
    delay = 3.0
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True) as client:
        for attempt in range(attempts):
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404 and attempt < attempts - 1:
                await _sleep(delay)            # recording not ready yet — wait + retry
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
            return resp.content
    resp.raise_for_status()   # exhausted retries on 404
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

    stream_url = _ws_stream_url(request, f"exotel/stream/{tenant.slug}")
    body = voicebot_xml(stream_url)
    log.info(
        "exotel voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.post("/exotel/voice/{tenant_slug}", response_class=Response)
async def exotel_voice_for_tenant(
    request: Request,
    tenant_slug: str,
    To: str | None = Form(None),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Slug-scoped answer URL for **outbound** calls we place (dev console /
    orchestrator) — resolves the placing tenant by slug, so the outbound
    caller-ID need not be registered in ``tenant_phone_numbers``. Symmetric to
    the Twilio slug route; inbound keeps the bare ``/exotel/voice``."""
    from src.auth.middleware import tenant_from_slug

    tenant = await tenant_from_slug(tenant_slug)
    stream_url = _ws_stream_url(request, f"exotel/stream/{tenant.slug}")
    log.info(
        "exotel voice webhook (slug-scoped)",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=voicebot_xml(stream_url), media_type="application/xml")


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
    if inspect.isawaitable(bridge):
        bridge = await bridge
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


async def _stringee_answer(request: Request, tenant: "TenantContext | None"):
    """Build the call's bridge for an already-resolved tenant and return the
    opening SCCO. Shared by the slug route (outbound, tenant from the URL) and the
    number route (inbound fallback, tenant from the dialed/caller number)."""
    data = await _stringee_params(request)
    log.info("stringee answer", extra={"method": request.method, "data": data})
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    if not call_id:
        log.warning("stringee answer missing call_id; keys=%s", sorted(data.keys()))
    if tenant is None:
        log.warning("stringee answer: no tenant; keys=%s", sorted(data.keys()))
        return Response(status_code=404)
    if _stringee_bridge_factory is None:
        return Response(status_code=503)
    # Derive the base URL for audio/event webhook links. Prefer the platform
    # webhook_base_url (always the public-internet address) over request headers,
    # which can resolve to internal/container addresses under Northflank's proxy.
    from src.config_tenant import platform_webhook_base_url as _plat_wb
    _raw_base = _plat_wb() or None
    stringee_base = (_raw_base.rstrip("/") + "/stringee") if _raw_base else _stringee_base(request)
    log.info("stringee answer base", extra={"base": stringee_base, "tenant": tenant.slug})
    # Reuse a pre-warmed bridge (synthesized during ringing) if available;
    # otherwise build a fresh one and synthesize on-demand.
    bridge = registry.get(call_id) if call_id else None
    if bridge is not None:
        log.info("stringee answer: reusing prewarmed bridge", extra={"call_id": call_id})
    else:
        bridge = _stringee_bridge_factory(
            call_id=call_id, tenant=tenant,
            base_url=stringee_base, fetch=_download,
        )
        if inspect.isawaitable(bridge):
            bridge = await bridge
        registry.put(bridge)
    scco = await bridge.start_call()
    log.info("stringee answer SCCO", extra={"call_id": call_id, "scco": scco})
    if call_id:
        from src.api import dev_call_control
        dev_call_control.monitor.set_status(call_id, "answered")
    log.info("stringee answer registered", extra={"tenant": tenant.slug, "call_id": call_id})
    return JSONResponse(scco)


@router.api_route("/stringee/answer/{tenant_slug}", methods=["GET", "POST"])
async def stringee_answer_for_tenant(tenant_slug: str, request: Request):
    """Outbound answer URL: the placing tenant is the slug in the path (set when we
    built the callout's answer_url), so attribution + the live bridge's config
    follow who placed the call — not the shared caller/dialed number."""
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except Exception:  # noqa: BLE001 - unknown slug → 404 below
        tenant = None
    return await _stringee_answer(request, tenant)


@router.api_route("/stringee/answer", methods=["GET", "POST"])
async def stringee_answer(request: Request):
    """Inbound fallback: resolve the tenant from the dialed/caller number when the
    answer URL carries no slug. Stringee fetches the answer_url via GET (call info
    in the query string); we accept POST/JSON too.
    """
    tenant = await _resolve_stringee_tenant(await _stringee_params(request))
    return await _stringee_answer(request, tenant)


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
