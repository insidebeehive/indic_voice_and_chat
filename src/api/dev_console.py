# src/api/dev_console.py
"""Dev-only browser voice console (gated by VOX_DEV_CONSOLE=1).

Serves a self-contained page at ``GET /dev/voice`` and runs a
``BrowserVoiceBridge`` at ``WS /api/v1/dev/voice``. Reuses the tenant's
provider stack exactly like the telephony bridges; intended for local
dialogue-management iteration with no telephony cost.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.api.gemini_live_bridge import RECORD_TURN_SIGNAL, GeminiLiveBridge
from src.auth.context import TenantContext
from src.auth.registry import TenantProviders
from src.bootstrap import DEFAULT_DEMO_SCRIPT
from src.dialogue.prompts import VoiceBotScript, build_s2s_system_instruction
from src.dialogue.slots import SlotSchema
from src.interfaces.realtime import RealtimeConfig
from src.providers.realtime.gemini_live import GeminiLiveSession
from src.interfaces.llm import LLMConfig
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.pipeline.vad import EnergyVAD, SileroVAD
from src.providers import get_streaming_stt_provider, get_telephony_provider

log = logging.getLogger(__name__)

_STATIC = Path(__file__).resolve().parents[2] / "static"

# WS path lives under the /api/v1 router; the page route is top-level.
ws_router = APIRouter(prefix="/dev", tags=["dev-console"])   # mounted under /api/v1
dev_router = APIRouter(tags=["dev-console"])                  # mounted at app root

# Factory: (websocket, tenant) -> BrowserVoiceBridge. Set during lifespan.
BrowserBridgeFactory = Callable[[WebSocket, TenantContext], BrowserVoiceBridge]
_browser_bridge_factory: Optional[BrowserBridgeFactory] = None


def dev_console_enabled() -> bool:
    return os.environ.get("VOX_DEV_CONSOLE", "") == "1"


def set_browser_bridge_factory(factory: Optional[BrowserBridgeFactory]) -> None:
    global _browser_bridge_factory
    _browser_bridge_factory = factory


# Factory: (websocket, tenant) -> GeminiLiveBridge (the S2S path). Set during lifespan.
LiveBridgeFactory = Callable[[WebSocket, TenantContext], GeminiLiveBridge]
_live_bridge_factory: Optional[LiveBridgeFactory] = None


def set_live_bridge_factory(factory: Optional[LiveBridgeFactory]) -> None:
    global _live_bridge_factory
    _live_bridge_factory = factory


@dev_router.get("/dev/voice")
async def dev_voice_page() -> FileResponse:
    return FileResponse(_STATIC / "dev_console.html", media_type="text/html")


@dev_router.get("/dev/voices")
async def dev_voices(request: Request, tenant: str = "dev") -> dict:
    """Voice options for the console's Voice dropdown, per mode — config-driven so
    the list matches the active stack instead of a hardcoded one:

    - ``layered`` (cascade): the configured TTS provider's voice roster.
    - ``s2s`` (Gemini Live): the tenant's realtime ``allowed_voices``.
    """
    from src.auth.middleware import tenant_from_slug

    try:
        tctx = await tenant_from_slug(tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")
    p = tctx.settings.pipeline

    # S2S: the tenant's allowed realtime voices (default = pipeline.realtime.voice).
    rt = p.realtime
    s2s_voices = list(rt.allowed_voices) if (rt and rt.allowed_voices) else []
    s2s_default = (rt.voice if rt else "") or (s2s_voices[0] if s2s_voices else "")

    # Layered: the configured TTS provider's roster (default = pipeline.tts.voice_id).
    layered_default = p.tts.voice_id or ""
    layered_voices: list[str] = []
    providers = getattr(request.app.state, "providers", None)
    if providers is not None:
        try:
            tts = providers.get_tts(tctx)
            roster = tts.get_available_voices(p.tts.language or "hi-IN")
            layered_voices = [v.get("voice_id") for v in roster if v.get("voice_id")]
        except Exception:  # noqa: BLE001 - never block the page on a provider hiccup
            log.warning("dev voices: could not list TTS voices", exc_info=True)
    if layered_default and layered_default not in layered_voices:
        layered_voices = [layered_default, *layered_voices]
    if not layered_default and layered_voices:
        layered_default = layered_voices[0]

    return {
        "layered": {"voices": layered_voices, "default": layered_default},
        "s2s": {"voices": s2s_voices, "default": s2s_default},
    }


@dev_router.get("/dev/campaigns")
async def dev_campaigns(tenant: str = "dev") -> dict:
    """The tenant's campaigns, for the console's campaign selector. The selected
    id is passed to the voice WS as ``?campaign=<id>`` and drives the agent's
    script + slots for that call (per-tenant, no global fallback)."""
    from sqlalchemy import select

    from src.auth.middleware import tenant_from_slug
    from src.models.campaign import Campaign
    from src.models.database import get_sessionmaker

    try:
        tctx = await tenant_from_slug(tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")
    async with get_sessionmaker()() as s:
        rows = (await s.execute(
            select(Campaign).where(Campaign.tenant_id == tctx.id)
            .order_by(Campaign.status.desc(), Campaign.created_at.desc())
        )).scalars().all()
    return {"campaigns": [{"id": c.id, "name": c.name, "status": c.status} for c in rows]}


# --- Telephony control panel: place an outbound call + poll its status --------
#
# WebConsole runs in-browser (the WS routes above). The Telephony dropdown picks
# the provider; Twilio/Exotel place a real outbound call that runs the agent over
# the phone, and the console polls the in-memory call monitor the bridge writes
# to (keyed by the provider Call SID) for lifecycle (calling -> answered ->
# ended) + outcome. Mode/Voice are threaded via a one-shot override the bridge
# factory consumes. The selected provider's adapter is built on demand (creds
# resolve from the provider's env vars); the caller-ID comes from
# pipeline.telephony.outbound_from[provider]. Needs a publicly reachable host.

# Providers the dev console can place an outbound call with, and the answer-webhook
# path each one uses. Twilio/Exotel run the media-stream bridge (S2S or cascade,
# per the Mode override); Stringee is a turn-based IVR with its own /stringee/answer.
_ANSWER_PATH = {"twilio": "twilio/voice", "exotel": "exotel/voice", "stringee": "stringee/answer"}
_PLACE_CALL_PROVIDERS = tuple(_ANSWER_PATH)
# Stringee is IVR-only — Mode/Voice (S2S vs cascade) don't apply to it.
_STREAM_PROVIDERS = ("twilio", "exotel")


class PlaceCallRequest(BaseModel):
    provider: str               # "twilio" | "exotel"
    to_number: str
    mode: str = "s2s"           # "s2s" | "layered" — drives the placed call
    voice: str = ""             # S2S voice; "" -> tenant default
    lead_name: str = ""
    tenant: str = "dev"


@dev_router.post("/dev/place-call")
async def dev_place_call(req: PlaceCallRequest) -> dict:
    from src.auth.middleware import tenant_from_slug
    from src.interfaces.telephony import CallConfig

    from src.api import dev_call_control

    try:
        tenant = await tenant_from_slug(req.tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")

    tel = tenant.settings.pipeline.telephony
    provider = req.provider.strip().lower()
    if provider not in _PLACE_CALL_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider '{provider}' can't be placed from here; use {list(_PLACE_CALL_PROVIDERS)}")
    webhook_base = tel.effective_webhook_base_url()   # tenant value, else platform WEBHOOK_BASE_URL
    if not webhook_base:
        raise HTTPException(
            status_code=400,
            detail="no telephony webhook base URL — set tenant telephony.webhook_base_url or platform WEBHOOK_BASE_URL")

    # The dropdown drives the provider — build *its* adapter (creds resolve from the
    # provider's env vars) and dial from *its* configured caller-ID, independent of
    # the tenant's default/inbound provider.
    from_number = (tel.outbound_from or {}).get(provider)
    if not from_number and (tel.provider or "").lower() == provider:
        from_number = tel.from_number          # default block's number, if it matches
    if not from_number:
        raise HTTPException(
            status_code=400,
            detail=(f"no caller-ID configured for '{provider}'. Set "
                    f"pipeline.telephony.outbound_from.{provider} in config/tenants/{req.tenant}.yaml."))

    # Build the adapter with the SELECTED provider's per-tenant creds (the dropdown
    # provider may differ from the tenant's default) — no platform-env fallback.
    # Stringee's server adapter reads api_key_sid/api_key_secret (stored as the
    # account_sid/auth_token), the others account_sid/auth_token.
    def _cred(name):
        try:
            return tenant.secret(name) if name else None
        except Exception:  # noqa: BLE001 - missing env → let the adapter decide
            return None

    pcreds = tel.creds_for(provider)
    acct, auth = _cred(pcreds.account_sid_env), _cred(pcreds.auth_token_env)
    # Stringee: the callout needs a non-null userId, or it goes out as a
    # phone->phone external call and the Answer URL/SCCO never runs (silent bot).
    uid = _cred(pcreds.user_id_env)
    try:
        adapter = get_telephony_provider({
            "provider": provider,
            "account_sid": acct, "auth_token": auth,
            "api_key_sid": acct, "api_key_secret": auth,
            "user_id": uid,
        })
    except Exception as e:  # noqa: BLE001 - e.g. missing per-tenant credentials
        raise HTTPException(status_code=400, detail=f"telephony adapter for '{provider}' unavailable: {e}")

    # Thread the console's Mode/Voice/lead to the media-stream bridge factory.
    # Stringee ignores it (turn-based IVR), so don't leave a stale override.
    if provider in _STREAM_PROVIDERS:
        dev_call_control.set_override(
            tenant.slug, mode=req.mode, voice=req.voice.strip(), lead_name=req.lead_name.strip())
    cfg = CallConfig(
        to_number=req.to_number.strip(),
        from_number=from_number,
        webhook_url=f"{webhook_base.rstrip('/')}/{_ANSWER_PATH[provider]}",
    )
    try:
        session = await adapter.initiate_call(cfg)
    except Exception as e:  # noqa: BLE001 - don't leave a stale override on failure
        if provider in _STREAM_PROVIDERS:
            dev_call_control.pop_override(tenant.slug)
        log.exception("dev place-call failed", extra={"tenant": tenant.slug, "provider": provider})
        raise HTTPException(status_code=502, detail=f"call failed: {e}")

    dev_call_control.monitor.set_status(session.session_id, "calling")
    log.info("dev console placed call", extra={
        "tenant": tenant.slug, "provider": provider, "call_sid": session.session_id})
    return {"call_sid": session.session_id, "status": "calling"}


@dev_router.get("/dev/call-status/{call_sid}")
async def dev_call_status(call_sid: str) -> dict:
    from src.api import dev_call_control

    item = dev_call_control.monitor.get(call_sid)
    if item is None:
        return {"status": "unknown", "outcome": None}
    return item


def _parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


async def _run_billed_session(tenant, bridge, *, mode: str) -> None:
    """Run a browser-console bridge as a recorded + billed conversation.

    Inserts an in_progress conversation row (channel='webconsole') keyed by a
    fresh call_id, runs the bridge, then finalizes the row (status/outcome +
    derived duration + platform cost). ``mode`` is the actual path used
    ('s2s' for the live console, 'layered' for the cascade console) so the cost
    is billed correctly even when it differs from the tenant default. Telephony
    is excluded from the cost (the browser path uses no telephony). Failures here
    never break the call.
    """
    from src.api.call_store import insert_call, record_outcome
    from src.models.database import get_sessionmaker

    sm = get_sessionmaker()
    call_id = f"call_{uuid.uuid4().hex[:16]}"
    try:
        async with sm() as s:
            await insert_call(s, call_id=call_id, tenant=tenant,
                              provider_call_sid=call_id, channel="webconsole", mode=mode)
    except Exception:  # noqa: BLE001
        log.exception("webconsole: failed to start call record")
    try:
        await bridge.run()
    finally:
        payload = getattr(bridge, "_outcome_payload", None) or {}
        try:
            async with sm() as s:
                await record_outcome(
                    s, call_id, status="ended",
                    outcome=payload.get("outcome"), summary=payload.get("summary"),
                    notes=payload.get("notes"),
                    callback_at=_parse_iso(payload.get("callback_datetime")))
        except Exception:  # noqa: BLE001
            log.exception("webconsole: failed to finalize call record")


@ws_router.websocket("/voice")
async def dev_voice_ws(websocket: WebSocket) -> None:
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    if _browser_bridge_factory is None:
        await websocket.close(code=1011, reason="browser bridge factory unset")
        return
    try:
        tenant = await tenant_from_slug(
            websocket.query_params.get("tenant", "dev")
        )
    except Exception as e:  # noqa: BLE001
        log.warning("dev console tenant resolution failed: %s", e)
        await websocket.close(code=1008, reason="unknown tenant")
        return

    try:
        bridge = await _browser_bridge_factory(websocket, tenant)
    except Exception as e:  # noqa: BLE001 - e.g. tenant has no campaign configured
        log.warning("dev console bridge build failed: %s", e)
        await websocket.close(code=1011, reason="no campaign configured for tenant")
        return
    try:
        await _run_billed_session(tenant, bridge, mode="layered")
    except WebSocketDisconnect:
        log.info("dev console client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("dev console bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


@ws_router.websocket("/voice-live")
async def dev_voice_live_ws(websocket: WebSocket) -> None:
    """Speech-to-speech (Gemini Live) path. Same client; different bridge."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    if _live_bridge_factory is None:
        await websocket.close(code=1011, reason="live bridge factory unset")
        return
    try:
        tenant = await tenant_from_slug(websocket.query_params.get("tenant", "dev"))
    except Exception as e:  # noqa: BLE001
        log.warning("dev console (s2s) tenant resolution failed: %s", e)
        await websocket.close(code=1008, reason="unknown tenant")
        return
    try:
        bridge = await _live_bridge_factory(websocket, tenant)
    except Exception as e:  # noqa: BLE001 - e.g. tenant has no realtime config
        log.warning("dev console (s2s) bridge build failed: %s", e)
        await websocket.close(code=1011, reason="s2s not configured for tenant")
        return
    try:
        await _run_billed_session(tenant, bridge, mode="s2s")
    except WebSocketDisconnect:
        log.info("dev console (s2s) client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("dev console (s2s) bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _build_browser_vad():
    """VAD for the dev console: prefer Silero (robust speech/noise discrimination
    so turns end cleanly on speakers); fall back to EnergyVAD if onnxruntime or
    the model isn't available. Silero needs 32 ms / 512-sample frames at 16 kHz.
    """
    try:
        vad = SileroVAD(sample_rate=16000, frame_ms=32, threshold=0.5)
        vad._ensure_model()  # load now so a failure falls back here, not mid-call
        log.info("dev console using SileroVAD")
        return vad
    except Exception as e:  # noqa: BLE001
        log.warning("SileroVAD unavailable (%s); using EnergyVAD", e)
        return EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0)


def _build_stream_provider(tenant: TenantContext):
    """Build a streaming-STT provider from pipeline.stt_streaming, or None.

    Returns None when no streaming config is present (batch behaviour) or when
    the provider can't be constructed (e.g. missing key) — the bridge then
    falls back to batch Groq, so this never blocks a call.
    """
    cfg = getattr(tenant.settings.pipeline, "stt_streaming", None)
    if cfg is None or not getattr(cfg, "provider", None):
        return None
    try:
        merged = {
            "provider": cfg.provider,
            "model": cfg.model,
            "language": cfg.language,
            "endpointing": cfg.endpointing,
            "utterance_end_ms": cfg.utterance_end_ms,
            "api_key": tenant.secret(cfg.api_key_env) if cfg.api_key_env else None,
        }
        return get_streaming_stt_provider(merged)
    except Exception as e:  # noqa: BLE001 - never block a call on streaming setup
        log.warning("streaming STT provider unavailable (%s); using batch", e)
        return None


def make_browser_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    campaign_resolver=None,
) -> BrowserBridgeFactory:
    """Build a BrowserVoiceBridge per connection, wired to the tenant stack.

    Mirrors ``src.bootstrap.make_bridge_factory`` but returns a browser bridge.
    When a ``campaign_resolver`` is supplied, the agent's script + slots are
    resolved per call from the tenant's DB campaign (``?campaign=<id>``, else the
    tenant's active campaign, else the YAML fallback) instead of the global
    closure script/slots.
    """

    async def factory(websocket: WebSocket, tenant: TenantContext) -> BrowserVoiceBridge:
        import uuid

        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            qp0 = getattr(websocket, "query_params", {}) or {}
            lc = await campaign_resolver.resolve(tenant.id, qp0.get("campaign") or None)
            cur_script, cur_slots = lc.script, lc.slots

        stt = providers.get_stt(tenant)
        llm = providers.get_llm(tenant)
        tts = providers.get_tts(tenant)
        tts_language = tenant.settings.pipeline.tts.language or "hi-IN"
        # Voice: ?voice= overrides the configured default (validated against the
        # TTS provider's roster), so the console's Voice dropdown applies in
        # layered mode just like it does for S2S.
        tts_voice = tenant.settings.pipeline.tts.voice_id
        sel_voice = ((getattr(websocket, "query_params", {}) or {}).get("voice") or "").strip()
        if sel_voice:
            try:
                roster = {v.get("voice_id") for v in tts.get_available_voices(tts_language)}
            except Exception:  # noqa: BLE001
                roster = set()
            if sel_voice in roster:
                tts_voice = sel_voice
        pipeline_cfg = PipelineConfig(
            stt=STTConfig(language=tenant.settings.pipeline.stt.language or "hi-IN"),
            llm=LLMConfig(
                temperature=tenant.settings.pipeline.llm.temperature or 0.5,
                max_tokens=tenant.settings.pipeline.llm.max_tokens or 256,
                response_format=tenant.settings.pipeline.llm.response_format or "json",
            ),
            tts=TTSConfig(
                language=tts_language,
                voice_id=tts_voice,
                sample_rate=16000,
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)
        session_id = f"web_{uuid.uuid4().hex[:12]}"
        # The dev console has no CRM lead, so let the page supply a test lead
        # name via the WS query string (?lead_name=...). This feeds the spoken
        # opening, the rendered opening in the prompt, and "Known lead data".
        query_params = getattr(websocket, "query_params", {}) or {}
        lead_name = (query_params.get("lead_name") or "").strip()
        lead_data = {"lead_name": lead_name, "name": lead_name} if lead_name else {}
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id, lead_data=lead_data),
            state_machine=AgentStateMachine(),
            slot_schema=cur_slots,
            script=cur_script,
            engine=engine,
            store=None,
        )
        log.info("dev console built call", extra={"tenant": tenant.slug, "session_id": session_id})
        return BrowserVoiceBridge(
            websocket=websocket,
            agent=agent,
            vad=_build_browser_vad(),
            config=BrowserBridgeConfig(),
            stream_provider=_build_stream_provider(tenant),
            llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
        )

    return factory


def make_live_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    campaign_resolver=None,
) -> LiveBridgeFactory:
    """Build a GeminiLiveBridge (S2S) per connection from pipeline.realtime.

    With a ``campaign_resolver``, the agent's script + slots come per call from
    the tenant's DB campaign (``?campaign=<id>`` → active → YAML fallback)."""

    async def factory(websocket: WebSocket, tenant: TenantContext) -> GeminiLiveBridge:
        import uuid

        rt = getattr(tenant.settings.pipeline, "realtime", None)
        if rt is None or not getattr(rt, "provider", None):
            raise RuntimeError("tenant has no pipeline.realtime config for S2S")

        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            qp0 = getattr(websocket, "query_params", {}) or {}
            lc = await campaign_resolver.resolve(tenant.id, qp0.get("campaign") or None)
            cur_script, cur_slots = lc.script, lc.slots

        llm = providers.get_llm(tenant)
        # The agent is the same; only the bridge differs. The engine is required by
        # the constructor (the Live path doesn't synthesize via it).
        engine = PipelineEngine(
            providers.get_stt(tenant), llm, providers.get_tts(tenant),
            PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(sample_rate=16000)),
        )
        qp = getattr(websocket, "query_params", {}) or {}
        lead_name = (qp.get("lead_name") or "").strip()
        lead_data = {"lead_name": lead_name, "name": lead_name} if lead_name else {}
        session_id = f"live_{uuid.uuid4().hex[:12]}"
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id, lead_data=lead_data),
            state_machine=AgentStateMachine(), slot_schema=cur_slots, script=cur_script,
            engine=engine, store=None,
        )

        # Voice: ?voice= overrides the config default (validated against allowed_voices).
        voice = (qp.get("voice") or "").strip() or rt.voice
        if rt.allowed_voices and voice not in rt.allowed_voices:
            voice = rt.voice
        key = tenant.secret(rt.api_key_env) if rt.api_key_env else None
        config = RealtimeConfig(
            model=rt.model, voice=voice, language_code=rt.language_code,
            system_instruction=build_s2s_system_instruction(cur_script, cur_slots, lead_data),
            tools=[RECORD_TURN_SIGNAL],
        )

        async def connect(cfg: RealtimeConfig):
            return await GeminiLiveSession.connect(cfg, api_key=key)

        log.info("dev console built S2S call", extra={
            "tenant": tenant.slug, "session_id": session_id, "voice": voice, "model": rt.model})
        return GeminiLiveBridge(
            websocket=websocket, agent=agent, config=config, connect_session=connect,
            llm=llm, tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"))

    return factory
