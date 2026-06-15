"""Production wiring: per-call bridge factory + provider construction.

Lives in its own module (rather than ``main.py``) so individual pieces are
unit-testable without spinning up the FastAPI lifespan.

The bridge factory is the moment of truth — it's what turns a tenant's
declared provider preferences into a live conversation. Flow:

    inbound WS connect
        -> twilio_stream() resolves tenant from ?tenant= query param
        -> calls registered bridge_factory(websocket, tenant)
        -> builds STT/LLM/TTS/scheduler/etc. for that tenant via the
           cached TenantProviders registry
        -> assembles a VoiceBotAgent with a minimal demo script
        -> wraps in a TwilioMediaBridge that:
             * calls agent.start() then agent.play_opening(sink)
             * pumps Twilio media frames into agent.handle_turn()
             * sends agent TTS audio back as μ-law frames
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import WebSocket

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api.telephony_exotel import ExotelBridgeConfig, ExotelMediaBridge
from src.api.telephony_twilio import TwilioBridgeConfig, TwilioMediaBridge
from src.auth.context import TenantContext
from src.auth.registry import TenantProviders
from src.dialogue.context import SessionStore
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import LLMConfig
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.pipeline.vad import EnergyVAD
from src.providers import (
    get_llm_provider,
    get_stt_provider,
    get_telephony_provider,
    get_tts_provider,
    get_vector_store,
)

log = logging.getLogger(__name__)


# --- Demo script -------------------------------------------------------


DEFAULT_DEMO_SCRIPT = VoiceBotScript(
    agent_name="Priya",
    agent_role="Customer Engagement Specialist",
    company_name="Vox Demo",
    language_default="hi-IN",
    opening=(
        "Namaste! Main Priya bol rahi hoon Vox Demo se. "
        "Aapse ek choti si baat karni thi — kya aapke paas do minute hain?"
    ),
    talking_points=[
        "Vox Demo ek end-to-end AI voice agent platform hai.",
    ],
    qualifying_questions=["Aap abhi kya use kar rahe hain customer calls ke liye?"],
    objection_responses={
        "is_ai": "Haan, main ek AI assistant hoon — Vox Demo ki taraf se.",
        "busy": "Bilkul, samajh sakti hoon. Kya main baad mein call karun?",
    },
    closing={
        "positive": "Bahut accha! Dhanyavaad aapke time ke liye.",
        "negative": "Koi baat nahi. Aapka din shubh ho!",
    },
)


# --- Per-tenant runtime builders ---------------------------------------


def build_provider_registry(
    global_defaults: dict, base_vector_path: Path = Path("data/faiss"),
) -> TenantProviders:
    """One ``TenantProviders`` per process; caches per-tenant clients."""
    return TenantProviders(
        global_defaults=global_defaults,
        stt_factory=get_stt_provider,
        llm_factory=get_llm_provider,
        tts_factory=get_tts_provider,
        telephony_factory=get_telephony_provider,
        vector_store_factory=get_vector_store,
        base_vector_path=base_vector_path,
    )


# --- The bridge factory ------------------------------------------------


@dataclass
class _CallSpec:
    """Resolved per-call wiring."""

    tenant: TenantContext
    session_id: str
    lead_data: dict


def _override_lead_data(override: dict | None) -> dict:
    """Build agent lead_data from a dev-console override's lead_name (if any)."""
    name = (override or {}).get("lead_name", "").strip()
    return {"lead_name": name, "name": name} if name else {}


def _build_s2s_telephony_bridge(
    providers: TenantProviders, tenant: TenantContext, script: VoiceBotScript,
    slots: SlotSchema, websocket: WebSocket, session_store: SessionStore | None,
    *, encoding: str, sid_field: str, supports_clear: bool,
    call_sid_field: str = "callSid", voice_override: str | None = None,
    lead_data: dict | None = None,
):
    """Build a TelephonyLiveBridge (Gemini Live over the media stream) for a call
    whose tenant has pipeline.mode == 's2s'. Mirrors the cascade agent assembly
    but returns the S2S bridge; reuses the dev-console S2S wiring shape."""
    import uuid

    from src.api.live_bridge_base import RECORD_TURN_SIGNAL
    from src.api.telephony_live_bridge import TelephonyLiveBridge
    from src.dialogue.prompts import build_s2s_system_instruction
    from src.interfaces.realtime import RealtimeConfig
    from src.providers.realtime.gemini_live import GeminiLiveSession

    rt = tenant.settings.pipeline.realtime
    llm = providers.get_llm(tenant)
    # The agent is the same; the engine is only needed to satisfy the constructor
    # (the Live path doesn't synthesize via it).
    engine = PipelineEngine(
        providers.get_stt(tenant), llm, providers.get_tts(tenant),
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(sample_rate=16000)))
    store: SessionStore | None = None
    if session_store is not None:
        store = SessionStore(redis=session_store.redis, ttl_seconds=session_store.ttl,
                             tenant_id=tenant.id)
    session_id = f"call_{uuid.uuid4().hex[:12]}"
    lead_data = lead_data or {}
    agent = VoiceBotAgent(
        session=AgentSession(session_id=session_id, lead_data=lead_data),
        state_machine=AgentStateMachine(),
        slot_schema=slots, script=script, engine=engine, store=store)
    # Voice: a dev-console override wins (validated against allowed_voices); else config.
    voice = (voice_override or "").strip() or rt.voice
    allowed = getattr(rt, "allowed_voices", None)
    if allowed and voice not in allowed:
        voice = rt.voice
    key = tenant.secret(rt.api_key_env) if rt.api_key_env else None
    config = RealtimeConfig(
        model=rt.model, voice=voice, language_code=rt.language_code,
        system_instruction=build_s2s_system_instruction(script, slots, lead_data),
        tools=[RECORD_TURN_SIGNAL])

    async def connect(cfg: RealtimeConfig):
        return await GeminiLiveSession.connect(cfg, api_key=key)

    log.info("s2s telephony bridge built call", extra={
        "tenant": tenant.slug, "session_id": session_id, "voice": voice,
        "model": rt.model, "encoding": encoding})
    return TelephonyLiveBridge(
        websocket=websocket, agent=agent, config=config, connect_session=connect, llm=llm,
        tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
        encoding=encoding, sid_field=sid_field, supports_clear=supports_clear,
        call_sid_field=call_sid_field)


def make_bridge_factory(
    providers: TenantProviders,
    session_store: SessionStore | None = None,
    bridge_config: TwilioBridgeConfig | None = None,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
) -> Callable[[WebSocket, TenantContext], object]:
    """Return a callable suitable for ``set_bridge_factory(...)``.

    The returned factory closes over the shared registries so every
    inbound call lands on the right tenant-scoped provider clients
    without rebuilding anything.
    """
    cfg = bridge_config or TwilioBridgeConfig()

    def factory(websocket: WebSocket, tenant: TenantContext):
        from src.api import dev_call_control

        # A dev-console "place call" can override mode/voice for this one call.
        override = dev_call_control.pop_override(tenant.slug)
        mode = (override or {}).get("mode") or getattr(
            tenant.settings.pipeline, "mode", "layered")
        # Speech-to-speech path: when the tenant is in s2s mode, drive Gemini Live
        # over the Twilio media stream instead of the STT->LLM->TTS cascade.
        if mode == "s2s":
            return _build_s2s_telephony_bridge(
                providers, tenant, script, slots, websocket, session_store,
                encoding="mulaw", sid_field="streamSid", supports_clear=True,
                call_sid_field="callSid", voice_override=(override or {}).get("voice"),
                lead_data=_override_lead_data(override))
        # Build a fresh agent per call; provider clients are cached on the
        # registry so we don't pay reconstruction cost.
        stt = providers.get_stt(tenant)
        llm = providers.get_llm(tenant)
        tts = providers.get_tts(tenant)

        # Tenant-namespaced Redis session store (one per tenant; the same
        # instance is fine across calls since the keys carry session_id).
        store: SessionStore | None = None
        if session_store is not None:
            store = SessionStore(
                redis=session_store.redis,
                ttl_seconds=session_store.ttl,
                tenant_id=tenant.id,
            )

        # Use Sarvam/etc. defaults for the per-call configs — providers
        # have already been built with tenant overrides applied.
        pipeline_cfg = PipelineConfig(
            stt=STTConfig(language=tenant.settings.pipeline.stt.language or "hi-IN"),
            llm=LLMConfig(
                temperature=tenant.settings.pipeline.llm.temperature or 0.5,
                max_tokens=tenant.settings.pipeline.llm.max_tokens or 256,
                response_format=tenant.settings.pipeline.llm.response_format or "json",
            ),
            tts=TTSConfig(
                language=tenant.settings.pipeline.tts.language or "hi-IN",
                voice_id=tenant.settings.pipeline.tts.voice_id,
                sample_rate=16000,
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)

        import uuid

        session_id = f"call_{uuid.uuid4().hex[:12]}"
        session = AgentSession(session_id=session_id)
        sm = AgentStateMachine()
        agent = VoiceBotAgent(
            session=session,
            state_machine=sm,
            slot_schema=slots,
            script=script,
            engine=engine,
            store=store,
        )

        log.info(
            "bridge factory built call",
            extra={"tenant": tenant.slug, "session_id": session_id},
        )

        return _AgentBridge(
            websocket=websocket,
            agent=agent,
            vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
            config=cfg,
            llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
        )

    return factory


class _AgentBridge(TwilioMediaBridge):
    """Subclass of TwilioMediaBridge that plays an opening line on connect.

    Crucial ordering: Twilio sends ``connected`` then ``start`` events on
    the WS before any media. ``_send_pcm`` needs ``self._stream_sid``,
    which is only populated when we process the ``start`` event. So
    ``play_opening`` MUST run after the start event, not before — otherwise
    the opening audio is silently dropped (``_send_pcm`` returns early
    when ``_stream_sid is None``).
    """

    async def run(self) -> None:
        import json

        await self._agent.start()
        opening_played = False
        try:
            while not self._stopped.is_set():
                raw = await self._ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "connected":
                    continue
                if event == "start":
                    self._stream_sid = (
                        msg.get("start", {}).get("streamSid") or msg.get("streamSid")
                    )
                    log.info("twilio stream started", extra={"streamSid": self._stream_sid})
                    # NOW we can play the opening — Twilio is ready to receive media.
                    if not opening_played:
                        opening_played = True
                        await self._agent.play_opening(self._send_pcm)  # type: ignore[arg-type]
                        log.info("agent opening played", extra={"streamSid": self._stream_sid})
                elif event == "media":
                    await self._on_media_frame(msg["media"])
                elif event == "stop":
                    log.info("twilio stream stopped")
                    break
        finally:
            try:
                await self._record_outcome()
            except Exception:  # noqa: BLE001 - never let analysis break teardown
                log.exception("record outcome failed")
            await self._agent.handle_hangup()


# --- Exotel bridge factory (shares the agent stack with Twilio) ----------


def make_exotel_bridge_factory(
    providers: TenantProviders,
    session_store: SessionStore | None = None,
    bridge_config: ExotelBridgeConfig | None = None,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
) -> Callable[[WebSocket, TenantContext], ExotelMediaBridge]:
    """Build an Exotel WS bridge per call, wired to the tenant's provider stack.

    Identical agent assembly to ``make_bridge_factory`` (Twilio) — only the
    last step differs: returns an ``_ExotelAgentBridge`` instead of an
    ``_AgentBridge``. The agent itself is encoding-agnostic; only the bridge
    knows whether to ship μ-law or PCM16 over the wire.
    """
    cfg = bridge_config or ExotelBridgeConfig()

    def factory(websocket: WebSocket, tenant: TenantContext):
        from src.api import dev_call_control

        override = dev_call_control.pop_override(tenant.slug)
        mode = (override or {}).get("mode") or getattr(
            tenant.settings.pipeline, "mode", "layered")
        # S2S path: drive Gemini Live over the Exotel media stream (raw PCM16@8k,
        # snake_case stream_sid, no `clear` frame) when the tenant is in s2s mode.
        if mode == "s2s":
            return _build_s2s_telephony_bridge(
                providers, tenant, script, slots, websocket, session_store,
                encoding="pcm", sid_field="stream_sid", supports_clear=False,
                call_sid_field="call_sid", voice_override=(override or {}).get("voice"),
                lead_data=_override_lead_data(override))
        stt = providers.get_stt(tenant)
        llm = providers.get_llm(tenant)
        tts = providers.get_tts(tenant)

        store: SessionStore | None = None
        if session_store is not None:
            store = SessionStore(
                redis=session_store.redis,
                ttl_seconds=session_store.ttl,
                tenant_id=tenant.id,
            )

        pipeline_cfg = PipelineConfig(
            stt=STTConfig(language=tenant.settings.pipeline.stt.language or "hi-IN"),
            llm=LLMConfig(
                temperature=tenant.settings.pipeline.llm.temperature or 0.5,
                max_tokens=tenant.settings.pipeline.llm.max_tokens or 256,
                response_format=tenant.settings.pipeline.llm.response_format or "json",
            ),
            tts=TTSConfig(
                language=tenant.settings.pipeline.tts.language or "hi-IN",
                voice_id=tenant.settings.pipeline.tts.voice_id,
                sample_rate=16000,
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)

        import uuid

        session_id = f"call_{uuid.uuid4().hex[:12]}"
        session = AgentSession(session_id=session_id)
        sm = AgentStateMachine()
        agent = VoiceBotAgent(
            session=session,
            state_machine=sm,
            slot_schema=slots,
            script=script,
            engine=engine,
            store=store,
        )

        log.info(
            "exotel bridge factory built call",
            extra={"tenant": tenant.slug, "session_id": session_id},
        )

        return _ExotelAgentBridge(
            websocket=websocket,
            agent=agent,
            vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
            config=cfg,
            llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
        )

    return factory


# --- Stringee IVR bridge factory (HTTP-driven, per-call) --------------------


def make_stringee_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
):
    """Build a StringeeIvrBridge per call, wired to the tenant's providers.

    Same agent assembly as make_exotel_bridge_factory; HTTP-driven instead of
    WS-driven, so the call_id/base_url/fetch are passed per call by the route.
    """
    from src.api.telephony_stringee_bridge import StringeeIvrBridge

    def factory(*, call_id, tenant, base_url, fetch):
        stt = providers.get_stt(tenant)
        llm = providers.get_llm(tenant)
        tts = providers.get_tts(tenant)

        pipeline_cfg = PipelineConfig(
            stt=STTConfig(language=tenant.settings.pipeline.stt.language or "hi-IN"),
            llm=LLMConfig(
                temperature=tenant.settings.pipeline.llm.temperature or 0.5,
                max_tokens=tenant.settings.pipeline.llm.max_tokens or 256,
                response_format=tenant.settings.pipeline.llm.response_format or "json",
            ),
            tts=TTSConfig(
                language=tenant.settings.pipeline.tts.language or "hi-IN",
                voice_id=tenant.settings.pipeline.tts.voice_id,
                sample_rate=16000,
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)

        import uuid

        session_id = f"call_{uuid.uuid4().hex[:12]}"
        session = AgentSession(session_id=session_id)
        sm = AgentStateMachine()
        agent = VoiceBotAgent(
            session=session,
            state_machine=sm,
            slot_schema=slots,
            script=script,
            engine=engine,
            store=None,
        )

        log.info(
            "stringee bridge factory built call",
            extra={"tenant": tenant.slug, "session_id": session_id},
        )

        return StringeeIvrBridge(
            call_id=str(call_id),
            agent=agent,
            llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
            tts_sample_rate=16000,
            base_url=base_url,
            tenant_slug=tenant.slug,
            fetch=fetch,
        )

    return factory


class _ExotelAgentBridge(ExotelMediaBridge):
    """ExotelMediaBridge that plays the opening line once the stream starts.

    Same ``start``-event ordering rule as the Twilio variant: ``_stream_sid``
    isn't populated until Exotel sends ``start``, and ``_send_pcm`` early-returns
    on a null stream sid — so the opening must wait for that event.
    """

    async def run(self) -> None:
        import json

        await self._agent.start()
        opening_played = False
        try:
            while not self._stopped.is_set():
                raw = await self._ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "connected":
                    continue
                if event == "start":
                    self._stream_sid = (
                        msg.get("stream_sid")
                        or msg.get("start", {}).get("stream_sid")
                        or msg.get("start", {}).get("streamSid")
                    )
                    log.info("exotel stream started", extra={"stream_sid": self._stream_sid})
                    if not opening_played:
                        opening_played = True
                        await self._agent.play_opening(self._send_pcm)  # type: ignore[arg-type]
                        log.info("agent opening played (exotel)", extra={"stream_sid": self._stream_sid})
                elif event == "media":
                    await self._on_media_frame(msg["media"])
                elif event == "stop":
                    log.info("exotel stream stopped")
                    break
        finally:
            try:
                await self._record_outcome()
            except Exception:  # noqa: BLE001 - never let analysis break teardown
                log.exception("record outcome failed")
            await self._agent.handle_hangup()


def make_sip_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    sip_call_factory=None,
    session_store: SessionStore | None = None,
):
    """Build a SipMediaBridge per outbound SIP-trunk call (e.g. DiDLogic).

    S2S only for now: the tenant must have ``pipeline.realtime`` configured.
    The SIP user/password come from the tenant's encrypted telephony secrets
    (account_sid_env / auth_token_env) and the host from ``telephony.sip_server``.
    ``sip_call_factory`` places the INVITE — defaults to the pyVoIP implementation,
    injectable for tests.
    """
    import uuid

    from src.api.live_bridge_base import RECORD_TURN_SIGNAL
    from src.api.sip_media_bridge import SipMediaBridge
    from src.dialogue.prompts import build_s2s_system_instruction
    from src.interfaces.realtime import RealtimeConfig
    from src.providers.realtime.gemini_live import GeminiLiveSession
    from src.providers.telephony.sip.transport import SipCallParams

    async def _default_factory(params: SipCallParams):
        from src.providers.telephony.sip.pyvoip_call import place_pyvoip_call
        return await place_pyvoip_call(params)

    place = sip_call_factory or _default_factory

    async def factory(tenant: TenantContext, to_number: str, *, lead_data: dict | None = None):
        rt = tenant.settings.pipeline.realtime
        if rt is None or not getattr(rt, "provider", None):
            raise RuntimeError(
                "SIP outbound currently requires pipeline.mode=s2s (realtime config)")
        tel = tenant.settings.pipeline.telephony
        sip_user = tenant.secret(tel.account_sid_env) if tel.account_sid_env else None
        sip_password = tenant.secret(tel.auth_token_env) if tel.auth_token_env else None
        if not (sip_user and sip_password and tel.sip_server):
            raise RuntimeError(
                "DiDLogic SIP requires telephony account_sid+auth_token (SIP user/pass) "
                "and telephony.sip_server")

        llm = providers.get_llm(tenant)
        engine = PipelineEngine(
            providers.get_stt(tenant), llm, providers.get_tts(tenant),
            PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(sample_rate=16000)))
        store: SessionStore | None = None
        if session_store is not None:
            store = SessionStore(redis=session_store.redis, ttl_seconds=session_store.ttl,
                                 tenant_id=tenant.id)
        lead = lead_data or {}
        session_id = f"call_{uuid.uuid4().hex[:12]}"
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id, lead_data=lead),
            state_machine=AgentStateMachine(), slot_schema=slots, script=script,
            engine=engine, store=store)
        key = tenant.secret(rt.api_key_env) if rt.api_key_env else None
        config = RealtimeConfig(
            model=rt.model, voice=rt.voice, language_code=rt.language_code,
            system_instruction=build_s2s_system_instruction(script, slots, lead),
            tools=[RECORD_TURN_SIGNAL])

        async def connect(cfg: RealtimeConfig):
            return await GeminiLiveSession.connect(cfg, api_key=key)

        params = SipCallParams(
            to_number=to_number, from_number=tel.from_number or "",
            sip_user=sip_user, sip_password=sip_password, sip_server=tel.sip_server)
        sip_call = await place(params)   # sends the INVITE
        log.info("sip bridge built call", extra={
            "tenant": tenant.slug, "to": to_number, "session_id": session_id})
        return SipMediaBridge(
            sip_call=sip_call, agent=agent, config=config, connect_session=connect,
            llm=llm, tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"))

    return factory
