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


def build_platform_retriever(base_vector_path: Path = Path("data/faiss")) -> "HybridRetriever":
    """One shared FAISS+BM25 retriever for the global KB (all tenants)."""
    from src.providers.vector_store.faiss_store import FAISSAdapter
    from src.rag.embeddings import GeminiEmbedder
    from src.rag.retriever import HybridRetriever, RetrievalConfig

    platform_path = base_vector_path / "platform"
    platform_path.mkdir(parents=True, exist_ok=True)
    vector_store = FAISSAdapter({
        "embedding_dim": 384,
        "index_path": str(platform_path / "index"),
    })
    return HybridRetriever(
        embedder=GeminiEmbedder(dim=384),
        vector_store=vector_store,
        config=RetrievalConfig(),
    )


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


@dataclass
class TenantDnd:
    """A tenant's DND filter + calling-hours policy, built together (the scheduler
    and the per-tenant compliance gate consume both)."""
    filter: object   # DNDFilter
    hours: object    # CallingHoursPolicy


def build_runtime_registry(providers: TenantProviders, base_session_store: SessionStore):
    """Instantiate the per-tenant ``TenantRuntimeRegistry`` (previously dead code):
    one lazily-built instance per tenant for DND, scheduler, retriever, session
    store, chat channel, CRM, webhooks. Real impls where they exist
    (dnd/scheduler/retriever/session_store); honest stubs where only a ``fake``
    exists (crm/chat); inert ``WebhookManager`` for webhooks (the live outbound
    path is ``tenant_events``). Reuses the shared ``providers`` cache."""
    from src.auth.registry import TenantRuntimeRegistry, _PerTenantRegistry
    from src.campaign.dnd_filter import CallingHoursPolicy, DNDFilter, InMemoryDNDStore
    from src.campaign.scheduler import CallScheduler, RateLimitConfig, RetryConfig
    from src.integration.crm_client import FakeChatChannel, FakeCRMClient
    from src.integration.webhooks import WebhookManager
    from src.rag.embeddings import GeminiEmbedder
    from src.rag.retriever import HybridRetriever, RetrievalConfig

    def _dnd(tenant: TenantContext) -> TenantDnd:
        c = getattr(tenant.settings, "compliance", None)
        enabled = getattr(c, "dnd_check_enabled", None)
        return TenantDnd(
            filter=DNDFilter(InMemoryDNDStore(), enabled=True if enabled is None else enabled),
            hours=CallingHoursPolicy(
                start=getattr(c, "calling_hours_start", None) or "10:00",
                end=getattr(c, "calling_hours_end", None) or "19:00"))

    dnd_reg = _PerTenantRegistry(_dnd)

    def _scheduler(tenant: TenantContext) -> CallScheduler:
        d = dnd_reg.get(tenant)
        c = getattr(tenant.settings, "compliance", None)
        return CallScheduler(
            hours=d.hours, dnd_filter=d.filter,
            retry=RetryConfig(
                max_retry_attempts=getattr(c, "max_retry_attempts", None) or 3,
                retry_interval_hours=getattr(c, "retry_interval_hours", None) or 2),
            rate_limit=RateLimitConfig(
                max_concurrent_calls=tenant.settings.max_concurrent_calls or 10))

    def _retriever(tenant: TenantContext) -> HybridRetriever:
        # Semantic multilingual embeddings via Gemini (384-dim, matches the vector
        # store) — no torch/sentence-transformers, so the deploy image stays slim
        # and it reuses the platform GEMINI_API_KEY. The client is built lazily on
        # first ingest/query.
        return HybridRetriever(
            embedder=GeminiEmbedder(dim=384),
            vector_store=providers.get_vector_store(tenant),
            config=RetrievalConfig())

    def _session_store(tenant: TenantContext) -> SessionStore:
        return SessionStore(redis=base_session_store.redis,
                            ttl_seconds=base_session_store.ttl, tenant_id=tenant.id)

    return TenantRuntimeRegistry(
        providers=providers,
        retrievers=_PerTenantRegistry(_retriever),
        dnd=dnd_reg,
        schedulers=_PerTenantRegistry(_scheduler),
        webhooks=_PerTenantRegistry(lambda t: WebhookManager()),
        chat_channels=_PerTenantRegistry(lambda t: FakeChatChannel()),
        session_stores=_PerTenantRegistry(_session_store),
        crms=_PerTenantRegistry(lambda t: FakeCRMClient()),
    )


# --- ChatBot factory ---------------------------------------------------


def _crm_params_to_schema(params: dict) -> dict:
    """Turn a PRD-style ``{name: {type, description, source}}`` map into the
    JSON-Schema the LLM tool declaration needs. LLM-sourced params are required;
    session-sourced ones are filled from session context, so not required."""
    props: dict = {}
    required: list[str] = []
    for name, spec in (params or {}).items():
        spec = spec or {}
        prop = {"type": spec.get("type", "string")}
        if spec.get("description"):
            prop["description"] = spec["description"]
        props[name] = prop
        if spec.get("source", "llm") == "llm":
            required.append(name)
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def make_chatbot_factory(registry, sessionmaker=None, platform_retriever=None):
    """Per-(tenant, session) ChatBotAgent factory for ``chat.set_chatbot_factory``.

    Reuses the tenant's LLM, per-tenant hybrid retriever, and Redis session store
    from the runtime registry — the same per-tenant wiring the voice bridges use,
    so a chat and a call for one tenant share providers + the knowledge index.
    Loads the tenant's registered CRM tools (chat_tools) and enables the agentic
    tool loop (builtin search/escalate/offer + CRM tools).
    """
    from sqlalchemy import select

    from src.agents.base import AgentSession
    from src.agents.chatbot import ChatBotAgent
    from src.auth import secrets as crypto
    from src.chatbot.tool_executor import execute_crm_tool
    from src.interfaces.llm import ToolSpec
    from src.models.chat import ChatTool
    from src.models.tenant import TenantSecret

    async def _load_crm_tools(tenant: TenantContext):
        """Return (tool_specs, {name: exec_spec}) for the tenant's CRM tools.

        Priority:
        1. Tenant-specific tools registered in the chat_tools DB table.
        2. Platform catalog (catalog.ALL_TOOLS) built from the tenant's
           ``crm:*`` secrets + ``PLATFORM_CRM_BASE_URL`` env fallback — no
           per-tenant DB registration required.
        """
        import os
        from src.chatbot.catalog import ALL_TOOLS

        specs: list[ToolSpec] = []
        execs: dict[str, dict] = {}
        if sessionmaker is not None:
            async with sessionmaker() as db:
                rows = (await db.execute(
                    select(ChatTool).where(ChatTool.tenant_id == tenant.id)
                )).scalars().all()
                for r in rows:
                    specs.append(ToolSpec(
                        name=r.name, description=r.description,
                        parameters=_crm_params_to_schema(r.parameters)))
                    token = None
                    secret_name = (r.auth_config or {}).get("token_secret_name")
                    if secret_name:
                        sec = (await db.execute(
                            select(TenantSecret).where(
                                TenantSecret.tenant_id == tenant.id,
                                TenantSecret.name == secret_name)
                        )).scalar_one_or_none()
                        if sec is not None:
                            try:
                                token = crypto.decrypt(sec.value_encrypted)
                            except Exception:  # noqa: BLE001
                                token = None
                    execs[r.name] = {
                        "endpoint": r.endpoint, "method": r.method,
                        "parameters": r.parameters or {}, "auth_type": r.auth_type,
                        "token": token,
                        "extra_headers": (r.auth_config or {}).get("extra_headers")}

        if specs:
            return specs, execs  # tenant-specific tools take precedence

        # ── Platform catalog fallback ─────────────────────────────────────
        sr = tenant.secrets_resolved
        base_url = (sr.get("crm:base_url") or os.environ.get("PLATFORM_CRM_BASE_URL", "")).rstrip("/")
        if not base_url:
            return [], {}

        api_token = sr.get("crm:api_token")
        auth_type = sr.get("crm:auth_type") or "api_key"
        for name, spec in ALL_TOOLS.items():
            endpoint = base_url + spec["default_path"]
            specs.append(ToolSpec(
                name=name, description=spec["description"],
                parameters=_crm_params_to_schema(spec["parameters"])))
            execs[name] = {
                "endpoint": endpoint, "method": spec.get("method", "GET"),
                "parameters": spec["parameters"], "auth_type": auth_type,
                "token": api_token, "extra_headers": None,
            }
        return specs, execs

    async def factory(tenant: TenantContext, session_id: str) -> ChatBotAgent:
        # Load the chat session so we have customer_id (= logged-in user/player ID)
        # for CRM tool calls that need player-specific context.
        user_id: str | None = None
        if sessionmaker is not None:
            from src.models.chat import ChatSession as _ChatSession
            # session_id may be Redis-scoped as "{tenant_id}:{bare_id}" — strip the prefix.
            bare_session_id = session_id.split(":", 1)[-1] if ":" in session_id else session_id
            async with sessionmaker() as _db:
                _row = await _db.get(_ChatSession, bare_session_id)
                user_id = _row.customer_id if _row else None

        crm_specs, crm_execs = await _load_crm_tools(tenant)

        # operator_id = the CRM's operator identifier for this tenant. Set via
        # crm_operator_id at tenant registration; falls back to tenant.id.
        # user_id = the logged-in player's ID passed at chat session creation.
        # CRM tool parameters declared with source="session" pull from this dict.
        _operator_id = (
            getattr(tenant.settings.crm, "operator_id", None) or tenant.id
        )
        _crm_context = {"operator_id": _operator_id, "user_id": user_id}

        async def crm_executor(tc) -> dict:
            spec = crm_execs.get(tc.name)
            if spec is None:
                return {"error": f"unknown tool {tc.name}"}
            return await execute_crm_tool(
                endpoint=spec["endpoint"], method=spec["method"],
                parameters=spec["parameters"], auth_type=spec["auth_type"],
                token=spec["token"], args=tc.arguments or {}, context=_crm_context,
                extra_headers=spec.get("extra_headers"))

        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=registry.providers.get_llm(tenant),
            retriever=registry.retrievers.get(tenant),
            platform_retriever=platform_retriever,
            company_name=tenant.name,
            language_default=getattr(tenant.settings, "default_language", None) or "en",
            store=registry.session_stores.get(tenant),
            enable_tools=True,
            crm_tools=crm_specs,
            crm_executor=crm_executor,
        )

    return factory


# --- The bridge factory ------------------------------------------------


@dataclass
class _CallSpec:
    """Resolved per-call wiring."""

    tenant: TenantContext
    session_id: str
    lead_data: dict


def _override_lead_data(override: dict | None) -> dict:
    """Build agent lead_data from a dev-console override (lead_name, lead_gender)."""
    d = override or {}
    name = d.get("lead_name", "").strip()
    gender = d.get("lead_gender", "").strip()
    data: dict = {}
    if name:
        data["lead_name"] = name
        data["name"] = name
    if gender:
        data["lead_gender"] = gender
    return data


def _build_s2s_telephony_bridge(
    providers: TenantProviders, tenant: TenantContext, script: VoiceBotScript,
    slots: SlotSchema, websocket: WebSocket, session_store: SessionStore | None,
    *, encoding: str, sid_field: str, supports_clear: bool,
    call_sid_field: str = "callSid", voice_override: str | None = None,
    lead_data: dict | None = None, kb_context: str | None = None,
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
    # PLATFORM-level key (not per-tenant): pass None so GeminiLiveSession.connect
    # reads the platform GEMINI_API_KEY / GOOGLE_API_KEY. Resolving a per-tenant
    # realtime key here is what crashed s2s calls — a placeholder/invalid tenant
    # key reached connect and Gemini rejected it.
    key = None
    config = RealtimeConfig(
        model=rt.model, voice=voice, language_code=rt.language_code,
        system_instruction=build_s2s_system_instruction(script, slots, lead_data, kb_context=kb_context),
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


def _build_kb_context(platform_retriever, tenant_retriever) -> str:
    """Build a static KB context string from platform + tenant retrievers for voicebot."""
    from src.rag.context_builder import build_voicebot_kb_context
    retrievers = [r for r in [platform_retriever, tenant_retriever] if r is not None]
    return build_voicebot_kb_context(retrievers)


def make_bridge_factory(
    providers: TenantProviders,
    session_store: SessionStore | None = None,
    bridge_config: TwilioBridgeConfig | None = None,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    campaign_resolver=None,
    platform_retriever=None,
) -> Callable[[WebSocket, TenantContext], object]:
    """Return a callable suitable for ``set_bridge_factory(...)``.

    The returned factory closes over the shared registries so every
    inbound call lands on the right tenant-scoped provider clients
    without rebuilding anything.
    """
    cfg = bridge_config or TwilioBridgeConfig()

    async def factory(websocket: WebSocket, tenant: TenantContext):
        from src.api import dev_call_control

        # Per-call overrides: SID-keyed (production outbound) takes priority;
        # tenant-slug-keyed is the dev-console fallback (single call at a time).
        call_sid = (getattr(websocket, "path_params", {}) or {}).get("call_sid")
        override = (
            dev_call_control.pop_sid_override(call_sid) if call_sid else None
        ) or dev_call_control.pop_override(tenant.slug)
        mode = (override or {}).get("mode") or getattr(
            tenant.settings.pipeline, "mode", "layered")
        # Per-tenant campaign: resolve this call's script + slots from the DB
        # (?campaign=<id> on the stream URL, else from SID override, else active campaign).
        # Note: Twilio strips query strings from stream URLs, so the SID override
        # is the only way to pass campaign_id for outbound telephony calls.
        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            cid = (
                (override or {}).get("campaign_id")
                or (getattr(websocket, "query_params", {}) or {}).get("campaign")
                or None
            )
            lc = await campaign_resolver.resolve(tenant.id, cid)
            cur_script, cur_slots = lc.script, lc.slots
        # Dev-console overrides: voice (caller/agent name, gender auto-derived from voice).
        voice_override = (override or {}).get("voice", "").strip()
        caller_name_override = (override or {}).get("caller_name", "").strip()
        tts_voice_id = voice_override or tenant.settings.pipeline.tts.voice_id
        if voice_override or caller_name_override:
            from dataclasses import replace as _dc_replace
            from src.providers.voice_catalog import gender_from_voice_id
            replacements: dict = {}
            derived_gender = gender_from_voice_id(voice_override) if voice_override else ""
            if derived_gender:
                replacements["gender"] = derived_gender
            if caller_name_override:
                replacements["agent_name"] = caller_name_override
            if replacements:
                cur_script = _dc_replace(cur_script, **replacements)
        # Speech-to-speech path: when the tenant is in s2s mode, drive Gemini Live
        # over the Twilio media stream instead of the STT->LLM->TTS cascade.
        kb_ctx = _build_kb_context(platform_retriever, None)
        if mode == "s2s":
            return _build_s2s_telephony_bridge(
                providers, tenant, cur_script, cur_slots, websocket, session_store,
                encoding="mulaw", sid_field="streamSid", supports_clear=True,
                call_sid_field="callSid", voice_override=voice_override or None,
                lead_data=_override_lead_data(override), kb_context=kb_ctx)
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
                voice_id=tts_voice_id,
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
            slot_schema=cur_slots,
            script=cur_script,
            engine=engine,
            store=store,
            kb_context=kb_ctx or None,
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
    *,
    campaign_resolver=None,
    platform_retriever=None,
) -> Callable[[WebSocket, TenantContext], ExotelMediaBridge]:
    """Build an Exotel WS bridge per call, wired to the tenant's provider stack.

    Identical agent assembly to ``make_bridge_factory`` (Twilio) — only the
    last step differs: returns an ``_ExotelAgentBridge`` instead of an
    ``_AgentBridge``. The agent itself is encoding-agnostic; only the bridge
    knows whether to ship μ-law or PCM16 over the wire.
    """
    cfg = bridge_config or ExotelBridgeConfig()

    async def factory(websocket: WebSocket, tenant: TenantContext):
        from src.api import dev_call_control

        override = dev_call_control.pop_override(tenant.slug)
        mode = (override or {}).get("mode") or getattr(
            tenant.settings.pipeline, "mode", "layered")
        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            cid = (getattr(websocket, "query_params", {}) or {}).get("campaign") or None
            lc = await campaign_resolver.resolve(tenant.id, cid)
            cur_script, cur_slots = lc.script, lc.slots
        voice_override = (override or {}).get("voice", "").strip()
        caller_name_override = (override or {}).get("caller_name", "").strip()
        tts_voice_id = voice_override or tenant.settings.pipeline.tts.voice_id
        if voice_override or caller_name_override:
            from dataclasses import replace as _dc_replace
            from src.providers.voice_catalog import gender_from_voice_id
            replacements: dict = {}
            derived_gender = gender_from_voice_id(voice_override) if voice_override else ""
            if derived_gender:
                replacements["gender"] = derived_gender
            if caller_name_override:
                replacements["agent_name"] = caller_name_override
            if replacements:
                cur_script = _dc_replace(cur_script, **replacements)
        # S2S path: drive Gemini Live over the Exotel media stream (raw PCM16@8k,
        # snake_case stream_sid, no `clear` frame) when the tenant is in s2s mode.
        kb_ctx = _build_kb_context(platform_retriever, None)
        if mode == "s2s":
            return _build_s2s_telephony_bridge(
                providers, tenant, cur_script, cur_slots, websocket, session_store,
                encoding="pcm", sid_field="stream_sid", supports_clear=False,
                call_sid_field="call_sid", voice_override=voice_override or None,
                lead_data=_override_lead_data(override), kb_context=kb_ctx)
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
                voice_id=tts_voice_id,
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
            slot_schema=cur_slots,
            script=cur_script,
            engine=engine,
            store=store,
            kb_context=kb_ctx or None,
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
    *,
    campaign_resolver=None,
    platform_retriever=None,
):
    """Build a StringeeIvrBridge per call, wired to the tenant's providers.

    Same agent assembly as make_exotel_bridge_factory; HTTP-driven instead of
    WS-driven, so the call_id/base_url/fetch are passed per call by the route.
    """
    from src.api.telephony_stringee_bridge import StringeeIvrBridge

    async def factory(*, call_id, tenant, base_url, fetch):
        # Per-tenant campaign (IVR has no query string → the tenant's active one).
        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            lc = await campaign_resolver.resolve(tenant.id, None)
            cur_script, cur_slots = lc.script, lc.slots
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
            slot_schema=cur_slots,
            script=cur_script,
            engine=engine,
            store=None,
            kb_context=_build_kb_context(platform_retriever, None) or None,
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
