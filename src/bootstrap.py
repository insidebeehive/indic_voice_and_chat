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
from typing import Optional

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
from src.models.turn_metrics import record_turn_metric
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


def build_crm_retriever(
    crm_id: str,
    global_defaults: dict | None = None,
) -> Optional["HybridRetriever"]:
    """One retriever for a single CRM's shared KB (pgvector only).

    Returns ``None`` if the configured vector_store provider isn't pgvector
    (FAISS has no CRM-level tier — out of scope per
    docs/superpowers/specs/2026-07-23-crm-kb-design.md) or if construction
    fails for any reason (e.g. the pgvector table/column isn't set up yet in
    this environment) — callers treat ``None`` exactly like "this CRM has no
    KB", never an error.
    """
    from src.providers import get_vector_store
    from src.rag.embeddings import GeminiEmbedder
    from src.rag.retriever import HybridRetriever, RetrievalConfig

    vs_cfg = dict((global_defaults or {}).get("vector_store", {}))
    provider = vs_cfg.get("provider", "faiss")
    if provider != "pgvector":
        return None
    vs_cfg["embedding_dim"] = vs_cfg.get("embedding_dim", 384)
    vs_cfg["crm_id"] = crm_id
    vs_cfg.pop("tenant_id", None)

    try:
        vector_store = get_vector_store(vs_cfg)
    except Exception:
        log.exception("build_crm_retriever: failed to build vector store", extra={"crm_id": crm_id})
        return None
    return HybridRetriever(
        embedder=GeminiEmbedder(dim=384),
        vector_store=vector_store,
        config=RetrievalConfig(),
    )


class PerCrmRetrieverRegistry:
    """Lazy ``crm_id -> HybridRetriever | None`` cache (pgvector only).

    Same shape as ``_PerTenantRegistry`` (``src/auth/registry.py``), keyed by
    a plain ``crm_id`` string instead of a ``TenantContext`` — there is no
    per-CRM eviction need today (pgvector reads live from the DB on every
    query; nothing about a CRM's KB docs is cached client-side beyond the
    thin adapter object itself).
    """

    def __init__(self, global_defaults: dict | None = None) -> None:
        self._global_defaults = global_defaults or {}
        self._items: dict[str, Optional["HybridRetriever"]] = {}

    def get(self, crm_id: str) -> Optional["HybridRetriever"]:
        if crm_id not in self._items:
            self._items[crm_id] = build_crm_retriever(crm_id, self._global_defaults)
        return self._items[crm_id]


def _crm_retriever_for(
    tenant: TenantContext, crm_retrievers: Optional[PerCrmRetrieverRegistry],
) -> Optional["HybridRetriever"]:
    """Resolve this tenant's linked CRM's retriever, or None (no link / no
    registry / that CRM has no usable KB) — never an error."""
    if crm_retrievers is None:
        return None
    crm_id = getattr(tenant.settings, "crm_id", None)
    if not crm_id:
        return None
    return crm_retrievers.get(crm_id)


def _tenant_retriever_for(
    tenant: TenantContext, registry: "TenantRuntimeRegistry | None",
) -> Optional["HybridRetriever"]:
    """Resolve this tenant's own opt-in KB retriever, or None (no runtime
    registry available at this call site) — never an error.

    Mirrors ``_crm_retriever_for`` above, but for the per-tenant tier
    (``src.api.knowledge``'s chat-side ``_active_retrievers`` equivalent for
    voice). Callers merge this with ``_crm_retriever_for``'s result via
    ``_build_kb_context`` so voice gets the same [CRM, tenant] KB coverage
    chat already has.
    """
    if registry is None:
        return None
    return registry.retrievers.get(tenant)


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


async def resolve_crm_tools(
    tenant: TenantContext, sessionmaker,
) -> tuple[list, dict, str]:
    """Resolve a tenant's CRM tools fresh from the DB/env — no caching.

    This is the single source of truth for "what CRM tools does this tenant
    actually get" — both the cached chat-runtime path (``_load_crm_tools_uncached``
    below, via the per-tenant registry) and the ``GET /chat/tools/resolved``
    diagnostic endpoint call into this same function.

    Priority:
    1. Tenant-specific tools registered in the chat_tools DB table.
    2. The catalog of the ``Crm`` entity this tenant is linked to
       (``tenant.settings.crm_id`` -> ``Crm``/``CrmTool`` DB rows), using the
       tenant's own ``crm:*`` secrets for auth — no per-tenant DB tool
       registration required.

    Returns ``(tool_specs, {name: exec_spec}, source)`` where ``source`` is
    one of: ``"tenant"`` (tenant-registered chat_tools rows), ``"crm_catalog"``
    (the linked Crm's DB-backed tool catalog), or ``"none"`` (nothing
    resolved — no tools available at all).
    """
    from sqlalchemy import select

    from src.auth import secrets as crypto
    from src.interfaces.llm import ToolSpec
    from src.models.chat import ChatTool
    from src.models.tenant import TenantSecret

    specs: list[ToolSpec] = []
    execs: dict[str, dict] = {}
    # Moved to the top so it's available to BOTH branches below (tenant-
    # registered tools and the platform-fallback catalog) — needed for the
    # new, independent, tenant-level crm:x_api_key secret (sent unconditionally
    # as X-API-Key alongside whatever the existing token/auth_type produces).
    sr = tenant.secrets_resolved
    x_api_key = sr.get("crm:x_api_key")
    if sessionmaker is not None:
        async with sessionmaker() as db:
            rows = (await db.execute(
                select(ChatTool).where(ChatTool.tenant_id == tenant.id)
            )).scalars().all()
            # One batched query for every tool's token secret instead of one
            # query per tool (this loop used to be N+1 — with tenants
            # registering a dozen-plus tools, that's a dozen-plus sequential
            # round-trips on every single new chat session).
            secret_names = {
                (r.auth_config or {}).get("token_secret_name")
                for r in rows
                if (r.auth_config or {}).get("token_secret_name")
            }
            secrets_by_name: dict[str, str] = {}
            if secret_names:
                sec_rows = (await db.execute(
                    select(TenantSecret).where(
                        TenantSecret.tenant_id == tenant.id,
                        TenantSecret.name.in_(secret_names))
                )).scalars().all()
                secrets_by_name = {s.name: s.value_encrypted for s in sec_rows}
            for r in rows:
                specs.append(ToolSpec(
                    name=r.name, description=r.description,
                    parameters=_crm_params_to_schema(r.parameters)))
                token = None
                secret_name = (r.auth_config or {}).get("token_secret_name")
                if secret_name and secret_name in secrets_by_name:
                    try:
                        token = crypto.decrypt(secrets_by_name[secret_name])
                    except Exception:  # noqa: BLE001
                        token = None
                execs[r.name] = {
                    "endpoint": r.endpoint, "method": r.method,
                    "parameters": r.parameters or {}, "auth_type": r.auth_type,
                    "token": token, "x_api_key": x_api_key,
                    "extra_headers": (r.auth_config or {}).get("extra_headers")}

    if specs:
        return specs, execs, "tenant"  # tenant-specific tools take precedence

    # ── CRM catalog (tenant linked to a Crm entity) ─────────────────────
    crm_id = tenant.settings.crm_id
    if not crm_id or sessionmaker is None:
        return [], {}, "none"

    from src.models.crm import Crm, CrmTool

    async with sessionmaker() as db:
        crm = await db.get(Crm, crm_id)
        if crm is None:
            return [], {}, "none"
        crm_tool_rows = (await db.execute(
            select(CrmTool).where(CrmTool.crm_id == crm_id)
        )).scalars().all()

    if not crm_tool_rows:
        return [], {}, "none"

    api_token = sr.get("crm:api_token")
    # Same operator_id resolution as the crm_executor closure in
    # make_chatbot_factory below: the CRM's operator identifier for this
    # tenant, falling back to the tenant's own id. Every crm-catalog tool
    # must carry this as the "operatorid" header — previously hardcoded
    # to None here, so the old platform-fallback path never sent it at all
    # (only the tenant-registered chat_tools branch did, via auth_config).
    operator_id = getattr(tenant.settings.crm, "operator_id", None) or tenant.id
    extra_headers = {"operatorid": operator_id}

    for row in crm_tool_rows:
        endpoint = crm.base_url.rstrip("/") + row.endpoint
        specs.append(ToolSpec(
            name=row.name, description=row.description,
            parameters=_crm_params_to_schema(row.parameters)))
        execs[row.name] = {
            "endpoint": endpoint, "method": row.method,
            "parameters": row.parameters or {}, "auth_type": crm.auth_type,
            "token": api_token, "x_api_key": x_api_key,
            "extra_headers": extra_headers,
        }
    return specs, execs, "crm_catalog"


def make_chatbot_factory(registry, sessionmaker=None, crm_retrievers: "PerCrmRetrieverRegistry | None" = None):
    """Per-(tenant, session) ChatBotAgent factory for ``chat.set_chatbot_factory``.

    Uses the platform-level LLM (global defaults, no tenant override) so all chat
    sessions use the same model regardless of per-tenant pipeline_config.llm.
    Loads the tenant's registered CRM tools (chat_tools) and enables the agentic
    tool loop (builtin search/escalate/offer + CRM tools).
    """
    from src.agents.base import AgentSession
    from src.agents.chatbot import ChatBotAgent
    from src.auth.registry import _AsyncPerTenantRegistry
    from src.chatbot.deposit_verification import submit_deposit_verification
    from src.chatbot.tool_executor import execute_crm_tool
    from src.chatbot.tools import SUBMIT_DEPOSIT_VERIFICATION_TOOL_SPEC

    async def _load_crm_tools_uncached(tenant: TenantContext):
        """Return (tool_specs, {name: exec_spec}) for the tenant's CRM tools.

        Thin delegate to the module-level ``resolve_crm_tools`` (also used by
        the ``GET /chat/tools/resolved`` diagnostic endpoint) — drops the
        ``source`` value since this cached path's callers only expect a
        2-tuple.
        """
        specs, execs, _source = await resolve_crm_tools(tenant, sessionmaker)
        return specs, execs

    # A tenant's registered CRM tools rarely change, but _load_crm_tools_uncached
    # was being re-run from scratch on EVERY new chat session (N+1 DB queries
    # each) — under concurrent session bursts this exhausts the DB connection
    # pool and new sessions hang waiting for a connection. Cache per tenant,
    # invalidated via the same registry.evict_all/evict_tenant path a tenant
    # config reload already uses (src/main.py: resolver.on_reload).
    if registry.crm_tools is None:
        registry.crm_tools = _AsyncPerTenantRegistry(_load_crm_tools_uncached)
    crm_tools_registry = registry.crm_tools

    async def _load_crm_tools(tenant: TenantContext):
        return await crm_tools_registry.get(tenant)

    # Sentinel distinguishing "caller didn't supply customer_id" from a session
    # that genuinely has none — an anonymous session's None must not trigger a
    # redundant DB lookup.
    _CUSTOMER_ID_UNSET = object()

    async def factory(
        tenant: TenantContext, session_id: str, *,
        customer_id: object = _CUSTOMER_ID_UNSET,
        ticket_id: str | None = None,
    ) -> ChatBotAgent:
        # customer_id (= logged-in user/player ID) feeds CRM tool calls that
        # need player-specific context. The WS connect path passes it from the
        # ChatSession row it already fetched; only legacy callers without the
        # row fall back to loading it here.
        # session_id may be Redis-scoped as "{tenant_id}:{bare_id}" — strip the
        # prefix; this is the bare id chat_sessions/chat_messages/
        # deposit_verification_requests rows are keyed by.
        bare_session_id = session_id.split(":", 1)[-1] if ":" in session_id else session_id

        if customer_id is not _CUSTOMER_ID_UNSET:
            user_id = customer_id
        else:
            user_id = None
            if sessionmaker is not None:
                from src.models.chat import ChatSession as _ChatSession
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

        async def crm_executor(tc, *, timeout_s: float) -> dict:
            spec = crm_execs.get(tc.name)
            if spec is None:
                return {"error": f"unknown tool {tc.name}"}
            return await execute_crm_tool(
                endpoint=spec["endpoint"], method=spec["method"],
                parameters=spec["parameters"], auth_type=spec["auth_type"],
                token=spec["token"], args=tc.arguments or {}, context=_crm_context,
                x_api_key=spec.get("x_api_key"),
                extra_headers=spec.get("extra_headers"),
                session_id=bare_session_id, ticket_id=ticket_id,
                timeout_s=timeout_s)

        tool_specs = list(crm_specs)
        dv_config = getattr(tenant.settings, "deposit_verification", None)
        deposit_verification_executor = None
        # A resolvable signing secret is REQUIRED, not optional: the inbound
        # verdict callback (src/api/deposit_verification.py) rejects any
        # callback it can't HMAC-verify with 401, and treats a missing secret
        # as a verification failure rather than "unsigned is fine". Registering
        # the tool without one would offer the LLM a flow whose verdict can
        # never come back — every request would sit pending until it timed out
        # and escalated, looking like an unresponsive vendor rather than the
        # misconfiguration it actually is.
        dv_secret = (
            tenant.secret_optional(dv_config.webhook_secret_env)
            if dv_config is not None else None
        )
        if (
            dv_config is not None
            and dv_config.enabled
            and dv_config.webhook_url
            and dv_secret
            and sessionmaker is not None
        ):
            tool_specs.append(SUBMIT_DEPOSIT_VERIFICATION_TOOL_SPEC)

            async def deposit_verification_executor(tc, *, timeout_s: float) -> dict:
                # Lazy per-call lookup of the module-level media store injected
                # via chat.set_media_store(...) at app startup (src/main.py).
                # There's no public getter for it (only the setter) and
                # src/bootstrap.py is imported by src/api/chat.py's own
                # dependency graph at startup, so a module-level `from
                # src.api.chat import ...` here would risk an import cycle;
                # this lazy import (evaluated fresh on every call, matching
                # the lazy-import pattern already used elsewhere in this
                # closure/module) reads the current value each time, which is
                # what set_media_store's later reassignment requires anyway.
                from src.api import chat as _chat_api
                return await submit_deposit_verification(
                    tenant=tenant,
                    session_id=bare_session_id,
                    order_id=(tc.arguments or {}).get("order_id", ""),
                    sessionmaker=sessionmaker,
                    media_store=_chat_api._media_store,
                    timeout_s=timeout_s,
                    ticket_id=ticket_id,
                )

        elif dv_config is not None and dv_config.enabled and dv_config.webhook_url and not dv_secret:
            log.warning(
                "deposit verification is enabled with a webhook_url but no resolvable "
                "webhook secret — the tool is NOT being registered, because the verdict "
                "callback would be rejected as unsigned (401). Set webhook_secret_env "
                "(or PATCH the tenant's deposit_verification.webhook_secret).",
                extra={
                    "tenant_id": tenant.id,
                    "webhook_secret_env": dv_config.webhook_secret_env,
                },
            )

        # Platform-level LLM identity (provider + model) — same global default
        # dict get_platform_llm() itself builds the client from — threaded
        # into the agent for src/api/chat_cost.py's per-turn cost lookup.
        # getattr-defensive: some tests stub registry.providers as a bare
        # SimpleNamespace(get_platform_llm=...) without global_defaults.
        _llm_defaults = getattr(registry.providers, "global_defaults", {}).get("llm", {})
        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=registry.providers.get_platform_llm(),
            retriever=registry.retrievers.get(tenant),
            crm_retriever=_crm_retriever_for(tenant, crm_retrievers),
            company_name=tenant.name,
            language_default=getattr(tenant.settings, "default_language", None) or "en",
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
            prompt_pack=getattr(tenant.settings, "prompt_pack", None) or "generic",
            store=registry.session_stores.get(tenant),
            enable_tools=True,
            crm_tools=tool_specs,
            crm_executor=crm_executor,
            deposit_verification_executor=deposit_verification_executor,
            llm_provider=_llm_defaults.get("provider") or "",
            llm_model=_llm_defaults.get("model") or "",
            session_id=bare_session_id,
            ticket_id=ticket_id,
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


def _build_s2s_agent_and_config(
    providers: TenantProviders, tenant: TenantContext, script: VoiceBotScript,
    slots: SlotSchema, session_store: SessionStore | None,
    *, voice_override: str | None = None, lead_data: dict | None = None,
    kb_context: str | None = None,
):
    """Assemble everything a Gemini-Live S2S call needs EXCEPT the transport-
    specific bridge itself: the agent, the ``RealtimeConfig``, the session
    connector, the LLM/TTS provider clients, and the tenant's timezone.

    Shared by both transport shapes:
    - ``_build_s2s_telephony_bridge`` (Twilio/Exotel — wraps the result in a
      ``TelephonyLiveBridge`` bound to an already-open WebSocket).
    - ``make_livekit_bridge_factory`` (LiveKit — the caller instead gets back
      a builder closure, because the real room transport objects don't exist
      until a separate runner connects to the room).

    Returns ``(agent, config, connect_session, llm, tts, tenant_timezone)``.
    """
    import uuid

    from src.api.live_bridge_base import RECORD_TURN_SIGNAL
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
        slot_schema=slots, script=script, engine=engine, store=store,
        record_metric=lambda payload: record_turn_metric(tenant_id=tenant.id, **payload))
    # Voice: a dev-console override wins (validated against allowed_voices); else config.
    voice = (voice_override or "").strip() or rt.voice
    allowed = getattr(rt, "allowed_voices", None)
    if allowed and voice not in allowed:
        voice = rt.voice
    # PLATFORM-level key (not per-tenant): pass None so GeminiLiveSession.connect
    # reads the platform GEMINI_API_KEY. Resolving a per-tenant
    # realtime key here is what crashed s2s calls — a placeholder/invalid tenant
    # key reached connect and Gemini rejected it.
    key = None
    config = RealtimeConfig(
        model=rt.model, voice=voice, language_code=rt.language_code,
        system_instruction=build_s2s_system_instruction(script, slots, lead_data, kb_context=kb_context),
        tools=[RECORD_TURN_SIGNAL])

    async def connect(cfg: RealtimeConfig):
        return await GeminiLiveSession.connect(cfg, api_key=key)

    log.info("s2s agent+config built", extra={
        "tenant": tenant.slug, "session_id": session_id, "voice": voice, "model": rt.model})
    tts = providers.get_tts(tenant)
    tenant_timezone = getattr(tenant.settings, "timezone", "Asia/Kolkata")
    return agent, config, connect, llm, tts, tenant_timezone


def _build_s2s_telephony_bridge(
    providers: TenantProviders, tenant: TenantContext, script: VoiceBotScript,
    slots: SlotSchema, websocket: WebSocket, session_store: SessionStore | None,
    *, encoding: str, sid_field: str, supports_clear: bool,
    call_sid_field: str = "callSid", voice_override: str | None = None,
    lead_data: dict | None = None, kb_context: str | None = None,
    transfer_webhook_url_override: str | None = None,
):
    """Build a TelephonyLiveBridge (Gemini Live over the media stream) for a call
    whose tenant has pipeline.mode == 's2s'. Mirrors the cascade agent assembly
    but returns the S2S bridge; reuses the dev-console S2S wiring shape."""
    from src.api.telephony_live_bridge import TelephonyLiveBridge

    agent, config, connect, llm, tts, tenant_timezone = _build_s2s_agent_and_config(
        providers, tenant, script, slots, session_store,
        voice_override=voice_override, lead_data=lead_data, kb_context=kb_context)

    log.info("s2s telephony bridge built call", extra={
        "tenant": tenant.slug, "voice": config.voice, "model": config.model, "encoding": encoding})
    # Transfer-hold: TTS for failure apology; webhook so CS learns to look for a human.
    # A per-call override (e.g. bridge console) takes priority over tenant config.
    _wh_url = transfer_webhook_url_override or getattr(tenant.settings, "events_webhook_url", None)
    _wh_sec_env = getattr(tenant.settings, "events_webhook_secret_env", None)
    _wh_secret = (tenant.secret_optional(_wh_sec_env)
                  if _wh_sec_env and hasattr(tenant, "secret_optional") else None)
    return TelephonyLiveBridge(
        websocket=websocket, agent=agent, config=config, connect_session=connect, llm=llm,
        tts=tts, tenant_timezone=tenant_timezone, tenant_id=tenant.id,
        pronunciation_overrides=getattr(tenant.settings, "pronunciation_overrides", None),
        encoding=encoding, sid_field=sid_field, supports_clear=supports_clear,
        call_sid_field=call_sid_field,
        transfer_webhook_url=_wh_url,
        transfer_webhook_secret=_wh_secret)


async def _build_kb_context(crm_retriever, tenant_retriever) -> str:
    """Build a static KB context string from the CRM + tenant retrievers for voicebot."""
    from src.rag.context_builder import build_voicebot_kb_context
    retrievers = [r for r in [crm_retriever, tenant_retriever] if r is not None]
    return await build_voicebot_kb_context(retrievers)


class LiveKitModeNotSupported(Exception):
    """Raised by ``make_livekit_bridge_factory``'s factory when the tenant isn't
    in s2s pipeline mode. LiveKit room-join is s2s-only — there is no cascade
    (STT->LLM->TTS) LiveKit path, and no per-call mode override for it."""


def make_livekit_bridge_factory(
    providers: TenantProviders,
    session_store: SessionStore | None = None,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    campaign_resolver=None,
    crm_retrievers: "PerCrmRetrieverRegistry | None" = None,
    registry: "TenantRuntimeRegistry | None" = None,
):
    """Returns a factory(tenant, room_name, meta) -> builder-callable.

    The builder callable takes the real LiveKit transport objects (only
    available once a runner has actually connected to the room) and
    constructs the ``LiveKitBridge``. This two-stage split exists because,
    unlike Twilio/Exotel (which build the bridge inside an already-open
    WebSocket route), LiveKit has no inbound request to hang off of — the
    room connection happens separately, after tenant/campaign resolution
    (Phase 4's runner owns that second stage; this module only resolves
    config and hands back a builder).

    LiveKit room-join is s2s-only: there is no per-call mode override here
    (unlike Twilio/Exotel's dev-console override) — mode is tenant-level
    only, and any ``mode`` key present in ``meta`` is logged and ignored.
    """

    async def factory(tenant: TenantContext, room_name: str, meta: dict):
        meta = meta or {}
        if meta.get("mode") is not None:
            log.warning(
                "livekit bridge factory: ignoring 'mode' in call metadata — "
                "LiveKit room-join mode is tenant-level only, no per-call override",
                extra={"tenant": tenant.slug, "room_name": room_name, "meta_mode": meta.get("mode")})
        if tenant.settings.pipeline.mode != "s2s":
            raise LiveKitModeNotSupported(
                f"tenant {tenant.slug!r} is not in s2s mode — LiveKit room-join requires s2s")

        campaign_id = meta.get("campaign_id")
        voice_override = (meta.get("voice") or "").strip() or None
        lead = meta.get("lead") or {}
        lead_data = _override_lead_data({
            "lead_name": lead.get("name", ""),
            "lead_gender": lead.get("gender", ""),
        })
        # call_ref isn't agent config — it's for the runner/webhook layer to echo
        # back in outbound webhooks, so it's intentionally not consumed here.

        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            lc = await campaign_resolver.resolve(tenant.id, campaign_id)
            cur_script, cur_slots = lc.script, lc.slots

        kb_ctx = await _build_kb_context(
            _crm_retriever_for(tenant, crm_retrievers), _tenant_retriever_for(tenant, registry))

        agent, config, connect_session, llm, tts, tenant_timezone = _build_s2s_agent_and_config(
            providers, tenant, cur_script, cur_slots, session_store,
            voice_override=voice_override, lead_data=lead_data, kb_context=kb_ctx)

        log.info("livekit bridge factory resolved call config", extra={
            "tenant": tenant.slug, "room_name": room_name, "voice": config.voice})

        async def build(*, audio_stream, audio_source, frame_factory, on_hangup=None):
            from src.api.livekit_bridge import LiveKitBridge
            return LiveKitBridge(
                agent=agent, config=config, connect_session=connect_session,
                llm=llm, tts=tts, tenant_timezone=tenant_timezone,
                audio_stream=audio_stream, audio_source=audio_source,
                frame_factory=frame_factory, room_name=room_name, on_hangup=on_hangup,
                tenant_id=tenant.id)

        return build

    return factory


def make_bridge_factory(
    providers: TenantProviders,
    session_store: SessionStore | None = None,
    bridge_config: TwilioBridgeConfig | None = None,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    campaign_resolver=None,
    crm_retrievers: "PerCrmRetrieverRegistry | None" = None,
    registry: "TenantRuntimeRegistry | None" = None,
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
        kb_ctx = await _build_kb_context(
            _crm_retriever_for(tenant, crm_retrievers), _tenant_retriever_for(tenant, registry))
        if mode == "s2s":
            return _build_s2s_telephony_bridge(
                providers, tenant, cur_script, cur_slots, websocket, session_store,
                encoding="mulaw", sid_field="streamSid", supports_clear=True,
                call_sid_field="callSid", voice_override=voice_override or None,
                lead_data=_override_lead_data(override), kb_context=kb_ctx,
                transfer_webhook_url_override=(override or {}).get("transfer_webhook_url") or None)
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
                extra_pronunciations=getattr(tenant.settings, "pronunciation_overrides", None),
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
            record_metric=lambda payload: record_turn_metric(tenant_id=tenant.id, **payload),
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
    crm_retrievers: "PerCrmRetrieverRegistry | None" = None,
    registry: "TenantRuntimeRegistry | None" = None,
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
        kb_ctx = await _build_kb_context(
            _crm_retriever_for(tenant, crm_retrievers), _tenant_retriever_for(tenant, registry))
        if mode == "s2s":
            return _build_s2s_telephony_bridge(
                providers, tenant, cur_script, cur_slots, websocket, session_store,
                encoding="pcm", sid_field="stream_sid", supports_clear=False,
                call_sid_field="call_sid", voice_override=voice_override or None,
                lead_data=_override_lead_data(override), kb_context=kb_ctx,
                transfer_webhook_url_override=(override or {}).get("transfer_webhook_url") or None)
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
                extra_pronunciations=getattr(tenant.settings, "pronunciation_overrides", None),
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
            record_metric=lambda payload: record_turn_metric(tenant_id=tenant.id, **payload),
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
    crm_retrievers: "PerCrmRetrieverRegistry | None" = None,
    registry: "TenantRuntimeRegistry | None" = None,
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
                extra_pronunciations=getattr(tenant.settings, "pronunciation_overrides", None),
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)

        import uuid

        kb_ctx = await _build_kb_context(
            _crm_retriever_for(tenant, crm_retrievers),
            _tenant_retriever_for(tenant, registry),
        )
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
            kb_context=kb_ctx or None,
            record_metric=lambda payload: record_turn_metric(tenant_id=tenant.id, **payload),
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
