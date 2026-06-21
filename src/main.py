"""FastAPI app entry point.

Lifespan-based startup:
- configure structured logging
- initialize SQLAlchemy async engine + Redis pool
- discover every tenant in ``config/tenants/`` and register it on the
  in-memory ``TenantResolver``
- build the ``TenantRuntimeRegistry`` so per-tenant providers, retrievers,
  DND stores, schedulers, webhook managers, etc. are lazily wired on first
  use of each tenant

``GET /health`` probes infrastructure + reports per-tenant provider routing.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env into the process environment for local runs, so settings read via
# os.environ (VOX_SECRET_KEY, VOX_ADMIN_TOKENS, TENANT_*_API_TOKENS, …) work
# without a manual `source .env`. override=False → real env (e.g. Northflank)
# always wins, and a missing file is a no-op. Skipped under pytest so test
# fixtures control the environment.
if "pytest" not in sys.modules:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except ImportError:
        pass

import redis.asyncio as redis_async
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sqlalchemy import text

from src.api import (
    api_router,
    chat as chat_api,
    knowledge as knowledge_api,
    telephony_hooks,
)
from src.api.dev_console import (
    dev_console_enabled,
    dev_router,
    make_browser_bridge_factory,
    make_live_bridge_factory,
    set_browser_bridge_factory,
    set_live_bridge_factory,
)
from src.api.dev_console import (
    ws_router as dev_ws_router,
)
from src.api.call_store import (
    record_outcome,
    set_call_outcome_persister,
    set_tenant_event_notifier,
)
from src.integration.tenant_events import deliver as deliver_tenant_event
from src.auth.db_resolver import DbTenantResolver
from src.auth.middleware import set_admin_tokens, set_tenant_resolver
from src.auth.seed import seed_campaigns_if_empty, seed_if_empty, seed_provider_costs
from src.bootstrap import (
    build_platform_retriever,
    build_provider_registry,
    build_runtime_registry,
    make_bridge_factory,
    make_chatbot_factory,
    make_exotel_bridge_factory,
    make_stringee_bridge_factory,
)
from src.config import Settings, get_settings
from src.config_tenant import TenantSettings
from src.dialogue.campaign_loader import active_campaign_slug, load_campaign
from src.dialogue.campaign_resolver import DbCampaignResolver
from src.dialogue.context import SessionStore
from src.models.database import dispose_engine, ensure_schema, get_engine, get_sessionmaker
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _admin_tokens_from_env() -> list[str]:
    """Comma-separated admin tokens in ``VOX_ADMIN_TOKENS``. Empty if unset."""
    raw = os.environ.get("VOX_ADMIN_TOKENS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_callback(value):
    """Parse an ISO callback datetime from an outcome payload (None-safe)."""
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


# How often the background sweep auto-closes calls stuck in an active status
# (their finalization never fired). See call_store.reap_stale_calls.
_REAP_INTERVAL_S = 600


async def _reap_stale_calls_loop() -> None:
    """Periodically close conversation rows stuck `in_progress`/`answered` so a
    call whose finalization never fired (e.g. no recording webhook) doesn't
    linger forever. Runs once at startup, then every _REAP_INTERVAL_S."""
    from src.api.call_store import reap_stale_calls

    sm = get_sessionmaker()
    while True:
        try:
            async with sm() as s:
                n = await reap_stale_calls(s)
            if n:
                log.info("reaped stale active calls", extra={"count": n})
        except Exception:  # noqa: BLE001 - the reaper must never die (CancelledError still propagates)
            log.exception("stale-call reaper failed")
        await asyncio.sleep(_REAP_INTERVAL_S)


async def _seed_global_kb(platform_retriever, sessionmaker) -> None:
    """Re-ingest bundled global KB docs into the platform FAISS index at startup.

    FAISS is in-memory and lost on each deploy; this seeds it on boot from docs
    bundled at data/kb/global/. Uses deterministic doc_ids (filename-based) so
    KBDocument DB rows are replaced, not duplicated, across restarts. Docs are
    stored with tenant_id=NULL (platform-level, shared across all tenants).
    """
    from sqlalchemy import delete as sa_delete

    from src.interfaces.vector_store import Document
    from src.models.benchmark import KBDocument
    from src.rag.ingestion import ChunkConfig, detect_language, get_chunker, parse_document

    kb_dir = Path("data/kb/global")
    if not kb_dir.is_dir():
        return
    exts = {".md", ".txt", ".pdf", ".docx", ".csv"}
    files = sorted(
        p for p in kb_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
    )
    if not files:
        return

    chunker = get_chunker(ChunkConfig())
    total = 0
    for f in files:
        doc_id = f"global_kb_{f.stem}"
        try:
            text = parse_document(f.name, f.read_bytes())
            if not text.strip():
                continue
            language = detect_language(text)
            raw_chunks = chunker(text, {
                "filename": f.name, "document_id": doc_id, "language": language,
            })
            if not raw_chunks:
                continue
            docs = [
                Document(
                    id=f"{doc_id}::chunk-{c.index}",
                    content=c.text,
                    metadata={**c.metadata, "section": c.index, "page": c.index},
                )
                for c in raw_chunks
            ]
            n = await platform_retriever.index(docs)
            total += n
            async with sessionmaker() as session:
                await session.execute(
                    sa_delete(KBDocument).where(KBDocument.id == doc_id)
                )
                session.add(KBDocument(
                    id=doc_id, tenant_id=None, filename=f.name,
                    source_type=f.suffix.lstrip(".").lower(),
                    language=language, chunk_count=n,
                    extra_data={"chunk_ids": [d.id for d in docs]},
                ))
                await session.commit()
        except Exception:  # noqa: BLE001 - one bad file must not abort the whole seed
            log.exception("global KB seed failed", extra={"file": f.name})
    if total:
        log.info("platform KB seeded", extra={"files": len(files), "chunks": total})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_logging(settings.app.log_level)
    log.info("startup", extra={"app": settings.app.name, "version": settings.app.version})

    # Eagerly create engine + redis pool so missing config fails on boot, not first request.
    get_engine(settings.database.url)
    # Ensure our schema exists before anything touches a table (no-op on SQLite).
    await ensure_schema(settings.database.url)
    redis_client = redis_async.from_url(settings.redis.url, decode_responses=False)
    app.state.redis = redis_client
    app.state.settings = settings

    # --- Tenants: DB-backed (YAML is migrated in on first boot, then ignored) ---
    sessionmaker = get_sessionmaker()
    seeded = await seed_if_empty(sessionmaker)
    if seeded:
        log.info("seeded tenants from YAML into DB", extra={"count": seeded})
    await seed_provider_costs(sessionmaker)
    seeded_campaigns = await seed_campaigns_if_empty(sessionmaker)
    if seeded_campaigns:
        log.info("seeded default campaigns from VOX_CAMPAIGN", extra={"count": seeded_campaigns})
    resolver = DbTenantResolver(sessionmaker)
    await resolver.reload()
    set_tenant_resolver(resolver)
    set_admin_tokens(_admin_tokens_from_env())
    app.state.tenant_resolver = resolver
    app.state.tenants = resolver.loaded_settings()

    # Bridges persist a finished call's outcome + cost to its conversations row
    # (keyed by provider Call SID) through this hook at teardown.
    async def _persist_call_outcome(call_sid: str, payload: dict) -> None:
        async with sessionmaker() as session:
            await record_outcome(
                session, call_sid,
                outcome=payload.get("outcome"),
                summary=payload.get("summary"),
                notes=payload.get("notes"),
                callback_at=_parse_callback(payload.get("callback_datetime")),
            )
    set_call_outcome_persister(_persist_call_outcome)

    # Outbound per-tenant call-event webhook: call_store hands us a ready-built
    # envelope at call start (call.initiated) and end (call.completed + outcome);
    # we resolve the tenant's events_webhook_url + secret and POST it signed,
    # fire-and-forget so a slow tenant endpoint never blocks call handling. The
    # config fields live on TenantSettings (read via getattr so this is a clean
    # no-op until the active_creds WIP adds events_webhook_url/secret_env).
    def _tenant_settings_by_id(tenant_id: str):
        for t in (getattr(app.state, "tenants", {}) or {}).values():
            if getattr(t, "id", None) == tenant_id:
                return t
        return None

    async def _notify_tenant_event(envelope: dict) -> None:
        settings = _tenant_settings_by_id(envelope.get("tenant_id"))
        tel = getattr(getattr(settings, "pipeline", None), "telephony", None)
        url = getattr(tel, "events_webhook_url", None) if tel else None
        if not url:
            return
        secret_env = getattr(tel, "events_webhook_secret_env", None)
        # Resolve the signing secret from the tenant's DECRYPTED secrets first
        # (so it can be stored per-tenant like the telephony keys), falling back
        # to process env. secret_optional() never raises — a missing secret just
        # means the event is sent unsigned.
        secret = None
        if secret_env:
            resolver = getattr(app.state, "tenant_resolver", None)
            ctx = None
            if resolver is not None and settings is not None and hasattr(resolver, "resolve_by_slug"):
                ctx = await resolver.resolve_by_slug(settings.slug)
            secret = ctx.secret_optional(secret_env) if ctx else os.environ.get(secret_env)
        # Detached so delivery (retries/backoff) never blocks the caller.
        asyncio.create_task(deliver_tenant_event(url, envelope, secret))
    set_tenant_event_notifier(_notify_tenant_event)

    # --- Bridge factory: turn an inbound Twilio WS into a live agent ----
    providers = build_provider_registry(
        global_defaults={
            "stt": settings.pipeline.stt.model_dump(),
            "llm": settings.pipeline.llm.model_dump(),
            "tts": settings.pipeline.tts.model_dump(),
            "telephony": settings.pipeline.telephony.model_dump(),
            "vector_store": settings.pipeline.vector_store.model_dump(),
        },
    )
    base_session_store = SessionStore(
        redis=redis_client, ttl_seconds=settings.redis.session_ttl_seconds
    )

    # Per-tenant runtime registry: one lazily-built instance per tenant for
    # providers + DND / scheduler / retriever / session store / chat / CRM /
    # webhooks. Real where impls exist, honest stubs for the fake-only ones.
    runtime_registry = build_runtime_registry(providers, base_session_store)
    app.state.registry = runtime_registry
    platform_retriever = build_platform_retriever()
    # Drop ALL cached per-tenant instances (providers + sub-registries) whenever
    # tenants reload (e.g. a key/config update via the tenant API) so the new
    # config takes effect.
    resolver.on_reload = runtime_registry.evict_all
    campaign = load_campaign(active_campaign_slug())
    log.info(
        "campaign loaded",
        extra={"slug": active_campaign_slug(), "agent": campaign.script.agent_name,
               "slots": list(campaign.slots.specs.keys())},
    )
    # Per-tenant campaign resolution: EVERY bridge (telephony + dev console)
    # resolves this call's script + slots from the tenant's DB campaign
    # (?campaign=<id> on the media-stream URL, else the tenant's active campaign)
    # — no global fallback. Every tenant is seeded a campaign on boot, so a call
    # never runs a shared/global script.
    campaign_resolver = DbCampaignResolver(sessionmaker)
    telephony_hooks.set_bridge_factory(
        make_bridge_factory(
            providers=providers, session_store=base_session_store,
            campaign_resolver=campaign_resolver,
            platform_retriever=platform_retriever,
        )
    )
    telephony_hooks.set_exotel_bridge_factory(
        make_exotel_bridge_factory(
            providers=providers, session_store=base_session_store,
            campaign_resolver=campaign_resolver,
            platform_retriever=platform_retriever,
        )
    )
    telephony_hooks.set_stringee_bridge_factory(
        make_stringee_bridge_factory(
            providers=providers, campaign_resolver=campaign_resolver,
            platform_retriever=platform_retriever,
        )
    )
    # The browser voice bridge is wired ALWAYS (not just for the dev console) so
    # the chat→voice handoff (/api/v1/chat/voice) works in prod; the dev console's
    # own WS routes stay behind VOX_DEV_CONSOLE. handoff_store lets a ?handoff token
    # carry the chat summary into the voice agent.
    set_browser_bridge_factory(
        make_browser_bridge_factory(
            providers=providers, campaign_resolver=campaign_resolver,
            handoff_store=base_session_store,
            platform_retriever=platform_retriever,
        )
    )
    if dev_console_enabled():
        set_live_bridge_factory(
            make_live_bridge_factory(
                providers=providers, campaign_resolver=campaign_resolver,
                platform_retriever=platform_retriever,
            )
        )
        log.info("dev console enabled at /dev/voice")
    # Browser softphone recording webhook transcribes + analyzes with the
    # tenant's STT + LLM, so it needs the same per-tenant provider registry.
    telephony_hooks.set_softphone_providers(providers)
    # ChatBot: per-(tenant, session) agent factory + the sessionmaker the WS uses
    # to resolve the tenant from the chat_sessions row and persist messages.
    chat_api.set_chatbot_factory(
        make_chatbot_factory(runtime_registry, sessionmaker,
                             platform_retriever=platform_retriever))
    chat_api.set_chat_sessionmaker(sessionmaker)
    chat_api.set_chat_handoff_store(base_session_store)
    # Knowledge ingest/query resolve the SAME per-tenant retriever the chatbot
    # uses (registry.retrievers), so ingested docs are retrievable in chat.
    knowledge_api.set_retriever_factory(lambda t: runtime_registry.retrievers.get(t))
    knowledge_api.set_platform_retriever(platform_retriever)
    app.state.providers = providers

    reaper_task = asyncio.create_task(_reap_stale_calls_loop())
    kb_seed_task = asyncio.create_task(_seed_global_kb(platform_retriever, sessionmaker))

    try:
        yield
    finally:
        log.info("shutdown")
        reaper_task.cancel()
        kb_seed_task.cancel()
        telephony_hooks.set_bridge_factory(None)
        telephony_hooks.set_exotel_bridge_factory(None)
        telephony_hooks.set_stringee_bridge_factory(None)
        telephony_hooks.set_softphone_providers(None)
        chat_api.set_chatbot_factory(None)
        chat_api.set_chat_sessionmaker(None)
        chat_api.set_chat_handoff_store(None)
        knowledge_api.set_retriever_factory(None)
        knowledge_api.set_platform_retriever(None)
        set_browser_bridge_factory(None)
        set_call_outcome_persister(None)
        set_tenant_event_notifier(None)
        await redis_client.aclose()
        await dispose_engine()
        set_tenant_resolver(None)


app = FastAPI(
    title="vox-agent",
    version="1.0.0",
    description="Vendor-agnostic agentic framework for multilingual VoiceBot + ChatBot",
    lifespan=lifespan,
)

if dev_console_enabled():
    api_router.include_router(dev_ws_router)    # WS  /api/v1/dev/voice

app.include_router(api_router)

if dev_console_enabled():
    app.include_router(dev_router)              # GET /dev/voice


_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@app.get("/console", include_in_schema=False)
async def api_console() -> FileResponse:
    """Tenant browser UI (campaigns, calls, reference) over /api/v1.

    Always available — it bypasses no auth: it just calls the API with the
    tenant bearer the operator pastes in, so the API enforces access as usual.
    """
    return FileResponse(_STATIC_DIR / "api_console.html", media_type="text/html")


@app.get("/admin", include_in_schema=False)
async def admin_console() -> FileResponse:
    """Admin browser UI (register tenants, maintain provider costs) over /api/v1.

    Uses an admin bearer pasted in by the operator — bypasses no auth.
    """
    return FileResponse(_STATIC_DIR / "admin_console.html", media_type="text/html")


@app.get("/admin/tenants", include_in_schema=False)
async def backoffice() -> FileResponse:
    """Admin backoffice: tenant list + per-tenant analytics & billing."""
    return FileResponse(_STATIC_DIR / "backoffice.html", media_type="text/html")


@app.get("/softphone.js", include_in_schema=False)
async def softphone_helper() -> FileResponse:
    """Provider-agnostic browser softphone helper (wraps Twilio + Stringee SDKs).

    The CRM embeds this to dial with one API regardless of the tenant's provider:
    ``Softphone.create(tokenResponse).then(p => p.dial(leadNumber))``.
    """
    return FileResponse(
        _STATIC_DIR / "softphone.js", media_type="application/javascript")


@app.get("/softphone-test", include_in_schema=False)
async def softphone_test_page() -> FileResponse:
    """Single-page test harness: mint a token + place a browser call via softphone.js.

    Test-only (it mints the token in the browser); a real CRM mints server-side.
    """
    return FileResponse(_STATIC_DIR / "softphone_test.html", media_type="text/html")


@app.get("/chat-widget", include_in_schema=False)
async def chat_widget() -> FileResponse:
    """Reference chat UI for demos/testing. Creates a session via
    POST /api/v1/chat/sessions, then connects the returned ws_url. A real CRM
    builds its own UI against the same APIs and mints the session server-side."""
    return FileResponse(_STATIC_DIR / "chat_widget.html", media_type="text/html")


@app.get("/health")
async def health() -> dict:
    """Liveness + dependency probe + per-tenant provider routing."""
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()

    redis_status = "down"
    try:
        if hasattr(app.state, "redis"):
            await app.state.redis.ping()
            redis_status = "ok"
    except Exception as e:  # noqa: BLE001
        log.warning("redis ping failed", extra={"error": str(e)})

    db_status = "down"
    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:  # noqa: BLE001
        log.warning("db probe failed", extra={"error": str(e)})

    tenants_summary = []
    tenants: dict[str, TenantSettings] = getattr(app.state, "tenants", {})
    for slug, t in tenants.items():
        tenants_summary.append({
            "slug": slug,
            "name": t.name,
            "status": t.status,
            "providers": {
                "stt": t.pipeline.stt.provider or settings.pipeline.stt.provider,
                "llm": t.pipeline.llm.provider or settings.pipeline.llm.provider,
                "tts": t.pipeline.tts.provider or settings.pipeline.tts.provider,
                "telephony": t.pipeline.telephony.provider or settings.pipeline.telephony.provider,
                "vector_store": t.pipeline.vector_store.provider or settings.pipeline.vector_store.provider,
            },
        })

    overall = "ok" if redis_status == "ok" and db_status == "ok" else "degraded"
    return {
        "status": overall,
        "version": settings.app.version,
        "platform_defaults": {
            "stt": settings.pipeline.stt.provider,
            "llm": settings.pipeline.llm.provider,
            "tts": settings.pipeline.tts.provider,
            "telephony": settings.pipeline.telephony.provider,
            "vector_store": settings.pipeline.vector_store.provider,
        },
        "tenants": tenants_summary,
        "tenant_count": len(tenants_summary),
        "redis": redis_status,
        "db": db_status,
    }
