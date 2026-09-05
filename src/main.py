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
from typing import Optional

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
    crm_kb as crm_kb_api,
    external_chat as ext_chat_api,
    knowledge as knowledge_api,
    livekit_routes,
    telephony_hooks,
)
from src.api.dev_console import (
    dev_console_enabled,
    dev_page_router,
    dev_router,
    make_browser_bridge_factory,
    make_live_bridge_factory,
    set_browser_bridge_factory,
    set_live_bridge_factory,
)
from src.api.dev_console import (
    ws_router as dev_ws_router,
)
from src.api.bridge_console import page_router as bridge_page_router
from src.api.bridge_console import router as bridge_router
from src.api.call_store import (
    record_outcome,
    set_call_outcome_persister,
    set_tenant_event_notifier,
)
from src.integration.tenant_events import deliver as deliver_tenant_event
from src.integration.tenant_events import resolve_events_webhook_url
from src.auth.db_resolver import DbTenantResolver
from src.auth.middleware import admin_token_labels, set_admin_tokens, set_tenant_resolver
from src.auth.seed import seed_if_empty, seed_provider_costs, sync_telephony_from_yaml
from src.bootstrap import (
    PerCrmRetrieverRegistry,
    build_provider_registry,
    build_runtime_registry,
    make_bridge_factory,
    make_chatbot_factory,
    make_exotel_bridge_factory,
    make_livekit_bridge_factory,
    make_stringee_bridge_factory,
)
from src.config import Settings, get_settings
from src.config_tenant import TenantSettings
from src.dialogue.campaign_resolver import DbCampaignResolver
from src.dialogue.context import SessionStore
from src.models.database import dispose_engine, ensure_schema, get_engine, get_sessionmaker
from src.utils.client_ip import ClientIPMiddleware
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _admin_tokens_from_env() -> list[str]:
    """Comma-separated admin tokens in ``VOX_ADMIN_TOKENS``. Empty if unset."""
    raw = os.environ.get("VOX_ADMIN_TOKENS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _kb_auto_prune_enabled() -> bool:
    """Whether ``_seed_crm_kb``'s reconcile step is allowed to actually delete
    orphaned CrmKBDocument rows + pgvector chunks it finds, vs. just reporting
    them. Defaults OFF: the destructive action stays opt-in until an operator
    has reviewed ``scripts/purge_stale_kb_docs.py --dry-run`` output on their
    own deployment and explicitly enables this. Same on/off string convention
    as VOX_DEV_CONSOLE."""
    return os.environ.get("VOX_KB_AUTO_PRUNE", "") == "1"


def _parse_callback(value):
    """Parse an ISO callback datetime from an outcome payload (None-safe)."""
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


async def _resolve_tenant_event_secret(
    settings,
    secret_env: Optional[str],
    resolver,
    *,
    tenant_id: Optional[str],
    event_type: Optional[str],
) -> Optional[str]:
    """Resolve the outbound tenant-event webhook signing secret.

    Per-tenant DECRYPTED secret first (so it can be stored per-tenant like the
    telephony keys), falling back to the platform-level ``EVENTS_WEBHOOK_SECRET``
    env var. Never raises — a missing secret just means the event is sent
    unsigned, but that's a real security gap for the tenant's CRM (it can't
    verify the event genuinely came from us), so we log a loud warning rather
    than sending silently unsigned.
    """
    secret = None
    if secret_env:
        ctx = None
        if resolver is not None and settings is not None and hasattr(resolver, "resolve_by_slug"):
            ctx = await resolver.resolve_by_slug(settings.slug)
        secret = ctx.secret_optional(secret_env) if ctx else os.environ.get(secret_env)
    # Fall back to platform-level signing key when no per-tenant secret is set.
    if not secret:
        secret = os.environ.get("EVENTS_WEBHOOK_SECRET") or None
    if not secret:
        log.warning(
            "tenant event webhook sending UNSIGNED (no events_webhook_secret_env or "
            "platform EVENTS_WEBHOOK_SECRET configured) — configure a webhook secret; "
            "see docs/integrations/chat-widget-backend-integration.md#4-webhook-events",
            extra={"tenant_id": tenant_id, "event_type": event_type},
        )
    return secret


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


async def _seed_crm_kb(
    crm_retrievers: "PerCrmRetrieverRegistry",
    sessionmaker,
    kb_dir: Path = Path("data/kb/packs/betting-default"),
    auto_prune: Optional[bool] = None,
    bundled_kb_pack: str = "betting-default",
) -> None:
    """Re-ingest a bundled KB pack into every CRM that's opted into it.

    Docs bundled at ``kb_dir`` (default data/kb/packs/betting-default/) are
    seeded ONLY into CRM rows whose ``bundled_kb_pack`` column equals
    ``bundled_kb_pack`` (default "betting-default", matching ``kb_dir``'s
    default) — a CRM with ``bundled_kb_pack`` unset (NULL) or set to a
    different pack name gets none of this pack's docs. This is an explicit
    per-CRM opt-in, not automatic-for-every-CRM. Uses deterministic doc_ids
    (filename-based) so CrmKBDocument DB rows are replaced, not duplicated,
    across restarts.

    Also self-heals: after seeding, any file that used to exist under
    ``kb_dir`` but has since been renamed/deleted/moved to a different tier
    leaves behind an orphaned ``CrmKBDocument`` row (and pgvector chunks)
    that nothing else ever cleans up. See the reconcile step below.

    ``auto_prune`` gates whether the reconcile step below is actually allowed
    to delete what it finds, vs. only reporting it. Defaults (``None``) to
    ``_kb_auto_prune_enabled()`` (i.e. the ``VOX_KB_AUTO_PRUNE`` env var,
    off unless explicitly set) — overridable here so this function stays
    unit-testable without monkeypatching the environment, same pattern as
    ``kb_dir``.

    ``kb_dir`` defaults to the real bundled-docs directory but is overridable
    so this function is unit-testable against a tmp directory. ``bundled_kb_pack``
    likewise defaults to the pack name matching that default directory but is
    overridable for testability (e.g. pairing a tmp ``kb_dir`` with a
    made-up pack name so the real DB/pack names are never touched by tests).
    """
    from sqlalchemy import delete as sa_delete, select

    from src.interfaces.vector_store import Document
    from src.models.crm import Crm, CrmKBDocument
    from src.rag.ingestion import ChunkConfig, detect_language, get_chunker, parse_document

    if auto_prune is None:
        auto_prune = _kb_auto_prune_enabled()

    if not kb_dir.is_dir():
        return
    async with sessionmaker() as session:
        crm_ids = [
            r[0] for r in (
                await session.execute(
                    select(Crm.id).where(Crm.bundled_kb_pack == bundled_kb_pack)
                )
            ).all()
        ]
    if not crm_ids:
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
    # Per-crm bookkeeping for the reconcile step below: the doc_ids we
    # actually (re)wrote for each crm this pass. `failed_stems` tracks which
    # *files* (by stem) raised during parsing/indexing this pass — isolated
    # to that file only, not smeared across every crm/file. A parse failure
    # on one file must never disable pruning for unrelated files or other
    # crms; it only protects that one file's own derived doc ids from being
    # mistaken for "gone" by the reconcile step below.
    expected_ids_by_crm: dict[str, set[str]] = {crm_id: set() for crm_id in crm_ids}
    failed_stems: set[str] = set()
    for f in files:
        try:
            text = parse_document(f.name, f.read_bytes())
            if not text.strip():
                continue
            language = detect_language(text)
            for crm_id in crm_ids:
                retriever = crm_retrievers.get(crm_id)
                if retriever is None:
                    continue
                doc_id = f"crm_kb_{crm_id}_{f.stem}"
                expected_ids_by_crm[crm_id].add(doc_id)
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
                n = await retriever.index(docs)
                total += n
                async with sessionmaker() as session:
                    await session.execute(
                        sa_delete(CrmKBDocument).where(CrmKBDocument.id == doc_id)
                    )
                    session.add(CrmKBDocument(
                        id=doc_id, crm_id=crm_id, filename=f.name,
                        source_type=f.suffix.lstrip(".").lower(),
                        language=language, chunk_count=n,
                        extra_data={"chunk_ids": [d.id for d in docs]},
                    ))
                    await session.commit()
        except Exception:  # noqa: BLE001 - one bad file must not abort the whole seed
            log.exception("crm KB seed failed", extra={"file": f.name})
            # Isolate the damage to this file only: record its stem so the
            # reconcile step below protects *this file's* derived doc ids
            # (across all crms) from being pruned as stale — a transient
            # parse/index error must never be mistaken for "this file no
            # longer exists". Every other file/crm is unaffected and still
            # reconciles normally this pass.
            failed_stems.add(f.stem)
    if total:
        log.info("CRM KB seeded", extra={"files": len(files), "chunks": total, "crms": len(crm_ids)})

    # Reconcile: prune CrmKBDocument rows (and their pgvector chunks) that
    # this seeder itself wrote in a previous pass but that no longer
    # correspond to a file under kb_dir today — e.g. the file was renamed,
    # deleted, or moved to a different tier. Without this, such rows are
    # orphaned forever (this was the root cause of duplicate-looking docs
    # in the KB list).
    for crm_id in crm_ids:
        retriever = crm_retrievers.get(crm_id)
        if retriever is None:
            continue  # can't safely delete pgvector chunks without a retriever
        expected_ids = expected_ids_by_crm[crm_id]
        # Rows tied to a file that failed to parse/index this pass are
        # protected from pruning (in both the current and legacy id
        # namespaces) — a transient failure on that file must not cascade
        # into deleting its previously-seeded doc. Everything else not in
        # expected_ids is genuinely stale and still gets pruned normally,
        # even if some unrelated file failed elsewhere in this same pass.
        protected_ids = {f"crm_kb_{crm_id}_{stem}" for stem in failed_stems} | {
            f"global_kb_{stem}" for stem in failed_stems
        }
        async with sessionmaker() as session:
            rows = (await session.execute(
                select(CrmKBDocument).where(CrmKBDocument.crm_id == crm_id)
            )).scalars().all()
            # Only prune ids in the two namespaces this seeder has ever
            # written under: current (`crm_kb_{crm_id}_*`) and legacy
            # pre-migration (`global_kb_*`). Admin-uploaded docs get
            # `crmdoc_*` ids (see POST /crms/{id}/kb/ingest in
            # src/api/crm_kb.py) and never match either prefix, so they're
            # never touched by this auto-reconcile.
            seeder_prefixes = (f"crm_kb_{crm_id}_", "global_kb_")
            stale = [
                r for r in rows
                if r.id.startswith(seeder_prefixes)
                and r.id not in expected_ids
                and r.id not in protected_ids
            ]
            if not stale:
                continue
            stale_ids = [r.id for r in stale]
            if not auto_prune:
                # Destructive, unattended by default: report what would be
                # pruned but do not touch it. See scripts/purge_stale_kb_docs.py
                # for the reviewable dry-run/--execute path, or set
                # VOX_KB_AUTO_PRUNE=1 to let this reconcile step delete
                # automatically on every future boot.
                log.warning(
                    "Found %d stale CRM KB doc(s) for crm_id=%s but "
                    "VOX_KB_AUTO_PRUNE is not enabled -- not pruning: %s. "
                    "Review with `python scripts/purge_stale_kb_docs.py`, then "
                    "either run it with --execute or set VOX_KB_AUTO_PRUNE=1.",
                    len(stale_ids), crm_id, stale_ids,
                )
                continue
            pruned_ids = []
            for row in stale:
                # Mirror src.api.crm_kb.delete_crm_document's chunk-id
                # resolution exactly, so manual delete and this
                # auto-reconcile behave identically.
                chunk_ids = (row.extra_data or {}).get("chunk_ids") or [
                    f"{row.id}::chunk-{i}" for i in range(row.chunk_count or 0)
                ]
                await retriever.delete(chunk_ids)
                await session.delete(row)
                pruned_ids.append(row.id)
            await session.commit()
            # Safe under concurrent boots: expected_ids is derived purely
            # and deterministically from kb_dir's current contents, so any
            # number of replicas computing it independently agree on the
            # same "stale" set — re-deleting an already-deleted row/chunk
            # is a no-op, so no locking is needed.
            log.warning("Pruned stale CRM KB doc(s) for crm_id=%s: %s", crm_id, pruned_ids)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_logging(settings.app.log_level)
    log.info("startup", extra={"app": settings.app.name, "version": settings.app.version})

    # Eagerly create engine + redis pool so missing config fails on boot, not first request.
    get_engine(settings.database.url)
    # Ensure our schema exists before anything touches a table (no-op on SQLite).
    # Wrapped in a timeout: if the DB is temporarily unavailable during a rolling
    # restart the schema already exists from the last boot, so it's safe to proceed.
    import asyncio as _asyncio
    try:
        await _asyncio.wait_for(ensure_schema(settings.database.url), timeout=20.0)
    except Exception:
        log.warning("ensure_schema skipped (timeout or error); schema assumed current")
    redis_client = redis_async.from_url(settings.redis.url, decode_responses=False)
    app.state.redis = redis_client
    app.state.settings = settings

    # --- Tenants: DB-backed (YAML migrated in on first boot, then DB is authoritative) ---
    sessionmaker = get_sessionmaker()
    seeded = await seed_if_empty(sessionmaker)
    if seeded:
        log.info("seeded tenants from YAML into DB", extra={"count": seeded})
    try:
        await _asyncio.wait_for(sync_telephony_from_yaml(sessionmaker), timeout=10.0)
    except Exception:
        log.warning("sync_telephony_from_yaml skipped (timeout or error)")
    await seed_provider_costs(sessionmaker)
    resolver = DbTenantResolver(sessionmaker)
    await resolver.reload()
    set_tenant_resolver(resolver)
    set_admin_tokens(_admin_tokens_from_env())
    labels = admin_token_labels()
    log.info("admin tokens configured", extra={"count": len(labels), "labels": labels})
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
                turns=payload.get("turns"),
                # Only set for LiveKit (room names collide across tenants; Twilio/
                # Exotel/Stringee Call SIDs are globally unique and never set this
                # key, so tenant_id stays None -> unscoped lookup, unchanged).
                tenant_id=payload.get("tenant_id"),
            )
    set_call_outcome_persister(_persist_call_outcome)

    # Outbound per-tenant event webhook: call_store hands us a ready-built
    # envelope at call start/end; we resolve the tenant's events_webhook_url +
    # secret from TenantSettings (top-level, not under telephony) and POST it
    # signed, fire-and-forget so a slow tenant endpoint never blocks call handling.
    def _tenant_settings_by_id(tenant_id: str):
        for t in (getattr(app.state, "tenants", {}) or {}).values():
            if getattr(t, "id", None) == tenant_id:
                return t
        return None

    async def _notify_tenant_event(envelope: dict) -> None:
        settings = _tenant_settings_by_id(envelope.get("tenant_id"))
        url = await resolve_events_webhook_url(settings, sessionmaker)
        if not url:
            return
        secret_env = getattr(settings, "events_webhook_secret_env", None)
        resolver = getattr(app.state, "tenant_resolver", None)
        secret = await _resolve_tenant_event_secret(
            settings, secret_env, resolver,
            tenant_id=envelope.get("tenant_id"),
            event_type=envelope.get("event_type"),
        )
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
    crm_retrievers = PerCrmRetrieverRegistry(
        global_defaults={"vector_store": settings.pipeline.vector_store.model_dump()},
    )
    # Drop ALL cached per-tenant instances (providers + sub-registries) whenever
    # tenants reload (e.g. a key/config update via the tenant API) so the new
    # config takes effect.
    resolver.on_reload = runtime_registry.evict_all
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
            crm_retrievers=crm_retrievers,
            registry=runtime_registry,
        )
    )
    telephony_hooks.set_exotel_bridge_factory(
        make_exotel_bridge_factory(
            providers=providers, session_store=base_session_store,
            campaign_resolver=campaign_resolver,
            crm_retrievers=crm_retrievers,
            registry=runtime_registry,
        )
    )
    telephony_hooks.set_stringee_bridge_factory(
        make_stringee_bridge_factory(
            providers=providers, campaign_resolver=campaign_resolver,
            crm_retrievers=crm_retrievers,
            registry=runtime_registry,
        )
    )
    # LiveKit room-join (the CRM's SIP trunk fronts LiveKit): our webhook route
    # spawns a per-call runner that builds this factory's bridge. Same providers /
    # session store / campaign resolver / CRM retrievers as Twilio + Exotel.
    livekit_routes.set_livekit_bridge_factory(
        make_livekit_bridge_factory(
            providers=providers, session_store=base_session_store,
            campaign_resolver=campaign_resolver,
            crm_retrievers=crm_retrievers,
            registry=runtime_registry,
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
            crm_retrievers=crm_retrievers,
            registry=runtime_registry,
        )
    )
    if dev_console_enabled():
        set_live_bridge_factory(
            make_live_bridge_factory(
                providers=providers, campaign_resolver=campaign_resolver,
                crm_retrievers=crm_retrievers,
                registry=runtime_registry,
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
                             crm_retrievers=crm_retrievers))
    chat_api.set_chat_sessionmaker(sessionmaker)
    chat_api.set_chat_handoff_store(base_session_store)
    ext_chat_api.set_ext_redis(redis_client)
    if settings.media_storage is not None:
        from src.providers.media.s3 import S3MediaStorage
        ms = settings.media_storage
        chat_api.set_media_store(S3MediaStorage(
            endpoint_url=ms.endpoint_url,
            access_key=ms.access_key,
            secret_key=ms.secret_key,
            bucket=ms.bucket,
            region=ms.region,
        ))
    else:
        from src.providers.media.local import LocalMediaStorage
        log.info("media storage: using local filesystem fallback (/tmp/chat_media)")
        chat_api.set_media_store(LocalMediaStorage())
        # json_ticket_relay vendors need a real, publicly-fetchable signed URL
        # (see src/chatbot/deposit_verification.py); LocalMediaStorage only
        # ever produces a relative, unsigned path, so any tenant on that
        # contract can never successfully submit a verification while the
        # local fallback is in effect. Warn (not a hard failure) so this is
        # caught at boot instead of silently erroring on first submission.
        broken_slugs = [
            slug for slug, t in (getattr(app.state, "tenants", {}) or {}).items()
            if getattr(t.deposit_verification, "contract", None) == "json_ticket_relay"
        ]
        if broken_slugs:
            log.warning(
                "deposit verification: tenant(s) use contract='json_ticket_relay' while "
                "media storage is the local filesystem fallback — this vendor requires a "
                "real public signed URL, which LocalMediaStorage cannot provide; "
                "verification submissions for these tenants will fail",
                extra={"tenant_slugs": broken_slugs},
            )
    # Knowledge ingest/query resolve the SAME per-tenant retriever the chatbot
    # uses (registry.retrievers), so ingested docs are retrievable in chat.
    knowledge_api.set_retriever_factory(lambda t: runtime_registry.retrievers.get(t))
    knowledge_api.set_crm_retrievers(crm_retrievers)
    crm_kb_api.set_crm_retrievers(crm_retrievers)
    app.state.providers = providers

    reaper_task = asyncio.create_task(_reap_stale_calls_loop())
    kb_seed_task = asyncio.create_task(_seed_crm_kb(crm_retrievers, sessionmaker))

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
        livekit_routes.set_livekit_bridge_factory(None)
        chat_api.set_chatbot_factory(None)
        chat_api.set_chat_sessionmaker(None)
        chat_api.set_chat_handoff_store(None)
        chat_api.set_media_store(None)
        ext_chat_api.set_ext_redis(None)
        knowledge_api.set_retriever_factory(None)
        knowledge_api.set_crm_retrievers(None)
        crm_kb_api.set_crm_retrievers(None)
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

# Resolve the client IP once per HTTP request / WS connection and publish it on
# a ContextVar, so every log line emitted while serving that connection carries
# it (see src/utils/client_ip.py and the log filter in src/utils/logging.py).
# Pure-ASGI so it covers websocket scopes, which BaseHTTPMiddleware does not.
app.add_middleware(ClientIPMiddleware)

# Fail closed: the dev/bridge consoles place real billed outbound calls and run
# billed AI voice sessions, so their data/action routes now require a
# per-request admin token. If VOX_DEV_CONSOLE is on but no admin token is
# configured, that gate can never be satisfied — mounting those routes would
# only publish unusable-but-reachable endpoints, so we refuse to mount them and
# say so loudly. The open HTML shells still mount; they carry no data.
#
# NOTE: this reads the env var directly rather than src.auth.middleware's
# _admin_token_labels, because set_admin_tokens() runs in lifespan while
# router mounting happens here at import time — that map is still empty
# at this point.
_dev_console_on = dev_console_enabled()
_dev_console_authed = _dev_console_on and bool(_admin_tokens_from_env())
if _dev_console_on and not _dev_console_authed:
    log.error(
        "VOX_DEV_CONSOLE=1 but VOX_ADMIN_TOKENS is empty — refusing to mount the "
        "admin-gated dev/bridge console routes (place-call, voice WS, reanalyze, …). "
        "Set VOX_ADMIN_TOKENS to enable them."
    )

if _dev_console_authed:
    api_router.include_router(dev_ws_router)    # WS  /api/v1/dev/voice{,-live}

app.include_router(api_router)

if _dev_console_on:
    app.include_router(dev_page_router)         # GET /dev/voice   (open page)
    app.include_router(bridge_page_router)      # GET /dev/bridge  (open page)
if _dev_console_authed:
    app.include_router(dev_router)              # /dev/voices, /dev/place-call, …
    app.include_router(bridge_router)           # /dev/bridge/tenants, /dev/bridge/place-call


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


@app.get("/bo-agent", include_in_schema=False)
async def bo_agent_console() -> FileResponse:
    """BO support-agent console: claim escalated sessions and chat with customers."""
    return FileResponse(_STATIC_DIR / "bo_agent.html", media_type="text/html")


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
