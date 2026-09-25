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
from fastapi.responses import FileResponse, JSONResponse
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
from src.config import Settings, drain_load_diagnostics, get_settings
from src.config_tenant import TenantSettings
from src.dialogue.campaign_resolver import DbCampaignResolver
from src.dialogue.context import SessionStore
from src.models.database import dispose_engine, ensure_schema, get_engine, get_sessionmaker
from src.utils.client_ip import ClientIPMiddleware
from src.utils.logging import configure_logging, debug_event, get_logger
from src.utils.redact import redact_url

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
    except (ValueError, TypeError) as exc:
        # Silent before this line: a malformed callback datetime in a call
        # outcome payload used to just vanish -- the callback was silently
        # never scheduled, with nothing at any level to say why.
        debug_event(
            log, "call_outcome callback_parse failed",
            raw_value=value, exc_type=type(exc).__name__,
        )
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
    # has_secret only -- secret itself is the outbound webhook's HMAC signing
    # key and never appears at any level, per docs/debug-logging.md.
    debug_event(
        log, "tenant_event secret_resolve decision",
        tenant_id=tenant_id, event_type=event_type, secret_env=secret_env,
        has_secret=bool(secret),
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


async def _push_turn_metrics_loop(
    interval_s: float, push_url: Optional[str], push_auth: Optional[str]
) -> None:
    """Periodically aggregate recent TurnMetric rows (per-turn voice-call
    latency, see src.models.turn_metrics) into Prometheus metric families and
    push them to Grafana Cloud (Phase 2 observability), THEN do the same for
    ChatBot's per-turn/per-tool metrics (turn-metrics plan, Phase 3, §5) —
    sequentially, in the same iteration, on the same interval/config
    (METRICS_PUSH_INTERVAL_S is reused, not duplicated: no second background
    task, no new config knob). Each push is independently a clean no-op when
    push_url is unset — see aggregate_and_push_turn_metrics /
    aggregate_and_push_chat_metrics. Runs once at startup, then every
    interval_s."""
    from src.observability.chat_metrics_push import aggregate_and_push_chat_metrics
    from src.observability.turn_metrics_push import aggregate_and_push_turn_metrics

    sm = get_sessionmaker()
    while True:
        try:
            n = await aggregate_and_push_turn_metrics(
                sm, push_url, push_auth, window_s=interval_s * 2,
            )
            if n:
                log.info("pushed turn metrics", extra={"count": n})
        except Exception:  # noqa: BLE001 - the push loop must never die (CancelledError still propagates)
            log.exception("turn-metrics push failed")
        try:
            n_chat = await aggregate_and_push_chat_metrics(
                sm, push_url, push_auth, window_s=interval_s * 2,
            )
            if n_chat:
                log.info("pushed chat turn metrics", extra={"count": n_chat})
        except Exception:  # noqa: BLE001 - the push loop must never die (CancelledError still propagates)
            log.exception("chat-metrics push failed")
        await asyncio.sleep(interval_s)


# How often the chat-turn-metrics retention prune runs. Not itself
# configurable (unlike the retention WINDOW -- see
# Secrets.CHAT_METRICS_RETENTION_DAYS): this only needs to keep pace with
# ~4.7k parent rows/month (turn-metrics plan §11.2), so a fixed interval is
# enough. Mirrors _REAP_INTERVAL_S's hardcoded-constant convention above.
_CHAT_METRICS_PRUNE_INTERVAL_S = 6 * 3600

# Per-DELETE row cap for the prune loop below. A single unbounded DELETE over
# a large, continuously-written table can hold a long-lived lock and bloat
# the table/WAL -- deleting oldest-first in bounded batches avoids both.
_CHAT_METRICS_PRUNE_BATCH_SIZE = 1000


async def prune_chat_turn_metrics(sessionmaker, retention_days: float) -> int:
    """Delete ``chat_turn_metrics`` rows older than ``retention_days``,
    returning the number of parent rows deleted (turn-metrics plan, Phase 3,
    §11.2).

    Chat runs ~90x voice's ``TurnMetric`` volume (~4.7k parent rows/month +
    ~12-15k child rows/month, vs. voice's 747 LIFETIME rows) -- unlike voice,
    which is deliberately left unbounded because it will never grow enough to
    matter, unbounded growth here is not acceptable, hence this prune job.

    ``chat_tool_metrics`` rows are FK ``ondelete="CASCADE"`` to
    ``chat_turn_metrics.id`` (see ``src/models/chat_turn_metrics.py``), so
    deleting only the parent here is sufficient -- the DB removes matching
    children itself; no separate child-table delete is needed. (On Postgres
    this CASCADE is always enforced; SQLite -- test-only -- only enforces it
    when ``PRAGMA foreign_keys=ON`` has been set on the connection, same as
    ``tests/unit/test_chat_turn_metrics_model.py`` already does for its own
    CASCADE assertion.)

    Deletes oldest-first in bounded batches (``_CHAT_METRICS_PRUNE_BATCH_SIZE``
    rows per statement) rather than one unbounded ``DELETE`` — a single huge
    delete against a large, continuously-written table can hold a long-lived
    lock and bloat the table/WAL; batching avoids both.

    Pure logic, deliberately separate from ``_prune_chat_turn_metrics_loop``
    below (mirrors ``src/api/call_store.py::reap_stale_calls`` vs.
    ``_reap_stale_calls_loop``'s split) so this is directly unit-testable
    without fighting an infinite loop.
    """
    from sqlalchemy import delete, func, select

    from src.models.chat_turn_metrics import ChatTurnMetric

    # Cutoff evaluated against the DB SERVER's own clock (not Python's
    # datetime.utcnow()) on Postgres -- matches turn_metrics_push.py's
    # identical reasoning: created_at is DateTime(timezone=False) populated
    # by the server's own func.now(), so both sides of the `<` comparison
    # below must be evaluated by that same clock, or a server clock in a
    # non-UTC timezone would silently make `retention_days` mean something
    # else. Built as a reusable SQL expression (not evaluated to a Python
    # value here) so the SAME cutoff applies across every batch iteration's
    # own session below, evaluated fresh by the DB each time -- exactly
    # turn_metrics_push.py's pattern, never a pre-fetched Python datetime
    # compared against a naive column (which risks an aware/naive mismatch).
    # SQLite (test-only) has no separate server clock to disagree with
    # Python's, so it gets a plain Python cutoff instead.
    async with sessionmaker() as probe:
        dialect_name = probe.get_bind().dialect.name
    if dialect_name == "sqlite":
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
    else:
        cutoff = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, retention_days * 86400.0)
    debug_event(
        log, "chat_metrics_prune cutoff decision",
        dialect=dialect_name, retention_days=retention_days,
        cutoff_source="python_utcnow" if dialect_name == "sqlite" else "db_server_clock",
    )

    total_deleted = 0
    while True:
        async with sessionmaker() as s:
            ids = (await s.execute(
                select(ChatTurnMetric.id)
                .where(ChatTurnMetric.created_at < cutoff)
                .order_by(ChatTurnMetric.created_at)
                .limit(_CHAT_METRICS_PRUNE_BATCH_SIZE)
            )).scalars().all()
            if not ids:
                break
            await s.execute(delete(ChatTurnMetric).where(ChatTurnMetric.id.in_(ids)))
            await s.commit()
        total_deleted += len(ids)
        if len(ids) < _CHAT_METRICS_PRUNE_BATCH_SIZE:
            break  # last (partial) batch -- nothing older left to prune this pass
    return total_deleted


async def _prune_chat_turn_metrics_loop(retention_days: float) -> None:
    """Periodically run ``prune_chat_turn_metrics`` (turn-metrics plan, Phase
    3, §11.2). Best-effort like every other background loop here: never dies,
    one try/except per iteration. Runs once at startup, then every
    ``_CHAT_METRICS_PRUNE_INTERVAL_S``."""
    sm = get_sessionmaker()
    while True:
        try:
            n = await prune_chat_turn_metrics(sm, retention_days)
            if n:
                log.info("pruned old chat turn metrics", extra={"count": n})
        except Exception:  # noqa: BLE001 - the prune loop must never die (CancelledError still propagates)
            log.exception("chat-turn-metrics prune failed")
        await asyncio.sleep(_CHAT_METRICS_PRUNE_INTERVAL_S)


# How often the webhook-outbox retry loop wakes up to claim due rows. Kept
# short (unlike the multi-hour metrics-prune interval) because the whole
# point of this queue is to recover a session_closed delivery that missed its
# original ~16s in-line budget as soon as reasonably possible, not hours
# later.
_WEBHOOK_OUTBOX_INTERVAL_S = 60

# How many pending, due rows one pass claims at most. Small on purpose: this
# queue only ever holds session_closed deliveries that already exhausted a
# 3-attempt in-line budget -- an expected-rare, not bulk, workload. Also
# bounds one pass's worst-case wall time against the lease below: at up to
# _WEBHOOK_OUTBOX_PER_ROW_TIMEOUT_S per row, 10 rows is at most 450s --
# comfortably inside _WEBHOOK_OUTBOX_LEASE_S with margin (see N1 below).
_WEBHOOK_OUTBOX_CLAIM_BATCH = 10

# Per-row delivery bound (N1): each row's _deliver_webhook_outbox_claim call
# is wrapped in asyncio.wait_for(..., this). Without it, a single row that
# hangs past httpx's own 5s-per-attempt timeout (e.g. stuck in tenant/URL
# resolution, a wedged DB call) has no upper bound at all, and
# _WEBHOOK_OUTBOX_CLAIM_BATCH rows' worth of such hangs could push one pass
# past the lease window, at which point a DIFFERENT pod could re-claim and
# double-send a row this pod is still (very slowly) working on.
#
# 10 rows (_WEBHOOK_OUTBOX_CLAIM_BATCH) x 45s = 450s < 600s lease
# (_WEBHOOK_OUTBOX_LEASE_S below), a 150s / 25% margin. 45s itself is
# generous headroom over the ~16.8s a normal delivery attempt can already
# take (3 httpx attempts x 5s timeout + backoff) plus tenant/URL resolution.
# On timeout, the row is treated as retryable (see _finalize_webhook_outbox_row)
# -- never dead, since a timeout says nothing about whether the request
# itself was bad.
_WEBHOOK_OUTBOX_PER_ROW_TIMEOUT_S = 45

# How long a claimed row is leased for (see _claim_webhook_outbox_rows):
# next_attempt_at is pushed this far into the future the moment a row is
# claimed, committed immediately, BEFORE any delivery attempt. That's what
# makes a crash/cancelled pass (every rolling deploy cancels in-flight
# background tasks -- see src/main.py's shutdown block) cheap: the row is
# just not due again until the lease expires, at worst delaying it by one
# lease window -- never a re-send of a row this same pass already finished
# and committed as delivered (see run_webhook_outbox_once's docstring, H1/H2).
# Must exceed a full batch's worst-case wall time (batch x per-row timeout,
# see the two constants above) with margin.
_WEBHOOK_OUTBOX_LEASE_S = 600  # 10 minutes

# Backoff schedule (seconds) indexed by `attempts` (1-based) after a retry
# fails: 1m, 2m, 5m, 15m, 30m, then hourly for every attempt beyond that --
# matches the spec's "1m, 2m, 5m, 15m, 30m, then hourly" exactly. Used for
# EVERY reschedule -- a retryable failure and a permanent failure that
# hasn't yet hit the 3-strike cap (see _finalize_webhook_outbox_row) both
# back off on this same schedule.
_WEBHOOK_OUTBOX_BACKOFF_S = [60, 120, 300, 900, 1800, 3600]

# A row still PENDING this long after it was first enqueued is retired
# (`dead`, last_error="max_age") directly at claim time (N3 -- see
# _claim_webhook_outbox_rows) rather than being claimed for yet another
# delivery attempt. Age-based (not an attempts-count cap): the backoff
# schedule above settles at hourly, so ~24h since creation is also roughly
# "24 more attempts after settling" -- but age is what's actually being
# bounded (how long a CRM ticket can plausibly stay incorrectly open), so
# that's what's measured directly. Distinct from the permanent-failure
# 3-strike rule below, which can retire a row much sooner than 24h.
_WEBHOOK_OUTBOX_MAX_AGE_S = 24 * 3600

# A row is given up as dead on a PERMANENT (non-retryable 4xx) delivery
# outcome only once it has accumulated at least this many delivery attempts
# AND the latest one was itself permanent (see _finalize_webhook_outbox_row).
# Simplest-correct rule of the two considered ("N consecutive permanent
# failures" needs a second counter this table doesn't have; "attempts >= N
# with the latest outcome permanent" reuses the `attempts` column already
# there) -- chosen specifically so a transient 403/404 during a brief CRM
# deploy (permanent on attempt 1, succeeds or times out normally afterward)
# never kills a row: it only takes hold after the row has already been
# rescheduled with backoff twice (~1m + ~2m elapsed, see the backoff
# schedule above) and is STILL seeing a permanent failure on its 3rd try.
_WEBHOOK_OUTBOX_PERMANENT_DEAD_THRESHOLD = 3

# How long a `delivered`/`dead` row is kept before being pruned (L3). These
# terminal rows carry the full session_closed body -- transcript text and the
# tenant's webhook URL -- and are no longer actionable once terminal, so
# unlike the retry bookkeeping itself they must not accumulate indefinitely.
_WEBHOOK_OUTBOX_PRUNE_RETENTION_DAYS = 7
_WEBHOOK_OUTBOX_PRUNE_BATCH_SIZE = 500


def _webhook_outbox_backoff_s(attempts: int) -> int:
    """Backoff delay for the Nth failure (`attempts` is 1-based: the delay to
    apply AFTER attempt number `attempts` has just failed). Clamped to the
    schedule's last entry (hourly) once `attempts` runs past it."""
    idx = min(max(attempts, 1), len(_WEBHOOK_OUTBOX_BACKOFF_S)) - 1
    return _WEBHOOK_OUTBOX_BACKOFF_S[idx]


async def _claim_webhook_outbox_rows(sessionmaker, *, now) -> list[dict]:
    """Claim up to ``_WEBHOOK_OUTBOX_CLAIM_BATCH`` due ``pending`` rows in one
    SHORT transaction (H2): ``SELECT ... FOR UPDATE SKIP LOCKED`` (Postgres
    only -- SQLite, tests-only and always single-process, has no such clause
    and needs none). For each due row:

    - N3: if it's older than ``_WEBHOOK_OUTBOX_MAX_AGE_S`` since it was first
      enqueued, retire it directly HERE -- ``status=dead``,
      ``last_error="max_age"`` -- rather than claiming it for a delivery
      attempt that would only be thrown away. This also means an over-age
      row never occupies a claim-batch slot that could go to a still-viable
      one.
    - Otherwise, bump ``attempts`` and push ``next_attempt_at`` out by the
      lease window.

    Either way, commits immediately -- BEFORE any HTTP delivery is
    attempted. That's the fix for an earlier design, which held one
    ``FOR UPDATE SKIP LOCKED`` transaction open across an entire batch's
    worth of httpx calls (up to ~7 minutes idle-in-transaction for a 25-row
    batch) -- a real risk under ``idle_in_transaction_session_timeout`` or a
    PgBouncer transaction-pooling proxy (this project's own
    ``src/models/database.py`` already sets ``statement_cache_size=0``
    specifically for PgBouncer compatibility).

    Returns plain dicts (rows actually claimed for delivery only -- retired
    over-age rows are NOT included), not live ORM rows bound to the now-
    closed session that read them: ``id``, ``tenant_id``, ``session_id``,
    ``event_type``, ``url``, ``body``, ``attempts`` (POST-bump -- the count
    this delivery attempt about to run IS attempt number ``attempts``),
    ``created_at``.
    """
    from datetime import timedelta

    from sqlalchemy import select

    from src.models.webhook_outbox import STATUS_DEAD, STATUS_PENDING, WebhookOutbox

    async with sessionmaker() as db:
        dialect_name = db.get_bind().dialect.name
        stmt = (
            select(WebhookOutbox)
            .where(WebhookOutbox.status == STATUS_PENDING)
            .where(WebhookOutbox.next_attempt_at <= now)
            .order_by(WebhookOutbox.next_attempt_at)
            .limit(_WEBHOOK_OUTBOX_CLAIM_BATCH)
        )
        if dialect_name != "sqlite":
            stmt = stmt.with_for_update(skip_locked=True)
        rows = (await db.execute(stmt)).scalars().all()
        claimed: list[dict] = []
        for row in rows:
            age_s = (now - row.created_at).total_seconds() if row.created_at else 0.0
            if age_s >= _WEBHOOK_OUTBOX_MAX_AGE_S:
                row.status = STATUS_DEAD
                row.last_error = "max_age"
                log.warning(
                    "webhook outbox row given up as dead (max age reached)",
                    extra={
                        "id": row.id, "tenant_id": row.tenant_id, "session_id": row.session_id,
                        "event_type": row.event_type, "attempts": row.attempts,
                    },
                )
                continue
            row.attempts += 1
            row.next_attempt_at = now + timedelta(seconds=_WEBHOOK_OUTBOX_LEASE_S)
            claimed.append({
                "id": row.id, "tenant_id": row.tenant_id, "session_id": row.session_id,
                "event_type": row.event_type, "url": row.url, "body": row.body,
                "attempts": row.attempts, "created_at": row.created_at,
            })
        await db.commit()
    return claimed


async def _deliver_webhook_outbox_claim(claim: dict, *, resolve_tenant, deliver_fn) -> tuple[str, Optional[str]]:
    """Attempt one delivery for an already-claimed, already-leased row (a
    dict from ``_claim_webhook_outbox_rows``). Runs entirely OUTSIDE any DB
    transaction -- only network I/O plus, at most, a read-only tenant/URL
    resolution (which opens and closes its own short-lived session).

    Returns ``(outcome, error)`` where ``outcome`` is one of ``"delivered"``,
    ``"permanent"`` (a non-retryable 4xx -- see
    ``src.integration.tenant_events.DeliveryResult.permanent``, M2), or
    ``"retryable"`` (anything else: 5xx, timeout/transport failure, a
    retryable 4xx, tenant/URL resolution failure).

    Re-resolves the tenant's CURRENT webhook URL + signing secret rather than
    trusting the claim's own stored ``url`` (captured at enqueue time) or
    re-using any signature: either can legitimately change between enqueue
    and eventual delivery (a tenant rotating its secret, or moving/unsetting
    its CRM webhook entirely). When re-resolution comes back with no URL at
    all, this deliberately does NOT fall back to the claim's stored ``url``
    -- that stored value could be stale/revoked from the tenant's own point
    of view by now, and silently POSTing a customer transcript to a URL the
    tenant no longer authorizes is a real data-exposure risk this loop must
    not take. It's treated as retryable instead, in case this was only a
    transient resolution hiccup (e.g. a DB blip).

    Can itself raise (a resolver bug, an unexpected exception from
    ``deliver_fn``) -- ``_finalize_webhook_outbox_row``'s own try/except is
    the outermost safety net for that (H1); this function does not catch
    broadly on purpose, so a genuine bug surfaces in that one row's
    ``last_error`` rather than being silently absorbed here too.
    """
    import os

    from src.integration.tenant_events import resolve_events_webhook_url
    from src.models.database import get_sessionmaker

    tenant = await resolve_tenant(claim["tenant_id"])
    if tenant is None:
        return "retryable", "tenant_unresolved"

    settings = getattr(tenant, "settings", tenant)
    url = await resolve_events_webhook_url(tenant, get_sessionmaker())
    if not url:
        return "retryable", "no_webhook_url_configured"

    secret_env = getattr(settings, "events_webhook_secret_env", None)
    secret = (
        tenant.secret_optional(secret_env)
        if secret_env and hasattr(tenant, "secret_optional") else None
    )
    if not secret:
        secret = os.environ.get("EVENTS_WEBHOOK_SECRET") or None

    result = await deliver_fn(url, claim["body"], secret)
    if result.ok:
        return "delivered", None
    if getattr(result, "permanent", False):
        return "permanent", f"http_{result.final_status}"
    if result.final_status == -1:
        return "retryable", "delivery_failed"
    return "retryable", f"http_{result.final_status}"


async def _finalize_webhook_outbox_row(
    sessionmaker, claim: dict, *, resolve_tenant, deliver_fn, now_fn,
) -> str:
    """Deliver one claimed row (bounded, N1) and apply the result via a
    single atomic ``UPDATE ... WHERE id=:id AND status='pending' AND
    attempts=:claimed_attempts`` (H2 + N1 fencing).

    N1 fencing: the ``attempts=:claimed_attempts`` clause means this UPDATE
    only applies if the row's ``attempts`` is STILL exactly what this claim
    bumped it to -- i.e. nothing else (another pod re-claiming after this
    pass overran its lease, a retried finalize after a prior timeout) has
    already claimed or finalized this row again in the meantime. Without it,
    a slow/delayed finalize could clobber a NEWER claim's state with a
    STALE result.

    N1 bound: the delivery attempt itself is wrapped in
    ``asyncio.wait_for(..., _WEBHOOK_OUTBOX_PER_ROW_TIMEOUT_S)`` so one
    hung row can't let a whole pass run long enough to outlive the lease
    (see that constant's own comment for the batch x per-row-timeout <
    lease arithmetic). A timeout is treated as retryable, same as any other
    exception (H1) -- it says nothing about whether the request itself was
    bad, only that it took too long.

    H1: wraps delivery (now including the wait_for above) in try/except so
    ANY exception (a resolver bug, a custom ``deliver_fn`` raising, a
    timeout) reschedules this one row rather than propagating out of
    ``run_webhook_outbox_once`` and aborting the rest of the batch -- with
    an earlier single-batch-commit design, one bad row used to roll back
    every row already marked delivered earlier in the same pass.

    N4: ``now`` is read fresh here (via ``now_fn()``), AFTER delivery
    completes -- not the timestamp ``_claim_webhook_outbox_rows`` used to
    decide which rows were due. Delivery can itself take up to
    ``_WEBHOOK_OUTBOX_PER_ROW_TIMEOUT_S``, so reusing a pre-delivery
    timestamp for ``delivered_at`` / the next backoff's ``next_attempt_at``
    would silently understate how much time has actually passed.

    Item 3 (permanent-4xx 3-strike rule): a ``"permanent"`` outcome only
    marks the row dead once ``claim["attempts"] >=
    _WEBHOOK_OUTBOX_PERMANENT_DEAD_THRESHOLD`` (3) -- otherwise it's
    rescheduled with the SAME backoff schedule a retryable failure gets,
    ``last_error`` still recording the permanent status (e.g. ``"http_401"``)
    so it's visible either way. This is what keeps a transient 403/404
    during a brief CRM deploy from killing a row on its first try.

    Returns one of ``"delivered"``, ``"dead"`` (permanent x3, or an
    exception/timeout on an already-old-enough row is NOT how a row goes
    dead here -- see N3, that's now handled entirely at claim time),
    ``"rescheduled"``, or ``"skipped"`` (the N1 fence didn't match -- someone
    else already finalized or re-claimed this row) -- for the caller's own
    logging.
    """
    from datetime import timedelta

    from sqlalchemy import update as sa_update

    from src.models.webhook_outbox import STATUS_DEAD, STATUS_DELIVERED, STATUS_PENDING, WebhookOutbox

    try:
        outcome, error = await asyncio.wait_for(
            _deliver_webhook_outbox_claim(claim, resolve_tenant=resolve_tenant, deliver_fn=deliver_fn),
            timeout=_WEBHOOK_OUTBOX_PER_ROW_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001 - H1/N1: one bad or slow row must never abort the batch
        outcome, error = "retryable", type(e).__name__

    now = now_fn()  # N4: fresh clock read, taken AFTER delivery completed

    # N1 fencing on every branch below: only apply if attempts is still
    # exactly what THIS claim bumped it to.
    fence = (
        WebhookOutbox.id == claim["id"],
        WebhookOutbox.status == STATUS_PENDING,
        WebhookOutbox.attempts == claim["attempts"],
    )

    if outcome == "delivered":
        stmt = (
            sa_update(WebhookOutbox).where(*fence)
            .values(status=STATUS_DELIVERED, delivered_at=now, last_error=None)
        )
        result_label = "delivered"
    elif outcome == "permanent" and claim["attempts"] >= _WEBHOOK_OUTBOX_PERMANENT_DEAD_THRESHOLD:
        stmt = sa_update(WebhookOutbox).where(*fence).values(status=STATUS_DEAD, last_error=error)
        result_label = "dead"
        log.warning(
            "webhook outbox row given up as dead (permanent failure, 3rd+ attempt)",
            extra={
                "id": claim["id"], "tenant_id": claim["tenant_id"], "session_id": claim["session_id"],
                "event_type": claim["event_type"], "attempts": claim["attempts"], "last_error": error,
            },
        )
    else:  # "retryable", or "permanent" but not yet at the 3-strike threshold
        next_at = now + timedelta(seconds=_webhook_outbox_backoff_s(claim["attempts"]))
        stmt = (
            sa_update(WebhookOutbox).where(*fence)
            .values(next_attempt_at=next_at, last_error=error)
        )
        result_label = "rescheduled"

    async with sessionmaker() as db:
        result = await db.execute(stmt)
        await db.commit()
        applied = (result.rowcount or 0) > 0

    if result_label == "delivered" and applied:
        log.info(
            "webhook outbox row delivered",
            extra={
                "id": claim["id"], "tenant_id": claim["tenant_id"], "session_id": claim["session_id"],
                "event_type": claim["event_type"], "attempts": claim["attempts"],
            },
        )
    return result_label if applied else "skipped"


async def _default_webhook_outbox_deliver_fn(url: str, body: dict, secret: Optional[str]):
    """The outbox's default ``deliver_fn``: ``deliver_detailed`` with
    ``stop_on_permanent=True`` (N6) -- opt-in fast-fail-within-one-attempt
    behavior used ONLY here, never by ``deliver()``/the in-line
    ``session_closed`` path (see ``deliver_detailed``'s own docstring)."""
    from src.integration.tenant_events import deliver_detailed
    return await deliver_detailed(url, body, secret, stop_on_permanent=True)


def _webhook_outbox_now():
    """The real, naive-UTC clock reading used by default wherever this
    module needs "now" for the outbox (N4: called fresh at finalize time,
    not reused from claim time) -- overridable via ``now``/``now_fn``
    parameters for tests."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def run_webhook_outbox_once(
    sessionmaker, *, resolve_tenant=None, deliver_fn=None, now=None, now_fn=None,
) -> int:
    """One pass of the webhook-outbox retry loop: claim due ``pending`` rows
    (``_claim_webhook_outbox_rows``, a short transaction, evaluated against
    ``now``), then deliver + finalize each OUTSIDE any transaction, one row
    at a time (``_finalize_webhook_outbox_row``, N4: each finalize reads its
    OWN fresh "now" via ``now_fn``, not this function's ``now``). Returns the
    number of rows claimed (and thus processed) this pass -- a row retired
    directly at claim time for being over-age (N3) is NOT included, since no
    delivery was attempted for it.

    Pure logic, deliberately separate from ``_webhook_outbox_loop`` below
    (mirrors ``prune_chat_turn_metrics`` vs. ``_prune_chat_turn_metrics_loop``,
    and ``reap_stale_calls`` vs. ``_reap_stale_calls_loop``) so this is
    directly unit-testable without fighting an infinite loop.

    Multi-pod safety (H2 + N1): claiming uses ``SELECT ... FOR UPDATE SKIP
    LOCKED`` on Postgres, but that transaction is held only long enough to
    retire over-age rows / bump ``attempts``+``next_attempt_at`` and commit
    -- never across the actual HTTP delivery (which is itself bounded by
    ``_WEBHOOK_OUTBOX_PER_ROW_TIMEOUT_S`` per row, N1, so a whole pass can't
    run long enough to outlive the lease). The lease (``next_attempt_at``
    pushed ``_WEBHOOK_OUTBOX_LEASE_S`` into the future at claim time) is what
    keeps two pods from double-sending the same row even after that short
    transaction's lock is released. Each row's eventual outcome is then
    applied via one atomic, FENCED (N1: ``attempts=:claimed_attempts``, not
    just ``id``+``status``) ``UPDATE`` (see ``_finalize_webhook_outbox_row``),
    so a row another pod already re-claimed/finalized in the meantime is a
    safe no-op here rather than clobbering newer state with a stale result.

    ``resolve_tenant`` defaults to ``src.auth.middleware.tenant_from_id``;
    ``deliver_fn`` defaults to ``_default_webhook_outbox_deliver_fn`` (N6:
    ``deliver_detailed`` with ``stop_on_permanent=True`` -- opt-in, only
    here); ``now_fn`` defaults to ``_webhook_outbox_now`` -- all three
    overridable so tests never need a real tenant resolver, network call, or
    wall clock.
    """
    if resolve_tenant is None:
        from src.auth.middleware import tenant_from_id
        resolve_tenant = tenant_from_id
    if deliver_fn is None:
        deliver_fn = _default_webhook_outbox_deliver_fn
    if now_fn is None:
        now_fn = _webhook_outbox_now

    claim_now = now if now is not None else now_fn()

    claims = await _claim_webhook_outbox_rows(sessionmaker, now=claim_now)
    for claim in claims:
        # H1: _finalize_webhook_outbox_row's own try/except means a bad row
        # here can never raise out of this loop and abort rows not yet
        # processed, nor undo rows already finalized earlier in this same
        # pass (each is its own committed transaction, not a shared one).
        await _finalize_webhook_outbox_row(
            sessionmaker, claim, resolve_tenant=resolve_tenant, deliver_fn=deliver_fn, now_fn=now_fn,
        )
    return len(claims)


async def prune_webhook_outbox(
    sessionmaker, retention_days: float = _WEBHOOK_OUTBOX_PRUNE_RETENTION_DAYS,
) -> int:
    """Delete ``webhook_outbox`` rows in a terminal state (``delivered`` /
    ``dead``) older than ``retention_days`` (L3). Terminal rows carry the
    full session_closed body -- transcript text and the tenant's webhook URL
    -- and are no longer actionable once terminal, so unlike the retry
    bookkeeping itself they must not accumulate indefinitely. Rows still
    ``pending`` are never touched here regardless of age -- the 24h dead-cap
    in ``run_webhook_outbox_once``/``_finalize_webhook_outbox_row`` is what
    retires those.

    Same DB-clock-vs-Python-clock + batched-delete approach as
    ``prune_chat_turn_metrics`` above -- see that function's own comments for
    the full reasoning (SQLite gets a Python cutoff since it has no separate
    server clock; Postgres evaluates the cutoff fresh each batch against the
    DB server's own clock).
    """
    from sqlalchemy import delete, func, select

    from src.models.webhook_outbox import STATUS_DEAD, STATUS_DELIVERED, WebhookOutbox

    async with sessionmaker() as probe:
        dialect_name = probe.get_bind().dialect.name
    if dialect_name == "sqlite":
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
    else:
        cutoff = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, retention_days * 86400.0)

    total_deleted = 0
    while True:
        async with sessionmaker() as s:
            ids = (await s.execute(
                select(WebhookOutbox.id)
                .where(WebhookOutbox.status.in_([STATUS_DELIVERED, STATUS_DEAD]))
                .where(WebhookOutbox.created_at < cutoff)
                .order_by(WebhookOutbox.created_at)
                .limit(_WEBHOOK_OUTBOX_PRUNE_BATCH_SIZE)
            )).scalars().all()
            if not ids:
                break
            await s.execute(delete(WebhookOutbox).where(WebhookOutbox.id.in_(ids)))
            await s.commit()
        total_deleted += len(ids)
        if len(ids) < _WEBHOOK_OUTBOX_PRUNE_BATCH_SIZE:
            break  # last (partial) batch -- nothing older left to prune this pass
    return total_deleted


def webhook_outbox_enabled() -> bool:
    """``WEBHOOK_OUTBOX_ENABLED`` kill switch for the retry loop, read on every
    pass so flipping it needs no restart. Unset means on; 0/false/no/off
    pauses retries (e.g. if the CRM objects to repeated session_closed)."""
    raw = os.environ.get("WEBHOOK_OUTBOX_ENABLED", "")
    return raw.strip().lower() not in ("0", "false", "no", "off")


async def _webhook_outbox_loop() -> None:
    """Periodically retry ``session_closed`` webhook deliveries that
    exhausted their in-line budget (see ``src/models/webhook_outbox.py``),
    then prune old terminal rows (L3). Runs once at startup, then every
    ``_WEBHOOK_OUTBOX_INTERVAL_S``. Never dies, same one-try/except-per-
    iteration convention as every other background loop in this module --
    two separate try/excepts so a prune failure never skips next pass's
    retry attempt (and vice versa)."""
    sm = get_sessionmaker()
    paused_logged = False
    while True:
        if not webhook_outbox_enabled():
            # Kill switch: retries stop, rows stay pending (and still age out
            # via max_age), the in-line first attempt is unaffected.
            if not paused_logged:
                log.warning("webhook outbox retries paused (WEBHOOK_OUTBOX_ENABLED is off)")
                paused_logged = True
        else:
            paused_logged = False
            try:
                n = await run_webhook_outbox_once(sm)
                if n:
                    log.info("webhook outbox pass processed rows", extra={"count": n})
            except Exception:  # noqa: BLE001 - the outbox loop must never die (CancelledError still propagates)
                log.exception("webhook outbox pass failed")
        try:
            n_pruned = await prune_webhook_outbox(sm)
            if n_pruned:
                log.info("pruned old webhook outbox rows", extra={"count": n_pruned})
        except Exception:  # noqa: BLE001 - same never-dies convention
            log.exception("webhook outbox prune failed")
        await asyncio.sleep(_WEBHOOK_OUTBOX_INTERVAL_S)


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

    auto_prune_explicit = auto_prune is not None
    if auto_prune is None:
        auto_prune = _kb_auto_prune_enabled()
    debug_event(
        log, "startup kb_seed auto_prune decision",
        auto_prune=auto_prune,
        source="explicit_arg" if auto_prune_explicit else "env_VOX_KB_AUTO_PRUNE",
    )

    if not kb_dir.is_dir():
        # Whole reseed is a no-op with nothing at any level otherwise --
        # indistinguishable from "no bundled KB pack shipped" from outside.
        debug_event(
            log, "startup kb_seed skipped",
            reason="kb_dir_missing", kb_dir=str(kb_dir), bundled_kb_pack=bundled_kb_pack,
        )
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
        debug_event(
            log, "startup kb_seed skipped",
            reason="no_crm_opted_in", bundled_kb_pack=bundled_kb_pack, kb_dir=str(kb_dir),
        )
        return
    exts = {".md", ".txt", ".pdf", ".docx", ".csv"}
    files = sorted(
        p for p in kb_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
    )
    if not files:
        debug_event(
            log, "startup kb_seed skipped",
            reason="no_files_found", kb_dir=str(kb_dir), extensions=sorted(exts),
        )
        return
    debug_event(
        log, "startup kb_seed scope resolved",
        file_count=len(files), crm_count=len(crm_ids), bundled_kb_pack=bundled_kb_pack,
    )

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
    else:
        # Files and opted-in CRMs both existed (the two early returns above
        # didn't fire) but nothing was indexed -- e.g. every file parsed to
        # empty text, or chunking produced no chunks. log.info above never
        # fires in this case, so this was previously silent.
        debug_event(
            log, "startup kb_seed indexed_zero",
            file_count=len(files), crm_count=len(crm_ids), failed_files=sorted(failed_stems),
        )

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
    configure_logging(
        settings.app.log_level,
        loki_url=settings.secrets.GRAFANA_LOKI_PUSH_URL,
        loki_auth=settings.secrets.GRAFANA_LOKI_PUSH_AUTH,
        service_name=settings.app.name,
    )
    log.info("startup", extra={"app": settings.app.name, "version": settings.app.version})

    # get_settings() above had to run FIRST -- it is where the log level comes
    # from -- so everything load_settings() decided was decided while the root
    # logger was still unconfigured, and its own debug_event calls went nowhere.
    # This is the first moment they can be seen. Without it the unknown-key
    # sweep would be silent on every production boot, which is the one place it
    # is worth having: config/default.yaml's `tts.model` has been inert for
    # months precisely because nothing reported it.
    if (load_diagnostics := drain_load_diagnostics()):
        debug_event(
            log, "config load_diagnostics replayed", diagnostics=load_diagnostics,
        )

    # Eagerly create engine + redis pool so missing config fails on boot, not first request.
    get_engine(settings.database.url)
    # Ensure our schema exists before anything touches a table (no-op on SQLite).
    # Wrapped in a timeout: if the DB is temporarily unavailable during a rolling
    # restart the schema already exists from the last boot, so it's safe to proceed.
    import asyncio as _asyncio
    try:
        await _asyncio.wait_for(ensure_schema(settings.database.url), timeout=20.0)
    except Exception as exc:
        log.warning("ensure_schema skipped (timeout or error); schema assumed current")
        # The WARNING above is deliberately terse (no exception detail); this
        # is the "which was it" companion -- timeout vs. a real DB error, and
        # what the error actually was.
        debug_event(
            log, "startup ensure_schema failed",
            exc_type=type(exc).__name__, exc_message=str(exc),
        )
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
    except Exception as exc:
        log.warning("sync_telephony_from_yaml skipped (timeout or error)")
        debug_event(
            log, "startup sync_telephony_from_yaml failed",
            exc_type=type(exc).__name__, exc_message=str(exc),
        )
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
        # A call-outcome envelope referencing a tenant_id absent from the
        # currently loaded app.state.tenants -- e.g. a reload race, or a
        # deleted tenant with a call still in flight.
        debug_event(log, "tenant_event tenant_lookup miss", tenant_id=tenant_id)
        return None

    async def _notify_tenant_event(envelope: dict) -> None:
        settings = _tenant_settings_by_id(envelope.get("tenant_id"))
        url = await resolve_events_webhook_url(settings, sessionmaker)
        if not url:
            # No events_webhook_url configured for this tenant/CRM -- the
            # event is dropped here, silently, with nothing at any level.
            debug_event(
                log, "tenant_event dispatch skipped",
                tenant_id=envelope.get("tenant_id"), event_type=envelope.get("event_type"),
                reason="no_webhook_url_configured",
            )
            return
        secret_env = getattr(settings, "events_webhook_secret_env", None)
        resolver = getattr(app.state, "tenant_resolver", None)
        secret = await _resolve_tenant_event_secret(
            settings, secret_env, resolver,
            tenant_id=envelope.get("tenant_id"),
            event_type=envelope.get("event_type"),
        )
        debug_event(
            log, "tenant_event dispatch request",
            tenant_id=envelope.get("tenant_id"), event_type=envelope.get("event_type"),
            # redact_url like every other url in this file: this one is
            # tenant-configured, so it can carry an api key as a query param
            # or basic-auth userinfo -- both of which redact_url drops.
            url=redact_url(url), has_secret=bool(secret),
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
    debug_event(
        log, "startup provider_registry built",
        stt_provider=settings.pipeline.stt.provider, stt_model=settings.pipeline.stt.model,
        llm_provider=settings.pipeline.llm.provider, llm_model=settings.pipeline.llm.model,
        tts_provider=settings.pipeline.tts.provider,
        telephony_provider=settings.pipeline.telephony.provider,
        vector_store_provider=settings.pipeline.vector_store.provider,
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
    else:
        debug_event(log, "startup dev_console skipped", reason="VOX_DEV_CONSOLE not set to '1'")
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
    # Voice-note replies: injects the SAME TenantProviders instance the
    # voice/telephony paths use, but chat resolves its TTS client through
    # `get_chat_tts` (pipeline.chat_voice, falling back to pipeline.tts) under
    # its own cache key — no longer the same source of truth as the
    # voice-call cascade's `get_tts` (see src/auth/registry.py).
    chat_api.set_tts_providers(providers)
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
        debug_event(
            log, "startup media_storage configured", backend="s3",
            endpoint_url=ms.endpoint_url, bucket=ms.bucket, region=ms.region,
            has_access_key=bool(ms.access_key), has_secret_key=bool(ms.secret_key),
        )
    else:
        from src.providers.media.local import LocalMediaStorage
        log.info("media storage: using local filesystem fallback (/tmp/chat_media)")
        chat_api.set_media_store(LocalMediaStorage())
        debug_event(log, "startup media_storage configured", backend="local_filesystem")
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
    # URL redacted (never raw): GRAFANA_PROMETHEUS_PUSH_URL isn't itself a
    # credential, but redact_url is cheap insurance against a push URL that
    # embeds one. The paired GRAFANA_PROMETHEUS_PUSH_AUTH never appears here.
    debug_event(
        log, "startup metrics_push scheduled",
        interval_s=settings.secrets.METRICS_PUSH_INTERVAL_S,
        push_url=redact_url(settings.secrets.GRAFANA_PROMETHEUS_PUSH_URL)
        if settings.secrets.GRAFANA_PROMETHEUS_PUSH_URL else None,
    )
    metrics_push_task = asyncio.create_task(_push_turn_metrics_loop(
        settings.secrets.METRICS_PUSH_INTERVAL_S,
        settings.secrets.GRAFANA_PROMETHEUS_PUSH_URL,
        settings.secrets.GRAFANA_PROMETHEUS_PUSH_AUTH,
    ))
    debug_event(
        log, "startup chat_metrics_prune scheduled",
        retention_days=settings.secrets.CHAT_METRICS_RETENTION_DAYS,
    )
    chat_metrics_prune_task = asyncio.create_task(_prune_chat_turn_metrics_loop(
        settings.secrets.CHAT_METRICS_RETENTION_DAYS,
    ))
    webhook_outbox_task = asyncio.create_task(_webhook_outbox_loop())

    try:
        yield
    finally:
        log.info("shutdown")
        reaper_task.cancel()
        kb_seed_task.cancel()
        metrics_push_task.cancel()
        chat_metrics_prune_task.cancel()
        webhook_outbox_task.cancel()
        # None of the five are awaited after cancel() -- a task that ignores
        # or is slow to honor cancellation would leave this event as the only
        # trace that teardown even asked it to stop.
        debug_event(
            log, "shutdown background_tasks cancel_requested",
            tasks=["reap_stale_calls", "seed_crm_kb", "push_turn_metrics",
                   "prune_chat_turn_metrics", "webhook_outbox"],
        )
        telephony_hooks.set_bridge_factory(None)
        telephony_hooks.set_exotel_bridge_factory(None)
        telephony_hooks.set_stringee_bridge_factory(None)
        telephony_hooks.set_softphone_providers(None)
        livekit_routes.set_livekit_bridge_factory(None)
        chat_api.set_chatbot_factory(None)
        chat_api.set_chat_sessionmaker(None)
        chat_api.set_chat_handoff_store(None)
        chat_api.set_media_store(None)
        chat_api.set_tts_providers(None)
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


async def _probe_dependencies(app: FastAPI) -> tuple[str, str]:
    """Probe Redis and the DB, returning (redis_status, db_status) as "ok"/"down".

    Shared by /health and /ready so the two routes can never drift on what
    counts as "down".
    """
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

    return redis_status, db_status


@app.get("/health")
async def health() -> dict:
    """Liveness + dependency probe + per-tenant provider routing."""
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()

    redis_status, db_status = await _probe_dependencies(app)

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


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe: 503 when Redis or the DB is down.

    Unlike /health (which always returns 200 for Northflank's liveness
    check), /ready reflects real dependency health so callers that need to
    gate on it (e.g. a load balancer or orchestrator doing readiness-based
    routing) can do so.
    """
    redis_status, db_status = await _probe_dependencies(app)

    if redis_status == "down" or db_status == "down":
        return JSONResponse(
            status_code=503,
            content={"redis": redis_status, "db": db_status, "status": "not_ready"},
        )

    return JSONResponse(
        content={"redis": redis_status, "db": db_status, "status": "ready"},
    )
