"""Batch-aggregates ``ChatTurnMetric``/``ChatToolMetricRow`` rows (per-chat-turn
and per-tool-call latency/outcome, see ``src/models/chat_turn_metrics.py``)
into Prometheus metric families and pushes them to Grafana Cloud (turn-metrics
plan, Phase 3, §5). Runs from the same periodic in-process loop voice's push
already uses (``src/main.py``'s ``_push_turn_metrics_loop``, called
sequentially alongside ``aggregate_and_push_turn_metrics``) — never on the
live chat-turn path.

**Sibling module, not an extension of ``turn_metrics_push.py`` — deliberately.**
See the turn-metrics plan's §5 for the full argument; the short version:

- The grouping keys are entirely different shapes: chat's turn gauges group
  by ``(tenant_id, path)`` and its tool gauges by
  ``(tenant_id, tool_name, outcome)``, vs. voice's provider-combo
  ``(mode, stt_provider, llm_provider, tts_provider)``. There is no shared
  grouping key to factor out.
- **Decisively:** a Prometheus Pushgateway PUT *replaces the entire
  job/group's prior state* — this module's ``aggregate_and_push_turn_metrics``
  counterpart relies on exactly that to clear stale data when a window goes
  quiet (see its own docstring). Sharing a job name between voice and chat
  would let one product's empty push window silently wipe the other's still-
  live gauges. Hence ``_JOB_NAME`` here (``vox_chat_turn_metrics``) is
  distinct from voice's (``vox_turn_metrics``) — never reuse one job name for
  both.

**Reuse, not duplication, of the parts that ARE shape-independent:**
``_percentile`` (a pure function over a sorted list) and ``_push`` (the actual
Pushgateway PUT, now taking ``job_name`` as a parameter — see
``turn_metrics_push.py``'s own docstring on that function for why) are
imported from ``turn_metrics_push.py`` rather than reimplemented.
``_PushFailureWarner`` is likewise imported and given its OWN instance here
(module-level, below) — chat's push outages must warn/re-arm independently of
voice's, never share one "already warned" flag.

**Cardinality note.** ``(tenant_id, tool_name, outcome)`` is fine at today's
scale (~1 live tenant, ~23 catalog tools, 5 outcome values — at most ~115
label combinations for the tool gauges). If tenant count grows meaningfully,
drop ``tenant_id`` from the TOOL gauges first (keep it on the turn gauges,
where it's the primary dashboard filter) rather than dropping ``tool_name``
or ``outcome``, which are the actual diagnostic signal (`turn-metrics plan §5`).

**Aggregation window.** Same rolling-window (not watermarked) design as
``aggregate_and_push_turn_metrics`` — see that module's docstring for why a
bit of overlap between consecutive runs is fine here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from prometheus_client import CollectorRegistry, Gauge
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.models.chat_turn_metrics import (
    WS_TURN_FAILURE_ACTIONS,
    ChatToolMetricRow,
    ChatTurnMetric,
)
from src.observability.turn_metrics_push import _PushFailureWarner, _percentile, _push
from src.utils.redact import redact_url

log = logging.getLogger(__name__)

# Distinct from voice's "vox_turn_metrics" -- see module docstring for why
# sharing would be actively wrong (a Pushgateway PUT replaces the whole
# job/group's prior state).
_JOB_NAME = "vox_chat_turn_metrics"

# Own instance, deliberately separate from turn_metrics_push.py's module-level
# warner -- chat's push failures/recoveries must never silence or re-arm
# voice's warnings, or vice versa.
_push_failure_warner = _PushFailureWarner()

_TURN_GROUP_LABELS = ("tenant_id", "path")
_TURN_LATENCY_COLUMNS = ("total_ms", "llm_total_ms", "tool_total_ms", "kb_search_ms")

# WS-layer failure rows (action in WS_TURN_FAILURE_ACTIONS -- a turn that
# raised or timed out before the agent produced a ChatTurnResult, see
# src/api/chat.py::_record_ws_turn_failure_metric) are pushed under their OWN
# gauge/group, never mixed into _TURN_GROUP_LABELS's count/latency gauges:
# their llm_total_ms/tool_total_ms/kb_search_ms are all 0 placeholders, and
# folding them into the same percentile population as completed turns would
# drag every turn-level latency gauge toward 0 during a WS-layer outage --
# exactly backwards for an alert meant to catch that outage. This is what
# makes a chat-turn FAILURE rate alertable at all (turn-metrics plan §5's
# whole point for the push module), which grouping only by (tenant_id, path)
# cannot answer.
_TURN_FAILURE_GROUP_LABELS = ("tenant_id", "action")

_TOOL_GROUP_LABELS = ("tenant_id", "tool_name", "outcome")

_PERCENTILES = (50, 95)


def _build_registry(
    turn_groups: dict[tuple[str, ...], list[ChatTurnMetric]],
    failure_groups: dict[tuple[str, ...], list[ChatTurnMetric]],
    tool_groups: dict[tuple[str, ...], list[ChatToolMetricRow]],
) -> CollectorRegistry:
    registry = CollectorRegistry()

    turn_count_gauge = Gauge(
        "vox_chat_turn_metric_count",
        "Number of COMPLETED (non-failure) ChatTurnMetric rows aggregated in "
        "this push window -- see vox_chat_turn_failure_count for WS-layer "
        "failure rows, tracked separately",
        _TURN_GROUP_LABELS,
        registry=registry,
    )
    turn_latency_gauge = Gauge(
        "vox_chat_turn_metric_latency_ms",
        "Pre-computed percentile latency (ms) per turn-level stage over this "
        "push window, over COMPLETED (non-failure) turns only",
        _TURN_GROUP_LABELS + ("stage", "quantile"),
        registry=registry,
    )
    for key, rows in turn_groups.items():
        labels = dict(zip(_TURN_GROUP_LABELS, key, strict=True))
        turn_count_gauge.labels(**labels).set(len(rows))
        for column in _TURN_LATENCY_COLUMNS:
            values = sorted(getattr(r, column) for r in rows)
            if not values:
                continue
            for pct in _PERCENTILES:
                turn_latency_gauge.labels(**labels, stage=column, quantile=f"p{pct}").set(
                    _percentile(values, pct)
                )

    turn_failure_gauge = Gauge(
        "vox_chat_turn_failure_count",
        "Number of chat_turn_metrics rows in this push window whose action "
        "marks a WS-layer turn failure (src/api/chat.py's "
        "_record_ws_turn_failure_metric) -- grouped by the specific failure "
        "reason so e.g. a spike in failed_timeout is distinguishable from "
        "failed_llm_quota. Divide by (this + vox_chat_turn_metric_count) for "
        "an alertable chat-turn failure rate.",
        _TURN_FAILURE_GROUP_LABELS,
        registry=registry,
    )
    for key, rows in failure_groups.items():
        labels = dict(zip(_TURN_FAILURE_GROUP_LABELS, key, strict=True))
        turn_failure_gauge.labels(**labels).set(len(rows))

    tool_count_gauge = Gauge(
        "vox_chat_tool_metric_count",
        "Number of chat_tool_metrics rows aggregated in this push window, "
        "by outcome -- the input to a timeout/failure-rate alert "
        "(e.g. sum(outcome='timeout') / sum(all outcomes) by tool_name)",
        _TOOL_GROUP_LABELS,
        registry=registry,
    )
    tool_latency_gauge = Gauge(
        "vox_chat_tool_metric_latency_ms",
        "Pre-computed percentile latency (ms) per tool call over this push window",
        _TOOL_GROUP_LABELS + ("quantile",),
        registry=registry,
    )
    for key, rows in tool_groups.items():
        labels = dict(zip(_TOOL_GROUP_LABELS, key, strict=True))
        tool_count_gauge.labels(**labels).set(len(rows))
        values = sorted(r.latency_ms for r in rows)
        if not values:
            continue
        for pct in _PERCENTILES:
            tool_latency_gauge.labels(**labels, quantile=f"p{pct}").set(_percentile(values, pct))

    return registry


async def aggregate_and_push_chat_metrics(
    sessionmaker: async_sessionmaker,
    push_url: Optional[str],
    push_auth: Optional[str],
    *,
    window_s: float = 120.0,
) -> int:
    """Aggregate ``ChatTurnMetric``/``ChatToolMetricRow`` rows created in the
    last ``window_s`` seconds and push count + p50/p95 latency Gauges to a
    Pushgateway-compatible endpoint, under job name ``vox_chat_turn_metrics``.

    Mirrors ``turn_metrics_push.py::aggregate_and_push_turn_metrics`` in every
    structural respect (no-op when ``push_url`` is unset, still pushes an
    empty registry when no rows fall in the window so stale gauges get
    cleared, never raises). Returns the number of ``ChatTurnMetric`` rows
    aggregated (0 when ``push_url`` is unset, a query fails, or no rows fall
    in the window).
    """
    if not push_url:
        log.debug("chat-metrics push skipped (GRAFANA_PROMETHEUS_PUSH_URL unset)")
        return 0

    try:
        async with sessionmaker() as db:
            dialect_name = db.get_bind().dialect.name
            if dialect_name == "sqlite":
                # SQLite is test-only (production always runs Postgres) -- see
                # turn_metrics_push.py's identical comment for the reasoning.
                cutoff = datetime.utcnow() - timedelta(seconds=window_s)
            else:
                cutoff = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, window_s)
            turn_rows = (
                await db.execute(select(ChatTurnMetric).where(ChatTurnMetric.created_at >= cutoff))
            ).scalars().all()
            # Filtered by JOINING to the parent's indexed created_at, NOT
            # ChatToolMetricRow's own created_at column -- that column has no
            # standalone index (only `(tool_name, created_at)`, whose leading
            # column can't serve a bare created_at predicate; see
            # src/models/chat_turn_metrics.py's __table_args__), so filtering
            # on it directly would full-scan chat_tool_metrics every push
            # cycle -- the exact mistake alembic/versions/0019 exists to
            # document and fix for the PARENT table. The parent's created_at
            # IS indexed (both standalone and composite with tenant_id), and
            # every chat_tool_metrics row is written in the same transaction
            # as its parent (see record_chat_turn_metric), so this is
            # equivalent to filtering on the child's own created_at, just
            # index-backed.
            tool_rows = (
                await db.execute(
                    select(ChatToolMetricRow)
                    .join(ChatTurnMetric, ChatToolMetricRow.turn_id == ChatTurnMetric.id)
                    .where(ChatTurnMetric.created_at >= cutoff)
                )
            ).scalars().all()
    except Exception:  # noqa: BLE001 - must never affect a live chat turn
        log.warning("chat-metrics aggregation query failed", exc_info=True)
        return 0

    # Deliberately does NOT return early when everything's empty -- see
    # aggregate_and_push_turn_metrics's identical comment: a Pushgateway PUT
    # replaces the job's entire prior state, so an empty push is what
    # correctly clears stale gauges once chat traffic stops.
    #
    # WS-layer failure rows (action in WS_TURN_FAILURE_ACTIONS) are split out
    # into their own group -- see _TURN_FAILURE_GROUP_LABELS's comment for why
    # mixing them into the regular turn latency/count gauges would be wrong.
    turn_groups: dict[tuple[str, ...], list[ChatTurnMetric]] = {}
    failure_groups: dict[tuple[str, ...], list[ChatTurnMetric]] = {}
    for row in turn_rows:
        if row.action in WS_TURN_FAILURE_ACTIONS:
            key = tuple(getattr(row, label) or "" for label in _TURN_FAILURE_GROUP_LABELS)
            failure_groups.setdefault(key, []).append(row)
        else:
            key = tuple(getattr(row, label) or "" for label in _TURN_GROUP_LABELS)
            turn_groups.setdefault(key, []).append(row)

    tool_groups: dict[tuple[str, ...], list[ChatToolMetricRow]] = {}
    for row in tool_rows:
        key = tuple(getattr(row, label) or "" for label in _TOOL_GROUP_LABELS)
        tool_groups.setdefault(key, []).append(row)

    try:
        registry = _build_registry(turn_groups, failure_groups, tool_groups)
    except Exception:  # noqa: BLE001 - must never affect a live chat turn
        log.warning("chat-metrics registry build failed", exc_info=True)
        return 0

    try:
        await _push(registry, push_url, push_auth, job_name=_JOB_NAME)
    except Exception as exc:  # noqa: BLE001 - must never affect a live chat turn
        # Same redaction rationale as turn_metrics_push.py's identical block:
        # never log the raw exception (may embed push_url userinfo) or a full
        # traceback -- and rate-limited to one warning per outage (own
        # instance of _PushFailureWarner, independent of voice's).
        _push_failure_warner.warn_once(
            f"chat-metrics push failed: {type(exc).__name__} ({redact_url(push_url)})"
        )
        return len(turn_rows)

    _push_failure_warner.mark_recovered()
    return len(turn_rows)
