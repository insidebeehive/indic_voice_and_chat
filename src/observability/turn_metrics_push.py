"""Batch-aggregates ``TurnMetric`` rows (per-turn voice-call latency, see
``src/models/turn_metrics.py``) into Prometheus metric families and pushes
them to Grafana Cloud (Phase 2 observability). Runs from a periodic
in-process loop (``src/main.py``'s ``_push_turn_metrics_loop``) — never on
the live call path.

**Protocol note — read before pointing this at a real endpoint.** This
speaks the classic Prometheus **Pushgateway** wire protocol: a PUT of the
plain-text exposition format to ``{GRAFANA_PROMETHEUS_PUSH_URL}/metrics/job/
<job>`` (exactly what ``prometheus_client.exposition.push_to_gateway`` does
under the hood — see its ``_use_gateway`` helper). It does **not** speak
Prometheus's remote-write protocol, which is a completely different wire
format (protobuf messages, snappy-compressed). ``prometheus_client`` — the
library used here to build the metric families — has no built-in
remote-write support at all, only the Pushgateway push path and the standard
``/metrics`` scrape format; there's no well-maintained pure-Python
remote-write encoder to lean on, so hand-rolling protobuf+snappy would be a
much heavier lift than this phase warrants. This module therefore
implements the simpler, standard option: classic Pushgateway-style push.

That means ``GRAFANA_PROMETHEUS_PUSH_URL`` must point at a Pushgateway-
compatible ingestion endpoint. **Confirm the actual URL and protocol against
the real Grafana Cloud account before this goes live** — Grafana Cloud's
usual ingestion path for Prometheus metrics is remote-write, and if the real
account only offers that (not a Pushgateway-compatible path), the push step
here (``_push``) will need to be swapped for a protobuf+snappy remote-write
implementation; the aggregation logic (grouping, percentiles, metric family
construction) is unaffected and can be reused as-is.

**Aggregation window.** Uses a simple rolling window (the last
``window_s`` seconds of ``TurnMetric`` rows) rather than a persistent
"last aggregated" watermark. This is a periodic rollup for dashboarding, not
an exactly-once ledger, so a bit of overlap between consecutive runs is
fine — Grafana panels naturally handle overlapping/duplicate-timestamp
samples via max/last aggregation — and it avoids the complexity (and
failure modes) of persisting cross-restart state for something that must
never affect a live call.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import httpx
from prometheus_client import CollectorRegistry, Gauge
from prometheus_client.exposition import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.models.turn_metrics import TurnMetric
from src.utils.redact import redact_url

log = logging.getLogger(__name__)

_JOB_NAME = "vox_turn_metrics"

# Grouping key: the provider/mode combination a Grafana panel would slice
# latency by. stt_provider/tts_provider are nullable on the model (e.g. a
# text-only turn has no tts_provider) -- None is rendered as "" for the
# Prometheus label value (labels can't be None).
_GROUP_LABELS = ("mode", "stt_provider", "llm_provider", "tts_provider")

# All are NOT NULL on the model (default 0), so there's no per-row NULL to
# filter out -- a stage that doesn't apply to a given mode (e.g.
# tts_total_ms for a text-only turn) is naturally isolated because we
# already group by `mode`, so its zeros don't bleed into a voice group's
# percentiles.
_LATENCY_COLUMNS = (
    "stt_latency_ms",
    "llm_ttft_ms",
    "llm_total_ms",
    "tts_first_chunk_ms",
    "tts_total_ms",
    "total_latency_ms",
)
_PERCENTILES = (50, 95)


def _percentile(sorted_values: list[int], pct: int) -> int:
    """Nearest-rank percentile over an already-sorted, non-empty list."""
    n = len(sorted_values)
    idx = max(0, min(n - 1, math.ceil(pct / 100 * n) - 1))
    return sorted_values[idx]


def _build_registry(groups: dict[tuple[Optional[str], ...], list[TurnMetric]]) -> CollectorRegistry:
    registry = CollectorRegistry()
    count_gauge = Gauge(
        "vox_turn_metric_count",
        "Number of TurnMetric rows aggregated in this push window",
        _GROUP_LABELS,
        registry=registry,
    )
    latency_gauge = Gauge(
        "vox_turn_metric_latency_ms",
        "Pre-computed percentile latency (ms) per pipeline stage over this push window",
        _GROUP_LABELS + ("stage", "quantile"),
        registry=registry,
    )
    for key, rows in groups.items():
        labels = dict(zip(
            _GROUP_LABELS, ("" if v is None else v for v in key), strict=True,
        ))
        count_gauge.labels(**labels).set(len(rows))
        for column in _LATENCY_COLUMNS:
            values = sorted(getattr(r, column) for r in rows)
            if not values:
                continue
            for pct in _PERCENTILES:
                latency_gauge.labels(**labels, stage=column, quantile=f"p{pct}").set(
                    _percentile(values, pct)
                )
    return registry


async def _push(registry: CollectorRegistry, push_url: str, push_auth: Optional[str]) -> None:
    """PUT the registry's exposition-format payload to
    ``{push_url}/metrics/job/<job>`` (classic Pushgateway protocol — see
    module docstring). Raises on failure/non-2xx; the caller catches it.

    Uses ``httpx.AsyncClient`` (not ``prometheus_client.push_to_gateway``,
    which shells out to a blocking ``urllib`` call) so this never blocks the
    event loop it runs on.
    """
    auth = None
    if push_auth:
        if ":" in push_auth:
            user, _, key = push_auth.partition(":")
            auth = (user, key)
        else:
            log.warning(
                "GRAFANA_PROMETHEUS_PUSH_AUTH set without a ':' separator; "
                "pushing unauthenticated (expected 'user:api_key')"
            )
    url = f"{push_url.rstrip('/')}/metrics/job/{_JOB_NAME}"
    data = generate_latest(registry)
    async with httpx.AsyncClient(timeout=10.0, auth=auth) as client:
        resp = await client.put(url, content=data, headers={"Content-Type": CONTENT_TYPE_LATEST})
        resp.raise_for_status()


async def aggregate_and_push_turn_metrics(
    sessionmaker: async_sessionmaker,
    push_url: Optional[str],
    push_auth: Optional[str],
    *,
    window_s: float = 120.0,
) -> int:
    """Aggregate ``TurnMetric`` rows created in the last ``window_s`` seconds,
    grouped by (mode, stt_provider, llm_provider, tts_provider), and push
    count + p50/p95 latency Gauges per group to a Pushgateway-compatible
    endpoint.

    Returns the number of ``TurnMetric`` rows aggregated (0 when ``push_url``
    is unset, the DB query itself fails, or no rows fall in the window --
    that last case still performs an otherwise-empty push, to clear any
    stale data a prior window may have left in the Pushgateway, see below).
    Never raises: DB and network failures are caught and logged, matching
    ``src/main.py``'s ``_reap_stale_calls_loop`` convention -- this runs off
    a periodic background loop and must never affect a live call/turn.
    """
    if not push_url:
        log.debug("turn-metrics push skipped (GRAFANA_PROMETHEUS_PUSH_URL unset)")
        return 0

    try:
        async with sessionmaker() as db:
            dialect_name = db.get_bind().dialect.name
            if dialect_name == "sqlite":
                # SQLite is used only in tests (production always runs
                # Postgres -- see src/models/database.py's _is_sqlite); an
                # in-memory test DB has no separate server clock to disagree
                # with Python's, so a Python-computed cutoff is fine here.
                cutoff = datetime.utcnow() - timedelta(seconds=window_s)
            else:
                # Compute "now - window_s" using the DB SERVER's own clock,
                # not Python's `datetime.utcnow()`, so this is correct
                # regardless of what timezone the server's clock is actually
                # in -- created_at is DateTime(timezone=False) populated by
                # the server's own func.now() (see TurnMetric), so both sides
                # of this comparison must be evaluated by that same clock.
                # func.make_interval(...) (not string interpolation) keeps
                # window_s a normal bound parameter.
                cutoff = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, window_s)
            rows = (
                await db.execute(select(TurnMetric).where(TurnMetric.created_at >= cutoff))
            ).scalars().all()
    except Exception:  # noqa: BLE001 - must never affect a live call
        log.warning("turn-metrics aggregation query failed", exc_info=True)
        return 0

    # Deliberately does NOT return early when `rows` is empty: Prometheus
    # Pushgateway retains the LAST pushed value for a job/group indefinitely
    # until explicitly overwritten -- so when traffic stops (overnight, or a
    # real outage), skipping the push here would leave the dashboard showing
    # stale, no-longer-true counts/latencies as if they were current. Instead
    # we still push, with an empty registry (`groups` stays `{}` below) --
    # a Pushgateway PUT replaces the entire job/group's prior state, so an
    # empty push correctly clears the stale data.
    groups: dict[tuple[Optional[str], ...], list[TurnMetric]] = {}
    for row in rows:
        key = tuple(getattr(row, label) for label in _GROUP_LABELS)
        groups.setdefault(key, []).append(row)

    try:
        registry = _build_registry(groups)
    except Exception:  # noqa: BLE001 - must never affect a live call
        log.warning("turn-metrics registry build failed", exc_info=True)
        return 0

    try:
        await _push(registry, push_url, push_auth)
    except Exception as exc:  # noqa: BLE001 - must never affect a live call
        # Deliberately NOT exc_info=True, and NOT interpolating str(exc):
        # httpx's raise_for_status() embeds the full request URL (including
        # any userinfo) in its exception message, and
        # GRAFANA_PROMETHEUS_PUSH_URL could in principle carry embedded
        # basic-auth credentials -- logging the raw exception (traceback or
        # message) could leak them, and since Phase 1's Loki shipping may be
        # active, potentially ship them to an external log aggregator too.
        # Log only the exception type + a redacted push_url instead, mirroring
        # LokiPushHandler's own push-failure logging (src/utils/logging.py).
        log.warning(
            "turn-metrics push failed: %s (%s)",
            type(exc).__name__, redact_url(push_url),
        )
        return len(rows)

    return len(rows)
