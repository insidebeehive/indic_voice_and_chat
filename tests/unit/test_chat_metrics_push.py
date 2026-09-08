"""Tests for the ChatBot turn-metrics -> Grafana Cloud Prometheus push job
(turn-metrics plan, Phase 3, §5). Mirrors tests/unit/test_turn_metrics_push.py's
sqlite-in-memory sessionmaker fixture and respx-mocked-HTTP style.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pytest_asyncio
import respx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.chat_turn_metrics import ChatToolMetricRow, ChatTurnMetric
from src.models.database import Base
from src.observability.chat_metrics_push import aggregate_and_push_chat_metrics
from src.observability.turn_metrics_push import _push

PUSH_URL = "https://prometheus-prod-example.grafana.net/api/prom/push"
CHAT_PUSH_ROUTE = f"{PUSH_URL}/metrics/job/vox_chat_turn_metrics"
VOICE_PUSH_ROUTE = f"{PUSH_URL}/metrics/job/vox_turn_metrics"


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()


_TURN_DEFAULTS = dict(
    tenant_id="dev", crm_id="betstudio", session_id="cs_1", trace_id="tr_1",
    path="tools", llm_provider="GeminiLLMAdapter", llm_model="gemini-2.0-flash",
    action="continue", total_ms=0, llm_total_ms=0, llm_calls=0, tool_total_ms=0,
    tool_calls=0, tool_failures=0, tool_timeouts=0, tool_calls_skipped=0,
    kb_search_ms=0, kb_searches=0, retrieved_chunks=0, rounds=0,
    rounds_exhausted=False, retry_fired=False, failure_directive_fired=False,
    failure_directive_escalated=False, guard_hallucination_fired=False,
    guard_no_grounding_fired=False, guard_unverified_data_fired=False, escalated=False,
)


async def _seed_turn(sessionmaker, tools=(), **overrides) -> int:
    fields = dict(_TURN_DEFAULTS)
    fields.update(overrides)
    async with sessionmaker() as db:
        row = ChatTurnMetric(**fields)
        db.add(row)
        await db.flush()
        for t in tools:
            db.add(ChatToolMetricRow(turn_id=row.id, tenant_id=fields["tenant_id"], **t))
        await db.commit()
        return row.id


class _ExplodingSessionmaker:
    def __call__(self):
        raise AssertionError("sessionmaker must not be invoked when push_url is unset")


async def test_noop_when_push_url_unset() -> None:
    n = await aggregate_and_push_chat_metrics(_ExplodingSessionmaker(), None, None)
    assert n == 0


async def test_noop_when_push_url_empty_string() -> None:
    n = await aggregate_and_push_chat_metrics(_ExplodingSessionmaker(), "", None)
    assert n == 0


async def test_push_default_job_name_unchanged_for_voice_caller() -> None:
    """The §4 refactor (parameterizing turn_metrics_push.py::_push's job name)
    must not regress the existing voice caller -- _push's default job name is
    still vox_turn_metrics when called with no job_name argument, exactly as
    aggregate_and_push_turn_metrics calls it."""
    from prometheus_client import CollectorRegistry

    with respx.mock:
        route = respx.put(VOICE_PUSH_ROUTE).mock(return_value=httpx.Response(200))
        await _push(CollectorRegistry(), PUSH_URL, None)
        assert route.called


@respx.mock
async def test_push_lands_on_distinct_chat_job_path(sessionmaker) -> None:
    chat_route = respx.put(CHAT_PUSH_ROUTE).mock(return_value=httpx.Response(200))
    voice_route = respx.put(VOICE_PUSH_ROUTE).mock(return_value=httpx.Response(200))
    await _seed_turn(sessionmaker, total_ms=100)

    n = await aggregate_and_push_chat_metrics(sessionmaker, PUSH_URL, None)

    assert n == 1
    assert chat_route.called
    assert not voice_route.called  # never touches voice's job/group


@respx.mock
async def test_empty_window_still_pushes_to_clear_stale_gauges(sessionmaker) -> None:
    route = respx.put(CHAT_PUSH_ROUTE).mock(return_value=httpx.Response(200))
    # A row exists but well outside the window -> aggregated count is 0.
    await _seed_turn(
        sessionmaker, total_ms=1, created_at=datetime.utcnow() - timedelta(seconds=99999),
    )

    n = await aggregate_and_push_chat_metrics(sessionmaker, PUSH_URL, None, window_s=10)

    assert n == 0
    assert route.called  # the push DOES happen, to clear any stale prior data
    body = route.calls.last.request.content.decode()
    assert "vox_chat_turn_metric_count{" not in body
    assert "vox_chat_tool_metric_count{" not in body


@respx.mock
async def test_turn_and_tool_groups_and_percentiles(sessionmaker) -> None:
    route = respx.put(CHAT_PUSH_ROUTE).mock(return_value=httpx.Response(200))
    now = datetime.utcnow()

    for i, v in enumerate([100, 200, 300]):
        await _seed_turn(
            sessionmaker, session_id=f"cs_{i}", path="tools", total_ms=v,
            llm_total_ms=v, tool_total_ms=v, kb_search_ms=v,
            created_at=now - timedelta(seconds=10 * (i + 1)),
            tools=[{
                "tool_name": "get_player_wallet", "kind": "crm", "latency_ms": v,
                "outcome": "ok" if i < 2 else "timeout",
                "budget_slice_ms": 5000, "round_index": 0,
            }],
        )
    # Different path -> separate turn group.
    await _seed_turn(
        sessionmaker, session_id="cs_single_shot", path="single_shot", total_ms=50,
        created_at=now - timedelta(seconds=5),
    )
    # Outside the window entirely.
    await _seed_turn(
        sessionmaker, session_id="cs_old", total_ms=999999,
        created_at=now - timedelta(seconds=99999),
    )

    n = await aggregate_and_push_chat_metrics(sessionmaker, PUSH_URL, "user:key", window_s=100)

    assert n == 4  # excludes the row outside the window
    assert route.called
    body = route.calls.last.request.content.decode()

    assert 'vox_chat_turn_metric_count{path="tools",tenant_id="dev"} 3.0' in body
    assert (
        'vox_chat_turn_metric_latency_ms{path="tools",quantile="p50",'
        'stage="total_ms",tenant_id="dev"} 200.0' in body
    )
    assert (
        'vox_chat_turn_metric_latency_ms{path="tools",quantile="p95",'
        'stage="total_ms",tenant_id="dev"} 300.0' in body
    )
    assert 'vox_chat_turn_metric_count{path="single_shot",tenant_id="dev"} 1.0' in body

    assert (
        'vox_chat_tool_metric_count{outcome="ok",tenant_id="dev",'
        'tool_name="get_player_wallet"} 2.0' in body
    )
    assert (
        'vox_chat_tool_metric_count{outcome="timeout",tenant_id="dev",'
        'tool_name="get_player_wallet"} 1.0' in body
    )
    assert "999999" not in body


@respx.mock
async def test_ws_failure_rows_pushed_under_own_gauge_not_mixed_into_turn_latency(
    sessionmaker,
) -> None:
    """A WS-layer failure row (action='failed_timeout') must land in
    vox_chat_turn_failure_count, grouped by (tenant_id, action) -- and must
    NOT be counted in vox_chat_turn_metric_count or dilute
    vox_chat_turn_metric_latency_ms's percentiles for completed turns."""
    route = respx.put(CHAT_PUSH_ROUTE).mock(return_value=httpx.Response(200))
    await _seed_turn(sessionmaker, session_id="cs_ok", path="tools", total_ms=100)
    await _seed_turn(
        sessionmaker, session_id="cs_failed", path="tools", total_ms=90000,
        action="failed_timeout", llm_total_ms=0, rounds=0,
    )

    n = await aggregate_and_push_chat_metrics(sessionmaker, PUSH_URL, None, window_s=100)

    assert n == 2
    assert route.called
    body = route.calls.last.request.content.decode()

    assert 'vox_chat_turn_failure_count{action="failed_timeout",tenant_id="dev"} 1.0' in body
    # Only the one healthy turn counted -- the failure row didn't join this group.
    assert 'vox_chat_turn_metric_count{path="tools",tenant_id="dev"} 1.0' in body
    # The failure row's total_ms=90000 must not appear as a "tools" p95.
    assert "90000" not in body


@respx.mock
async def test_push_http_failure_is_caught_and_does_not_raise(sessionmaker) -> None:
    respx.put(CHAT_PUSH_ROUTE).mock(return_value=httpx.Response(500))
    await _seed_turn(sessionmaker, total_ms=100)

    n = await aggregate_and_push_chat_metrics(sessionmaker, PUSH_URL, None)

    assert n == 1  # aggregation itself succeeded; only the push failed


async def test_db_query_failure_is_caught_and_does_not_raise() -> None:
    class _BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("db unavailable")

    with respx.mock:
        # No route registered -- a network call here would also fail the test.
        n = await aggregate_and_push_chat_metrics(_BrokenSessionmaker(), PUSH_URL, None)
        assert n == 0
