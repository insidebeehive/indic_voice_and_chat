"""Route test for GET /tenants/{tenant_id}/chat-turn-metrics (turn-metrics
plan, Phase 3, §5). Mirrors tests/unit/test_benchmarks_turn_metrics_route.py's
fixture shape."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import tenants
from src.api.deps import get_db_session
from src.auth.middleware import set_admin_tokens
from src.models.chat_turn_metrics import ChatToolMetricRow, ChatTurnMetric
from src.models.database import Base
from src.models.tenant import Tenant

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}

# Deliberately distinctive values -- used to assert neither ever leaks into
# the aggregate-only response body (turn-metrics plan §7).
SECRET_SESSION_ID = "cs_super_secret_ws_capability_token"
SECRET_TRACE_ID = "trace_do_not_leak_me"


@pytest_asyncio.fixture
async def ctx():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        s.add(Tenant(id="dev", slug="dev", name="Dev Tenant"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_admin_tokens(["admin-token"])
    app = FastAPI()
    app.include_router(tenants.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, sm
    set_admin_tokens([])
    await engine.dispose()


_TURN_DEFAULTS = dict(
    tenant_id="dev", crm_id="betstudio",
    session_id=SECRET_SESSION_ID, trace_id=SECRET_TRACE_ID,
    path="tools", llm_provider="GeminiLLMAdapter", llm_model="gemini-2.0-flash",
    action="continue",
    total_ms=0, llm_total_ms=0, llm_calls=0, tool_total_ms=0, tool_calls=0,
    tool_failures=0, tool_timeouts=0, tool_calls_skipped=0, kb_search_ms=0,
    kb_searches=0, retrieved_chunks=0, rounds=0, rounds_exhausted=False,
    retry_fired=False, failure_directive_fired=False, failure_directive_escalated=False,
    guard_hallucination_fired=False, guard_no_grounding_fired=False,
    guard_unverified_data_fired=False, escalated=False,
)


async def _seed_turn(sm, tools=(), **overrides) -> int:
    fields = dict(_TURN_DEFAULTS)
    fields.update(overrides)
    async with sm() as db:
        row = ChatTurnMetric(**fields)
        db.add(row)
        await db.flush()
        for t in tools:
            db.add(ChatToolMetricRow(turn_id=row.id, tenant_id=fields["tenant_id"], **t))
        await db.commit()
        return row.id


async def test_requires_admin(ctx) -> None:
    client, _sm = ctx
    resp = await client.get("/tenants/dev/chat-turn-metrics")
    assert resp.status_code == 401


async def test_unknown_tenant_404s(ctx) -> None:
    client, _sm = ctx
    resp = await client.get("/tenants/nope/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 404


async def test_turns_aggregate_and_percentiles(ctx) -> None:
    client, sm = ctx
    for total_ms in (100, 200, 300):
        await _seed_turn(
            sm, total_ms=total_ms, llm_total_ms=total_ms // 2, tool_total_ms=10,
            kb_search_ms=5, rounds=2,
        )
    # One turn with a tool failure and a fired guard -- exercises the rate
    # computations (1/4 == 25.0%).
    await _seed_turn(
        sm, total_ms=400, tool_failures=1, guard_hallucination_fired=True,
        rounds_exhausted=True,
    )

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    turns = body["turns"]
    assert turns["samples"] == 4
    assert turns["p50_total_ms"] == 200  # nearest-rank p50 of [100,200,300,400]
    assert turns["p95_total_ms"] == 400
    assert turns["avg_total_ms"] == 250.0  # (100+200+300+400)/4
    assert turns["turns_with_tool_failure"] == 1
    assert turns["guard_hallucination_fired_rate_pct"] == 25.0
    assert turns["rounds_exhausted_rate_pct"] == 25.0
    assert turns["retry_fired_rate_pct"] == 0.0
    assert turns["escalated_rate_pct"] == 0.0


async def test_ws_failure_rows_excluded_from_averages_but_visible_separately(ctx) -> None:
    """WS-layer failure rows (src/api/chat.py::_record_ws_turn_failure_metric)
    carry total_ms but leave every other numeric/boolean column at its 0/False
    default. They must be counted in failed_turns/turn_failure_rate_pct, and
    EXCLUDED from every average/percentile/rate — a turn that never ran must
    not silently drag down the numbers for turns that did."""
    client, sm = ctx
    for total_ms in (1000, 1000):
        await _seed_turn(
            sm, total_ms=total_ms, llm_total_ms=1000, rounds=2,
            input_tokens=100, output_tokens=20, cached_tokens=10,
        )
    # A WS-layer timeout row: only total_ms + the failure-marker action are
    # real; everything else stays at its dataclass default (0/False), which
    # for the new token columns means input/output/cached_tokens=0 too.
    await _seed_turn(
        sm, total_ms=90000, llm_total_ms=0, rounds=0, action="failed_timeout",
        input_tokens=999999, output_tokens=999999, cached_tokens=999999,
    )

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    turns = resp.json()["turns"]

    assert turns["failed_turns"] == 1
    assert turns["turn_failure_rate_pct"] == round(100 / 3, 1)  # 1 failed / 3 total turns
    # Unaffected by the failure row's total_ms=90000 or its zeroed rounds --
    # if it weren't excluded, avg_total_ms would be (1000+1000+90000)/3.
    assert turns["samples"] == 2
    assert turns["avg_total_ms"] == 1000.0
    assert turns["avg_llm_total_ms"] == 1000.0
    assert turns["avg_rounds"] == 2.0
    assert turns["p95_total_ms"] == 1000
    # Same exclusion for the new token averages/ratio -- if the failure row's
    # 999999 token counts leaked in, these would be wildly higher than 100/
    # 20/10/10.0.
    assert turns["avg_input_tokens"] == 100.0
    assert turns["avg_output_tokens"] == 20.0
    assert turns["avg_cached_tokens"] == 10.0
    assert turns["cache_hit_rate_pct"] == 10.0


async def test_token_averages(ctx) -> None:
    client, sm = ctx
    await _seed_turn(sm, input_tokens=100, output_tokens=20, cached_tokens=10)
    await _seed_turn(sm, input_tokens=300, output_tokens=40, cached_tokens=30)

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    turns = resp.json()["turns"]
    assert turns["avg_input_tokens"] == 200.0  # (100+300)/2
    assert turns["avg_output_tokens"] == 30.0  # (20+40)/2
    assert turns["avg_cached_tokens"] == 20.0  # (10+30)/2
    # sum(cached)/sum(input) = (10+30)/(100+300) = 10.0%
    assert turns["cache_hit_rate_pct"] == 10.0


async def test_cache_hit_rate_is_ratio_of_sums_not_average_of_ratios(ctx) -> None:
    """cache_hit_rate_pct MUST be sum(cached_tokens)/sum(input_tokens), not an
    average of each turn's own cached/input ratio -- averaging per-turn
    ratios over-weights low-token turns. A 10-token turn with a 100% hit and
    a 1000-token turn with a 0% hit average to a misleading 50%, but the
    ratio of sums (10 / 1010) correctly reports ~1.0%."""
    client, sm = ctx
    await _seed_turn(sm, input_tokens=10, cached_tokens=10)
    await _seed_turn(sm, input_tokens=1000, cached_tokens=0)

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    turns = resp.json()["turns"]
    assert turns["cache_hit_rate_pct"] == round(10 * 100 / 1010, 1)
    assert turns["cache_hit_rate_pct"] != 50.0


async def test_window_h_excludes_out_of_window_rows(ctx) -> None:
    client, sm = ctx
    await _seed_turn(sm, total_ms=111, created_at=datetime.utcnow() - timedelta(hours=1))
    await _seed_turn(sm, total_ms=999999, created_at=datetime.utcnow() - timedelta(hours=48))

    resp = await client.get(
        "/tenants/dev/chat-turn-metrics", params={"window_h": 24}, headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["turns"]["samples"] == 1
    assert body["turns"]["avg_total_ms"] == 111.0
    assert "999999" not in resp.text


async def test_reply_length_stats_and_pre_migration_rows_dont_500(ctx) -> None:
    """B1 regression test (turn-metrics plan §2/§3): every chat_turn_metrics
    row written before migration 0026 has reply_chars/reply_words IS NULL.
    The percentile row-fetch this endpoint does for total_ms/reply_words/
    reply_chars must tolerate that NULL without ever calling sorted() on a
    list containing a bare None -- this endpoint is a bare route handler with
    no `except`, so a TypeError here is a hard 500 for the tenant's entire
    retention window, for every tenant, until enough pre-migration rows age
    out."""
    client, sm = ctx
    for words, chars in ((10, 40), (20, 90), (30, 140), (40, 190)):
        await _seed_turn(sm, total_ms=100, reply_words=words, reply_chars=chars)
    # Pre-migration-shaped row: reply_chars/reply_words left at their
    # ChatTurnMetric default (None -- simply not passed here), exactly what
    # an older, pre-0026 agent process's row looks like. Still a healthy,
    # non-failure row, so it must count toward `samples` but must be
    # silently skipped by the reply-length avg/percentile/sample-count
    # fields, not crash the endpoint.
    await _seed_turn(sm, total_ms=100)
    # A WS-layer failure row: excluded from `healthy` entirely (twice over --
    # both by the `action` filter and because its own reply_words/reply_chars
    # are NULL too), so even a wildly out-of-range reply_words here must
    # never reach the reply-length stats.
    await _seed_turn(
        sm, total_ms=90000, action="failed_timeout",
        reply_words=999999, reply_chars=999999,
    )

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    turns = resp.json()["turns"]

    assert turns["samples"] == 5  # 4 measured + 1 pre-migration-shaped, both healthy
    assert turns["failed_turns"] == 1
    assert turns["reply_length_samples"] == 4  # the pre-migration row is excluded
    assert turns["avg_reply_words"] == 25.0  # (10+20+30+40)/4
    assert turns["p50_reply_words"] == 20  # nearest-rank p50 of [10,20,30,40]
    assert turns["p95_reply_words"] == 40
    assert turns["avg_reply_chars"] == 115.0  # (40+90+140+190)/4
    # The WS-failure row's out-of-range values never leaked into the response.
    assert "999999" not in resp.text


async def test_tools_grouped_by_name_with_percentiles_and_failure_rate(ctx) -> None:
    client, sm = ctx
    await _seed_turn(sm, tools=[
        {"tool_name": "get_player_wallet", "kind": "crm", "latency_ms": 100,
         "outcome": "ok", "budget_slice_ms": 5000, "round_index": 0},
        {"tool_name": "get_player_wallet", "kind": "crm", "latency_ms": 200,
         "outcome": "ok", "budget_slice_ms": 5000, "round_index": 0},
        {"tool_name": "get_player_wallet", "kind": "crm", "latency_ms": 300,
         "outcome": "timeout", "budget_slice_ms": 5000, "round_index": 1},
        {"tool_name": "get_player_transactions", "kind": "crm", "latency_ms": 50,
         "outcome": "skipped_budget", "budget_slice_ms": 0, "round_index": 0},
    ])

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    tools = {t["tool_name"]: t for t in resp.json()["tools"]}
    assert set(tools) == {"get_player_wallet", "get_player_transactions"}

    wallet = tools["get_player_wallet"]
    assert wallet["calls"] == 3
    assert wallet["p50_latency_ms"] == 200  # nearest-rank p50 of [100,200,300]
    assert wallet["p95_latency_ms"] == 300
    assert wallet["timeouts"] == 1
    assert wallet["transport_errors"] == 0
    assert wallet["skipped"] == 0
    assert wallet["avg_budget_slice_ms"] == 5000.0
    # 1 timeout / 3 calls (denominator INCLUDES skipped_budget calls per
    # tool_calls' own convention, but this group has none) -> 33.3%.
    assert wallet["failure_rate_pct"] == round(100 / 3, 1)

    transactions = tools["get_player_transactions"]
    assert transactions["calls"] == 1
    assert transactions["skipped"] == 1
    # skipped_budget is "never tried" -- but chatbot.py's own docstring on
    # ChatTurnMetrics.tool_calls pins this exact case: a naive
    # tool_failures/tool_calls UNDERSTATES degradation whenever calls were
    # skipped, so the correct "how bad" number folds skipped_budget into the
    # numerator too (a 100%-unusable tool must not report a 0% failure rate).
    assert transactions["failure_rate_pct"] == 100.0


async def test_skipped_budget_calls_excluded_from_latency_percentiles(ctx) -> None:
    """A skipped_budget call was never dispatched -- chatbot.py records its
    latency_ms as ~0 (the time to decide to skip, not a real call duration).
    Mixing that into the percentile population would drag p50/p95 toward 0
    for a tool with many skips, answering "is this tool saturating its
    timeout" (the point of this endpoint per the plan's §2) with a wrong
    near-zero number."""
    client, sm = ctx
    await _seed_turn(sm, tools=[
        {"tool_name": "get_player_transactions", "kind": "crm", "latency_ms": 30000,
         "outcome": "timeout", "budget_slice_ms": 30000, "round_index": 0},
        {"tool_name": "get_player_transactions", "kind": "crm", "latency_ms": 1,
         "outcome": "skipped_budget", "budget_slice_ms": 0, "round_index": 1},
        {"tool_name": "get_player_transactions", "kind": "crm", "latency_ms": 1,
         "outcome": "skipped_budget", "budget_slice_ms": 0, "round_index": 1},
        {"tool_name": "get_player_transactions", "kind": "crm", "latency_ms": 1,
         "outcome": "skipped_budget", "budget_slice_ms": 0, "round_index": 1},
        {"tool_name": "get_player_transactions", "kind": "crm", "latency_ms": 1,
         "outcome": "skipped_budget", "budget_slice_ms": 0, "round_index": 1},
    ])

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    tx = {t["tool_name"]: t for t in resp.json()["tools"]}["get_player_transactions"]
    assert tx["calls"] == 5
    assert tx["skipped"] == 4
    # The single real (timeout) call's latency, NOT dragged toward 0 by the
    # 4 skipped_budget rows' near-zero latency_ms.
    assert tx["p50_latency_ms"] == 30000
    assert tx["p95_latency_ms"] == 30000


async def test_response_never_leaks_session_id_or_trace_id(ctx) -> None:
    client, sm = ctx
    await _seed_turn(sm, total_ms=100, tools=[
        {"tool_name": "get_player_wallet", "kind": "crm", "latency_ms": 100,
         "outcome": "ok", "budget_slice_ms": 5000, "round_index": 0},
    ])

    resp = await client.get("/tenants/dev/chat-turn-metrics", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body_text = resp.text

    assert SECRET_SESSION_ID not in body_text
    assert SECRET_TRACE_ID not in body_text
    assert "session_id" not in body_text
    assert "trace_id" not in body_text
