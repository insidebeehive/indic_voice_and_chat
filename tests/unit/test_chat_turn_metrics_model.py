"""Round-trip test for ChatTurnMetric/ChatToolMetricRow + the
record_chat_turn_metric helper (turn-metrics plan Phase 2). Mirrors
test_turn_metrics_model.py's shape."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.chat_turn_metrics import (
    ChatToolMetricRow,
    ChatTurnMetric,
    record_chat_turn_metric,
)
from src.models.database import Base
from src.models.tenant import Tenant


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        # Foreign-key enforcement is off by default on sqlite -- turn it on so
        # the CASCADE-delete assertion below actually exercises the FK, not
        # just app-level plumbing.
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    # chat_turn_metrics.tenant_id FKs to tenants.id -- with FK enforcement on,
    # a parent-row insert needs a real tenants row to point at.
    async with sm() as db:
        db.add(Tenant(id="dev", slug="dev", name="Dev Tenant"))
        await db.commit()
    yield sm
    await engine.dispose()


def _metrics_dict(**overrides) -> dict:
    base = {
        "total_ms": 850,
        "llm_total_ms": 600,
        "llm_calls": 2,
        "tool_total_ms": 200,
        "tool_calls": 2,
        "tool_failures": 1,
        "tool_timeouts": 1,
        "tool_calls_skipped": 0,
        "kb_search_ms": 120,
        "kb_searches": 1,
        "retrieved_chunks": 3,
        "rounds": 2,
        "rounds_exhausted": False,
        "retry_fired": False,
        "failure_directive_fired": True,
        "failure_directive_escalated": False,
        "guard_hallucination_fired": False,
        "guard_no_grounding_fired": False,
        "guard_unverified_data_fired": True,
        "escalated": False,
    }
    base.update(overrides)
    return base


def _tools() -> list[dict]:
    return [
        {"tool_name": "search_knowledge_base", "kind": "kb", "latency_ms": 120,
         "outcome": "ok", "budget_slice_ms": 15000, "round_index": 0},
        {"tool_name": "get_player_wallet", "kind": "crm", "latency_ms": 80,
         "outcome": "ok", "budget_slice_ms": 5000, "round_index": 0},
        {"tool_name": "get_player_transactions", "kind": "crm", "latency_ms": 120,
         "outcome": "timeout", "budget_slice_ms": 5000, "round_index": 1},
    ]


async def test_record_chat_turn_metric_round_trips_parent_and_children(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.chat_turn_metrics.get_sessionmaker", lambda: sessionmaker)

    await record_chat_turn_metric(
        tenant_id="dev",
        crm_id="betstudio",
        session_id="chat_abc123",
        trace_id="trace-xyz",
        path="tools",
        llm_provider="GeminiLLMAdapter",
        llm_model="gemini-2.0-flash",
        action="continue",
        metrics=_metrics_dict(),
        tools=_tools(),
    )

    async with sessionmaker() as db:
        turns = (await db.execute(select(ChatTurnMetric))).scalars().all()
        tools = (await db.execute(select(ChatToolMetricRow))).scalars().all()

    assert len(turns) == 1
    row = turns[0]
    assert row.tenant_id == "dev"
    assert row.crm_id == "betstudio"
    assert row.session_id == "chat_abc123"
    assert row.trace_id == "trace-xyz"
    assert row.path == "tools"
    assert row.llm_provider == "GeminiLLMAdapter"
    assert row.llm_model == "gemini-2.0-flash"
    assert row.action == "continue"
    assert row.total_ms == 850
    assert row.llm_total_ms == 600
    assert row.llm_calls == 2
    assert row.tool_total_ms == 200
    assert row.tool_calls == 2
    assert row.tool_failures == 1
    assert row.tool_timeouts == 1
    assert row.tool_calls_skipped == 0
    assert row.kb_search_ms == 120
    assert row.kb_searches == 1
    assert row.retrieved_chunks == 3
    assert row.rounds == 2
    assert row.rounds_exhausted is False
    assert row.failure_directive_fired is True
    assert row.guard_unverified_data_fired is True
    assert row.created_at is not None

    assert len(tools) == 3
    by_name = {t.tool_name: t for t in tools}
    assert by_name["search_knowledge_base"].kind == "kb"
    assert by_name["search_knowledge_base"].turn_id == row.id
    assert by_name["search_knowledge_base"].tenant_id == "dev"  # denormalized
    assert by_name["get_player_transactions"].outcome == "timeout"
    assert by_name["get_player_transactions"].round_index == 1


async def test_cascade_delete_removes_child_rows(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.chat_turn_metrics.get_sessionmaker", lambda: sessionmaker)

    await record_chat_turn_metric(
        tenant_id="dev", crm_id=None, session_id="chat_1", trace_id=None,
        path="tools", llm_provider="p", llm_model="m", action="continue",
        metrics=_metrics_dict(), tools=_tools(),
    )

    async with sessionmaker() as db:
        turn = (await db.execute(select(ChatTurnMetric))).scalars().one()
        await db.delete(turn)
        await db.commit()

    async with sessionmaker() as db:
        remaining_turns = (await db.execute(select(ChatTurnMetric))).scalars().all()
        remaining_tools = (await db.execute(select(ChatToolMetricRow))).scalars().all()

    assert remaining_turns == []
    assert remaining_tools == []  # CASCADE removed all 3 children


async def test_record_chat_turn_metric_swallows_db_errors(monkeypatch) -> None:
    def _broken_sessionmaker():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("src.models.chat_turn_metrics.get_sessionmaker", _broken_sessionmaker)

    # Must not raise.
    await record_chat_turn_metric(
        tenant_id="dev", crm_id=None, session_id="chat_x", trace_id=None,
        path="single_shot", llm_provider="p", llm_model="m", action="continue",
        metrics=_metrics_dict(), tools=[],
    )


async def test_record_chat_turn_metric_never_raises_logs_warning(monkeypatch, caplog) -> None:
    def _broken_sessionmaker():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("src.models.chat_turn_metrics.get_sessionmaker", _broken_sessionmaker)

    with caplog.at_level("WARNING", logger="src.models.chat_turn_metrics"):
        await record_chat_turn_metric(
            tenant_id="dev", crm_id=None, session_id="chat_x", trace_id=None,
            path="single_shot", llm_provider="p", llm_model="m", action="continue",
            metrics=_metrics_dict(), tools=[],
        )

    assert any("record_chat_turn_metric failed" in rec.message for rec in caplog.records)


async def test_record_chat_turn_metric_truncates_overlong_tool_name(sessionmaker, monkeypatch) -> None:
    """Review fix 3: tool_name is copied straight from the model's own
    tool-call response (src/agents/chatbot.py) -- unlike every other string
    field here, it is NOT bounded by construction, so a hallucinated or
    malformed tool name could exceed the column width. On a real Postgres DB
    an overflow raises StringDataRightTruncation, which -- being swallowed by
    this module's never-raises contract -- would otherwise silently drop the
    ENTIRE row (parent + all its children), not just that one tool's entry.
    Truncating defensively at insert time costs a truncated label instead."""
    monkeypatch.setattr("src.models.chat_turn_metrics.get_sessionmaker", lambda: sessionmaker)

    overlong_name = "x" * 500
    await record_chat_turn_metric(
        tenant_id="dev", crm_id=None, session_id="chat_overlong", trace_id=None,
        path="tools", llm_provider="p", llm_model="m", action="continue",
        metrics=_metrics_dict(),
        tools=[{"tool_name": overlong_name, "kind": "crm", "latency_ms": 10,
                "outcome": "ok", "budget_slice_ms": 100, "round_index": 0}],
    )

    async with sessionmaker() as db:
        turns = (await db.execute(select(ChatTurnMetric))).scalars().all()
        tools = (await db.execute(select(ChatToolMetricRow))).scalars().all()

    assert len(turns) == 1  # the row was NOT dropped
    assert len(tools) == 1
    assert len(tools[0].tool_name) == 100
    assert tools[0].tool_name == overlong_name[:100]
