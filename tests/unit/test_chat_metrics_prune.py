"""Tests for the chat_turn_metrics retention prune job (turn-metrics plan,
Phase 3, §11.2). Mirrors tests/unit/test_call_store.py's
test_reap_stale_calls_closes_only_old_active style: test the pure logic
function directly, not the infinite `_*_loop` wrapper (see
src/main.py::prune_chat_turn_metrics's own docstring on why it's split out
that way, same as reap_stale_calls vs. _reap_stale_calls_loop).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import src.main as main
from src.models.chat_turn_metrics import ChatToolMetricRow, ChatTurnMetric
from src.models.database import Base
from src.models.tenant import Tenant

RETENTION_DAYS = 90


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        # Foreign-key enforcement is off by default on sqlite -- turn it on so
        # the CASCADE-delete assertion below actually exercises the real FK,
        # not just app-level plumbing (mirrors
        # test_chat_turn_metrics_model.py's identical fixture).
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(Tenant(id="dev", slug="dev", name="Dev Tenant"))
        await db.commit()
    yield sm
    await engine.dispose()


async def _seed_turn(sessionmaker, *, created_at: datetime, with_tool: bool = False) -> int:
    async with sessionmaker() as db:
        row = ChatTurnMetric(
            tenant_id="dev", session_id="cs_1", path="tools",
            llm_provider="GeminiLLMAdapter", llm_model="gemini-2.0-flash",
            action="continue", created_at=created_at,
        )
        db.add(row)
        await db.flush()
        if with_tool:
            db.add(ChatToolMetricRow(
                turn_id=row.id, tenant_id="dev", tool_name="get_player_wallet",
                kind="crm", outcome="ok", created_at=created_at,
            ))
        await db.commit()
        return row.id


async def test_prune_deletes_rows_older_than_window_keeps_newer(sessionmaker) -> None:
    now = datetime.utcnow()
    old_id = await _seed_turn(sessionmaker, created_at=now - timedelta(days=RETENTION_DAYS + 1))
    new_id = await _seed_turn(sessionmaker, created_at=now - timedelta(days=RETENTION_DAYS - 1))

    n = await main.prune_chat_turn_metrics(sessionmaker, RETENTION_DAYS)

    assert n == 1
    async with sessionmaker() as db:
        remaining_ids = set((await db.execute(select(ChatTurnMetric.id))).scalars().all())
    assert remaining_ids == {new_id}
    assert old_id not in remaining_ids


async def test_prune_cascades_to_child_tool_rows(sessionmaker) -> None:
    now = datetime.utcnow()
    old_id = await _seed_turn(
        sessionmaker, created_at=now - timedelta(days=RETENTION_DAYS + 1), with_tool=True,
    )

    n = await main.prune_chat_turn_metrics(sessionmaker, RETENTION_DAYS)

    assert n == 1
    async with sessionmaker() as db:
        remaining_tools = (await db.execute(
            select(ChatToolMetricRow).where(ChatToolMetricRow.turn_id == old_id)
        )).scalars().all()
    assert remaining_tools == []  # CASCADE removed the child along with its parent


async def test_prune_noop_when_nothing_is_old_enough(sessionmaker) -> None:
    await _seed_turn(sessionmaker, created_at=datetime.utcnow())

    n = await main.prune_chat_turn_metrics(sessionmaker, RETENTION_DAYS)

    assert n == 0
    async with sessionmaker() as db:
        assert (await db.execute(select(ChatTurnMetric))).scalars().all()


async def test_prune_deletes_in_bounded_batches(sessionmaker, monkeypatch) -> None:
    """25 old rows with a batch size of 10 must take 3 round trips (10+10+5),
    not one unbounded DELETE -- proving the batching loop actually iterates
    rather than the whole retention window being deleted in a single
    statement."""
    monkeypatch.setattr(main, "_CHAT_METRICS_PRUNE_BATCH_SIZE", 10)

    now = datetime.utcnow()
    old_created_at = now - timedelta(days=RETENTION_DAYS + 1)
    for _ in range(25):
        await _seed_turn(sessionmaker, created_at=old_created_at)

    call_count = 0
    real_sessionmaker_call = sessionmaker.__call__

    class _CountingSessionmaker:
        def __call__(self):
            nonlocal call_count
            call_count += 1
            return real_sessionmaker_call()

    n = await main.prune_chat_turn_metrics(_CountingSessionmaker(), RETENTION_DAYS)

    assert n == 25
    # 1 (dialect probe, to pick the cutoff expression -- see the function's
    # own docstring) + 3 batches (10 + 10 + 5) -- the loop actually batched,
    # not one giant delete.
    assert call_count == 4
    async with sessionmaker() as db:
        assert (await db.execute(select(ChatTurnMetric))).scalars().all() == []
