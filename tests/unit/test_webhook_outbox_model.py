"""Round-trip test for WebhookOutbox / enqueue_webhook_outbox
(src/models/webhook_outbox.py). Mirrors test_chat_turn_metrics_model.py's
shape: an isolated in-memory engine, get_sessionmaker monkeypatched onto the
module under test.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.tenant import Tenant
from src.models.webhook_outbox import STATUS_PENDING, WebhookOutbox, enqueue_webhook_outbox


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(Tenant(id="t1", slug="acme", name="Acme", pipeline_config={}))
        await db.commit()
    yield sm
    await engine.dispose()


async def test_enqueue_writes_a_pending_row(sessionmaker, monkeypatch):
    monkeypatch.setattr("src.models.webhook_outbox.get_sessionmaker", lambda: sessionmaker)

    await enqueue_webhook_outbox(
        tenant_id="t1", session_id="cs_1", event_type="session_closed",
        url="https://crm.example/hook", body={"event": "session_closed", "session_id": "cs_1"},
    )

    async with sessionmaker() as db:
        rows = (await db.execute(select(WebhookOutbox))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.tenant_id == "t1"
    assert row.session_id == "cs_1"
    assert row.event_type == "session_closed"
    assert row.url == "https://crm.example/hook"
    assert row.body == {"event": "session_closed", "session_id": "cs_1"}
    assert row.attempts == 0
    assert row.status == STATUS_PENDING


async def test_enqueue_sets_created_at_in_python_not_db_server_default(sessionmaker, monkeypatch):
    """L1: created_at must be set explicitly in Python (naive UTC), not left
    to the column's server_default=func.now() -- a non-UTC DB server clock
    would otherwise silently shift the 24h dead-cap (measured from
    created_at in src/main.py's run_webhook_outbox_once)."""
    monkeypatch.setattr("src.models.webhook_outbox.get_sessionmaker", lambda: sessionmaker)

    before = datetime.now(timezone.utc).replace(tzinfo=None)
    await enqueue_webhook_outbox(
        tenant_id="t1", session_id="cs_1", event_type="session_closed",
        url="https://crm.example/hook", body={"event": "session_closed"},
    )
    after = datetime.now(timezone.utc).replace(tzinfo=None)

    async with sessionmaker() as db:
        row = (await db.execute(select(WebhookOutbox))).scalars().one()

    assert before <= row.created_at <= after
    # created_at and next_attempt_at come from the SAME Python "now" call --
    # they must match exactly, not merely both be "close to now".
    assert row.created_at == row.next_attempt_at


async def test_enqueue_never_raises_when_db_is_broken(monkeypatch, caplog):
    import logging

    def _broken_sessionmaker():
        raise RuntimeError("db is on fire")

    monkeypatch.setattr("src.models.webhook_outbox.get_sessionmaker", _broken_sessionmaker)

    with caplog.at_level(logging.WARNING):
        await enqueue_webhook_outbox(
            tenant_id="t1", session_id="cs_1", event_type="session_closed",
            url="https://crm.example/hook", body={"event": "session_closed"},
        )

    assert any(
        r.levelno == logging.WARNING and "webhook_outbox enqueue failed" in r.message
        for r in caplog.records
    )
