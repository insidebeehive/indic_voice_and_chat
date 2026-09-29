"""Round-trip test for EmbeddingUsage + the record_embedding_usage helper
(chat-cost-widening plan, Phase 2 write path). Mirrors
test_chat_turn_metrics_model.py's shape."""

from __future__ import annotations

import math

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.crm import Crm
from src.models.database import Base
from src.models.embedding_usage import EmbeddingUsage, record_embedding_usage
from src.models.tenant import ProviderCost, Tenant


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(Tenant(id="dev", slug="dev", name="Dev Tenant"))
        db.add(Crm(id="crm1", name="CRM One", base_url="https://crm.example"))
        db.add(ProviderCost(
            kind="embedding", provider="gemini", model="gemini-embedding-001",
            cost_per_1k_input_tokens=0.00015,
        ))
        await db.commit()
    yield sm
    await engine.dispose()


async def test_record_embedding_usage_estimates_cost_from_chars(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.embedding_usage.get_sessionmaker", lambda: sessionmaker)

    row_id = await record_embedding_usage(
        tenant_id="dev", crm_id=None, purpose="ingest",
        provider="gemini", model="gemini-embedding-001", input_chars=4000,
    )

    assert isinstance(row_id, int)
    async with sessionmaker() as db:
        row = await db.get(EmbeddingUsage, row_id)
    assert row is not None
    assert row.tenant_id == "dev"
    assert row.crm_id is None
    assert row.purpose == "ingest"
    assert row.provider == "gemini"
    assert row.model == "gemini-embedding-001"
    assert row.input_chars == 4000
    assert row.tokens is None  # never backfilled -- no explicit tokens passed
    expected = 0.00015 * math.ceil(4000 / 4.0) / 1000.0
    assert row.cost == pytest.approx(expected)


async def test_record_embedding_usage_explicit_tokens_override_chars(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.embedding_usage.get_sessionmaker", lambda: sessionmaker)

    row_id = await record_embedding_usage(
        tenant_id="dev", crm_id=None, purpose="search",
        provider="gemini", model="gemini-embedding-001", input_chars=4000, tokens=2000,
    )

    async with sessionmaker() as db:
        row = await db.get(EmbeddingUsage, row_id)
    assert row.tokens == 2000
    expected = 0.00015 * 2000 / 1000.0
    assert row.cost == pytest.approx(expected)
    # cost computed from the explicit token count, NOT from the char estimate
    # (which would be math.ceil(4000/4) = 1000, a different number).
    assert row.cost != pytest.approx(0.00015 * 1000 / 1000.0)


async def test_record_embedding_usage_crm_scoped_row(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.embedding_usage.get_sessionmaker", lambda: sessionmaker)

    row_id = await record_embedding_usage(
        tenant_id=None, crm_id="crm1", purpose="ingest",
        provider="gemini", model="gemini-embedding-001", input_chars=800,
    )

    async with sessionmaker() as db:
        row = await db.get(EmbeddingUsage, row_id)
    assert row.tenant_id is None
    assert row.crm_id == "crm1"


async def test_record_embedding_usage_swallows_db_errors_and_returns_none(monkeypatch, caplog) -> None:
    def _broken_sessionmaker():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("src.models.embedding_usage.get_sessionmaker", _broken_sessionmaker)

    with caplog.at_level("WARNING", logger="src.models.embedding_usage"):
        rv = await record_embedding_usage(
            tenant_id="dev", crm_id=None, purpose="ingest",
            provider="gemini", model="gemini-embedding-001", input_chars=100,
        )

    assert rv is None
    assert any("record_embedding_usage failed" in rec.message for rec in caplog.records)


async def test_record_embedding_usage_unknown_purpose_writes_nothing(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.embedding_usage.get_sessionmaker", lambda: sessionmaker)

    rv = await record_embedding_usage(
        tenant_id="dev", crm_id=None, purpose="bogus",
        provider="gemini", model="gemini-embedding-001", input_chars=100,
    )

    assert rv is None
    async with sessionmaker() as db:
        rows = (await db.execute(select(EmbeddingUsage))).scalars().all()
    assert rows == []
