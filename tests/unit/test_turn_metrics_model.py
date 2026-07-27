"""Round-trip test for the TurnMetric model + record_turn_metric helper."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.turn_metrics import TurnMetric, record_turn_metric


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()


async def test_record_turn_metric_inserts_row(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.turn_metrics.get_sessionmaker", lambda: sessionmaker)

    await record_turn_metric(
        tenant_id="dev",
        session_id="call_abc123",
        campaign_id="bharat_matka",
        mode="layered",
        stt_provider="GroqSTTAdapter",
        llm_provider="GeminiLLMAdapter",
        tts_provider="SarvamTTSAdapter",
        action="continue",
        metrics={
            "stt_latency_ms": 300,
            "llm_ttft_ms": 1200,
            "llm_total_ms": 4000,
            "tts_first_chunk_ms": 2000,
            "tts_total_ms": 2500,
            "total_latency_ms": 4300,
            "tts_segments_dropped": 1,
        },
    )

    async with sessionmaker() as db:
        rows = (await db.execute(select(TurnMetric))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.tenant_id == "dev"
    assert row.session_id == "call_abc123"
    assert row.campaign_id == "bharat_matka"
    assert row.mode == "layered"
    assert row.stt_provider == "GroqSTTAdapter"
    assert row.llm_provider == "GeminiLLMAdapter"
    assert row.tts_provider == "SarvamTTSAdapter"
    assert row.action == "continue"
    assert row.stt_latency_ms == 300
    assert row.llm_ttft_ms == 1200
    assert row.llm_total_ms == 4000
    assert row.tts_first_chunk_ms == 2000
    assert row.tts_total_ms == 2500
    assert row.total_latency_ms == 4300
    assert row.tts_segments_dropped == 1
    assert row.created_at is not None


async def test_record_turn_metric_swallows_db_errors(monkeypatch) -> None:
    def _broken_sessionmaker():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("src.models.turn_metrics.get_sessionmaker", _broken_sessionmaker)

    # Must not raise.
    await record_turn_metric(
        tenant_id="dev",
        session_id="call_x",
        campaign_id=None,
        mode="layered",
        stt_provider=None,
        llm_provider="GeminiLLMAdapter",
        tts_provider=None,
        action="continue",
        metrics={
            "stt_latency_ms": 0,
            "llm_ttft_ms": 0,
            "llm_total_ms": 0,
            "tts_first_chunk_ms": 0,
            "tts_total_ms": 0,
            "total_latency_ms": 0,
            "tts_segments_dropped": 0,
        },
    )
