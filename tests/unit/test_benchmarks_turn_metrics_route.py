"""Route test for the turn-metrics benchmarking summary endpoint."""

from __future__ import annotations

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import benchmarks
from src.api.deps import get_db_session
from src.auth.middleware import set_admin_tokens
from src.models.database import Base
from src.models.turn_metrics import TurnMetric

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async def _seed():
        async with sm() as session:
            session.add_all([
                TurnMetric(
                    tenant_id="dev", session_id="c1", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="GeminiLLMAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=300, llm_ttft_ms=1200,
                    llm_total_ms=4000, tts_first_chunk_ms=2000, tts_total_ms=2500,
                    total_latency_ms=4300,
                ),
                TurnMetric(
                    tenant_id="dev", session_id="c1", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="GeminiLLMAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=280, llm_ttft_ms=1400,
                    llm_total_ms=4200, tts_first_chunk_ms=2100, tts_total_ms=2600,
                    total_latency_ms=4500,
                ),
                TurnMetric(
                    tenant_id="dev", session_id="c2", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="AnthropicClaudeAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=300, llm_ttft_ms=0,
                    llm_total_ms=5000, tts_first_chunk_ms=1900, tts_total_ms=2400,
                    total_latency_ms=5300,
                ),
            ])
            await session.commit()

    await _seed()

    async def _session_override():
        async with sm() as session:
            yield session

    set_admin_tokens(["admin-token"])
    app = FastAPI()
    app.include_router(benchmarks.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_admin_tokens([])
    await engine.dispose()


async def test_summary_groups_by_combo(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    combos = {(e["stt_provider"], e["llm_provider"], e["tts_provider"]): e for e in body["entries"]}
    assert len(combos) == 2

    gemini_combo = combos[("GroqSTTAdapter", "GeminiLLMAdapter", "SarvamTTSAdapter")]
    assert gemini_combo["samples"] == 2
    assert gemini_combo["avg_total_latency_ms"] == 4400.0  # (4300 + 4500) / 2

    claude_combo = combos[("GroqSTTAdapter", "AnthropicClaudeAdapter", "SarvamTTSAdapter")]
    assert claude_combo["samples"] == 1
    assert claude_combo["avg_total_latency_ms"] == 5300.0


async def test_summary_requires_admin(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary")
    assert resp.status_code == 401
