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
from src.models.turn_metrics import TurnMetric, record_turn_metric

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
                    total_latency_ms=4300, tts_segments_dropped=1,
                ),
                TurnMetric(
                    tenant_id="dev", session_id="c1", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="GeminiLLMAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=280, llm_ttft_ms=1400,
                    llm_total_ms=4200, tts_first_chunk_ms=2100, tts_total_ms=2600,
                    total_latency_ms=4500, tts_segments_dropped=0,
                ),
                TurnMetric(
                    tenant_id="dev", session_id="c2", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="AnthropicClaudeAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=300, llm_ttft_ms=0,
                    llm_total_ms=5000, tts_first_chunk_ms=1900, tts_total_ms=2400,
                    total_latency_ms=5300, tts_segments_dropped=0,
                ),
                TurnMetric(
                    tenant_id="dev", session_id="s2s1", campaign_id="bharat_matka",
                    mode="s2s", stt_provider=None,
                    llm_provider="GeminiLiveSession", tts_provider=None,
                    action="continue", stt_latency_ms=0, llm_ttft_ms=0,
                    llm_total_ms=0, tts_first_chunk_ms=1400, tts_total_ms=0,
                    total_latency_ms=3800, tts_segments_dropped=0,
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
        # Exposed so tests can seed rows through the real record_turn_metric
        # helper (monkeypatching its get_sessionmaker) against this SAME
        # in-memory engine/sessionmaker, instead of only the fixture's direct
        # TurnMetric(...) seeding above.
        c.sm = sm  # type: ignore[attr-defined]
        yield c
    set_admin_tokens([])
    await engine.dispose()


async def test_summary_groups_by_combo(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    combos = {(e["stt_provider"], e["llm_provider"], e["tts_provider"]): e for e in body["entries"]}
    # 2 layered combos + 1 s2s combo (distinct stt/llm/tts triplet even
    # without considering `mode`); mode-specific separation is covered by
    # test_summary_includes_mode_and_separates_s2s_combo below.
    assert len(combos) == 3

    gemini_combo = combos[("GroqSTTAdapter", "GeminiLLMAdapter", "SarvamTTSAdapter")]
    assert gemini_combo["samples"] == 2
    assert gemini_combo["avg_total_latency_ms"] == 4400.0  # (4300 + 4500) / 2
    assert gemini_combo["avg_tts_segments_dropped"] == 0.5  # (1 + 0) / 2

    claude_combo = combos[("GroqSTTAdapter", "AnthropicClaudeAdapter", "SarvamTTSAdapter")]
    assert claude_combo["samples"] == 1
    assert claude_combo["avg_total_latency_ms"] == 5300.0


async def test_summary_includes_mode_and_separates_s2s_combo(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    combos = {
        (e["mode"], e["stt_provider"], e["llm_provider"], e["tts_provider"]): e
        for e in body["entries"]
    }
    assert len(combos) == 3  # the 2 pre-existing layered combos + 1 new s2s combo

    s2s_combo = combos[("s2s", None, "GeminiLiveSession", None)]
    assert s2s_combo["samples"] == 1
    assert s2s_combo["avg_tts_first_chunk_ms"] == 1400.0
    assert s2s_combo["avg_total_latency_ms"] == 3800.0


async def test_summary_requires_admin(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary")
    assert resp.status_code == 401


async def test_record_turn_metric_then_summary_e2e(
    client: AsyncClient, monkeypatch
) -> None:
    """End-to-end: record_turn_metric (the real helper VoiceBotAgent.apply_signal
    calls in production, via src/bootstrap.py's
    ``record_metric=lambda payload: record_turn_metric(tenant_id=tenant.id, **payload)``)
    actually produces a row that /benchmarks/turn-metrics/summary can find and
    aggregate. Everything else in this file substitutes a fake callback or
    seeds rows via direct TurnMetric(...) construction; this is the one test
    that exercises the real chain end to end."""
    monkeypatch.setattr(
        "src.models.turn_metrics.get_sessionmaker", lambda: client.sm  # type: ignore[attr-defined]
    )

    # Exactly the 8 keys VoiceBotAgent.apply_signal builds for _record_metric
    # (src/agents/voicebot.py, the ``if self._record_metric is not None:`` block).
    payload = {
        "session_id": "e2e_call_1",
        "campaign_id": "e2e_campaign",
        "mode": "layered",
        "stt_provider": "WhisperSTTAdapter",
        "llm_provider": "OpenAILLMAdapter",
        "tts_provider": "ElevenLabsTTSAdapter",
        "action": "continue",
        "metrics": {
            "stt_latency_ms": 250,
            "llm_ttft_ms": 900,
            "llm_total_ms": 3000,
            "tts_first_chunk_ms": 1500,
            "tts_total_ms": 1800,
            "total_latency_ms": 3300,
            "tts_segments_dropped": 2,
        },
    }

    # Mirrors the real production call site exactly:
    # record_turn_metric(tenant_id=tenant.id, **payload)
    await record_turn_metric(tenant_id="dev", **payload)

    resp = await client.get("/benchmarks/turn-metrics/summary", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    combos = {(e["stt_provider"], e["llm_provider"], e["tts_provider"]): e for e in body["entries"]}
    entry = combos[("WhisperSTTAdapter", "OpenAILLMAdapter", "ElevenLabsTTSAdapter")]
    assert entry["samples"] == 1
    assert entry["avg_stt_latency_ms"] == 250.0
    assert entry["avg_llm_ttft_ms"] == 900.0
    assert entry["avg_llm_total_ms"] == 3000.0
    assert entry["avg_tts_first_chunk_ms"] == 1500.0
    assert entry["avg_tts_total_ms"] == 1800.0
    assert entry["avg_total_latency_ms"] == 3300.0
    assert entry["avg_tts_segments_dropped"] == 2.0
