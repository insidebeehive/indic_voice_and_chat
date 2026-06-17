"""Unit tests for the call-record persistence helpers."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api.call_store import (
    compute_call_cost,
    count_active_calls,
    deliver_to_persister,
    record_outcome,
    set_call_outcome_persister,
)
from src.models.database import Base
from src.models.conversation import Conversation
from src.models.tenant import ProviderCost, Tenant


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        # catalog: provider-level ("") rows + a per-model llm rate
        s.add_all([
            ProviderCost(kind="stt", provider="groq", cost_per_min=0.01),
            ProviderCost(kind="llm", provider="gemini", cost_per_min=0.02),       # provider-level
            ProviderCost(kind="llm", provider="gemini", model="gemini-2.5-pro", cost_per_min=0.50),
            ProviderCost(kind="tts", provider="sarvam", cost_per_min=0.03),
            ProviderCost(kind="telephony", provider="twilio", cost_per_min=0.10),
            ProviderCost(kind="s2s", provider="gemini_live", cost_per_min=0.50),
        ])
        await s.commit()
    yield maker
    await engine.dispose()


def _conv(**over):
    base = dict(
        id="call_1", tenant_id="t1", agent_type="voicebot", channel="voice",
        status="in_progress", pipeline_config={}, provider_call_sid="SID-1",
        mode="layered", stt_provider="groq", llm_provider="gemini",
        tts_provider="sarvam", telephony_provider="twilio",
    )
    base.update(over)
    return Conversation(**base)


async def test_compute_cost_layered_excludes_telephony(sm):
    async with sm() as s:
        # 2 min layered, platform only (telephony NOT billed): (0.01+0.02+0.03)*2 = 0.12
        cost = await compute_call_cost(
            s, mode="layered", stt_provider="groq", llm_provider="gemini",
            tts_provider="sarvam", telephony_provider="twilio", duration_ms=120_000)
    assert cost == pytest.approx(0.12)


async def test_compute_cost_s2s_excludes_telephony(sm):
    async with sm() as s:
        # 1 min s2s, platform only: 0.50 (telephony excluded)
        cost = await compute_call_cost(
            s, mode="s2s", realtime_provider="gemini_live",
            telephony_provider="twilio", duration_ms=60_000)
    assert cost == pytest.approx(0.50)


async def test_compute_cost_uses_model_rate(sm):
    async with sm() as s:
        # gemini-2.5-pro rate: 1 min = stt 0.01 + llm-pro 0.50 + tts 0.03 = 0.54
        cost = await compute_call_cost(
            s, mode="layered", stt_provider="groq", llm_provider="gemini",
            llm_model="gemini-2.5-pro", tts_provider="sarvam",
            telephony_provider="twilio", duration_ms=60_000)
    assert cost == pytest.approx(0.54)


async def test_compute_cost_unknown_model_falls_back_to_provider(sm):
    async with sm() as s:
        # unpriced model falls back to gemini provider-level 0.02: 0.01+0.02+0.03 = 0.06
        cost = await compute_call_cost(
            s, mode="layered", stt_provider="groq", llm_provider="gemini",
            llm_model="gemini-9-ultra", tts_provider="sarvam",
            telephony_provider="twilio", duration_ms=60_000)
    assert cost == pytest.approx(0.06)


async def test_compute_cost_zero_duration(sm):
    async with sm() as s:
        assert await compute_call_cost(
            s, mode="layered", telephony_provider="twilio", duration_ms=0) == 0.0


async def test_telephony_tentative_cost(sm):
    from src.api.call_store import telephony_tentative_cost
    async with sm() as s:
        # telephony twilio 0.10/min * 2 min = 0.20 — tentative, never in the total
        assert await telephony_tentative_cost(s, "twilio", 120_000) == pytest.approx(0.20)
        assert await telephony_tentative_cost(s, None, 120_000) == 0.0
        assert await telephony_tentative_cost(s, "twilio", 0) == 0.0


async def test_insert_call_snapshots_config(sm):
    from src.api.call_store import insert_call
    from src.auth.context import TenantContext
    from src.config_tenant import (
        TenantLLMConfig, TenantPipelineConfig, TenantSettings, TenantSTTConfig,
        TenantTelephonyConfig, TenantTTSConfig,
    )
    ctx = TenantContext(settings=TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            stt=TenantSTTConfig(provider="groq", model="whisper-large-v3"),
            llm=TenantLLMConfig(provider="gemini", model="gemini-2.5-flash"),
            tts=TenantTTSConfig(provider="sarvam", model="bulbul:v2", voice_id="anushka"),
            telephony=TenantTelephonyConfig(provider="webconsole"))))
    async with sm() as s:
        await insert_call(s, call_id="call_x", tenant=ctx, provider_call_sid="call_x",
                          channel="webconsole")
    async with sm() as s:
        got = await s.get(Conversation, "call_x")
    assert got.channel == "webconsole"
    assert got.status == "in_progress"
    assert (got.stt_provider, got.llm_provider, got.tts_provider) == ("groq", "gemini", "sarvam")
    assert got.telephony_provider == "webconsole"
    assert got.voice == "anushka"
    assert got.pipeline_config["tts"]["model"] == "bulbul:v2"


async def test_insert_call_mode_override_records_s2s(sm):
    # A layered-default tenant run through the S2S browser path must be recorded
    # as s2s (with the realtime provider), not layered — drives correct cost.
    from src.api.call_store import insert_call
    from src.auth.context import TenantContext
    from src.config_tenant import (
        TenantPipelineConfig, TenantRealtimeConfig, TenantSettings, TenantTTSConfig,
    )
    ctx = TenantContext(settings=TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            mode="layered",                       # tenant default is layered
            tts=TenantTTSConfig(provider="sarvam", voice_id="anushka"),
            realtime=TenantRealtimeConfig(provider="gemini_live",
                                          model="gemini-3.1-flash-live-preview", voice="Aoede"))))
    async with sm() as s:
        await insert_call(s, call_id="call_s2s", tenant=ctx, provider_call_sid="call_s2s",
                          channel="webconsole", mode="s2s")
    async with sm() as s:
        got = await s.get(Conversation, "call_s2s")
    assert got.mode == "s2s"
    assert got.realtime_provider == "gemini_live"


async def test_count_active_calls(sm):
    async with sm() as s:
        s.add(_conv(id="c1", provider_call_sid="s1", status="in_progress"))
        s.add(_conv(id="c2", provider_call_sid="s2", status="answered"))
        s.add(_conv(id="c3", provider_call_sid="s3", status="ended"))
        await s.commit()
        assert await count_active_calls(s, "t1") == 2


async def test_record_outcome_writes_and_computes_cost(sm):
    async with sm() as s:
        s.add(_conv(id="c1", provider_call_sid="SID-9"))
        await s.commit()
    async with sm() as s:
        row = await record_outcome(
            s, "SID-9", outcome="interested", summary="Wants a callback",
            notes="Prefers evenings", duration_ms=120_000)
    assert row is not None
    assert row.status == "ended"
    assert row.outcome == "interested"
    assert row.summary == "Wants a callback"
    assert row.duration_ms == 120_000
    # platform cost only (stt+llm+tts) — telephony excluded: (0.01+0.02+0.03)*2
    assert row.cost == pytest.approx(0.12)
    assert row.ended_at is not None


async def test_record_outcome_unknown_sid_returns_none(sm):
    async with sm() as s:
        assert await record_outcome(s, "missing-sid", outcome="x") is None


async def test_deliver_to_persister_writes_outcome(sm):
    """Simulates a bridge teardown handing its outcome payload to the wired
    persister, which writes the conversations row (the main.py wiring shape)."""
    async with sm() as s:
        s.add(_conv(id="c1", provider_call_sid="SID-P"))
        await s.commit()

    async def _persister(call_sid, payload):
        async with sm() as s:
            await record_outcome(
                s, call_sid, outcome=payload.get("outcome"),
                summary=payload.get("summary"), notes=payload.get("notes"))

    set_call_outcome_persister(_persister)
    try:
        await deliver_to_persister("SID-P", {
            "outcome": "interested", "summary": "Wants a callback", "notes": "Evenings"})
    finally:
        set_call_outcome_persister(None)

    async with sm() as s:
        from sqlalchemy import select
        got = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "SID-P")
        )).scalar_one()
    assert got.status == "ended"
    assert got.outcome == "interested"
    assert got.summary == "Wants a callback"
    # Cost derived (duration from started_at -> ended_at; tiny but >= 0).
    assert got.cost is not None and got.cost >= 0.0


async def test_deliver_to_persister_noop_without_persister(sm):
    # Unset persister + missing sid must both be safe no-ops (never raise).
    set_call_outcome_persister(None)
    await deliver_to_persister("any-sid", {"outcome": "x"})
    await deliver_to_persister(None, {"outcome": "x"})


async def test_deliver_to_persister_swallows_errors(sm):
    async def _boom(call_sid, payload):
        raise RuntimeError("db down")

    set_call_outcome_persister(_boom)
    try:
        # Must not propagate — teardown has to survive a failing persister.
        await deliver_to_persister("sid", {"outcome": "x"})
    finally:
        set_call_outcome_persister(None)


async def test_reap_stale_calls_closes_only_old_active(sm):
    from datetime import datetime, timedelta

    from sqlalchemy import select

    from src.api.call_store import reap_stale_calls
    from src.models.conversation import Conversation

    old = datetime.utcnow() - timedelta(hours=2)
    async with sm() as s:
        s.add_all([
            _conv(id="c_old", provider_call_sid="old", status="in_progress", started_at=old),
            _conv(id="c_answered_old", provider_call_sid="ans", status="answered", started_at=old),
            _conv(id="c_recent", provider_call_sid="recent", status="in_progress",
                  started_at=datetime.utcnow()),
            _conv(id="c_ended", provider_call_sid="end", status="ended", started_at=old),
        ])
        await s.commit()

        n = await reap_stale_calls(s, older_than_minutes=30)
        assert n == 2   # only the two OLD active rows

        rows = {r.id: r for r in (await s.execute(select(Conversation))).scalars().all()}
        assert rows["c_old"].status == "ended" and rows["c_old"].ended_at is not None
        assert "auto-closed" in (rows["c_old"].notes or "")
        assert rows["c_answered_old"].status == "ended"
        assert rows["c_recent"].status == "in_progress"   # too recent — untouched
        assert rows["c_ended"].status == "ended"          # already ended — untouched

        # Idempotent: a second sweep closes nothing new.
        assert await reap_stale_calls(s, older_than_minutes=30) == 0
