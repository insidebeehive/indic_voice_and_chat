"""Unit tests for the chat (text) per-token cost helpers."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api.chat_cost import compute_chat_turn_cost, token_rates
from src.auth.seed import seed_provider_costs
from src.models.database import Base
from src.models.tenant import ProviderCost


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add_all([
            ProviderCost(kind="llm", provider="gemini", model="",
                         cost_per_1k_input_tokens=0.0001, cost_per_1k_output_tokens=0.0005),
            ProviderCost(kind="llm", provider="gemini", model="gemini-3.5-flash",
                         cost_per_1k_input_tokens=0.0003, cost_per_1k_output_tokens=0.0025),
        ])
        await s.commit()
    yield maker
    await engine.dispose()


async def test_token_rates_uses_model_level_row(sm):
    async with sm() as s:
        in_rate, out_rate = await token_rates(s, "gemini", "gemini-3.5-flash")
    assert in_rate == pytest.approx(0.0003)
    assert out_rate == pytest.approx(0.0025)


async def test_token_rates_falls_back_to_provider_level(sm):
    async with sm() as s:
        in_rate, out_rate = await token_rates(s, "gemini", "gemini-9-ultra")
    assert in_rate == pytest.approx(0.0001)
    assert out_rate == pytest.approx(0.0005)


async def test_token_rates_unknown_provider_returns_zero(sm):
    async with sm() as s:
        in_rate, out_rate = await token_rates(s, "unknown-provider", "some-model")
    assert (in_rate, out_rate) == (0.0, 0.0)


async def test_compute_chat_turn_cost_math(sm):
    async with sm() as s:
        cost = await compute_chat_turn_cost(
            s, provider="gemini", model="gemini-3.5-flash",
            input_tokens=1000, output_tokens=500,
        )
    # 1000/1000 * 0.0003 + 500/1000 * 0.0025 = 0.0003 + 0.00125 = 0.00155
    assert cost == pytest.approx(0.00155)


async def test_compute_chat_turn_cost_uses_provider_fallback(sm):
    async with sm() as s:
        cost = await compute_chat_turn_cost(
            s, provider="gemini", model="unpriced-model",
            input_tokens=2000, output_tokens=0,
        )
    # 2000/1000 * 0.0001 (provider-level fallback) = 0.0002
    assert cost == pytest.approx(0.0002)


async def test_compute_chat_turn_cost_zero_tokens(sm):
    async with sm() as s:
        cost = await compute_chat_turn_cost(
            s, provider="gemini", model="gemini-3.5-flash",
            input_tokens=0, output_tokens=0,
        )
    assert cost == 0.0


async def test_compute_chat_turn_cost_no_provider(sm):
    async with sm() as s:
        cost = await compute_chat_turn_cost(
            s, provider="", model="", input_tokens=100, output_tokens=50,
        )
    assert cost == 0.0


@pytest_asyncio.fixture
async def real_seeded_sm():
    """A DB seeded from the ACTUAL config/provider_costs.yaml (not a hand-built
    fixture) via seed_provider_costs — this is the real production shape."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    await seed_provider_costs(maker)   # default path = config/provider_costs.yaml
    yield maker
    await engine.dispose()


async def test_token_rates_real_yaml_fallback_for_zero_rate_model(real_seeded_sm):
    """gemini-2.5-flash has a per-minute (cost_per_min) row seeded from the
    `llm:` block of config/provider_costs.yaml, but NO dedicated entry in
    llm_token_rates — so its ProviderCost row exists with token rates left at
    their 0.0 default. This is the exact production shape that was broken:
    token_rates()'s old fallback only triggered when the exact-model row was
    entirely absent (`row is None`), never when it existed with zero rates.
    Must resolve to the provider-level ("") fallback rate, not (0.0, 0.0)."""
    async with real_seeded_sm() as s:
        # Sanity check: the exact-model row exists but has zero token rates.
        row = await s.get(ProviderCost, ("llm", "gemini", "gemini-2.5-flash"))
        assert row is not None
        assert row.cost_per_1k_input_tokens == 0.0
        assert row.cost_per_1k_output_tokens == 0.0

        in_rate, out_rate = await token_rates(s, "gemini", "gemini-2.5-flash")
    assert in_rate > 0.0
    assert out_rate > 0.0


async def test_compute_chat_turn_cost_warns_when_rate_resolves_to_zero(sm, caplog):
    """A provider/model with no ProviderCost row at all (no exact-model row and
    no provider-level fallback) must resolve to $0 cost *and* log a warning —
    silent $0 billing for real tokens should never pass without a trace."""
    with caplog.at_level("WARNING", logger="src.api.chat_cost"):
        async with sm() as s:
            cost = await compute_chat_turn_cost(
                s, provider="unknown-provider", model="some-model",
                input_tokens=100, output_tokens=50,
            )
    assert cost == 0.0
    assert any(
        "chat cost resolved to $0" in rec.message for rec in caplog.records
    )
