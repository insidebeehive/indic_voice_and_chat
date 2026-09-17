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
            # Provider-level fallback row: no cached rate configured (left at
            # the column's default None) -- deliberately, to exercise the
            # "unconfigured" path distinctly from the model-level row below.
            ProviderCost(kind="llm", provider="gemini", model="",
                         cost_per_1k_input_tokens=0.0001, cost_per_1k_output_tokens=0.0005),
            # Model-level row WITH a configured cached rate.
            ProviderCost(kind="llm", provider="gemini", model="gemini-3.5-flash",
                         cost_per_1k_input_tokens=0.0003, cost_per_1k_output_tokens=0.0025,
                         cost_per_1k_cached_tokens=0.0001),
            # A second provider with NO cached rate configured anywhere (not
            # even a fallback row) -- the clean "never configured" case for
            # the fallback-to-input-rate tests below.
            ProviderCost(kind="llm", provider="nocache", model="model-x",
                         cost_per_1k_input_tokens=0.001, cost_per_1k_output_tokens=0.002),
        ])
        await s.commit()
    yield maker
    await engine.dispose()


async def test_token_rates_uses_model_level_row(sm):
    async with sm() as s:
        in_rate, out_rate, cached_rate = await token_rates(s, "gemini", "gemini-3.5-flash")
    assert in_rate == pytest.approx(0.0003)
    assert out_rate == pytest.approx(0.0025)
    assert cached_rate == pytest.approx(0.0001)


async def test_token_rates_falls_back_to_provider_level(sm):
    async with sm() as s:
        in_rate, out_rate, cached_rate = await token_rates(s, "gemini", "gemini-9-ultra")
    assert in_rate == pytest.approx(0.0001)
    assert out_rate == pytest.approx(0.0005)
    # The provider-level fallback row has no cached rate configured -- must
    # come back None, not 0.0 (see compute_chat_turn_cost's fallback).
    assert cached_rate is None


async def test_token_rates_unknown_provider_returns_zero(sm):
    async with sm() as s:
        in_rate, out_rate, cached_rate = await token_rates(s, "unknown-provider", "some-model")
    assert (in_rate, out_rate, cached_rate) == (0.0, 0.0, None)


async def test_compute_chat_turn_cost_math(sm):
    async with sm() as s:
        cost = await compute_chat_turn_cost(
            s, provider="gemini", model="gemini-3.5-flash",
            input_tokens=1000, output_tokens=500,
        )
    # No cached_tokens passed (defaults to 0) -- identical to the pre-cache
    # arithmetic: 1000/1000 * 0.0003 + 500/1000 * 0.0025 = 0.00155, even
    # though this row DOES have a configured cached rate. Proves a caller
    # with no cached figure gets today's behaviour unchanged.
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


async def test_compute_chat_turn_cost_split_rate_arithmetic(sm):
    """The core new arithmetic: fresh and cached input tokens must bill at
    DIFFERENT rates. gemini-3.5-flash: in=0.0003/1k, out=0.0025/1k,
    cached=0.0001/1k (fixture above). 1000 input tokens, 400 of them cached,
    200 output:
        fresh   = (1000-400)/1000 * 0.0003 = 0.00018
        cached  =      400 /1000 * 0.0001 = 0.00004
        output  =      200 /1000 * 0.0025 = 0.0005
        total   = 0.00072
    Also asserts this is cheaper than billing every input token at the fresh
    rate (0.0003 + 0.0005 = 0.0008) -- if the cached/fresh split collapsed
    back into "bill everything at in_rate" this second assertion fails even
    though a lazier version of the first could be tuned to still pass."""
    async with sm() as s:
        cost = await compute_chat_turn_cost(
            s, provider="gemini", model="gemini-3.5-flash",
            input_tokens=1000, output_tokens=200, cached_tokens=400,
        )
    assert cost == pytest.approx(0.00072)
    uncached_equivalent = (1000 / 1000 * 0.0003) + (200 / 1000 * 0.0025)
    assert cost < uncached_equivalent


async def test_compute_chat_turn_cost_unset_cached_rate_bills_at_full_input_rate(sm):
    """THE important test. (nocache, model-x) has no cost_per_1k_cached_tokens
    row at all. Reported cached_tokens must make ZERO difference to the bill
    when no cached rate is configured -- cached tokens fall back to the FULL
    input rate, never to 0.0. Mutating compute_chat_turn_cost's fallback from
    `cached_rate if cached_rate is not None else in_rate` to
    `cached_rate or 0.0` was run by hand against this test and it failed
    (0.0003 instead of 0.001) -- see the report for the exact diff used."""
    async with sm() as s:
        cost_with_cached = await compute_chat_turn_cost(
            s, provider="nocache", model="model-x",
            input_tokens=1000, output_tokens=0, cached_tokens=700,
        )
        cost_with_no_cached_reported = await compute_chat_turn_cost(
            s, provider="nocache", model="model-x",
            input_tokens=1000, output_tokens=0, cached_tokens=0,
        )
    assert cost_with_cached == pytest.approx(0.001)          # all 1000 @ in_rate
    assert cost_with_cached == cost_with_no_cached_reported   # cached_tokens was a no-op


async def test_compute_chat_turn_cost_clamps_cached_tokens_exceeding_input(sm, caplog):
    """A provider report with cached_tokens > input_tokens (should never
    happen per ChatTurnMetric's own docstring, but nothing upstream enforces
    it) must clamp to input_tokens, not go negative. Without the clamp,
    fresh_tokens = 100 - 1000 = -900, giving cost = -900/1000*0.0003 +
    1000/1000*0.0001 = -0.00017 -- a NEGATIVE bill that gets cheaper as the
    bogus cached figure grows, the opposite of what a real cache hit does.
    Mutating the clamp's `if cached_tokens > input_tokens` out entirely was
    run by hand against this test and it failed on the `cost >= 0` assertion
    with exactly that negative value -- see the report."""
    async with sm() as s:
        with caplog.at_level("WARNING", logger="src.api.chat_cost"):
            cost = await compute_chat_turn_cost(
                s, provider="gemini", model="gemini-3.5-flash",
                input_tokens=100, output_tokens=0, cached_tokens=1000,
            )
    assert cost >= 0
    # Clamped: all 100 input tokens treated as cached -- 100/1000 * 0.0001.
    assert cost == pytest.approx(0.00001)
    assert any("exceeds input_tokens" in rec.message for rec in caplog.records)


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

        in_rate, out_rate, _cached_rate = await token_rates(s, "gemini", "gemini-2.5-flash")
    assert in_rate > 0.0
    assert out_rate > 0.0


async def test_token_rates_real_yaml_seeds_gemini_3_5_flash_cached_rate(real_seeded_sm):
    """The real config/provider_costs.yaml seeds an authoritative cached rate
    for gemini/gemini-3.5-flash (see that file's comment for the source and
    date) -- this is the "feature is live, not dormant" requirement: without
    this row, GEMINI_EXPLICIT_CACHE reporting near-100% cache hits would
    still bill every one of those tokens at the full input rate."""
    async with real_seeded_sm() as s:
        in_rate, out_rate, cached_rate = await token_rates(s, "gemini", "gemini-3.5-flash")
    assert in_rate > 0.0
    assert out_rate > 0.0
    assert cached_rate is not None
    assert 0.0 < cached_rate < in_rate  # a cache hit must be cheaper than a fresh token


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
