from __future__ import annotations

import textwrap

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.db_resolver import DbTenantResolver
from src.auth.seed import seed_if_empty, seed_provider_costs
from src.models import Base
from src.models.tenant import ProviderCost


@pytest_asyncio.fixture
async def sm(monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    crypto.reset_cache_for_tests()
    eng = create_async_engine("sqlite+aiosqlite://")
    async with eng.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(eng, expire_on_commit=False)
    await eng.dispose()
    crypto.reset_cache_for_tests()


@pytest.mark.asyncio
async def test_seed_from_yaml_then_resolve(sm, tmp_path, monkeypatch):
    (tmp_path / "demo.yaml").write_text(textwrap.dedent("""
        id: t_demo
        slug: demo
        name: Demo
        timezone: Asia/Kolkata
        max_concurrent_calls: 5
        pipeline:
          mode: layered
          tts: {provider: sarvam, voice_id: anushka, api_key_env: SARVAM_API_KEY}
          telephony: {provider: stringee, from_number: "+91123",
                      account_sid_env: DEMO_SID, auth_token_env: DEMO_TOKEN}
        phone_numbers: ["+91123"]
    """))
    monkeypatch.setenv("VOX_TENANT_DIR", str(tmp_path))
    monkeypatch.setenv("TENANT_DEMO_API_TOKENS", "demo-token")
    monkeypatch.setenv("SARVAM_API_KEY", "master-sarvam")

    assert await seed_if_empty(sm) == 1
    assert await seed_if_empty(sm) == 0          # idempotent — already populated

    r = DbTenantResolver(sm)
    await r.reload()

    ctx = await r.resolve_by_slug("demo")
    assert ctx is not None and ctx.id == "t_demo"
    assert ctx.settings.max_concurrent_calls == 5
    assert ctx.settings.pipeline.tts.voice_id == "anushka"
    assert ctx.settings.pipeline.telephony.provider == "stringee"
    assert ctx.secret("SARVAM_API_KEY") == "master-sarvam"        # master env
    assert (await r.resolve_by_token(hash_api_token("demo-token"))).slug == "demo"
    assert (await r.resolve_by_phone_number("+91123")).slug == "demo"


@pytest.mark.asyncio
async def test_seed_provider_costs_inserts_missing_preserves_existing(sm, tmp_path):
    costs_yaml = tmp_path / "costs.yaml"
    # nested = per-model (llm); scalar = provider-level model="" (telephony).
    costs_yaml.write_text(textwrap.dedent("""
        llm:
          gemini:
            gemini-2.5-flash: 0.002
            gemini-2.5-pro: 0.012
        telephony: {twilio: 0.014, exotel: 0.007}
    """))

    # First seed inserts all four rows (2 model-level llm + 2 telephony).
    assert await seed_provider_costs(sm, costs_yaml) == 4
    async with sm() as s:
        # model-level row keyed by (kind, provider, model)
        assert (await s.get(ProviderCost, ("llm", "gemini", "gemini-2.5-pro"))).cost_per_min == 0.012
        # telephony stored with empty model
        assert (await s.get(ProviderCost, ("telephony", "twilio", ""))).cost_per_min == 0.014

    # An admin edits the twilio rate after seeding.
    async with sm() as s:
        row = await s.get(ProviderCost, ("telephony", "twilio", ""))
        row.cost_per_min = 0.99
        await s.commit()

    # Re-seeding is insert-missing-only: nothing new, the edited rate is preserved.
    assert await seed_provider_costs(sm, costs_yaml) == 0
    async with sm() as s:
        assert (await s.get(ProviderCost, ("telephony", "twilio", ""))).cost_per_min == 0.99


@pytest.mark.asyncio
async def test_seed_provider_costs_missing_file_is_noop(sm, tmp_path):
    assert await seed_provider_costs(sm, tmp_path / "does-not-exist.yaml") == 0


@pytest.mark.asyncio
async def test_seed_provider_costs_upserts_llm_token_rates(sm, tmp_path):
    """``llm_token_rates`` is not a `kind` — it must be excluded from the
    per-minute loop and instead upsert cost_per_1k_input/output_tokens onto
    the matching (kind='llm', provider, model) row, creating it if the
    per-minute pass above didn't already seed one."""
    costs_yaml = tmp_path / "costs.yaml"
    costs_yaml.write_text(textwrap.dedent("""
        llm:
          gemini:
            gemini-3.5-flash: 0.002
        llm_token_rates:
          gemini:
            gemini-3.5-flash: {input_per_1k: 0.0003, output_per_1k: 0.0025}
          groq:
            llama-3.1-8b-instant: {input_per_1k: 0.00005, output_per_1k: 0.00008}
    """))

    # 1 per-minute row (gemini/gemini-3.5-flash, matched by token rates) +
    # 1 new per-token-only row (groq/llama-3.1-8b-instant, no per-minute entry).
    assert await seed_provider_costs(sm, costs_yaml) == 2
    async with sm() as s:
        gem = await s.get(ProviderCost, ("llm", "gemini", "gemini-3.5-flash"))
        assert gem.cost_per_min == 0.002                     # per-minute pass
        assert gem.cost_per_1k_input_tokens == pytest.approx(0.0003)
        assert gem.cost_per_1k_output_tokens == pytest.approx(0.0025)

        groq = await s.get(ProviderCost, ("llm", "groq", "llama-3.1-8b-instant"))
        assert groq is not None
        assert groq.cost_per_min == 0.0                      # never in the per-minute YAML
        assert groq.cost_per_1k_input_tokens == pytest.approx(0.00005)
        assert groq.cost_per_1k_output_tokens == pytest.approx(0.00008)

    # An admin edits the token rate after seeding — re-seeding UPSERTS (unlike
    # the per-minute pass, which is insert-missing-only): the YAML value wins.
    async with sm() as s:
        row = await s.get(ProviderCost, ("llm", "gemini", "gemini-3.5-flash"))
        row.cost_per_1k_input_tokens = 0.99
        await s.commit()
    assert await seed_provider_costs(sm, costs_yaml) == 0     # no new rows
    async with sm() as s:
        gem = await s.get(ProviderCost, ("llm", "gemini", "gemini-3.5-flash"))
        assert gem.cost_per_1k_input_tokens == pytest.approx(0.0003)  # reset from YAML


@pytest.mark.asyncio
async def test_seed_provider_costs_real_yaml_seeds_llm_token_fallback_row(sm):
    """The real config/provider_costs.yaml must seed a provider-level ("")
    fallback row for llm_token_rates/gemini, not just the exact-model row —
    otherwise token_rates()'s fallback lookup in src/api/chat_cost.py is dead
    in production and chat cost silently goes to $0 the moment
    pipeline.llm.model points at any gemini variant other than
    gemini-3.5-flash."""
    await seed_provider_costs(sm)   # default path = config/provider_costs.yaml
    async with sm() as s:
        fallback = await s.get(ProviderCost, ("llm", "gemini", ""))
        assert fallback is not None
        assert fallback.cost_per_1k_input_tokens > 0
        assert fallback.cost_per_1k_output_tokens > 0
