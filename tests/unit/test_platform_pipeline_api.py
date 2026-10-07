"""Route tests for src/api/platform.py -- the admin-only platform pipeline-
defaults endpoints (GET/PUT/DELETE /api/v1/platform/pipeline...)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import platform
from src.api.deps import get_db_session
from src.auth.audit import reset_suppression_state
from src.auth.middleware import set_admin_tokens
from src.models.database import Base

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest.fixture(autouse=True)
def _reset_audit_suppression_state():
    reset_suppression_state()
    yield
    reset_suppression_state()


@pytest_asyncio.fixture
async def ctx():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async def _session_override():
        async with sm() as session:
            yield session

    set_admin_tokens(["admin-token"])

    app = FastAPI()
    app.include_router(platform.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, app
    set_admin_tokens([])
    await engine.dispose()


# --- GET: yaml-sourced by default ----------------------------------------


async def test_get_pipeline_defaults_to_yaml_source(ctx) -> None:
    client, _ = ctx
    resp = await client.get("/platform/pipeline", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    layers = {layer["layer"]: layer for layer in body["layers"]}
    assert set(layers) == {"stt", "llm", "tts"}
    for layer in layers.values():
        assert layer["source"] == "yaml"
        assert layer["effective"] == layer["yaml"]
        assert layer["updated_at"] is None
        assert layer["updated_by"] is None
    # Sanity: reflects config/default.yaml's actual platform defaults.
    assert layers["llm"]["effective"]["provider"] == "gemini"
    assert layers["stt"]["effective"]["provider"] == "sarvam"


async def test_get_effective_reads_live_registry_not_db_row_when_wired(ctx) -> None:
    """`effective` comes from `providers.global_defaults[layer]` (the live,
    in-process value), not the DB row -- the two can diverge (e.g. this
    worker applied a live override through a different path than the one
    that wrote the row, or hasn't reloaded yet). `source` still reflects
    row presence regardless."""
    client, app = ctx
    await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    # The live registry has since moved on to a different value than what's
    # stored in the DB row.
    app.state.providers = SimpleNamespace(
        global_defaults={"llm": {"provider": "anthropic", "model": "claude-haiku-4-5"}},
    )

    resp = await client.get("/platform/pipeline", headers=ADMIN_HEADERS)
    llm = next(layer for layer in resp.json()["layers"] if layer["layer"] == "llm")
    assert llm["source"] == "override"  # DB row still exists
    assert llm["effective"]["provider"] == "anthropic"  # but reports the LIVE value
    assert llm["effective"]["model"] == "claude-haiku-4-5"


# --- PUT: valid override upserts, reflected on next GET -------------------


async def test_put_valid_override_upserts_and_get_reflects_it(ctx) -> None:
    client, _ = ctx
    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "override"
    assert body["effective"] == {"provider": "groq", "model": "llama-3.3-70b-versatile"}
    assert body["updated_at"] is not None

    resp2 = await client.get("/platform/pipeline", headers=ADMIN_HEADERS)
    llm = next(layer for layer in resp2.json()["layers"] if layer["layer"] == "llm")
    assert llm["source"] == "override"
    assert llm["effective"]["provider"] == "groq"
    assert llm["effective"]["model"] == "llama-3.3-70b-versatile"


async def test_put_provider_with_no_model_dimension_clears_model(ctx) -> None:
    """Azure TTS has no model dimension (empty catalog list) -- a PUT with no
    model (or one sent anyway) must store model=None, not reject."""
    client, _ = ctx
    resp = await client.put(
        "/platform/pipeline/tts", headers=ADMIN_HEADERS,
        json={"provider": "azure"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["effective"]["provider"] == "azure"
    assert body["effective"]["model"] is None


# --- PUT: validation errors -------------------------------------------


async def test_put_unknown_provider_returns_422(ctx) -> None:
    client, _ = ctx
    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "not-a-real-provider"},
    )
    assert resp.status_code == 422


async def test_put_unknown_model_for_known_provider_returns_422(ctx) -> None:
    client, _ = ctx
    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "not-a-real-model"},
    )
    assert resp.status_code == 422


async def test_put_missing_model_for_provider_that_requires_one_returns_422(ctx) -> None:
    client, _ = ctx
    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq"},
    )
    assert resp.status_code == 422


# --- PUT applies live ------------------------------------------------------


async def test_put_applies_live_via_reset_platform_defaults_and_evict_all(ctx) -> None:
    client, app = ctx
    fake_providers = SimpleNamespace(
        global_defaults={"llm": {"provider": "gemini", "model": "gemini-3.8-flash"}},
        reset_platform_defaults=MagicMock(),
    )
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    assert resp.status_code == 200
    fake_providers.reset_platform_defaults.assert_called_once()
    call_args = fake_providers.reset_platform_defaults.call_args
    assert call_args[0][0] == "llm"
    assert call_args[0][1]["provider"] == "groq"
    assert call_args[0][1]["model"] == "llama-3.3-70b-versatile"
    fake_registry.evict_all.assert_called_once()


async def test_put_applies_live_is_getattr_safe_with_bare_providers_stub(ctx) -> None:
    """Some test harnesses stub `providers` as a bare SimpleNamespace with no
    `global_defaults`/`reset_platform_defaults` -- the route must not blow up
    on that, just skip the live-apply."""
    client, app = ctx
    app.state.providers = SimpleNamespace()  # no global_defaults, no reset method

    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    assert resp.status_code == 200


# --- DELETE: revert to yaml -------------------------------------------


async def test_delete_reverts_to_yaml_and_get_reflects_it(ctx) -> None:
    client, _ = ctx
    await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    resp = await client.delete("/platform/pipeline/llm", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "yaml"
    assert body["effective"]["provider"] == "gemini"  # back to config/default.yaml

    resp2 = await client.get("/platform/pipeline", headers=ADMIN_HEADERS)
    llm = next(layer for layer in resp2.json()["layers"] if layer["layer"] == "llm")
    assert llm["source"] == "yaml"


async def test_delete_with_no_existing_override_is_a_no_op_200(ctx) -> None:
    client, _ = ctx
    resp = await client.delete("/platform/pipeline/stt", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["source"] == "yaml"


# --- PUT: buildability check (provider must actually construct) ----------
#
# The catalog check above only guards against typos/unlisted entries -- it
# doesn't know a provider is registered under a DIFFERENT dict than the one
# `providers.<layer>_factory` looks up (deepgram: catalog lists it for "stt",
# but it only lives in STREAMING_STT_PROVIDERS, not STT_PROVIDERS, which is
# what stt_factory=get_stt_provider uses), or that a provider's platform API
# key is unset in THIS environment. Both would otherwise only surface at the
# next real voice call / chat session. These tests wire real factories (not
# the bare SimpleNamespace the tests above use) so the check actually runs.


def _real_factory_providers(**global_defaults):
    from src.providers import get_llm_provider, get_stt_provider, get_tts_provider
    return SimpleNamespace(
        global_defaults=global_defaults,
        stt_factory=get_stt_provider,
        llm_factory=get_llm_provider,
        tts_factory=get_tts_provider,
        reset_platform_defaults=MagicMock(),
    )


async def test_put_stt_deepgram_cannot_be_built_returns_422_and_leaves_db_and_registry_unchanged(
    ctx,
) -> None:
    """deepgram is a valid catalog entry for "stt" but is only registered in
    STREAMING_STT_PROVIDERS -- get_stt_provider (stt_factory) doesn't know
    it, so building it raises UnknownProviderError. Must 422, not commit a
    row that would break every voice call inheriting the platform STT."""
    client, app = ctx
    fake_providers = _real_factory_providers(stt={"provider": "sarvam"})
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    resp = await client.put(
        "/platform/pipeline/stt", headers=ADMIN_HEADERS,
        json={"provider": "deepgram", "model": "nova-3"},
    )
    assert resp.status_code == 422
    assert "deepgram" in resp.json()["detail"]
    fake_providers.reset_platform_defaults.assert_not_called()
    fake_registry.evict_all.assert_not_called()

    resp2 = await client.get("/platform/pipeline", headers=ADMIN_HEADERS)
    stt = next(layer for layer in resp2.json()["layers"] if layer["layer"] == "stt")
    assert stt["source"] == "yaml"  # no row was committed


async def test_put_llm_anthropic_missing_api_key_returns_422_and_leaves_db_and_registry_unchanged(
    ctx, monkeypatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, app = ctx
    fake_providers = _real_factory_providers(llm={"provider": "gemini", "model": "gemini-3.8-flash"})
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "anthropic", "model": "claude-haiku-4-5"},
    )
    assert resp.status_code == 422
    assert "anthropic" in resp.json()["detail"]
    fake_providers.reset_platform_defaults.assert_not_called()
    fake_registry.evict_all.assert_not_called()

    resp2 = await client.get("/platform/pipeline", headers=ADMIN_HEADERS)
    llm = next(layer for layer in resp2.json()["layers"] if layer["layer"] == "llm")
    assert llm["source"] == "yaml"  # no row was committed


async def test_put_tts_elevenlabs_missing_api_key_returns_422_and_leaves_db_and_registry_unchanged(
    ctx, monkeypatch,
) -> None:
    """elevenlabs.py stores a missing key as `self._api_key = "" ` in
    __init__ and only raises once `synthesize()` is actually called -- so
    `factory(cfg)` alone would build "successfully" here with no key at all.
    `_check_buildable` must also check the adapter's own `_api_key` attribute
    so this still 422s, same as the deepgram/anthropic cases above."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    client, app = ctx
    fake_providers = _real_factory_providers(tts={"provider": "sarvam"})
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    resp = await client.put(
        "/platform/pipeline/tts", headers=ADMIN_HEADERS,
        json={"provider": "elevenlabs", "model": "eleven_flash_v2_5"},
    )
    assert resp.status_code == 422
    assert "elevenlabs" in resp.json()["detail"]
    fake_providers.reset_platform_defaults.assert_not_called()
    fake_registry.evict_all.assert_not_called()

    resp2 = await client.get("/platform/pipeline", headers=ADMIN_HEADERS)
    tts = next(layer for layer in resp2.json()["layers"] if layer["layer"] == "tts")
    assert tts["source"] == "yaml"  # no row was committed


async def test_put_tts_elevenlabs_with_api_key_set_returns_200(ctx, monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key-123")
    client, app = ctx
    fake_providers = _real_factory_providers(tts={"provider": "sarvam"})
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    resp = await client.put(
        "/platform/pipeline/tts", headers=ADMIN_HEADERS,
        json={"provider": "elevenlabs", "model": "eleven_flash_v2_5"},
    )
    assert resp.status_code == 200
    fake_providers.reset_platform_defaults.assert_called_once()
    fake_registry.evict_all.assert_called_once()


async def test_put_valid_provider_still_works_with_real_factories_wired(ctx) -> None:
    """GROQ_API_KEY is set globally in tests/conftest.py -- a real
    GroqLLMAdapter builds cleanly, so the buildability check must not block
    an ordinary valid save."""
    client, app = ctx
    fake_providers = _real_factory_providers(llm={"provider": "gemini", "model": "gemini-3.8-flash"})
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    assert resp.status_code == 200
    fake_providers.reset_platform_defaults.assert_called_once()
    fake_registry.evict_all.assert_called_once()


# --- PUT: updated_by records the admin actor label -------------------------


async def test_put_records_admin_actor_label_as_updated_by(ctx) -> None:
    client, _ = ctx
    set_admin_tokens(["lbl=ops-jane:admin-token"])

    resp = await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    assert resp.status_code == 200
    assert resp.json()["updated_by"] == "ops-jane"


# --- DELETE: live-apply + buildability -------------------------------------


async def test_delete_applies_live_via_reset_platform_defaults_and_evict_all(ctx) -> None:
    client, app = ctx
    await client.put(
        "/platform/pipeline/llm", headers=ADMIN_HEADERS,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )
    fake_providers = SimpleNamespace(
        global_defaults={"llm": {"provider": "groq", "model": "llama-3.3-70b-versatile"}},
        reset_platform_defaults=MagicMock(),
    )
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    resp = await client.delete("/platform/pipeline/llm", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    fake_providers.reset_platform_defaults.assert_called_once()
    call_args = fake_providers.reset_platform_defaults.call_args
    assert call_args[0][0] == "llm"
    assert call_args[0][1]["provider"] == "gemini"  # back to config/default.yaml
    fake_registry.evict_all.assert_called_once()


async def test_delete_reverts_even_when_yaml_config_cannot_be_built(ctx, monkeypatch, caplog) -> None:
    """tts's yaml default is sarvam -- if SARVAM_API_KEY is unset even the
    YAML config can't build. DELETE must still succeed (yaml is the
    deploy-time truth); only a warning is logged."""
    client, app = ctx
    await client.put(
        "/platform/pipeline/tts", headers=ADMIN_HEADERS,
        json={"provider": "azure"},
    )
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    fake_providers = _real_factory_providers(tts={"provider": "azure"})
    fake_registry = SimpleNamespace(evict_all=MagicMock())
    app.state.providers = fake_providers
    app.state.registry = fake_registry

    with caplog.at_level("WARNING"):
        resp = await client.delete("/platform/pipeline/tts", headers=ADMIN_HEADERS)

    assert resp.status_code == 200
    assert resp.json()["effective"]["provider"] == "sarvam"
    fake_providers.reset_platform_defaults.assert_called_once()  # still applied
    fake_registry.evict_all.assert_called_once()
    assert any("tts" in rec.message and "could not be built" in rec.message for rec in caplog.records)


# --- auth ---------------------------------------------------------------


async def test_no_admin_bearer_is_rejected_on_all_three_verbs(ctx) -> None:
    client, _ = ctx
    assert (await client.get("/platform/pipeline")).status_code == 401
    assert (await client.put(
        "/platform/pipeline/llm", json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )).status_code == 401
    assert (await client.delete("/platform/pipeline/llm")).status_code == 401


async def test_invalid_admin_bearer_is_rejected_on_all_three_verbs(ctx) -> None:
    client, _ = ctx
    bad = {"Authorization": "Bearer not-a-real-admin-token"}
    assert (await client.get("/platform/pipeline", headers=bad)).status_code == 403
    assert (await client.put(
        "/platform/pipeline/llm", headers=bad,
        json={"provider": "groq", "model": "llama-3.3-70b-versatile"},
    )).status_code == 403
    assert (await client.delete("/platform/pipeline/llm", headers=bad)).status_code == 403
