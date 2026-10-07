"""Bootstrap tests for src/main.py lifespan logic.

Validates admin-token parsing. Tenant loading moved from YAML-on-boot to the
DB resolver + seed (see test_seed.py / test_db_resolver.py).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.config_tenant import discover_tenant_slugs, load_tenant
from src.main import (
    _admin_tokens_from_env,
    _load_platform_pipeline_overrides,
    _overlay_platform_pipeline_overrides,
)
from src.models.database import Base
from src.models.platform_pipeline import PlatformPipelineDefault


def test_example_tenant_yaml_still_loads() -> None:
    """The shipped example tenant must still parse (it's what the seed reads)."""
    from pathlib import Path
    assert "example" in discover_tenant_slugs(Path("config/tenants"))
    assert load_tenant("example", Path("config/tenants")).name == "Example Telecom"


def test_admin_tokens_from_env_empty(monkeypatch) -> None:
    monkeypatch.delenv("VOX_ADMIN_TOKENS", raising=False)
    assert _admin_tokens_from_env() == []


def test_admin_tokens_from_env_comma_separated(monkeypatch) -> None:
    monkeypatch.setenv("VOX_ADMIN_TOKENS", "tok-a, tok-b , tok-c")
    assert _admin_tokens_from_env() == ["tok-a", "tok-b", "tok-c"]


def test_admin_tokens_from_env_skips_empty_entries(monkeypatch) -> None:
    monkeypatch.setenv("VOX_ADMIN_TOKENS", "a,,b,")
    assert _admin_tokens_from_env() == ["a", "b"]


# --- _load_platform_pipeline_overrides ------------------------------------


@pytest_asyncio.fixture
async def _sessionmaker_with_schema():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()


@pytest.mark.asyncio
async def test_load_platform_pipeline_overrides_empty_db_returns_empty_dict(
    _sessionmaker_with_schema,
) -> None:
    result = await _load_platform_pipeline_overrides(_sessionmaker_with_schema)
    assert result == {}


@pytest.mark.asyncio
async def test_load_platform_pipeline_overrides_missing_table_returns_empty_dict_not_raise() -> None:
    """A fresh DB the very first time this ships, before 0033 has run -- the
    table genuinely doesn't exist yet. Must not blow up startup."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    # Deliberately no Base.metadata.create_all -- no tables exist at all.
    result = await _load_platform_pipeline_overrides(sm)
    assert result == {}
    await engine.dispose()


@pytest.mark.asyncio
async def test_load_platform_pipeline_overrides_returns_seeded_rows(
    _sessionmaker_with_schema,
) -> None:
    from datetime import datetime

    sm = _sessionmaker_with_schema
    async with sm() as session:
        session.add(PlatformPipelineDefault(
            layer="llm", provider="groq", model="llama-3.3-70b-versatile",
            updated_at=datetime(2026, 1, 1), updated_by="admin",
        ))
        session.add(PlatformPipelineDefault(
            layer="tts", provider="azure", model=None,
            updated_at=datetime(2026, 1, 1), updated_by="admin",
        ))
        await session.commit()

    result = await _load_platform_pipeline_overrides(sm)
    assert result == {
        "llm": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "tts": {"provider": "azure", "model": None},
    }


# --- _overlay_platform_pipeline_overrides ----------------------------------
#
# The lifespan loop (src/main.py, ~1348) that applies `_load_platform_
# pipeline_overrides`'s result onto the yaml-built `global_defaults` dict
# BEFORE `build_provider_registry` is called -- i.e. what makes a DB-set
# platform pipeline default already be in effect the moment the process
# boots, not just from the next PUT. Extracted into this small, directly
# testable function rather than exercised through the full `lifespan`
# (too heavy for what this is checking).


def test_overlay_platform_pipeline_overrides_applies_onto_global_defaults() -> None:
    global_defaults = {
        "stt": {"provider": "sarvam", "model": "saaras:v3", "language": "hi-IN"},
        "llm": {"provider": "gemini", "model": "gemini-3.8-flash", "temperature": 0.7},
        "tts": {"provider": "sarvam", "voice_id": "anushka"},
    }
    overrides = {"llm": {"provider": "groq", "model": "llama-3.3-70b-versatile"}}

    result = _overlay_platform_pipeline_overrides(global_defaults, overrides)

    assert result is global_defaults  # mutated in place and returned, not rebuilt
    assert global_defaults["llm"] == {
        "provider": "groq", "model": "llama-3.3-70b-versatile", "temperature": 0.7,
    }
    # Layers with no override are untouched.
    assert global_defaults["stt"] == {"provider": "sarvam", "model": "saaras:v3", "language": "hi-IN"}
    assert global_defaults["tts"] == {"provider": "sarvam", "voice_id": "anushka"}


def test_overlay_platform_pipeline_overrides_empty_overrides_is_a_no_op() -> None:
    global_defaults = {"stt": {"provider": "sarvam"}, "llm": {"provider": "gemini"}, "tts": {"provider": "sarvam"}}
    original = {k: dict(v) for k, v in global_defaults.items()}

    _overlay_platform_pipeline_overrides(global_defaults, {})

    assert global_defaults == original


def test_overlay_platform_pipeline_overrides_drops_provider_specific_fields_on_switch() -> None:
    """Same cross-provider guard as apply_platform_pipeline_override itself --
    switching tts provider with no model must drop the old provider's model/
    voice_id, not carry them onto the new provider."""
    global_defaults = {"tts": {"provider": "sarvam", "model": "bulbul:v3", "voice_id": "anushka"}}
    overrides = {"tts": {"provider": "elevenlabs", "model": None}}

    _overlay_platform_pipeline_overrides(global_defaults, overrides)

    assert global_defaults["tts"] == {"provider": "elevenlabs"}
