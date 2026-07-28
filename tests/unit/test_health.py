"""Smoke test for the /health endpoint.

Patches the redis client and DB sessionmaker so the route can run without
real infrastructure.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient


@asynccontextmanager
async def _no_lifespan(app):
    yield


@pytest.mark.asyncio
async def test_health_reports_provider_names(monkeypatch) -> None:
    # Patch lifespan-bound globals before importing main.
    from src import main as main_module

    # Disable real lifespan — we'll seed app.state manually.
    monkeypatch.setattr(main_module, "lifespan", _no_lifespan)

    # Reload-style swap: rebuild the app with the patched lifespan.
    from fastapi import FastAPI

    test_app = FastAPI(lifespan=_no_lifespan)

    # Wire the same /health handler on the test app.
    test_app.add_api_route("/health", main_module.health, methods=["GET"])
    main_module.app = test_app  # so handler's app.state references work

    # Seed state.
    redis_stub = SimpleNamespace(ping=AsyncMock(return_value=True))
    test_app.state.redis = redis_stub
    test_app.state.settings = main_module.get_settings()

    # Patch DB sessionmaker to a stub that yields a session whose execute is mocked.
    fake_session = MagicMock()
    fake_session.execute = AsyncMock(return_value=None)

    @asynccontextmanager
    async def fake_session_cm(*a, **kw):
        yield fake_session

    fake_sm = MagicMock(side_effect=fake_session_cm)
    monkeypatch.setattr(main_module, "get_sessionmaker", lambda: fake_sm)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["platform_defaults"] == {
        "stt": "sarvam",
        "llm": "gemini",
        "tts": "sarvam",
        "telephony": "twilio",
        "vector_store": "pgvector",
    }
    assert body["redis"] == "ok"
    assert body["db"] == "ok"
    assert body["status"] == "ok"
    # Tenants list is empty when not seeded — the test bypasses lifespan.
    assert body["tenant_count"] == 0
    assert body["tenants"] == []


@pytest.mark.asyncio
async def test_health_reports_tenants(monkeypatch) -> None:
    """When tenants are loaded, /health surfaces them in the response."""
    from src import main as main_module
    from src.config_tenant import (
        TenantSettings, TenantPipelineConfig, TenantSTTConfig, TenantLLMConfig,
    )

    monkeypatch.setattr(main_module, "lifespan", _no_lifespan)
    from fastapi import FastAPI
    test_app = FastAPI(lifespan=_no_lifespan)
    test_app.add_api_route("/health", main_module.health, methods=["GET"])
    main_module.app = test_app

    test_app.state.redis = SimpleNamespace(ping=AsyncMock(return_value=True))
    test_app.state.settings = main_module.get_settings()
    test_app.state.tenants = {
        "acme": TenantSettings(
            id="t_acme", slug="acme", name="Acme",
            pipeline=TenantPipelineConfig(
                stt=TenantSTTConfig(provider="sarvam"),
                llm=TenantLLMConfig(provider="gemini"),  # tenant override
            ),
        ),
    }

    fake_session = MagicMock()
    fake_session.execute = AsyncMock(return_value=None)

    @asynccontextmanager
    async def fake_session_cm(*a, **kw):
        yield fake_session

    monkeypatch.setattr(main_module, "get_sessionmaker", lambda: MagicMock(side_effect=fake_session_cm))

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        body = (await c.get("/health")).json()

    assert body["tenant_count"] == 1
    assert body["tenants"][0]["slug"] == "acme"
    # Tenant override is reflected
    assert body["tenants"][0]["providers"]["llm"] == "gemini"
    # Falls back to platform default for unspecified layers
    assert body["tenants"][0]["providers"]["telephony"] == "twilio"
