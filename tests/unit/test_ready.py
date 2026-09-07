"""Smoke tests for the /ready endpoint.

Mirrors tests/unit/test_health.py's fixture/monkeypatch conventions:
patches the redis client and DB sessionmaker so the route can run without
real infrastructure, and covers all four up/down combinations.
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


def _wire_test_app(monkeypatch):
    """Build a test app with /ready wired up, returning (test_app, main_module)."""
    from src import main as main_module

    monkeypatch.setattr(main_module, "lifespan", _no_lifespan)

    from fastapi import FastAPI

    test_app = FastAPI(lifespan=_no_lifespan)
    test_app.add_api_route("/ready", main_module.ready, methods=["GET"])
    main_module.app = test_app  # so handler's app.state references work

    test_app.state.settings = main_module.get_settings()
    return test_app, main_module


def _patch_db(monkeypatch, main_module, *, up: bool) -> None:
    if up:
        fake_session = MagicMock()
        fake_session.execute = AsyncMock(return_value=None)

        @asynccontextmanager
        async def fake_session_cm(*a, **kw):
            yield fake_session

        monkeypatch.setattr(
            main_module, "get_sessionmaker", lambda: MagicMock(side_effect=fake_session_cm)
        )
    else:
        def broken_sessionmaker():
            raise RuntimeError("db unavailable")

        monkeypatch.setattr(main_module, "get_sessionmaker", broken_sessionmaker)


@pytest.mark.asyncio
async def test_ready_both_up_returns_200(monkeypatch) -> None:
    test_app, main_module = _wire_test_app(monkeypatch)
    test_app.state.redis = SimpleNamespace(ping=AsyncMock(return_value=True))
    _patch_db(monkeypatch, main_module, up=True)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"redis": "ok", "db": "ok", "status": "ready"}


@pytest.mark.asyncio
async def test_ready_redis_down_returns_503(monkeypatch) -> None:
    test_app, main_module = _wire_test_app(monkeypatch)
    test_app.state.redis = SimpleNamespace(
        ping=AsyncMock(side_effect=RuntimeError("redis unavailable"))
    )
    _patch_db(monkeypatch, main_module, up=True)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body == {"redis": "down", "db": "ok", "status": "not_ready"}


@pytest.mark.asyncio
async def test_ready_db_down_returns_503(monkeypatch) -> None:
    test_app, main_module = _wire_test_app(monkeypatch)
    test_app.state.redis = SimpleNamespace(ping=AsyncMock(return_value=True))
    _patch_db(monkeypatch, main_module, up=False)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body == {"redis": "ok", "db": "down", "status": "not_ready"}


@pytest.mark.asyncio
async def test_ready_both_down_returns_503(monkeypatch) -> None:
    test_app, main_module = _wire_test_app(monkeypatch)
    test_app.state.redis = SimpleNamespace(
        ping=AsyncMock(side_effect=RuntimeError("redis unavailable"))
    )
    _patch_db(monkeypatch, main_module, up=False)

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body == {"redis": "down", "db": "down", "status": "not_ready"}
