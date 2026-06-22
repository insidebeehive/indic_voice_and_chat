"""Tests for GET /api/v1/chat/media/{message_id}."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from unittest.mock import AsyncMock, MagicMock

from src.api import chat as chat_api
from src.api.deps import get_db_session
from src.auth.middleware import set_tenant_resolver, set_admin_tokens
from src.models.database import Base
from src.models.chat import ChatMessage, ChatSession


class _FakeStore:
    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        return f"https://cdn.example.com/{key}?ttl={ttl_seconds}"


class _FakeTenant:
    id = "t1"
    slug = "demo"
    name = "Demo"
    settings = MagicMock()


class _FakeResolver:
    async def resolve_by_token(self, token_hash: str):
        # token_hash is the SHA-256 of the plaintext token
        from src.auth.context import hash_api_token
        if token_hash == hash_api_token("good-token"):
            return _FakeTenant()
        return None

    async def resolve_by_slug(self, slug):
        return None

    async def resolve_by_id(self, tenant_id):
        return None

    async def resolve_by_phone_number(self, phone):
        return None


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(ChatSession(id="sess1", tenant_id="t1", language="hi",
                           status="active", mode="ai", extra_data={}))
        msg = ChatMessage(session_id="sess1", role="customer", type="audio",
                          content="[audio]", media_url="chat/t1/sess1/abc.webm",
                          media_mime="audio/webm")
        db.add(msg)
        await db.flush()
        msg_id = msg.id
        await db.commit()

    chat_api.set_media_store(_FakeStore())
    chat_api.set_chat_sessionmaker(sm)
    set_tenant_resolver(_FakeResolver())
    set_admin_tokens(["admin-token"])

    app = FastAPI()

    async def _session():
        async with sm() as s:
            yield s

    app.dependency_overrides[get_db_session] = _session
    app.include_router(chat_api.router, prefix="/api/v1")

    yield app, msg_id

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    set_tenant_resolver(None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_media_endpoint_bearer_redirects(ctx):
    app, msg_id = ctx
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get(
            f"/api/v1/chat/media/{msg_id}",
            headers={"Authorization": "Bearer good-token"},
            follow_redirects=False,
        )
    assert resp.status_code == 302
    assert "cdn.example.com" in resp.headers["location"]


@pytest.mark.asyncio
async def test_media_endpoint_session_id_redirects(ctx):
    app, msg_id = ctx
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get(
            f"/api/v1/chat/media/{msg_id}?session_id=sess1",
            follow_redirects=False,
        )
    assert resp.status_code == 302


@pytest.mark.asyncio
async def test_media_endpoint_wrong_session_id_401(ctx):
    app, msg_id = ctx
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get(
            f"/api/v1/chat/media/{msg_id}?session_id=wrong",
            follow_redirects=False,
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_media_endpoint_no_auth_401(ctx):
    app, msg_id = ctx
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get(f"/api/v1/chat/media/{msg_id}", follow_redirects=False)
    assert resp.status_code == 401
