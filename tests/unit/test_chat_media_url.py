"""Tests for fetching chat image/video attachments via a caller-supplied
media_url (e.g. a presigned R2/S3 URL) instead of inline base64 bytes."""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
import respx
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat as chat_api
from src.models.database import Base
from src.models.chat import ChatSession


class _FakeMediaStore:
    def __init__(self):
        self.uploaded = []

    async def upload(self, data, key, content_type):
        self.uploaded.append((data, key, content_type))

    async def signed_url(self, key, ttl_seconds):
        return f"https://cdn/{key}"


class _FakeTurnResult:
    class _Resp:
        response_text = "Image received"
        sources_used = []
        suggested_followups = []
        action = "none"
    response = _Resp()
    escalation = None
    call_offer = None


@pytest.mark.asyncio
async def test_image_ws_media_url_fetches_and_uploads():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(ChatSession(id="s3", tenant_id="t1", language="hi",
                           status="active", mode="ai", extra_data={}))
        await db.commit()

    media_store = _FakeMediaStore()
    chat_api.set_media_store(media_store)
    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent.handle_image = AsyncMock(return_value=_FakeTurnResult())
    fake_agent.session = MagicMock()

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    chat_api.set_chatbot_factory(AsyncMock(return_value=fake_agent))

    img_bytes = b"\x89PNG\r\n" + b"x" * 100

    async def _fake_fetch(url):
        assert url == "https://bucket.r2.example.com/photo.png?sig=abc"
        return img_bytes, "image/png"

    import src.auth.middleware as mw
    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)), \
         patch.object(chat_api, "_fetch_media_url", _fake_fetch):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)
        with client.websocket_connect("/api/v1/chat/ws/s3") as ws:
            ws.send_text(json.dumps({
                "type": "image",
                "media_url": "https://bucket.r2.example.com/photo.png?sig=abc",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"

    assert len(media_store.uploaded) == 1
    data, key, ct = media_store.uploaded[0]
    assert data == img_bytes
    assert ct == "image/png"
    fake_agent.handle_image.assert_awaited_once()
    call_args = fake_agent.handle_image.await_args
    assert call_args.args[0] == img_bytes
    assert call_args.args[1] == "image/png"

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_image_ws_media_url_fetch_failure_sends_error():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(ChatSession(id="s4", tenant_id="t1", language="hi",
                           status="active", mode="ai", extra_data={}))
        await db.commit()

    chat_api.set_media_store(_FakeMediaStore())
    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent.handle_image = AsyncMock(return_value=_FakeTurnResult())
    fake_agent.session = MagicMock()

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    chat_api.set_chatbot_factory(AsyncMock(return_value=fake_agent))

    async def _fake_fetch(url):
        raise ValueError("media_url content exceeds size limit")

    import src.auth.middleware as mw
    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)), \
         patch.object(chat_api, "_fetch_media_url", _fake_fetch):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)
        with client.websocket_connect("/api/v1/chat/ws/s4") as ws:
            ws.send_text(json.dumps({
                "type": "image",
                "media_url": "https://bucket.r2.example.com/huge.png",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"

    fake_agent.handle_image.assert_not_awaited()

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_fetch_media_url_rejects_oversized_body():
    url = "https://bucket.r2.example.com/huge.png"
    body = b"x" * (chat_api._MAX_MEDIA_FETCH_BYTES + 1)
    with respx.mock, patch.object(chat_api, "_is_public_host", return_value=True):
        respx.get(url).mock(return_value=httpx.Response(
            200, headers={"content-type": "image/png"}, content=body))
        with pytest.raises(ValueError, match="exceeds size limit"):
            await chat_api._fetch_media_url(url)


@pytest.mark.asyncio
async def test_fetch_media_url_rejects_bad_content_type():
    url = "https://bucket.r2.example.com/not-media.txt"
    with respx.mock, patch.object(chat_api, "_is_public_host", return_value=True):
        respx.get(url).mock(return_value=httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"hello"))
        with pytest.raises(ValueError, match="unsupported content-type"):
            await chat_api._fetch_media_url(url)


@pytest.mark.asyncio
async def test_fetch_media_url_rejects_non_https():
    with pytest.raises(ValueError, match="https"):
        await chat_api._fetch_media_url("http://bucket.r2.example.com/photo.png")


@pytest.mark.asyncio
async def test_fetch_media_url_rejects_private_host():
    with pytest.raises(ValueError, match="non-public"):
        await chat_api._fetch_media_url("https://127.0.0.1/photo.png")


@pytest.mark.asyncio
async def test_fetch_media_url_accepts_valid_image():
    url = "https://bucket.r2.example.com/photo.png"
    body = b"\x89PNG\r\n" + b"x" * 50
    with respx.mock, patch.object(chat_api, "_is_public_host", return_value=True):
        respx.get(url).mock(return_value=httpx.Response(
            200, headers={"content-type": "image/png"}, content=body))
        data, content_type = await chat_api._fetch_media_url(url)
    assert data == body
    assert content_type == "image/png"
