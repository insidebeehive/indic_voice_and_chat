"""Test that image/video WS frames upload to S3 and persist media_url."""

from __future__ import annotations

import base64
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat as chat_api
from src.models.database import Base
from src.models.chat import ChatMessage, ChatSession


class _FakeMediaStore:
    def __init__(self):
        self.uploaded = []
    async def upload(self, data, key, content_type):
        self.uploaded.append((key, content_type))
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
async def test_image_ws_uploads_to_s3():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(ChatSession(id="s2", tenant_id="t1", language="hi",
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

    import src.auth.middleware as mw
    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)
        img_bytes = b"\x89PNG\r\n..."
        encoded = base64.b64encode(img_bytes).decode()
        with client.websocket_connect("/api/v1/chat/ws/s2") as ws:
            ws.send_text(json.dumps({"type": "image", "data": encoded, "mime": "image/png"}))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"

    assert len(media_store.uploaded) == 1
    key, ct = media_store.uploaded[0]
    assert key.startswith("chat/t1/s2/")
    assert key.endswith(".png")

    # Verify media_url persisted in DB
    from sqlalchemy import select
    async with sm() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "s2", ChatMessage.type == "image")
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].media_url is not None
    assert rows[0].media_url.endswith(".png")

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()
