"""Test audio message handling over WebSocket."""

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
        self.uploaded.append((key, content_type, data))

    async def signed_url(self, key, ttl_seconds):
        return f"https://cdn/{key}"


class _FakeTurnResult:
    class _Resp:
        response_text = "I heard you"
        sources_used = []
        suggested_followups = []
        action = "none"
        language = "en"
        confidence = "high"

    response = _Resp()
    escalation = None
    call_offer = None
    input_tokens = 0
    output_tokens = 0
    llm_provider = ""
    llm_model = ""


@pytest_asyncio.fixture
async def ws_ctx():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(ChatSession(id="sess1", tenant_id="t1", language="hi",
                           status="active", mode="ai", extra_data={}))
        await db.commit()

    media_store = _FakeMediaStore()
    chat_api.set_media_store(media_store)
    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent.handle_message = AsyncMock(return_value=_FakeTurnResult())
    fake_agent._llm = None  # bare MagicMock() would auto-vivify a truthy _llm and win over .llm below
    fake_agent.llm = MagicMock()
    fake_agent.llm.transcribe_audio = AsyncMock(return_value="hello there")
    fake_agent.session = MagicMock()

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    async def fake_factory(tenant, scoped_id, *, customer_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, media_store, fake_agent

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_audio_ws_uploads_and_acks(ws_ctx):
    sm, media_store, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        audio_bytes = b"fake_audio_data"
        encoded = base64.b64encode(audio_bytes).decode()

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "data": encoded,
                "mime": "audio/webm;codecs=opus",
            }))
            # Expect typing frame
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            # Expect audio_ack
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "audio_ack"
            assert "/api/v1/chat/media/" in ack["media_url"]
            # Expect AI reply
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"
            assert "I heard you" in reply["text"]

    assert len(media_store.uploaded) == 1
    key, content_type, data = media_store.uploaded[0]
    assert key.startswith("chat/t1/sess1/")
    assert key.endswith(".webm")
    assert data == audio_bytes

    fake_agent.handle_message.assert_called_once_with("hello there")


@pytest.mark.asyncio
async def test_audio_ws_media_url_fetches_and_transcribes(ws_ctx):
    """`type:audio` accepts media_url as an alternative to base64 data,
    matching the image/video contract (CRM relays forward presigned URLs)."""
    sm, media_store, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    audio_bytes = b"fake_ogg_audio"

    async def _fake_fetch(url):
        assert url == "https://bucket.r2.example.com/voice.ogg?sig=abc"
        return audio_bytes, "audio/ogg"

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)), \
         patch.object(chat_api, "_fetch_media_url", _fake_fetch):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "media_url": "https://bucket.r2.example.com/voice.ogg?sig=abc",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "audio_ack"
            assert "/api/v1/chat/media/" in ack["media_url"]
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"
            assert "I heard you" in reply["text"]

    assert len(media_store.uploaded) == 1
    key, content_type, data = media_store.uploaded[0]
    assert data == audio_bytes
    assert content_type == "audio/ogg"

    fake_agent.llm.transcribe_audio.assert_awaited_once()
    assert fake_agent.llm.transcribe_audio.await_args.args[0] == audio_bytes
    fake_agent.handle_message.assert_called_once_with("hello there")


@pytest.mark.asyncio
async def test_audio_ws_media_url_fetch_failure_sends_error(ws_ctx):
    sm, media_store, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    async def _fake_fetch(url):
        raise ValueError("media_url content exceeds size limit")

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)), \
         patch.object(chat_api, "_fetch_media_url", _fake_fetch):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "media_url": "https://bucket.r2.example.com/huge.ogg",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"

    assert len(media_store.uploaded) == 0
    fake_agent.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_audio_ws_media_url_non_audio_content_returns_error(ws_ctx):
    """A media_url that serves a non-audio content type (and no explicit
    audio/* mime in the frame) must be rejected, not transcribed."""
    sm, media_store, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    async def _fake_fetch(url):
        return b"\x89PNG\r\n" + b"x" * 50, "image/png"

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)), \
         patch.object(chat_api, "_fetch_media_url", _fake_fetch):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "media_url": "https://bucket.r2.example.com/photo.png",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert "audio" in err["message"].lower()

    assert len(media_store.uploaded) == 0
    fake_agent.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_audio_ws_no_media_store_returns_error(ws_ctx):
    """When _media_store is None, return an error frame instead of crashing."""
    sm, media_store, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    # Temporarily clear the media store
    chat_api.set_media_store(None)
    try:
        with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
            app = FastAPI()
            app.include_router(chat_api.router, prefix="/api/v1")
            client = TestClient(app)

            audio_bytes = b"fake_audio_data"
            encoded = base64.b64encode(audio_bytes).decode()

            with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
                ws.send_text(json.dumps({
                    "type": "audio",
                    "data": encoded,
                    "mime": "audio/webm;codecs=opus",
                }))
                err = json.loads(ws.receive_text())
                assert err["type"] == "error"
                assert "media storage" in err["message"].lower()
    finally:
        chat_api.set_media_store(media_store)


@pytest.mark.asyncio
async def test_audio_ws_missing_mime_returns_error(ws_ctx):
    """Missing mime field should return an error frame."""
    sm, media_store, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        audio_bytes = b"fake_audio_data"
        encoded = base64.b64encode(audio_bytes).decode()

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "data": encoded,
                # missing "mime"
            }))
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"


@pytest.mark.asyncio
async def test_audio_ws_no_transcribe_method_uploads_only(ws_ctx):
    """If agent.llm does not have transcribe_audio, upload still works but no AI reply."""
    sm, media_store, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    # Remove transcribe_audio from the llm mock
    del fake_agent.llm.transcribe_audio

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        audio_bytes = b"fake_audio_no_transcription"
        encoded = base64.b64encode(audio_bytes).decode()

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio",
                "data": encoded,
                "mime": "audio/webm;codecs=opus",
            }))
            # Expect typing frame
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            # No transcription → audio_ack still sent (audio persisted)
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "audio_ack"
            # Error frame indicating transcription failed
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert "transcribe" in err["message"].lower()

    # Upload should still have happened
    assert len(media_store.uploaded) == 1
    fake_agent.handle_message.assert_not_called()
