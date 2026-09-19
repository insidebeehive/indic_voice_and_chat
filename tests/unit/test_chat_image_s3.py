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
    input_tokens = 0
    output_tokens = 0
    llm_provider = ""
    llm_model = ""


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


@pytest.mark.asyncio
async def test_image_ws_persists_customer_row_before_the_turn_runs():
    """The actual production bug: submit_deposit_verification looks up the
    customer's screenshot mid-turn by querying chat_messages for the newest
    type='image' row (src/chatbot/deposit_verification.py's
    screenshot_predicate), through its own separate DB session. If the
    customer's row is only written AFTER handle_image returns (the old
    ordering), that query finds nothing and the tool wrongly tells the
    customer to upload a screenshot they already sent.

    `handle_image` here queries the DB itself, from a session opened fresh
    off the same sessionmaker, standing in for that separate-session lookup
    — so this fails against the old code (row written only in the
    `_persist_turn` call after `_run_turn`) and passes against the new
    pre-persist-then-turn ordering."""
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

    seen_image_row_mid_turn = {}

    async def _handle_image(data, mime, caption):
        from sqlalchemy import select
        async with sm() as mid_turn_db:
            row = (await mid_turn_db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == "s3", ChatMessage.type == "image")
                .order_by(ChatMessage.id.desc())
            )).scalars().first()
        seen_image_row_mid_turn["row"] = row
        return _FakeTurnResult()

    fake_agent = MagicMock()
    fake_agent.handle_image = AsyncMock(side_effect=_handle_image)
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
        with client.websocket_connect("/api/v1/chat/ws/s3") as ws:
            ws.send_text(json.dumps({"type": "image", "data": encoded, "mime": "image/png"}))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"

    assert seen_image_row_mid_turn["row"] is not None, (
        "the customer's image row must already be committed and visible to "
        "a separate DB session while handle_image (the turn) is still running"
    )
    assert seen_image_row_mid_turn["row"].media_url is not None
    assert seen_image_row_mid_turn["row"].media_url.endswith(".png")

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_image_ws_falls_back_to_full_customer_row_when_pre_persist_returns_none():
    """Regression test for the WS image/video branch's `_persist_turn` call
    (src/api/chat.py, right after `_persist_inbound_media_message`) silently
    dropping `user_type=`/`media_mime=`/`media_url=`/`source_media_url=`.

    When `_persist_inbound_media_message` fails (returns `None` — a transient
    DB error) `_persist_turn` falls back to writing the customer row itself,
    but only from those kwargs the call site still passes alongside
    `customer_message_id=None`. Drop any of them from that call site and the
    fallback silently writes a corrupt type="text"/no-media row instead of
    the real image row — exactly the bug a prior fix round claimed to cover
    with a unit test that never actually drove this call site (see
    tests/unit/test_chat_persist.py's now-corrected docstring).

    This monkeypatches the pre-persist helper to return `None` (simulating
    that failure) and drives a real WS image frame end-to-end through the
    actual call site, so it fails the moment any of those kwargs are removed
    there — verified by temporarily deleting them and re-running this test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(ChatSession(id="s4", tenant_id="t1", language="hi",
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
    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)), \
         patch.object(chat_api, "_persist_inbound_media_message",
                      AsyncMock(return_value=None)), \
         patch.object(chat_api, "_fetch_media_url",
                      AsyncMock(return_value=(b"pngdata", "image/png"))):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)
        with client.websocket_connect("/api/v1/chat/ws/s4") as ws:
            ws.send_text(json.dumps({
                "type": "image",
                "media_url": "https://crm.example.com/uploads/shot.png",
                "mime": "image/png",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"

    from sqlalchemy import select
    async with sm() as db:
        rows = (await db.execute(
            select(ChatMessage).where(
                ChatMessage.session_id == "s4", ChatMessage.role == "customer")
        )).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.type == "image"
    assert row.media_mime == "image/png"
    assert row.media_url is not None
    assert row.media_url.endswith(".png")
    assert row.source_media_url == "https://crm.example.com/uploads/shot.png"

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_image_ws_pre_persisted_customer_row_survives_a_raising_turn():
    """Regression coverage for the pre-persist helper's documented guarantee
    (see `_persist_inbound_media_message`'s docstring): the customer's row is
    committed BEFORE the turn runs, so if the turn itself then raises
    (provider timeout, tool error), that row must survive exactly as-is — one
    committed customer row, no agent row, `message_count` bumped by 1 (not
    rolled back, not duplicated).

    Drives a real WS image frame whose `handle_image` raises, through the
    actual per-turn try/except in `chat_websocket` (which logs, records a
    failure metric, and sends an `error` frame rather than propagating).
    Replaces test_chat_persist.py's
    test_pre_persist_helper_row_survives_a_raising_turn, which used a
    locally-defined coroutine that raised and was immediately caught by
    `pytest.raises` right there in the test — it never called `_persist_turn`
    or drove any real turn-failure path, so it could not have caught a
    regression in this behaviour."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(ChatSession(id="s5", tenant_id="t1", language="hi",
                           status="active", mode="ai", extra_data={}))
        await db.commit()

    media_store = _FakeMediaStore()
    chat_api.set_media_store(media_store)
    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent.handle_image = AsyncMock(side_effect=RuntimeError("provider timeout"))
    fake_agent.session = MagicMock()
    fake_agent._enable_tools = False
    fake_agent._llm_provider = ""
    fake_agent._llm_model = ""

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
        with client.websocket_connect("/api/v1/chat/ws/s5") as ws:
            ws.send_text(json.dumps({"type": "image", "data": encoded, "mime": "image/png"}))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"

    from sqlalchemy import select
    async with sm() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "s5")
        )).scalars().all()
        session_row = await db.get(ChatSession, "s5")
    assert len(rows) == 1
    assert rows[0].role == "customer"
    assert rows[0].type == "image"
    assert session_row.message_count == 1

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()
