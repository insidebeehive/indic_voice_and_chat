"""Capture side of `previous_conversation` (src/api/chat.py's
`_capture_previous_conversation`/`_warn_late_previous_conversation`/
`_stored_previous_conversation`, and the `session_first_frame_pending`
bookkeeping in `chat_websocket`). See
tests/unit/test_chatbot_previous_conversation.py for the fold-into-contents
side in src/agents/chatbot.py.

Fixture scaffolding (`_connect`, `_make_tenant`, `_FakeTurnResult`) mirrors
tests/unit/test_chat_duplicate_turn_guard.py -- see that file for the
reasoning behind each piece.
"""

from __future__ import annotations

import base64
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat as chat_api
from src.models.chat import ChatSession
from src.models.database import Base


class _FakeTurnResult:
    class _Resp:
        response_text = "ok"
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
    llm_provider = None
    llm_model = None


@pytest_asyncio.fixture
async def ws_pc_ctx():
    """Like test_chat_duplicate_turn_guard.py's ws_ctx, except the factory
    hands back a FRESH agent double per call (tracked in `created_agents`) --
    needed here to distinguish "the connection that captured it" from "a
    later reconnect's freshly-built agent" when proving persistence/reconnect
    behaviour, which a single shared fake_agent (that file's own fixture)
    can't do."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(ChatSession(id="sess1", tenant_id="t1", language="hi",
                            status="active", mode="ai", extra_data={}))
        await db.commit()

    chat_api.set_chat_sessionmaker(sm)

    created_agents: list = []

    def _new_agent():
        agent = MagicMock()
        agent._llm = None
        agent.llm = MagicMock()
        agent.session = MagicMock()
        agent.session.turns = []
        agent.handle_message = AsyncMock(return_value=_FakeTurnResult())
        agent.handle_image = AsyncMock(return_value=_FakeTurnResult())
        agent.summarize_session = AsyncMock(return_value="")
        # Real ChatBotAgent defaults this to None too (see __init__) --
        # matching it here means an agent that never gets touched by the
        # capture logic reads as "nothing captured", same as production.
        agent._previous_conversation = None
        created_agents.append(agent)
        return agent

    async def fake_factory(tenant, scoped_id, *, customer_id=None, ticket_id=None):
        return _new_agent()

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, created_agents

    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


def _make_tenant():
    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300
    fake_tenant.settings.events_webhook_url = None
    return fake_tenant


def _connect(fake_tenant):
    import src.auth.middleware as mw

    patcher = patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant))
    patcher.start()
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    client = TestClient(app)
    return patcher, client


# --- 1. First-frame capture, text/message shape -----------------------------


@pytest.mark.asyncio
async def test_first_frame_captures_previous_conversation_text_shape(ws_pc_ctx, monkeypatch) -> None:
    sm, agents = ws_pc_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "message", "text": "hi, my withdrawal is stuck",
                "previous_conversation": "Customer asked about a pending withdrawal.",
            }))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"
    finally:
        patcher.stop()

    assert len(agents) == 1
    assert agents[0]._previous_conversation == "Customer asked about a pending withdrawal."

    async with sm() as db:
        row = await db.get(ChatSession, "sess1")
    assert row.extra_data.get("previous_conversation") == "Customer asked about a pending withdrawal."


# --- 2. First-frame capture, image shape ------------------------------------


@pytest.mark.asyncio
async def test_first_frame_captures_previous_conversation_image_shape(ws_pc_ctx, monkeypatch) -> None:
    sm, agents = ws_pc_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)
    encoded = base64.b64encode(b"fake png bytes").decode()

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "image", "data": encoded, "mime": "image/png", "text": "screenshot",
                "previous_conversation": "Customer previously asked about KYC documents.",
            }))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"
    finally:
        patcher.stop()

    assert agents[0]._previous_conversation == "Customer previously asked about KYC documents."
    async with sm() as db:
        row = await db.get(ChatSession, "sess1")
    assert row.extra_data.get("previous_conversation") == "Customer previously asked about KYC documents."


# --- 3. A later frame's value is ignored; the first one is kept ------------


@pytest.mark.asyncio
async def test_later_frame_previous_conversation_ignored_first_kept(ws_pc_ctx, monkeypatch, caplog) -> None:
    sm, agents = ws_pc_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "message", "text": "first message",
                "previous_conversation": "ORIGINAL SUMMARY",
            }))
            json.loads(ws.receive_text())  # typing
            json.loads(ws.receive_text())  # message

            with caplog.at_level(logging.WARNING, logger="src.api.chat"):
                ws.send_text(json.dumps({
                    "type": "message", "text": "second message",
                    "previous_conversation": "A CLIENT-REWRITTEN REPLACEMENT SUMMARY",
                }))
                json.loads(ws.receive_text())  # typing
                json.loads(ws.receive_text())  # message
    finally:
        patcher.stop()

    # Same connection -> same agent instance -> still holds the FIRST value.
    assert len(agents) == 1
    assert agents[0]._previous_conversation == "ORIGINAL SUMMARY"

    async with sm() as db:
        row = await db.get(ChatSession, "sess1")
    assert row.extra_data.get("previous_conversation") == "ORIGINAL SUMMARY"

    assert any("non-first frame" in r.getMessage() for r in caplog.records), (
        "expected a warning log line for the rejected late previous_conversation"
    )


# --- 4. Persists across turns and survives a reconnect ----------------------


@pytest.mark.asyncio
async def test_previous_conversation_persists_across_turns_and_reconnect(ws_pc_ctx, monkeypatch) -> None:
    sm, agents = ws_pc_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "message", "text": "first message",
                "previous_conversation": "Customer previously asked about bonus wagering.",
            }))
            json.loads(ws.receive_text())
            json.loads(ws.receive_text())

            # A second turn on the SAME connection, no previous_conversation
            # on this frame -- must still benefit (every turn, not just the
            # first frame's own turn).
            ws.send_text(json.dumps({"type": "message", "text": "second message"}))
            json.loads(ws.receive_text())
            json.loads(ws.receive_text())

        # A brand-new connection (reconnect) for the SAME session: its
        # freshly-built agent must pick the summary back up from
        # ChatSession.extra_data, even though this connection's own first
        # frame never carries it.
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws2:
            ws2.send_text(json.dumps({"type": "message", "text": "third message, after reconnect"}))
            json.loads(ws2.receive_text())
            json.loads(ws2.receive_text())
    finally:
        patcher.stop()

    assert len(agents) == 2, "expected one fresh agent per connection"
    assert agents[0]._previous_conversation == "Customer previously asked about bonus wagering."
    assert agents[1]._previous_conversation == "Customer previously asked about bonus wagering."


# --- 5. Empty / missing / non-string is a silent no-op ----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "   ", 123, ["a", "b"], {"a": 1}, True])
async def test_invalid_previous_conversation_is_a_noop(ws_pc_ctx, monkeypatch, value) -> None:
    sm, agents = ws_pc_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "message", "text": "hello", "previous_conversation": value,
            }))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"
    finally:
        patcher.stop()

    assert agents[0]._previous_conversation is None
    async with sm() as db:
        row = await db.get(ChatSession, "sess1")
    assert "previous_conversation" not in (row.extra_data or {})


@pytest.mark.asyncio
async def test_missing_previous_conversation_key_is_a_noop(ws_pc_ctx, monkeypatch) -> None:
    sm, agents = ws_pc_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hello, no summary field at all"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"
    finally:
        patcher.stop()

    assert agents[0]._previous_conversation is None
    async with sm() as db:
        row = await db.get(ChatSession, "sess1")
    assert "previous_conversation" not in (row.extra_data or {})


# --- 6. The cap is applied at capture time -----------------------------------


@pytest.mark.asyncio
async def test_previous_conversation_is_capped_at_capture_time(ws_pc_ctx, monkeypatch) -> None:
    from src.agents.chatbot import PREVIOUS_CONVERSATION_MAX_CHARS

    sm, agents = ws_pc_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)
    oversized = "word " * 2000  # far over the cap

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "message", "text": "hi", "previous_conversation": oversized,
            }))
            json.loads(ws.receive_text())
            json.loads(ws.receive_text())
    finally:
        patcher.stop()

    stored = agents[0]._previous_conversation
    assert stored is not None
    assert len(stored) <= PREVIOUS_CONVERSATION_MAX_CHARS + 1  # +1 for the trailing ellipsis
    assert stored.endswith("…")

    async with sm() as db:
        row = await db.get(ChatSession, "sess1")
    assert row.extra_data.get("previous_conversation") == stored


def test_stored_summary_is_capped_on_read_not_only_on_capture() -> None:
    """The frame path caps before storing, but it is not the only writer:
    ChatSession.extra_data is written wholesale from a caller-supplied
    `metadata` dict at session creation, which carries no length constraint.
    An unbounded value seeded that way rides every round of every turn for the
    life of the session, and `contents` never caches -- so the bound has to be
    a property of what reaches the model, not of one writer.
    """
    from src.agents.chatbot import PREVIOUS_CONVERSATION_MAX_CHARS

    oversized = "word " * 5000
    assert len(oversized) > PREVIOUS_CONVERSATION_MAX_CHARS

    row = MagicMock()
    row.extra_data = {"previous_conversation": oversized}

    out = chat_api._stored_previous_conversation(row)

    assert out is not None
    assert len(out) <= PREVIOUS_CONVERSATION_MAX_CHARS + 1, (
        f"read path returned {len(out)} chars, cap is {PREVIOUS_CONVERSATION_MAX_CHARS}"
    )


def test_stored_summary_under_cap_is_returned_verbatim() -> None:
    """The read-path cap must not perturb a normal value -- otherwise every
    session would silently get an altered summary."""
    row = MagicMock()
    row.extra_data = {"previous_conversation": "Customer asked about a withdrawal."}

    assert chat_api._stored_previous_conversation(row) == "Customer asked about a withdrawal."
