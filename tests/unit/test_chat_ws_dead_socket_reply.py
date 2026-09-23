"""A slow turn (e.g. a 32s CRM tool call) can outlast the customer's socket:
the client disconnects and reconnects mid-turn, then the turn finishes and
`_send_reply` tries to write to the now-dead connection. Starlette raises a
plain ``RuntimeError('Cannot call "send" once a close message has been
sent.')`` from that write, which used to escape `_send_reply` entirely,
getting logged twice (`chat turn failed`, then `chat websocket crashed`) and
killing the connection handler -- even though the reply was already
persisted to `chat_messages` by the `_persist_turn` call immediately above
every `_send_reply` call site, so the customer gets it from history on
reconnect regardless.

Covers:
- `_is_dead_socket_send_error`'s narrow predicate (type + exact message +
  `application_state`, not a blanket ``except RuntimeError``).
- `_send_reply` returning False (not raising) on a dead socket, logging a
  WARNING that names the session, and stopping before any later frame in the
  same call raises the identical error again.
- A genuine ``RuntimeError`` from inside `_send_reply` -- not the closed
  socket case -- still propagates. This is the regression guard for the
  predicate: a naive `except RuntimeError: return False` would silently eat
  a real bug in frame construction.
- The normal path is unchanged: a live socket still gets the message frame,
  and escalation/call_offer frames still follow when present.
- End to end through the real WS route: a dead-socket reply does not raise
  out of `chat_websocket`, does not log `chat turn failed` or `chat
  websocket crashed`, and the turn is still visible in `chat_messages`.
- What still happens when the socket is gone: `_end_session` /
  `_send_close_webhook` (bookkeeping the customer's reconnect and the
  tenant's CRM depend on) still run for an agent-resolved turn even though
  the closing `ended` frame could not be delivered.
"""

from __future__ import annotations

import json
import logging
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from fastapi import FastAPI
from fastapi.testclient import TestClient
import starlette.websockets as sw

from src.api import chat as chat_api
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base

_CLOSED_SOCKET_MESSAGE = 'Cannot call "send" once a close message has been sent.'


# --- _is_dead_socket_send_error: the narrow predicate --------------------


class _WS:
    def __init__(self, application_state=sw.WebSocketState.CONNECTED):
        self.application_state = application_state


def test_predicate_matches_the_exact_closed_socket_error():
    ws = _WS(sw.WebSocketState.DISCONNECTED)
    exc = RuntimeError(_CLOSED_SOCKET_MESSAGE)
    assert chat_api._is_dead_socket_send_error(ws, exc) is True


def test_predicate_rejects_a_different_runtime_error_even_when_disconnected():
    """The message must match too -- a socket that happens to be marked
    DISCONNECTED for some other reason must not make an unrelated
    RuntimeError look like a routine disconnect."""
    ws = _WS(sw.WebSocketState.DISCONNECTED)
    exc = RuntimeError("something else entirely went wrong")
    assert chat_api._is_dead_socket_send_error(ws, exc) is False


def test_predicate_rejects_the_right_message_when_still_connected():
    """The state must agree too -- belt and suspenders against a coincidental
    string match."""
    ws = _WS(sw.WebSocketState.CONNECTED)
    exc = RuntimeError(_CLOSED_SOCKET_MESSAGE)
    assert chat_api._is_dead_socket_send_error(ws, exc) is False


def test_predicate_rejects_non_runtime_error():
    ws = _WS(sw.WebSocketState.DISCONNECTED)
    assert chat_api._is_dead_socket_send_error(ws, ValueError(_CLOSED_SOCKET_MESSAGE)) is False


# --- Direct _send_reply tests ---------------------------------------------


class _FakeResponse:
    def __init__(self, *, action: str = "none"):
        self.response_text = "here is your answer"
        self.sources_used = []
        self.suggested_followups = []
        self.action = action


class _FakeTurnResult:
    def __init__(self, *, action="none", escalation=None, call_offer=None):
        self.response = _FakeResponse(action=action)
        self.escalation = escalation
        self.call_offer = call_offer


class _FakeWS:
    """Direct stand-in for a starlette WebSocket -- only what `_send_reply`
    touches: `send_text` and `application_state`."""

    def __init__(self, *, fail_on_call: int | None = None, fail_message: str = _CLOSED_SOCKET_MESSAGE):
        self.sent: list[str] = []
        self.application_state = sw.WebSocketState.CONNECTED
        self._fail_on_call = fail_on_call
        self._fail_message = fail_message
        self._calls = 0

    async def send_text(self, data: str) -> None:
        self._calls += 1
        if self._fail_on_call is not None and self._calls == self._fail_on_call:
            self.application_state = sw.WebSocketState.DISCONNECTED
            raise RuntimeError(self._fail_message)
        self.sent.append(data)


async def test_send_reply_dead_socket_returns_false_and_warns(caplog):
    ws = _FakeWS(fail_on_call=1)
    result = _FakeTurnResult()
    with caplog.at_level(logging.WARNING, logger="src.api.chat"):
        delivered = await chat_api._send_reply(
            ws, "sess1", result, "t1", ticket_id="tix1")
    assert delivered is False
    assert ws.sent == []  # the one frame never landed
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].session_id == "sess1"
    assert warnings[0].ticket_id == "tix1"
    assert "persisted" in warnings[0].message or "persisted" in warnings[0].getMessage()


async def test_send_reply_stops_after_first_dead_frame_no_double_raise():
    """Escalation frame would also hit the closed-socket error -- must not
    be attempted (and must not raise) once the message frame already found
    the socket gone."""
    ws = _FakeWS(fail_on_call=1)
    result = _FakeTurnResult(escalation={"reason": "r", "summary": "s"})
    delivered = await chat_api._send_reply(ws, "sess1", result, "t1")
    assert delivered is False
    assert ws.sent == []


async def test_send_reply_genuine_runtime_error_propagates():
    """The regression guard: a RuntimeError that is NOT the closed-socket
    message must still raise out of `_send_reply` -- proving the predicate
    is not a blanket `except RuntimeError`."""
    ws = _FakeWS(fail_on_call=1, fail_message="boom, this is a real bug")
    result = _FakeTurnResult()
    with pytest.raises(RuntimeError, match="boom, this is a real bug"):
        await chat_api._send_reply(ws, "sess1", result, "t1")


async def test_send_reply_same_message_but_still_connected_propagates():
    """Same closed-socket wording, but `application_state` never actually
    flipped to DISCONNECTED -- the state half of the predicate must also
    hold, or a coincidental message match would swallow a real error."""
    class _StillConnectedWS(_FakeWS):
        async def send_text(self, data: str) -> None:
            self._calls += 1
            if self._calls == 1:
                # Deliberately do NOT flip application_state here.
                raise RuntimeError(_CLOSED_SOCKET_MESSAGE)
            self.sent.append(data)

    ws = _StillConnectedWS()
    result = _FakeTurnResult()
    with pytest.raises(RuntimeError, match="Cannot call"):
        await chat_api._send_reply(ws, "sess1", result, "t1")


async def test_send_reply_normal_path_sends_message_escalation_and_call_offer():
    ws = _FakeWS()
    result = _FakeTurnResult(
        escalation={"reason": "wants human", "summary": "summary text"},
        call_offer={"reason": "offer a call"},
    )
    delivered = await chat_api._send_reply(
        ws, "sess1", result, "t1", call_url="wss://example/call/abc")
    assert delivered is True
    assert len(ws.sent) == 3
    message_frame = json.loads(ws.sent[0])
    assert message_frame["type"] == "message"
    assert message_frame["text"] == "here is your answer"
    escalation_frame = json.loads(ws.sent[1])
    assert escalation_frame["type"] == "escalation"
    assert escalation_frame["reason"] == "wants human"
    call_offer_frame = json.loads(ws.sent[2])
    assert call_offer_frame["type"] == "call_offer"
    assert call_offer_frame["call_url"] == "wss://example/call/abc"


# --- End to end through the real WS route ---------------------------------


class _RouteTurnResult:
    """`_persist_turn`-shaped result for the full-route tests below --
    mirrors tests/unit/test_chat_keepalive.py's `_FakeTurnResult`."""

    class _Resp:
        response_text = "the answer, after a slow tool call"
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


class _ResolvedTurnResult(_RouteTurnResult):
    class _Resp(_RouteTurnResult._Resp):
        action = "resolved"

    response = _Resp()


@pytest.fixture
async def ws_ctx():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(ChatSession(id="sess1", tenant_id="t1", language="en",
                            status="active", mode="ai", extra_data={}))
        await db.commit()

    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent.handle_message = AsyncMock(return_value=_RouteTurnResult())
    fake_agent.summarize_session = AsyncMock(return_value="a summary")
    fake_agent._llm = None
    fake_agent.llm = MagicMock()
    fake_agent.session = MagicMock()

    async def fake_factory(tenant, scoped_id, *, customer_id=None, ticket_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, fake_agent

    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


def _dying_send_text(fail_at: int, message: str = _CLOSED_SOCKET_MESSAGE):
    """Monkeypatch target for `starlette.websockets.WebSocket.send_text`
    (identical to `fastapi.WebSocket`) that behaves normally except on the
    `fail_at`-th call, where it raises `message` and -- only for the
    closed-socket message -- flips `application_state` to DISCONNECTED
    exactly as Starlette's real `send()` does when it takes that branch."""
    orig = sw.WebSocket.send_text
    state = {"n": 0}

    async def _send_text(self, data):
        state["n"] += 1
        if state["n"] == fail_at:
            if message == _CLOSED_SOCKET_MESSAGE:
                self.application_state = sw.WebSocketState.DISCONNECTED
            raise RuntimeError(message)
        return await orig(self, data)

    return _send_text, state


def _wait_for_call_count(state: dict, n: int, timeout: float = 5.0) -> None:
    """Block (main test thread) until the background ASGI app thread's
    `send_text` call counter reaches `n`.

    Needed whenever the frame we're waiting on never actually reaches the
    TestClient transport (the whole point of the dead-socket scenario: the
    patched `send_text` raises before calling through) -- with nothing to
    `ws.receive_text()`, exiting `client.websocket_connect(...)`'s `with`
    block immediately after the prior frame races the server-side turn and
    tears the connection down (cancelling the app task) before it ever
    attempts the failing send."""
    deadline = time.monotonic() + timeout
    while state["n"] < n and time.monotonic() < deadline:
        time.sleep(0.01)
    assert state["n"] >= n, f"send_text was only called {state['n']} times, expected >= {n}"


def _fake_tenant():
    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300
    fake_tenant.settings.events_webhook_url = None
    return fake_tenant


async def test_ws_route_dead_socket_reply_does_not_raise_and_turn_is_persisted(
    ws_ctx, monkeypatch, caplog,
):
    sm, fake_agent = ws_ctx
    # Call 1 = the "typing" frame (succeeds), call 2 = _send_reply's message
    # frame (fails as a dead socket).
    send_text, state = _dying_send_text(fail_at=2)
    monkeypatch.setattr(sw.WebSocket, "send_text", send_text)

    import src.auth.middleware as mw

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=_fake_tenant())):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with caplog.at_level(logging.WARNING, logger="src.api.chat"):
            with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
                ws.send_text(json.dumps({"type": "message", "text": "hi"}))
                typing = json.loads(ws.receive_text())
                assert typing["type"] == "typing"
                # The reply frame never actually reaches the transport (the
                # patched send_text raises before calling through) -- wait
                # for the server side to have attempted it before letting the
                # `with` block exit, or exiting here races the server's own
                # turn and tears the connection down first.
                _wait_for_call_count(state, 2)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(r.session_id == "sess1" for r in warnings), (
        "expected a WARNING naming the session for the undelivered reply")

    async with sm() as db:
        from sqlalchemy import select
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "sess1")
            .order_by(ChatMessage.id)
        )).scalars().all()
    assert [r.role for r in rows] == ["customer", "agent"], (
        "the turn must be persisted even though live delivery failed")
    assert rows[1].content == "the answer, after a slow tool call"


async def test_ws_route_dead_socket_reply_logs_neither_turn_failed_nor_crashed(
    ws_ctx, monkeypatch, caplog,
):
    sm, fake_agent = ws_ctx
    send_text, state = _dying_send_text(fail_at=2)
    monkeypatch.setattr(sw.WebSocket, "send_text", send_text)

    import src.auth.middleware as mw

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=_fake_tenant())):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="src.api.chat"):
            with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
                ws.send_text(json.dumps({"type": "message", "text": "hi"}))
                json.loads(ws.receive_text())  # typing
                _wait_for_call_count(state, 2)

    messages = [r.getMessage() for r in caplog.records]
    assert not any("chat turn failed" in m for m in messages)
    assert not any("chat websocket crashed" in m for m in messages)


async def test_ws_route_genuine_runtime_error_still_surfaces_as_turn_failed(
    ws_ctx, monkeypatch, caplog,
):
    """Regression guard end to end: a real bug in frame construction (not a
    closed socket) must still be caught by the existing per-turn handler and
    reported as `chat turn failed`, not silently treated as a routine
    disconnect."""
    sm, fake_agent = ws_ctx
    send_text, _ = _dying_send_text(fail_at=2, message="boom, a real bug")
    monkeypatch.setattr(sw.WebSocket, "send_text", send_text)

    import src.auth.middleware as mw

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=_fake_tenant())):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with caplog.at_level(logging.ERROR, logger="src.api.chat"):
            with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
                ws.send_text(json.dumps({"type": "message", "text": "hi"}))
                json.loads(ws.receive_text())  # typing
                err = json.loads(ws.receive_text())
                assert err["type"] == "error"
                assert err["reason"] == "internal"

    messages = [r.getMessage() for r in caplog.records]
    assert any("chat turn failed" in m for m in messages)


async def test_ws_route_resolved_action_dead_socket_still_ends_session(
    ws_ctx, monkeypatch,
):
    """Pin the decision on what still happens when the socket is gone:
    `_end_session` (marks the ChatSession row ended) and `_send_close_webhook`
    (notifies the tenant's CRM) are bookkeeping the customer's reconnect and
    the CRM depend on, not socket writes -- they must still run even though
    the closing `ended` frame cannot be delivered."""
    sm, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(return_value=_ResolvedTurnResult())
    send_text, state = _dying_send_text(fail_at=2)
    monkeypatch.setattr(sw.WebSocket, "send_text", send_text)

    webhook_mock = AsyncMock(return_value=True)
    monkeypatch.setattr("src.api.chat_webhooks.send_bo_webhook", webhook_mock)

    import src.auth.middleware as mw

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=_fake_tenant())):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            json.loads(ws.receive_text())  # typing
            # No "ended" frame arrives -- the socket is gone -- but nothing
            # raises inside this block either. Wait not just for the failed
            # send but for the bookkeeping AFTER it (_end_session /
            # send_bo_webhook) to actually run, or exiting the `with` block
            # races the server's still-in-flight turn and cancels it early.
            _wait_for_call_count(state, 2)
            deadline = time.monotonic() + 5.0
            while not webhook_mock.called and time.monotonic() < deadline:
                time.sleep(0.01)

    webhook_mock.assert_awaited_once()
    async with sm() as db:
        row = await db.get(ChatSession, "sess1")
    assert row.status == "ended"
    assert row.mode == "closed"
