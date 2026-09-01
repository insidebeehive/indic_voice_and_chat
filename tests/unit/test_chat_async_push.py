"""Tests for ``push_async_message`` (src/api/chat.py) and the ``chat_websocket``
async-push queue race: a background/webhook-driven flow (deposit-verification
vendor callback, its timeout sweep) needs to speak into a bot-mode chat WS
from outside the normal request/response turn.

Part A exercises ``push_async_message`` directly against the DB layer (no
websocket involved). Part B exercises the real WS loop's
``asyncio.wait({ws_task, aq_task}, FIRST_COMPLETED)`` race, mirroring the
harness in ``test_chat_keepalive.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat as chat_api
from src.models.database import Base
from src.models.chat import ChatMessage, ChatSession
from src.models.tenant import Tenant


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
    llm_provider = None
    llm_model = None


# --- Part A: push_async_message at the DB layer --------------------------


@pytest_asyncio.fixture
async def push_ctx():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(Tenant(id="t1", slug="t1", name="Acme"))
        db.add(ChatSession(id="sess1", tenant_id="t1", language="hi",
                            status="active", mode="ai", extra_data={}))
        await db.commit()

    chat_api.set_chat_sessionmaker(sm)

    yield sm

    chat_api.set_chat_sessionmaker(None)
    # Narrow to the one session id this fixture's tests use, rather than
    # wiping the whole module-level dict.
    chat_api._async_push_queues.pop("sess1", None)
    await engine.dispose()


class _FailingQueue:
    async def put(self, item):
        raise RuntimeError("queue put failed")


@pytest.mark.asyncio
async def test_push_persists_system_message_and_bumps_count_when_no_live_queue(push_ctx):
    sm = push_ctx
    assert "sess1" not in chat_api._async_push_queues

    msg_id = await chat_api.push_async_message("sess1", "verdict text")

    assert isinstance(msg_id, int)
    async with sm() as db:
        row = await db.get(ChatMessage, msg_id)
        session_row = await db.get(ChatSession, "sess1")
    assert row.role == "system"
    assert row.type == "text"
    assert row.content == "verdict text"
    assert session_row.message_count == 1


@pytest.mark.asyncio
async def test_push_returns_none_for_unknown_session_and_writes_nothing(push_ctx):
    sm = push_ctx

    msg_id = await chat_api.push_async_message("no_such_session", "verdict text")

    assert msg_id is None
    async with sm() as db:
        rows = (await db.execute(select(ChatMessage))).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_push_enqueues_frame_when_a_queue_is_registered(push_ctx):
    q = asyncio.Queue()
    chat_api._async_push_queues["sess1"] = q

    msg_id = await chat_api.push_async_message("sess1", "verdict text")

    assert msg_id is not None
    assert q.qsize() == 1
    frame = json.loads(q.get_nowait())
    assert frame == {"type": "async_message", "session_id": "sess1", "text": "verdict text"}


@pytest.mark.asyncio
async def test_push_honours_custom_role_and_frame_type(push_ctx):
    sm = push_ctx
    q = asyncio.Queue()
    chat_api._async_push_queues["sess1"] = q

    msg_id = await chat_api.push_async_message(
        "sess1", "timeout notice", role="human_agent", frame_type="verification_timeout")

    async with sm() as db:
        row = await db.get(ChatMessage, msg_id)
    assert row.role == "human_agent"

    frame = json.loads(q.get_nowait())
    assert frame["type"] == "verification_timeout"
    assert frame["text"] == "timeout notice"
    assert frame["session_id"] == "sess1"


@pytest.mark.asyncio
async def test_push_returns_msg_id_even_when_queue_delivery_fails(push_ctx, caplog):
    sm = push_ctx
    chat_api._async_push_queues["sess1"] = _FailingQueue()

    with caplog.at_level(logging.ERROR, logger="src.api.chat"):
        msg_id = await chat_api.push_async_message("sess1", "verdict text")

    assert isinstance(msg_id, int)
    async with sm() as db:
        row = await db.get(ChatMessage, msg_id)
        session_row = await db.get(ChatSession, "sess1")
    assert row is not None
    assert row.content == "verdict text"
    assert session_row.message_count == 1
    assert "push_async_message queue delivery failed" in caplog.text


@pytest.mark.asyncio
async def test_push_never_raises_on_persistence_failure(push_ctx, monkeypatch, caplog):
    def _boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(chat_api, "_sm", _boom)

    with caplog.at_level(logging.ERROR, logger="src.api.chat"):
        msg_id = await chat_api.push_async_message("sess1", "verdict text")

    assert msg_id is None
    assert "push_async_message persistence failed" in caplog.text


# --- Part B: WS queue race, full integration ------------------------------
#
# Mirrors test_chat_keepalive.py's harness exactly: TestClient runs the app
# in its own thread with its own event loop, so `_async_push_queues["sess1"]`
# is an asyncio.Queue bound to THAT loop. Every push below is triggered from
# a background task scheduled (via asyncio.ensure_future, from inside the
# fake agent's handle_message) onto the APP's own loop -- never awaited
# directly from this test coroutine's separate loop, which would make
# Queue.put's internal call_soon cross-thread and intermittently flaky.


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

    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent._llm = None
    fake_agent.llm = MagicMock()
    fake_agent.session = MagicMock()

    async def fake_factory(tenant, scoped_id, *, customer_id=None, ticket_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, fake_agent

    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    chat_api._async_push_queues.pop("sess1", None)
    chat_api._bo_queues.pop("sess1", None)
    chat_api._customer_queues.pop("sess1", None)
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


@pytest.mark.asyncio
async def test_async_push_lands_mid_conversation_without_a_customer_turn(ws_ctx) -> None:
    sm, fake_agent = ws_ctx
    # The persistence check runs INSIDE the same background task, on the
    # app's own loop, right after the push -- not as a separate query from
    # this test coroutine's own loop after the WS closes. TestClient's
    # websocket runs the app in a different thread with its own loop bound
    # to the single StaticPool sqlite connection; querying that connection
    # from this loop right after cross-thread WS activity has been observed
    # to corrupt it (a later, unrelated query then fails with "no such
    # table"), so verification has to stay on the loop that owns it. The test
    # must also wait for that verification to finish WHILE the WS is still
    # open: TestClient tears down the portal thread/loop once the `with`
    # block exits, which would silently strand (never resume) a background
    # task still in flight at that point. A `threading.Event` (not an
    # asyncio.Event -- this task and this test coroutine run on two
    # different loops, see test_chat_keepalive.py's same convention) lets
    # this test block on that completion from inside the `with` block.
    verified: dict = {}
    done = threading.Event()

    async def _delayed_push():
        await asyncio.sleep(0.05)
        await chat_api.push_async_message("sess1", "VERDICT", role="system")
        async with sm() as db:
            rows = (await db.execute(
                select(ChatMessage).where(
                    ChatMessage.session_id == "sess1", ChatMessage.role == "system")
            )).scalars().all()
        verified["rows"] = [(r.role, r.content) for r in rows]
        done.set()

    async def _handle(text):
        asyncio.ensure_future(_delayed_push())
        return _FakeTurnResult()

    fake_agent.handle_message = _handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            assert json.loads(ws.receive_text())["type"] == "message"

            pushed = json.loads(ws.receive_text())
            assert pushed["type"] == "async_message"
            assert pushed["session_id"] == "sess1"
            assert pushed["text"] == "VERDICT"

            assert await asyncio.to_thread(done.wait, 2.0), "verification task never finished"
    finally:
        patcher.stop()

    assert verified.get("rows") == [("system", "VERDICT")]


@pytest.mark.asyncio
async def test_bot_loop_resumes_after_an_async_push_when_mode_is_unchanged(ws_ctx) -> None:
    sm, fake_agent = ws_ctx

    async def _delayed_push():
        await asyncio.sleep(0.05)
        await chat_api.push_async_message("sess1", "VERDICT", role="system")

    async def _handle(text):
        asyncio.ensure_future(_delayed_push())
        return _FakeTurnResult()

    fake_agent.handle_message = _handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            assert json.loads(ws.receive_text())["type"] == "message"

            pushed = json.loads(ws.receive_text())
            assert pushed["type"] == "async_message"

            ws.send_text(json.dumps({"type": "message", "text": "still there?"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            reply2 = json.loads(ws.receive_text())
            assert reply2["type"] == "message"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_ws_and_async_push_completing_in_same_cycle_forwards_pending_customer_text(ws_ctx) -> None:
    """F5 (race): when `asyncio.wait({ws_task, aq_task}, FIRST_COMPLETED)`
    (chat.py ~1351-1354) comes back with BOTH tasks done in the same cycle --
    a customer message arrives in the very same instant an out-of-band
    escalation push lands -- the already-consumed customer message must be
    forwarded into the BO queue via `_forward_pending_text_to_bo`
    (~1405-1424), not silently dropped when the socket hands off to
    `_run_human_mode` (which starts a fresh `receive_text()` and would never
    see it again).

    Forcing genuine simultaneity through the TestClient-based harness turns
    out to be achievable deterministically -- no arbitrary sleep/timing race
    needed -- by making sure BOTH the client's next WS frame and the pushed
    queue item are already sitting in their respective buffers *before* the
    loop's next iteration creates `ws_task`/`aq_task`:

    - The test sends the customer's second message ("still there?") right
      after the first ("hi"), without reading any replies in between, via
      `WebSocketTestSession.send_text` -> `portal.call(self._receive_tx.send,
      ...)` (starlette.testclient): that lands directly on the app's own
      event loop through an *unbounded* anyio memory stream, so the frame is
      already buffered well before the server gets around to reading it.
    - The fake agent's `handle_message` for the FIRST turn ("hi") -- which
      runs on the app's own loop as part of the WS coroutine itself, so no
      cross-loop/cross-thread hazard -- flips `ChatSession.mode` to
      `"awaiting_human"` and does `async_q.put_nowait(...)` directly, both
      *before* returning its result. `put_nowait` never suspends, so the push
      is fully queued before the bot-turn reply is even sent.

    By the time the main loop finishes the first turn and re-enters the
    `while True` top to build the SECOND iteration's `ws_task`/`aq_task`,
    both `websocket.receive_text()` and `async_q.get()` find their data
    already available and complete in their very first step -- i.e. within
    the same `asyncio.wait()` cycle -- landing exactly in the
    `if aq_task in done: ... if ws_task in done:` branch this test targets.
    """
    sm, fake_agent = ws_ctx

    real_forward = chat_api._forward_pending_text_to_bo
    forward_calls: list[tuple[str, str]] = []
    verified: dict = {}
    done = threading.Event()

    async def _spy_forward(session_id, text):
        # Runs on the app's own loop (it's a direct replacement for the
        # module-level function chat.py calls in-place) -- safe to do the
        # verification DB read here too, for the same cross-loop-corruption
        # reason the other Part B tests' comments explain.
        forward_calls.append((session_id, text))
        await real_forward(session_id, text)
        async with sm() as db:
            rows = (await db.execute(
                select(ChatMessage).where(
                    ChatMessage.session_id == session_id, ChatMessage.role == "customer")
            )).scalars().all()
        verified["messages"] = [(r.role, r.content) for r in rows]
        bq = chat_api._bo_queues.get("sess1")
        verified["bq_frame"] = bq.get_nowait() if bq is not None and not bq.empty() else None
        done.set()

    async def _handle(text):
        if text == "hi":
            async with sm() as db:
                row = await db.get(ChatSession, "sess1")
                row.mode = "awaiting_human"
                await db.commit()
            q = chat_api._async_push_queues["sess1"]
            q.put_nowait(json.dumps({
                "type": "async_message", "session_id": "sess1", "text": "VERDICT",
            }))
        return _FakeTurnResult()

    fake_agent.handle_message = _handle

    patcher, client = _connect(_make_tenant())
    try:
        with patch.object(chat_api, "_forward_pending_text_to_bo", _spy_forward):
            with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
                ws.send_text(json.dumps({"type": "message", "text": "hi"}))
                # Queued NOW, before reading any reply to "hi" -- it must
                # already be sitting in the buffered receive stream by the
                # time the loop's second iteration builds a fresh ws_task
                # (see docstring above).
                ws.send_text(json.dumps({"type": "message", "text": "still there?"}))

                assert json.loads(ws.receive_text())["type"] == "typing"
                assert json.loads(ws.receive_text())["type"] == "message"

                pushed = json.loads(ws.receive_text())
                assert pushed["type"] == "async_message"
                assert pushed["text"] == "VERDICT"

                assert await asyncio.to_thread(done.wait, 2.0), (
                    "_forward_pending_text_to_bo was never called -- the "
                    "ws_task/aq_task same-cycle race path did not trigger"
                )
    finally:
        patcher.stop()

    assert forward_calls == [("sess1", "still there?")]
    # "hi" (the first, normal turn) is also persisted as a customer message
    # by the regular bot-turn flow -- assert membership, not exact equality.
    assert ("customer", "still there?") in (verified.get("messages") or [])
    assert verified.get("bq_frame") is not None
    frame = json.loads(verified["bq_frame"])
    assert frame == {"type": "customer_message", "text": "still there?", "session_id": "sess1"}


@pytest.mark.asyncio
async def test_async_push_after_out_of_band_escalation_hands_off_to_human_mode(ws_ctx) -> None:
    sm, fake_agent = ws_ctx
    # As above (test 7): the final DB check runs from inside the same
    # app-loop background task, and this test waits for it to finish via a
    # threading.Event while still inside the `with` block -- both to avoid
    # the cross-loop connection corruption and because the portal thread is
    # torn down as soon as the `with` block exits.
    verified: dict = {}
    done = threading.Event()

    async def _delayed_escalate_and_decline():
        await asyncio.sleep(0.05)
        async with sm() as db:
            row = await db.get(ChatSession, "sess1")
            row.mode = "awaiting_human"
            await db.commit()
        await chat_api.push_async_message("sess1", "VERDICT", role="system")
        await asyncio.sleep(0.05)
        cq = chat_api._customer_queues.setdefault("sess1", asyncio.Queue())
        await cq.put(json.dumps({"type": "declined"}))
        await asyncio.sleep(0.1)
        async with sm() as db:
            final_row = await db.get(ChatSession, "sess1")
        verified["mode"] = final_row.mode if final_row else None
        done.set()

    async def _handle(text):
        asyncio.ensure_future(_delayed_escalate_and_decline())
        return _FakeTurnResult()

    fake_agent.handle_message = _handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            assert json.loads(ws.receive_text())["type"] == "message"

            pushed = json.loads(ws.receive_text())
            assert pushed["type"] == "async_message"
            assert pushed["text"] == "VERDICT"

            mode_change = json.loads(ws.receive_text())
            assert mode_change["type"] == "mode_change"
            assert mode_change["mode"] == "bot"

            bot_msg = json.loads(ws.receive_text())
            assert bot_msg["type"] == "message"

            assert await asyncio.to_thread(done.wait, 2.0), "verification task never finished"
    finally:
        patcher.stop()

    assert verified.get("mode") == "bot"


@pytest.mark.asyncio
async def test_queue_is_removed_from_async_push_queues_on_disconnect(ws_ctx) -> None:
    sm, fake_agent = ws_ctx

    async def _handle(text):
        return _FakeTurnResult()

    fake_agent.handle_message = _handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            assert json.loads(ws.receive_text())["type"] == "message"
            assert "sess1" in chat_api._async_push_queues
    finally:
        patcher.stop()

    await asyncio.sleep(0.2)
    assert "sess1" not in chat_api._async_push_queues


@pytest.mark.asyncio
async def test_reconnect_installs_a_fresh_queue_object(ws_ctx) -> None:
    sm, fake_agent = ws_ctx
    # Hold a strong reference to each captured queue (not just its id()) --
    # otherwise the first queue can be garbage-collected once the first
    # connection's `finally` pops it, and CPython is then free to reuse the
    # same memory address (and therefore the same id()) for the second
    # connection's queue, producing a false-positive "same object" failure.
    seen: list[asyncio.Queue] = []

    async def _capture_queue():
        await asyncio.sleep(0.02)
        q = chat_api._async_push_queues.get("sess1")
        seen.append(q)

    async def _handle(text):
        asyncio.ensure_future(_capture_queue())
        return _FakeTurnResult()

    fake_agent.handle_message = _handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            assert json.loads(ws.receive_text())["type"] == "message"
            await asyncio.sleep(0.05)

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi again"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            assert json.loads(ws.receive_text())["type"] == "message"
            await asyncio.sleep(0.05)
    finally:
        patcher.stop()

    assert len(seen) == 2
    assert seen[0] is not None and seen[1] is not None
    assert seen[0] is not seen[1]
