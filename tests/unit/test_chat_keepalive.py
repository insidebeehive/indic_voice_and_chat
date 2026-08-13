"""Long-turn WS keepalive (``typing`` heartbeat) added to guard against the
CRM's downstream relay declaring the connection dead (1006) during a turn
that makes sequential CRM tool calls and goes tens of seconds without any
application traffic on the socket.

Covers:
- ``_typing_keepalive`` / ``_run_turn_with_keepalive`` wired into the real WS
  message loop (periodic frames during a slow turn, none after the reply,
  none after an error, no change to the fast-turn frame sequence).
- Direct unit tests of ``_run_turn_with_keepalive`` proving no leaked asyncio
  tasks and that a keepalive send failure never masks the turn's result.
"""

from __future__ import annotations

import asyncio
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
from src.models.chat import ChatSession


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
    # _persist_turn (chat-cost-tracking) reads these; llm_provider=None keeps
    # it out of the cost-computation branch entirely.
    input_tokens = 0
    output_tokens = 0
    llm_provider = None
    llm_model = None


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

    async def fake_factory(tenant, scoped_id, *, customer_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, fake_agent

    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


class _FakeMediaStore:
    """Upload that can be made artificially slow, to stand in for a large
    voice-note/video upload stalling the pre-turn media window (Bug 2)."""

    def __init__(self, delay: float = 0.0) -> None:
        self.uploaded: list[tuple] = []
        self._delay = delay

    async def upload(self, data, key, content_type) -> None:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.uploaded.append((key, content_type, data))


@pytest_asyncio.fixture
async def ws_media_ctx():
    """Like ``ws_ctx`` but with a chatbot agent that also answers to
    ``handle_image``/``llm.transcribe_audio``, for the audio and image/video
    pre-turn-media-window keepalive coverage tests (Bug 2)."""
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
    fake_agent.llm.transcribe_audio = AsyncMock(return_value="hello there")
    fake_agent.handle_message = AsyncMock(return_value=_FakeTurnResult())
    fake_agent.handle_image = AsyncMock(return_value=_FakeTurnResult())
    fake_agent.session = MagicMock()

    async def fake_factory(tenant, scoped_id, *, customer_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, fake_agent

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


def _make_tenant():
    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300
    fake_tenant.settings.events_webhook_url = None  # skip the best-effort webhook call
    return fake_tenant


def _connect(fake_tenant):
    import src.auth.middleware as mw

    patcher = patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant))
    patcher.start()
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    client = TestClient(app)
    return patcher, client


# --- WS-level: keepalive wired into the real message loop ----------------


@pytest.mark.asyncio
async def test_long_turn_sends_periodic_typing(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.05)

    async def _slow_handle(text):
        await asyncio.sleep(0.4)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            typing_count = 0
            frame = json.loads(ws.receive_text())
            while frame["type"] == "typing":
                typing_count += 1
                frame = json.loads(ws.receive_text())
            assert typing_count >= 3, f"expected >=3 typing frames, got {typing_count}"
            assert frame["type"] == "message"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_no_typing_frames_after_the_reply(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.05)

    async def _slow_handle(text):
        await asyncio.sleep(0.15)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            while frame["type"] == "typing":
                frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"

            # Several keepalive intervals' worth of real time with the turn
            # already finished — no frame should show up on its own.
            await asyncio.sleep(0.3)
            ws.send_text(json.dumps({"type": "end"}))
            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_keepalive_stops_on_turn_error(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.05)

    async def _slow_fail(text):
        await asyncio.sleep(0.2)
        raise RuntimeError("boom")

    fake_agent.handle_message = _slow_fail

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            typing_count = 0
            frame = json.loads(ws.receive_text())
            while frame["type"] == "typing":
                typing_count += 1
                frame = json.loads(ws.receive_text())
            assert typing_count >= 1
            assert frame["type"] == "error"

            # No stray typing frame should arrive after the error, even after
            # waiting through several keepalive intervals.
            await asyncio.sleep(0.3)
            ws.send_text(json.dumps({"type": "end"}))
            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended"
    finally:
        patcher.stop()


# --- Regression: fast turns unchanged (default 8.0s interval never fires) -


@pytest.mark.asyncio
async def test_fast_turn_frame_sequence_unchanged(ws_ctx) -> None:
    sm, fake_agent = ws_ctx

    async def _fast_handle(text):
        return _FakeTurnResult()

    fake_agent.handle_message = _fast_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            message = json.loads(ws.receive_text())
            assert message["type"] == "message"

            ws.send_text(json.dumps({"type": "end"}))
            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended"
    finally:
        patcher.stop()


# --- WS-level: keepalive covers the pre-turn media window (Bug 2) --------


@pytest.mark.asyncio
async def test_audio_keepalive_covers_pre_turn_media_window(ws_media_ctx, monkeypatch) -> None:
    """A slow S3 upload/transcription — BEFORE the LLM turn even starts — must
    still get keepalive frames. Regression for Bug 2: previously the audio
    branch only wrapped the turn call, leaving this window silent."""
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.05)
    media_store = _FakeMediaStore(delay=0.3)
    chat_api.set_media_store(media_store)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            audio_bytes = b"fake audio bytes"
            encoded = base64.b64encode(audio_bytes).decode()
            ws.send_text(json.dumps({
                "type": "audio", "data": encoded, "mime": "audio/webm",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"

            # The upload delay (0.3s) hasn't reached the LLM turn yet — a
            # keepalive frame arriving here proves coverage extends into the
            # pre-turn media window, not just the turn itself.
            second = json.loads(ws.receive_text())
            assert second["type"] == "typing", "no keepalive frame during pre-turn media upload"

            frame = second
            typing_count = 1
            while frame["type"] == "typing":
                typing_count += 1
                frame = json.loads(ws.receive_text())
            assert typing_count >= 2
            assert frame["type"] == "audio_ack"
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"
    finally:
        patcher.stop()

    assert len(media_store.uploaded) == 1


@pytest.mark.asyncio
async def test_image_keepalive_covers_pre_turn_media_window(ws_media_ctx, monkeypatch) -> None:
    """A slow media_url fetch — BEFORE the LLM turn even starts — must still
    get keepalive frames. Regression for Bug 2: previously the image/video
    branch only wrapped the turn call, leaving this window silent."""
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.05)

    async def _slow_fetch(url):
        await asyncio.sleep(0.3)
        return b"fake image bytes", "image/jpeg"

    patcher, client = _connect(_make_tenant())
    try:
        with patch.object(chat_api, "_fetch_media_url", _slow_fetch):
            with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
                ws.send_text(json.dumps({
                    "type": "image", "media_url": "https://example.com/pic.jpg",
                }))
                typing = json.loads(ws.receive_text())
                assert typing["type"] == "typing"

                second = json.loads(ws.receive_text())
                assert second["type"] == "typing", "no keepalive frame during pre-turn media fetch"

                frame = second
                typing_count = 1
                while frame["type"] == "typing":
                    typing_count += 1
                    frame = json.loads(ws.receive_text())
                assert typing_count >= 2
                assert frame["type"] == "message"
    finally:
        patcher.stop()

    fake_agent.handle_image.assert_awaited_once()


# --- Regression: keepalive must not survive into human handoff (Bug 1) ----
#
# `_run_human_mode` is an unbounded loop for the rest of a human agent's
# conversation with the customer. The audio and image/video branches used to
# stop their branch-level keepalive only in the `finally` wrapping the whole
# branch body — which doesn't run until `_run_human_mode` itself returns, so
# a `typing` frame kept firing every `_KEEPALIVE_INTERVAL_S` for the entire
# human conversation. `_handle_escalation`/`_run_human_mode` are faked out
# here (their own DB/webhook/queue plumbing is exercised elsewhere) so these
# tests isolate exactly the thing that regressed: is the real keepalive task
# stopped *before* the potentially-unbounded human-mode wait starts, not only
# after it ends.
#
# The fake `_run_human_mode` never sends anything and the real handler
# doesn't explicitly close the socket on a plain `break` (Starlette only
# closes on an explicit `websocket.close()` or an exception), so draining
# "until disconnect" would hang forever on a passing (fixed) run. Instead,
# drain for a bounded window using a background thread so a `receive_text()`
# call that would otherwise block forever can be timed out.


async def _drain_for(ws, seconds: float) -> list[dict]:
    """Collect whatever text frames arrive on `ws` within `seconds`, without
    blocking past the deadline if nothing (more) ever arrives."""
    frames: list[dict] = []
    loop = asyncio.get_event_loop()
    deadline = loop.time() + seconds
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(asyncio.to_thread(ws.receive_text), timeout=remaining)
        except (asyncio.TimeoutError, Exception):
            break
        frames.append(json.loads(raw))
    return frames


@pytest.mark.asyncio
async def test_no_typing_during_human_handoff_after_image_escalation(ws_media_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.05)

    fake_result = _FakeTurnResult()
    fake_result.escalation = {"reason": "user needs help", "summary": "refund query"}
    fake_agent.handle_image = AsyncMock(return_value=fake_result)

    async def _fake_handle_escalation(websocket, session_id, tenant, row, result):
        await websocket.send_text(json.dumps({"type": "mode_change", "mode": "awaiting_human"}))
        return True

    async def _fake_human_mode(websocket, session_id, tenant):
        # Stand-in for "the rest of a human agent's conversation": several
        # keepalive intervals' worth of real time with the WS otherwise idle
        # — exactly the window that used to leak `typing` frames.
        await asyncio.sleep(0.3)
        return True  # session ended normally, matching the real return contract

    monkeypatch.setattr(chat_api, "_handle_escalation", _fake_handle_escalation)
    monkeypatch.setattr(chat_api, "_run_human_mode", _fake_human_mode)

    patcher, client = _connect(_make_tenant())
    frames: list[dict] = []
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "image", "data": base64.b64encode(b"fake image bytes").decode(),
                "mime": "image/jpeg",
            }))
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            while frame["type"] != "mode_change":
                frame = json.loads(ws.receive_text())
                frames.append(frame)

            # `_run_human_mode` (faked above) is "in the human conversation"
            # for 0.3s — several keepalive intervals. Drain a bit longer than
            # that to catch any leaked `typing` frame.
            frames += await _drain_for(ws, 0.5)
    finally:
        patcher.stop()

    types = [f["type"] for f in frames]
    mode_change_idx = types.index("mode_change")
    assert "typing" not in types[mode_change_idx + 1:], (
        f"typing frame leaked during human handoff: {types}"
    )


@pytest.mark.asyncio
async def test_no_typing_during_human_handoff_after_audio_escalation(ws_media_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.05)

    fake_result = _FakeTurnResult()
    fake_result.escalation = {"reason": "user needs help", "summary": "refund query"}
    fake_agent.handle_message = AsyncMock(return_value=fake_result)
    chat_api.set_media_store(_FakeMediaStore())

    async def _fake_handle_escalation(websocket, session_id, tenant, row, result):
        await websocket.send_text(json.dumps({"type": "mode_change", "mode": "awaiting_human"}))
        return True

    async def _fake_human_mode(websocket, session_id, tenant):
        await asyncio.sleep(0.3)
        return True

    monkeypatch.setattr(chat_api, "_handle_escalation", _fake_handle_escalation)
    monkeypatch.setattr(chat_api, "_run_human_mode", _fake_human_mode)

    patcher, client = _connect(_make_tenant())
    frames: list[dict] = []
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            audio_bytes = b"fake audio bytes"
            ws.send_text(json.dumps({
                "type": "audio", "data": base64.b64encode(audio_bytes).decode(),
                "mime": "audio/webm",
            }))
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            while frame["type"] != "mode_change":
                frame = json.loads(ws.receive_text())
                frames.append(frame)

            frames += await _drain_for(ws, 0.5)
    finally:
        patcher.stop()

    types = [f["type"] for f in frames]
    mode_change_idx = types.index("mode_change")
    assert "typing" not in types[mode_change_idx + 1:], (
        f"typing frame leaked during human handoff: {types}"
    )


# --- Direct unit tests of _run_turn_with_keepalive ------------------------


class _FakeWS:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self._fail = fail

    async def send_text(self, s: str) -> None:
        if self._fail:
            raise RuntimeError("send failed — client gone")
        self.sent.append(s)


@pytest.mark.asyncio
async def test_keepalive_does_not_leak_tasks(monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.02)
    fake_ws = _FakeWS()

    async def _slow_ok():
        await asyncio.sleep(0.08)
        return "turn-result"

    current = asyncio.current_task()
    result = await chat_api._run_turn_with_keepalive(fake_ws, _slow_ok())
    assert result == "turn-result"

    sent_after_return = len(fake_ws.sent)
    await asyncio.sleep(0.1)  # several keepalive intervals after the turn ended
    assert len(fake_ws.sent) == sent_after_return, "keepalive kept sending after the turn finished"

    remaining = asyncio.all_tasks() - {current}
    assert not remaining, f"leaked asyncio tasks: {remaining}"


@pytest.mark.asyncio
async def test_keepalive_send_failure_does_not_mask_result(monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.02)
    fake_ws = _FakeWS(fail=True)

    async def _slow_ok():
        await asyncio.sleep(0.08)
        return "turn-result"

    result = await chat_api._run_turn_with_keepalive(fake_ws, _slow_ok())
    assert result == "turn-result"


class _WedgedWS:
    """A websocket whose ``send_text`` blocks forever (until released),
    simulating a socket write that never completes — the scenario that puts
    ``_run_turn_with_keepalive``'s ``finally`` block into its own
    ``await asyncio.wait({ka}, timeout=_KEEPALIVE_DRAIN_S)`` with ``ka`` still
    genuinely in flight."""

    def __init__(self) -> None:
        self.send_started = asyncio.Event()
        self.release = asyncio.Event()

    async def send_text(self, s: str) -> None:
        self.send_started.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_outer_cancellation_during_drain_propagates(monkeypatch) -> None:
    """Bug 1 regression.

    Proves that a cancellation of the *enclosing* task, landing while
    ``_run_turn_with_keepalive``'s ``finally`` block is itself inside its own
    drain await (``await asyncio.wait({ka}, timeout=_KEEPALIVE_DRAIN_S)``),
    genuinely propagates out instead of being silently absorbed by the old
    bare ``except BaseException: pass``.

    Setup: the keepalive's ``send_text`` blocks on an event, so `ka` is
    guaranteed to still be mid-send (not done) once the turn coroutine
    finishes and control enters the `finally` block. A large
    ``_KEEPALIVE_DRAIN_S`` keeps the drain `await` open long enough for the
    test to cancel the enclosing task from outside while execution is inside
    it. If the bug were still present, the enclosing task would swallow the
    cancellation and complete normally with ``"turn-result"``; with the fix,
    ``await task`` raises ``CancelledError`` and ``task.cancelled()`` is True.
    """
    monkeypatch.setattr(chat_api, "_KEEPALIVE_INTERVAL_S", 0.01)
    monkeypatch.setattr(chat_api, "_KEEPALIVE_DRAIN_S", 2.0)

    fake_ws = _WedgedWS()

    async def _turn():
        # Give the keepalive time to fire once and get stuck inside
        # send_text before the turn itself finishes, so `ka` is guaranteed
        # mid-send when `finally` starts its drain wait.
        await asyncio.sleep(0.05)
        return "turn-result"

    task = asyncio.ensure_future(chat_api._run_turn_with_keepalive(fake_ws, _turn()))
    try:
        await asyncio.wait_for(fake_ws.send_started.wait(), timeout=1.0)
        # Margin over the turn's own 0.05s sleep so it has definitely
        # returned and `finally` has definitely entered its drain await by
        # the time we cancel.
        await asyncio.sleep(0.15)
        assert not task.done(), "test setup assumption broken: task finished before the drain window"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled(), "outer cancellation was swallowed instead of propagating"
    finally:
        fake_ws.release.set()
        # Let the (now-cancelled) keepalive task settle so nothing leaks
        # into the next test.
        await asyncio.sleep(0)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_outer_cancellation_during_reap_propagates(monkeypatch) -> None:
    """Bug 2 regression.

    Targets the REAP step specifically (the final ``await ka``/
    ``await asyncio.wait({ka})`` in ``_stop_keepalive``, reached *after* the
    drain has already timed out and called ``ka.cancel()``) — not the drain
    step, which ``test_outer_cancellation_during_drain_propagates`` already
    covers.

    With the pre-fix bare ``await ka`` reap: cancelling the enclosing task
    while it is suspended in ``await ka`` makes CPython deliver that
    cancellation by cancelling `ka` itself (since `ka` is the task's current
    `_fut_waiter`) — so by the time `await ka` raises, `ka.cancelled()` is
    already True, and the old guard (``if not ka.cancelled(): raise``) reads
    that as "this is ka's own cancellation, not ours" and swallows it. The
    enclosing task then completes normally instead of ending up cancelled.

    With the fix (``await asyncio.wait({ka})`` for the reap): the enclosing
    task's `_fut_waiter` is `asyncio.wait`'s own internal waiter future, not
    `ka` — so cancelling the enclosing task never touches `ka`, and the
    `CancelledError` raised out of that `asyncio.wait` call is unambiguously
    ours and propagates untouched (no `except` wraps that second call).

    Setup: `_KEEPALIVE_DRAIN_S` is patched tiny so the drain step times out
    almost immediately and calls `ka.cancel()`. `ka` itself is a fake task
    that, once cancelled, spends a real 0.2s in a `finally`/`except` block
    before actually finishing — a real multi-tick gap between `ka.cancel()`
    being called and `ka` reaching `done()`, which is exactly the window the
    REAP step (not the drain step) sits in. The enclosing `_stop_keepalive`
    call is cancelled from outside while parked in that window.
    """
    monkeypatch.setattr(chat_api, "_KEEPALIVE_DRAIN_S", 0.01)

    async def _slow_to_cancel() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # Real cleanup work that takes multiple event-loop ticks — keeps
            # `ka` un-done for a real window after `ka.cancel()` is called.
            await asyncio.sleep(0.2)
            raise

    stop = asyncio.Event()
    ka = asyncio.ensure_future(_slow_to_cancel())
    await asyncio.sleep(0)  # let ka actually start running (parked in sleep(10))

    async def _call_stop() -> None:
        await chat_api._stop_keepalive(ka, stop)

    task = asyncio.ensure_future(_call_stop())
    try:
        # Past the drain timeout (0.01s, so ka.cancel() has already been
        # called) but well within ka's own 0.2s cleanup window — i.e. inside
        # the reap step's `await asyncio.wait({ka})`, with `ka` still
        # genuinely unwinding, not yet done.
        await asyncio.sleep(0.08)
        assert not task.done(), "test setup assumption broken: reap finished before we could cancel the outer task"
        assert not ka.done(), "test setup assumption broken: ka finished before we could cancel the outer task"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert task.cancelled(), "outer cancellation was swallowed instead of propagating (reap path)"
    finally:
        # Let ka finish unwinding so nothing leaks into the next test.
        try:
            await ka
        except asyncio.CancelledError:
            pass
