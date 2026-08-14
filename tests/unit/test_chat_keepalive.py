"""Long-turn WS keepalive added to guard against the CRM's downstream relay
declaring the connection dead (1006) during a turn that makes sequential CRM
tool calls and goes tens of seconds without any application traffic on the
socket.

The keepalive's task lifecycle (cancellation-safe drain/reap, idempotent
stop) predates this file's current form and is NOT under test here beyond
the direct `_run_turn_with_keepalive`/`_stop_keepalive` regression tests
below -- only the *content* of what's repeated changed: instead of a silent
`{"type":"typing"}` frame repeating every 8s, the platform now sends a real,
visible "still working on it" chat message (a `message` frame tagged
`"interim": true`) every `_INTERIM_INTERVAL_S` (15s in production, patched
down in these tests).

Covers:
- ``_interim_wait_keepalive`` / ``_run_turn_with_keepalive`` wired into the
  real WS message loop (periodic interim frames during a slow turn, none
  after the reply, none after an error, no change to the fast-turn frame
  sequence).
- Localization of the interim text, and its two-variant progression.
- Interim messages are never persisted to chat_messages / message_count.
- The ordering fix: the audio and image/video branches' branch-level
  keepalive must stop before persisting/replying, not after -- otherwise a
  slow persist/reply lets a (now visible) interim message land after the
  real answer, which looks broken.
- Direct unit tests of ``_run_turn_with_keepalive`` proving no leaked asyncio
  tasks and that a keepalive send failure never masks the turn's result.
- A guard test tying `_TURN_TIMEOUT_S` to the per-turn cumulative tool budget
  (`chatbot._TOOL_BUDGET_S`) rather than a fixed tool-call-count assumption.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents import chatbot
from src.api import chat as chat_api
from src.models.database import Base
from src.models.chat import ChatMessage, ChatSession


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


async def _add_session(sm, session_id: str, language: str) -> None:
    """Insert an extra ChatSession row (beyond the fixture's default "sess1")
    for tests that need a specific language code."""
    async with sm() as db:
        db.add(ChatSession(id=session_id, tenant_id="t1", language=language,
                            status="active", mode="ai", extra_data={}))
        await db.commit()


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
async def test_long_turn_sends_periodic_interim_messages(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    async def _slow_handle(text):
        await asyncio.sleep(0.4)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))

            # The one-time `typing` frame at t=0 is unchanged.
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"

            interim_count = 0
            frame = json.loads(ws.receive_text())
            while frame.get("interim"):
                assert frame["type"] == "message"
                interim_count += 1
                frame = json.loads(ws.receive_text())
            assert interim_count >= 3, f"expected >=3 interim frames, got {interim_count}"
            assert frame["type"] == "message"
            assert not frame.get("interim")
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_no_typing_frames_after_the_reply(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    async def _slow_handle(text):
        await asyncio.sleep(0.15)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            while frame["type"] == "typing" or frame.get("interim"):
                frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"
            assert not frame.get("interim")

            # Several interim intervals' worth of real time with the turn
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
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    async def _slow_fail(text):
        await asyncio.sleep(0.2)
        raise RuntimeError("boom")

    fake_agent.handle_message = _slow_fail

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            interim_count = 0
            frame = json.loads(ws.receive_text())
            while frame["type"] == "typing" or frame.get("interim"):
                if frame.get("interim"):
                    interim_count += 1
                frame = json.loads(ws.receive_text())
            assert interim_count >= 1
            assert frame["type"] == "error"

            # No stray interim frame should arrive after the error, even after
            # waiting through several interim intervals.
            await asyncio.sleep(0.3)
            ws.send_text(json.dumps({"type": "end"}))
            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended"
    finally:
        patcher.stop()


# --- Regression: fast turns unchanged (default 15.0s interval never fires) -


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
            assert not message.get("interim")

            ws.send_text(json.dumps({"type": "end"}))
            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended"
    finally:
        patcher.stop()


# --- Localization of the interim message ----------------------------------


@pytest.mark.asyncio
async def test_interim_message_localized_hindi(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx  # "sess1" is language="hi" per the fixture
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.03)

    async def _slow_handle(text):
        await asyncio.sleep(0.15)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            while not (frame["type"] == "message" and frame.get("interim")):
                frame = json.loads(ws.receive_text())
            assert frame["text"] == chat_api._INTERIM_WAIT_MESSAGES["hi"][0]
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_interim_message_localized_tamil(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    await _add_session(sm, "sess_ta", "ta")
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.03)

    async def _slow_handle(text):
        await asyncio.sleep(0.15)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess_ta") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            while not (frame["type"] == "message" and frame.get("interim")):
                frame = json.loads(ws.receive_text())
            assert frame["text"] == chat_api._INTERIM_WAIT_MESSAGES["ta"][0]
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_interim_message_unmapped_language_falls_back_to_english(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    await _add_session(sm, "sess_xx", "xx")
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.03)

    async def _slow_handle(text):
        await asyncio.sleep(0.15)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess_xx") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            while not (frame["type"] == "message" and frame.get("interim")):
                frame = json.loads(ws.receive_text())
            assert frame["text"] == chat_api._INTERIM_WAIT_MESSAGES["en"][0]
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_interim_message_variant_progression(ws_ctx, monkeypatch) -> None:
    """First interim message uses variant 0; every one after that uses
    variant 1 (repeats -- never indexes past the list)."""
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.03)

    async def _slow_handle(text):
        await asyncio.sleep(0.13)  # several interim intervals
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    interim_texts: list[str] = []
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            while frame["type"] == "typing" or frame.get("interim"):
                if frame.get("interim"):
                    interim_texts.append(frame["text"])
                frame = json.loads(ws.receive_text())
    finally:
        patcher.stop()

    assert len(interim_texts) >= 3, f"expected >=3 interim messages, got {len(interim_texts)}"
    variants = chat_api._INTERIM_WAIT_MESSAGES["hi"]
    assert interim_texts[0] == variants[0]
    assert interim_texts[1] == variants[1]
    assert interim_texts[2] == variants[1]


@pytest.mark.asyncio
async def test_interim_frame_shape(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.03)

    async def _slow_handle(text):
        await asyncio.sleep(0.1)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            while not (frame["type"] == "message" and frame.get("interim")):
                frame = json.loads(ws.receive_text())
    finally:
        patcher.stop()

    assert frame["type"] == "message"
    assert frame["session_id"] == "sess1"
    assert frame["sources"] == []
    assert frame["suggestions"] == []
    assert frame["action"] == "none"
    assert frame["interim"] is True


@pytest.mark.asyncio
async def test_interim_messages_not_persisted(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.02)

    async def _slow_handle(text):
        await asyncio.sleep(0.12)
        return _FakeTurnResult()

    fake_agent.handle_message = _slow_handle

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            interim_seen = 0
            frame = json.loads(ws.receive_text())
            while frame["type"] == "typing" or frame.get("interim"):
                if frame.get("interim"):
                    interim_seen += 1
                frame = json.loads(ws.receive_text())
            assert interim_seen >= 2, "test setup assumption broken: no interim frames fired"
            assert frame["type"] == "message"
    finally:
        patcher.stop()

    async with sm() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "sess1")
        )).scalars().all()
        session_row = await db.get(ChatSession, "sess1")

    assert len(rows) == 2, (
        f"expected exactly 2 persisted rows (customer + agent reply), "
        f"got {len(rows)}: {[r.role for r in rows]}"
    )
    assert session_row.message_count == 2
    assert session_row.cost == 0.0  # llm_provider=None on the fake result -> cost path never runs


# --- WS-level: keepalive covers the pre-turn media window (Bug 2) --------


@pytest.mark.asyncio
async def test_audio_keepalive_covers_pre_turn_media_window(ws_media_ctx, monkeypatch) -> None:
    """A slow S3 upload/transcription — BEFORE the LLM turn even starts — must
    still get keepalive frames. Regression for Bug 2: previously the audio
    branch only wrapped the turn call, leaving this window silent."""
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)
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
            assert typing["type"] == "typing"  # one-time frame, unchanged

            # The upload delay (0.3s) hasn't reached the LLM turn yet — an
            # interim frame arriving here proves coverage extends into the
            # pre-turn media window, not just the turn itself.
            second = json.loads(ws.receive_text())
            assert second["type"] == "message" and second.get("interim"), (
                "no interim frame during pre-turn media upload")

            frame = second
            interim_count = 1
            while frame.get("interim"):
                interim_count += 1
                frame = json.loads(ws.receive_text())
            assert interim_count >= 2
            assert frame["type"] == "audio_ack"
            reply = json.loads(ws.receive_text())
            assert reply["type"] == "message"
            assert not reply.get("interim")
    finally:
        patcher.stop()

    assert len(media_store.uploaded) == 1


@pytest.mark.asyncio
async def test_image_keepalive_covers_pre_turn_media_window(ws_media_ctx, monkeypatch) -> None:
    """A slow media_url fetch — BEFORE the LLM turn even starts — must still
    get keepalive frames. Regression for Bug 2: previously the image/video
    branch only wrapped the turn call, leaving this window silent."""
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

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
                assert second["type"] == "message" and second.get("interim"), (
                    "no interim frame during pre-turn media fetch")

                frame = second
                interim_count = 1
                while frame.get("interim"):
                    interim_count += 1
                    frame = json.loads(ws.receive_text())
                assert interim_count >= 2
                assert frame["type"] == "message"
                assert not frame.get("interim")
    finally:
        patcher.stop()

    fake_agent.handle_image.assert_awaited_once()


# --- Regression: keepalive must not survive into human handoff (Bug 1) ----
#
# `_run_human_mode` is an unbounded loop for the rest of a human agent's
# conversation with the customer. The audio and image/video branches used to
# stop their branch-level keepalive only in the `finally` wrapping the whole
# branch body — which doesn't run until `_run_human_mode` itself returns, so
# an interim frame kept firing every `_INTERIM_INTERVAL_S` for the entire
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
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    fake_result = _FakeTurnResult()
    fake_result.escalation = {"reason": "user needs help", "summary": "refund query"}
    fake_agent.handle_image = AsyncMock(return_value=fake_result)

    async def _fake_handle_escalation(websocket, session_id, tenant, row, result):
        await websocket.send_text(json.dumps({"type": "mode_change", "mode": "awaiting_human"}))
        return True

    async def _fake_human_mode(websocket, session_id, tenant):
        # Stand-in for "the rest of a human agent's conversation": several
        # interim intervals' worth of real time with the WS otherwise idle —
        # exactly the window that used to leak interim frames.
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
            # for 0.3s — several interim intervals. Drain a bit longer than
            # that to catch any leaked interim frame.
            frames += await _drain_for(ws, 0.5)
    finally:
        patcher.stop()

    after = frames[[f["type"] for f in frames].index("mode_change") + 1:]
    assert not any(f.get("interim") for f in after), (
        f"interim frame leaked during human handoff: {frames}"
    )


@pytest.mark.asyncio
async def test_no_typing_during_human_handoff_after_audio_escalation(ws_media_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

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

    after = frames[[f["type"] for f in frames].index("mode_change") + 1:]
    assert not any(f.get("interim") for f in after), (
        f"interim frame leaked during human handoff: {frames}"
    )


# --- Ordering regression: keepalive must stop BEFORE persist/reply, -------
# --- not after (the bug fixed alongside this rework) -----------------------
#
# With the OLD silent `typing` frame, stopping the branch-level keepalive
# late (right before the escalation check, i.e. AFTER `_persist_turn` /
# `audio_ack` / `_send_reply` had already run) was harmless — an extra
# `typing` frame is invisible and idempotent. With a REAL visible interim
# message this is a bug: a slow persist/reply gives the still-running
# keepalive room to fire again, and that frame could land right after the
# real answer — the customer would see the actual reply and THEN see
# "still working on it" appear after it, which looks broken.
#
# These tests use event-controlled synchronization (not wall-clock racing)
# so the assertion is deterministic: `_persist_turn` (or `_fetch_media_url`
# for the error-path variants) is patched to signal once it's entered and
# then block until released, letting the test observe directly whether the
# keepalive is still alive AT THE MOMENT persistence/fetch begins — which is
# exactly the moment the fix's ordering promise covers.
#
# NOTE: ``threading.Event`` (not ``asyncio.Event``) on purpose. Starlette's
# ``TestClient.websocket_connect`` runs the ASGI app -- and therefore
# ``_gated_persist``/``_gated_fetch`` below -- in a separate portal thread
# with its OWN event loop. An ``asyncio.Event`` is only safe to ``.set()``/
# ``.wait()`` from the single event loop that created it; sharing one across
# the portal's loop and this test's loop silently hangs (a waiter registered
# on one loop is never woken by a ``.set()`` call running on the other).
# ``threading.Event`` has no such restriction, and ``asyncio.to_thread`` lets
# each side await it without blocking its own loop.


@pytest.mark.asyncio
async def test_audio_interim_stops_before_persist_and_reply(ws_media_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.02)
    chat_api.set_media_store(_FakeMediaStore())

    persist_entered = threading.Event()
    release_persist = threading.Event()
    real_persist = chat_api._persist_turn

    async def _gated_persist(*a, **k):
        persist_entered.set()
        await asyncio.to_thread(release_persist.wait)
        return await real_persist(*a, **k)

    monkeypatch.setattr(chat_api, "_persist_turn", _gated_persist)

    patcher, client = _connect(_make_tenant())
    frames: list[dict] = []
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio", "data": base64.b64encode(b"fake audio bytes").decode(),
                "mime": "audio/webm",
            }))

            async def _reader():
                while True:
                    raw = await asyncio.to_thread(ws.receive_text)
                    frames.append(json.loads(raw))

            reader_task = asyncio.ensure_future(_reader())
            try:
                entered = await asyncio.to_thread(persist_entered.wait, 2.0)
                assert entered, "test setup assumption broken: _persist_turn never entered"
                # Several interim intervals' worth of real time while
                # persistence is deliberately held open. If the keepalive
                # were still alive at this point (the bug), an interim frame
                # would arrive here.
                await asyncio.sleep(0.1)
                assert not any(f.get("interim") for f in frames), (
                    f"interim frame arrived while _persist_turn was in "
                    f"flight (keepalive not stopped before persist/reply): {frames}"
                )
                release_persist.set()
                # Let the reply go out, then drain a bit more to catch any
                # trailing interim frame.
                await asyncio.sleep(0.1)
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
    finally:
        patcher.stop()

    reply_positions = [i for i, f in enumerate(frames)
                        if f["type"] == "message" and not f.get("interim")]
    assert reply_positions, f"real reply never arrived: {frames}"
    reply_idx = reply_positions[0]
    assert not any(f.get("interim") for f in frames[reply_idx + 1:]), (
        f"interim frame arrived after the real reply: {frames}"
    )


@pytest.mark.asyncio
async def test_image_interim_stops_before_persist_and_reply(ws_media_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.02)

    persist_entered = threading.Event()
    release_persist = threading.Event()
    real_persist = chat_api._persist_turn

    async def _gated_persist(*a, **k):
        persist_entered.set()
        await asyncio.to_thread(release_persist.wait)
        return await real_persist(*a, **k)

    monkeypatch.setattr(chat_api, "_persist_turn", _gated_persist)

    patcher, client = _connect(_make_tenant())
    frames: list[dict] = []
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "image", "data": base64.b64encode(b"fake image bytes").decode(),
                "mime": "image/jpeg",
            }))

            async def _reader():
                while True:
                    raw = await asyncio.to_thread(ws.receive_text)
                    frames.append(json.loads(raw))

            reader_task = asyncio.ensure_future(_reader())
            try:
                entered = await asyncio.to_thread(persist_entered.wait, 2.0)
                assert entered, "test setup assumption broken: _persist_turn never entered"
                await asyncio.sleep(0.1)
                assert not any(f.get("interim") for f in frames), (
                    f"interim frame arrived while _persist_turn was in "
                    f"flight (keepalive not stopped before persist/reply): {frames}"
                )
                release_persist.set()
                await asyncio.sleep(0.1)
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
    finally:
        patcher.stop()

    reply_positions = [i for i, f in enumerate(frames)
                        if f["type"] == "message" and not f.get("interim")]
    assert reply_positions, f"real reply never arrived: {frames}"
    reply_idx = reply_positions[0]
    assert not any(f.get("interim") for f in frames[reply_idx + 1:]), (
        f"interim frame arrived after the real reply: {frames}"
    )


@pytest.mark.asyncio
async def test_audio_no_interim_after_media_fetch_error(ws_media_ctx, monkeypatch) -> None:
    """Error-path variant: a media_url fetch failure must stop the keepalive
    before the error frame is sent, so no interim frame can follow it.

    Unlike the persist/reply ordering tests above, ``_stop_keepalive`` for
    this path only runs *inside* the ``except`` handler once
    ``_fetch_media_url`` actually raises (src/api/chat.py, the
    ``if audio_bytes is None:`` block) -- so while the fetch is still in
    flight the keepalive is legitimately still running and MAY send interim
    frames (that's the pre-turn-media-window coverage proven by
    ``test_audio_keepalive_covers_pre_turn_media_window`` above). This test's
    own job is narrower: prove no interim frame lands AFTER the error frame,
    not that the keepalive is silent during the fetch itself.

    A plain ``asyncio.sleep``-then-raise fake stands in for the slow/failing
    fetch -- no cross-loop ``threading.Event`` gate needed here, since
    there's nothing to synchronize on except elapsed time relative to the
    (patched tiny) interim interval.
    """
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.02)
    chat_api.set_media_store(_FakeMediaStore())

    async def _slow_failing_fetch(url):
        await asyncio.sleep(0.1)
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(chat_api, "_fetch_media_url", _slow_failing_fetch)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio", "media_url": "https://example.com/a.webm",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"

            frame = json.loads(ws.receive_text())
            while frame.get("interim"):
                frame = json.loads(ws.receive_text())
            assert frame["type"] == "error", f"expected error frame, got: {frame}"

            trailing = await _drain_for(ws, 0.1)
            assert not any(f.get("interim") for f in trailing), (
                f"interim frame arrived after the error frame: {trailing}"
            )
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_image_no_interim_after_media_fetch_error(ws_media_ctx, monkeypatch) -> None:
    """Same as ``test_audio_no_interim_after_media_fetch_error`` above, for
    the image/video branch's ``_fetch_media_url`` failure path."""
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.02)

    async def _slow_failing_fetch(url):
        await asyncio.sleep(0.1)
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(chat_api, "_fetch_media_url", _slow_failing_fetch)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "image", "media_url": "https://example.com/pic.jpg",
            }))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"

            frame = json.loads(ws.receive_text())
            while frame.get("interim"):
                frame = json.loads(ws.receive_text())
            assert frame["type"] == "error", f"expected error frame, got: {frame}"

            trailing = await _drain_for(ws, 0.1)
            assert not any(f.get("interim") for f in trailing), (
                f"interim frame arrived after the error frame: {trailing}"
            )
    finally:
        patcher.stop()


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
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.02)
    fake_ws = _FakeWS()

    async def _slow_ok():
        await asyncio.sleep(0.08)
        return "turn-result"

    current = asyncio.current_task()
    result = await chat_api._run_turn_with_keepalive(
        fake_ws, _slow_ok(), session_id="sess1", language="en")
    assert result == "turn-result"

    sent_after_return = len(fake_ws.sent)
    await asyncio.sleep(0.1)  # several interim intervals after the turn ended
    assert len(fake_ws.sent) == sent_after_return, "keepalive kept sending after the turn finished"

    remaining = asyncio.all_tasks() - {current}
    assert not remaining, f"leaked asyncio tasks: {remaining}"


@pytest.mark.asyncio
async def test_keepalive_send_failure_does_not_mask_result(monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.02)
    fake_ws = _FakeWS(fail=True)

    async def _slow_ok():
        await asyncio.sleep(0.08)
        return "turn-result"

    result = await chat_api._run_turn_with_keepalive(
        fake_ws, _slow_ok(), session_id="sess1", language="en")
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
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.01)
    monkeypatch.setattr(chat_api, "_KEEPALIVE_DRAIN_S", 2.0)

    fake_ws = _WedgedWS()

    async def _turn():
        # Give the keepalive time to fire once and get stuck inside
        # send_text before the turn itself finishes, so `ka` is guaranteed
        # mid-send when `finally` starts its drain wait.
        await asyncio.sleep(0.05)
        return "turn-result"

    task = asyncio.ensure_future(chat_api._run_turn_with_keepalive(
        fake_ws, _turn(), session_id="sess1", language="en"))
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


# --- Pure-function tests ----------------------------------------------------


def test_interim_wait_text_index_clamping_and_language_fallback() -> None:
    # Index clamping: past the last variant, the final variant repeats --
    # never an IndexError.
    variants_hi = chat_api._INTERIM_WAIT_MESSAGES["hi"]
    assert chat_api._interim_wait_text("hi", 0) == variants_hi[0]
    assert chat_api._interim_wait_text("hi", 1) == variants_hi[1]
    assert chat_api._interim_wait_text("hi", 2) == variants_hi[-1]
    assert chat_api._interim_wait_text("hi", 99) == variants_hi[-1]

    # Language fallback: unmapped code -> English.
    assert chat_api._interim_wait_text("xx", 0) == chat_api._INTERIM_WAIT_MESSAGES["en"][0]


def test_interim_messages_carry_no_masculine_self_reference() -> None:
    # Hindi/Marathi/Gujarati/Punjabi grammatically mark the speaker's gender
    # on first-person verbs -- a blunt substring check for known masculine
    # markers across every variant guards against regressing to gendered
    # self-reference (e.g. Hindi "काम कर रहा हूँ", Marathi "कळवतो").
    masculine_markers = (
        "रहा हूँ", "ता हूँ", "कळवतो", "करतो", "રહ્યો છું", "ਰਿਹਾ ਹਾਂ", "ਦੱਸਾਂਗਾ",
    )
    for lang, variants in chat_api._INTERIM_WAIT_MESSAGES.items():
        for text in variants:
            for marker in masculine_markers:
                assert marker not in text, f"{lang!r} variant has masculine marker {marker!r}: {text!r}"
    # _GREETINGS already established the neutral convention these messages
    # mirror -- lock in that it stays that way too.
    for lang, text in chat_api._GREETINGS.items():
        for marker in masculine_markers:
            assert marker not in text, f"{lang!r} greeting has masculine marker {marker!r}: {text!r}"


def test_turn_timeout_leaves_room_for_llm_overhead_beyond_the_tool_budget() -> None:
    # Tool time within a turn is no longer bounded by a fixed "2 calls total"
    # assumption (a turn can carry more than 2 tool calls -- up to
    # _max_tool_rounds rounds, each of which can itself carry several
    # function_call parts in one LLM response). Instead it's bounded by a
    # per-turn CUMULATIVE budget (chatbot._TOOL_BUDGET_S) regardless of how
    # many calls make it up, with a per-call ceiling below that budget so one
    # call can never consume the whole thing.
    assert chatbot._TOOL_CALL_CEILING_S < chatbot._TOOL_BUDGET_S
    # 40.0s is a documented floor for the realistic LLM-side worst case within
    # a turn: up to 3 generate() calls (2 tool rounds + 1 final synthesis, or
    # the forced-final path) plus one _CHAT_RETRY_TIMEOUT_S-bounded (12s)
    # retry -- ~5-8s per call x 3 (~15-24s) + 12s retry (~27-36s realistic),
    # comfortably under 40s of margin.
    assert chat_api._TURN_TIMEOUT_S - chatbot._TOOL_BUDGET_S >= 40.0

    # KB search (chatbot._KB_SEARCH_TIMEOUT_S) deliberately sits OUTSIDE
    # _TOOL_BUDGET_S (see _exec_kb_tool) so a slow/exhausted CRM budget can
    # never starve it. That means the two assertions above don't cover the
    # full worst case: a turn carrying BOTH a fully-saturated CRM budget AND
    # a slow KB search in every round (up to _max_tool_rounds=2) can push
    # total tool time higher than _TOOL_BUDGET_S alone. This is a known,
    # accepted edge case (needs both conditions at once, and even then
    # degrades to the existing turn-timeout error path, not a silent
    # failure) -- documented here so the number is visible, not re-derived.
    worst_case_tool_time_s = chatbot._TOOL_BUDGET_S + 2 * chatbot._KB_SEARCH_TIMEOUT_S
    assert worst_case_tool_time_s == 75.0
    assert chat_api._TURN_TIMEOUT_S - worst_case_tool_time_s == 15.0
