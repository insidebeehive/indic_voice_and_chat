"""Duplicate-turn guard: dedupes inbound WS chat frames so a single customer
message can never produce two independent bot turns.

Root cause of the reported bug: the WS message loop had zero deduplication
of inbound frames, and the CRM's downstream relay can replay a still-pending
customer message after reconnecting mid-turn (either on the same connection,
if the relay resends before seeing an ack, or on a brand-new connection for
the same session_id). The WS loop itself is strictly sequential (the text-turn
call site does an inline `await _run_turn_with_keepalive(...)` — see
chat.py:1753-ish), so a same-connection duplicate is never even read until the
first turn has already fully finished; that's why the guard needs both an
in-flight check (Part A) AND a post-completion echo window (Part B), and why
it's keyed by session_id rather than by connection (see the module comment
above `_TurnGuard` in src/api/chat.py for the full rationale).

Fixture scaffolding (``_FakeTurnResult``, ``ws_ctx``/``ws_media_ctx``,
``_make_tenant``, ``_connect``, ``_drain_for``) is copied from
tests/unit/test_chat_keepalive.py -- see that file for the reasoning behind
each piece (in particular: why ``threading.Event``, not ``asyncio.Event``, is
used for cross-thread gating with the TestClient's portal thread).
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

from src.api import chat as chat_api
from src.models.database import Base
from src.models.chat import ChatMessage, ChatSession

# Note: no local `_clear_turn_guards` autouse fixture here -- the shared
# `_reset_chat_turn_guards` autouse fixture in tests/conftest.py already
# clears `chat_api._turn_guards` before and after every test in the suite
# (same logic, same scope), so a per-file duplicate would be redundant.


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
    await engine.dispose()


class _FakeMediaStore:
    def __init__(self, delay: float = 0.0) -> None:
        self.uploaded: list[tuple] = []
        self._delay = delay

    async def upload(self, data, key, content_type) -> None:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.uploaded.append((key, content_type, data))


@pytest_asyncio.fixture
async def ws_media_ctx():
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

    async def fake_factory(tenant, scoped_id, *, customer_id=None, ticket_id=None):
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
        except asyncio.TimeoutError:
            # Expected, normal exit: nothing more arrived before the deadline.
            # Every call site here drains a `ws` that is still open (none call
            # this after closing the connection), so a real disconnect is not
            # an expected outcome of this loop -- if `ws.receive_text()` ever
            # raises anything else, let it propagate instead of silently
            # treating it as "the drain just ended", which would mask a real
            # error as normal completion.
            break
        frames.append(json.loads(raw))
    return frames


# --- 1. Headline regression test ------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_frames_produce_one_turn(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    handle_calls = []

    async def _fast_handle(text):
        handle_calls.append(text)
        return _FakeTurnResult()

    fake_agent.handle_message = AsyncMock(side_effect=_fast_handle)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            # Two identical frames back-to-back, before draining either.
            ws.send_text(json.dumps({"type": "message", "text": "How to deposit"}))
            ws.send_text(json.dumps({"type": "message", "text": "How to deposit"}))

            frames = []
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            assert frame["type"] == "message" and not frame.get("interim")

            # Bounded drain to prove nothing else shows up (no second typing,
            # no second reply, no error frame for the dropped duplicate).
            frames += await _drain_for(ws, 0.3)
    finally:
        patcher.stop()

    fake_agent.handle_message.assert_awaited_once()
    assert len(handle_calls) == 1
    message_frames = [f for f in frames if f["type"] == "message" and not f.get("interim")]
    assert len(message_frames) == 1
    assert not any(f["type"] == "error" for f in frames)
    assert not any(f["type"] == "typing" for f in frames[1:])

    async with sm() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "sess1")
        )).scalars().all()
        session_row = await db.get(ChatSession, "sess1")

    assert len(rows) == 2, (
        f"expected exactly 2 persisted rows (1 customer + 1 bot reply), got "
        f"{len(rows)}: {[r.role for r in rows]}"
    )
    assert session_row.message_count == 2


# --- 2. Guard releases after echo window elapses --------------------------


@pytest.mark.asyncio
async def test_guard_releases_after_turn_completes(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    fake_agent.handle_message = AsyncMock(return_value=_FakeTurnResult())

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "How to deposit"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")

            # Shrink the echo window and wait past it (no real multi-second sleep).
            monkeypatch.setattr(chat_api, "_DUPLICATE_ECHO_WINDOW_S", 0.05)
            import time as _time
            _time.sleep(0.1)

            # Same text again -> must run as a real second turn now.
            ws.send_text(json.dumps({"type": "message", "text": "How to deposit"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")

            # A third, different message also runs.
            ws.send_text(json.dumps({"type": "message", "text": "something else"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")
    finally:
        patcher.stop()

    assert fake_agent.handle_message.await_count == 3


# --- 3. Part B is fingerprint-scoped, not a blanket cooldown --------------


@pytest.mark.asyncio
async def test_different_message_after_reply_not_blocked(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)
    fake_agent.handle_message = AsyncMock(return_value=_FakeTurnResult())

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "msg one"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"

            # Immediately (no window wait) send a DIFFERENT message.
            ws.send_text(json.dumps({"type": "message", "text": "msg two"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")
    finally:
        patcher.stop()

    assert fake_agent.handle_message.await_count == 2


# --- 4. finally-release fires even on turn exception -----------------------


@pytest.mark.asyncio
async def test_guard_released_after_turn_exception(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    calls = {"n": 0}

    async def _flaky(text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return _FakeTurnResult()

    fake_agent.handle_message = AsyncMock(side_effect=_flaky)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "msg one"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"

            # A DIFFERENT second message must be processed normally afterward.
            ws.send_text(json.dumps({"type": "message", "text": "msg two"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")
    finally:
        patcher.stop()

    assert calls["n"] == 2


# --- 5. finally-release fires even on turn timeout -------------------------


@pytest.mark.asyncio
async def test_guard_released_after_turn_timeout(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)
    monkeypatch.setattr(chat_api, "_TURN_TIMEOUT_S", 0.05)

    calls = {"n": 0}

    async def _slow_then_fast(text):
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(0.3)
        return _FakeTurnResult()

    fake_agent.handle_message = AsyncMock(side_effect=_slow_then_fast)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "msg one"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            while frame.get("interim"):
                frame = json.loads(ws.receive_text())
            assert frame["type"] == "error"
            assert frame.get("reason") == "timeout"

            ws.send_text(json.dumps({"type": "message", "text": "different message"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")
    finally:
        patcher.stop()

    assert calls["n"] == 2


# --- 6. Keyed by session_id, catches a SECOND concurrent connection -------


@pytest.mark.asyncio
async def test_second_connection_dropped_while_first_mid_turn(ws_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)

    turn_entered = threading.Event()
    release_turn = threading.Event()
    calls = {"n": 0}

    async def _gated_handle(text):
        calls["n"] += 1
        turn_entered.set()
        await asyncio.to_thread(release_turn.wait)
        return _FakeTurnResult()

    fake_agent.handle_message = AsyncMock(side_effect=_gated_handle)

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws_a, \
                client.websocket_connect("/api/v1/chat/ws/sess1") as ws_b:
            ws_a.send_text(json.dumps({"type": "message", "text": "How to deposit"}))

            entered = await asyncio.to_thread(turn_entered.wait, 2.0)
            assert entered, "test setup assumption broken: turn never entered"

            # Connection A is provably still mid-turn. Send the identical
            # message on connection B — must be silently dropped, no
            # typing/message frame.
            ws_b.send_text(json.dumps({"type": "message", "text": "How to deposit"}))
            b_frames = await _drain_for(ws_b, 0.3)
            assert not any(f["type"] in ("typing", "message") for f in b_frames), (
                f"connection B was not dropped while A was mid-turn: {b_frames}"
            )

            release_turn.set()

            # Connection A's real turn completes normally (possibly preceded
            # by one or more interim "still working on it" keepalive frames,
            # since the gating above can take longer than the patched
            # _INTERIM_INTERVAL_S).
            frame = json.loads(ws_a.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws_a.receive_text())
            while frame.get("interim"):
                frame = json.loads(ws_a.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")
    finally:
        patcher.stop()

    assert calls["n"] == 1


# --- 7. Media turns are deduplicated too (guard sits above mtype dispatch) -


@pytest.mark.asyncio
async def test_media_turn_deduplicated(ws_media_ctx, monkeypatch) -> None:
    sm, fake_agent = ws_media_ctx
    monkeypatch.setattr(chat_api, "_INTERIM_INTERVAL_S", 0.05)
    chat_api.set_media_store(_FakeMediaStore())

    patcher, client = _connect(_make_tenant())
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            audio_bytes = b"fake audio bytes"
            encoded = base64.b64encode(audio_bytes).decode()
            frame_dict = {"type": "audio", "data": encoded, "mime": "audio/webm"}
            ws.send_text(json.dumps(frame_dict))
            ws.send_text(json.dumps(frame_dict))

            frames = []
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            assert frame["type"] == "audio_ack"
            frame = json.loads(ws.receive_text())
            frames.append(frame)
            assert frame["type"] == "message" and not frame.get("interim")

            frames += await _drain_for(ws, 0.3)
    finally:
        patcher.stop()

    fake_agent.handle_message.assert_awaited_once()
    assert not any(f["type"] == "error" for f in frames)
    assert not any(f["type"] == "typing" for f in frames[1:])
    assert not any(f["type"] == "audio_ack" for f in frames[2:])


# --- 8. Pure-function tests -------------------------------------------------


def test_turn_fingerprint_stable_and_distinguishes_content() -> None:
    fp1 = chat_api._turn_fingerprint({"type": "message", "text": "How to deposit"})
    fp2 = chat_api._turn_fingerprint({"type": "message", "text": "How to deposit"})
    assert fp1 == fp2

    fp3 = chat_api._turn_fingerprint({"type": "message", "text": "different text"})
    assert fp3 != fp1

    fp_audio1 = chat_api._turn_fingerprint({
        "type": "audio", "mime": "audio/webm", "data": "AAAA",
    })
    fp_audio2 = chat_api._turn_fingerprint({
        "type": "audio", "mime": "audio/webm", "data": "AAAA",
    })
    assert fp_audio1 == fp_audio2
    assert fp_audio1 != fp1  # different mtype/content entirely

    fp_audio3 = chat_api._turn_fingerprint({
        "type": "audio", "mime": "audio/webm", "data": "BBBB",
    })
    assert fp_audio3 != fp_audio1

    fp_image_caption1 = chat_api._turn_fingerprint({
        "type": "image", "mime": "image/jpeg", "media_url": "https://x/1.jpg", "text": "look",
    })
    fp_image_caption2 = chat_api._turn_fingerprint({
        "type": "image", "mime": "image/jpeg", "media_url": "https://x/1.jpg", "text": "different caption",
    })
    assert fp_image_caption1 != fp_image_caption2


def test_try_begin_turn_blocks_while_in_flight_and_releases() -> None:
    chat_api._turn_guards.clear()
    token = chat_api._try_begin_turn("s1", "fp1")
    assert token

    # In flight -> a second attempt (even with a different fingerprint) is blocked.
    assert chat_api._try_begin_turn("s1", "fp1") is None
    assert chat_api._try_begin_turn("s1", "fp2") is None

    chat_api._end_turn("s1", token)

    # Released -> a fresh attempt succeeds and returns a new token.
    token2 = chat_api._try_begin_turn("s1", "fp2")
    assert token2
    assert token2 is not token


def test_try_begin_turn_takes_over_stale_in_flight_entry(monkeypatch) -> None:
    chat_api._turn_guards.clear()
    token = chat_api._try_begin_turn("s1", "fp1")
    assert token

    # Age the entry past the max-hold ceiling without a real sleep.
    entry = chat_api._turn_guards["s1"]
    entry.started_at -= (chat_api._TURN_GUARD_MAX_HOLD_S + 1.0)

    # A fresh call now succeeds instead of being blocked (stale takeover).
    token2 = chat_api._try_begin_turn("s1", "fp2")
    assert token2
    assert token2 is not token


def test_end_turn_with_stale_token_does_not_clobber_newer_entry() -> None:
    chat_api._turn_guards.clear()
    token1 = chat_api._try_begin_turn("s1", "fp1")
    assert token1

    # Simulate a stale takeover superseding token1 with a new in-flight entry.
    entry = chat_api._turn_guards["s1"]
    entry.started_at -= (chat_api._TURN_GUARD_MAX_HOLD_S + 1.0)
    token2 = chat_api._try_begin_turn("s1", "fp2")
    assert token2 and token2 is not token1

    # The old (superseded) token must not be able to release the new entry.
    chat_api._end_turn("s1", token1)
    entry_after = chat_api._turn_guards["s1"]
    assert entry_after.in_flight is True
    assert entry_after.token is token2

    # The real (current) token still releases it correctly.
    chat_api._end_turn("s1", token2)
    assert chat_api._turn_guards["s1"].in_flight is False


def test_dict_bounding_sweep_removes_stale_completed_entries(monkeypatch) -> None:
    chat_api._turn_guards.clear()
    monkeypatch.setattr(chat_api, "_TURN_GUARD_MAX_ENTRIES", 3)
    monkeypatch.setattr(chat_api, "_DUPLICATE_ECHO_WINDOW_S", 5.0)

    # Fill past the cap with completed (non-in-flight), old entries.
    for i in range(4):
        sid = f"stale-{i}"
        tok = chat_api._try_begin_turn(sid, "fp")
        chat_api._end_turn(sid, tok)
        chat_api._turn_guards[sid].finished_at -= 100.0  # well past the echo window

    assert len(chat_api._turn_guards) > 3

    # The next acquire call should trigger the bounding sweep and remove the
    # stale entries.
    tok = chat_api._try_begin_turn("new-session", "fp")
    assert tok
    assert all(sid.startswith("stale-") is False or sid not in chat_api._turn_guards
               for sid in [f"stale-{i}" for i in range(4)])
    for i in range(4):
        assert f"stale-{i}" not in chat_api._turn_guards
