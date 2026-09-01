"""Deposit dispute screenshot verification timeout coverage.

Covers ``_check_and_timeout_verification``, ``schedule_verification_timeout``,
and the reconnect-time timeout sweep inside ``chat_websocket`` — all in
``src/api/chat.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat as chat_api
from src.api.call_store import set_tenant_event_notifier
from src.auth.middleware import register_tenant_for_test, set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base
from src.models.deposit_verification import DepositVerificationRequest


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _tenant_settings(tenant_id: str = "t1") -> TenantSettings:
    return TenantSettings(id=tenant_id, slug=f"tenant-{tenant_id}", name="Demo Tenant")


@pytest_asyncio.fixture
async def timeout_ctx(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(ChatSession(
            id="cs_1", tenant_id="t1", customer_name="Raju", customer_id="cust_7",
            language="hi", status="active", mode="ai", extra_data={},
        ))
        await db.commit()

    chat_api.set_chat_sessionmaker(sm)
    tenant_ctx = register_tenant_for_test(_tenant_settings("t1"))

    webhook = AsyncMock(return_value=True)
    monkeypatch.setattr("src.api.chat_webhooks.send_bo_webhook", webhook)

    yield sm, tenant_ctx, webhook

    set_tenant_resolver(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    # Narrow to the one session id this file's tests ever register (checked
    # via grep) rather than wiping the whole module-level dict, which would
    # be broader than this fixture owns.
    chat_api._bo_queues.pop("cs_1", None)
    chat_api._customer_queues.pop("cs_1", None)
    chat_api._async_push_queues.pop("cs_1", None)
    await engine.dispose()


async def _seed_request(
    sm, request_id: str = "dvr_1", *, status: str = "pending", session_id: str = "cs_1",
    tenant_id: str = "t1", order_id: str = "ORD-9", overdue: bool = True,
) -> None:
    timeout_at = _now() + (timedelta(minutes=-1) if overdue else timedelta(minutes=10))
    async with sm() as db:
        db.add(DepositVerificationRequest(
            id=request_id, tenant_id=tenant_id, session_id=session_id,
            order_id=order_id, status=status, timeout_at=timeout_at,
        ))
        await db.commit()


async def _get_request(sm, request_id: str) -> DepositVerificationRequest:
    async with sm() as db:
        return await db.get(DepositVerificationRequest, request_id)


async def _get_session(sm, session_id: str = "cs_1") -> ChatSession:
    async with sm() as db:
        return await db.get(ChatSession, session_id)


async def _get_messages(sm, session_id: str = "cs_1") -> list[ChatMessage]:
    async with sm() as db:
        return (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id)
        )).scalars().all()


# --- _check_and_timeout_verification: core coverage -----------------------


@pytest.mark.asyncio
async def test_timed_out_pending_request_marks_row_escalates_and_pushes(timeout_ctx) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm)

    events: list[dict] = []

    async def _notifier(env: dict) -> None:
        events.append(env)

    set_tenant_event_notifier(_notifier)
    try:
        await chat_api._check_and_timeout_verification("dvr_1")
    finally:
        set_tenant_event_notifier(None)

    row = await _get_request(sm, "dvr_1")
    session = await _get_session(sm)
    messages = await _get_messages(sm)

    assert row.status == "timed_out"
    assert session.mode == "awaiting_human"

    webhook.assert_awaited_once()
    tenant_arg, event_type, payload = webhook.await_args.args
    assert tenant_arg is tenant_ctx
    assert event_type == "escalation_requested"
    assert payload["session_id"] == "cs_1"
    assert payload["reason"] == "deposit verification timed out"
    assert "ORD-9" in payload["summary"]
    assert payload["customer"] == {"name": "Raju", "id": "cust_7"}
    assert payload["bo_available"] is True

    assert "cs_1" in chat_api._bo_queues
    assert "cs_1" in chat_api._customer_queues

    assert any(
        m.role == "system" and "connecting" in m.content.lower()
        for m in messages
    )

    assert len(events) == 1
    env = events[0]
    assert env["event_type"] == "chat.escalated"
    assert env["call_id"] == "cs_1"
    assert env["tenant_id"] == "t1"
    assert env["data"]["reason"] == "deposit verification timed out"


@pytest.mark.asyncio
async def test_timeout_with_failed_bo_webhook_reverts_to_bot_and_pushes_softer_message(
    timeout_ctx,
) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    webhook.return_value = False
    await _seed_request(sm)

    events: list[dict] = []

    async def _notifier(env: dict) -> None:
        events.append(env)

    set_tenant_event_notifier(_notifier)
    try:
        await chat_api._check_and_timeout_verification("dvr_1")
    finally:
        set_tenant_event_notifier(None)

    row = await _get_request(sm, "dvr_1")
    session = await _get_session(sm)
    messages = await _get_messages(sm)

    assert row.status == "timed_out"
    assert session.mode == "bot"
    assert events == []
    assert any(
        m.role == "system" and "taking longer than expected" in m.content
        for m in messages
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_status", ["verified", "rejected", "timed_out", "error"])
async def test_already_resolved_request_is_a_no_op(timeout_ctx, prior_status) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm, status=prior_status, overdue=True)

    await chat_api._check_and_timeout_verification("dvr_1")

    row = await _get_request(sm, "dvr_1")
    session = await _get_session(sm)
    messages = await _get_messages(sm)

    assert row.status == prior_status
    assert session.mode == "ai"
    webhook.assert_not_awaited()
    assert messages == []


@pytest.mark.asyncio
async def test_not_yet_due_request_is_a_no_op(timeout_ctx) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm, overdue=False)

    await chat_api._check_and_timeout_verification("dvr_1")

    row = await _get_request(sm, "dvr_1")
    session = await _get_session(sm)

    assert row.status == "pending"
    assert session.mode == "ai"
    webhook.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_request_id_is_a_no_op(timeout_ctx) -> None:
    sm, tenant_ctx, webhook = timeout_ctx

    await chat_api._check_and_timeout_verification("does_not_exist")

    webhook.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_tenant_marks_timed_out_but_does_not_escalate(timeout_ctx, caplog) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm, tenant_id="unregistered_tenant")

    with caplog.at_level(logging.WARNING, logger="src.api.chat"):
        await chat_api._check_and_timeout_verification("dvr_1")

    row = await _get_request(sm, "dvr_1")
    session = await _get_session(sm)
    messages = await _get_messages(sm)

    # Fixed: tenant/chat-session resolution now happens BEFORE the row is
    # flipped to "timed_out" and committed, so an unresolvable tenant leaves
    # the row "pending" for a later reconnect-time sweep to retry, instead of
    # permanently stuck.
    assert row.status == "pending"
    assert session.mode == "ai"
    webhook.assert_not_awaited()
    assert messages == []
    assert any("tenant unavailable" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_missing_chat_session_marks_timed_out_but_does_not_escalate(timeout_ctx, caplog) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm, session_id="cs_does_not_exist")

    with caplog.at_level(logging.WARNING, logger="src.api.chat"):
        await chat_api._check_and_timeout_verification("dvr_1")

    row = await _get_request(sm, "dvr_1")

    # Fixed: tenant/chat-session resolution now happens BEFORE the row is
    # flipped to "timed_out" and committed, so an unresolvable chat session
    # leaves the row "pending" for a later reconnect-time sweep to retry,
    # instead of permanently stuck.
    assert row.status == "pending"
    webhook.assert_not_awaited()
    assert any("chat session missing" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_check_never_raises_on_db_failure(timeout_ctx, monkeypatch, caplog) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm)

    def _boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(chat_api, "_sm", _boom)

    with caplog.at_level(logging.ERROR, logger="src.api.chat"):
        await chat_api._check_and_timeout_verification("dvr_1")

    # Match the specific message _check_and_timeout_verification's own
    # except-block logs (chat.py's "deposit verification timeout handling
    # failed" log.exception call), not just any exception log — consistent
    # with this file's other tests, which all assert on specific message text.
    assert any(
        r.exc_info and r.getMessage() == "deposit verification timeout handling failed"
        for r in caplog.records
    )


# --- schedule_verification_timeout wiring ----------------------------------


@pytest.mark.asyncio
async def test_schedule_with_zero_minutes_fires_the_check_immediately(monkeypatch) -> None:
    evt = asyncio.Event()
    calls: list[str] = []

    async def _fake_check(request_id: str) -> None:
        calls.append(request_id)
        evt.set()

    monkeypatch.setattr(chat_api, "_check_and_timeout_verification", _fake_check)

    chat_api.schedule_verification_timeout("dvr_1", "cs_1", 0)
    await asyncio.wait_for(evt.wait(), 1.0)

    assert calls == ["dvr_1"]


@pytest.mark.asyncio
async def test_schedule_sleeps_timeout_minutes_times_sixty(monkeypatch) -> None:
    evt = asyncio.Event()
    calls: list[str] = []
    sleep_calls: list[float] = []
    # ``chat_api.asyncio is asyncio`` (same module object), so
    # `monkeypatch.setattr(chat_api.asyncio, "sleep", ...)` unavoidably
    # replaces the real `asyncio.sleep` process-wide for this test's
    # lifetime -- there's no separate seam in `schedule_verification_timeout`
    # to patch instead. Narrow the blast radius: only intercept (and record)
    # the one delay this test actually cares about (300.0s, from
    # timeout_minutes=5); delegate every other duration to a saved reference
    # to the real `asyncio.sleep`, so any other concurrent/background task
    # needing a real sleep during this window still gets one.
    _real_sleep = asyncio.sleep

    async def _fake_check(request_id: str) -> None:
        calls.append(request_id)
        evt.set()

    async def _fake_sleep(delay, *args, **kwargs) -> None:
        if delay == 300.0:
            sleep_calls.append(delay)
            return
        await _real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(chat_api, "_check_and_timeout_verification", _fake_check)
    monkeypatch.setattr(chat_api.asyncio, "sleep", _fake_sleep)

    chat_api.schedule_verification_timeout("dvr_1", "cs_1", 5)
    await asyncio.wait_for(evt.wait(), 1.0)

    assert sleep_calls == [300.0]
    assert calls == ["dvr_1"]


@pytest.mark.asyncio
async def test_schedule_registers_and_releases_the_background_task(monkeypatch) -> None:
    evt = asyncio.Event()

    async def _fake_check(request_id: str) -> None:
        evt.set()

    monkeypatch.setattr(chat_api, "_check_and_timeout_verification", _fake_check)

    before = len(chat_api._webhook_tasks)
    chat_api.schedule_verification_timeout("dvr_1", "cs_1", 0)
    assert len(chat_api._webhook_tasks) == before + 1

    await asyncio.wait_for(evt.wait(), 1.0)
    await asyncio.sleep(0)  # let the done-callback discard the finished task

    assert len(chat_api._webhook_tasks) == before


# --- Reconnect sweep (WS-level) --------------------------------------------


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


def _fake_agent() -> MagicMock:
    agent = MagicMock()
    agent.session = MagicMock()
    agent.handle_message = AsyncMock(return_value=_FakeTurnResult())
    agent.summarize_session = AsyncMock(return_value="summary")
    return agent


async def _fake_factory(tenant, scoped_id, *, customer_id=None, ticket_id=None):
    return _fake_agent()


def _connect_ws() -> TestClient:
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    return TestClient(app)


@pytest.mark.asyncio
async def test_reconnect_sweep_times_out_an_overdue_pending_request(timeout_ctx) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm)
    chat_api.set_chatbot_factory(_fake_factory)

    client = _connect_ws()
    # The sweep runs on connect, before any message exchange, and its
    # escalation flips the session straight into human-handoff mode — the
    # client here sees no bot frames at all, just the connect/close. Give the
    # server-side coroutine a moment to reach its steady "awaiting a claim"
    # wait point before tearing the connection down, so the TestClient
    # portal's shutdown doesn't cancel it mid-DB-write (which would corrupt
    # the shared in-memory sqlite connection for the assertions below).
    with client.websocket_connect("/api/v1/chat/ws/cs_1"):
        await asyncio.sleep(0.2)

    await asyncio.sleep(0.05)

    row = await _get_request(sm, "dvr_1")
    session = await _get_session(sm)
    messages = await _get_messages(sm)

    assert row.status == "timed_out"
    assert session.mode == "awaiting_human"
    assert any(m.role == "system" for m in messages)


async def _drain_for(ws, seconds: float) -> list[dict]:
    """Collect whatever text frames arrive on `ws` within `seconds`, without
    blocking past the deadline if nothing (more) ever arrives. Mirrors
    test_chat_keepalive.py's helper of the same name/shape."""
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
async def test_reconnect_sweep_escalation_puts_socket_in_human_mode_not_bot_loop(timeout_ctx) -> None:
    """F1: chat.py (~1292-1301) re-fetches `row` after the reconnect sweep so
    that, if the sweep just escalated the session to human mode, the
    mode-branch a few lines below (~1329) sees `row.mode == "awaiting_human"`
    and enters `_run_human_mode` instead of falling through into the bot
    loop. `test_reconnect_sweep_times_out_an_overdue_pending_request` above
    only asserts DB state (row/session/message) -- it never confirms the
    actual socket behavior, so deleting the re-fetch block still passes it.

    Assert the client-visible consequence instead: after the sweep escalates,
    a customer send must NOT get a normal bot turn (no `typing`/`message`
    frames -- the real bot loop always answers with those; `_run_human_mode`
    silently persists a plain customer text and forwards it to the BO queue).
    Also confirm the message was in fact persisted+forwarded (proving the
    socket really is inside `_run_human_mode`'s ws-handling branch, not just
    a socket that went silent for some unrelated reason)."""
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm)
    chat_api.set_chatbot_factory(_fake_factory)

    client = _connect_ws()
    with client.websocket_connect("/api/v1/chat/ws/cs_1") as ws:
        # Give the sweep time to run and escalate before sending anything —
        # matches the timing already relied on by the test above.
        await asyncio.sleep(0.2)

        ws.send_text(json.dumps({"type": "message", "text": "still there?"}))
        frames = await _drain_for(ws, 0.5)

    assert not any(f.get("type") in ("typing", "message") for f in frames), (
        "the bot loop answered a customer send after the reconnect sweep "
        f"escalated to human mode -- the row re-fetch regressed: {frames}"
    )

    messages = await _get_messages(sm)
    assert any(
        m.role == "customer" and m.content == "still there?" for m in messages
    ), "customer text should have been persisted+forwarded by _run_human_mode, not silently dropped by the bot loop"
    assert "cs_1" in chat_api._bo_queues


@pytest.mark.asyncio
async def test_reconnect_sweep_ignores_not_due_and_already_resolved_requests(timeout_ctx) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm, request_id="dvr_future", overdue=False)
    await _seed_request(sm, request_id="dvr_verified", status="verified", overdue=True)
    chat_api.set_chatbot_factory(_fake_factory)

    client = _connect_ws()
    with client.websocket_connect("/api/v1/chat/ws/cs_1") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        frame = json.loads(ws.receive_text())
        assert frame["type"] == "typing"
        frame = json.loads(ws.receive_text())
        assert frame["type"] == "message"

    future_row = await _get_request(sm, "dvr_future")
    verified_row = await _get_request(sm, "dvr_verified")
    session = await _get_session(sm)

    assert future_row.status == "pending"
    assert verified_row.status == "verified"
    assert session.mode == "ai"


@pytest.mark.asyncio
async def test_reconnect_sweep_failure_does_not_block_the_connection(
    timeout_ctx, monkeypatch, caplog,
) -> None:
    sm, tenant_ctx, webhook = timeout_ctx
    await _seed_request(sm)
    chat_api.set_chatbot_factory(_fake_factory)

    async def _boom(request_id: str) -> None:
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(chat_api, "_check_and_timeout_verification", _boom)

    client = _connect_ws()
    with caplog.at_level(logging.ERROR, logger="src.api.chat"):
        with client.websocket_connect("/api/v1/chat/ws/cs_1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message"
            assert frame["text"] == "ok"

    assert any("reconnect sweep failed" in r.getMessage() for r in caplog.records)
