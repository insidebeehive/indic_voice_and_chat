"""Classification of per-turn chat WS exceptions into (reason, customer_message).

Covers the four branches in ``_classify_turn_error`` plus:
- WS-level tests proving quota/timeout errors surface a distinct ``reason``
  and the socket stays usable for the next turn.
- REST-level tests proving ``POST /chat/message`` maps provider failures to
  honest HTTP errors instead of a bare 500.
- A WS ``end``-frame test proving a failing ``summarize_session`` still lets
  the customer receive the ``ended`` frame.
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
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.chat import ChatSession


class _FakeProviderError(Exception):
    """Stand-in for google.genai.errors.ClientError — tests must not import
    the real SDK, only need something with a ``.code`` attribute."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# --- _classify_turn_error unit tests -------------------------------------


def test_classify_billing_cap_429():
    exc = _FakeProviderError(
        429,
        "429 RESOURCE_EXHAUSTED. Your project has exceeded its monthly "
        "spending cap. See https://ai.studio/spend for details.",
    )
    reason, message = chat_api._classify_turn_error(exc)
    assert reason == "llm_billing"
    assert message == (
        "The AI assistant is temporarily unavailable due to a service limit "
        "on our side. Our team has been notified — please try again later."
    )


def test_classify_billing_cap_case_insensitive():
    exc = _FakeProviderError(429, "SPENDING CAP exceeded, RESOURCE_EXHAUSTED")
    reason, _ = chat_api._classify_turn_error(exc)
    assert reason == "llm_billing"


def test_classify_plain_quota_429():
    exc = _FakeProviderError(429, "Resource exhausted, retry after backoff")
    reason, message = chat_api._classify_turn_error(exc)
    assert reason == "llm_quota"
    assert message == ("We're experiencing very high demand right now — "
                        "please try again in a little while.")


def test_classify_timeout_asyncio():
    reason, message = chat_api._classify_turn_error(asyncio.TimeoutError())
    assert reason == "timeout"
    assert message == "That took longer than expected — please try again."


def test_classify_timeout_builtin():
    reason, message = chat_api._classify_turn_error(TimeoutError())
    assert reason == "timeout"
    assert message == "That took longer than expected — please try again."


def test_classify_internal_fallback():
    reason, message = chat_api._classify_turn_error(ValueError("boom"))
    assert reason == "internal"
    assert message == "Sorry, something went wrong — please try again."


# --- WS-level test: billing-cap 429 surfaces reason, socket stays open ---


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
    billing_exc = _FakeProviderError(
        429,
        "429 RESOURCE_EXHAUSTED. Your project has exceeded its monthly "
        "spending cap. See https://ai.studio/spend for details.",
    )
    fake_agent.handle_message = AsyncMock(side_effect=billing_exc)
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


@pytest.mark.asyncio
async def test_ws_billing_cap_error_frame_and_socket_stays_open(ws_ctx):
    sm, fake_agent = ws_ctx

    import src.auth.middleware as mw

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)

        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert err["reason"] == "llm_billing"
            assert err["message"] == (
                "The AI assistant is temporarily unavailable due to a "
                "service limit on our side. Our team has been notified — "
                "please try again later."
            )

            # Socket must stay usable — send another turn, expect the same
            # typing/error sequence rather than a disconnect.
            ws.send_text(json.dumps({"type": "message", "text": "hi again"}))
            typing2 = json.loads(ws.receive_text())
            assert typing2["type"] == "typing"
            err2 = json.loads(ws.receive_text())
            assert err2["type"] == "error"
            assert err2["reason"] == "llm_billing"

    assert fake_agent.handle_message.await_count == 2


def _connect(fake_tenant):
    import src.auth.middleware as mw

    patcher = patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant))
    patcher.start()
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    client = TestClient(app)
    return patcher, client


@pytest.mark.asyncio
async def test_ws_plain_quota_429_error_frame(ws_ctx):
    sm, fake_agent = ws_ctx
    quota_exc = _FakeProviderError(429, "Resource exhausted, retry after backoff")
    fake_agent.handle_message = AsyncMock(side_effect=quota_exc)

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    patcher, client = _connect(fake_tenant)
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert err["reason"] == "llm_quota"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_ws_timeout_error_frame(ws_ctx):
    sm, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(side_effect=asyncio.TimeoutError())

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300

    patcher, client = _connect(fake_tenant)
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            typing = json.loads(ws.receive_text())
            assert typing["type"] == "typing"
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert err["reason"] == "timeout"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_ws_end_frame_survives_summarize_failure(ws_ctx):
    """summarize_session raising on an explicit 'end' must not black-hole the
    socket — the customer must still get the 'ended' frame (empty summary)."""
    sm, fake_agent = ws_ctx
    fake_agent.summarize_session = AsyncMock(side_effect=RuntimeError("llm down"))

    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = 300
    fake_tenant.settings.events_webhook_url = None  # skip the best-effort webhook call

    patcher, client = _connect(fake_tenant)
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "end"}))
            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended"
            assert ended["summary"] == ""
            assert ended["reason"] == "customer_ended"
    finally:
        patcher.stop()

    fake_agent.summarize_session.assert_awaited_once()


# --- REST /chat/message error-mapping tests ------------------------------


@pytest_asyncio.fixture
async def rest_ctx():
    """Minimal app exposing POST /chat/message backed by a fake agent factory —
    no DB / RAG setup needed since the agent raises before any persistence."""
    fake_agent = MagicMock()

    async def fake_factory(tenant, session_id, *, customer_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )

    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    client = TestClient(app)

    yield client, fake_agent

    chat_api.set_chatbot_factory(None)
    set_tenant_resolver(None)


def test_rest_chat_message_billing_cap_returns_503(rest_ctx):
    client, fake_agent = rest_ctx
    billing_exc = _FakeProviderError(
        429,
        "429 RESOURCE_EXHAUSTED. Your project has exceeded its monthly "
        "spending cap. See https://ai.studio/spend for details.",
    )
    fake_agent.handle_message = AsyncMock(side_effect=billing_exc)

    resp = client.post(
        "/api/v1/chat/message",
        json={"message": "hi"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["reason"] == "llm_billing"
    assert detail["message"] == (
        "The AI assistant is temporarily unavailable due to a "
        "service limit on our side. Our team has been notified — "
        "please try again later."
    )


def test_rest_chat_message_internal_error_returns_500(rest_ctx):
    client, fake_agent = rest_ctx
    fake_agent.handle_message = AsyncMock(side_effect=ValueError("boom"))

    resp = client.post(
        "/api/v1/chat/message",
        json={"message": "hi"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["reason"] == "internal"
    assert detail["message"] == "Sorry, something went wrong — please try again."
