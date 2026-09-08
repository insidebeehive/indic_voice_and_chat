"""Tests for the WS-layer chat-turn-metrics failure row (turn-metrics plan,
Phase 3): a turn that raises or times out must still write a minimal
``chat_turn_metrics`` row, closing Phase 2's deliberately-accepted gap (see
``src/models/chat_turn_metrics.py``'s module docstring and
``src/api/chat.py::_record_ws_turn_failure_metric``).

Mirrors ``tests/unit/test_chat_routes.py``'s ``app``/``cost_app`` fixture
shape and its existing
``test_websocket_turn_timeout_sends_error_keeps_socket``/
``test_websocket_quota_exhaustion_sends_high_demand_message`` tests for how
to drive a hung/erroring fake agent through the real WS route.
"""

from __future__ import annotations

import asyncio as aio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from src.api import chat
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.audit import reset_suppression_state
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.dialogue.context import SessionStore
from src.models.chat_turn_metrics import ChatToolMetricRow, ChatTurnMetric
from src.models.tenant import Tenant

HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _reset_audit_suppression_state():
    reset_suppression_state()
    yield
    reset_suppression_state()


@pytest.fixture
async def app(test_engine: AsyncEngine, fake_redis):
    sm = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    session_store = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t1")

    chat.set_chatbot_factory(None)  # set per-test below
    chat.set_chat_sessionmaker(sm)
    chat.set_chat_handoff_store(session_store)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    # Exposed so tests can query chat_turn_metrics/chat_tool_metrics rows
    # against this SAME in-memory engine, and monkeypatch
    # record_chat_turn_metric's get_sessionmaker to write to it.
    a.state.sm = sm  # type: ignore[attr-defined]
    yield a
    chat.set_chatbot_factory(None)
    chat.set_chat_sessionmaker(None)
    chat.set_chat_handoff_store(None)
    set_tenant_resolver(None)


def _create_session(client: TestClient, **body) -> str:
    resp = client.post("/chat/sessions", json=body, headers=HEADERS)
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


class _HangingAgent:
    """Bare test double -- deliberately has NONE of ChatBotAgent's private
    attrs (_enable_tools/_llm_provider/_llm_model), so this also exercises
    _record_ws_turn_failure_metric's getattr-defensive reads against a
    minimal agent shape."""

    async def handle_message(self, text):
        await aio.sleep(5)

    async def summarize_session(self):
        return "summary"


class _ErrorAgent:
    async def handle_message(self, text):
        raise RuntimeError("boom")

    async def summarize_session(self):
        return "summary"


async def test_ws_turn_timeout_writes_failure_row(app: FastAPI, monkeypatch) -> None:
    monkeypatch.setattr(chat, "_TURN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(
        "src.models.chat_turn_metrics.get_sessionmaker", lambda: app.state.sm  # type: ignore[attr-defined]
    )
    client = TestClient(app)
    sid = _create_session(client)

    async def _factory(tenant, session_id, *, customer_id=None, ticket_id=None):
        return _HangingAgent()

    chat.set_chatbot_factory(_factory)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["reason"] == "timeout"

    async with app.state.sm() as db:  # type: ignore[attr-defined]
        rows = (await db.execute(select(ChatTurnMetric))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.tenant_id == "t1"
    assert row.session_id == sid
    assert row.action == "failed_timeout"
    assert row.total_ms > 0
    # Never a free-text exception string -- see the PII section of
    # src/models/chat_turn_metrics.py's module docstring.
    assert "boom" not in row.action
    # Everything else defaults to 0/False -- nothing honest to fill in for a
    # turn that never produced a ChatTurnResult.
    assert row.llm_calls == 0
    assert row.tool_calls == 0
    assert row.rounds == 0
    assert row.rounds_exhausted is False
    assert row.escalated is False


async def test_ws_turn_exception_writes_failure_row_with_no_tool_children(
    app: FastAPI, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.models.chat_turn_metrics.get_sessionmaker", lambda: app.state.sm  # type: ignore[attr-defined]
    )
    client = TestClient(app)
    sid = _create_session(client)

    async def _factory(tenant, session_id, *, customer_id=None, ticket_id=None):
        return _ErrorAgent()

    chat.set_chatbot_factory(_factory)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["reason"] == "internal"

    async with app.state.sm() as db:  # type: ignore[attr-defined]
        turn_rows = (await db.execute(select(ChatTurnMetric))).scalars().all()
        tool_rows = (await db.execute(select(ChatToolMetricRow))).scalars().all()
    assert len(turn_rows) == 1
    assert turn_rows[0].action == "failed_internal"
    assert tool_rows == []  # minimal row only -- no per-tool detail for a turn that never ran tools


async def test_ws_turn_failure_metric_write_is_best_effort(app: FastAPI, monkeypatch) -> None:
    """A DB failure on the failure-row write itself must not break the
    customer-visible error frame or the socket -- same never-raises
    contract as every other metrics write in this codebase."""
    class _BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("db unavailable")

    monkeypatch.setattr(
        "src.models.chat_turn_metrics.get_sessionmaker",
        lambda: _BrokenSessionmaker(),
    )
    client = TestClient(app)
    sid = _create_session(client)

    async def _factory(tenant, session_id, *, customer_id=None, ticket_id=None):
        return _ErrorAgent()

    chat.set_chatbot_factory(_factory)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        # Socket still alive.
        ws.send_text(json.dumps({"type": "end"}))
        ended = json.loads(ws.receive_text())
        assert ended["type"] == "ended"
