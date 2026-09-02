"""Route-level tests for /api/v1/chat/* endpoints."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.api import chat
from src.api.deps import get_db_session
from src.auth import TenantContext, register_tenant_for_test
from src.auth.audit import reset_suppression_state, token_fingerprint
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.dialogue.context import SessionStore
from src.interfaces.llm import ILLMProvider, LLMResult, ToolCall
from src.interfaces.vector_store import Document
from src.models.database import Base
from src.models.tenant import Tenant
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import HybridRetriever, RetrievalConfig


HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _reset_audit_suppression_state():
    """log_denied's per-(reason, client_ip) suppression window is module-level
    state — without resetting it, one test's auth-rejection log gets silently
    suppressed by a prior test's rejections for the same reason."""
    reset_suppression_state()
    yield
    reset_suppression_state()


class _FakeLLM(ILLMProvider):
    def __init__(self, payload: dict, usage: dict | None = None) -> None:
        self._json = json.dumps(payload)
        self._usage = usage or {}

    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(text=self._json, finish_reason="stop", usage=self._usage)

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


@pytest.fixture
async def app(tmp_faiss_index: str, fake_redis):
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8, reranking=False),
    )
    await retriever.index([
        Document(id="c1", content="Plan B has 500GB unlimited.", metadata={"filename": "plans.md", "page": 0})
    ])

    session_store = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t1")

    # In-memory DB for chat_sessions / chat_messages.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    async def factory(tenant: TenantContext, session_id: str, *, customer_id=None, ticket_id=None) -> ChatBotAgent:
        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=_FakeLLM({
                "response_text": "Plan B has 500GB.",
                "language": "en",
                "sources_used": ["plans.md:0"],
                "confidence": "high",
                "action": "none",
            }),
            retriever=retriever,
            company_name=tenant.settings.name,
            store=session_store,
        )

    chat.set_chatbot_factory(factory)
    chat.set_chat_sessionmaker(sm)
    chat.set_chat_handoff_store(session_store)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    yield a
    chat.set_chatbot_factory(None)
    chat.set_chat_sessionmaker(None)
    chat.set_chat_handoff_store(None)
    set_tenant_resolver(None)
    await engine.dispose()


@pytest.fixture
async def cost_app(tmp_faiss_index: str, fake_redis):
    """Like ``app``, but the agent reports real token usage + an
    llm_provider/model (as bootstrap.py's factory now threads through), and
    the DB is seeded with a matching token-rate ProviderCost row — for
    exercising _persist_turn's chat cost bookkeeping end to end."""
    from src.models.tenant import ProviderCost

    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8, reranking=False),
    )
    await retriever.index([
        Document(id="c1", content="Plan B has 500GB unlimited.", metadata={"filename": "plans.md", "page": 0})
    ])

    session_store = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t1")

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        s.add(ProviderCost(
            kind="llm", provider="gemini", model="gemini-3.5-flash",
            cost_per_1k_input_tokens=0.0003, cost_per_1k_output_tokens=0.0025))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    async def factory(tenant: TenantContext, session_id: str, *, customer_id=None, ticket_id=None) -> ChatBotAgent:
        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=_FakeLLM({
                "response_text": "Plan B has 500GB.",
                "language": "en",
                "sources_used": ["plans.md:0"],
                "confidence": "high",
                "action": "none",
            }, usage={"prompt_tokens": 100, "completion_tokens": 40}),
            retriever=retriever,
            company_name=tenant.settings.name,
            store=session_store,
            llm_provider="gemini",
            llm_model="gemini-3.5-flash",
        )

    chat.set_chatbot_factory(factory)
    chat.set_chat_sessionmaker(sm)
    chat.set_chat_handoff_store(session_store)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    yield a, sm
    chat.set_chatbot_factory(None)
    chat.set_chat_sessionmaker(None)
    chat.set_chat_handoff_store(None)
    set_tenant_resolver(None)
    await engine.dispose()


def _create_session(client: TestClient, **body) -> str:
    resp = client.post("/chat/sessions", json=body, headers=HEADERS)
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


# --- Session lifecycle --------------------------------------------------


def test_create_session_returns_id_greeting_ws_url(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/chat/sessions",
                       json={"customer_name": "Raju", "language": "en"}, headers=HEADERS)
    assert resp.status_code == 201
    body = resp.json()
    assert body["session_id"].startswith("cs_")
    assert "Raju" in body["greeting"]
    assert body["ws_url"].endswith(f"/api/v1/chat/ws/{body['session_id']}")


def test_create_session_greeting_in_requested_language(app: FastAPI) -> None:
    """CRM told us the customer's language up front -- greet in it, not English."""
    client = TestClient(app)
    resp = client.post("/chat/sessions",
                       json={"customer_name": "Ravi", "language": "ta"}, headers=HEADERS)
    assert resp.status_code == 201
    greeting = resp.json()["greeting"]
    assert greeting.startswith("வணக்கம் Ravi")
    assert "Hello" not in greeting


def test_create_session_greeting_strips_region_suffix(app: FastAPI) -> None:
    """"mr-IN" must hit the same template as "mr"."""
    client = TestClient(app)
    resp = client.post("/chat/sessions",
                       json={"customer_name": "Sneha", "language": "mr-IN"}, headers=HEADERS)
    assert resp.json()["greeting"].startswith("नमस्कार Sneha")


def test_create_session_greeting_defaults_to_tenant_language(app: FastAPI) -> None:
    """No language from the CRM -> tenant default (hi), not English."""
    client = TestClient(app)
    resp = client.post("/chat/sessions", json={"customer_name": "Amit"}, headers=HEADERS)
    body = resp.json()
    assert body["greeting"].startswith("नमस्ते Amit")


def test_create_session_greeting_unknown_language_falls_back_to_english(app: FastAPI) -> None:
    """A language we have no template for still gets a sane opener."""
    client = TestClient(app)
    resp = client.post("/chat/sessions",
                       json={"customer_name": "Lee", "language": "fr"}, headers=HEADERS)
    assert resp.json()["greeting"] == "Hello Lee, how can I help?"


def test_create_session_greeting_without_name(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/chat/sessions", json={"language": "hi"}, headers=HEADERS)
    assert resp.json()["greeting"] == "नमस्ते, मैं आपकी कैसे मदद करूँ?"


def test_create_session_language_name_falls_back_to_tenant_default(app: FastAPI) -> None:
    """A spelled-out name ("Hindi") is not a code: fall back per the CRM contract,
    and don't persist the junk value on the session row."""
    client = TestClient(app)
    resp = client.post("/chat/sessions",
                       json={"customer_name": "Amit", "language": "Hindi"}, headers=HEADERS)
    sid = resp.json()["session_id"]
    assert resp.json()["greeting"].startswith("नमस्ते Amit")
    detail = client.get(f"/chat/sessions/{sid}", headers=HEADERS).json()
    assert detail["language"] == "hi"


def test_create_session_persists_normalized_language(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/chat/sessions",
                       json={"customer_name": "Sneha", "language": "MR-IN"}, headers=HEADERS)
    sid = resp.json()["session_id"]
    detail = client.get(f"/chat/sessions/{sid}", headers=HEADERS).json()
    assert detail["language"] == "mr"


def test_greeting_map_covers_all_contract_languages() -> None:
    """Every code in docs/crm-chat-media-contract.md has a template.
    Note "od" (this platform's Odia code), not the ISO "or"."""
    from src.api import chat
    expected = {"hi", "en", "bn", "gu", "kn", "ml", "mr", "od", "pa", "ta", "te", "as"}
    assert expected <= set(chat._GREETINGS)
    for code, template in chat._GREETINGS.items():
        assert "{who}" in template, code
        assert chat._greeting("Acme", "Raju", code).startswith(template.split("{")[0] + " Raju")


def test_create_session_requires_auth(app: FastAPI) -> None:
    client = TestClient(app)
    assert client.post("/chat/sessions", json={}).status_code == 401


def test_list_sessions_scoped_to_tenant(app: FastAPI) -> None:
    client = TestClient(app)
    _create_session(client, customer_id="cust_1")
    _create_session(client, customer_id="cust_2")
    resp = client.get("/chat/sessions", headers=HEADERS)
    assert resp.status_code == 200
    assert len(resp.json()["sessions"]) == 2


def test_get_session_detail_includes_messages(app: FastAPI) -> None:
    client = TestClient(app)
    sid = _create_session(client, customer_name="Raju")
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "Plan B?"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        assert json.loads(ws.receive_text())["type"] == "message"

    resp = client.get(f"/chat/sessions/{sid}", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["message_count"] == 2
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["customer", "agent"]
    assert body["messages"][0]["content"] == "Plan B?"
    assert body["messages"][1]["content"] == "Plan B has 500GB."


def test_get_unknown_session_404(app: FastAPI) -> None:
    client = TestClient(app)
    assert client.get("/chat/sessions/nope", headers=HEADERS).status_code == 404


# --- WebSocket (capability model) --------------------------------------


def test_websocket_round_trip(app: FastAPI) -> None:
    client = TestClient(app)
    sid = _create_session(client)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "Plan B?"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        reply = json.loads(ws.receive_text())
    assert reply["type"] == "message"
    assert reply["session_id"] == sid
    assert reply["text"] == "Plan B has 500GB."
    assert "plans.md:0" in reply["sources"]


def test_websocket_normal_turn_still_logs_session_id_in_clear(
    app: FastAPI, caplog
) -> None:
    """Control test for the fingerprinting carve-out: a NORMAL successful WS
    turn (not a rejection) must still show session_id in the clear in the
    existing disconnect log line -- proving the carve-out added for the
    session-not-found close (site 6) is narrow, not a blanket change across
    the file."""
    client = TestClient(app)
    sid = _create_session(client)
    with caplog.at_level(logging.INFO):
        with client.websocket_connect(f"/chat/ws/{sid}") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "Plan B?"}))
            assert json.loads(ws.receive_text())["type"] == "typing"
            assert json.loads(ws.receive_text())["type"] == "message"

    disconnect_records = [
        r for r in caplog.records if r.getMessage() == "chat ws client disconnected"
    ]
    assert len(disconnect_records) == 1, caplog.records
    assert disconnect_records[0].session_id == sid

    # No auth_rejected record for a normal, successful connection.
    assert not [r for r in caplog.records if getattr(r, "event", None) == "auth_rejected"]


def test_websocket_unknown_session_closes(app: FastAPI) -> None:
    client = TestClient(app)
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/chat/ws/cs_doesnotexist") as ws:
            ws.receive_text()


def test_websocket_unknown_session_logs_auth_rejected_fingerprinted(
    app: FastAPI, caplog
) -> None:
    """WS /ws/{session_id} — session_id IS the entire credential on this route,
    so an unknown/guessed session_id must be logged (this is the highest-value
    site in the whole remediation plan), but only as a fingerprint. The raw
    guessed session_id must never appear in any log record from the attempt."""
    from starlette.websockets import WebSocketDisconnect

    guessed = "cs_guessedvalue123"
    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/chat/ws/{guessed}") as ws:
                ws.receive_text()

    rejected = [r for r in caplog.records if getattr(r, "event", None) == "auth_rejected"]
    assert len(rejected) == 1, rejected
    record = rejected[0]
    assert record.reason == "chat_ws_session_not_found"
    assert record.levelno == logging.WARNING
    assert record.session_fp == token_fingerprint(guessed, domain="vox-logfp-sid-v1")

    # The raw guessed session_id must never leak into any captured record.
    for r in caplog.records:
        assert guessed not in r.getMessage()
        for value in vars(r).values():
            assert value != guessed


def test_websocket_factory_failure_closes_cleanly(app: FastAPI) -> None:
    # A factory failure (e.g. DB connection pool exhausted under concurrent
    # session load) must close the socket with a clear code instead of
    # leaving the client waiting forever for a response that never comes.
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    sid = _create_session(client)

    async def _broken_factory(tenant, session_id, *, customer_id=None, ticket_id=None):
        raise RuntimeError("DB connection pool exhausted")

    chat.set_chatbot_factory(_broken_factory)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/chat/ws/{sid}") as ws:
            ws.receive_text()
    assert exc_info.value.code == 1011


def test_websocket_passes_customer_id_to_factory(app: FastAPI) -> None:
    # The WS connect path already fetched the ChatSession row — it must hand
    # customer_id to the factory so the factory doesn't re-query the same row
    # on another pool checkout (part of the concurrent-session-burst fixes).
    client = TestClient(app)
    sid = _create_session(client, user_id="cust-7")

    seen: dict = {}
    orig_factory = chat._factory

    async def _recording_factory(tenant, session_id, *, customer_id=None, ticket_id=None):
        seen["customer_id"] = customer_id
        return await orig_factory(tenant, session_id)

    chat.set_chatbot_factory(_recording_factory)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        ws.receive_text()  # typing
        ws.receive_text()  # reply
    assert seen["customer_id"] == "cust-7"


def test_websocket_turn_timeout_sends_error_keeps_socket(app: FastAPI, monkeypatch) -> None:
    # A hung provider call must not black-hole the turn: the wait_for ceiling
    # fires, the customer gets an error frame, and the socket stays usable.
    import asyncio as aio

    monkeypatch.setattr(chat, "_TURN_TIMEOUT_S", 0.05)
    client = TestClient(app)
    sid = _create_session(client)

    class _HangingAgent:
        async def handle_message(self, text):
            await aio.sleep(5)

        async def summarize_session(self):
            return "summary"

    async def _factory(tenant, session_id, *, customer_id=None, ticket_id=None):
        return _HangingAgent()

    chat.set_chatbot_factory(_factory)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        # Socket still alive — a clean end works.
        ws.send_text(json.dumps({"type": "end"}))
        ended = json.loads(ws.receive_text())
        assert ended["type"] == "ended"


def test_websocket_quota_exhaustion_sends_high_demand_message(app: FastAPI) -> None:
    # A 429 (provider quota exhausted) must tell the customer it's demand,
    # not a bug — the generic "something went wrong" invites retrying now.
    client = TestClient(app)
    sid = _create_session(client)

    class _QuotaError(Exception):
        code = 429

    class _QuotaAgent:
        async def handle_message(self, text):
            raise _QuotaError("429 RESOURCE_EXHAUSTED")

    async def _factory(tenant, session_id, *, customer_id=None, ticket_id=None):
        return _QuotaAgent()

    chat.set_chatbot_factory(_factory)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "hi"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert "high demand" in err["message"]


def test_create_session_returns_fast_when_webhook_slow(app: FastAPI, monkeypatch) -> None:
    # The session_started BO webhook is fire-and-forget: a slow/unreachable
    # tenant CRM endpoint must not stall session creation (it used to block
    # up to ~16s inline, wedging session bursts).
    import asyncio as aio
    import time

    import src.api.chat_webhooks as chat_webhooks

    async def _slow_webhook(tenant, event_type, payload):
        await aio.sleep(1.5)
        return True

    monkeypatch.setattr(chat_webhooks, "send_bo_webhook", _slow_webhook)
    client = TestClient(app)
    t0 = time.monotonic()
    resp = client.post("/chat/sessions", json={}, headers=HEADERS)
    elapsed = time.monotonic() - t0
    assert resp.status_code == 201
    assert elapsed < 1.0, f"session create blocked on webhook ({elapsed:.2f}s)"


def test_websocket_invalid_json_returns_error(app: FastAPI) -> None:
    client = TestClient(app)
    sid = _create_session(client)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text("not json")
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"


def test_websocket_image_message_persists_as_image(app: FastAPI) -> None:
    import base64
    client = TestClient(app)
    sid = _create_session(client)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({
            "type": "image", "mime": "image/jpeg",
            "data": base64.b64encode(b"imgbytes").decode(), "text": "what plan is this?"}))
        assert json.loads(ws.receive_text())["type"] == "typing"
        reply = json.loads(ws.receive_text())
    assert reply["type"] == "message"
    detail = client.get(f"/chat/sessions/{sid}", headers=HEADERS).json()
    assert detail["messages"][0]["type"] == "image"


def test_websocket_image_missing_data_errors(app: FastAPI) -> None:
    client = TestClient(app)
    sid = _create_session(client)
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "image", "mime": "image/jpeg"}))
        assert json.loads(ws.receive_text())["type"] == "error"


async def test_request_call_returns_voice_url_and_stores_context(app: FastAPI, fake_redis) -> None:
    client = TestClient(app)
    sid = _create_session(client, customer_name="Raju")
    # one turn so there's something to summarize
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "my plan?"}))
        json.loads(ws.receive_text())  # typing
        json.loads(ws.receive_text())  # message

    resp = client.post(f"/chat/{sid}/call")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "/chat/voice?tenant=t1&handoff=" in body["call_url"]
    token = body["call_id"]
    # The handoff context was stashed in Redis for the voice bridge to pick up.
    raw = await fake_redis.get(f"chat_handoff:{token}")
    assert raw is not None
    ctx = json.loads(raw)
    assert ctx["chat_session_id"] == sid
    assert ctx["customer_name"] == "Raju"
    assert "chat_summary" in ctx


def test_voice_ws_unknown_tenant_logs_auth_rejected(app: FastAPI, caplog) -> None:
    """chat_voice_ws's bare `except Exception:` around tenant resolution used
    to swallow the failure silently. It must now log with exc_info populated.

    A downstream tenant-resolution helper in src/auth/middleware.py (Package B,
    landing in parallel) may ALSO legitimately log its own auth_rejected record
    for the same underlying rejection (reason="unknown_tenant_slug") -- that's
    expected, so this only filters by OUR reason rather than asserting a total
    record count for the whole connection attempt."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/chat/voice?tenant=doesnotexist") as ws:
                ws.receive_text()

    ours = [
        r for r in caplog.records
        if getattr(r, "reason", None) == "voice_ws_unknown_tenant"
    ]
    assert len(ours) == 1, caplog.records
    record = ours[0]
    assert record.event == "auth_rejected"
    assert record.levelno == logging.WARNING
    assert record.exc_info is not None


def test_upload_endpoint_processes_and_persists(app: FastAPI) -> None:
    import io
    client = TestClient(app)
    sid = _create_session(client)
    resp = client.post(
        f"/chat/{sid}/upload",
        files={"file": ("err.jpg", io.BytesIO(b"imgbytes"), "image/jpeg")},
        data={"text": "what is this error?"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["session_id"] == sid
    detail = client.get(f"/chat/sessions/{sid}", headers=HEADERS).json()
    assert detail["messages"][0]["type"] == "image"


def test_websocket_end_marks_session_ended(app: FastAPI) -> None:
    client = TestClient(app)
    sid = _create_session(client)
    # one turn so there's something to summarize
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "Plan B?"}))
        json.loads(ws.receive_text())  # typing
        json.loads(ws.receive_text())  # message
        ws.send_text(json.dumps({"type": "end"}))
        ended = json.loads(ws.receive_text())
    assert ended["type"] == "ended"
    detail = client.get(f"/chat/sessions/{sid}", headers=HEADERS).json()
    assert detail["status"] == "ended"
    assert detail["summary"]  # a session summary was stored


# --- HTTP single-turn channel ------------------------------------------


async def test_persist_turn_writes_cost_onto_agent_message_and_session(cost_app) -> None:
    """The customer message row gets no cost fields; the agent message row
    gets input/output tokens + cost + llm identity; the session's running
    totals are bumped by the same turn cost."""
    from sqlalchemy import select

    from src.models.chat import ChatMessage, ChatSession

    app, sm = cost_app
    client = TestClient(app)
    sid = _create_session(client)
    resp = client.post(
        "/chat/message", json={"session_id": sid, "message": "Tell me about Plan B"}, headers=HEADERS)
    assert resp.status_code == 200

    async with sm() as s:
        rows = (await s.execute(
            select(ChatMessage).where(ChatMessage.session_id == sid).order_by(ChatMessage.id)
        )).scalars().all()
        session_row = await s.get(ChatSession, sid)

    assert [r.role for r in rows] == ["customer", "agent"]
    customer_msg, agent_msg = rows
    # Customer message: no cost fields.
    assert customer_msg.input_tokens is None
    assert customer_msg.output_tokens is None
    assert customer_msg.cost is None
    assert customer_msg.llm_provider is None
    # Agent message: usage=100 prompt / 40 completion tokens @ 0.0003/0.0025 per 1k.
    # (100/1000 * 0.0003) + (40/1000 * 0.0025) = 0.00003 + 0.0001 = 0.00013
    assert agent_msg.input_tokens == 100
    assert agent_msg.output_tokens == 40
    assert agent_msg.cost == pytest.approx(0.00013)
    assert agent_msg.llm_provider == "gemini"
    assert agent_msg.llm_model == "gemini-3.5-flash"
    # Session running totals match the single turn.
    assert session_row.input_tokens == 100
    assert session_row.output_tokens == 40
    assert session_row.cost == pytest.approx(0.00013)


async def test_persist_turn_survives_cost_computation_failure(cost_app, monkeypatch) -> None:
    """Regression test: a failing cost computation (e.g. the ProviderCost rate
    lookup SELECT erroring out -- on Postgres this aborts the *whole*
    transaction if it shares a session with the message inserts) must never
    take the customer/agent message rows down with it. _persist_turn now runs
    the rate lookup on its own independent DB session so a failure there can't
    poison the transaction the message rows are persisted on."""
    from sqlalchemy import select

    from src.models.chat import ChatMessage, ChatSession

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated rate-lookup failure")

    monkeypatch.setattr(chat, "compute_chat_turn_cost", _boom)

    app, sm = cost_app
    client = TestClient(app)
    sid = _create_session(client)
    resp = client.post(
        "/chat/message", json={"session_id": sid, "message": "Tell me about Plan B"}, headers=HEADERS)
    assert resp.status_code == 200

    async with sm() as s:
        rows = (await s.execute(
            select(ChatMessage).where(ChatMessage.session_id == sid).order_by(ChatMessage.id)
        )).scalars().all()
        session_row = await s.get(ChatSession, sid)

    # Both message rows persisted despite the cost-computation exception.
    assert [r.role for r in rows] == ["customer", "agent"]
    customer_msg, agent_msg = rows
    assert customer_msg.content == "Tell me about Plan B"
    assert agent_msg.content == "Plan B has 500GB."
    # Cost bookkeeping was skipped (exception swallowed) -- no cost recorded,
    # but message persistence and the session's message_count were unaffected.
    assert agent_msg.cost is None
    assert session_row.message_count == 2


async def test_persist_turn_cost_computation_uses_isolated_session(cost_app, monkeypatch) -> None:
    """Regression test for the actual session-isolation mechanism (the
    predecessor test above only proved the outer exception handler swallows
    errors -- it made compute_chat_turn_cost raise before touching any
    session, so it passed even with the isolation fix reverted, i.e. cost_db
    aliased back to db). This test proves the session object passed into
    compute_chat_turn_cost is NOT the same object as the session the
    customer/agent ChatMessage inserts are staged on -- the mechanism the fix
    actually relies on. It would fail if _persist_turn were reverted to reuse
    `db` for the cost lookup instead of opening `cost_db`."""
    from src.api.chat_cost import compute_chat_turn_cost as real_compute_chat_turn_cost

    app, sm = cost_app

    created_session_ids: list[int] = []
    real_sessionmaker = sm

    def tracking_sessionmaker():
        session = real_sessionmaker()
        created_session_ids.append(id(session))
        return session

    monkeypatch.setattr(chat, "_sm", lambda: tracking_sessionmaker)

    cost_call_session_id: dict[str, int] = {}

    async def _tracking_compute(session, **kwargs):
        cost_call_session_id["id"] = id(session)
        return await real_compute_chat_turn_cost(session, **kwargs)

    monkeypatch.setattr(chat, "compute_chat_turn_cost", _tracking_compute)

    client = TestClient(app)
    sid = _create_session(client)
    resp = client.post(
        "/chat/message", json={"session_id": sid, "message": "Tell me about Plan B"}, headers=HEADERS)
    assert resp.status_code == 200

    # _persist_turn opens (at least) two sessions via _sm(): the outer `db`
    # the message rows are staged on (created first), and the isolated
    # `cost_db` passed to compute_chat_turn_cost (created after).
    assert len(created_session_ids) >= 2
    db_session_id = created_session_ids[0]
    assert "id" in cost_call_session_id, "compute_chat_turn_cost was never called"
    assert cost_call_session_id["id"] == created_session_ids[-1]
    assert cost_call_session_id["id"] != db_session_id


def test_post_message_creates_session_and_returns_answer(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/chat/message", json={"message": "Tell me about Plan B"}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["response_text"] == "Plan B has 500GB."
    assert body["session_id"].startswith("cs_")
    assert "plans.md:0" in body["sources_used"]


def test_get_history_returns_persisted_turns(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/chat/message", json={"session_id": "hist-test", "message": "First Q"}, headers=HEADERS)
    client.post("/chat/message", json={"session_id": "hist-test", "message": "Second Q"}, headers=HEADERS)
    resp = client.get("/chat/history/hist-test", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["history"]) == 4  # 2 user + 2 agent
    assert body["history"][0]["content"] == "First Q"


def test_post_when_factory_unset_returns_503() -> None:
    chat.set_chatbot_factory(None)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="X"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    client = TestClient(a)
    resp = client.post("/chat/message", json={"message": "x"}, headers=HEADERS)
    assert resp.status_code == 503
    set_tenant_resolver(None)


def test_post_without_auth_returns_401(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/chat/message", json={"message": "x"})
    assert resp.status_code == 401


# --- BO handover: claim + agent WebSocket ---------------------------------

# Fixture using an escalating FakeLLM so tests can trigger escalation via the
# customer WS without touching the DB directly (avoids aiosqlite/event-loop issues).
class _FakeLLMEscalating(_FakeLLM):
    """Returns an escalate_to_human tool call on the first tool-enabled call,
    then plain text on the second (after tool result is appended)."""
    def __init__(self):
        super().__init__({
            "response_text": "Connecting you to a human agent.",
            "language": "en",
            "sources_used": [],
            "confidence": "high",
            "action": "escalate",
        })
        self._round = 0

    async def generate(self, messages, config) -> LLMResult:  # type: ignore[override]
        self._round += 1
        if self._round == 1 and getattr(config, "tools", None):
            return LLMResult(
                text="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(
                    id="tc1", name="escalate_to_human",
                    arguments={"reason": "user needs help", "summary": "refund query"},
                )],
            )
        return await super().generate(messages, config)


@pytest.fixture
async def escalating_app(tmp_faiss_index: str, fake_redis, tmp_path):
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8, reranking=False),
    )
    await retriever.index([
        Document(id="c1", content="refund policy", metadata={"filename": "policy.md", "page": 0})
    ])
    session_store = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t1")
    # NullPool + file DB so TestClient's background event loop creates fresh
    # connections (not reused from pytest-asyncio's event loop), and all
    # connections share the same persistent data.
    db_path = tmp_path / "test_handover.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    async def factory(tenant: TenantContext, session_id: str, *, customer_id=None, ticket_id=None) -> ChatBotAgent:
        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=_FakeLLMEscalating(),
            retriever=retriever,
            company_name="Acme",
            store=session_store,
            enable_tools=True,
        )

    chat.set_chatbot_factory(factory)
    chat.set_chat_sessionmaker(sm)
    chat.set_chat_handoff_store(session_store)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    yield a
    chat.set_chatbot_factory(None)
    chat.set_chat_sessionmaker(None)
    chat.set_chat_handoff_store(None)
    set_tenant_resolver(None)
    await engine.dispose()


def test_claim_session_and_agent_ws(escalating_app: FastAPI) -> None:
    """Full BO handover: customer WS escalates → claim → agent-ws history + reply."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(escalating_app)
    sid = _create_session(client)

    # --- Trigger escalation via the customer WS ---
    with client.websocket_connect(f"/chat/ws/{sid}") as ws:
        ws.send_text(json.dumps({"type": "message", "text": "I want a refund"}))
        # Consume: typing, message, escalation, mode_change
        frames = []
        for _ in range(4):
            try:
                frames.append(json.loads(ws.receive_text()))
            except WebSocketDisconnect:  # pragma: no cover
                break
    types = [f["type"] for f in frames]
    assert "mode_change" in types, f"expected mode_change, got: {types}"

    # --- Claim ---
    r = client.post(
        f"/chat/sessions/{sid}/claim",
        json={"agent_id": "a01", "agent_name": "Priya"},
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "claimed", "agent_id": "a01"}

    # Re-claim returns 409.
    r2 = client.post(
        f"/chat/sessions/{sid}/claim",
        json={"agent_id": "a01", "agent_name": "Priya"},
        headers=HEADERS,
    )
    assert r2.status_code == 409

    # --- Agent WebSocket: history + reply + end ---
    with client.websocket_connect(
        f"/chat/sessions/{sid}/agent-ws?token=test-token"
    ) as ws:
        hist = json.loads(ws.receive_text())
        assert hist["type"] == "history"
        assert len(hist["messages"]) >= 1
        assert hist["messages"][0]["role"] in ("customer", "agent")

        # Agent sends a reply and ends the session.
        ws.send_text(json.dumps({"type": "reply", "text": "Hi, I'm Priya. Let me help!"}))
        ws.send_text(json.dumps({"type": "end"}))
        # Drain all frames (possibly a "Customer disconnected" system notification)
        # until the server closes — that close happens AFTER _end_session commits.
        while True:
            try:
                ws.receive_text()
            except WebSocketDisconnect:
                break

    # Verify via HTTP: session ended and agent message stored.
    detail = client.get(f"/chat/sessions/{sid}", headers=HEADERS).json()
    assert detail["status"] == "ended", f"expected ended, got: {detail['status']}"
    ha = [m for m in detail["messages"] if m["role"] == "human_agent"]
    assert len(ha) == 1, f"expected 1 human_agent message: {detail['messages']}"
    assert ha[0]["content"] == "Hi, I'm Priya. Let me help!"

    # Teardown queues so they don't bleed into other tests.
    chat._bo_queues.pop(sid, None)
    chat._customer_queues.pop(sid, None)


def test_agent_ws_invalid_token_closes(app: FastAPI) -> None:
    """WS closes with 'invalid token' when token is wrong (no DB access needed)."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    sid = _create_session(client)
    # Session is in 'ai' mode — token check runs before mode check, so
    # wrong-token always closes regardless of mode.
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/chat/sessions/{sid}/agent-ws?token=wrong-token"
        ) as ws:
            ws.receive_text()


def test_agent_ws_wrong_mode_closes(app: FastAPI) -> None:
    """WS closes when session is not in handover mode (ai mode by default)."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    sid = _create_session(client)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/chat/sessions/{sid}/agent-ws?token=test-token"
        ) as ws:
            ws.receive_text()


def test_agent_ws_wrong_mode_close_is_not_logged(app: FastAPI, caplog) -> None:
    """The 'session not in handover mode' close (and the cross-tenant
    `row is None or row.tenant_id != tenant.id` close it sits behind) are
    explicitly OUT OF SCOPE for this change -- confirm neither one produces
    an auth_rejected record."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    sid = _create_session(client)

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/chat/sessions/{sid}/agent-ws?token=test-token"
            ) as ws:
                ws.receive_text()

    assert not [r for r in caplog.records if getattr(r, "event", None) == "auth_rejected"]


def test_agent_ws_missing_token_logs_auth_rejected(app: FastAPI, caplog) -> None:
    """Site 2: no ?token= at all -- benign (client just didn't send one), so
    this logs at INFO, not WARNING. session_id is an ordinary correlation
    field on this route (the credential here is the token, not session_id),
    so it's logged in the clear."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    sid = _create_session(client)

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/chat/sessions/{sid}/agent-ws") as ws:
                ws.receive_text()

    rejected = [r for r in caplog.records if getattr(r, "event", None) == "auth_rejected"]
    assert len(rejected) == 1, rejected
    record = rejected[0]
    assert record.reason == "agent_ws_missing_token"
    assert record.levelno == logging.INFO
    assert record.session_id == sid


def test_agent_ws_invalid_token_logs_auth_rejected(app: FastAPI, caplog) -> None:
    """Site 4: token resolved but no tenant found for it (existing
    test_agent_ws_invalid_token_closes covers the close behavior; this
    confirms the log line added alongside it)."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    sid = _create_session(client)

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/chat/sessions/{sid}/agent-ws?token=wrong-token"
            ) as ws:
                ws.receive_text()

    rejected = [r for r in caplog.records if getattr(r, "event", None) == "auth_rejected"]
    assert len(rejected) == 1, rejected
    record = rejected[0]
    assert record.reason == "agent_ws_token_unknown"
    assert record.levelno == logging.WARNING
    assert record.session_id == sid
    assert record.token_fp == token_fingerprint("wrong-token")


def test_agent_ws_token_resolve_error_logs_with_exc_info(
    app: FastAPI, caplog, monkeypatch
) -> None:
    """Site 3: the token resolver itself throwing (not just an unknown token)
    is an ops signal, not merely an attacker signal -- exc_info must be
    populated."""
    from starlette.websockets import WebSocketDisconnect

    async def _boom(token: str):
        raise RuntimeError("resolver backend unavailable")

    monkeypatch.setattr("src.auth.middleware.tenant_from_bearer_token", _boom)

    client = TestClient(app)
    sid = _create_session(client)

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/chat/sessions/{sid}/agent-ws?token=some-token"
            ) as ws:
                ws.receive_text()

    rejected = [r for r in caplog.records if getattr(r, "event", None) == "auth_rejected"]
    assert len(rejected) == 1, rejected
    record = rejected[0]
    assert record.reason == "agent_ws_token_resolve_error"
    assert record.levelno == logging.WARNING
    assert record.session_id == sid
    assert record.token_fp == token_fingerprint("some-token")
    assert record.exc_info is not None


# --- Cross-tenant chat session access (audit logging) --------------------
#
# An authenticated tenant probing another tenant's chat session ids must be
# logged (event="cross_tenant_access_denied", distinct from "auth_rejected"
# so it can never collide with the auth-failure counts asserted above),
# without changing any existing HTTP/WS response.


async def _seed_foreign_chat_session(session_id: str, *, tenant_id: str = "t2", mode: str = "ai") -> None:
    """Insert a ChatSession row owned by a tenant OTHER than the default test
    tenant ("t1"/HEADERS) -- simulates tenant A guessing or reusing tenant
    B's real session id. Uses the sessionmaker already injected into `chat`
    by the `app` fixture (chat.set_chat_sessionmaker)."""
    from src.models.chat import ChatSession
    from src.models.tenant import Tenant

    sm = chat._sm()
    async with sm() as s:
        if await s.get(Tenant, tenant_id) is None:
            s.add(Tenant(id=tenant_id, slug=tenant_id, name="Other Co"))
        s.add(ChatSession(
            id=session_id, tenant_id=tenant_id, language="en", status="active",
            extra_data={}, mode=mode,
        ))
        await s.commit()


async def _seed_own_chat_session(session_id: str, *, mode: str = "ai") -> None:
    """Insert a ChatSession row owned by the default test tenant ("t1")."""
    from src.models.chat import ChatSession

    sm = chat._sm()
    async with sm() as s:
        s.add(ChatSession(
            id=session_id, tenant_id="t1", language="en", status="active",
            extra_data={}, mode=mode,
        ))
        await s.commit()


def _cross_tenant_records(caplog):
    return [r for r in caplog.records if getattr(r, "event", None) == "cross_tenant_access_denied"]


# get_session ---------------------------------------------------------------


async def test_get_session_cross_tenant_logs_and_404(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    other_sid = "cs_foreign_get_1"
    await _seed_foreign_chat_session(other_sid)

    with caplog.at_level(logging.INFO):
        resp = client.get(f"/chat/sessions/{other_sid}", headers=HEADERS)

    assert resp.status_code == 404
    assert resp.json()["detail"] == "chat session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "chat_session_not_owned"
    assert rec.levelno == logging.WARNING
    assert rec.found is True
    assert rec.owner_tenant_id == "t2"
    assert rec.session_id == other_sid
    assert rec.resource == "chat_session"
    assert rec.resource_id == other_sid


async def test_get_session_nonexistent_logs_and_404(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    missing_sid = "cs_does_not_exist_1"

    with caplog.at_level(logging.INFO):
        resp = client.get(f"/chat/sessions/{missing_sid}", headers=HEADERS)

    assert resp.status_code == 404
    assert resp.json()["detail"] == "chat session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "chat_session_not_found"
    assert rec.levelno == logging.WARNING
    assert rec.found is False
    assert rec.owner_tenant_id is None
    assert rec.session_id == missing_sid


def test_get_session_same_tenant_no_cross_tenant_log(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    sid = _create_session(client)

    with caplog.at_level(logging.INFO):
        resp = client.get(f"/chat/sessions/{sid}", headers=HEADERS)

    assert resp.status_code == 200
    assert not _cross_tenant_records(caplog)


# claim_session ---------------------------------------------------------------


async def test_claim_session_cross_tenant_logs_and_404(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    other_sid = "cs_foreign_claim_1"
    await _seed_foreign_chat_session(other_sid, mode="awaiting_human")

    with caplog.at_level(logging.INFO):
        resp = client.post(
            f"/chat/sessions/{other_sid}/claim",
            json={"agent_id": "a01", "agent_name": "Priya"},
            headers=HEADERS,
        )

    assert resp.status_code == 404
    assert resp.json()["detail"] == "chat session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "chat_session_not_owned"
    assert rec.levelno == logging.WARNING
    assert rec.found is True
    assert rec.owner_tenant_id == "t2"
    assert rec.session_id == other_sid


async def test_claim_session_nonexistent_logs_and_404(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    missing_sid = "cs_does_not_exist_2"

    with caplog.at_level(logging.INFO):
        resp = client.post(
            f"/chat/sessions/{missing_sid}/claim",
            json={"agent_id": "a01", "agent_name": "Priya"},
            headers=HEADERS,
        )

    assert resp.status_code == 404
    assert resp.json()["detail"] == "chat session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "chat_session_not_found"
    assert rec.levelno == logging.WARNING
    assert rec.found is False
    assert rec.owner_tenant_id is None


async def test_claim_session_same_tenant_no_cross_tenant_log(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    sid = "cs_own_claim_1"
    await _seed_own_chat_session(sid, mode="awaiting_human")

    with caplog.at_level(logging.INFO):
        resp = client.post(
            f"/chat/sessions/{sid}/claim",
            json={"agent_id": "a01", "agent_name": "Priya"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json() == {"status": "claimed", "agent_id": "a01"}
    assert not _cross_tenant_records(caplog)


# decline_session ---------------------------------------------------------------


async def test_decline_session_cross_tenant_logs_and_404(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    other_sid = "cs_foreign_decline_1"
    await _seed_foreign_chat_session(other_sid, mode="awaiting_human")

    with caplog.at_level(logging.INFO):
        resp = client.post(f"/chat/sessions/{other_sid}/decline", headers=HEADERS)

    assert resp.status_code == 404
    assert resp.json()["detail"] == "chat session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "chat_session_not_owned"
    assert rec.levelno == logging.WARNING
    assert rec.found is True
    assert rec.owner_tenant_id == "t2"
    assert rec.session_id == other_sid


async def test_decline_session_nonexistent_logs_and_404(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    missing_sid = "cs_does_not_exist_3"

    with caplog.at_level(logging.INFO):
        resp = client.post(f"/chat/sessions/{missing_sid}/decline", headers=HEADERS)

    assert resp.status_code == 404
    assert resp.json()["detail"] == "chat session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "chat_session_not_found"
    assert rec.levelno == logging.WARNING
    assert rec.found is False
    assert rec.owner_tenant_id is None


async def test_decline_session_same_tenant_no_cross_tenant_log(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    sid = "cs_own_decline_1"
    await _seed_own_chat_session(sid, mode="awaiting_human")

    with caplog.at_level(logging.INFO):
        resp = client.post(f"/chat/sessions/{sid}/decline", headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json() == {"status": "declined"}
    assert not _cross_tenant_records(caplog)


# agent_websocket (BO handover WS close) -------------------------------------


async def test_agent_ws_cross_tenant_logs_and_closes(app: FastAPI, caplog) -> None:
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    other_sid = "cs_foreign_agentws_1"
    await _seed_foreign_chat_session(other_sid)

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/chat/sessions/{other_sid}/agent-ws?token=test-token"
            ) as ws:
                ws.receive_text()

    assert exc_info.value.code == 4004
    assert exc_info.value.reason == "session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "agent_ws_session_not_owned"
    assert rec.levelno == logging.WARNING
    assert rec.found is True
    assert rec.owner_tenant_id == "t2"
    assert rec.session_id == other_sid
    assert rec.resource == "chat_session"
    assert rec.resource_id == other_sid


async def test_agent_ws_nonexistent_session_logs_and_closes(app: FastAPI, caplog) -> None:
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    missing_sid = "cs_does_not_exist_4"

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/chat/sessions/{missing_sid}/agent-ws?token=test-token"
            ) as ws:
                ws.receive_text()

    assert exc_info.value.code == 4004
    assert exc_info.value.reason == "session not found"

    records = _cross_tenant_records(caplog)
    assert len(records) == 1, records
    rec = records[0]
    assert rec.reason == "agent_ws_session_not_found"
    assert rec.levelno == logging.WARNING
    assert rec.found is False
    assert rec.owner_tenant_id is None
    assert rec.session_id == missing_sid


def test_agent_ws_same_tenant_no_cross_tenant_log(app: FastAPI, caplog) -> None:
    """A same-tenant connection that passes the ownership check (but still
    fails the wrong-mode check further below, out of scope for this change)
    must not produce a cross_tenant_access_denied record."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    sid = _create_session(client)

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/chat/sessions/{sid}/agent-ws?token=test-token"
            ) as ws:
                ws.receive_text()

    assert not _cross_tenant_records(caplog)
