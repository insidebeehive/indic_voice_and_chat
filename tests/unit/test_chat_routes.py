"""Route-level tests for /api/v1/chat/* endpoints."""

from __future__ import annotations

import json
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


class _FakeLLM(ILLMProvider):
    def __init__(self, payload: dict) -> None:
        self._json = json.dumps(payload)

    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(text=self._json, finish_reason="stop")

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

    async def factory(tenant: TenantContext, session_id: str) -> ChatBotAgent:
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


def test_websocket_unknown_session_closes(app: FastAPI) -> None:
    client = TestClient(app)
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/chat/ws/cs_doesnotexist") as ws:
            ws.receive_text()


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

    async def factory(tenant: TenantContext, session_id: str) -> ChatBotAgent:
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
