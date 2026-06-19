"""End-to-end ChatBot test.

Drives the full pipeline:
1. Ingest several markdown documents through the knowledge admin API.
2. Open a websocket session and ask multiple follow-up questions.
3. Assert the LLM saw retrieved context, the hallucination guard fires
   when the LLM cites bogus sources, the no-retrieval branch returns the
   fallback message, and Redis history persists across turns.

The LLM is a fake that responds with a different canned answer per call.
"""

from __future__ import annotations

import io
import json
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.api import chat, knowledge
from src.api.deps import get_db_session
from src.auth import TenantContext, register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import Tenant

HEADERS = {"Authorization": "Bearer test-token"}
from src.dialogue.context import SessionStore
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.ingestion import ChunkConfig
from src.rag.retriever import HybridRetriever, RetrievalConfig


class ScriptedLLM(ILLMProvider):
    """Pops the next pre-canned response per call."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.last_messages: list[list[LLMMessage]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.last_messages.append(list(messages))
        if not self._responses:
            payload = {"response_text": "ok", "language": "en", "action": "none"}
        else:
            payload = self._responses.pop(0)
        return LLMResult(text=json.dumps(payload), finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


@pytest.fixture
async def app(tmp_faiss_index: str, fake_redis):
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=3,
            oversample_k=10,
            reranking=True,
            similarity_threshold=0.0,
        ),
    )
    knowledge.set_retriever(retriever, ChunkConfig(chunk_size=10, chunk_overlap=2, strategy="recursive"))

    session_store = SessionStore(fake_redis, ttl_seconds=300)
    llm = ScriptedLLM(responses=[
        # Turn 1: clean answer with valid citation
        {
            "response_text": "Plan B has 500GB unlimited data.",
            "language": "en",
            "sources_used": ["plans.md:0"],
            "confidence": "high",
            "action": "none",
            "suggested_followups": ["What about Plan A?"],
        },
        # Turn 2: LLM invents a citation; guard should strip + downgrade
        {
            "response_text": "Plan B is Rs 699/month and supports 6G.",
            "language": "en",
            "sources_used": ["plans.md:0", "wireless.md:99"],  # wireless.md never ingested
            "confidence": "high",
            "action": "none",
        },
        # Turn 3: question that has no good retrieval -> fallback fires
        {
            "response_text": "Yes, the answer is 42.",
            "language": "en",
            "sources_used": [],
            "confidence": "high",
            "action": "none",
        },
    ])

    async def factory(tenant: TenantContext, session_id: str) -> ChatBotAgent:
        return ChatBotAgent(
            session=AgentSession(session_id=session_id),
            llm=llm,
            retriever=retriever,
            company_name=tenant.settings.name,
            language_default="en",
            store=session_store,
        )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme Telecom"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    chat.set_chatbot_factory(factory)
    chat.set_chat_sessionmaker(sm)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme Telecom"),
        plaintext_tokens=["test-token"],
    )
    a = FastAPI()
    a.include_router(knowledge.router)
    a.include_router(chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    yield a, llm, session_store, retriever
    chat.set_chatbot_factory(None)
    chat.set_chat_sessionmaker(None)
    knowledge.set_retriever(None, None)  # type: ignore[arg-type]
    set_tenant_resolver(None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_e2e_ingest_then_chat(app) -> None:
    fastapi_app, llm, session_store, retriever = app
    client = TestClient(fastapi_app)

    # 1. Ingest documents — at least 3 so BM25 IDF behaves sensibly.
    docs = [
        ("plans.md", b"Plan B has 500GB unlimited data per month."),
        ("pricing.md", b"Plan B costs Rs 699 per month, annual is Rs 599."),
        ("cookbook.md", b"Recipes for biryani and other dishes."),
    ]
    for filename, body in docs:
        resp = client.post(
            "/knowledge/ingest",
            files={"file": (filename, io.BytesIO(body), "text/markdown")},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    stats = client.get("/knowledge/stats", headers=HEADERS).json()
    assert stats["document_count"] == 3
    assert stats["chunk_count"] >= 3

    # 2. Turn 1: clean, valid answer
    r1 = client.post(
        "/chat/message",
        json={"session_id": "e2e-1", "message": "Tell me about Plan B"},
        headers=HEADERS,
    ).json()
    assert r1["response_text"] == "Plan B has 500GB unlimited data."
    assert r1["confidence"] == "high"
    assert "plans.md:0" in r1["sources_used"]

    # The LLM saw retrieved context in its system prompt.
    sys_msg = llm.last_messages[0][0]
    assert sys_msg.role == "system"
    assert "Plan B" in sys_msg.content

    # 3. Turn 2: invented citation -> guard kicks in
    r2 = client.post(
        "/chat/message",
        json={"session_id": "e2e-1", "message": "How much and what speed?"},
        headers=HEADERS,
    ).json()
    # The bogus source was stripped.
    assert "wireless.md:99" not in r2["sources_used"]
    # And confidence dropped because the LLM lied.
    assert r2["confidence"] == "low"

    # 4. Turn 3: ask something completely unrelated.
    r3 = client.post(
        "/chat/message",
        json={"session_id": "e2e-1", "message": "What's the meaning of life?"},
        headers=HEADERS,
    ).json()
    assert "42" not in r3["response_text"]

    # 5. History persisted across all turns
    hist = client.get("/chat/history/e2e-1", headers=HEADERS).json()
    assert len(hist["history"]) == 6  # 3 user + 3 agent
    user_msgs = [h["content"] for h in hist["history"] if h["role"] == "user"]
    assert user_msgs == [
        "Tell me about Plan B",
        "How much and what speed?",
        "What's the meaning of life?",
    ]


@pytest.mark.asyncio
async def test_e2e_delete_document_removes_from_retrieval(app) -> None:
    fastapi_app, _, _, retriever = app
    client = TestClient(fastapi_app)

    # Ingest
    ingested = client.post(
        "/knowledge/ingest",
        files={"file": ("delete-me.md", io.BytesIO(b"Plan X has unique-token-abc"), "text/markdown")},
        headers=HEADERS,
    ).json()
    doc_id = ingested["document_id"]

    # Verify retrievable
    pre = client.post("/knowledge/query", json={"query": "unique-token-abc", "top_k": 5}, headers=HEADERS).json()
    assert pre["total"] >= 1

    # Delete
    client.delete(f"/knowledge/documents/{doc_id}", headers=HEADERS)

    # No longer retrievable
    post = client.post("/knowledge/query", json={"query": "unique-token-abc", "top_k": 5}, headers=HEADERS).json()
    assert all("unique-token-abc" not in h["content"] for h in post["hits"])
