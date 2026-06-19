"""Phase 5 integration: per-tenant knowledge isolation + Hindi/English round-trip.

Two tenants get their own retriever (separate FAISS index, as the runtime
registry wires in prod). A doc ingested by one tenant must never surface in the
other's queries, and a Devanagari doc must round-trip.
"""

from __future__ import annotations

import io

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import knowledge
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import Tenant
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.ingestion import ChunkConfig
from src.rag.retriever import HybridRetriever, RetrievalConfig

T1 = {"Authorization": "Bearer tok1"}
T2 = {"Authorization": "Bearer tok2"}


def _retriever(path: str) -> HybridRetriever:
    return HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": path}),
        reranker=IdentityReranker(),
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8,
                               reranking=True, similarity_threshold=0.0))


@pytest_asyncio.fixture
async def ctx(tmp_path):
    retrievers = {"t1": _retriever(str(tmp_path / "t1")), "t2": _retriever(str(tmp_path / "t2"))}

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        s.add(Tenant(id="t2", slug="t2", name="Globex"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(TenantSettings(id="t1", slug="t1", name="Acme"), plaintext_tokens=["tok1"])
    register_tenant_for_test(TenantSettings(id="t2", slug="t2", name="Globex"), plaintext_tokens=["tok2"])
    knowledge.set_retriever_factory(
        lambda tenant: retrievers[tenant.id],
        ChunkConfig(chunk_size=40, chunk_overlap=5, strategy="recursive"))

    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    knowledge.set_retriever(None, None)  # type: ignore[arg-type]
    set_tenant_resolver(None)
    await engine.dispose()


async def _ingest(client, headers, filename, body):
    return await client.post(
        "/knowledge/ingest",
        files={"file": (filename, io.BytesIO(body), "text/markdown")}, headers=headers)


async def test_knowledge_is_isolated_per_tenant(ctx) -> None:
    client = ctx
    assert (await _ingest(client, T1, "plans.md", b"Acme Plan B has 500GB unlimited data.")).status_code == 200

    # Tenant 1 sees its own doc...
    r1 = await client.post("/knowledge/query", json={"query": "Plan B unlimited", "top_k": 3}, headers=T1)
    assert r1.json()["total"] >= 1

    # ...tenant 2's knowledge base is empty (no cross-tenant leak).
    r2 = await client.post("/knowledge/query", json={"query": "Plan B unlimited", "top_k": 3}, headers=T2)
    assert r2.json()["total"] == 0
    assert (await client.get("/knowledge/documents", headers=T2)).json()["total"] == 0
    assert (await client.get("/knowledge/documents", headers=T1)).json()["total"] == 1


async def test_devanagari_document_round_trips(ctx) -> None:
    client = ctx
    hi = "रिफंड नीति: सात दिन के भीतर रिफंड मिलेगा।".encode("utf-8")
    assert (await _ingest(client, T2, "refund_hi.md", hi)).status_code == 200
    resp = await client.post("/knowledge/query", json={"query": "रिफंड नीति", "top_k": 3}, headers=T2)
    body = resp.json()
    assert body["total"] >= 1
    assert "रिफंड" in body["hits"][0]["content"]
