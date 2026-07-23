"""Route-level tests for /api/v1/knowledge/* endpoints."""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
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


HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture
async def app(tmp_faiss_index: str) -> FastAPI:
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8),
    )
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    knowledge.set_retriever(retriever, ChunkConfig(chunk_size=10, chunk_overlap=2, strategy="recursive"))
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"),
        plaintext_tokens=["test-token"],
    )

    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[get_db_session] = _session_override
    yield app
    knowledge.set_retriever(None, None)  # type: ignore[arg-type]
    set_tenant_resolver(None)
    await engine.dispose()


def test_ingest_a_markdown_document(app: FastAPI) -> None:
    client = TestClient(app)
    text = "# Plans\n\nPlan B has 500GB unlimited.\n\nPlan A has 100GB."
    resp = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(text.encode("utf-8")), "text/markdown")},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "plans.md"
    assert body["chunks_indexed"] >= 1


def test_ingest_rejects_empty_upload(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest",
        files={"file": ("empty.md", io.BytesIO(b""), "text/markdown")},
        headers=HEADERS,
    )
    assert resp.status_code == 400


def test_list_documents_after_ingest(app: FastAPI) -> None:
    client = TestClient(app)
    client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        headers=HEADERS,
    )
    resp = client.get("/knowledge/documents", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert any(d["filename"] == "plans.md" for d in body["documents"])


def test_query_returns_hits(app: FastAPI) -> None:
    client = TestClient(app)
    client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        headers=HEADERS,
    )
    resp = client.post(
        "/knowledge/query",
        json={"query": "Plan B unlimited", "top_k": 3},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert "Plan B" in body["hits"][0]["content"]


def test_delete_document(app: FastAPI) -> None:
    client = TestClient(app)
    ingested = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        headers=HEADERS,
    ).json()
    doc_id = ingested["document_id"]

    resp = client.delete(f"/knowledge/documents/{doc_id}", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["chunks_removed"] >= 1

    after = client.get("/knowledge/documents", headers=HEADERS).json()
    assert after["total"] == 0


def test_delete_unknown_document_404(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.delete("/knowledge/documents/missing", headers=HEADERS)
    assert resp.status_code == 404


def test_stats_reflects_ingest(app: FastAPI) -> None:
    client = TestClient(app)
    before = client.get("/knowledge/stats", headers=HEADERS).json()
    assert before["document_count"] == 0

    client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        headers=HEADERS,
    )
    after = client.get("/knowledge/stats", headers=HEADERS).json()
    assert after["document_count"] == 1
    assert after["chunk_count"] >= 1


def test_missing_auth_returns_401(app: FastAPI) -> None:
    client = TestClient(app)
    assert client.get("/knowledge/documents").status_code == 401


def test_query_when_retriever_unset_returns_503(tmp_faiss_index: str) -> None:
    knowledge.set_retriever(None, None)  # type: ignore[arg-type]
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"),
        plaintext_tokens=["test-token"],
    )
    app = FastAPI()
    app.include_router(knowledge.router)
    client = TestClient(app)
    resp = client.post("/knowledge/query", json={"query": "x"}, headers=HEADERS)
    assert resp.status_code == 503
    set_tenant_resolver(None)


def test_query_mixes_tenant_and_linked_crm_docs(app: FastAPI) -> None:
    from src.api import knowledge
    from src.providers.vector_store.faiss_store import FAISSAdapter
    from src.rag.embeddings import HashEmbedder, IdentityReranker
    from src.rag.retriever import HybridRetriever, RetrievalConfig

    crm_store = FAISSAdapter({"embedding_dim": 64, "index_path": "/tmp/crm_kb_test_index"})
    crm_retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64), vector_store=crm_store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8),
    )
    import asyncio
    asyncio.get_event_loop().run_until_complete(crm_retriever.index([
        __import__("src.interfaces.vector_store", fromlist=["Document"]).Document(
            id="crm-chunk-1", content="CRM-level policy: refunds within 30 days.",
            metadata={}),
    ]))

    class _FakeRegistry:
        def get(self, crm_id: str):
            return crm_retriever if crm_id == "betstudio" else None

    knowledge.set_crm_retrievers(_FakeRegistry())
    try:
        from src.auth import register_tenant_for_test
        from src.config_tenant import TenantSettings
        register_tenant_for_test(
            TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio"),
            plaintext_tokens=["test-token"],
        )
        client = TestClient(app)
        resp = client.post(
            "/knowledge/query",
            json={"query": "refund policy", "top_k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert any("refund" in h["content"].lower() for h in body["hits"])
    finally:
        knowledge.set_crm_retrievers(None)
