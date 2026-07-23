"""tests/unit/test_crm_kb_routes.py"""
from __future__ import annotations

import io

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import crm_kb
from src.api.deps import get_db_session
from src.auth.middleware import set_admin_tokens
from src.models.crm import Crm
from src.models.database import Base
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import HybridRetriever, RetrievalConfig

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def client(tmp_faiss_index: str):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Crm(id="betstudio", name="BetStudio", base_url="https://x"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64), vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8),
    )

    class _FakeRegistry:
        def get(self, crm_id: str):
            return retriever if crm_id == "betstudio" else None

    set_admin_tokens(["admin-token"])
    crm_kb.set_crm_retrievers(_FakeRegistry())
    app = FastAPI()
    app.include_router(crm_kb.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_admin_tokens([])
    crm_kb.set_crm_retrievers(None)
    await engine.dispose()


async def test_ingest_list_delete_round_trip(client: AsyncClient) -> None:
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    doc_id = resp.json()["document_id"]

    listed = await client.get("/crms/betstudio/kb/documents", headers=ADMIN_HEADERS)
    assert listed.status_code == 200
    assert any(d["id"] == doc_id for d in listed.json()["documents"])

    deleted = await client.delete(f"/crms/betstudio/kb/documents/{doc_id}", headers=ADMIN_HEADERS)
    assert deleted.status_code == 200

    after = await client.get("/crms/betstudio/kb/documents", headers=ADMIN_HEADERS)
    assert after.json()["total"] == 0


async def test_unknown_crm_id_404s(client: AsyncClient) -> None:
    resp = await client.get("/crms/does-not-exist/kb/documents", headers=ADMIN_HEADERS)
    assert resp.status_code == 404


async def test_requires_admin(client: AsyncClient) -> None:
    resp = await client.get("/crms/betstudio/kb/documents")
    assert resp.status_code == 401
