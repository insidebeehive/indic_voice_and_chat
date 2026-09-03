"""tests/unit/test_crm_kb_routes.py"""
from __future__ import annotations

import io
import logging

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import crm_kb
from src.api.deps import get_db_session
from src.auth.audit import reset_suppression_state
from src.auth.middleware import set_admin_tokens
from src.models.crm import Crm
from src.models.database import Base
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import HybridRetriever, RetrievalConfig

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest.fixture(autouse=True)
def _reset_audit_suppression_state():
    reset_suppression_state()
    yield
    reset_suppression_state()


@pytest_asyncio.fixture
async def client(tmp_faiss_index: str):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Crm(id="betstudio", name="BetStudio", base_url="https://x"))
        # A second CRM so cross-CRM-scope tests can reach a real, existing
        # crm_id whose _require_crm check passes and whose retriever is also
        # configured (delete_crm_document resolves the retriever BEFORE the
        # row-ownership check, so it must not 503 first).
        s.add(Crm(id="othercrm", name="OtherCrm", base_url="https://y"))
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
            return retriever if crm_id in ("betstudio", "othercrm") else None

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


async def test_delete_crm_document_cross_crm_404_logs_denied(client: AsyncClient, caplog) -> None:
    """An admin hitting a document that belongs to a DIFFERENT (but real) CRM
    still gets the pre-existing 404, but an admin_scope_denied INFO record is
    emitted and the document itself is left untouched."""
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        headers=ADMIN_HEADERS,
    )
    doc_id = resp.json()["document_id"]

    with caplog.at_level(logging.INFO):
        deleted = await client.delete(
            f"/crms/othercrm/kb/documents/{doc_id}", headers=ADMIN_HEADERS)
    assert deleted.status_code == 404
    assert deleted.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "admin_scope_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.INFO
    assert denied[0].reason == "crm_kb_document_not_in_crm"
    assert denied[0].found is True

    still_there = await client.get("/crms/betstudio/kb/documents", headers=ADMIN_HEADERS)
    assert any(d["id"] == doc_id for d in still_there.json()["documents"])


async def test_delete_crm_document_unknown_id_404_logs_denied(client: AsyncClient, caplog) -> None:
    with caplog.at_level(logging.INFO):
        resp = await client.delete(
            "/crms/betstudio/kb/documents/missing", headers=ADMIN_HEADERS)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "admin_scope_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.INFO
    assert denied[0].reason == "crm_kb_document_not_found"
    assert denied[0].found is False


async def test_delete_crm_document_same_crm_logs_no_denial(client: AsyncClient, caplog) -> None:
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        headers=ADMIN_HEADERS,
    )
    doc_id = resp.json()["document_id"]

    with caplog.at_level(logging.INFO):
        deleted = await client.delete(
            f"/crms/betstudio/kb/documents/{doc_id}", headers=ADMIN_HEADERS)
    assert deleted.status_code == 200

    denied = [r for r in caplog.records if getattr(r, "event", None) == "admin_scope_denied"]
    assert denied == []


async def test_download_crm_document_cross_crm_404_logs_denied(client: AsyncClient, caplog) -> None:
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        headers=ADMIN_HEADERS,
    )
    doc_id = resp.json()["document_id"]

    with caplog.at_level(logging.INFO):
        dl = await client.get(
            f"/crms/othercrm/kb/documents/{doc_id}/download", headers=ADMIN_HEADERS)
    assert dl.status_code == 404
    assert dl.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "admin_scope_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.INFO
    assert denied[0].reason == "crm_kb_document_not_in_crm"
    assert denied[0].found is True


async def test_download_crm_document_unknown_id_404_logs_denied(client: AsyncClient, caplog) -> None:
    with caplog.at_level(logging.INFO):
        resp = await client.get(
            "/crms/betstudio/kb/documents/missing/download", headers=ADMIN_HEADERS)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "admin_scope_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.INFO
    assert denied[0].reason == "crm_kb_document_not_found"
    assert denied[0].found is False


async def test_download_crm_document_same_crm_logs_no_denial(client: AsyncClient, caplog) -> None:
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        headers=ADMIN_HEADERS,
    )
    doc_id = resp.json()["document_id"]

    with caplog.at_level(logging.INFO):
        dl = await client.get(
            f"/crms/betstudio/kb/documents/{doc_id}/download", headers=ADMIN_HEADERS)
    assert dl.status_code == 200

    denied = [r for r in caplog.records if getattr(r, "event", None) == "admin_scope_denied"]
    assert denied == []


async def test_unknown_crm_id_404s(client: AsyncClient) -> None:
    resp = await client.get("/crms/does-not-exist/kb/documents", headers=ADMIN_HEADERS)
    assert resp.status_code == 404


@pytest.mark.parametrize("reserved_id", ["crm_kb_betstudio_hijack", "global_kb_hijack"])
async def test_ingest_rejects_reserved_namespace_document_id(
    client: AsyncClient, reserved_id: str
) -> None:
    """A caller-supplied document_id in the seeder/purge-script's protected
    namespace (crm_kb_ / global_kb_) must be rejected, not silently accepted
    -- accepting it would let an admin upload collide with an id the purge
    script or the seeder's auto-reconcile treats as safe to prune."""
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        data={"document_id": reserved_id},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 400
    assert reserved_id in resp.json()["detail"]

    # Non-regression: the id must never have been persisted.
    listed = await client.get("/crms/betstudio/kb/documents", headers=ADMIN_HEADERS)
    assert not any(d["id"] == reserved_id for d in listed.json()["documents"])


async def test_ingest_without_document_id_still_succeeds(client: AsyncClient) -> None:
    """Non-regression: omitting document_id (the common case) still works."""
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["document_id"]


async def test_ingest_with_normal_document_id_still_succeeds(client: AsyncClient) -> None:
    """Non-regression: a normal caller-chosen document_id (outside the
    reserved namespace) still works."""
    resp = await client.post(
        "/crms/betstudio/kb/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB unlimited"), "text/markdown")},
        data={"document_id": "my-own-doc-id"},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["document_id"] == "my-own-doc-id"


async def test_requires_admin(client: AsyncClient) -> None:
    resp = await client.get("/crms/betstudio/kb/documents")
    assert resp.status_code == 401
