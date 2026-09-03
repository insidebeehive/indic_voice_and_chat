"""Route-level tests for /api/v1/knowledge/* endpoints."""

from __future__ import annotations

import io
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import knowledge
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.audit import reset_suppression_state
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import Tenant
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.ingestion import ChunkConfig
from src.rag.retriever import HybridRetriever, RetrievalConfig


HEADERS = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _reset_audit_suppression_state():
    reset_suppression_state()
    yield
    reset_suppression_state()


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


@pytest.mark.parametrize("reserved_id", ["crm_kb_betstudio_hijack", "global_kb_hijack"])
def test_ingest_rejects_reserved_namespace_document_id(app: FastAPI, reserved_id: str) -> None:
    """A caller-supplied document_id in the seeder/purge-script's protected
    namespace (crm_kb_ / global_kb_) must be rejected, not silently accepted
    -- accepting it would let a tenant upload collide with an id the purge
    script or the seeder's auto-reconcile treats as safe to prune."""
    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        data={"document_id": reserved_id},
        headers=HEADERS,
    )
    assert resp.status_code == 400
    assert reserved_id in resp.json()["detail"]

    listed = client.get("/knowledge/documents", headers=HEADERS).json()
    assert not any(d["id"] == reserved_id for d in listed["documents"])


def test_ingest_with_normal_document_id_still_succeeds(app: FastAPI) -> None:
    """Non-regression: a normal caller-chosen document_id still works."""
    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        data={"document_id": "my-own-doc-id"},
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["document_id"] == "my-own-doc-id"


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


def test_delete_document_cross_tenant_404_logs_denied(app: FastAPI, caplog) -> None:
    """A tenant deleting another tenant's document gets the same 404 as
    before, but a cross_tenant_access_denied WARNING record is emitted and
    the document itself is left untouched (not actually deleted)."""
    client = TestClient(app)
    ingested = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        headers=HEADERS,
    ).json()
    doc_id = ingested["document_id"]

    register_tenant_for_test(
        TenantSettings(id="t2", slug="t2", name="T2"), plaintext_tokens=["t2-token"],
    )
    with caplog.at_level(logging.INFO):
        resp = client.delete(
            f"/knowledge/documents/{doc_id}", headers={"Authorization": "Bearer t2-token"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.WARNING
    assert denied[0].reason == "kb_document_not_owned"
    assert denied[0].found is True

    still_there = client.get("/knowledge/documents", headers=HEADERS).json()
    assert any(d["id"] == doc_id for d in still_there["documents"])


def test_delete_unknown_document_404_logs_denied(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        resp = client.delete("/knowledge/documents/missing", headers=HEADERS)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.WARNING
    assert denied[0].reason == "kb_document_not_found"
    assert denied[0].found is False


def test_delete_document_same_tenant_logs_no_denial(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    ingested = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        headers=HEADERS,
    ).json()
    doc_id = ingested["document_id"]

    with caplog.at_level(logging.INFO):
        resp = client.delete(f"/knowledge/documents/{doc_id}", headers=HEADERS)
    assert resp.status_code == 200

    denied = [r for r in caplog.records if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert denied == []


def test_download_document_cross_tenant_404_logs_denied(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    ingested = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        headers=HEADERS,
    ).json()
    doc_id = ingested["document_id"]

    register_tenant_for_test(
        TenantSettings(id="t2", slug="t2", name="T2"), plaintext_tokens=["t2-token"],
    )
    with caplog.at_level(logging.INFO):
        resp = client.get(
            f"/knowledge/documents/{doc_id}/download",
            headers={"Authorization": "Bearer t2-token"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.WARNING
    assert denied[0].reason == "kb_document_not_owned"
    assert denied[0].found is True


def test_download_document_unknown_id_404_logs_denied(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        resp = client.get("/knowledge/documents/missing/download", headers=HEADERS)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "document not found"

    denied = [r for r in caplog.records if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, caplog.records
    assert denied[0].levelno == logging.WARNING
    assert denied[0].reason == "kb_document_not_found"
    assert denied[0].found is False


def test_download_document_same_tenant_logs_no_denial(app: FastAPI, caplog) -> None:
    client = TestClient(app)
    ingested = client.post(
        "/knowledge/ingest",
        files={"file": ("plans.md", io.BytesIO(b"Plan B has 500GB"), "text/markdown")},
        headers=HEADERS,
    ).json()
    doc_id = ingested["document_id"]

    with caplog.at_level(logging.INFO):
        resp = client.get(f"/knowledge/documents/{doc_id}/download", headers=HEADERS)
    assert resp.status_code == 200

    denied = [r for r in caplog.records if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert denied == []


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


def test_ingest_layout_valid_name(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest-layout", json={"layout": "layout-1"}, headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    docs_out = body["documents"]
    assert len(docs_out) == 1
    assert docs_out[0]["document_id"] == "layout_layout-1"
    assert docs_out[0]["filename"] == "layout-1.md"
    assert docs_out[0]["chunks_indexed"] >= 1

    docs = client.get("/knowledge/documents", headers=HEADERS).json()["documents"]
    assert any(d["id"] == "layout_layout-1" for d in docs)


@pytest.mark.parametrize("bad_layout", ["layout-99", "../../etc/passwd"])
def test_ingest_layout_rejects_unknown_name(app: FastAPI, bad_layout: str) -> None:
    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest-layout", json={"layout": bad_layout}, headers=HEADERS,
    )
    assert resp.status_code == 400
    docs = client.get("/knowledge/documents", headers=HEADERS).json()["documents"]
    assert docs == []


def test_ingest_layout_reingest_is_idempotent(app: FastAPI) -> None:
    client = TestClient(app)
    first = client.post(
        "/knowledge/ingest-layout", json={"layout": "layout-1"}, headers=HEADERS,
    )
    assert first.status_code == 200
    second = client.post(
        "/knowledge/ingest-layout", json={"layout": "layout-1"}, headers=HEADERS,
    )
    assert second.status_code == 200
    docs = client.get("/knowledge/documents", headers=HEADERS).json()["documents"]
    matching = [d for d in docs if d["id"] == "layout_layout-1"]
    assert len(matching) == 1


def test_ingest_layout_missing_auth_returns_401(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/knowledge/ingest-layout", json={"layout": "layout-1"})
    assert resp.status_code == 401


def test_ingest_layout_paired_module_ingests_both_files(app: FastAPI) -> None:
    """A product-module key (casino) ingests its backend doc AND its UI-help
    counterpart as two separate KBDocument rows — never merged into one, since
    voicebot doc-priority matching keys off each file's own filename."""
    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest-layout", json={"layout": "casino"}, headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    docs_out = resp.json()["documents"]
    assert len(docs_out) == 2
    assert [d["filename"] for d in docs_out] == ["06-casino-games.md", "ui-05-casino.md"]
    assert [d["document_id"] for d in docs_out] == [
        "layout_casino_06-casino-games", "layout_casino_ui-05-casino",
    ]
    assert len({d["document_id"] for d in docs_out}) == 2
    assert all(d["chunks_indexed"] >= 1 for d in docs_out)

    listed = client.get("/knowledge/documents", headers=HEADERS).json()
    assert listed["total"] == 2
    assert {d["id"] for d in listed["documents"]} == {
        "layout_casino_06-casino-games", "layout_casino_ui-05-casino",
    }


@pytest.mark.parametrize("missing_index", [0, 1])
def test_ingest_layout_paired_key_missing_file_ingests_nothing(
    app: FastAPI, tmp_path, monkeypatch, missing_index: int,
) -> None:
    """A paired key must be all-or-nothing: if ANY of its files is missing on
    disk, the request 404s with ZERO writes. Previously the existence check ran
    inside the ingest loop, so a missing 2nd file left the 1st already
    committed (KBDocument row + chunks) with no rollback and no signal."""
    present = tmp_path / "06-casino-games.md"
    present.write_text("# Casino\n\nSlots, live tables and jackpots.", encoding="utf-8")
    absent = tmp_path / "ui-05-casino.md"  # deliberately never created
    pair = [present, absent] if missing_index == 1 else [absent, present]
    monkeypatch.setitem(knowledge._INGESTIBLE_DOCS, "casino", list(pair))

    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest-layout", json={"layout": "casino"}, headers=HEADERS,
    )
    assert resp.status_code == 404, resp.text
    assert "ui-05-casino.md" in resp.json()["detail"]

    assert client.get("/knowledge/documents", headers=HEADERS).json()["documents"] == []
    stats = client.get("/knowledge/stats", headers=HEADERS).json()
    assert stats["document_count"] == 0
    assert stats["chunk_count"] == 0


def test_every_ingestible_doc_exists_on_disk() -> None:
    """_INGESTIBLE_DOCS is the endpoint's whole path-traversal guard AND its
    file lookup — a stale entry means a 404 at ingest time, not a test failure
    at build time, so pin it here. Paths are cwd-relative exactly as the route
    resolves them."""
    missing = [
        f"{key} -> {path}"
        for key, paths in knowledge._INGESTIBLE_DOCS.items()
        for path in paths
        if not path.is_file()
    ]
    assert missing == [], f"missing bundled KB docs: {missing}"


@pytest.mark.parametrize("stem", ["06-casino-games", "ui-05-casino", "README"])
def test_ingest_layout_rejects_raw_filename_stem(app: FastAPI, stem: str) -> None:
    """Only the registered keys are ingestible — a real file's stem under
    data/kb/modules/ is NOT a key, so it must still 400 rather than resolve to
    a path. (README.md is reference-only and must never be ingestible.)"""
    client = TestClient(app)
    resp = client.post(
        "/knowledge/ingest-layout", json={"layout": stem}, headers=HEADERS,
    )
    assert resp.status_code == 400
    assert client.get("/knowledge/documents", headers=HEADERS).json()["documents"] == []


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
