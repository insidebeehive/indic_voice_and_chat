"""Knowledge base / RAG endpoints (PRD §7.5).

A retriever instance is injected via ``set_retriever`` at app startup so the
endpoints stay testable in isolation. The retriever pairs a vector store
with an embedder and (optionally) a reranker — see ``src.rag.retriever``.

Endpoints:
- POST   /knowledge/ingest         multipart upload, parses + chunks + indexes
- POST   /knowledge/ingest-layout  ingest a bundled frontend-layout KB doc (server-side file read)
- GET    /knowledge/documents      list ingested documents (Phase 5+: paginate)
- DELETE /knowledge/documents/{id} remove an ingested document
- POST   /knowledge/query          retrieve top-k chunks for a query (debug)
- GET    /knowledge/stats          basic counts
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.interfaces.vector_store import Document
from src.models.benchmark import KBDocument
from src.rag.context_builder import search_combined
from src.rag.ingestion import ChunkConfig, detect_language, get_chunker, parse_document
from src.rag.retriever import HybridRetriever

log = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# --- DI -----------------------------------------------------------------


# A factory that returns the TENANT retriever (for ingest + tenant-specific search).
# The tenant's linked CRM's retriever (if any) is resolved separately per-request
# from the per-CRM registry, via tenant.settings.crm_id.
RetrieverFactory = Callable[[TenantContext], HybridRetriever]
_retriever_factory: Optional[RetrieverFactory] = None
_crm_retrievers: "object | None" = None  # src.bootstrap.PerCrmRetrieverRegistry
_chunk_config: ChunkConfig = ChunkConfig()


def set_retriever_factory(
    factory: Optional[RetrieverFactory], chunk_config: Optional[ChunkConfig] = None,
) -> None:
    global _retriever_factory, _chunk_config
    _retriever_factory = factory
    if chunk_config is not None:
        _chunk_config = chunk_config


def set_crm_retrievers(registry) -> None:
    """Inject the per-CRM retriever registry (typed loosely to avoid an
    import cycle with src.bootstrap)."""
    global _crm_retrievers
    _crm_retrievers = registry


def set_retriever(
    retriever: Optional[HybridRetriever], chunk_config: Optional[ChunkConfig] = None,
) -> None:
    """Convenience: use ONE retriever for every tenant (tests). ``None`` clears."""
    set_retriever_factory(
        (lambda _tenant: retriever) if retriever is not None else None, chunk_config)


def _retriever_for(tenant: TenantContext) -> HybridRetriever:
    if _retriever_factory is None:
        raise HTTPException(
            status_code=503,
            detail="knowledge base not initialized; set_retriever() not called",
        )
    return _retriever_factory(tenant)


def _crm_retriever_for(tenant: TenantContext) -> Optional[HybridRetriever]:
    if _crm_retrievers is None:
        return None
    crm_id = getattr(tenant.settings, "crm_id", None)
    if not crm_id:
        return None
    return _crm_retrievers.get(crm_id)


def _active_retrievers(tenant: TenantContext) -> list[HybridRetriever]:
    """The tenant's linked CRM's retriever + the tenant's own retriever, skipping None."""
    return [r for r in [_crm_retriever_for(tenant), _retriever_for(tenant)] if r is not None]


# --- Schemas ------------------------------------------------------------


class DocumentInfo(BaseModel):
    id: str
    filename: str
    language: Optional[str] = None
    chunk_count: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    language: Optional[str]


class IngestedFileInfo(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    language: Optional[str] = None


class IngestLayoutResponse(BaseModel):
    """Response for POST /ingest-layout — one entry per file ingested for the
    requested key. Single-file keys (existing frontend layouts) return a
    one-item list; paired vertical-module keys (casino/sports/matka) return
    two."""
    documents: list[IngestedFileInfo]


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict] = None


class QueryHit(BaseModel):
    chunk_id: str
    content: str
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: dict


class QueryResponse(BaseModel):
    hits: list[QueryHit]
    total: int


class StatsResponse(BaseModel):
    document_count: int
    chunk_count: int


# Fixed allow-list mapping an ingest key to the bundled doc(s) it pulls in.
# This dict lookup IS the path-traversal guard: request values only ever index
# into this map, never build a filesystem path directly — never change that.
#
# - Existing frontend-layout keys (data/kb/layouts/) — one file each, unchanged
#   from the old _ALLOWED_LAYOUTS behavior. Deliberately excludes
#   operator-to-layout.md and README.md (reference-only, never ingestible).
# - Product-module keys (data/kb/modules/) — each ingests a *pair* of files
#   (backend doc + its UI-help counterpart) as two separate KBDocument rows,
#   never merged, since voicebot doc-priority matching keys off each file's
#   own original filename (see _VOICE_KB_PRIORITY in src/rag/context_builder.py).
_INGESTIBLE_DOCS: dict[str, list[Path]] = {
    "layout-1": [Path("data/kb/layouts/layout-1.md")],
    "layout-2": [Path("data/kb/layouts/layout-2.md")],
    "layout-3": [Path("data/kb/layouts/layout-3.md")],
    "layout-4": [Path("data/kb/layouts/layout-4.md")],
    "layout-5": [Path("data/kb/layouts/layout-5.md")],
    "layout-6": [Path("data/kb/layouts/layout-6.md")],
    "layout-7": [Path("data/kb/layouts/layout-7.md")],
    "layout-8": [Path("data/kb/layouts/layout-8.md")],
    "layout-9": [Path("data/kb/layouts/layout-9.md")],
    "layout-sports": [Path("data/kb/layouts/layout-sports.md")],
    "casino": [
        Path("data/kb/modules/06-casino-games.md"),
        Path("data/kb/modules/ui-05-casino.md"),
    ],
    "sports": [
        Path("data/kb/modules/07-sports-betting.md"),
        Path("data/kb/modules/ui-06-sports.md"),
    ],
    "matka": [
        Path("data/kb/modules/08-matka-lottery-games.md"),
        Path("data/kb/modules/ui-07-matka-lottery.md"),
    ],
}


class IngestLayoutRequest(BaseModel):
    # Field name kept as `layout` for backward compatibility with existing
    # callers, though it now also accepts product-module keys (casino/sports/
    # matka) — see _INGESTIBLE_DOCS.
    layout: str


# --- Routes -------------------------------------------------------------


async def _ingest_text(
    tenant: TenantContext, session: AsyncSession, *,
    filename: str, text: str, document_id: Optional[str] = None,
) -> IngestResponse:
    """Chunk+embed+index `text` into the tenant's own retriever, persist a
    KBDocument row. Shared by the multipart file-upload route and the
    server-side layout-doc ingestion route below."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="document parsed to empty text")
    retriever = _retriever_for(tenant)
    doc_id = document_id or _new_id()
    language = detect_language(text)
    chunker = get_chunker(_chunk_config)
    raw_chunks = chunker(text, {
        "filename": filename, "document_id": doc_id, "language": language,
        "tenant_id": tenant.id,
    })
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="no chunks produced")
    docs = [
        Document(
            id=f"{doc_id}::chunk-{c.index}", content=c.text,
            metadata={**c.metadata, "section": c.index, "page": c.index},
        )
        for c in raw_chunks
    ]
    indexed = await retriever.index(docs)
    # merge (not add): re-ingesting the same document_id (e.g. the deterministic
    # layout_<layout> id used by /ingest-layout) updates the existing row instead
    # of raising a primary-key conflict — safe to call twice for the same id.
    await session.merge(KBDocument(
        id=doc_id, tenant_id=tenant.id, filename=filename,
        source_type=(Path(filename).suffix.lstrip(".").lower() if filename else None),
        language=language, chunk_count=indexed,
        extra_data={"chunk_ids": [d.id for d in docs]},
    ))
    await session.commit()
    log.info("ingested document", extra={"document_id": doc_id, "chunks": indexed})
    return IngestResponse(
        document_id=doc_id, filename=filename, chunks_indexed=indexed, language=language,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> IngestResponse:
    if document_id and (document_id.startswith("crm_kb_") or document_id.startswith("global_kb_")):
        raise HTTPException(
            status_code=400,
            detail=(
                f"document_id {document_id!r} is not allowed: the 'crm_kb_' / "
                "'global_kb_' prefixes are reserved for the bundled-KB seeder "
                "and purge script. Choose a different id, or omit document_id "
                "to have one generated."
            ),
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    text = parse_document(file.filename or "uploaded", data)
    return await _ingest_text(
        tenant, session, filename=file.filename or "uploaded",
        text=text, document_id=document_id,
    )


@router.post("/ingest-layout", response_model=IngestLayoutResponse)
async def ingest_layout_document(
    req: IngestLayoutRequest,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> IngestLayoutResponse:
    """Ingest one or more bundled KB docs for a given key into this tenant's
    own KB — used by the backoffice's per-tenant doc selector (admin-triggered
    via X-Tenant-Slug + admin token) or by a tenant calling this directly with
    its own bearer token. `layout` is validated against a fixed allow-list
    (_INGESTIBLE_DOCS), since it is used to look up server-side file path(s)
    — never accept an arbitrary value here.

    Single-file keys (existing frontend layouts) behave exactly as before.
    Paired keys (casino/sports/matka product modules) ingest BOTH files as
    separate KBDocument rows — never concatenated into one — since each file
    keeps its own filename for voicebot doc-priority matching."""
    paths = _INGESTIBLE_DOCS.get(req.layout)
    if paths is None:
        raise HTTPException(status_code=400, detail=f"unknown layout {req.layout!r}")
    # Pre-flight ALL files for this key before ingesting ANY of them.
    # _ingest_text commits per file, and there is no enclosing transaction, so
    # checking existence inside the loop meant a paired key (casino/sports/
    # matka) whose 1st file exists and 2nd is missing left a half-ingested KB:
    # file 1's KBDocument row + chunks committed, then a 404 with no rollback
    # and no signal that only 1 of 2 landed. All-or-nothing instead: any
    # missing file => 404 with zero writes.
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise HTTPException(
            status_code=404, detail=f"doc not found: {', '.join(missing)}")
    results: list[IngestedFileInfo] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # Deterministic document_id: re-ingesting the same key for the same
        # tenant updates the existing KBDocument row(s) (retriever.index
        # upserts by chunk id) rather than creating duplicates — safe to
        # click twice. Single-file keys keep the exact `layout_{key}` id used
        # before this change (no orphaned rows for existing tenants); paired
        # keys append the file stem to keep each file's id distinct.
        doc_id = (
            f"layout_{req.layout}"
            if len(paths) == 1
            else f"layout_{req.layout}_{path.stem}"
        )
        ingested = await _ingest_text(
            tenant, session, filename=path.name, text=text, document_id=doc_id,
        )
        results.append(IngestedFileInfo(
            document_id=ingested.document_id, filename=ingested.filename,
            chunks_indexed=ingested.chunks_indexed, language=ingested.language,
        ))
    return IngestLayoutResponse(documents=results)


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> DocumentsResponse:
    rows = (await session.execute(
        select(KBDocument).where(KBDocument.tenant_id == tenant.id)
        .order_by(KBDocument.ingested_at.desc())
    )).scalars().all()
    items = [
        DocumentInfo(
            id=r.id, filename=r.filename or r.id,
            language=r.language, chunk_count=r.chunk_count or 0,
        )
        for r in rows
    ]
    return DocumentsResponse(documents=items, total=len(items))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    retriever = _retriever_for(tenant)
    row = await session.get(KBDocument, document_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="document not found")
    chunk_ids = (row.extra_data or {}).get("chunk_ids") or [
        f"{document_id}::chunk-{i}" for i in range(row.chunk_count or 0)]
    await session.delete(row)
    await session.commit()
    n = await retriever.delete(chunk_ids)
    return {"document_id": document_id, "chunks_removed": n}


def _chunks_for_doc(retriever: HybridRetriever, document_id: str) -> str:
    """Reconstruct document text from stored chunks, sorted by section index."""
    all_chunks = retriever.list_all(max_chunks=50000)
    doc_chunks = [c for c in all_chunks
                  if c.id.startswith(f"{document_id}::") or
                  (c.metadata or {}).get("document_id") == document_id]
    doc_chunks.sort(key=lambda c: (c.metadata or {}).get("section", 0))
    return "\n\n".join(c.content for c in doc_chunks)


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Download a tenant KB document as reconstructed text."""
    row = await session.get(KBDocument, document_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="document not found")
    retriever = _retriever_for(tenant)
    text = _chunks_for_doc(retriever, document_id)
    filename = (row.filename or document_id).rsplit(".", 1)[0] + ".txt"
    return Response(content=text, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest, tenant: TenantContext = Depends(current_tenant),
) -> QueryResponse:
    retrievers = _active_retrievers(tenant)
    # Tenant-specific retriever is filtered to tenant_id; the linked CRM's retriever
    # is scoped to that crm_id (see PerCrmRetrieverRegistry / CrmKBDocument.crm_id) so
    # it only ever holds docs for tenants sharing the same CRM — no cross-tenant
    # leakage across unrelated CRMs.
    results = await search_combined(req.query, retrievers, top_k=req.top_k)
    hits = [
        QueryHit(
            chunk_id=r.document.id,
            content=r.document.content,
            score=r.score,
            dense_score=r.dense_score,
            bm25_score=r.bm25_score,
            rerank_score=r.rerank_score,
            metadata=r.document.metadata or {},
        )
        for r in results
    ]
    return QueryResponse(hits=hits, total=len(hits))


@router.get("/stats", response_model=StatsResponse)
async def stats(
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> StatsResponse:
    from src.models.crm import CrmKBDocument

    tenant_rows = (await session.execute(
        select(KBDocument).where(KBDocument.tenant_id == tenant.id)
    )).scalars().all()
    crm_id = getattr(tenant.settings, "crm_id", None)
    crm_rows = []
    if crm_id:
        crm_rows = (await session.execute(
            select(CrmKBDocument).where(CrmKBDocument.crm_id == crm_id)
        )).scalars().all()
    all_rows = tenant_rows + crm_rows
    return StatsResponse(
        document_count=len(all_rows),
        chunk_count=sum(r.chunk_count or 0 for r in all_rows),
    )


def _new_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"
