"""Knowledge base / RAG endpoints (PRD §7.5).

A retriever instance is injected via ``set_retriever`` at app startup so the
endpoints stay testable in isolation. The retriever pairs a vector store
with an embedder and (optionally) a reranker — see ``src.rag.retriever``.

Endpoints:
- POST   /knowledge/ingest         multipart upload, parses + chunks + indexes
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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.auth.middleware import require_admin
from src.interfaces.vector_store import Document
from src.models.benchmark import KBDocument, PlatformKBDocument
from src.rag.context_builder import search_combined
from src.rag.ingestion import ChunkConfig, detect_language, get_chunker, parse_document
from src.rag.retriever import HybridRetriever

log = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# --- DI -----------------------------------------------------------------


# A factory that returns the TENANT retriever (for ingest + tenant-specific search).
# A separate platform retriever holds the global KB shared across all tenants.
RetrieverFactory = Callable[[TenantContext], HybridRetriever]
_retriever_factory: Optional[RetrieverFactory] = None
_platform_retriever: Optional[HybridRetriever] = None
_chunk_config: ChunkConfig = ChunkConfig()


def set_retriever_factory(
    factory: Optional[RetrieverFactory], chunk_config: Optional[ChunkConfig] = None,
) -> None:
    global _retriever_factory, _chunk_config
    _retriever_factory = factory
    if chunk_config is not None:
        _chunk_config = chunk_config


def set_platform_retriever(retriever: Optional[HybridRetriever]) -> None:
    """Set the platform-level shared retriever (global KB, all tenants)."""
    global _platform_retriever
    _platform_retriever = retriever


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


def _active_retrievers(tenant: TenantContext) -> list[HybridRetriever]:
    """Platform retriever + tenant retriever, skipping None."""
    return [r for r in [_platform_retriever, _retriever_for(tenant)] if r is not None]


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


# --- Routes -------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> IngestResponse:
    retriever = _retriever_for(tenant)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    text = parse_document(file.filename or "uploaded", data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="document parsed to empty text")

    doc_id = document_id or _new_id()
    language = detect_language(text)
    chunker = get_chunker(_chunk_config)
    raw_chunks = chunker(text, {
        "filename": file.filename,
        "document_id": doc_id,
        "language": language,
        "tenant_id": tenant.id,
    })
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="no chunks produced")

    docs = [
        Document(
            id=f"{doc_id}::chunk-{c.index}",
            content=c.text,
            metadata={
                **c.metadata,
                "section": c.index,
                "page": c.index,
            },
        )
        for c in raw_chunks
    ]
    indexed = await retriever.index(docs)
    session.add(KBDocument(
        id=doc_id,
        tenant_id=tenant.id,
        filename=file.filename or doc_id,
        source_type=(Path(file.filename).suffix.lstrip(".").lower() if file.filename else None),
        language=language,
        chunk_count=indexed,
        extra_data={"chunk_ids": [d.id for d in docs]},
    ))
    await session.commit()
    log.info("ingested document", extra={"document_id": doc_id, "chunks": indexed})
    return IngestResponse(
        document_id=doc_id,
        filename=file.filename or "",
        chunks_indexed=indexed,
        language=language,
    )


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


@router.get("/platform-documents/{document_id}/download")
async def download_platform_document(
    document_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Admin: download a platform KB document as reconstructed text."""
    await require_admin(request)
    if _platform_retriever is None:
        raise HTTPException(status_code=503, detail="platform retriever not initialised")
    row = await session.get(PlatformKBDocument, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="platform document not found")
    text = _chunks_for_doc(_platform_retriever, document_id)
    filename = (row.filename or document_id).rsplit(".", 1)[0] + ".txt"
    return Response(content=text, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/platform-documents", response_model=DocumentsResponse)
async def list_platform_documents(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentsResponse:
    """Admin: list all documents in the platform (global) knowledge base."""
    await require_admin(request)
    rows = (await session.execute(
        select(PlatformKBDocument).order_by(PlatformKBDocument.ingested_at.desc())
    )).scalars().all()
    items = [
        DocumentInfo(id=r.id, filename=r.filename or r.id,
                     language=r.language, chunk_count=r.chunk_count or 0)
        for r in rows
    ]
    return DocumentsResponse(documents=items, total=len(items))


@router.post("/platform-ingest", response_model=IngestResponse)
async def platform_ingest_document(
    request: Request,
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_db_session),
) -> IngestResponse:
    """Admin: upload a document into the platform (global) knowledge base."""
    await require_admin(request)
    if _platform_retriever is None:
        raise HTTPException(status_code=503, detail="platform retriever not initialised")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    text = parse_document(file.filename or "uploaded", data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="document parsed to empty text")

    doc_id = document_id or f"platform_{_new_id()}"
    language = detect_language(text)
    chunker = get_chunker(_chunk_config)
    raw_chunks = chunker(text, {
        "filename": file.filename,
        "document_id": doc_id,
        "language": language,
    })
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="no chunks produced")

    docs = [
        Document(
            id=f"{doc_id}::chunk-{c.index}",
            content=c.text,
            metadata={**c.metadata, "section": c.index, "page": c.index},
        )
        for c in raw_chunks
    ]
    indexed = await _platform_retriever.index(docs)
    session.add(PlatformKBDocument(
        id=doc_id,
        filename=file.filename or doc_id,
        source_type=(Path(file.filename).suffix.lstrip(".").lower() if file.filename else None),
        language=language,
        chunk_count=indexed,
        extra_data={"chunk_ids": [d.id for d in docs]},
    ))
    await session.commit()
    log.info("platform: ingested document", extra={"document_id": doc_id, "chunks": indexed})
    return IngestResponse(document_id=doc_id, filename=file.filename or "",
                          chunks_indexed=indexed, language=language)


@router.delete("/platform-documents/{document_id}")
async def delete_platform_document(
    document_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Admin: delete a document from the platform (global) knowledge base."""
    await require_admin(request)
    if _platform_retriever is None:
        raise HTTPException(status_code=503, detail="platform retriever not initialised")
    row = await session.get(PlatformKBDocument, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="platform document not found")
    chunk_ids = (row.extra_data or {}).get("chunk_ids") or [
        f"{document_id}::chunk-{i}" for i in range(row.chunk_count or 0)]
    await session.delete(row)
    await session.commit()
    n = await _platform_retriever.delete(chunk_ids)
    return {"document_id": document_id, "chunks_removed": n}


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest, tenant: TenantContext = Depends(current_tenant),
) -> QueryResponse:
    retrievers = _active_retrievers(tenant)
    # Tenant-specific retriever is filtered to tenant_id; platform retriever is
    # unfiltered (platform docs have no tenant_id). Pass tenant filter only to the
    # tenant retriever via search_combined's per-retriever filter support is not yet
    # implemented — instead, platform docs are naturally unscoped (no tenant_id in
    # metadata) and FAISS has no FK, so cross-tenant leakage via the platform index
    # is fine (it only holds shared global docs).
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
    tenant_rows = (await session.execute(
        select(KBDocument).where(KBDocument.tenant_id == tenant.id)
    )).scalars().all()
    platform_rows = (await session.execute(
        select(PlatformKBDocument)
    )).scalars().all()
    all_rows = tenant_rows + platform_rows
    return StatsResponse(
        document_count=len(all_rows),
        chunk_count=sum(r.chunk_count or 0 for r in all_rows),
    )


def _new_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"
