"""Admin CRUD for a CRM's shared Knowledge Base documents.

- ``POST   /crms/{crm_id}/kb/ingest``               upload a doc into that CRM's KB
- ``GET    /crms/{crm_id}/kb/documents``             list that CRM's KB docs
- ``DELETE /crms/{crm_id}/kb/documents/{doc_id}``    remove a doc
- ``GET    /crms/{crm_id}/kb/documents/{doc_id}/download``  reconstructed text

Admin-only (``require_admin``) — a CRM's KB is shared platform config, not
tenant-scoped, same as ``src/api/crms.py``. A retriever registry is injected
via ``set_crm_retrievers`` at app startup (same DI pattern as
``src/api/knowledge.py``'s ``set_retriever_factory``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth.audit import log_denied
from src.auth.middleware import require_admin
from src.interfaces.vector_store import Document
from src.models.crm import Crm, CrmKBDocument
from src.rag.ingestion import ChunkConfig, detect_language, get_chunker, parse_document

router = APIRouter(prefix="/crms", tags=["crm-kb"])

_crm_retrievers: "object | None" = None
_chunk_config = ChunkConfig()


def set_crm_retrievers(registry) -> None:
    """Inject the per-CRM retriever registry (``src.bootstrap.PerCrmRetrieverRegistry``,
    typed loosely here to avoid an import cycle with ``src.bootstrap``)."""
    global _crm_retrievers
    _crm_retrievers = registry


def _retriever_for_crm(crm_id: str):
    if _crm_retrievers is None:
        raise HTTPException(status_code=503, detail="CRM KB not initialized")
    retriever = _crm_retrievers.get(crm_id)
    if retriever is None:
        raise HTTPException(status_code=503, detail=f"CRM {crm_id!r} has no usable KB retriever")
    return retriever


async def _require_crm(session: AsyncSession, crm_id: str) -> None:
    if await session.get(Crm, crm_id) is None:
        raise HTTPException(status_code=404, detail=f"CRM {crm_id!r} not found")


async def _scoped_crm_doc(
    session: AsyncSession, document_id: str, crm_id: str
) -> CrmKBDocument:
    """Fetch a CRM KB document row and 404 if it doesn't belong to ``crm_id``."""
    row = await session.get(CrmKBDocument, document_id)
    if row is None or row.crm_id != crm_id:
        log_denied(
            logging.INFO, "cross-CRM KB document access denied",
            event="admin_scope_denied",
            reason=("crm_kb_document_not_in_crm" if row is not None
                    else "crm_kb_document_not_found"),
            resource="crm_kb_document", resource_id=document_id,
            crm_id=crm_id,
            found=row is not None,
            owner_crm_id=(row.crm_id if row is not None else None),
        )
        raise HTTPException(status_code=404, detail="document not found")
    return row


class DocumentInfo(BaseModel):
    id: str
    filename: str
    language: Optional[str] = None
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    language: Optional[str]


def _new_id() -> str:
    import uuid
    return f"crmdoc_{uuid.uuid4().hex[:12]}"


@router.post("/{crm_id}/kb/ingest", response_model=IngestResponse)
async def ingest_crm_document(
    crm_id: str,
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> IngestResponse:
    await _require_crm(session, crm_id)
    retriever = _retriever_for_crm(crm_id)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    text = parse_document(file.filename or "uploaded", data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="document parsed to empty text")

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
    doc_id = document_id or _new_id()
    language = detect_language(text)
    chunker = get_chunker(_chunk_config)
    raw_chunks = chunker(text, {
        "filename": file.filename, "document_id": doc_id, "language": language,
    })
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="no chunks produced")

    docs = [
        Document(id=f"{doc_id}::chunk-{c.index}", content=c.text,
                 metadata={**c.metadata, "section": c.index, "page": c.index})
        for c in raw_chunks
    ]
    indexed = await retriever.index(docs)
    session.add(CrmKBDocument(
        id=doc_id, crm_id=crm_id, filename=file.filename or doc_id,
        source_type=(Path(file.filename).suffix.lstrip(".").lower() if file.filename else None),
        language=language, chunk_count=indexed,
        extra_data={"chunk_ids": [d.id for d in docs]},
    ))
    await session.commit()
    return IngestResponse(document_id=doc_id, filename=file.filename or "",
                          chunks_indexed=indexed, language=language)


@router.get("/{crm_id}/kb/documents", response_model=DocumentsResponse)
async def list_crm_documents(
    crm_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> DocumentsResponse:
    await _require_crm(session, crm_id)
    rows = (await session.execute(
        select(CrmKBDocument).where(CrmKBDocument.crm_id == crm_id)
        .order_by(CrmKBDocument.ingested_at.desc())
    )).scalars().all()
    items = [
        DocumentInfo(id=r.id, filename=r.filename or r.id,
                     language=r.language, chunk_count=r.chunk_count or 0)
        for r in rows
    ]
    return DocumentsResponse(documents=items, total=len(items))


@router.delete("/{crm_id}/kb/documents/{document_id}")
async def delete_crm_document(
    crm_id: str,
    document_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> dict:
    await _require_crm(session, crm_id)
    retriever = _retriever_for_crm(crm_id)
    row = await _scoped_crm_doc(session, document_id, crm_id)
    chunk_ids = (row.extra_data or {}).get("chunk_ids") or [
        f"{document_id}::chunk-{i}" for i in range(row.chunk_count or 0)]
    await session.delete(row)
    await session.commit()
    n = await retriever.delete(chunk_ids)
    return {"document_id": document_id, "chunks_removed": n}


def _chunks_for_doc(retriever, document_id: str) -> str:
    all_chunks = retriever.list_all(max_chunks=50000)
    doc_chunks = [c for c in all_chunks
                  if c.id.startswith(f"{document_id}::") or
                  (c.metadata or {}).get("document_id") == document_id]
    doc_chunks.sort(key=lambda c: (c.metadata or {}).get("section", 0))
    return "\n\n".join(c.content for c in doc_chunks)


@router.get("/{crm_id}/kb/documents/{document_id}/download")
async def download_crm_document(
    crm_id: str,
    document_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> Response:
    await _require_crm(session, crm_id)
    row = await _scoped_crm_doc(session, document_id, crm_id)
    retriever = _retriever_for_crm(crm_id)
    text = _chunks_for_doc(retriever, document_id)
    filename = (row.filename or document_id).rsplit(".", 1)[0] + ".txt"
    return Response(content=text, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
