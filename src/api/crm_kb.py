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
from src.utils.logging import debug_event

log = logging.getLogger(__name__)
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
        # Process-wiring gap, not a per-CRM one: set_crm_retrievers() never
        # ran (app startup order / DI wiring), so EVERY CRM KB route 503s
        # until that's fixed — worth distinguishing from the per-CRM case
        # below, where wiring is fine but this one CRM has no usable index.
        debug_event(log, "crm_kb retriever_lookup registry_not_initialized", crm_id=crm_id)
        raise HTTPException(status_code=503, detail="CRM KB not initialized")
    retriever = _crm_retrievers.get(crm_id)
    if retriever is None:
        debug_event(log, "crm_kb retriever_lookup no_retriever_for_crm", crm_id=crm_id)
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
    if indexed != len(docs):
        # Decision the retriever made, not this route -- it accepted fewer
        # (or more) chunks than were produced. chunk_count below is stored as
        # `indexed`, so a silent mismatch here would make the document list
        # and the actual retrievable content disagree with no trace.
        debug_event(
            log, "crm_kb ingest chunk_count_mismatch", crm_id=crm_id, document_id=doc_id,
            chunks_produced=len(docs), chunks_indexed=indexed,
        )
    session.add(CrmKBDocument(
        id=doc_id, crm_id=crm_id, filename=file.filename or doc_id,
        source_type=(Path(file.filename).suffix.lstrip(".").lower() if file.filename else None),
        language=language, chunk_count=indexed,
        extra_data={"chunk_ids": [d.id for d in docs]},
    ))
    # CRUD boundary: a CRM KB doc is shared across every tenant linked to
    # this CRM (module docstring). Full non-PII values — this is platform
    # reference content the model is meant to cite, not customer data.
    debug_event(
        log, "crm_kb ingest result", crm_id=crm_id, document_id=doc_id,
        document_filename=file.filename, language=language, text_chars=len(text),
        chunks_produced=len(docs), chunks_indexed=indexed,
    )
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
    if n != len(chunk_ids):
        # The DB row is gone either way (already committed above); this is
        # whether the vector index actually shed every chunk it should have —
        # a stale chunk left behind after "delete" reports success is exactly
        # the kind of drift that surfaces later as a citation from a document
        # that no longer appears in list_crm_documents.
        debug_event(
            log, "crm_kb delete chunk_removal_mismatch", crm_id=crm_id,
            document_id=document_id, chunk_ids_expected=len(chunk_ids), chunks_removed=n,
        )
    debug_event(
        log, "crm_kb delete result", crm_id=crm_id, document_id=document_id,
        document_filename=row.filename, chunks_removed=n,
    )
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
    if not text and (row.chunk_count or 0) > 0:
        # The DB row claims chunks exist but none of them matched this
        # document_id in the retriever's own index (_chunks_for_doc's id-
        # prefix / metadata match) — the download silently comes back empty
        # rather than 404ing, which otherwise looks identical to "this
        # document genuinely has no content".
        debug_event(
            log, "crm_kb download empty_despite_chunk_count", crm_id=crm_id,
            document_id=document_id, stored_chunk_count=row.chunk_count,
        )
    filename = (row.filename or document_id).rsplit(".", 1)[0] + ".txt"
    return Response(content=text, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
