"""KB-embedding call cost tracking (Phase 2 of the chat-cost-widening plan;
Phase 1 was voice-note TTS/STT media cost — see ``src/models/chat_turn_metrics.py``).

One ``EmbeddingUsage`` row per embed batch, covering both the ingest path
(``HybridRetriever.index()`` embedding new/uploaded chunks) and the search
path (``HybridRetriever._dense_search()`` embedding the query). Written
best-effort from ``record_embedding_usage`` below (see its own docstring for
the never-raises contract) so a DB hiccup never breaks KB ingest or a live
chat/voice turn's search — same posture as
``src/models/chat_turn_metrics.py::record_chat_turn_metric`` and
``src/models/turn_metrics.py::record_turn_metric``.

Scoping: exactly one of ``tenant_id``/``crm_id`` is set per row (or, for a
legacy/global row the app does not currently write, both None) — mirrors the
pgvector chunk-scoping rule (a tenant-KB row has ``tenant_id`` set/``crm_id``
NULL; a CRM shared-KB row has ``crm_id`` set/``tenant_id`` NULL). See
``tests/unit/test_pgvector_crm_scoping.py`` for the analogous chunk-level
guarantee this mirrors.

``tokens`` NULLABLE, no backfill: the Gemini Developer API's
``embed_content`` response reports no token count (Vertex-only field), so
NULL means "the embedder reported no usage count" — cost was then estimated
from ``input_chars`` (see ``src/api/chat_cost.py``'s
``compute_embedding_cost``/``_EMBED_CHARS_PER_TOKEN``). Never backfill NULL
to 0 — that would claim a real, reported zero-token batch where none was
observed.

No PII: only provider/model names (short catalog strings), lengths, and
numbers — this table must NEVER store chunk text or query text.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base, get_sessionmaker
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

EMBEDDING_PURPOSES = frozenset({"ingest", "search"})


class EmbeddingUsage(Base):
    __tablename__ = "embedding_usage"
    __table_args__ = (
        Index("idx_embedding_usage_tenant_created", "tenant_id", "created_at"),
        Index("idx_embedding_usage_crm_created", "crm_id", "created_at"),
        Index("idx_embedding_usage_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
    )
    crm_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="CASCADE"),
    )
    purpose: Mapped[str] = mapped_column(String(20), nullable=False)  # "ingest" | "search"
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    input_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # NULL means the embedder reported no usage count -- see module docstring.
    tokens: Mapped[Optional[int]] = mapped_column(Integer)
    cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(),
    )


async def record_embedding_usage(
    *,
    tenant_id: Optional[str],
    crm_id: Optional[str],
    purpose: str,
    provider: str,
    model: str,
    input_chars: int,
    tokens: Optional[int] = None,
) -> Optional[int]:
    """Record one embed-batch's cost, best-effort. Never raises into the
    ingest/search caller (``HybridRetriever``, via its injected
    ``record_embedding_usage`` callback — see ``src/rag/retriever.py`` and
    ``src/bootstrap.py``'s ``_embedding_usage_recorder``), same never-raises
    contract as ``src/models/chat_turn_metrics.py::record_chat_turn_metric``.

    Exactly one of ``tenant_id``/``crm_id`` should be set (or both None for a
    legacy/global row, which the app does not currently write). ``tokens`` is
    None when the embedder reported no usage count; ``cost`` is then
    estimated from ``input_chars`` (see ``src/api/chat_cost.py``'s
    ``compute_embedding_cost``).

    Returns the new row's id, or None if the purpose is unrecognized or the
    write failed.
    """
    if purpose not in EMBEDDING_PURPOSES:
        log.warning("record_embedding_usage: unknown purpose %r; not recorded", purpose)
        return None
    try:
        from src.api.chat_cost import compute_embedding_cost  # lazy: avoid models->api import at load

        sessionmaker = get_sessionmaker()
        async with sessionmaker() as db:
            cost = await compute_embedding_cost(
                db, provider=provider, model=model, input_chars=input_chars, tokens=tokens,
            )
            row = EmbeddingUsage(
                tenant_id=tenant_id, crm_id=crm_id, purpose=purpose, provider=provider,
                model=model, input_chars=input_chars, tokens=tokens, cost=cost,
            )
            db.add(row)
            await db.commit()
            debug_event(
                log, "metrics embedding_usage_write response",
                row_id=row.id, tenant_id=tenant_id, crm_id=crm_id, purpose=purpose,
                provider=provider, model=model, input_chars=input_chars, tokens=tokens, cost=cost,
            )
            return row.id
    except Exception:  # noqa: BLE001 - must never break a live ingest/search call
        log.warning(
            "record_embedding_usage failed; continuing without persistence", exc_info=True,
        )
        debug_event(
            log, "metrics embedding_usage_write failed",
            tenant_id=tenant_id, crm_id=crm_id, purpose=purpose, provider=provider, model=model,
        )
        return None
