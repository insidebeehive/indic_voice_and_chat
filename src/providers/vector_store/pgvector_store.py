"""PGVector adapter for the IVectorStore interface.

Stores all knowledge chunks in a single ``voicebot.knowledge_chunks`` Postgres
table, scoped by EITHER ``tenant_id`` (a tenant's own KB) OR ``crm_id`` (a
CRM's shared KB) — exactly one is set per adapter instance, never both, never
neither.  Uses the ``pgvector`` extension's cosine-distance operator (``<=>``)
with an HNSW index.

Configuration keys (config dict passed from TenantProviders._config_for or
src.bootstrap.build_crm_retriever):
  provider       : "pgvector"
  embedding_dim  : int  (default 384)
  tenant_id      : str | None  (a tenant's own KB)
  crm_id         : str | None  (a CRM's shared KB)
  database_url   : str  (optional; falls back to DATABASE_URL env var)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import numpy as np

from src.config import strip_libpq_only_query_params
from src.interfaces.vector_store import Document, IVectorStore, SearchResult

log = logging.getLogger(__name__)

# Module-level shared pool — one pool for all PGVectorAdapter instances.
_pool: Any = None  # asyncpg.Pool
_pool_lock: Optional[asyncio.Lock] = None
_schema_ready = False
_schema_lock: Optional[asyncio.Lock] = None


def _to_asyncpg_dsn(url: str) -> str:
    # database_url falls back to the raw DATABASE_URL env var (unlike
    # src.config's SQLAlchemy-facing settings.database.url), so libpq-only
    # params a managed provider appended (e.g. Neon's channel_binding) are
    # still present here and must be stripped before this DSN reaches asyncpg.
    return strip_libpq_only_query_params(url.replace("postgresql+asyncpg://", "postgresql://"))


async def _init_conn(conn: Any) -> None:
    """Register pgvector codec on every new connection."""
    try:
        import pgvector.asyncpg as _pva  # type: ignore[import-not-found]
        await _pva.register_vector(conn)
    except Exception:
        log.exception("pgvector asyncpg codec registration failed")


async def _get_pool(database_url: str) -> Any:
    global _pool, _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            import asyncpg  # type: ignore[import-not-found]
            dsn = _to_asyncpg_dsn(database_url)
            _pool = await asyncpg.create_pool(
                dsn, min_size=1, max_size=10, init=_init_conn
            )
            log.info("pgvector pool created")
    return _pool


async def _ensure_schema(pool: Any, dim: int) -> None:
    """Verify the table exists; do NOT attempt CREATE EXTENSION (needs superuser).
    Run the setup SQL from docs/pgvector_setup.sql once as the postgres user."""
    global _schema_ready, _schema_lock
    if _schema_ready:
        return
    if _schema_lock is None:
        _schema_lock = asyncio.Lock()
    async with _schema_lock:
        if _schema_ready:
            return
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'voicebot' AND table_name = 'knowledge_chunks'"
            )
            if not exists:
                raise RuntimeError(
                    "voicebot.knowledge_chunks table not found. "
                    "Run docs/pgvector_setup.sql as the postgres superuser first."
                )
        _schema_ready = True
        log.info("pgvector schema verified (dim=%d)", dim)


class PGVectorAdapter(IVectorStore):
    """IVectorStore backed by PostgreSQL + pgvector.

    Each adapter instance is scoped to exactly one of ``tenant_id`` (a
    tenant's own KB) or ``crm_id`` (a CRM's shared KB) — never both, never
    neither, for any instance built by this codebase today.
    """

    def __init__(self, config: dict) -> None:
        self._dim = int(config.get("embedding_dim", 384))
        self._tenant_id: Optional[str] = config.get("tenant_id")
        self._crm_id: Optional[str] = config.get("crm_id")
        self._database_url: str = (
            config.get("database_url")
            or os.environ.get("DATABASE_URL", "")
        )
        if not self._database_url:
            raise ValueError(
                "PGVectorAdapter requires database_url config or DATABASE_URL env var"
            )

    # --- helpers -----------------------------------------------------------

    async def _pool(self) -> Any:
        pool = await _get_pool(self._database_url)
        await _ensure_schema(pool, self._dim)
        return pool

    def _scope_clause(self, param_start: int) -> tuple[str, list]:
        """SQL WHERE clause + params scoping rows to this adapter's tenant or
        CRM. ``param_start`` is the next free ``$N`` placeholder index."""
        if self._tenant_id is not None:
            return f"tenant_id = ${param_start}", [self._tenant_id]
        if self._crm_id is not None:
            return f"crm_id = ${param_start}", [self._crm_id]
        return "tenant_id IS NULL AND crm_id IS NULL", []

    # --- IVectorStore ------------------------------------------------------

    async def index(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        pool = await self._pool()
        rows = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id!r} has no embedding")
            emb = np.array(doc.embedding, dtype="float32")
            rows.append((doc.id, doc.content, doc.metadata or {}, emb,
                         self._tenant_id, self._crm_id))

        import json as _json
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO voicebot.knowledge_chunks
                    (id, content, metadata, embedding, tenant_id, crm_id)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        tenant_id = EXCLUDED.tenant_id,
                        crm_id = EXCLUDED.crm_id
                """,
                [(r[0], r[1], _json.dumps(r[2]), r[3], r[4], r[5]) for r in rows],
            )
        return len(documents)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        pool = await self._pool()
        q = np.array(query_embedding, dtype="float32")

        scope_clause, scope_params = self._scope_clause(2)

        filter_clause = ""
        filter_params: list = []
        if filters:
            import json as _json
            next_pos = 2 + len(scope_params)
            filter_clause = f" AND metadata @> ${next_pos}::jsonb"
            filter_params = [_json.dumps(filters)]

        limit_pos = 2 + len(scope_params) + len(filter_params)
        sql = f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM voicebot.knowledge_chunks
            WHERE {scope_clause}{filter_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${limit_pos}::integer
        """
        params = [q] + scope_params + filter_params + [top_k]

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for r in rows:
            raw = r["metadata"]
            meta = json.loads(raw) if isinstance(raw, str) else (dict(raw) if raw else {})
            doc = Document(id=r["id"], content=r["content"], metadata=meta)
            results.append(SearchResult(document=doc, score=float(r["score"])))
        return results

    async def delete(self, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        pool = await self._pool()
        scope_clause, scope_params = self._scope_clause(2)
        sql = f"DELETE FROM voicebot.knowledge_chunks WHERE id = ANY($1) AND {scope_clause}"
        async with pool.acquire() as conn:
            result = await conn.execute(sql, doc_ids, *scope_params)
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def count(self) -> int:
        pool = await self._pool()
        scope_clause, scope_params = self._scope_clause(1)
        sql = f"SELECT count(*) FROM voicebot.knowledge_chunks WHERE {scope_clause}"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *scope_params)
        return int(row["count"]) if row else 0
