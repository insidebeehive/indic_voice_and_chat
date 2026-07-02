"""PGVector adapter for the IVectorStore interface.

Stores all knowledge chunks in a single ``knowledge_chunks`` Postgres table,
scoped by ``tenant_id`` column (NULL = platform / global docs).  Uses the
``pgvector`` extension's cosine-distance operator (``<=>``) with an HNSW index.

Configuration keys (config dict passed from TenantProviders._config_for):
  provider       : "pgvector"
  embedding_dim  : int  (default 384)
  tenant_id      : str | None  (None → platform docs)
  database_url   : str  (optional; falls back to DATABASE_URL env var)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import numpy as np

from src.interfaces.vector_store import Document, IVectorStore, SearchResult

log = logging.getLogger(__name__)

# Module-level shared pool — one pool for all PGVectorAdapter instances.
_pool: Any = None  # asyncpg.Pool
_pool_lock: Optional[asyncio.Lock] = None
_schema_ready = False
_schema_lock: Optional[asyncio.Lock] = None


def _to_asyncpg_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


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
                "WHERE table_name = 'knowledge_chunks'"
            )
            if not exists:
                raise RuntimeError(
                    "knowledge_chunks table not found. "
                    "Run docs/pgvector_setup.sql as the postgres superuser first."
                )
        _schema_ready = True
        log.info("pgvector schema verified (dim=%d)", dim)


class PGVectorAdapter(IVectorStore):
    """IVectorStore backed by PostgreSQL + pgvector."""

    def __init__(self, config: dict) -> None:
        self._dim = int(config.get("embedding_dim", 384))
        self._tenant_id: Optional[str] = config.get("tenant_id")
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

    def _tenant_where(self) -> str:
        """SQL snippet that scopes rows to this adapter's tenant."""
        return "tenant_id = $%d" if self._tenant_id is not None else "tenant_id IS NULL"

    def _tenant_param(self, base: int) -> list:
        """Extra positional param(s) for tenant filtering."""
        return [self._tenant_id] if self._tenant_id is not None else []

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
            rows.append((doc.id, doc.content, doc.metadata or {}, emb, self._tenant_id))

        import json as _json
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO knowledge_chunks (id, content, metadata, embedding, tenant_id)
                VALUES ($1, $2, $3::jsonb, $4, $5)
                ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        tenant_id = EXCLUDED.tenant_id
                """,
                [(r[0], r[1], _json.dumps(r[2]), r[3], r[4]) for r in rows],
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

        # Tenant scope
        tenant_clause = (
            "tenant_id = $2" if self._tenant_id is not None else "tenant_id IS NULL"
        )
        tenant_params = [self._tenant_id] if self._tenant_id is not None else []

        # Optional extra metadata filters (JSONB containment)
        filter_clause = ""
        filter_params: list = []
        if filters:
            import json as _json
            next_pos = 2 + len(tenant_params) + 1  # $1=query, $2[opt]=tenant, next=filters
            filter_clause = f" AND metadata @> ${next_pos}::jsonb"
            filter_params = [_json.dumps(filters)]

        limit_pos = 2 + len(tenant_params) + len(filter_params) + 1
        sql = f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM knowledge_chunks
            WHERE {tenant_clause}{filter_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${limit_pos}
        """
        params = [q] + tenant_params + filter_params + [top_k]

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for r in rows:
            meta = dict(r["metadata"]) if r["metadata"] else {}
            doc = Document(id=r["id"], content=r["content"], metadata=meta)
            results.append(SearchResult(document=doc, score=float(r["score"])))
        return results

    async def delete(self, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        pool = await self._pool()
        tenant_clause = (
            "tenant_id = $2" if self._tenant_id is not None else "tenant_id IS NULL"
        )
        tenant_params = [self._tenant_id] if self._tenant_id is not None else []
        sql = f"DELETE FROM knowledge_chunks WHERE id = ANY($1) AND {tenant_clause}"
        async with pool.acquire() as conn:
            result = await conn.execute(sql, doc_ids, *tenant_params)
        # asyncpg returns "DELETE N" as a string
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def count(self) -> int:
        pool = await self._pool()
        tenant_clause = (
            "tenant_id = $1" if self._tenant_id is not None else "tenant_id IS NULL"
        )
        tenant_params = [self._tenant_id] if self._tenant_id is not None else []
        sql = f"SELECT count(*) FROM knowledge_chunks WHERE {tenant_clause}"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *tenant_params)
        return int(row["count"]) if row else 0
