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
import re
from typing import Any, Optional

import numpy as np

from src.config import strip_libpq_only_query_params
from src.interfaces.vector_store import Document, IVectorStore, SearchResult

log = logging.getLogger(__name__)

# Module-level shared pools, one per distinct DSN -- NOT one global pool.
# Two PGVectorAdapter instances configured with different database_url values
# (e.g. two tenants on separate managed Postgres instances) must never share a
# connection pool: keying by DSN is what stops the second adapter from
# silently querying the first adapter's database.
_pools: dict[str, Any] = {}  # dsn -> asyncpg.Pool
_pool_lock: Optional[asyncio.Lock] = None
_schema_ready: set[str] = set()  # dsn's whose table existence has been verified
_schema_lock: Optional[asyncio.Lock] = None
# (dsn, embedding_dim) pairs already checked against the live table's
# `embedding` column width. Tracked separately from _schema_ready (which only
# covers table *existence*) because a single DSN's pool is shared across
# every PGVectorAdapter instance pointed at it, and different instances
# (e.g. a tenant KB vs. a CRM KB) can be configured with different
# embedding_dim values. If we only ever ran the dimension check once per DSN,
# the first adapter to call _ensure_schema would mark that DSN "ready" and
# every later adapter against the SAME DSN with a *different* (possibly
# wrong) dim would skip verification entirely -- deferring a
# misconfiguration to a confusing asyncpg.DataError at INSERT time instead of
# failing loudly at connect time. Keyed by (dsn, dim) -- not dim alone -- so
# that keying stays consistent with _pools/_schema_ready now that pools are
# DSN-scoped: keying this cache by dim alone while the pool is keyed by DSN
# would let a second adapter against a *different* database skip
# verification just because some other database was already checked at the
# same dim.
_verified_dims: set[tuple[str, int]] = set()
_dim_lock: Optional[asyncio.Lock] = None


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


async def _get_pool(database_url: str) -> tuple[Any, str]:
    """Return (pool, normalized_dsn) for ``database_url``, creating a new
    pool the first time this DSN is seen. Returns the normalized DSN too so
    callers can key the dim-verification cache the same way this function
    keys the pool cache -- see _verified_dims above for why that consistency
    matters."""
    global _pools, _pool_lock
    dsn = _to_asyncpg_dsn(database_url)
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    if dsn in _pools:
        return _pools[dsn], dsn
    async with _pool_lock:
        if dsn not in _pools:
            import asyncpg  # type: ignore[import-not-found]
            _pools[dsn] = await asyncpg.create_pool(
                dsn, min_size=1, max_size=10, init=_init_conn
            )
            log.info("pgvector pool created")
    return _pools[dsn], dsn


async def _ensure_schema(pool: Any, dsn: str, dim: int) -> None:
    """Verify the table exists; do NOT attempt CREATE EXTENSION (needs superuser).
    Run the setup SQL from docs/pgvector_setup.sql once as the postgres user.

    Also verifies that this adapter's configured ``embedding_dim`` matches
    the live table's ``embedding`` column width, so a tenant with the wrong
    embedding_dim learns about it here -- at connect time, with both numbers
    named -- instead of from an opaque asyncpg.DataError on their first
    ingest. See _verified_dims above for why this is cached per-(dsn, dim)
    rather than folded into the one-shot-per-dsn _schema_ready set.

    ``dsn`` must be the same normalized DSN _get_pool used to look up
    ``pool`` -- both caches key on it so a second adapter against a
    *different* database never rides on the first adapter's verification.
    """
    global _schema_ready, _schema_lock, _verified_dims, _dim_lock
    if dsn not in _schema_ready:
        if _schema_lock is None:
            _schema_lock = asyncio.Lock()
        async with _schema_lock:
            if dsn not in _schema_ready:
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
                _schema_ready.add(dsn)
                log.info("pgvector schema verified (table exists)")

    dim_key = (dsn, dim)
    if dim_key in _verified_dims:
        return
    if _dim_lock is None:
        _dim_lock = asyncio.Lock()
    async with _dim_lock:
        if dim_key in _verified_dims:
            return
        async with pool.acquire() as conn:
            # format_type() renders the column's declared type, e.g.
            # "vector(384)" -- reading it beats inferring the dimension from
            # atttypmod's raw encoding, which is type-specific and not
            # documented as stable across pgvector versions.
            col_type = await conn.fetchval(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'voicebot'
                  AND c.relname = 'knowledge_chunks'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """
            )
            # re.search (not re.match): format_type() schema-qualifies the
            # type when it isn't on search_path (e.g. Supabase installs
            # pgvector into "extensions" by default, rendering
            # "extensions.vector(384)"). An anchored re.match would silently
            # fail to find the dimension in that case.
            match = re.search(r"vector\((\d+)\)", col_type or "")
            actual_dim = int(match.group(1)) if match else None
            if actual_dim is not None and actual_dim != dim:
                raise RuntimeError(
                    f"embedding_dim mismatch: voicebot.knowledge_chunks.embedding "
                    f"is vector({actual_dim}) but this PGVectorAdapter is "
                    f"configured with embedding_dim={dim}. Fix the "
                    f"'embedding_dim' config key to match the live table "
                    f"(the table is not altered automatically)."
                )
            if actual_dim is None:
                # col_type NULL (column renamed/dropped), a non-vector type,
                # or a vector column with no dimension modifier all land
                # here. Fail OPEN (let the caller proceed -- the table might
                # still work) but do NOT cache this as verified: an
                # indeterminate result must not be remembered as success, or
                # a real mismatch introduced later would never be caught.
                log.warning(
                    "pgvector dim check inconclusive for "
                    "voicebot.knowledge_chunks.embedding: format_type "
                    "returned %r (expected something matching "
                    "'vector(<dim>)'); configured embedding_dim=%d will NOT "
                    "be verified against the live column",
                    col_type, dim,
                )
                return
        _verified_dims.add(dim_key)
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
        pool, dsn = await _get_pool(self._database_url)
        await _ensure_schema(pool, dsn, self._dim)
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

    async def list_documents(self, limit: int = 2000) -> list[Document]:
        """Enumerate this scope's chunks straight from Postgres — no query,
        no embedding needed. This is what lets the voicebot's one-shot KB
        dump see chunks ingested by a *different* process/worker than the one
        serving the current call (unlike ``HybridRetriever.list_all``, which
        only sees chunks this process indexed into its in-memory BM25)."""
        pool = await self._pool()
        scope_clause, scope_params = self._scope_clause(1)
        limit_pos = 1 + len(scope_params)
        # ORDER BY id: without it Postgres is free to return an arbitrary
        # subset when LIMIT truncates, and free to order it differently on
        # each call. That's silent for a small corpus (nothing gets cut) but
        # breaks reproducibility once row count exceeds `limit` -- two
        # identical benchmark sweep runs could hydrate the BM25 arm with
        # different chunks while the dense arm (which has no such limit at
        # query time) sees everything, so the two arms drift apart between
        # runs for reasons that have nothing to do with the config being
        # swept. Ordering by filename/section instead would also work but id
        # is stable and cheap (no extra predicate on jsonb metadata); callers
        # that care about document order (src/rag/context_builder.py's
        # voicebot KB dump) already re-sort by (filename, section) themselves
        # before use, so this ordering choice is invisible to them.
        sql = f"""
            SELECT id, content, metadata
            FROM voicebot.knowledge_chunks
            WHERE {scope_clause}
            ORDER BY id
            LIMIT ${limit_pos}::integer
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *scope_params, limit)
        out: list[Document] = []
        for r in rows:
            raw = r["metadata"]
            meta = json.loads(raw) if isinstance(raw, str) else (dict(raw) if raw else {})
            out.append(Document(id=r["id"], content=r["content"], metadata=meta))
        return out
