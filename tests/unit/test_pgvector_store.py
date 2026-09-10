"""Unit tests for the pgvector store's DSN sanitization.

_to_asyncpg_dsn feeds a raw DATABASE_URL (read directly from the env, not
through src.config.normalize_db_url) into asyncpg.create_pool. Managed
providers (Neon, ...) append channel_binding=require to connection strings;
asyncpg's own DSN parser has no handler for it, so it falls into
server_settings and Postgres rejects it as an unrecognized configuration
parameter — a second, differently-shaped break from the same env var that
crashes SQLAlchemy's asyncpg dialect at startup.
"""

from __future__ import annotations

import pytest

from src.providers.vector_store.pgvector_store import PGVectorAdapter, _to_asyncpg_dsn


def test_to_asyncpg_dsn_strips_channel_binding():
    out = _to_asyncpg_dsn(
        "postgresql://u:p@h:5432/db?channel_binding=require&sslmode=require")
    assert "channel_binding" not in out
    assert "sslmode=require" in out  # asyncpg's own DSN parser handles this


def test_to_asyncpg_dsn_converts_sqlalchemy_scheme():
    out = _to_asyncpg_dsn("postgresql+asyncpg://u:p@h/db?channel_binding=require")
    assert out.startswith("postgresql://")
    assert "channel_binding" not in out


def test_to_asyncpg_dsn_passthrough_when_no_libpq_params():
    url = "postgresql://u:p@h/db"
    assert _to_asyncpg_dsn(url) == url


# --- list_documents ordering (Finding 7) -----------------------------------
#
# list_documents had no ORDER BY, just LIMIT -- past the cap, Postgres is
# free to return an arbitrary subset AND order it differently between calls,
# which silently desyncs the BM25 arm (hydrated from this) from the dense
# arm (queried fresh each time, no cap) across two otherwise-identical sweep
# runs. These tests fake the asyncpg pool/connection so they exercise the
# adapter's SQL-building without a live database.


class _FakeConn:
    def __init__(self, pool: "_FakePool") -> None:
        self._pool = pool

    async def fetch(self, sql, *params):
        self._pool.captured_sql = sql
        self._pool.captured_params = params
        return self._pool.rows


class _FakeAcquireCtx:
    def __init__(self, pool: "_FakePool") -> None:
        self._pool = pool

    async def __aenter__(self):
        return _FakeConn(self._pool)

    async def __aexit__(self, *exc_info):
        return False


class _FakePool:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.captured_sql: str = ""
        self.captured_params: tuple = ()

    def acquire(self):
        return _FakeAcquireCtx(self)


def _adapter_with_fake_pool(rows: list[dict]) -> tuple[PGVectorAdapter, _FakePool]:
    adapter = PGVectorAdapter({
        "embedding_dim": 8,
        "tenant_id": "tenant-1",
        "database_url": "postgresql://u:p@h/db",
    })
    fake_pool = _FakePool(rows)

    async def _fake_pool_getter():
        return fake_pool

    adapter._pool = _fake_pool_getter  # type: ignore[method-assign]
    return adapter, fake_pool


@pytest.mark.asyncio
async def test_list_documents_sql_orders_by_id_for_determinism() -> None:
    adapter, fake_pool = _adapter_with_fake_pool(
        rows=[{"id": "b", "content": "B", "metadata": "{}"}, {"id": "a", "content": "A", "metadata": "{}"}]
    )
    await adapter.list_documents(limit=10)
    assert "ORDER BY id" in fake_pool.captured_sql, (
        "without a deterministic ORDER BY, LIMIT can silently truncate/reorder "
        "past the cap and desync BM25 hydration from the dense arm between runs"
    )


@pytest.mark.asyncio
async def test_list_documents_returns_docs_in_the_order_the_ordered_query_produced() -> None:
    # The adapter itself doesn't re-sort client-side -- it trusts the SQL's
    # ORDER BY. Two calls returning rows in the same (server-sorted) order
    # must map to the same document order both times.
    rows = [{"id": "a", "content": "A", "metadata": "{}"}, {"id": "b", "content": "B", "metadata": "{}"}]
    adapter, _ = _adapter_with_fake_pool(rows)
    first = await adapter.list_documents(limit=10)
    second = await adapter.list_documents(limit=10)
    assert [d.id for d in first] == [d.id for d in second] == ["a", "b"]
