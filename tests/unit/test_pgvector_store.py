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

from src.providers.vector_store.pgvector_store import _to_asyncpg_dsn


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
