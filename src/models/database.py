"""SQLAlchemy 2.x async engine + session factory.

Engines are created lazily so tests can supply their own URL (e.g. an
in-memory SQLite) without booting postgres.

**Schema namespacing.** All our tables live under a single schema (default
``voicebot``, configurable via ``VOX_DB_SCHEMA``) inside whatever database the
URL points at — so we never need a database dedicated to us. The ORM models
declare no schema; instead every Postgres connection sets ``search_path`` to our
schema (via asyncpg ``server_settings``), so unqualified table names resolve to
it for both DML and DDL. SQLite (tests) has no schemas, so the search_path is
skipped there and tables stay in the default — test fixtures need no changes.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.schema import CreateSchema


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def get_schema(url: Optional[str] = None) -> Optional[str]:
    """The schema our tables live under — None on SQLite or when unconfigured."""
    from src.config import get_settings

    settings = get_settings()
    url = url or settings.database.url
    if _is_sqlite(url):
        return None
    return getattr(settings.database, "db_schema", None) or None


def _quote_ident(name: str) -> str:
    """Double-quote an identifier for use in a search_path value (so a schema
    name with special chars, e.g. a hyphen, still works). Inner quotes escaped."""
    return '"' + name.replace('"', '""') + '"'


def search_path_connect_args(url: str) -> dict:
    """asyncpg connect_args that pin search_path to our schema (+ public for
    shared types/extensions). Empty for SQLite / unconfigured.

    statement_cache_size=0 disables asyncpg's prepared-statement cache so
    queries work correctly through PgBouncer in transaction-pooling mode."""
    schema = get_schema(url)
    if _is_sqlite(url):
        return {}
    args: dict = {"statement_cache_size": 0}
    if schema:
        args["server_settings"] = {"search_path": f"{_quote_ident(schema)},public"}
    return args


def get_engine(url: Optional[str] = None) -> AsyncEngine:
    """Return the process-wide async engine, creating it on first call."""
    global _engine, _sessionmaker
    if _engine is None:
        if url is None:
            from src.config import get_settings

            url = get_settings().database.url
        _engine = create_async_engine(
            url, future=True, pool_pre_ping=True,
            connect_args=search_path_connect_args(url),
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def ensure_schema(url: Optional[str] = None) -> None:
    """Create our schema if it doesn't exist (no-op on SQLite). Idempotent.

    Checks for the schema first and only issues CREATE when it's actually
    missing — a least-privilege DB user may lack CREATE on the database (and
    Postgres rejects ``CREATE SCHEMA IF NOT EXISTS`` on a permission check even
    when the schema already exists). If it's missing and we can't create it, the
    error surfaces so the deploy can grant rights or pre-create the schema.
    """
    from sqlalchemy import text

    from src.config import get_settings

    url = url or get_settings().database.url
    schema = get_schema(url)
    if not schema:
        return
    engine = get_engine(url)
    async with engine.begin() as conn:
        def _ensure(sync_conn) -> None:
            # pg_namespace (not information_schema.schemata, which is filtered by
            # the caller's privileges) so a least-privilege user that already has
            # access to an existing schema doesn't trigger a CREATE attempt.
            exists = sync_conn.execute(
                text("SELECT 1 FROM pg_namespace WHERE nspname = :s"),
                {"s": schema},
            ).first()
            if exists:
                return
            # CreateSchema isn't a schema-qualified table ref, so the translate
            # map doesn't touch it — emits CREATE SCHEMA with the real name.
            sync_conn.execute(CreateSchema(schema))
        await conn.run_sync(_ensure)


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        get_engine()
    assert _sessionmaker is not None  # for type checker
    return _sessionmaker


async def dispose_engine() -> None:
    """Close the engine; used on FastAPI shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


def reset_engine_for_tests(url: str) -> AsyncEngine:
    """Reinitialize the engine against a fresh URL (test fixture helper)."""
    global _engine, _sessionmaker
    _engine = create_async_engine(
        url, future=True, connect_args=search_path_connect_args(url))
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine
