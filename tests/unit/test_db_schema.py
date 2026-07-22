"""Unit tests for DB URL normalization + schema search_path wiring."""

from __future__ import annotations

from src.config import normalize_db_url, strip_libpq_only_query_params
from src.models.database import get_schema, search_path_connect_args


def test_normalize_adds_asyncpg_driver():
    assert normalize_db_url("postgresql://u:p@h:5432/db").startswith("postgresql+asyncpg://")
    assert normalize_db_url("postgres://u:p@h/db").startswith("postgresql+asyncpg://")


def test_normalize_converts_sslmode_to_ssl():
    out = normalize_db_url("postgresql://u:p@h:29203/db?sslmode=require")
    assert out.startswith("postgresql+asyncpg://")
    assert "ssl=require" in out
    assert "sslmode" not in out


def test_normalize_drops_sslmode_disable():
    out = normalize_db_url("postgresql://u:p@h/db?sslmode=disable")
    assert "ssl=" not in out and "sslmode" not in out


def test_normalize_drops_channel_binding():
    # Neon (and some other managed Postgres providers) hand out connection
    # strings with channel_binding=require — a libpq/psql-only SCRAM setting.
    # asyncpg.connect() has no such parameter and raises TypeError on it.
    out = normalize_db_url("postgresql://u:p@h/db?channel_binding=require")
    assert "channel_binding" not in out


def test_normalize_drops_channel_binding_alongside_sslmode():
    out = normalize_db_url(
        "postgresql://u:p@h:29203/db?channel_binding=require&sslmode=require")
    assert "channel_binding" not in out
    assert "ssl=require" in out
    assert "sslmode" not in out


def test_strip_libpq_only_drops_channel_binding_only():
    # Raw-DSN consumers (asyncpg.create_pool) don't need the sslmode->ssl
    # rewrite — asyncpg's own DSN parser already understands sslmode — but
    # channel_binding has no such handling and must still be stripped.
    out = strip_libpq_only_query_params(
        "postgresql://u:p@h:5432/db?channel_binding=require&sslmode=require")
    assert "channel_binding" not in out
    assert "sslmode=require" in out
    assert "ssl=" not in out  # no scheme-style translation here


def test_strip_libpq_only_passthrough_when_absent():
    url = "postgresql://u:p@h/db?sslmode=require"
    assert strip_libpq_only_query_params(url) == url


def test_normalize_passthrough_sqlite_and_asyncpg():
    assert normalize_db_url("sqlite+aiosqlite:///:memory:") == "sqlite+aiosqlite:///:memory:"
    already = "postgresql+asyncpg://vox:vox@localhost/vox_agent"
    assert normalize_db_url(already) == already


def test_get_schema_none_for_sqlite():
    assert get_schema("sqlite+aiosqlite:///:memory:") is None


def test_search_path_connect_args_sqlite_empty():
    assert search_path_connect_args("sqlite+aiosqlite:///:memory:") == {}


def test_search_path_connect_args_postgres_quotes_schema():
    args = search_path_connect_args("postgresql+asyncpg://u:p@h/db")
    sp = args["server_settings"]["search_path"]
    # Schema is double-quoted (safe for any name, incl. special chars) + public.
    assert sp.endswith(",public")
    assert sp.startswith('"') and '"' in sp[1:]
    assert "voicebot" in sp
