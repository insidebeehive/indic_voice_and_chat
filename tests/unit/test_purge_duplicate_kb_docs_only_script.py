"""tests/unit/test_purge_duplicate_kb_docs_only_script.py

Tests for scripts/purge_duplicate_kb_docs_only.py -- the reduced-scope
sibling of scripts/purge_stale_kb_docs.py that runs ONLY targets (a)/(b)/(d)
(the true stale-'global_kb_'-id-scheme duplicates) and never touches
targets (c)/(e) (the relocated casino/sports/matka rows, which as of this
script's authoring have no opt-in-module replacement ingested yet -- see the
script's module docstring for the full rationale).

Coverage:
1. The script imports its query/delete functions directly from
   purge_stale_kb_docs rather than redefining any SQL -- asserted here by
   identity, so this test breaks loudly if that ever silently forks.
2. Dry-run / --execute control flow against a minimal FakeConn, proving:
   (a) a dry run issues zero DELETEs, (b) --execute issues exactly 3 DELETEs
   (a/b/d) and NEVER a relocated-query (c/e) SELECT or DELETE -- the whole
   point of this script's existence.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from scripts import purge_duplicate_kb_docs_only as dup_purge
from scripts import purge_stale_kb_docs as purge


# --- (1) reuses the real, already-tested query/delete functions ------------


def test_reuses_real_stale_query_and_delete_functions_by_identity() -> None:
    """Guards against this script ever drifting into re-deriving its own SQL
    -- it must call the EXACT SAME function objects purge_stale_kb_docs.py
    defines and purge_stale_kb_docs_script tests already cover."""
    assert dup_purge._select_stale_crm_kb is purge._select_stale_crm_kb
    assert dup_purge._select_stale_kb is purge._select_stale_kb
    assert dup_purge._select_stale_chunks_grouped is purge._select_stale_chunks_grouped
    assert dup_purge._delete_stale_crm_kb is purge._delete_stale_crm_kb
    assert dup_purge._delete_stale_kb is purge._delete_stale_kb
    assert dup_purge._delete_stale_chunks is purge._delete_stale_chunks


def test_never_imports_relocated_targets() -> None:
    """(c)/(e) -- the relocated-filename targets -- must never be reachable
    from this module at all, not merely unused."""
    for name in (
        "_select_relocated_crm_kb",
        "_delete_relocated_crm_kb",
        "_select_relocated_chunks_grouped",
        "_delete_relocated_chunks",
        "RELOCATED_DOCS_WHERE",
        "RELOCATED_CHUNKS_WHERE",
    ):
        assert not hasattr(dup_purge, name), (
            f"{name} must not be imported into purge_duplicate_kb_docs_only "
            "-- this script's entire purpose is to never touch the relocated "
            "targets"
        )


# --- (2) dry-run / --execute control flow -----------------------------------


class _NullAsyncCM:
    async def __aenter__(self) -> "_NullAsyncCM":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class FakeConn:
    """Minimal stand-in for an asyncpg.Connection. Only services the stale
    (a)/(b)/(d) query shapes -- if this script ever issues a relocated-query
    (c)/(e), _classify raises instead of silently misrouting, since that
    would defeat the whole point of this script."""

    def __init__(self, canned: dict[tuple[str, str], list[dict]], chunks_exist: bool = True):
        self.canned = canned
        self.chunks_exist = chunks_exist
        self.fetch_calls: list[str] = []
        self.executed_deletes: list[str] = []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if "information_schema.tables" in sql:
            return 1 if self.chunks_exist else None
        raise AssertionError(f"unexpected fetchval query: {sql}")

    async def fetch(self, sql: str, *args: Any) -> list[dict]:
        self.fetch_calls.append(sql)
        if sql.strip().startswith("DELETE"):
            self.executed_deletes.append(sql)
        if "filename = ANY" in sql or "filename' = ANY" in sql:
            raise AssertionError(
                f"purge_duplicate_kb_docs_only must never issue a relocated-"
                f"target (c/e) query, but got: {sql}"
            )
        if "crm_kb_documents" in sql:
            table = "crm_kb"
        elif "kb_documents" in sql:
            table = "kb"
        elif "knowledge_chunks" in sql:
            table = "chunks"
        else:
            raise AssertionError(f"unclassifiable query: {sql}")
        return self.canned.get((table, "stale"), [])

    def transaction(self) -> _NullAsyncCM:
        return _NullAsyncCM()

    async def close(self) -> None:
        pass


CANNED: dict[tuple[str, str], list[dict]] = {
    ("crm_kb", "stale"): [
        {"id": "global_kb_01-account-registration-login", "crm_id": "betstudio",
         "filename": "01-account-registration-login.md"},
        {"id": "global_kb_06-casino-games", "crm_id": "betstudio",
         "filename": "06-casino-games.md"},
    ],
    ("kb", "stale"): [],
    ("chunks", "stale"): [
        {"document_id": "global_kb_01-account-registration-login", "n": 2},
        {"document_id": "global_kb_06-casino-games", "n": 2},
    ],
}


def _args(*extra: str) -> Any:
    return dup_purge.parse_args(["--database-url", "postgresql://u:p@host/db",
                                  "--schema", "voicebot", *extra])


async def test_dry_run_reports_only_abd_and_issues_zero_deletes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    fake_conn = FakeConn(CANNED, chunks_exist=True)
    monkeypatch.setattr(dup_purge.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    rc = await dup_purge._run(_args())

    assert rc == 0
    assert fake_conn.executed_deletes == [], "dry run must never issue a DELETE"

    out = capsys.readouterr().out
    assert "TRUE DUPLICATES ONLY" in out
    assert "(a) crm_kb_documents -- stale 'global_kb_' id scheme: 2 row(s)" in out
    assert "(b) kb_documents (tenant tier) -- stale 'global_kb_' id scheme: 0 row(s)" in out
    assert "(d) knowledge_chunks -- stale 'global_kb_' id scheme: 4 chunk(s)" in out
    # (c)/(e) must never appear in this script's report at all.
    assert "(c)" not in out
    assert "(e)" not in out
    assert "relocated" not in out.lower() or "intentionally NOT touched" in out
    assert "Dry run complete -- no changes written" in out


async def test_execute_mode_issues_exactly_three_deletes_never_relocated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    fake_conn = FakeConn(CANNED, chunks_exist=True)
    monkeypatch.setattr(dup_purge.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    rc = await dup_purge._run(_args("--execute"))

    assert rc == 0
    # Exactly (a) crm_kb_documents, (b) kb_documents, (d) knowledge_chunks --
    # never a 4th/5th DELETE for the relocated targets (c)/(e).
    assert len(fake_conn.executed_deletes) == 3
    assert all(sql.strip().startswith("DELETE") for sql in fake_conn.executed_deletes)

    out = capsys.readouterr().out
    assert "EXECUTE (will delete rows)" in out
    assert "Executed --" in out
    assert "true duplicates only" in out.lower()


async def test_chunks_table_missing_skips_chunk_purge_without_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    fake_conn = FakeConn(CANNED, chunks_exist=False)
    monkeypatch.setattr(dup_purge.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    rc = await dup_purge._run(_args("--execute"))

    assert rc == 0
    # Only the 2 doc-level deletes (a, b) -- no chunk delete attempted.
    assert len(fake_conn.executed_deletes) == 2

    out = capsys.readouterr().out
    assert "knowledge_chunks table not found" in out
