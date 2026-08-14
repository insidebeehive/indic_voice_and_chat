"""tests/unit/test_purge_stale_kb_docs_script.py

Real tests for ``scripts/purge_stale_kb_docs.py``, replacing the placeholder
``test_noop``. That script talks to Postgres directly via ``asyncpg`` (not
the app's SQLAlchemy engine), and this test environment has no live Postgres
fixture (see tests/conftest.py: only aiosqlite is set up, and asyncpg's
POSIX-regex ``~``/``!~`` operators used throughout the script aren't
available in sqlite anyway). So coverage here is split two ways:

1. The id-namespace guard predicates (``STALE_ID_REGEX`` / ``SEEDER_ID_REGEX``
   / ``RELOCATED_DOCS_WHERE`` / ``RELOCATED_CHUNKS_WHERE``) are exercised
   directly with Python's ``re`` module against representative ids -- these
   constants use plain anchored regex syntax valid in both asyncpg and `re`.
2. The dry-run/--execute control flow in ``_run`` is exercised against a
   minimal fake asyncpg connection (``FakeConn`` below) that classifies each
   query by its table + shape and returns canned rows -- enough to prove (a)
   a dry run issues zero DELETEs and never calls ``_execute_deletes``, and
   (b) --execute does call it and the deletes it issues share the exact same
   WHERE-predicate shape as the preview.
"""
from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock

import pytest

from scripts import purge_stale_kb_docs as purge


# --- (1) id-namespace guard predicates ---------------------------------


@pytest.mark.parametrize(
    "doc_id,should_match",
    [
        ("crm_kb_betstudio_06-casino-games", True),
        ("global_kb_06-casino-games", True),
        ("crmdoc_abc123", False),
        ("crmdoc_abc123::chunk-0", False),
        ("platform_doc_x", False),
        # Postgres's `~` operator (which this regex is ALWAYS used via) is an
        # UNANCHORED search, unlike Python's re.match which only ever checks
        # position 0 regardless of a leading `^`. re.search correctly
        # emulates `~` here: it would find "global_kb_" or "crm_kb_" as a
        # substring anywhere in the id if the `^` anchor were ever dropped
        # from SEEDER_ID_REGEX. This crmdoc_* id (admin-uploaded, must never
        # match) embeds "global_kb_" mid-string specifically to prove the
        # anchor is load-bearing, not merely decorative.
        ("crmdoc_ab12_global_kb_note", False),
    ],
)
def test_seeder_id_regex_namespace(doc_id: str, should_match: bool) -> None:
    """SEEDER_ID_REGEX is the positive guard used in every filename-driven
    predicate: it must match the seeder's two id namespaces and NOT match
    admin-uploaded (crmdoc_*) or legacy platform (platform_doc_*) ids.

    Uses re.search (not re.match) to correctly emulate Postgres's `~`
    operator, which is an unanchored search -- re.match is implicitly
    anchored at position 0 regardless of whether the pattern itself has a
    leading `^`, so it cannot distinguish an anchored pattern from an
    accidentally-unanchored one the way re.search (and `~`) can.
    """
    assert bool(re.search(purge.SEEDER_ID_REGEX, doc_id)) is should_match


@pytest.mark.parametrize(
    "doc_id,should_match",
    [
        ("global_kb_06-casino-games", True),
        ("global_kb_06-casino-games::chunk-0", True),
        ("crm_kb_betstudio_06-casino-games", False),
        ("crmdoc_abc123", False),
        ("platform_doc_x", False),
        # The header comment explicitly warns STALE_ID_REGEX must be used via
        # ~ / !~, never LIKE 'global_kb_%' -- LIKE's '_' wildcard would also
        # match this. re.search on the anchored regex must NOT match it.
        ("globalxkb_not_a_real_match", False),
        # Same anchor-emulation rationale as SEEDER_ID_REGEX above: proves
        # re.search (unlike re.match) would catch an accidentally-dropped
        # `^` on STALE_ID_REGEX, since "global_kb_" appears mid-string here.
        ("crmdoc_ab12_global_kb_note", False),
    ],
)
def test_stale_id_regex_namespace(doc_id: str, should_match: bool) -> None:
    assert bool(re.search(purge.STALE_ID_REGEX, doc_id)) is should_match


def _relocated_docs_would_include(doc_id: str, filename: str) -> bool:
    """Mirrors RELOCATED_DOCS_WHERE's logic in Python. RELOCATED_DOCS_WHERE
    is a raw SQL string built from STALE_ID_REGEX/SEEDER_ID_REGEX via
    asyncpg's `~`/`!~` (unusable from sqlite/re directly against the SQL
    string), so this reimplements it against the same two constants -- the
    actual assertable seam the script's own structure offers.

    Uses re.search, not re.match, to correctly emulate Postgres's unanchored
    `~`/`!~` operators (see test_seeder_id_regex_namespace above for why
    re.match cannot detect a dropped `^` anchor)."""
    return (
        filename in purge.RELOCATED_FILENAMES
        and re.search(purge.STALE_ID_REGEX, doc_id) is None
        and re.search(purge.SEEDER_ID_REGEX, doc_id) is not None
    )


class TestAdminDocPreservationGuard:
    """(1) from the task brief: construct representative doc ids and verify
    the relocated-doc predicate includes the seeder's own namespace and
    excludes crmdoc_* / platform_doc_* ids that happen to share a relocated
    filename."""

    def test_admin_uploaded_doc_excluded_even_with_matching_filename(self) -> None:
        assert _relocated_docs_would_include("crmdoc_abc123", "06-casino-games.md") is False

    def test_legacy_platform_doc_excluded_even_with_matching_filename(self) -> None:
        assert _relocated_docs_would_include("platform_doc_x", "06-casino-games.md") is False

    def test_current_seeder_namespace_included(self) -> None:
        assert _relocated_docs_would_include(
            "crm_kb_betstudio_06-casino-games", "06-casino-games.md"
        ) is True

    def test_stale_legacy_namespace_excluded_from_relocated_predicate(self) -> None:
        """A global_kb_* id for a relocated filename is covered by the
        SEPARATE stale-id-scheme predicate (a), not (c) -- excluded here so
        it's never reported/deleted twice."""
        assert _relocated_docs_would_include(
            "global_kb_06-casino-games", "06-casino-games.md"
        ) is False

    def test_unrelated_filename_excluded_regardless_of_id(self) -> None:
        assert _relocated_docs_would_include(
            "crm_kb_betstudio_unrelated-doc", "unrelated-doc.md"
        ) is False

    def test_relocated_chunks_where_requires_crm_id_not_null(self) -> None:
        """The extra guard (beyond RELOCATED_DOCS_WHERE) that protects a
        tenant's own opted-in tenant_id-scoped chunks from a filename
        collision."""
        assert "crm_id IS NOT NULL" in purge.RELOCATED_CHUNKS_WHERE

    def test_where_clauses_share_identical_regex_constants(self) -> None:
        """The preview (SELECT) and delete (DELETE) queries interpolate the
        SAME RELOCATED_*_WHERE constants (see _select_relocated_crm_kb /
        _delete_relocated_crm_kb etc.), so the preview can never drift from
        what --execute actually deletes."""
        for where in (purge.RELOCATED_DOCS_WHERE, purge.RELOCATED_CHUNKS_WHERE):
            assert purge.STALE_ID_REGEX in where
            assert purge.SEEDER_ID_REGEX in where

    def test_where_clauses_use_correct_include_exclude_polarity(self) -> None:
        """Bare substring presence (as in the test above) is polarity-blind:
        it can't tell `id ~ SEEDER_ID_REGEX` (include the seeder's own
        namespace) apart from `id !~ SEEDER_ID_REGEX` (exclude it) -- flipping
        `~` to `!~` on either operand inverts the entire admin-doc
        preservation guard (deletes exactly the wrong rows) yet leaves a bare
        `SEEDER_ID_REGEX in where` assertion passing. These assertions pin the
        EXACT operator each regex constant is used with, character-for-
        character against the real RELOCATED_DOCS_WHERE / RELOCATED_CHUNKS_WHERE
        strings built in scripts/purge_stale_kb_docs.py."""
        for where in (purge.RELOCATED_DOCS_WHERE, purge.RELOCATED_CHUNKS_WHERE):
            # Must EXCLUDE the stale global_kb_ id scheme (handled separately
            # by targets (a)/(b), never double-deleted here).
            assert f"id !~ '{purge.STALE_ID_REGEX}'" in where
            # Must INCLUDE only the seeder's own id namespaces -- this is
            # what keeps an admin-uploaded crmdoc_* doc that happens to share
            # a relocated filename from being deleted.
            assert f"id ~ '{purge.SEEDER_ID_REGEX}'" in where
            # And the inverted forms must NOT be present -- guards against a
            # future edit accidentally introducing the wrong operator
            # alongside (rather than instead of) the correct one.
            assert f"id ~ '{purge.STALE_ID_REGEX}'" not in where
            assert f"id !~ '{purge.SEEDER_ID_REGEX}'" not in where

        # crm_id IS NOT NULL is itself an inclusion guard (not a negated
        # one) -- pin its exact presence too, since it's the other half of
        # (e)'s "never touch a tenant's own opted-in chunks" guarantee.
        assert "crm_id IS NOT NULL" in purge.RELOCATED_CHUNKS_WHERE


# --- (2) dry-run / --execute control flow -------------------------------


class _NullAsyncCM:
    async def __aenter__(self) -> "_NullAsyncCM":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class FakeConn:
    """Minimal stand-in for an asyncpg.Connection.

    Canned rows are keyed by (table, kind), classified from the SQL text,
    and returned regardless of whether the call is the SELECT-side report
    query or the --execute-side DELETE ... RETURNING query for the SAME
    logical target (a/b/c doc-level queries share one key between their
    SELECT and DELETE forms, since they use an identical WHERE clause) --
    mirroring the script's own "preview always matches reality" guarantee.
    """

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
        table, kind = self._classify(sql)
        return self.canned.get((table, kind), [])

    @staticmethod
    def _classify(sql: str) -> tuple[str, str]:
        # _table_refs only quotes the schema (e.g. "voicebot".crm_kb_documents),
        # not the table name -- and "kb_documents" is itself a substring of
        # "crm_kb_documents", so the more specific name must be checked first.
        if "crm_kb_documents" in sql:
            table = "crm_kb"
        elif "kb_documents" in sql:
            table = "kb"
        elif "knowledge_chunks" in sql:
            table = "chunks"
        else:
            table = "?"
        # Order matters: _select_other_legacy_prefixes' WHERE contains BOTH
        # "split_part(id, '_'" and "filename = ANY", so the '_'-delimited
        # split_part check must be tried first to land it correctly.
        #
        # The two RELOCATED_*_WHERE constants spell their filename match
        # differently: RELOCATED_DOCS_WHERE (crm_kb/kb tables) uses plain
        # "filename = ANY(...)", but RELOCATED_CHUNKS_WHERE (chunks table)
        # uses "metadata->>'filename' = ANY(...)" -- note the apostrophe
        # immediately before " = ANY", which makes the literal substring
        # "filename = ANY" NOT present (it's "filename' = ANY" instead).
        # Probing for only the doc-table spelling silently misroutes every
        # relocated-chunks query (_select_relocated_chunks_grouped,
        # _delete_relocated_chunks) to the wrong canned fixture bucket, so
        # both spellings must be checked here.
        if "split_part(id, '_'" in sql:
            kind = "legacy_prefixes"
        elif "filename = ANY" in sql or "filename' = ANY" in sql:
            kind = "relocated"
        elif "split_part(id, '::'" in sql:
            kind = "grouped"
        else:
            kind = "stale"
        return table, kind

    def transaction(self) -> _NullAsyncCM:
        return _NullAsyncCM()

    async def close(self) -> None:
        pass


CANNED: dict[tuple[str, str], list[dict]] = {
    ("crm_kb", "stale"): [
        {"id": "global_kb_09-old-doc", "crm_id": "betstudio", "filename": "09-old-doc.md"},
    ],
    ("kb", "stale"): [],
    ("crm_kb", "relocated"): [
        {"id": "crm_kb_betstudio_06-casino-games", "crm_id": "betstudio",
         "filename": "06-casino-games.md"},
    ],
    ("crm_kb", "legacy_prefixes"): [
        {"prefix": "platform", "n": 1, "sample_id": "platform_doc_x"},
    ],
    ("chunks", "grouped"): [
        {"document_id": "global_kb_09-old-doc", "n": 2},
    ],
    ("chunks", "stale"): [
        {"id": "global_kb_09-old-doc::chunk-0"},
        {"id": "global_kb_09-old-doc::chunk-1"},
    ],
    ("chunks", "relocated"): [
        {"document_id": "crm_kb_betstudio_06-casino-games", "n": 1,
         "id": "crm_kb_betstudio_06-casino-games::chunk-0", "crm_id": "betstudio"},
    ],
}


async def test_classify_routes_relocated_chunk_queries_to_relocated_fixture() -> None:
    """Regression test for a FakeConn._classify probe-string mismatch: the
    real RELOCATED_CHUNKS_WHERE spells its filename match
    "metadata->>'filename' = ANY(...)" (note the apostrophe before " = ANY"),
    which does NOT contain the literal substring "filename = ANY" that the
    classifier used to probe for exclusively. That made
    _select_relocated_chunks_grouped fall through to the "grouped" bucket
    (misrouted to the STALE chunks fixture) and _delete_relocated_chunks fall
    all the way through to the default "stale" bucket -- so
    CANNED[("chunks", "relocated")] was dead code, never actually exercised.

    Calls the real query functions (not a hand-rolled SQL string) against a
    FakeConn, so this fails again if the classifier's probe ever drifts from
    the real predicate spelling.
    """
    fake_conn = FakeConn(CANNED, chunks_exist=True)
    chunks_table = '"voicebot".knowledge_chunks'

    grouped_rows = await purge._select_relocated_chunks_grouped(fake_conn, chunks_table)
    assert grouped_rows == CANNED[("chunks", "relocated")]
    assert grouped_rows != CANNED[("chunks", "grouped")]  # the stale fixture -- must not leak in here

    deleted_rows = await purge._delete_relocated_chunks(fake_conn, chunks_table)
    assert deleted_rows == CANNED[("chunks", "relocated")]
    assert deleted_rows != CANNED[("chunks", "stale")]  # the default/fallback fixture -- must not leak in here

    # The stale-chunks path (a genuinely different query/predicate) must
    # still land in its own separate bucket, unaffected by this fix.
    stale_grouped_rows = await purge._select_stale_chunks_grouped(fake_conn, chunks_table)
    assert stale_grouped_rows == CANNED[("chunks", "grouped")]


def _args(*extra: str) -> Any:
    return purge.parse_args(["--database-url", "postgresql://u:p@host/db",
                              "--schema", "voicebot", *extra])


async def test_dry_run_reports_correct_counts_and_issues_zero_deletes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """(3) Happy path: dry-run reports the rows that WOULD be deleted, with
    correct counts, without executing anything."""
    fake_conn = FakeConn(CANNED, chunks_exist=True)
    monkeypatch.setattr(purge.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    rc = await purge._run(_args())

    assert rc == 0
    assert fake_conn.executed_deletes == [], "dry run must never issue a DELETE"

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "(a) crm_kb_documents -- stale 'global_kb_' id scheme: 1 row(s)" in out
    assert "global_kb_09-old-doc" in out
    assert "(b) kb_documents (tenant tier) -- stale 'global_kb_' id scheme: 0 row(s)" in out
    assert "(c) crm_kb_documents -- relocated filenames (all CRMs): 1 row(s)" in out
    assert "platform_*: 1 row(s)  (e.g. platform_doc_x)" in out
    assert "Dry run complete -- no changes written" in out


async def test_execute_deletes_never_called_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(2) Dry-run writes nothing: mock-based proof that the function which
    issues actual DELETE/writes is never invoked when --execute is absent."""
    fake_conn = FakeConn(CANNED, chunks_exist=True)
    monkeypatch.setattr(purge.asyncpg, "connect", AsyncMock(return_value=fake_conn))
    execute_deletes_mock = AsyncMock(wraps=purge._execute_deletes)
    monkeypatch.setattr(purge, "_execute_deletes", execute_deletes_mock)

    rc = await purge._run(_args())

    assert rc == 0
    execute_deletes_mock.assert_not_called()


async def test_execute_mode_calls_execute_deletes_and_issues_deletes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Mirror of the dry-run test: with --execute, the delete function IS
    called, and real DELETE statements are issued against the connection."""
    fake_conn = FakeConn(CANNED, chunks_exist=True)
    monkeypatch.setattr(purge.asyncpg, "connect", AsyncMock(return_value=fake_conn))
    execute_deletes_mock = AsyncMock(wraps=purge._execute_deletes)
    monkeypatch.setattr(purge, "_execute_deletes", execute_deletes_mock)

    rc = await purge._run(_args("--execute"))

    assert rc == 0
    execute_deletes_mock.assert_called_once()
    assert len(fake_conn.executed_deletes) == 5, (
        "expected one DELETE each for a/b/c doc-level targets plus d/e "
        "chunk-level targets"
    )
    assert all(sql.strip().startswith("DELETE") for sql in fake_conn.executed_deletes)

    out = capsys.readouterr().out
    assert "EXECUTE (will delete rows)" in out
    assert "Executed --" in out
    assert "total row(s) deleted" in out


async def test_chunks_table_missing_skips_chunk_purge_without_failing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """FAISS-based local/dev DBs may have no voicebot.knowledge_chunks table
    at all -- that must be skipped, not treated as a failure."""
    fake_conn = FakeConn(CANNED, chunks_exist=False)
    monkeypatch.setattr(purge.asyncpg, "connect", AsyncMock(return_value=fake_conn))

    rc = await purge._run(_args("--execute"))

    assert rc == 0
    # Only the 3 doc-level deletes (a, b, c) -- no chunk deletes attempted.
    assert len(fake_conn.executed_deletes) == 3

    out = capsys.readouterr().out
    assert "knowledge_chunks table not found -- skipping chunk purge" in out
