"""Purge stale/orphaned KB documents and their pgvector chunks.

This is primarily a MANUALLY-RUN cleanup tool, NOT an Alembic migration — the
repo's Dockerfile auto-runs ``alembic upgrade head`` on every container
start, and the deletes this script performs are irreversible. Running this
script (dry-run first, then ``--execute``) is the recommended first step on
any given deployment, so an operator reviews exactly what would be deleted
before anything automatic ever touches it.

Separately, ``_seed_crm_kb``'s reconcile step in ``src/main.py`` (the app's
own startup KB seeder) implements the same "stale seeder-owned id, no longer
backed by a file on disk" logic as targets (a)/(c) below, scoped to
``crm_kb_documents`` + its pgvector chunks. That reconcile step MAY now
auto-prune on every boot, but ONLY when the ``VOX_KB_AUTO_PRUNE`` env var is
explicitly set to ``1`` — it defaults OFF, so nothing is deleted
automatically until an operator has reviewed this script's dry-run output on
their own deployment and either runs this script with ``--execute`` or
enables ``VOX_KB_AUTO_PRUNE`` for future deploys. This script itself is never
wired into any automatic runner regardless of that flag.

Two KB tiers exist: CRM-wide (``crm_kb_documents`` / ``CrmKBDocument``) and
tenant-level (``kb_documents`` / ``KBDocument``), both under the ``voicebot``
Postgres schema. An old migration (0010_crm_kb_documents) renamed
``platform_kb_documents`` -> ``crm_kb_documents`` and backfilled ``crm_id``,
but left pre-existing rows under a stale id scheme (``global_kb_<stem>``) that
the current seeder (``_seed_crm_kb`` in ``src/main.py``) never cleans up —
it writes fresh rows as ``crm_kb_<crm_id>_<stem>`` instead. Separately, six
files have been relocated out of the CRM-wide tier (removed from disk); their
``crm_kb_documents`` rows, under any crm_id, are now orphaned and are purged
here too — but a tenant's own opted-in copy in ``kb_documents`` is NEVER
touched by filename, only by the stale id prefix.

Chunks for both tiers live in the raw ``voicebot.knowledge_chunks`` table
(not a SQLAlchemy model — created once, by hand, per docs/pgvector_setup.sql;
it may not exist at all in FAISS-based local/dev setups, in which case the
chunk-purge steps are skipped, not failed).

Deletion targets (each run as a SELECT for the report, then — in --execute
mode only — as a DELETE ... RETURNING sharing the identical WHERE predicate,
so the preview always matches reality):

  (a) crm_kb_documents rows under the stale ``global_kb_`` id scheme.
  (b) kb_documents rows under the stale ``global_kb_`` id scheme (id-prefix
      only — never filtered by filename; expected to typically be empty).
  (c) crm_kb_documents rows for the 6 relocated filenames, any crm_id --
      restricted to the seeder's own id namespaces (crm_kb_ / global_kb_), so
      an admin-uploaded doc (crmdoc_* id, POST /crms/{id}/kb/ingest) that
      happens to share one of these filenames is NEVER deleted.
  (d) knowledge_chunks rows for (a).
  (e) knowledge_chunks rows for (c), scoped to crm_id IS NOT NULL so a
      tenant's own opted-in copy (tenant_id-scoped chunks) is never touched --
      and to the same crm_kb_/global_kb_ id namespaces as (c); chunk ids are
      '{doc_id}::chunk-{n}', so the namespace is filterable on the chunk id
      itself. A crmdoc_* upload's chunks carry a filename in metadata too,
      and would otherwise match.

Also reports (read-only, never deletes) any OTHER crm_kb_documents id prefix
that looks like a legacy platform-era id — out of scope for this cleanup,
flagged for a human to look at separately.

Defaults to a dry run (report only, zero writes). Pass --execute to actually
delete. The resolved target (host/db, masked password) and mode are always
printed before anything else, because the default DB target is whatever
.env/settings resolve to — which may be production.

Usage:
    python scripts/purge_stale_kb_docs.py                    # dry run (default)
    python scripts/purge_stale_kb_docs.py --execute           # actually delete
    python scripts/purge_stale_kb_docs.py --database-url postgresql://...
    python scripts/purge_stale_kb_docs.py --schema voicebot

Must be run from the repo root (settings/.env resolution is CWD-relative,
same as every other script under scripts/).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg  # type: ignore[import-not-found]

from src.config import get_settings, strip_libpq_only_query_params

# The 6 files relocated out of the CRM-wide KB tier (already off disk); their
# crm_kb_documents rows, under any crm_id, are now orphaned.
RELOCATED_FILENAMES = [
    "06-casino-games.md",
    "07-sports-betting.md",
    "08-matka-lottery-games.md",
    "ui-05-casino.md",
    "ui-06-sports.md",
    "ui-07-matka-lottery.md",
]

# Anchored regex for the stale pre-crm_kb id scheme. Always used via ~ / !~ —
# NEVER via LIKE 'global_kb_%', since LIKE's '_' is a wildcard and would also
# match e.g. 'globalxkb_...'.
STALE_ID_REGEX = "^global_kb_"

# Anchored regex for the ONLY two id namespaces the seeder has ever written
# under: current (`crm_kb_<crm_id>_<stem>`) and legacy pre-migration
# (`global_kb_<stem>`) — mirrors `seeder_prefixes` in _seed_crm_kb
# (src/main.py). This is a POSITIVE guard, required on every filename-driven
# predicate below: an admin-uploaded doc gets a `crmdoc_*` id (see POST
# /crms/{crm_id}/kb/ingest in src/api/crm_kb.py) and may legitimately share a
# filename with one of the 6 relocated files. Without this guard such a doc —
# real admin content, not seeded/stale content — would be silently deleted.
SEEDER_ID_REGEX = "^(crm_kb_|global_kb_)"

# The filename-driven predicates, defined ONCE each and interpolated into both
# the dry-run SELECT and the --execute DELETE, so the preview can never drift
# from what actually gets deleted. $1 is RELOCATED_FILENAMES (text[]).
RELOCATED_DOCS_WHERE = (
    f"filename = ANY($1::text[]) "
    f"AND id !~ '{STALE_ID_REGEX}' "
    f"AND id ~ '{SEEDER_ID_REGEX}'"
)
# crm_id IS NOT NULL is essential: it's what guarantees a tenant's own opted-in
# copy (tenant_id-scoped chunks) is never touched. The id ~ SEEDER_ID_REGEX
# guard is the doc-namespace half — chunk ids are '{doc_id}::chunk-{n}' in
# every writer (src/main.py, src/api/crm_kb.py, src/api/knowledge.py), so the
# chunk id carries its document's namespace prefix. metadata->>'filename' is
# populated for crmdoc_* admin uploads too, which is exactly why the filename
# match alone is not safe here.
RELOCATED_CHUNKS_WHERE = (
    f"crm_id IS NOT NULL "
    f"AND metadata->>'filename' = ANY($1::text[]) "
    f"AND id !~ '{STALE_ID_REGEX}' "
    f"AND id ~ '{SEEDER_ID_REGEX}'"
)


# --- DSN / target resolution ------------------------------------------------


def _resolve_database_url(cli_value: Optional[str]) -> str:
    """--database-url > $DATABASE_URL > src.config.get_settings(). Raises
    RuntimeError with a clear message if none of the three are available."""
    if cli_value:
        return cli_value
    env_value = os.environ.get("DATABASE_URL")
    if env_value:
        return env_value
    try:
        settings = get_settings()
    except Exception as exc:  # pragma: no cover - depends on local .env/CWD
        raise RuntimeError(
            "no --database-url given, $DATABASE_URL not set, and "
            f"src.config.get_settings() failed to load ({exc!r}). Pass "
            "--database-url explicitly, or run this script from the repo "
            "root with a working .env / config/default.yaml."
        ) from exc
    return settings.database.url


def _build_dsn(raw_url: str) -> str:
    """Convert any accepted Postgres URL form (postgres://, postgresql://,
    postgresql+asyncpg://, with or without libpq-only query params like
    sslmode/channel_binding) into a plain postgresql:// DSN asyncpg.connect()
    accepts.

    Deliberately does NOT call src.config.normalize_db_url — that helper
    rewrites libpq's ``sslmode=`` into ``ssl=``, which is correct for a
    SQLAlchemy engine URL (where ``ssl`` is a recognized asyncpg
    connect-kwarg name) but wrong here: this function builds a DSN string
    for ``asyncpg.connect()`` directly, and asyncpg's own DSN parser
    recognizes the libpq keyword ``sslmode``, not ``ssl``. An ``ssl=`` key
    in the DSN's query string is not understood by that parser, so it
    silently falls back to ``sslmode=prefer`` (TLS becomes optional) while
    passing ``ssl`` through as a stray server_setting.

    raw_url may arrive in either form: ``get_settings()`` already runs
    normalize_db_url internally (via DatabaseConfig's field validator), so
    the settings path hands us ``ssl=``, while ``--database-url``/
    ``$DATABASE_URL`` bypass that and hand us the original ``sslmode=``
    straight from .env. Handle both by normalizing whichever is present
    back to ``sslmode=``.

    Mirrors src.providers.vector_store.pgvector_store._to_asyncpg_dsn, plus
    the ssl->sslmode normalization that helper doesn't need (it never sees
    a normalize_db_url-processed URL).
    """
    plain = raw_url
    if plain.startswith("postgresql+asyncpg://"):
        plain = "postgresql://" + plain[len("postgresql+asyncpg://"):]
    elif plain.startswith("postgres://"):
        plain = "postgresql://" + plain[len("postgres://"):]
    plain = strip_libpq_only_query_params(plain)

    parts = urlsplit(plain)
    if not parts.query:
        return plain
    params = parse_qsl(parts.query, keep_blank_values=True)
    out = [("sslmode", v) if k == "ssl" else (k, v) for k, v in params]
    return urlunsplit(parts._replace(query=urlencode(out)))


def _masked_target(dsn: str) -> str:
    """host/db (and user) for the resolved DSN, password never included."""
    parts = urlsplit(dsn)
    host = parts.hostname or "?"
    port = f":{parts.port}" if parts.port else ""
    db = parts.path.lstrip("/") or "?"
    user = parts.username or "?"
    return f"{user}@{host}{port}/{db}"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_refs(schema: str) -> tuple[str, str, str]:
    """Fully-qualified, schema-quoted table references. A bare asyncpg
    connection has no search_path set (unlike the app's SQLAlchemy engine),
    so every query must be schema-qualified explicitly."""
    q = _quote_ident(schema)
    return f"{q}.crm_kb_documents", f"{q}.kb_documents", f"{q}.knowledge_chunks"


# --- queries -----------------------------------------------------------------


async def _chunks_table_exists(conn: Any, schema: str) -> bool:
    row = await conn.fetchval(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = $1 AND table_name = 'knowledge_chunks'",
        schema,
    )
    return bool(row)


async def _select_stale_crm_kb(conn: Any, crm_kb: str):
    return await conn.fetch(
        f"SELECT id, crm_id, filename FROM {crm_kb} "
        f"WHERE id ~ '{STALE_ID_REGEX}' ORDER BY id"
    )


async def _delete_stale_crm_kb(conn: Any, crm_kb: str):
    return await conn.fetch(
        f"DELETE FROM {crm_kb} WHERE id ~ '{STALE_ID_REGEX}' "
        f"RETURNING id, crm_id, filename"
    )


async def _select_stale_kb(conn: Any, kb: str):
    return await conn.fetch(
        f"SELECT id, tenant_id, filename FROM {kb} "
        f"WHERE id ~ '{STALE_ID_REGEX}' ORDER BY id"
    )


async def _delete_stale_kb(conn: Any, kb: str):
    return await conn.fetch(
        f"DELETE FROM {kb} WHERE id ~ '{STALE_ID_REGEX}' "
        f"RETURNING id, tenant_id, filename"
    )


async def _select_relocated_crm_kb(conn: Any, crm_kb: str):
    return await conn.fetch(
        f"SELECT id, crm_id, filename FROM {crm_kb} "
        f"WHERE {RELOCATED_DOCS_WHERE} "
        f"ORDER BY id",
        RELOCATED_FILENAMES,
    )


async def _delete_relocated_crm_kb(conn: Any, crm_kb: str):
    return await conn.fetch(
        f"DELETE FROM {crm_kb} "
        f"WHERE {RELOCATED_DOCS_WHERE} "
        f"RETURNING id, crm_id, filename",
        RELOCATED_FILENAMES,
    )


async def _select_stale_chunks_grouped(conn: Any, chunks: str):
    """Grouped count by document id (chunk ids are '{doc_id}::chunk-{n}') —
    there can be many chunks per doc, so this is what gets reported/printed."""
    return await conn.fetch(
        f"SELECT split_part(id, '::', 1) AS document_id, count(*) AS n "
        f"FROM {chunks} WHERE id ~ '{STALE_ID_REGEX}' GROUP BY 1 ORDER BY 1"
    )


async def _delete_stale_chunks(conn: Any, chunks: str):
    return await conn.fetch(
        f"DELETE FROM {chunks} WHERE id ~ '{STALE_ID_REGEX}' RETURNING id"
    )


async def _select_relocated_chunks_grouped(conn: Any, chunks: str):
    return await conn.fetch(
        f"SELECT split_part(id, '::', 1) AS document_id, count(*) AS n "
        f"FROM {chunks} "
        f"WHERE {RELOCATED_CHUNKS_WHERE} "
        f"GROUP BY 1 ORDER BY 1",
        RELOCATED_FILENAMES,
    )


async def _delete_relocated_chunks(conn: Any, chunks: str):
    # Predicate shared verbatim with _select_relocated_chunks_grouped via
    # RELOCATED_CHUNKS_WHERE, so the preview can never drift from the delete.
    return await conn.fetch(
        f"DELETE FROM {chunks} "
        f"WHERE {RELOCATED_CHUNKS_WHERE} "
        f"RETURNING id, crm_id",
        RELOCATED_FILENAMES,
    )


async def _select_other_legacy_prefixes(conn: Any, crm_kb: str):
    """Informational only, never deleted: other crm_kb_documents id prefixes
    that aren't the stale global_kb_ scheme, aren't one of the 6 relocated
    filenames, and aren't the current live crm_kb_<crm_id>_<stem> scheme —
    e.g. a leftover platform_doc_... id. Flagged for manual human review.

    Groups by the FIRST id segment only. The old two-segment grouping
    (split_part(id,'_',1) || '_' || split_part(id,'_',2)) put every
    `crmdoc_<uuid-hex>` id in a group of its own — the uuid is segment 2 — so a
    CRM with N admin-uploaded docs produced N bogus 1-count "prefix groups"
    and drowned out the real signal. One `min(id)` sample per group is
    returned so a human still has a concrete row to go look at.
    """
    return await conn.fetch(
        f"SELECT split_part(id, '_', 1) AS prefix, count(*) AS n, "
        f"min(id) AS sample_id "
        f"FROM {crm_kb} "
        f"WHERE id !~ '{STALE_ID_REGEX}' "
        f"AND NOT (filename = ANY($1::text[])) "
        f"AND id !~ '^crm_kb_' "
        f"GROUP BY 1 ORDER BY 1",
        RELOCATED_FILENAMES,
    )


# --- report / execute --------------------------------------------------------


async def _build_report(conn: Any, crm_kb: str, kb: str, chunks: str, chunks_exist: bool) -> dict:
    report: dict = {
        "a_crm_kb_stale": await _select_stale_crm_kb(conn, crm_kb),
        "b_kb_stale": await _select_stale_kb(conn, kb),
        "c_crm_kb_relocated": await _select_relocated_crm_kb(conn, crm_kb),
        "other_legacy_prefixes": await _select_other_legacy_prefixes(conn, crm_kb),
    }
    if chunks_exist:
        report["d_chunks_stale_grouped"] = await _select_stale_chunks_grouped(conn, chunks)
        report["e_chunks_relocated_grouped"] = await _select_relocated_chunks_grouped(conn, chunks)
    else:
        report["d_chunks_stale_grouped"] = None
        report["e_chunks_relocated_grouped"] = None
    return report


def _print_report(report: dict, chunks_exist: bool) -> None:
    a = report["a_crm_kb_stale"]
    print(f"(a) crm_kb_documents -- stale 'global_kb_' id scheme: {len(a)} row(s)")
    for r in a:
        print(f"    {r['id']}  crm_id={r['crm_id']}  filename={r['filename']}")
    print()

    b = report["b_kb_stale"]
    print(f"(b) kb_documents (tenant tier) -- stale 'global_kb_' id scheme: {len(b)} row(s)")
    for r in b:
        print(f"    {r['id']}  tenant_id={r['tenant_id']}  filename={r['filename']}")
    print()

    c = report["c_crm_kb_relocated"]
    print(f"(c) crm_kb_documents -- relocated filenames (all CRMs): {len(c)} row(s)")
    for r in c:
        print(f"    {r['id']}  crm_id={r['crm_id']}  filename={r['filename']}")
    print()

    if not chunks_exist:
        print("(d) knowledge_chunks -- stale 'global_kb_' id scheme: skipped")
        print("(e) knowledge_chunks -- relocated filenames: skipped")
        print()
        print(
            "NOTE: voicebot.knowledge_chunks table not found -- skipping chunk "
            "purge (this DB may be FAISS-based local/dev; see "
            "docs/pgvector_setup.sql)."
        )
        print()
    else:
        d_grouped = report["d_chunks_stale_grouped"]
        d_total = sum(r["n"] for r in d_grouped)
        print(
            f"(d) knowledge_chunks -- stale 'global_kb_' id scheme: "
            f"{d_total} chunk(s) across {len(d_grouped)} document(s)"
        )
        for r in d_grouped:
            print(f"    {r['document_id']}: {r['n']} chunk(s)")
        print()

        e_grouped = report["e_chunks_relocated_grouped"]
        e_total = sum(r["n"] for r in e_grouped)
        print(
            f"(e) knowledge_chunks -- relocated filenames (crm-scoped only): "
            f"{e_total} chunk(s) across {len(e_grouped)} document(s)"
        )
        for r in e_grouped:
            print(f"    {r['document_id']}: {r['n']} chunk(s)")
        print()

    other = report["other_legacy_prefixes"]
    print(
        f"(info) other legacy crm_kb_documents id prefixes -- not deleted, "
        f"out of scope for this cleanup: {len(other)} prefix group(s)"
    )
    for r in other:
        print(f"    {r['prefix']}_*: {r['n']} row(s)  (e.g. {r['sample_id']})")
    print()


async def _execute_deletes(conn: Any, crm_kb: str, kb: str, chunks: str, chunks_exist: bool) -> dict:
    deleted: dict = {}
    async with conn.transaction():
        deleted["a_crm_kb_stale"] = await _delete_stale_crm_kb(conn, crm_kb)
        deleted["b_kb_stale"] = await _delete_stale_kb(conn, kb)
        deleted["c_crm_kb_relocated"] = await _delete_relocated_crm_kb(conn, crm_kb)
        if chunks_exist:
            deleted["d_chunks_stale"] = await _delete_stale_chunks(conn, chunks)
            deleted["e_chunks_relocated"] = await _delete_relocated_chunks(conn, chunks)
        else:
            deleted["d_chunks_stale"] = None
            deleted["e_chunks_relocated"] = None
    return deleted


def _print_deleted_summary(deleted: dict, chunks_exist: bool) -> int:
    print("Deleted:")
    print(f"  (a) crm_kb_documents (stale id scheme):  {len(deleted['a_crm_kb_stale'])} row(s)")
    print(f"  (b) kb_documents (stale id scheme):      {len(deleted['b_kb_stale'])} row(s)")
    print(f"  (c) crm_kb_documents (relocated files):  {len(deleted['c_crm_kb_relocated'])} row(s)")
    total = (
        len(deleted["a_crm_kb_stale"])
        + len(deleted["b_kb_stale"])
        + len(deleted["c_crm_kb_relocated"])
    )
    if chunks_exist:
        d_n = len(deleted["d_chunks_stale"])
        e_n = len(deleted["e_chunks_relocated"])
        print(f"  (d) knowledge_chunks (stale id scheme):  {d_n} row(s)")
        print(f"  (e) knowledge_chunks (relocated files):  {e_n} row(s)")
        total += d_n + e_n
    else:
        print("  (d) knowledge_chunks (stale id scheme):  skipped")
        print("  (e) knowledge_chunks (relocated files):  skipped")
    print()
    return total


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL to connect to. Default: $DATABASE_URL, else "
        "resolved via src.config.get_settings() (which reads .env / "
        "config/default.yaml — CWD-relative, so run from the repo root).",
    )
    parser.add_argument(
        "--schema",
        default=os.environ.get("VOX_DB_SCHEMA", "voicebot"),
        help="Postgres schema the app's tables live under (default: voicebot, "
        "or $VOX_DB_SCHEMA if set).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Report only, zero writes (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually delete the rows identified by the report. Irreversible.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    execute = bool(args.execute)

    try:
        raw_url = _resolve_database_url(args.database_url)
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    dsn = _build_dsn(raw_url)
    schema = args.schema
    crm_kb, kb, chunks = _table_refs(schema)

    # Default target is whatever .env/settings resolve to, which may be
    # production -- always surface it, loudly, before doing anything.
    print(f"Target: {_masked_target(dsn)}")
    print(f"Schema: {schema}")
    print(f"Mode:   {'EXECUTE (will delete rows)' if execute else 'DRY RUN (no writes)'}")
    print()

    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:
        print(f"FATAL: could not connect to database: {exc}", file=sys.stderr)
        return 1

    try:
        # Existence check runs before any transaction is opened: a failed
        # statement poisons an asyncpg transaction, which would abort/lose
        # the doc-table deletes that ran before it.
        chunks_exist = await _chunks_table_exists(conn, schema)

        report = await _build_report(conn, crm_kb, kb, chunks, chunks_exist)
        _print_report(report, chunks_exist)

        if not execute:
            print("Dry run complete -- no changes written. Re-run with --execute to apply.")
            return 0

        deleted = await _execute_deletes(conn, crm_kb, kb, chunks, chunks_exist)
        total = _print_deleted_summary(deleted, chunks_exist)
        print(f"Executed -- {total} total row(s) deleted across doc + chunk tables.")
        return 0
    finally:
        await conn.close()


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
