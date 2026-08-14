"""Purge ONLY the true stale-duplicate KB rows -- a reduced-scope sibling of
``scripts/purge_stale_kb_docs.py`` that runs its targets (a)/(b)/(d) and
DELIBERATELY SKIPS (c)/(e).

Why this script exists (do not "helpfully" merge it back into the full purge
without re-reading this): ``purge_stale_kb_docs.py``'s targets (c)/(e) delete
the CRM-wide ``crm_kb_documents`` rows (and their chunks) for the 6 casino/
sports/matka filenames that are moving to the new opt-in-module system (see
``_INGESTIBLE_DOCS`` "casino"/"sports"/"matka" keys in src/api/knowledge.py,
POST .../ingest-layout). As of this script's authoring, a direct check
against the database found ZERO opt-in-module rows ingested for ANY tenant
(``kb_documents`` has no ``layout_casino_*`` / ``layout_sports_*`` /
``layout_matka_*`` rows yet). That means running the FULL purge script's
--execute right now would delete the ONLY existing copy of casino/sports/
matka KB content -- not a duplicate cleanup, an outright content removal,
until an operator explicitly re-ingests those modules per tenant via the
backoffice.

Targets (a)/(b)/(d), by contrast, ARE true duplicates: a 2023 table-rename
migration (0010_crm_kb_documents) left old ``global_kb_<stem>`` rows behind
under a stale id scheme, while the current seeder (``_seed_crm_kb`` in
src/main.py) writes -- and keeps live -- a fresh ``crm_kb_<crm_id>_<stem>``
row for the SAME topic. So for every one of the 12 bundled KB topics, the
stale ``global_kb_<stem>`` row and the live ``crm_kb_<crm_id>_<stem>`` row
are two copies of the same content; deleting only the stale one leaves
exactly one (the current, already-served) copy behind. That is what this
script does, and only that.

This script imports and calls the SAME already-reviewed, already-unit-tested
query/delete functions from purge_stale_kb_docs.py
(_select_stale_crm_kb/_select_stale_kb/_select_stale_chunks_grouped/
_delete_stale_crm_kb/_delete_stale_kb/_delete_stale_chunks) rather than
re-deriving any SQL, so it can never drift from that script's own tested
stale-id-scheme predicate (STALE_ID_REGEX). It never touches
RELOCATED_DOCS_WHERE / RELOCATED_CHUNKS_WHERE / targets (c)/(e) at all.

Once casino/sports/matka have been re-ingested per-tenant via the opt-in
module flow, run the FULL scripts/purge_stale_kb_docs.py (with --execute) to
also clean up targets (c)/(e) -- the now-safely-superseded CRM-wide
casino/sports/matka rows.

Usage (same DSN/schema resolution as purge_stale_kb_docs.py -- run from the
repo root; .env/settings resolution is CWD-relative):
    python scripts/purge_duplicate_kb_docs_only.py                  # dry run (default)
    python scripts/purge_duplicate_kb_docs_only.py --execute         # actually delete
    python scripts/purge_duplicate_kb_docs_only.py --database-url postgresql://...
    python scripts/purge_duplicate_kb_docs_only.py --schema voicebot

Recommended flow: dry-run on stage, review, --execute on stage, confirm the
app still serves KB content correctly, then repeat dry-run -> --execute on
prod.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg  # type: ignore[import-not-found]

from scripts.purge_stale_kb_docs import (
    _build_dsn,
    _chunks_table_exists,
    _delete_stale_chunks,
    _delete_stale_crm_kb,
    _delete_stale_kb,
    _masked_target,
    _quote_ident,
    _resolve_database_url,
    _select_stale_chunks_grouped,
    _select_stale_crm_kb,
    _select_stale_kb,
)


def _table_refs(schema: str) -> tuple[str, str, str]:
    """Fully-qualified, schema-quoted table references -- mirrors
    purge_stale_kb_docs._table_refs (not imported directly since that name
    isn't part of the intentionally-narrow import list above, but the logic
    is one line and trivially kept identical)."""
    q = _quote_ident(schema)
    return f"{q}.crm_kb_documents", f"{q}.kb_documents", f"{q}.knowledge_chunks"


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
        "config/default.yaml -- CWD-relative, so run from the repo root).",
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

    print(f"Target: {_masked_target(dsn)}")
    print(f"Schema: {schema}")
    print(f"Mode:   {'EXECUTE (will delete rows)' if execute else 'DRY RUN (no writes)'}")
    print(
        "Scope:  TRUE DUPLICATES ONLY -- stale 'global_kb_' id scheme "
        "(targets a/b/d). Relocated casino/sports/matka rows (targets c/e) "
        "are intentionally NOT touched by this script -- see module "
        "docstring. Use purge_stale_kb_docs.py for those, after opt-in "
        "re-ingest."
    )
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

        a = await _select_stale_crm_kb(conn, crm_kb)
        print(f"(a) crm_kb_documents -- stale 'global_kb_' id scheme: {len(a)} row(s)")
        for r in a:
            print(f"    {r['id']}  crm_id={r['crm_id']}  filename={r['filename']}")
        print()

        b = await _select_stale_kb(conn, kb)
        print(f"(b) kb_documents (tenant tier) -- stale 'global_kb_' id scheme: {len(b)} row(s)")
        for r in b:
            print(f"    {r['id']}  tenant_id={r['tenant_id']}  filename={r['filename']}")
        print()

        if chunks_exist:
            d_grouped = await _select_stale_chunks_grouped(conn, chunks)
            d_total = sum(r["n"] for r in d_grouped)
            print(
                f"(d) knowledge_chunks -- stale 'global_kb_' id scheme: "
                f"{d_total} chunk(s) across {len(d_grouped)} document(s)"
            )
            for r in d_grouped:
                print(f"    {r['document_id']}: {r['n']} chunk(s)")
            print()
        else:
            print(
                "(d) knowledge_chunks -- stale 'global_kb_' id scheme: "
                "skipped (voicebot.knowledge_chunks table not found -- this "
                "DB may be FAISS-based local/dev)"
            )
            print()

        if not execute:
            print("Dry run complete -- no changes written. Re-run with --execute to apply.")
            return 0

        async with conn.transaction():
            deleted_a = await _delete_stale_crm_kb(conn, crm_kb)
            deleted_b = await _delete_stale_kb(conn, kb)
            deleted_d = await _delete_stale_chunks(conn, chunks) if chunks_exist else []

        total = len(deleted_a) + len(deleted_b) + len(deleted_d)
        print("Deleted:")
        print(f"  (a) crm_kb_documents (stale id scheme):  {len(deleted_a)} row(s)")
        print(f"  (b) kb_documents (stale id scheme):      {len(deleted_b)} row(s)")
        if chunks_exist:
            print(f"  (d) knowledge_chunks (stale id scheme):  {len(deleted_d)} row(s)")
        else:
            print("  (d) knowledge_chunks (stale id scheme):  skipped")
        print()
        print(f"Executed -- {total} total row(s) deleted (true duplicates only).")
        return 0
    finally:
        await conn.close()


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
