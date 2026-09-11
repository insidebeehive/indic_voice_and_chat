"""Reseed ``crm_tools`` rows for one CRM from ``src.chatbot.catalog.ALL_TOOLS``.

This is a NEW sibling script, not an extension of ``reseed_catalog_tools.py``:
that script patches ``chat_tools`` (tenant-specific tools, keyed by
``tenant_id`` — the "tenant-specific tools take precedence" branch of
``resolve_crm_tools`` in ``src/bootstrap.py``). This script patches a
different table, ``crm_tools`` (keyed by ``crm_id``), which is shared across
every tenant that points at that CRM. A write here has a much bigger blast
radius — one ``--apply`` run can change tool behaviour for every tenant on
that CRM at once — which is why this script defaults to a dry run and
requires the explicit ``--apply`` flag to write anything, unlike its sibling
and unlike ``scripts/update_matka_crm_tools.py`` (a one-off, already-reviewed
patch for a single CRM).

There is deliberately no delete channel: a ``crm_tools`` row present in the
DB but absent from ``ALL_TOOLS`` (``unregistered`` in the diff) is reported
only, never removed. Such a row can be a deliberate per-CRM rename — e.g.
betstudio's ``get_matka_markets``, which ``scripts/update_matka_crm_tools.py``
created by renaming the catalog's ``get_matka_config`` to point at that CRM's
actual ``/operators/{operator_id}/matka/markets`` endpoint. Deciding whether
such a row is intentional is not something this script can infer, so it is
surfaced for a human to review instead.

Uses ``asyncpg`` (the driver used everywhere else in this repo, not
``psycopg2``). ``DATABASE_URL`` is resolved via ``src.config``'s settings
first (which read ``.env``), falling back to the raw process env var — a
plain ``python scripts/...`` run does not have ``.env`` exported into the
process environment, so this means the caller doesn't need to export it by
hand. Passing ``DATABASE_URL=<url>`` explicitly still works and takes
precedence over whatever ``.env``/``config/default.yaml`` would resolve to.

Usage:
    python scripts/reseed_crm_catalog_tools.py --crm-id betstudio
    python scripts/reseed_crm_catalog_tools.py --crm-id betstudio --apply
    python scripts/reseed_crm_catalog_tools.py --crm-id betstudio \\
        --tools get_player_latest_deposit_order get_bet_limit get_market_holiday_schedule --apply

The third (narrowed) form is the recommended real betstudio command:
``get_matka_config`` must be excluded there because betstudio has it
registered under the renamed ``get_matka_markets`` instead — inserting the
catalog's ``get_matka_config`` under its old name/path alongside that rename
would create a duplicate, semantically-wrong tool. A full-catalog run would
also rewrite descriptions on the 3 existing rows that cross-reference the
old, pre-rename ``get_matka_config`` name in their own description text
(see ``scripts/update_matka_crm_tools.py``), which is exactly the kind of
drift the narrowed ``--tools`` form is meant to avoid.

``endpoint`` and ``method`` on existing rows are never read or modified by
this script — only ``description``/``parameters`` are ever updated on an
existing row, and inserts always use the catalog's bare ``default_path``.
The stored ``endpoint`` is always a bare path (e.g. ``/players/{user_id}/wallet``),
never prefixed with the CRM's ``base_url``: ``resolve_crm_tools`` in
``src/bootstrap.py`` does ``crm.base_url.rstrip("/") + row.endpoint`` at read
time, so a stored full URL would double-prefix and break every call for that
CRM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg  # type: ignore[import-not-found]

from src.chatbot.catalog import ALL_TOOLS
from src.config import get_settings
from src.models.database import search_path_connect_args
from src.providers.vector_store.pgvector_store import _to_asyncpg_dsn


@dataclass(frozen=True)
class ToolInsert:
    name: str
    description: str
    parameters: dict
    endpoint: str           # bare path == catalog default_path; THIS is what gets written to the DB
    method: str
    resolved_endpoint: str  # base_url.rstrip("/") + endpoint — for the human-readable report ONLY, never written


@dataclass(frozen=True)
class ToolUpdate:
    name: str
    row_id: int
    description: str        # catalog value to write
    parameters: dict        # catalog value to write
    description_changed: bool
    parameters_changed: bool


@dataclass(frozen=True)
class CatalogDiff:
    inserts: list[ToolInsert] = field(default_factory=list)
    updates: list[ToolUpdate] = field(default_factory=list)
    unregistered: list[str] = field(default_factory=list)   # in DB, absent from ALL_TOOLS — report only, NEVER delete
    unchanged: list[str] = field(default_factory=list)      # in DB and identical to catalog


def compute_diff(
    existing: dict[str, dict],
    base_url: str,
    tool_names: Sequence[str],
) -> CatalogDiff:
    """Pure diff between the catalog and a snapshot of ``crm_tools`` rows.

    ``existing`` maps tool name -> {"id": int, "description": str,
    "parameters": dict} (caller normalizes ``parameters`` to a dict before
    calling). No DB access, no I/O, no printing here — safe to unit test.
    """
    unknown = set(tool_names) - set(ALL_TOOLS)
    if unknown:
        raise ValueError(f"unknown tool names: {sorted(unknown)}")

    inserts: list[ToolInsert] = []
    updates: list[ToolUpdate] = []
    unchanged: list[str] = []
    stripped_base = base_url.rstrip("/")

    for name in sorted(set(tool_names)):
        spec = ALL_TOOLS[name]
        row = existing.get(name)
        if row is None:
            inserts.append(ToolInsert(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                endpoint=spec["default_path"],
                method=spec["method"],
                resolved_endpoint=stripped_base + spec["default_path"],
            ))
            continue

        description_changed = row["description"] != spec["description"]
        parameters_changed = row["parameters"] != spec["parameters"]
        if description_changed or parameters_changed:
            updates.append(ToolUpdate(
                name=name,
                row_id=row["id"],
                description=spec["description"],
                parameters=spec["parameters"],
                description_changed=description_changed,
                parameters_changed=parameters_changed,
            ))
        else:
            unchanged.append(name)

    unregistered = sorted(n for n in existing if n not in ALL_TOOLS)
    return CatalogDiff(inserts=inserts, updates=updates, unregistered=unregistered, unchanged=unchanged)


def _print_report(diff: CatalogDiff, crm_id: str) -> None:
    print(f"crm_id={crm_id!r}")

    print(f"\nINSERT ({len(diff.inserts)})")
    for i in diff.inserts:
        print(f"  + {i.name}  {i.method} {i.endpoint}  ->  {i.resolved_endpoint}")

    print(f"\nUPDATE ({len(diff.updates)})")
    for u in diff.updates:
        changed = []
        if u.description_changed:
            changed.append("description")
        if u.parameters_changed:
            changed.append("parameters")
        print(f"  ~ {u.name}  (id={u.row_id})  changed: {', '.join(changed)}")

    print(f"\nUNCHANGED ({len(diff.unchanged)})")
    for name in diff.unchanged:
        print(f"  = {name}")

    print(f"\nPRESENT IN DB BUT NOT IN catalog.py — LEFT UNTOUCHED ({len(diff.unregistered)})")
    for name in diff.unregistered:
        print(f"  ? {name}")
    if diff.unregistered:
        print(
            "  CAUTION: a name here may be a deliberate per-CRM rename rather "
            "than something missing from the catalog — e.g. betstudio's "
            "get_matka_markets is a rename of get_matka_config done by "
            "scripts/update_matka_crm_tools.py. Blindly inserting the "
            "catalog's version under the old name could create a duplicate, "
            "semantically-wrong tool. Review before --apply; narrow scope "
            "with --tools if needed."
        )

    if diff.updates and diff.unregistered:
        print(
            "\n  CAUTION: catalog descriptions may reference tool names this "
            "CRM has renamed away from, so a description rewrite here can "
            "introduce dangling tool-name references into the model's prompt."
        )


def _resolve_database_url() -> str:
    """src.config's settings first (it reads .env), the raw process env var
    as a fallback. A plain ``python scripts/...`` invocation does not have
    ``.env`` exported into the process environment — this has already bitten
    the benchmark CLI and ``VOX_SECRET_KEY`` — so resolving through settings
    first means the caller doesn't have to remember to export DATABASE_URL
    by hand. Mirrors ``_resolve_database_url`` in
    tests/unit/test_crm_catalog_seeding.py."""
    try:
        url = get_settings().secrets.DATABASE_URL
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--crm-id", required=True, help="crms.id to reseed crm_tools rows for")
    parser.add_argument("--tools", nargs="+", default=list(ALL_TOOLS.keys()),
                        help="Tool names to reseed (default: all catalog tools)")
    parser.add_argument("--apply", action="store_true",
                        help="Write changes (default: dry run, nothing written)")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    db_url = _resolve_database_url()
    if not db_url:
        print(
            "ERROR: DATABASE_URL not set (checked src.config settings and the "
            "process env)",
            file=sys.stderr,
        )
        return 1

    dsn = _to_asyncpg_dsn(db_url)
    # search_path_connect_args pins the connection's search_path to the app's
    # configured schema (default "voicebot"), the same way the SQLAlchemy
    # engine does for every ORM query against these tables — crm_tools/crms
    # are plain ORM models (src/models/crm.py) with no schema-qualified table
    # name of their own, so a bare asyncpg connection needs this to resolve
    # them at all.
    conn = await asyncpg.connect(dsn, **search_path_connect_args(db_url))
    try:
        crm_row = await conn.fetchrow("SELECT base_url FROM crms WHERE id = $1", args.crm_id)
        if crm_row is None:
            print(f"ERROR: no crm row with id={args.crm_id!r}", file=sys.stderr)
            return 1
        base_url = crm_row["base_url"]

        rows = await conn.fetch(
            "SELECT id, name, description, parameters FROM crm_tools WHERE crm_id = $1",
            args.crm_id,
        )
        existing = {
            row["name"]: {
                "id": row["id"],
                "description": row["description"],
                # asyncpg returns json/jsonb columns as text (no codec
                # registered on this bare connection), same shape psycopg2's
                # RealDictCursor produced — normalize the same way.
                "parameters": json.loads(row["parameters"]) if isinstance(row["parameters"], str)
                else (row["parameters"] or {}),
            }
            for row in rows
        }

        diff = compute_diff(existing, base_url, args.tools)
        _print_report(diff, args.crm_id)

        if not args.apply:
            print("\nDry run — nothing written. Re-run with --apply to write.")
            return 0

        if not diff.inserts and not diff.updates:
            print("\nNothing to apply — diff is empty. No write transaction opened.")
            return 0

        # Single transaction: a partial failure (e.g. mid-batch constraint
        # violation) must not leave crm_tools half-updated for a table shared
        # by every tenant on this CRM.
        async with conn.transaction():
            for i in diff.inserts:
                await conn.execute(
                    "INSERT INTO crm_tools (crm_id, name, description, endpoint, method, parameters) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    args.crm_id, i.name, i.description, i.endpoint, i.method, json.dumps(i.parameters),
                )
            for u in diff.updates:
                await conn.execute(
                    "UPDATE crm_tools SET description = $1, parameters = $2 WHERE id = $3",
                    u.description, json.dumps(u.parameters), u.row_id,
                )

        print(f"\nDone — {len(diff.inserts)} insert(s), {len(diff.updates)} update(s) applied.")
        if diff.inserts:
            print("  inserted: " + ", ".join(i.name for i in diff.inserts))
        if diff.updates:
            print("  updated:  " + ", ".join(u.name for u in diff.updates))
        return 0
    finally:
        await conn.close()


def main() -> int:
    args = parse_args()

    unknown = [t for t in args.tools if t not in ALL_TOOLS]
    if unknown:
        print(f"ERROR: unknown tool names: {unknown}", file=sys.stderr)
        print(f"Available: {list(ALL_TOOLS.keys())}", file=sys.stderr)
        return 1

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
