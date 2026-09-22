"""A revision id longer than 32 characters cannot be recorded, so it can never apply.

Alembic's own `alembic_version.version_num` column is `VARCHAR(32)`. A longer
id raises `StringDataRightTruncationError` on the `UPDATE alembic_version` that
concludes the migration -- the DDL itself succeeds and then rolls back with it,
so the migration is not half-applied; it is simply unapplicable, forever, on
every deploy.

This has now happened twice:

- 0019 (`0019_turn_metrics_created_at_index`, 34 chars) was unapplicable from
  the day it was authored, blocking 0020 behind it. Fixed in d889604, whose
  message notes that 0018 is *exactly* 32 and "one from the same fate".
- 0025 (`0025_deposit_verification_order_idx`, 35 chars) did the same thing and
  additionally blocked 0026, which the application had already been deployed
  expecting -- every chat-turn metric was dropped until it was fixed.

Both times the failure was invisible at deploy: the Dockerfile runs
`timeout 60 alembic upgrade head; exec uvicorn ...` -- `;`, not `&&` -- so the
app starts against an unmigrated database and serves happily on the old schema.
The only symptom is whatever the new code does when its columns are missing.

A warning in a commit message did not prevent the second occurrence. This test
is the thing that does.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

VERSIONS = pathlib.Path(__file__).resolve().parents[2] / "alembic" / "versions"

# Alembic's own column width. Not configurable without a custom
# `version_table_schema`/`version_table` setup, which this project does not use.
MAX_VERSION_NUM = 32


def _assignments(path: pathlib.Path) -> dict[str, str]:
    """Module-level string assignments, read without importing.

    Importing a migration executes it; parsing keeps this test cheap and safe
    to run against every revision in the tree.

    Handles both spellings present in this tree -- `revision = "..."` and the
    annotated `revision: str = "..."`. Missing the annotated form is not a
    cosmetic gap: those files would drop out of the length check entirely and
    it would pass for them vacuously, which is exactly the failure this file
    exists to prevent.
    """
    out: dict[str, str] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for t in targets:
                out[t.id] = value.value
    return out


def _revisions() -> list[tuple[str, dict[str, str]]]:
    found = [(p.name, _assignments(p)) for p in sorted(VERSIONS.glob("*.py"))]
    return [(n, a) for n, a in found if "revision" in a]


@pytest.fixture(scope="module")
def revisions() -> list[tuple[str, dict[str, str]]]:
    revs = _revisions()
    # Guard the guard: if the parse stops finding revisions, every assertion
    # below passes vacuously and the check silently stops existing.
    assert len(revs) >= 20, f"expected the migration tree, found {len(revs)} revisions"
    return revs


def test_every_revision_id_fits_the_version_table(revisions) -> None:
    too_long = [
        f"{name}: revision={a['revision']!r} is {len(a['revision'])} chars"
        for name, a in revisions
        if len(a["revision"]) > MAX_VERSION_NUM
    ]
    assert not too_long, (
        f"alembic_version.version_num is VARCHAR({MAX_VERSION_NUM}); a longer "
        "revision id raises StringDataRightTruncationError when alembic records "
        "it, so the migration can never apply and everything behind it is "
        "blocked:\n  " + "\n  ".join(too_long)
    )


def test_every_down_revision_fits_too(revisions) -> None:
    """`down_revision` is written into the same column when downgrading, and a
    mismatch between the two is how a rename goes half-done."""
    too_long = [
        f"{name}: down_revision={a['down_revision']!r} is {len(a['down_revision'])} chars"
        for name, a in revisions
        if a.get("down_revision") and len(a["down_revision"]) > MAX_VERSION_NUM
    ]
    assert not too_long, "down_revision exceeds the version_num column:\n  " + "\n  ".join(too_long)


def test_the_chain_is_linear_and_every_link_resolves(revisions) -> None:
    """A dangling `down_revision` fails at runtime, not at author time, and a
    second head means `upgrade head` is ambiguous. Both are cheap to catch here
    and expensive to catch on a deploy that fails open.
    """
    ids = {a["revision"] for _, a in revisions}
    dangling = [
        f"{name}: down_revision={a['down_revision']!r} matches no revision"
        for name, a in revisions
        if a.get("down_revision") and a["down_revision"] not in ids
    ]
    assert not dangling, "broken revision chain:\n  " + "\n  ".join(dangling)

    downs = {a["down_revision"] for _, a in revisions if a.get("down_revision")}
    heads = sorted(ids - downs)
    assert len(heads) == 1, f"expected exactly one head, found {len(heads)}: {heads}"


def test_the_limit_is_what_alembic_actually_uses() -> None:
    """Pin MAX_VERSION_NUM to alembic's real column width rather than a number
    copied into this file -- a constant checked against itself proves nothing,
    and alembic could widen it in a future release.
    """
    import inspect
    import re

    from alembic.ddl import impl

    # alembic builds the table in DefaultImpl._version (alembic/ddl/impl.py):
    #     Column("version_num", String(32), nullable=False)
    # There is no public accessor, so read the source rather than hardcode a
    # second copy of the number here.
    match = re.search(
        r'Column\(\s*["\']version_num["\']\s*,\s*String\((\d+)\)',
        inspect.getsource(impl),
    )
    assert match, (
        "could not find alembic's version_num column definition in "
        "alembic/ddl/impl.py -- its internals moved; re-derive MAX_VERSION_NUM "
        "rather than trusting the constant in this file"
    )
    assert MAX_VERSION_NUM == int(match.group(1)), (
        f"alembic now uses VARCHAR({match.group(1)}); update MAX_VERSION_NUM"
    )
