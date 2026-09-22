"""Tree-wide conventions for `debug_event` call sites, enforced by AST walk.

Every rule here was written after the convention was violated in real work,
twice each, by people (and agents) who had been told the rule explicitly. A
convention that depends on remembering is not a convention; these are the ones
cheap enough to check mechanically, so they are checked mechanically.

Two of them fail silently in production rather than loudly in review, which is
why a test is worth more here than a code comment:

- A reserved key does not raise (`debug_event` renames it), so the value lands
  under `filename_` while `filename` sits there holding `logging.py`. An
  operator's natural Loki query matches nothing, and the field that DOES match
  is misleading. Caught four times in `src/api/` after the rule was written
  down and restated in the brief.
- A flat event name is perfectly functional and simply cannot be scoped by a
  `component .*` query, which is the whole reason the naming convention exists.

See docs/debug-logging.md. This file is the enforcement of what that documents.
"""

from __future__ import annotations

import ast
import logging
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"

# Derived from a real LogRecord rather than hand-listed: a hardcoded copy of
# the blocklist validated against itself always passes, and `logging` is free
# to grow attributes between versions.
RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName", "event"}

# Shipped before the `<component> <operation> <phase>` convention existed. A
# shipped event name is a Loki query key, and renaming one silently splits any
# query spanning it inside the retention window -- so these stay, and the list
# does not grow. A new flat name is a bug; these three are history.
GRANDFATHERED_FLAT_NAMES = frozenset({
    "chat_frame_received",
    "chat_reply_frame_sent",
    "tts_reply_synthesized",
})


def _literal_names(node: ast.expr) -> list[str] | None:
    """Every event name a name-expression can evaluate to, or None if it can
    evaluate to something unknowable.

    A conditional between two literals is fine and is used deliberately --
    `"pipeline envelope_parse recovered" if recovered else "... failed"` in
    `src/pipeline/engine.py` picks between two stable identifiers, both of
    which are greppable and queryable. What is NOT fine is a name assembled at
    runtime, which no saved query can match. So this unwraps conditionals
    rather than rejecting them.
    """
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else None
    if isinstance(node, ast.IfExp):
        body, orelse = _literal_names(node.body), _literal_names(node.orelse)
        return None if body is None or orelse is None else body + orelse
    return None


def _call_sites() -> list[tuple[pathlib.Path, ast.Call]]:
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "debug_event":
                sites.append((path, node))
    return sites


@pytest.fixture(scope="module")
def call_sites() -> list[tuple[pathlib.Path, ast.Call]]:
    sites = _call_sites()
    # Guard the guard: if the walk stops finding call sites (a rename, an
    # import-style change), every assertion below would vacuously pass.
    assert len(sites) > 100, f"expected the tree's debug_event calls, found {len(sites)}"
    return sites


def test_no_keyword_collides_with_a_logrecord_attribute(call_sites) -> None:
    """`document_filename`, not `filename`; `crm_name`, not `name`.

    The helper's rename keeps the process up -- it does not make the value
    findable, which is the entire point of logging it.
    """
    bad = [
        f"{path.relative_to(SRC.parent)}:{node.lineno} passes {kw.arg}="
        for path, node in call_sites
        for kw in node.keywords
        if kw.arg in RESERVED
    ]
    assert not bad, (
        "debug_event keyword collides with a LogRecord attribute; the value "
        "will be renamed to <key>_ and the un-suffixed key will hold logging's "
        "own value instead:\n  " + "\n  ".join(bad)
    )


def test_event_names_are_namespaced(call_sites) -> None:
    """`<component> <operation> <phase>` -- so `event=~"retriever .*"` works."""
    flat = []
    for path, node in call_sites:
        if len(node.args) < 2:
            continue
        for name in _literal_names(node.args[1]) or []:
            if " " not in name and name not in GRANDFATHERED_FLAT_NAMES:
                flat.append(f"{path.relative_to(SRC.parent)}:{node.lineno} -> {name!r}")
    assert not flat, (
        "event name is not namespaced as '<component> <operation> <phase>'; a "
        "flat name cannot be scoped with a component query:\n  " + "\n  ".join(flat)
    )


def test_event_name_is_a_literal(call_sites) -> None:
    """An f-string or variable name is unqueryable -- the whole point is that
    the name is a stable identifier a saved query can match on."""
    dynamic = [
        f"{path.relative_to(SRC.parent)}:{node.lineno}"
        for path, node in call_sites
        if len(node.args) >= 2 and _literal_names(node.args[1]) is None
    ]
    assert not dynamic, (
        "debug_event's event name must be a string literal, not computed:\n  "
        + "\n  ".join(dynamic)
    )


def test_grandfathered_list_does_not_grow(call_sites) -> None:
    """Pin the exception list to exactly the three names that predate the
    convention. Every one still has to exist -- a stale entry here would let a
    new flat name of the same spelling through unnoticed."""
    all_names = {
        name
        for _, node in call_sites
        if len(node.args) >= 2
        for name in _literal_names(node.args[1]) or []
    }
    assert GRANDFATHERED_FLAT_NAMES <= all_names, (
        "a grandfathered name no longer exists in the tree; remove it from "
        f"GRANDFATHERED_FLAT_NAMES: {sorted(GRANDFATHERED_FLAT_NAMES - all_names)}"
    )
    assert len(GRANDFATHERED_FLAT_NAMES) == 3
