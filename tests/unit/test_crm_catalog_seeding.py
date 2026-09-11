"""tests/unit/test_crm_catalog_seeding.py

Tests for ``scripts/reseed_crm_catalog_tools.py``, which reseeds ``crm_tools``
rows (the CRM-catalog branch of ``resolve_crm_tools`` in ``src/bootstrap.py``)
from ``src.chatbot.catalog.ALL_TOOLS``. Kept in its own file rather than
appended to ``test_crm_tools_platform_fallback.py`` or ``test_catalog_routes.py``
— those cover unrelated runtime/API paths, not this seeding script.

The mandatory tests below exercise ``compute_diff`` directly: it takes no
DB/psycopg2 dependency (psycopg2 isn't even installed in this venv — the
script imports it lazily inside ``main()``), so they run with zero external
dependencies. One optional test at the bottom is DB-guarded and skipped
unless DATABASE_URL resolves.
"""
from __future__ import annotations

import json
import os

import pytest

from scripts.reseed_crm_catalog_tools import compute_diff
from src.chatbot.catalog import ALL_TOOLS

# Deliberately hardcoded, not derived from ALL_TOOLS — this is what forces a
# CI failure (and a deliberate second edit here) whenever catalog.py's tool
# set changes, instead of the drift silently reaching production the way
# get_player_latest_deposit_order did.
SEEDED_TOOL_NAMES = frozenset({
    "get_player_wallet",
    "get_player_transactions",
    "get_player_latest_deposit_order",
    "get_player_bets",
    "get_player_bonuses",
    "get_player_profile",
    "get_player_responsible_gaming",
    "get_payment_config",
    "get_referral_code",
    "get_sports_open_bets",
    "get_sports_match_status",
    "get_casino_game_history",
    "get_matka_bids",
    "get_game",
    "get_game_providers",
    "get_operator_games_config",
    "get_matka_config",
    "get_matka_result",
    "get_operator_promotions",
    "get_operator_platform_config",
    "get_bet_limit",
    "get_market_holiday_schedule",
})


def _existing_from_catalog(names, *, start_id: int = 1) -> dict[str, dict]:
    """Build fixture crm_tools rows that match catalog.py exactly, keyed by
    name, {"id", "description", "parameters"} — self-consistent by
    construction since it reads straight from ALL_TOOLS."""
    return {
        name: {
            "id": start_id + i,
            "description": ALL_TOOLS[name]["description"],
            "parameters": ALL_TOOLS[name]["parameters"],
        }
        for i, name in enumerate(sorted(names))
    }


def test_catalog_tool_names_match_seeded_set() -> None:
    catalog_names = set(ALL_TOOLS)
    if catalog_names != SEEDED_TOOL_NAMES:
        added = catalog_names - SEEDED_TOOL_NAMES
        removed = SEEDED_TOOL_NAMES - catalog_names
        pytest.fail(
            "src/chatbot/catalog.py's ALL_TOOLS has drifted from "
            f"SEEDED_TOOL_NAMES in this test file. Added: {sorted(added)}; "
            f"Removed: {sorted(removed)}. To fix: (a) update "
            "SEEDED_TOOL_NAMES in tests/unit/test_crm_catalog_seeding.py, "
            "and (b) for every CRM using the crm-catalog path, first check "
            "`unregistered` for a possible per-CRM rename of the added/changed "
            "tool(s), then run "
            "`DATABASE_URL=... python scripts/reseed_crm_catalog_tools.py "
            "--crm-id <crm> --tools <added/changed tool names> --apply` "
            "(narrowed to those tools, not a bare --crm-id --apply covering "
            "the whole catalog)."
        )


def test_diff_is_noop_against_fully_seeded_crm() -> None:
    existing = _existing_from_catalog(SEEDED_TOOL_NAMES)
    diff = compute_diff(existing, "https://api.example.test/api", sorted(SEEDED_TOOL_NAMES))
    assert diff.inserts == []
    assert diff.updates == []
    assert diff.unregistered == []
    assert len(diff.unchanged) == 22


def test_diff_reports_missing_tools_as_inserts() -> None:
    missing = {
        "get_player_latest_deposit_order",
        "get_bet_limit",
        "get_market_holiday_schedule",
        "get_matka_config",
    }
    existing = _existing_from_catalog(SEEDED_TOOL_NAMES - missing)
    diff = compute_diff(existing, "https://api.example.test/api", sorted(SEEDED_TOOL_NAMES))
    assert {i.name for i in diff.inserts} == missing
    assert diff.updates == []


def test_diff_reports_unknown_db_row_as_unregistered() -> None:
    existing = _existing_from_catalog(SEEDED_TOOL_NAMES)
    existing["get_matka_markets"] = {
        "id": 999,
        "description": "renamed matka markets endpoint",
        "parameters": {"operator_id": {"type": "string", "source": "session"}},
    }
    diff = compute_diff(existing, "https://api.example.test/api", sorted(SEEDED_TOOL_NAMES))
    assert diff.unregistered == ["get_matka_markets"]
    assert "get_matka_markets" not in {i.name for i in diff.inserts}
    assert "get_matka_markets" not in {u.name for u in diff.updates}


def test_insert_endpoints_are_bare_paths_and_resolve_under_base_url() -> None:
    missing = {
        "get_player_latest_deposit_order",
        "get_bet_limit",
        "get_market_holiday_schedule",
        "get_matka_config",
    }
    existing = _existing_from_catalog(SEEDED_TOOL_NAMES - missing)

    diff_no_slash = compute_diff(existing, "https://api.example.test/api", sorted(SEEDED_TOOL_NAMES))
    diff_with_slash = compute_diff(existing, "https://api.example.test/api/", sorted(SEEDED_TOOL_NAMES))

    by_name_no_slash = {i.name: i for i in diff_no_slash.inserts}
    by_name_with_slash = {i.name: i for i in diff_with_slash.inserts}

    for name in missing:
        insert = by_name_no_slash[name]
        assert insert.endpoint.startswith("/")
        assert "http" not in insert.endpoint
        assert insert.endpoint == ALL_TOOLS[name]["default_path"]

        # Trailing slash on base_url must not double-prefix or otherwise
        # change the resolved endpoint.
        assert insert.resolved_endpoint == "https://api.example.test/api" + insert.endpoint
        assert by_name_with_slash[name].resolved_endpoint == insert.resolved_endpoint


def test_diff_reports_description_and_parameter_drift_as_updates() -> None:
    existing = _existing_from_catalog(SEEDED_TOOL_NAMES)
    existing["get_player_wallet"]["description"] = "stale description text"
    existing["get_player_bets"]["parameters"] = {"user_id": {"type": "string", "source": "session"}}

    diff = compute_diff(existing, "https://api.example.test/api", sorted(SEEDED_TOOL_NAMES))
    updates = {u.name: u for u in diff.updates}

    assert set(updates) == {"get_player_wallet", "get_player_bets"}

    wallet_update = updates["get_player_wallet"]
    assert wallet_update.description_changed is True
    assert wallet_update.parameters_changed is False
    assert wallet_update.row_id == existing["get_player_wallet"]["id"]
    assert wallet_update.description == ALL_TOOLS["get_player_wallet"]["description"]
    assert wallet_update.parameters == ALL_TOOLS["get_player_wallet"]["parameters"]

    bets_update = updates["get_player_bets"]
    assert bets_update.description_changed is False
    assert bets_update.parameters_changed is True
    assert bets_update.row_id == existing["get_player_bets"]["id"]
    assert bets_update.description == ALL_TOOLS["get_player_bets"]["description"]
    assert bets_update.parameters == ALL_TOOLS["get_player_bets"]["parameters"]


def test_compute_diff_rejects_unknown_tool_names() -> None:
    with pytest.raises(ValueError):
        compute_diff({}, "https://api.example.test/api", ["not_a_real_tool"])


# --- optional, DB-guarded structural-sanity test ----------------------------


def _resolve_database_url() -> str:
    """Same resolution strategy as tests/unit/test_pgvector_crm_scoping.py:
    prefer the app's own settings (which read .env), fall back to the raw
    process env var, and swallow all failures so a missing/broken .env
    degrades to a clean skip rather than a collection error."""
    try:
        from src.config import get_settings
        url = get_settings().secrets.DATABASE_URL
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "")


_DATABASE_URL = _resolve_database_url()


@pytest.mark.skipif(
    not _DATABASE_URL,
    reason="integration test: requires a live Postgres with crms/crm_tools tables; set DATABASE_URL (or .env) to run it",
)
async def test_live_crm_tools_rows_have_bare_path_endpoints() -> None:
    """Structural-sanity check against the live DB, NOT a zero-drift check —
    this must NOT assert "every catalog tool is present for every CRM", since
    that would hard-code today's known-incomplete betstudio state (missing 4
    catalog tools, plus the get_matka_config -> get_matka_markets rename) as
    a permanent, unfixable-without-corrupting-data expectation.

    HARD RULE: this test may ONLY run SELECT statements. The DB is a live
    shared multi-tenant Neon instance with real customer data — no
    INSERT/UPDATE/DELETE, ever.
    """
    import asyncpg

    dsn = _DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )

    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT c.id AS crm_id, c.base_url, t.name, t.endpoint, t.method,
                   t.parameters, t.description
            FROM crms c JOIN crm_tools t ON t.crm_id = c.id
            """
        )
    finally:
        await conn.close()

    crm_ids: set[str] = set()
    for row in rows:
        crm_id = row["crm_id"]
        name = row["name"]
        endpoint = row["endpoint"]
        method = row["method"]
        description = row["description"]

        assert endpoint.startswith("/"), (
            f"crm_id={crm_id!r} tool={name!r}: endpoint {endpoint!r} is not a bare path"
        )
        assert "http" not in endpoint, (
            f"crm_id={crm_id!r} tool={name!r}: endpoint {endpoint!r} looks like a full URL"
        )
        assert row["base_url"] not in endpoint, (
            f"crm_id={crm_id!r} tool={name!r}: endpoint {endpoint!r} contains the CRM base_url "
            "(would double-prefix at read time)"
        )
        assert method, f"crm_id={crm_id!r} tool={name!r}: method is empty"
        assert description, f"crm_id={crm_id!r} tool={name!r}: description is empty"

        parameters = row["parameters"]
        parameters_dict = json.loads(parameters) if isinstance(parameters, str) else parameters
        assert isinstance(parameters_dict, dict), (
            f"crm_id={crm_id!r} tool={name!r}: parameters did not decode to a dict"
        )
        for param_name, param_spec in parameters_dict.items():
            assert isinstance(param_spec, dict) and "type" in param_spec, (
                f"crm_id={crm_id!r} tool={name!r}: parameter {param_name!r} spec {param_spec!r} "
                "missing a 'type' key"
            )

        crm_ids.add(crm_id)

    # Informational only, never fatal: surface drift per CRM without turning
    # today's known-incomplete state into a hard-fail zero-drift assertion.
    import warnings

    from scripts.reseed_crm_catalog_tools import compute_diff

    for crm_id in crm_ids:
        crm_row = next(r for r in rows if r["crm_id"] == crm_id)
        existing = {}
        for r in rows:
            if r["crm_id"] != crm_id:
                continue
            p = r["parameters"]
            existing[r["name"]] = {
                "id": 0,
                "description": r["description"],
                "parameters": json.loads(p) if isinstance(p, str) else (p or {}),
            }
        try:
            diff = compute_diff(existing, crm_row["base_url"], list(ALL_TOOLS.keys()))
        except Exception:
            continue
        if diff.inserts or diff.unregistered:
            missing = sorted(i.name for i in diff.inserts)
            warnings.warn(
                f"crm_id={crm_id!r} catalog drift — missing: {missing}, "
                f"unregistered: {diff.unregistered}. Some 'missing' names may "
                "already exist under a different name in 'unregistered' (a "
                "per-CRM rename, e.g. betstudio's get_matka_config -> "
                "get_matka_markets) — review unregistered for possible "
                "renames before choosing which --tools to pass to "
                "scripts/reseed_crm_catalog_tools.py --crm-id "
                f"{crm_id} --tools <reviewed names> --apply.",
                stacklevel=1,
            )
