"""0031 data migration: get_player_transactions `type` spec + description in
chat_tools and crm_tools, run against a throwaway sqlite schema."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_PATH = (Path(__file__).resolve().parents[2] / "alembic" / "versions"
         / "0031_txn_type_filter_enum.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_t0031", _PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Pinned independently of the migration module: the exact historical texts.
_HEAD = (
    "Get the player's transaction history: deposits, withdrawals, casino "
    "credits/debits, sports credits/debits. Supports filtering by type and "
    "date range via query params. Use to answer questions like 'did my "
    "deposit go through?' or 'show my recent withdrawals'."
)
V_5ECDC36 = _HEAD
V_937C168 = (
    _HEAD + " NOTE: this does NOT return the PgsOrderId needed to raise a "
    "deposit verification \u2014 call get_player_latest_deposit_order for that."
)
V_495DA20 = (
    _HEAD + " NOTE: for a deposit DISPUTE, prefer "
    "get_player_latest_deposit_order instead: it targets the specific "
    "recent attempt in one call and exposes the pending/failed status "
    "detail a dispute needs."
)
OLD_TYPE = {
    "type": "string", "source": "llm",
    "description": "Filter: deposit | withdrawal | casino | sports | all (default: all)",
}

V_0031_NEW = (
    "Get the player's transaction history: deposits, withdrawals, casino "
    "credits/debits, sports credits/debits. Most questions are about a "
    "deposit or a withdrawal: work out which, then filter with type "
    "(deposit or withdraw). If the message does not make clear whether it "
    "is a deposit or a withdrawal, ask the customer one short question "
    "before calling this. Use casino or sports only when the customer "
    "clearly asks about those; omit type only for a general request "
    "like 'show my recent transactions'. "
    "NOTE: for a deposit DISPUTE, prefer get_player_latest_deposit_order "
    "instead: it targets the specific recent attempt in one call and "
    "exposes the pending/failed status detail a dispute needs."
)
NEW_TYPE_0031 = {
    "type": "string", "source": "llm", "required": False,
    "enum": ["deposit", "withdraw", "casino", "sports"],
    "description": (
        "deposit for deposit questions, withdraw for withdrawal "
        "questions, casino or sports only when the customer clearly "
        "asks about those. Omit for a general request; never send 'all'."
    ),
}


def _seed(conn, mod):
    for tbl in ("chat_tools", "crm_tools"):
        conn.execute(sa.text(
            f"CREATE TABLE {tbl} (id INTEGER PRIMARY KEY, name VARCHAR, "
            "description TEXT, parameters JSON)"))
        old_params = {
            "user_id": {"type": "string", "source": "session"},
            "type": OLD_TYPE,
            "limit": {"type": "integer", "source": "llm"},
        }
        rows = [
            (1, "get_player_transactions", V_495DA20, old_params),
            (2, "get_player_transactions", "custom text", old_params),
            (4, "get_player_transactions", V_5ECDC36, old_params),
            (5, "get_player_transactions", V_937C168, old_params),
            (6, "get_player_transactions", "no type param", {"limit": {"type": "integer"}}),
            (3, "get_player_wallet", "w", {"type": {"type": "string"}}),
        ]
        for i, n, d, p in rows:
            conn.execute(
                sa.text(f"INSERT INTO {tbl} VALUES (:i,:n,:d,:p)"),
                {"i": i, "n": n, "d": d, "p": json.dumps(p)})


def _get(conn, tbl, i):
    d, p = conn.execute(sa.text(
        f"SELECT description, parameters FROM {tbl} WHERE id={i}")).one()
    return d, json.loads(p)


def test_revision_ids():
    m = _mod()
    assert m.revision == "0031_txn_type_filter_enum" and len(m.revision) <= 32
    assert m.down_revision == "0030_hot_issues"


def test_known_old_variants_are_pinned():
    m = _mod()
    assert set(m.KNOWN_OLD_DESCRIPTIONS) == {V_5ECDC36, V_937C168, V_495DA20}
    assert m.OLD_DESCRIPTION == V_495DA20


def test_new_texts_are_pinned_withdraw_era():
    # 0031 is checked against its own literals, not the live catalog (which
    # moved on to "withdrawal" in 0032).
    m = _mod()
    assert m.NEW_DESCRIPTION == V_0031_NEW
    assert m.NEW_TYPE_SPEC == NEW_TYPE_0031


def test_upgrade_and_downgrade_roundtrip():
    m = _mod()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _seed(conn, m)
        with Operations.context(MigrationContext.configure(conn)):
            m.upgrade()
        for tbl in ("chat_tools", "crm_tools"):
            d, p = _get(conn, tbl, 1)
            assert d == m.NEW_DESCRIPTION
            assert p["type"] == m.NEW_TYPE_SPEC
            assert p["limit"] == {"type": "integer", "source": "llm"}
            d, p = _get(conn, tbl, 2)
            assert d == "custom text"
            assert p["type"] == m.NEW_TYPE_SPEC
            for i in (4, 5):
                d, p = _get(conn, tbl, i)
                assert d == m.NEW_DESCRIPTION and p["type"] == m.NEW_TYPE_SPEC
            d, p = _get(conn, tbl, 6)
            assert d == "no type param" and p == {"limit": {"type": "integer"}}
            d, p = _get(conn, tbl, 3)
            assert d == "w" and p == {"type": {"type": "string"}}
        with Operations.context(MigrationContext.configure(conn)):
            m.downgrade()
        for tbl in ("chat_tools", "crm_tools"):
            d, p = _get(conn, tbl, 1)
            assert d == V_495DA20 and p["type"] == OLD_TYPE
            d, p = _get(conn, tbl, 2)
            assert d == "custom text" and p["type"] == OLD_TYPE
            d, p = _get(conn, tbl, 6)
            assert p == {"limit": {"type": "integer"}}
