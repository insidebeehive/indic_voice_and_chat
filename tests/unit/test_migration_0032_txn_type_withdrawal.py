"""0032 data migration: get_player_transactions withdraw -> withdrawal, run
against a throwaway sqlite schema."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from src.chatbot.catalog import PLAYER_TOOLS

_VERS = Path(__file__).resolve().parents[2] / "alembic" / "versions"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, _VERS / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mod():
    return _load("_t0032", "0032_txn_type_withdrawal.py")


_TAIL = (
    "NOTE: for a deposit DISPUTE, prefer get_player_latest_deposit_order "
    "instead: it targets the specific recent attempt in one call and "
    "exposes the pending/failed status detail a dispute needs."
)
_MID = (
    "Get the player's transaction history: deposits, withdrawals, casino "
    "credits/debits, sports credits/debits. Most questions are about a "
    "deposit or a withdrawal: work out which, then filter with type "
)
_REST = (
    ". If the message does not make clear whether it "
    "is a deposit or a withdrawal, ask the customer one short question "
    "before calling this. Use casino or sports only when the customer "
    "clearly asks about those; omit type only for a general request "
    "like 'show my recent transactions'. "
)
# Pinned independently of the migration module.
DESC_0031 = _MID + "(deposit or withdraw)" + _REST + _TAIL
DESC_FIXED = _MID + "(deposit or withdrawal)" + _REST + _TAIL
TYPE_0031 = {
    "type": "string", "source": "llm", "required": False,
    "enum": ["deposit", "withdraw", "casino", "sports"],
    "description": (
        "deposit for deposit questions, withdraw for withdrawal "
        "questions, casino or sports only when the customer clearly "
        "asks about those. Omit for a general request; never send 'all'."
    ),
}
TYPE_FIXED = {
    "type": "string", "source": "llm", "required": False,
    "enum": ["deposit", "withdrawal", "casino", "sports"],
    "description": (
        "deposit for deposit questions, withdrawal for withdrawal "
        "questions, casino or sports only when the customer clearly "
        "asks about those. Omit for a general request; never send 'all'."
    ),
}
CUSTOM_TYPE = {"type": "string", "source": "llm", "description": "custom"}


def _seed(conn):
    for tbl in ("chat_tools", "crm_tools"):
        conn.execute(sa.text(
            f"CREATE TABLE {tbl} (id INTEGER PRIMARY KEY, name VARCHAR, "
            "description TEXT, parameters JSON)"))
        base = {"user_id": {"type": "string", "source": "session"},
                "limit": {"type": "integer", "source": "llm"}}
        rows = [
            (1, "get_player_transactions", DESC_0031, {**base, "type": TYPE_0031}),
            (2, "get_player_transactions", "custom text", {**base, "type": TYPE_0031}),
            (3, "get_player_transactions", DESC_0031, {**base, "type": CUSTOM_TYPE}),
            (4, "get_player_transactions", "no type", {"limit": {"type": "integer"}}),
            (5, "get_player_wallet", "w", {"type": TYPE_0031}),
        ]
        for i, n, d, p in rows:
            conn.execute(sa.text(f"INSERT INTO {tbl} VALUES (:i,:n,:d,:p)"),
                         {"i": i, "n": n, "d": d, "p": json.dumps(p)})


def _get(conn, tbl, i):
    d, p = conn.execute(sa.text(
        f"SELECT description, parameters FROM {tbl} WHERE id={i}")).one()
    return d, json.loads(p)


def _run(conn, fn):
    with Operations.context(MigrationContext.configure(conn)):
        fn()


def test_revision_ids():
    m = _mod()
    assert m.revision == "0032_txn_type_withdrawal" and len(m.revision) <= 32
    assert m.down_revision == "0031_txn_type_filter_enum"


def test_texts_pinned():
    m = _mod()
    assert m.OLD_DESCRIPTION == DESC_0031 and m.OLD_TYPE_SPEC == TYPE_0031
    assert m.NEW_DESCRIPTION == DESC_FIXED and m.NEW_TYPE_SPEC == TYPE_FIXED


def test_corrected_texts_match_catalog():
    m = _mod()
    spec = PLAYER_TOOLS["get_player_transactions"]
    assert spec["description"] == m.NEW_DESCRIPTION
    assert spec["parameters"]["type"] == m.NEW_TYPE_SPEC


def test_upgrade_and_downgrade_roundtrip():
    m = _mod()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _seed(conn)
        _run(conn, m.upgrade)
        for tbl in ("chat_tools", "crm_tools"):
            d, p = _get(conn, tbl, 1)
            assert d == DESC_FIXED and p["type"] == TYPE_FIXED
            assert p["limit"] == {"type": "integer", "source": "llm"}
            d, p = _get(conn, tbl, 2)  # customised description untouched
            assert d == "custom text" and p["type"] == TYPE_FIXED
            d, p = _get(conn, tbl, 3)  # custom type spec untouched
            assert d == DESC_FIXED and p["type"] == CUSTOM_TYPE
            d, p = _get(conn, tbl, 4)  # no type key
            assert d == "no type" and p == {"limit": {"type": "integer"}}
            d, p = _get(conn, tbl, 5)  # other tool
            assert d == "w" and p == {"type": TYPE_0031}
        _run(conn, m.downgrade)
        for tbl in ("chat_tools", "crm_tools"):
            d, p = _get(conn, tbl, 1)
            assert d == DESC_0031 and p["type"] == TYPE_0031
            d, p = _get(conn, tbl, 2)
            assert d == "custom text" and p["type"] == TYPE_0031
            d, p = _get(conn, tbl, 3)
            assert p["type"] == CUSTOM_TYPE
            d, p = _get(conn, tbl, 4)
            assert p == {"limit": {"type": "integer"}}


def test_chain_0031_then_0032_yields_catalog():
    m31 = _load("_t0031c", "0031_txn_type_filter_enum.py")
    m32 = _mod()
    old_type = {
        "type": "string", "source": "llm",
        "description": "Filter: deposit | withdrawal | casino | sports | all (default: all)",
    }
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        for tbl in ("chat_tools", "crm_tools"):
            conn.execute(sa.text(
                f"CREATE TABLE {tbl} (id INTEGER PRIMARY KEY, name VARCHAR, "
                "description TEXT, parameters JSON)"))
            conn.execute(
                sa.text(f"INSERT INTO {tbl} VALUES (1,'get_player_transactions',"
                        ":d,:p)"),
                {"d": m31.OLD_DESCRIPTION,
                 "p": json.dumps({"type": old_type,
                                  "limit": {"type": "integer"}})})
        _run(conn, m31.upgrade)
        _run(conn, m32.upgrade)
        spec = PLAYER_TOOLS["get_player_transactions"]
        for tbl in ("chat_tools", "crm_tools"):
            d, p = _get(conn, tbl, 1)
            assert d == spec["description"]
            assert p["type"] == spec["parameters"]["type"]
