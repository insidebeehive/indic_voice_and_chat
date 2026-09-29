"""Round-trips the real 0028 migration module's upgrade()/downgrade() against
a throwaway sqlite table shaped like chat_turn_metrics -- proves both
directions actually execute the intended add/drop, rather than trusting the
file by inspection. Same mechanism as
test_migration_0022_provider_cost_cached_rate.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic" / "versions" / "0028_chat_turn_media_cost.py"
)

_NEW_COLUMNS = {
    "tts_provider", "tts_model", "tts_audio_ms", "tts_cost",
    "stt_audio_ms", "stt_cost",
}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "_test_migration_0028_chat_turn_media_cost", _MIGRATION_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _bare_chat_turn_metrics_table(metadata: sa.MetaData) -> sa.Table:
    """A minimal shape of chat_turn_metrics -- just enough columns for the
    migration's ADD COLUMN statements and a round-trip row to be meaningful.
    Not the full real model (this test only cares about the six new columns
    this migration adds)."""
    return sa.Table(
        "chat_turn_metrics", metadata,
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(50)),
        sa.Column("session_id", sa.String(100)),
        sa.Column("llm_provider", sa.String(100)),
        sa.Column("llm_model", sa.String(100)),
        sa.Column("action", sa.String(50)),
    )


def test_migration_0028_upgrade_adds_nullable_columns_without_data_loss():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    chat_turn_metrics = _bare_chat_turn_metrics_table(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        conn.execute(chat_turn_metrics.insert().values(
            tenant_id="dev", session_id="s1", llm_provider="gemini",
            llm_model="gemini-3.5-flash", action="reply",
        ))

        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        cols = {c["name"] for c in sa.inspect(conn).get_columns("chat_turn_metrics")}
        assert _NEW_COLUMNS <= cols

        row = conn.execute(sa.text(
            "SELECT llm_provider, tts_provider, tts_audio_ms, tts_cost, "
            "stt_audio_ms, stt_cost FROM chat_turn_metrics WHERE session_id = 's1'"
        )).one()
        # Pre-existing column untouched.
        assert row.llm_provider == "gemini"
        # Every new column on a pre-existing row: NULL, per the migration's
        # own NULL/0 semantics ("no TTS/STT on this turn or a pre-migration
        # row") -- upgrading must not invent a billed-$0 row out of nothing.
        assert row.tts_provider is None
        assert row.tts_audio_ms is None
        assert row.tts_cost is None
        assert row.stt_audio_ms is None
        assert row.stt_cost is None


def test_migration_0028_new_columns_round_trip_values():
    """A row written after upgrade() can actually store the "billed unknown"
    shape (tts_audio_ms=0, tts_cost=0.0 with provider/model set) and a
    token-priced STT row (stt_audio_ms NULL, stt_cost set)."""
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    chat_turn_metrics = _bare_chat_turn_metrics_table(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        conn.execute(sa.text(
            "INSERT INTO chat_turn_metrics "
            "(tenant_id, session_id, llm_provider, llm_model, action, "
            " tts_provider, tts_model, tts_audio_ms, tts_cost, stt_audio_ms, stt_cost) "
            "VALUES ('dev', 's2', 'gemini', 'gemini-3.5-flash', 'reply', "
            " 'sarvam', 'bulbul:v3', 0, 0.0, NULL, 0.0006)"
        ))
        row = conn.execute(sa.text(
            "SELECT tts_provider, tts_audio_ms, tts_cost, stt_audio_ms, stt_cost "
            "FROM chat_turn_metrics WHERE session_id = 's2'"
        )).one()
        assert row.tts_provider == "sarvam"
        assert row.tts_audio_ms == 0
        assert row.tts_cost == 0.0
        assert row.stt_audio_ms is None
        assert abs(row.stt_cost - 0.0006) < 1e-9


def test_migration_0028_downgrade_drops_all_six_columns():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    _bare_chat_turn_metrics_table(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        assert _NEW_COLUMNS <= {
            c["name"] for c in sa.inspect(conn).get_columns("chat_turn_metrics")
        }

        with Operations.context(ctx):
            mod.downgrade()
        cols = {c["name"] for c in sa.inspect(conn).get_columns("chat_turn_metrics")}
        assert _NEW_COLUMNS.isdisjoint(cols)
        assert "llm_provider" in cols  # untouched sibling column
