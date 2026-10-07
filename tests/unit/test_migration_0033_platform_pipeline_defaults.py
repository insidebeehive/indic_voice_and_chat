"""Round-trips the real 0033 migration module's upgrade()/downgrade() against
a throwaway sqlite schema -- proves both directions actually create/drop the
intended table. Same mechanism as test_migration_0030_hot_issues.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic" / "versions" / "0033_platform_pipeline_defaults.py"
)

_EXPECTED_COLUMNS = {"layer", "provider", "model", "updated_at", "updated_by"}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "_test_migration_0033_platform_pipeline_defaults", _MIGRATION_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_migration_0033_revision_and_down_revision():
    mod = _load_migration_module()
    assert mod.revision == "0033_platform_pipeline_defaults"
    assert len(mod.revision) <= 32  # alembic_version.version_num is VARCHAR(32)
    assert mod.down_revision == "0032_txn_type_withdrawal"
    assert mod.branch_labels is None
    assert mod.depends_on is None


def test_migration_0033_upgrade_creates_table():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        inspector = sa.inspect(conn)
        assert "platform_pipeline_defaults" in inspector.get_table_names()
        cols = {c["name"] for c in inspector.get_columns("platform_pipeline_defaults")}
        assert _EXPECTED_COLUMNS <= cols


def test_migration_0033_round_trips_row_with_model():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        conn.execute(sa.text(
            "INSERT INTO platform_pipeline_defaults (layer, provider, model, updated_at, updated_by) "
            "VALUES ('llm', 'groq', 'llama-3.3-70b-versatile', '2026-01-01 00:00:00', 'admin')"
        ))
        row = conn.execute(sa.text(
            "SELECT layer, provider, model, updated_by FROM platform_pipeline_defaults WHERE layer = 'llm'"
        )).one()
        assert row.layer == "llm"
        assert row.provider == "groq"
        assert row.model == "llama-3.3-70b-versatile"
        assert row.updated_by == "admin"


def test_migration_0033_round_trips_row_with_null_model():
    """Azure/Google TTS have no model dimension -- `model` is NULL."""
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        conn.execute(sa.text(
            "INSERT INTO platform_pipeline_defaults (layer, provider, model, updated_at, updated_by) "
            "VALUES ('tts', 'azure', NULL, '2026-01-01 00:00:00', NULL)"
        ))
        row = conn.execute(sa.text(
            "SELECT layer, provider, model, updated_by FROM platform_pipeline_defaults WHERE layer = 'tts'"
        )).one()
        assert row.provider == "azure"
        assert row.model is None
        assert row.updated_by is None


def test_migration_0033_layer_is_primary_key():
    """One row per layer -- a duplicate `layer` must be rejected."""
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        conn.execute(sa.text(
            "INSERT INTO platform_pipeline_defaults (layer, provider, model, updated_at, updated_by) "
            "VALUES ('stt', 'sarvam', 'saaras:v3', '2026-01-01 00:00:00', 'admin')"
        ))
        try:
            conn.execute(sa.text(
                "INSERT INTO platform_pipeline_defaults (layer, provider, model, updated_at, updated_by) "
                "VALUES ('stt', 'groq', 'whisper-large-v3', '2026-01-01 00:00:00', 'admin')"
            ))
            raised = False
        except sa.exc.IntegrityError:
            raised = True
        assert raised


def test_migration_0033_downgrade_drops_table():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        assert "platform_pipeline_defaults" in sa.inspect(conn).get_table_names()

        with Operations.context(ctx):
            mod.downgrade()
        assert "platform_pipeline_defaults" not in sa.inspect(conn).get_table_names()
