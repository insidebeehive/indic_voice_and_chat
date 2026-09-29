"""Round-trips the real 0029 migration module's upgrade()/downgrade() against
a throwaway sqlite schema (bare tenants/crms tables for the FKs) -- proves
both directions actually create/drop the intended table + indexes, rather
than trusting the file by inspection. Same mechanism as
test_migration_0028_chat_turn_media_cost.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic" / "versions" / "0029_embedding_usage.py"
)

_EXPECTED_COLUMNS = {
    "id", "tenant_id", "crm_id", "purpose", "provider", "model",
    "input_chars", "tokens", "cost", "created_at",
}
_EXPECTED_INDEXES = {
    "idx_embedding_usage_tenant_created",
    "idx_embedding_usage_crm_created",
    "idx_embedding_usage_created_at",
}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "_test_migration_0029_embedding_usage", _MIGRATION_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _bare_parent_tables(metadata: sa.MetaData) -> None:
    """Minimal tenants/crms tables -- just enough for embedding_usage's FKs
    to resolve. Not the full real models."""
    sa.Table("tenants", metadata, sa.Column("id", sa.String(50), primary_key=True))
    sa.Table("crms", metadata, sa.Column("id", sa.String(50), primary_key=True))


def test_migration_0029_revision_and_down_revision():
    mod = _load_migration_module()
    assert mod.revision == "0029_embedding_usage"
    assert len(mod.revision) <= 32  # alembic_version.version_num is VARCHAR(32)
    assert mod.down_revision == "0028_chat_turn_media_cost"
    assert mod.branch_labels is None
    assert mod.depends_on is None


def test_migration_0029_upgrade_creates_table_and_indexes():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    _bare_parent_tables(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        conn.execute(sa.text("INSERT INTO tenants (id) VALUES ('t1')"))
        conn.execute(sa.text("INSERT INTO crms (id) VALUES ('c1')"))

        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        inspector = sa.inspect(conn)
        cols = {c["name"] for c in inspector.get_columns("embedding_usage")}
        assert _EXPECTED_COLUMNS <= cols

        idx_names = {ix["name"] for ix in inspector.get_indexes("embedding_usage")}
        assert _EXPECTED_INDEXES <= idx_names


def test_migration_0029_round_trips_tenant_scoped_row_with_null_tokens():
    """tenant_id set / crm_id NULL / tokens NULL (estimated-from-chars case)."""
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    _bare_parent_tables(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        conn.execute(sa.text("INSERT INTO tenants (id) VALUES ('t1')"))
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        conn.execute(sa.text(
            "INSERT INTO embedding_usage "
            "(tenant_id, crm_id, purpose, provider, model, input_chars, tokens, cost) "
            "VALUES ('t1', NULL, 'ingest', 'gemini', 'gemini-embedding-001', 4000, NULL, 0.00015)"
        ))
        row = conn.execute(sa.text(
            "SELECT tenant_id, crm_id, purpose, provider, model, input_chars, tokens, cost, created_at "
            "FROM embedding_usage WHERE tenant_id = 't1'"
        )).one()
        assert row.tenant_id == "t1"
        assert row.crm_id is None
        assert row.purpose == "ingest"
        assert row.provider == "gemini"
        assert row.model == "gemini-embedding-001"
        assert row.input_chars == 4000
        assert row.tokens is None
        assert abs(row.cost - 0.00015) < 1e-9
        assert row.created_at is not None  # server_default populated it


def test_migration_0029_round_trips_crm_scoped_row_with_explicit_tokens():
    """crm_id set / tenant_id NULL / tokens explicit (provider-reported case)."""
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    _bare_parent_tables(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        conn.execute(sa.text("INSERT INTO crms (id) VALUES ('c1')"))
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        conn.execute(sa.text(
            "INSERT INTO embedding_usage "
            "(tenant_id, crm_id, purpose, provider, model, input_chars, tokens, cost) "
            "VALUES (NULL, 'c1', 'search', 'gemini', 'gemini-embedding-001', 40, 12, 0.0000018)"
        ))
        row = conn.execute(sa.text(
            "SELECT tenant_id, crm_id, purpose, tokens FROM embedding_usage WHERE crm_id = 'c1'"
        )).one()
        assert row.tenant_id is None
        assert row.crm_id == "c1"
        assert row.purpose == "search"
        assert row.tokens == 12


def test_migration_0029_downgrade_drops_indexes_and_table():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    _bare_parent_tables(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        assert "embedding_usage" in sa.inspect(conn).get_table_names()

        with Operations.context(ctx):
            mod.downgrade()
        assert "embedding_usage" not in sa.inspect(conn).get_table_names()
        # Sibling tables untouched.
        assert "tenants" in sa.inspect(conn).get_table_names()
        assert "crms" in sa.inspect(conn).get_table_names()
