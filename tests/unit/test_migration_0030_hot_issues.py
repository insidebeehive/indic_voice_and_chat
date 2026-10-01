"""Round-trips the real 0030 migration module's upgrade()/downgrade() against
a throwaway sqlite schema (bare tenants/crms tables for the FKs) -- proves
both directions actually create/drop the intended table + indexes, and that
the CHECK/UNIQUE constraints declared in the migration actually hold. Same
mechanism as test_migration_0029_embedding_usage.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic" / "versions" / "0030_hot_issues.py"
)

_EXPECTED_COLUMNS = {
    "id", "tenant_id", "crm_id", "key", "title", "body",
    "expires_at", "created_at", "updated_at",
}


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "_test_migration_0030_hot_issues", _MIGRATION_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _bare_parent_tables(metadata: sa.MetaData) -> None:
    """Minimal tenants/crms tables -- just enough for hot_issues' FKs to
    resolve. Not the full real models."""
    sa.Table("tenants", metadata, sa.Column("id", sa.String(50), primary_key=True))
    sa.Table("crms", metadata, sa.Column("id", sa.String(50), primary_key=True))


def _upgraded_conn(engine):
    """Context-manager-free helper: creates the parent tables + hot_issues on
    a fresh in-memory sqlite engine and returns the open connection."""
    metadata = sa.MetaData()
    _bare_parent_tables(metadata)
    conn = engine.connect()
    trans = conn.begin()
    metadata.create_all(conn)
    conn.execute(sa.text("INSERT INTO tenants (id) VALUES ('t1')"))
    conn.execute(sa.text("INSERT INTO crms (id) VALUES ('c1')"))
    ctx = MigrationContext.configure(conn)
    with Operations.context(ctx):
        _load_migration_module().upgrade()
    trans.commit()
    return conn


def test_migration_0030_revision_and_down_revision():
    mod = _load_migration_module()
    assert mod.revision == "0030_hot_issues"
    assert len(mod.revision) <= 32  # alembic_version.version_num is VARCHAR(32)
    assert mod.down_revision == "0029_embedding_usage"
    assert mod.branch_labels is None
    assert mod.depends_on is None


def test_migration_0030_upgrade_creates_table_and_no_redundant_indexes():
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
        cols = {c["name"] for c in inspector.get_columns("hot_issues")}
        assert _EXPECTED_COLUMNS <= cols

        # No index at all on hot_issues: each UNIQUE constraint already leads
        # with tenant_id/crm_id respectively, so a dedicated index on either
        # column would be redundant (see the migration's own comment).
        # Asserting the full list (not just absence of two specific names)
        # catches ANY index the migration might add, named or not.
        assert inspector.get_indexes("hot_issues") == []


def test_migration_0030_round_trips_tenant_scoped_row():
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
            "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
            "VALUES ('t1', NULL, 'pg-delay', 'Deposits delayed', "
            "'Deposits may take up to 30 minutes.', '2099-01-01 00:00:00')"
        ))
        row = conn.execute(sa.text(
            "SELECT tenant_id, crm_id, key, title, body FROM hot_issues WHERE tenant_id = 't1'"
        )).one()
        assert row.tenant_id == "t1"
        assert row.crm_id is None
        assert row.key == "pg-delay"
        assert row.title == "Deposits delayed"


def test_migration_0030_round_trips_crm_scoped_row():
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
            "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
            "VALUES (NULL, 'c1', 'kyc-outage', 'KYC vendor down', "
            "'Verification may be slow today.', '2099-01-01 00:00:00')"
        ))
        row = conn.execute(sa.text(
            "SELECT tenant_id, crm_id, key FROM hot_issues WHERE crm_id = 'c1'"
        )).one()
        assert row.tenant_id is None
        assert row.crm_id == "c1"
        assert row.key == "kyc-outage"


def test_migration_0030_check_constraint_rejects_both_scopes_set():
    engine = sa.create_engine("sqlite:///:memory:")
    conn = _upgraded_conn(engine)
    try:
        with pytest.raises(sa.exc.IntegrityError):
            conn.execute(sa.text(
                "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
                "VALUES ('t1', 'c1', 'both-set', 'x', 'y', '2099-01-01 00:00:00')"
            ))
            conn.commit()
    finally:
        conn.close()


def test_migration_0030_check_constraint_rejects_neither_scope_set():
    """SQLite doesn't enforce foreign keys by default, so this CHECK
    constraint is the only thing actually guarding the one-scope invariant
    in this test suite -- the FK alone would not catch NULL/NULL."""
    engine = sa.create_engine("sqlite:///:memory:")
    conn = _upgraded_conn(engine)
    try:
        with pytest.raises(sa.exc.IntegrityError):
            conn.execute(sa.text(
                "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
                "VALUES (NULL, NULL, 'neither-set', 'x', 'y', '2099-01-01 00:00:00')"
            ))
            conn.commit()
    finally:
        conn.close()


def test_migration_0030_unique_constraint_holds_per_tenant():
    engine = sa.create_engine("sqlite:///:memory:")
    conn = _upgraded_conn(engine)
    try:
        conn.execute(sa.text(
            "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
            "VALUES ('t1', NULL, 'dup-key', 'x', 'y', '2099-01-01 00:00:00')"
        ))
        conn.commit()
        with pytest.raises(sa.exc.IntegrityError):
            conn.execute(sa.text(
                "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
                "VALUES ('t1', NULL, 'dup-key', 'x2', 'y2', '2099-01-01 00:00:00')"
            ))
            conn.commit()
    finally:
        conn.close()


def test_migration_0030_unique_constraint_holds_per_crm():
    engine = sa.create_engine("sqlite:///:memory:")
    conn = _upgraded_conn(engine)
    try:
        conn.execute(sa.text(
            "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
            "VALUES (NULL, 'c1', 'dup-key', 'x', 'y', '2099-01-01 00:00:00')"
        ))
        conn.commit()
        with pytest.raises(sa.exc.IntegrityError):
            conn.execute(sa.text(
                "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
                "VALUES (NULL, 'c1', 'dup-key', 'x2', 'y2', '2099-01-01 00:00:00')"
            ))
            conn.commit()
    finally:
        conn.close()


def test_migration_0030_same_key_allowed_across_different_scopes():
    """The same literal key in a tenant-scoped row and a crm-scoped row must
    not collide -- each UNIQUE constraint only bites within its own scope
    (NULL is never equal to NULL under a unique index)."""
    engine = sa.create_engine("sqlite:///:memory:")
    conn = _upgraded_conn(engine)
    try:
        conn.execute(sa.text(
            "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
            "VALUES ('t1', NULL, 'shared-key', 'x', 'y', '2099-01-01 00:00:00')"
        ))
        conn.execute(sa.text(
            "INSERT INTO hot_issues (tenant_id, crm_id, key, title, body, expires_at) "
            "VALUES (NULL, 'c1', 'shared-key', 'x2', 'y2', '2099-01-01 00:00:00')"
        ))
        conn.commit()
        count = conn.execute(sa.text(
            "SELECT COUNT(*) FROM hot_issues WHERE key = 'shared-key'"
        )).scalar()
        assert count == 2
    finally:
        conn.close()


def test_migration_0030_downgrade_drops_indexes_and_table():
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    _bare_parent_tables(metadata)
    with engine.begin() as conn:
        metadata.create_all(conn)
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        assert "hot_issues" in sa.inspect(conn).get_table_names()

        with Operations.context(ctx):
            mod.downgrade()
        assert "hot_issues" not in sa.inspect(conn).get_table_names()
        # Sibling tables untouched.
        assert "tenants" in sa.inspect(conn).get_table_names()
        assert "crms" in sa.inspect(conn).get_table_names()
