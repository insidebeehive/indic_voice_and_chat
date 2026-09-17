"""Round-trips the real 0022 migration module's upgrade()/downgrade() against
a throwaway sqlite table shaped like provider_costs -- proves both directions
actually execute the intended add/drop, rather than trusting the file by
inspection. No other migration in this repo has a test like this (there's no
existing alembic-test infra), so this loads the migration file directly via
importlib (its module name starts with a digit, so a normal `import` can't
reach it) and drives it through alembic's own Operations.context(), the same
mechanism alembic's env.py uses to bind the module-level `op` object.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic" / "versions" / "0022_provider_cost_cached_rate.py"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "_test_migration_0022_provider_cost_cached_rate", _MIGRATION_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_migration_0022_upgrade_adds_nullable_column_without_data_loss():
    """upgrade() adds cost_per_1k_cached_tokens, and an existing row's other
    columns are untouched -- an ADD COLUMN that somehow required a table
    rewrite losing data would be a much worse bug than a missing migration."""
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    provider_costs = sa.Table(
        "provider_costs", metadata,
        sa.Column("kind", sa.String(20), primary_key=True),
        sa.Column("provider", sa.String(40), primary_key=True),
        sa.Column("model", sa.String(60), primary_key=True),
        sa.Column("cost_per_min", sa.Float),
        sa.Column("cost_per_1k_input_tokens", sa.Float),
        sa.Column("cost_per_1k_output_tokens", sa.Float),
    )
    with engine.begin() as conn:
        metadata.create_all(conn)
        conn.execute(provider_costs.insert().values(
            kind="llm", provider="gemini", model="gemini-3.5-flash",
            cost_per_min=0.002, cost_per_1k_input_tokens=0.0003,
            cost_per_1k_output_tokens=0.0025,
        ))

        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()

        cols = {c["name"] for c in sa.inspect(conn).get_columns("provider_costs")}
        assert "cost_per_1k_cached_tokens" in cols

        row = conn.execute(sa.text(
            "SELECT cost_per_min, cost_per_1k_input_tokens, cost_per_1k_cached_tokens "
            "FROM provider_costs WHERE model = 'gemini-3.5-flash'"
        )).one()
        assert row.cost_per_min == 0.002
        assert row.cost_per_1k_input_tokens == 0.0003
        # New column on a pre-existing row: NULL, i.e. "unconfigured" -- this
        # is the whole point (see the migration's own docstring): upgrading
        # must not change a single existing dollar figure.
        assert row.cost_per_1k_cached_tokens is None


def test_migration_0022_downgrade_drops_the_column():
    """downgrade() is the exact inverse: the column disappears and nothing
    else about the table does."""
    mod = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    sa.Table(
        "provider_costs", metadata,
        sa.Column("kind", sa.String(20), primary_key=True),
        sa.Column("provider", sa.String(40), primary_key=True),
        sa.Column("model", sa.String(60), primary_key=True),
        sa.Column("cost_per_min", sa.Float),
        sa.Column("cost_per_1k_input_tokens", sa.Float),
        sa.Column("cost_per_1k_output_tokens", sa.Float),
    )
    with engine.begin() as conn:
        metadata.create_all(conn)
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        assert "cost_per_1k_cached_tokens" in {
            c["name"] for c in sa.inspect(conn).get_columns("provider_costs")
        }

        with Operations.context(ctx):
            mod.downgrade()
        cols = {c["name"] for c in sa.inspect(conn).get_columns("provider_costs")}
        assert "cost_per_1k_cached_tokens" not in cols
        assert "cost_per_1k_input_tokens" in cols  # untouched sibling column
