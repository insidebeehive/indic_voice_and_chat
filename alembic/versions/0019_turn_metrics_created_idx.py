"""alembic/versions/0019_turn_metrics_created_idx.py

Adds a plain B-tree index on turn_metrics.created_at.

The Phase 2 observability aggregation job
(src/observability/turn_metrics_push.py) filters TurnMetric rows by
``created_at >= cutoff`` every ``METRICS_PUSH_INTERVAL_S`` seconds (default
60s) forever, but the table was previously indexed only on the
provider-combo columns (idx_turn_metrics_combo) and tenant_id
(idx_turn_metrics_tenant) -- neither covers this query, so every run did a
full sequential scan of an ever-growing table. created_at is a simple
monotonically-growing timestamp column, so a plain B-tree index is enough;
nothing more exotic (e.g. BRIN) is warranted here.

Revision: 0019_turn_metrics_created_idx
Down: 0018_crm_pronunciation_overrides
"""

from __future__ import annotations

from alembic import op

revision = "0019_turn_metrics_created_idx"
down_revision = "0018_crm_pronunciation_overrides"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("idx_turn_metrics_created_at", "turn_metrics", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_turn_metrics_created_at", table_name="turn_metrics")
