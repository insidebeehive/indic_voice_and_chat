"""alembic/versions/0034_tenant_billing_snapshots.py

Adds the ``tenant_billing_snapshots`` table: immutable monthly billing
snapshots per tenant, frozen so a bill already shown/sent to a tenant never
silently changes afterward even if the underlying ``conversations``/
``chat_sessions`` rows are deleted (e.g. via tenant deletion's cascade) or a
``provider_costs`` telephony rate is edited retroactively. See
``src/models/billing_snapshot.py``'s ``TenantBillingSnapshot`` for the same
columns restated next to the model, and ``src/api/billing.py`` for how a
snapshot is computed/written (``snapshot_tenant_month`` /
``snapshot_previous_month_all_tenants``) and ``src/api/tenants.py`` for the
admin endpoints that read/write this table.

``tenant_id`` is deliberately NOT a foreign key to ``tenants.id`` -- a
snapshot must survive the tenant row (and its cascade-deleted children)
being deleted.

Revision: 0034_tenant_billing_snapshots
Down: 0033_platform_pipeline_defaults
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0034_tenant_billing_snapshots"
down_revision = "0033_platform_pipeline_defaults"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenant_billing_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(length=50), nullable=False),
        sa.Column("tenant_name", sa.String(length=255), nullable=False),
        sa.Column("period_month", sa.Date(), nullable=False),
        sa.Column("total_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("billable_minutes", sa.Float(), nullable=False, server_default="0"),
        sa.Column("platform_cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_cost_per_call", sa.Float(), nullable=False, server_default="0"),
        sa.Column("tentative_telephony_cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("telephony_rates", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("chat_sessions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chat_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chat_output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chat_cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("timezone", sa.String(length=64), nullable=False, server_default="UTC"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.String(length=100), nullable=False, server_default="auto"),
        sa.UniqueConstraint("tenant_id", "period_month", name="uq_billing_snapshot_tenant_month"),
    )
    # No standalone tenant_id index -- the unique (tenant_id, period_month)
    # constraint above already indexes tenant_id as its leftmost column,
    # which covers a tenant_id-only lookup just as well as a second,
    # redundant index would (see src/models/billing_snapshot.py).


def downgrade() -> None:
    op.drop_table("tenant_billing_snapshots")
