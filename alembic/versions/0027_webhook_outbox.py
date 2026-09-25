"""alembic/versions/0027_webhook_outbox.py

Adds ``webhook_outbox``: a durable retry queue for ``session_closed``
tenant-lifecycle webhooks (see ``src/models/webhook_outbox.py`` for the full
rationale -- in short, the tenant's CRM endpoint can intermittently exceed
``src.integration.tenant_events.deliver``'s in-line retry budget, ~16s, and
without a durable queue that's the only chance the CRM ever gets to learn a
session ended, leaving its ticket open forever).

Deliberately scoped to ``session_closed`` only, never ``escalation_requested``
-- see the model module's docstring for why.

``body`` is JSON (not JSONB-specific) so this also renders on SQLite (tests).

Indexed on (status, next_attempt_at) for the background loop's claim query:
pending rows due now, oldest-due first.

Revision: 0027_webhook_outbox
Down: 0026_chat_turn_reply_length
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0027_webhook_outbox"
down_revision = "0026_chat_turn_reply_length"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "webhook_outbox",
        sa.Column("id", sa.String(length=50), nullable=False),
        sa.Column("tenant_id", sa.String(length=50), nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("url", sa.String(length=2000), nullable=False),
        sa.Column("body", sa.JSON(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("delivered_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "idx_webhook_outbox_tenant_id", "webhook_outbox", ["tenant_id"],
    )
    op.create_index(
        "idx_webhook_outbox_status", "webhook_outbox", ["status"],
    )
    op.create_index(
        "idx_webhook_outbox_status_next_attempt", "webhook_outbox",
        ["status", "next_attempt_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_webhook_outbox_status_next_attempt", table_name="webhook_outbox")
    op.drop_index("idx_webhook_outbox_status", table_name="webhook_outbox")
    op.drop_index("idx_webhook_outbox_tenant_id", table_name="webhook_outbox")
    op.drop_table("webhook_outbox")
