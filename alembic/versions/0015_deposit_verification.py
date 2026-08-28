"""Deposit dispute screenshot verification: deposit_verification_requests.

Tracks a customer's claim that a deposit went through but wasn't credited —
one row per claim, keyed to the tenant and the chat session it was raised in.
``status`` starts pending and moves to verified/rejected/timed_out/error;
the composite index supports a periodic sweep for requests still pending
past their ``timeout_at``.

Revision: 0015_deposit_verification
Down: 0014_chat_cost
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0015_deposit_verification"
down_revision = "0014_chat_cost"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "deposit_verification_requests",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(50),
                  sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("order_id", sa.String(200), nullable=False),
        sa.Column("screenshot_message_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("verdict_payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("timeout_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "idx_deposit_verification_requests_tenant",
        "deposit_verification_requests", ["tenant_id"],
    )
    op.create_index(
        "idx_deposit_verification_requests_session",
        "deposit_verification_requests", ["session_id"],
    )
    op.create_index(
        "idx_deposit_verification_requests_tenant_status_timeout",
        "deposit_verification_requests", ["tenant_id", "status", "timeout_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_deposit_verification_requests_tenant_status_timeout",
        table_name="deposit_verification_requests",
    )
    op.drop_index(
        "idx_deposit_verification_requests_session",
        table_name="deposit_verification_requests",
    )
    op.drop_index(
        "idx_deposit_verification_requests_tenant",
        table_name="deposit_verification_requests",
    )
    op.drop_table("deposit_verification_requests")
