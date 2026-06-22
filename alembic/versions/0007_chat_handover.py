"""Chat handover: add mode, claimed_by, claimed_at to chat_sessions.

mode values: 'ai' (default) | 'awaiting_human' | 'human' | 'closed'

Revision: 0007_chat_handover
Down: 0006_platform_kb
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_chat_handover"
down_revision = "0006_platform_kb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_sessions",
        sa.Column("mode", sa.String(20), nullable=False, server_default="ai"))
    op.add_column("chat_sessions",
        sa.Column("claimed_by", sa.String(100), nullable=True))
    op.add_column("chat_sessions",
        sa.Column("claimed_at", sa.DateTime, nullable=True))


def downgrade() -> None:
    op.drop_column("chat_sessions", "claimed_at")
    op.drop_column("chat_sessions", "claimed_by")
    op.drop_column("chat_sessions", "mode")
