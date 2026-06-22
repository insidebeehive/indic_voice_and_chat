"""Widen chat_messages.role from VARCHAR(10) to VARCHAR(20).

The original column was sized for 'customer'|'agent'|'system', but
'human_agent' (11 chars) was added for BO handover messages and didn't fit.

Revision: 0008_widen_chat_message_role
Down: 0007_chat_handover
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_widen_chat_message_role"
down_revision = "0007_chat_handover"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "chat_messages", "role",
        existing_type=sa.String(10),
        type_=sa.String(20),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "chat_messages", "role",
        existing_type=sa.String(20),
        type_=sa.String(10),
        existing_nullable=False,
    )
