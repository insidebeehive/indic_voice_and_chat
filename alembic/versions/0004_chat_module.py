"""ChatBot module: chat_sessions + chat_messages.

Customer-facing ChatBot conversations (PRD §7). ``kb_documents`` already
exists (0001); ``chat_tools`` ships with Phase 3.

Revision: 0004_chat_module
Down: 0003_provider_cost_model
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_chat_module"
down_revision = "0003_provider_cost_model"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("customer_id", sa.String(100)),
        sa.Column("customer_name", sa.String(255)),
        sa.Column("language", sa.String(10), server_default="hi"),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("message_count", sa.Integer, server_default="0"),
        sa.Column("started_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime),
        sa.Column("summary", sa.Text),
        sa.Column("satisfaction", sa.String(20)),
    )
    op.create_index("idx_chat_sessions_tenant", "chat_sessions", ["tenant_id", "status"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(50),
                  sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("type", sa.String(10), server_default="text"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("media_url", sa.String(500)),
        sa.Column("media_mime", sa.String(50)),
        sa.Column("sources", sa.JSON),
        sa.Column("tool_calls", sa.JSON),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_chat_messages_session", "chat_messages", ["session_id"])


def downgrade() -> None:
    op.drop_index("idx_chat_messages_session", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("idx_chat_sessions_tenant", table_name="chat_sessions")
    op.drop_table("chat_sessions")
