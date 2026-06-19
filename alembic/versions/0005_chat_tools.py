"""ChatBot CRM tool registry: chat_tools (PRD §4.6).

Revision: 0005_chat_tools
Down: 0004_chat_module
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_chat_tools"
down_revision = "0004_chat_module"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_tools",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("endpoint", sa.String(500), nullable=False),
        sa.Column("method", sa.String(10), server_default="GET"),
        sa.Column("auth_type", sa.String(20)),
        sa.Column("auth_config", sa.JSON, server_default="{}"),
        sa.Column("parameters", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "name", name="uq_chat_tools_tenant_name"),
    )
    op.create_index("idx_chat_tools_tenant", "chat_tools", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("idx_chat_tools_tenant", table_name="chat_tools")
    op.drop_table("chat_tools")
