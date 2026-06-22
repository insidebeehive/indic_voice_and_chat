"""Platform-level KB: dedicated table for platform docs (no tenant FK).

Instead of making kb_documents.tenant_id nullable (requires table ownership),
we create a separate platform_kb_documents table with no tenant constraint.

Revision: 0006_platform_kb
Down: 0005_chat_tools
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_platform_kb"
down_revision = "0005_chat_tools"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_kb_documents",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(50)),
        sa.Column("language", sa.String(10)),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("ingested_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("platform_kb_documents")
