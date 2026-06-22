"""Platform-level KB: make kb_documents.tenant_id nullable (NULL = platform).

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
    # Drop the FK + NOT NULL constraint, then re-add FK as nullable.
    with op.batch_alter_table("kb_documents") as batch_op:
        batch_op.alter_column(
            "tenant_id",
            existing_type=sa.String(50),
            nullable=True,
        )


def downgrade() -> None:
    # Delete platform rows (tenant_id IS NULL) before restoring NOT NULL.
    op.execute("DELETE FROM kb_documents WHERE tenant_id IS NULL")
    with op.batch_alter_table("kb_documents") as batch_op:
        batch_op.alter_column(
            "tenant_id",
            existing_type=sa.String(50),
            nullable=False,
        )
