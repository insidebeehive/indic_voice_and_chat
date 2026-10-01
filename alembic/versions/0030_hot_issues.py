"""alembic/versions/0030_hot_issues.py

Adds the ``hot_issues`` table: operator-posted live-incident notices, scoped
to exactly one of a tenant or a CRM, injected directly into the chat/voice
system prompt while active (not stored in the RAG KB — see
``src/chatbot/hot_issues.py``'s module docstring for why). See
``src/models/hot_issue.py``'s ``HotIssue`` for the same constraints restated
next to the columns.

``ck_hot_issues_one_scope`` is the first CHECK constraint in this repo —
valid on both SQLite and Postgres. The two UNIQUE constraints each only bite
within their own scope: NULL is never equal to NULL under a unique index, so
a tenant-scoped row (crm_id NULL) never collides with another tenant-scoped
row under ``uq_hot_issues_crm_key``, and vice versa.

Revision: 0030_hot_issues
Down: 0029_embedding_usage
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0030_hot_issues"
down_revision = "0029_embedding_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hot_issues",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id", sa.String(length=50),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column(
            "crm_id", sa.String(length=50),
            sa.ForeignKey("crms.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=100), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "(tenant_id IS NULL) <> (crm_id IS NULL)", name="ck_hot_issues_one_scope",
        ),
        sa.UniqueConstraint("tenant_id", "key", name="uq_hot_issues_tenant_key"),
        sa.UniqueConstraint("crm_id", "key", name="uq_hot_issues_crm_key"),
    )
    # No separate single-column indexes on tenant_id/crm_id: each UNIQUE
    # constraint above already leads with that same column, so a dedicated
    # index would be redundant. Matches src/models/hot_issue.py's HotIssue
    # (no index=True on either column) so alembic autogenerate sees no drift.


def downgrade() -> None:
    op.drop_table("hot_issues")
