"""alembic/versions/0010_crm_kb_documents.py

CRM-level KB documents: rename platform_kb_documents -> crm_kb_documents,
add a required crm_id FK, backfill existing rows to the one CRM that exists
today (fails loudly if more than one CRM exists — see _backfill_crm_id).

Revision: 0010_crm_kb_documents
Down: 0009_crm_entity
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_crm_kb_documents"
down_revision = "0009_crm_entity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table("platform_kb_documents", "crm_kb_documents")
    op.add_column(
        "crm_kb_documents",
        sa.Column("crm_id", sa.String(50), sa.ForeignKey("crms.id", ondelete="CASCADE"),
                  nullable=True),
    )
    op.create_index("idx_crm_kb_documents_crm", "crm_kb_documents", ["crm_id"])

    _backfill_crm_id()

    op.alter_column("crm_kb_documents", "crm_id", nullable=False)


def _backfill_crm_id() -> None:
    """Attribute every pre-existing (was-platform-wide) KB doc to the one CRM
    that exists today. Fails loudly (raises) if zero or more than one Crm row
    exists — silently guessing which CRM owns pre-existing global docs would
    be a real, silent data-attribution error."""
    conn = op.get_bind()
    crm_ids = [r[0] for r in conn.execute(sa.text("SELECT id FROM crms")).fetchall()]
    if len(crm_ids) != 1:
        raise RuntimeError(
            f"0010_crm_kb_documents: expected exactly one Crm row to backfill "
            f"crm_kb_documents.crm_id against, found {len(crm_ids)} ({crm_ids!r}). "
            "This migration cannot guess which CRM owns the pre-existing "
            "platform-wide KB documents — resolve manually (e.g. run "
            "'UPDATE crm_kb_documents SET crm_id = <the-right-id> WHERE crm_id "
            "IS NULL' by hand for each affected row) before retrying."
        )
    conn.execute(sa.text(
        "UPDATE crm_kb_documents SET crm_id = :crm_id WHERE crm_id IS NULL"
    ), {"crm_id": crm_ids[0]})

    # Best-effort: also backfill the pgvector knowledge_chunks table's platform
    # rows (tenant_id IS NULL) to the same crm_id, IF that table exists and
    # already has a crm_id column. It's NOT Alembic-managed (see
    # docs/pgvector_setup.sql — created once by a superuser, not by migrations),
    # so this is a data-only UPDATE, never a DDL statement, and is skipped
    # entirely (not a failure) if the table or column isn't there yet — e.g. a
    # FAISS-only environment, or an environment where the DDL snippet in
    # docs/pgvector_setup.sql hasn't been applied yet.
    has_table = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'voicebot' AND table_name = 'knowledge_chunks'"
    )).fetchone()
    if not has_table:
        return
    has_column = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'voicebot' AND table_name = 'knowledge_chunks' "
        "AND column_name = 'crm_id'"
    )).fetchone()
    if not has_column:
        return
    conn.execute(sa.text(
        "UPDATE voicebot.knowledge_chunks SET crm_id = :crm_id "
        "WHERE tenant_id IS NULL AND crm_id IS NULL"
    ), {"crm_id": crm_ids[0]})


def downgrade() -> None:
    op.drop_index("idx_crm_kb_documents_crm", table_name="crm_kb_documents")
    op.drop_column("crm_kb_documents", "crm_id")
    op.rename_table("crm_kb_documents", "platform_kb_documents")
