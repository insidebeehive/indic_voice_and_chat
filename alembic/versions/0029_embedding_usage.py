"""alembic/versions/0029_embedding_usage.py

Adds the ``embedding_usage`` table: one row per KB-embedding call (Gemini,
via ``GeminiEmbedder`` in ``src/rag/embeddings.py``), covering both the
ingest path (``HybridRetriever.index()``) and the search path
(``HybridRetriever._dense_search()``). See ``src/api/chat_cost.py`` for how
these rows are priced and ``src/models/embedding_usage.py``'s
``EmbeddingUsage`` for the same NULL/scoping semantics restated next to the
columns.

Scoping: exactly one of ``tenant_id``/``crm_id`` is set per row, mirroring
the pgvector chunk-scoping rule elsewhere in this codebase (a tenant-KB row
has ``tenant_id`` set and ``crm_id`` NULL; a CRM shared-KB row has ``crm_id``
set and ``tenant_id`` NULL).

``tokens`` is NULLABLE with no backfill: the Gemini Developer API's
``embed_content`` response reports no token count (unlike Vertex AI), so
NULL means "the embedder reported no usage count" and the row's ``cost`` was
instead estimated from ``input_chars`` (see ``src/api/chat_cost.py``'s
``_EMBED_CHARS_PER_TOKEN``). Never backfill a NULL ``tokens`` to 0 — that
would claim a real, reported zero-token batch where none was observed.

No PII: only provider/model names, lengths, and numbers — never chunk or
query text.

The write path (``src/models/embedding_usage.py``'s
``record_embedding_usage``) is best-effort and swallows every exception, so
a deploy that reaches this table missing (e.g. this migration hasn't run
yet on a given environment) produces zero rows and a WARNING log line, never
an outage.

Revision: 0029_embedding_usage
Down: 0028_chat_turn_media_cost
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0029_embedding_usage"
down_revision = "0028_chat_turn_media_cost"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embedding_usage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id", sa.String(length=50),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column(
            "crm_id", sa.String(length=50),
            sa.ForeignKey("crms.id", ondelete="CASCADE"), nullable=True,
        ),
        sa.Column("purpose", sa.String(length=20), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("input_chars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens", sa.Integer(), nullable=True),
        sa.Column("cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_embedding_usage_tenant_created", "embedding_usage",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "idx_embedding_usage_crm_created", "embedding_usage",
        ["crm_id", "created_at"],
    )
    op.create_index(
        "idx_embedding_usage_created_at", "embedding_usage", ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_embedding_usage_created_at", table_name="embedding_usage")
    op.drop_index("idx_embedding_usage_crm_created", table_name="embedding_usage")
    op.drop_index("idx_embedding_usage_tenant_created", table_name="embedding_usage")
    op.drop_table("embedding_usage")
