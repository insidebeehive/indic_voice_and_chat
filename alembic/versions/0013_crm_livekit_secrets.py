"""crms.livekit_url + crm_secrets table — CRM-level LiveKit credentials.

A LiveKit project belongs to the CRM partner, not any one tenant, so its
URL/key/secret move up to the shared CRM-level scope (mirrors how
``crms.base_url``/``events_webhook_url_template`` already work). ``crm_secrets``
mirrors ``tenant_secrets`` exactly (Fernet-encrypted at rest), just keyed by
``crm_id`` instead of ``tenant_id``.

Revision: 0013_crm_livekit_secrets
Down: 0012_tts_segments_dropped
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0013_crm_livekit_secrets"
down_revision = "0012_tts_segments_dropped"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("crms", sa.Column("livekit_url", sa.String(500)))
    op.create_table(
        "crm_secrets",
        sa.Column("crm_id", sa.String(50),
                  sa.ForeignKey("crms.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("value_encrypted", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("crm_secrets")
    op.drop_column("crms", "livekit_url")
