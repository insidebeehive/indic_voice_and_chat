"""alembic/versions/0033_platform_pipeline_defaults.py

Adds the ``platform_pipeline_defaults`` table: admin-editable, live-applied
overrides for the platform's STT/LLM/TTS provider+model, one row per layer
(``layer`` is the primary key, so at most one platform-wide default per
layer). See ``src/models/platform_pipeline.py``'s ``PlatformPipelineDefault``
for the same columns restated next to the model, and
``src/api/platform.py`` for the admin endpoints that read/write this table
and apply a change to the live process.

Revision: 0033_platform_pipeline_defaults
Down: 0032_txn_type_withdrawal
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0033_platform_pipeline_defaults"
down_revision = "0032_txn_type_withdrawal"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_pipeline_defaults",
        sa.Column("layer", sa.String(length=10), primary_key=True),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.String(length=100), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("platform_pipeline_defaults")
