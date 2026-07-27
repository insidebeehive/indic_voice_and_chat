"""alembic/versions/0011_turn_metrics.py

Per-turn STT/LLM/TTS latency metrics table, for cross-provider benchmarking.
Written by VoiceBotAgent.apply_signal (src/agents/voicebot.py) via
record_turn_metric (src/models/turn_metrics.py) on every completed layered
(cascade) turn.

Revision: 0011_turn_metrics
Down: 0010_crm_kb_documents
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_turn_metrics"
down_revision = "0010_crm_kb_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "turn_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(50), sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("campaign_id", sa.String(100), nullable=True),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("stt_provider", sa.String(100), nullable=True),
        sa.Column("llm_provider", sa.String(100), nullable=False),
        sa.Column("tts_provider", sa.String(100), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("stt_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_ttft_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_total_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tts_first_chunk_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tts_total_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_turn_metrics_tenant", "turn_metrics", ["tenant_id"])
    op.create_index(
        "idx_turn_metrics_combo", "turn_metrics",
        ["stt_provider", "llm_provider", "tts_provider"],
    )


def downgrade() -> None:
    op.drop_index("idx_turn_metrics_combo", table_name="turn_metrics")
    op.drop_index("idx_turn_metrics_tenant", table_name="turn_metrics")
    op.drop_table("turn_metrics")
