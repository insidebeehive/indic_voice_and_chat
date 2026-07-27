"""alembic/versions/0012_tts_segments_dropped.py

Add turn_metrics.tts_segments_dropped — a count of TTS sentences that
failed/timed out and were skipped during a turn, silently before this
change (a log line nobody greps for). Written by
VoiceBotAgent.apply_signal via record_turn_metric, same as every other
turn_metrics column.

Note: originally authored as "0012_turn_metrics_dropped_segments" (34
chars), which exceeds Alembic's default alembic_version.version_num
column width (VARCHAR(32)) and failed the version-stamp UPDATE against
real Postgres every time (the ADD COLUMN itself succeeded but the whole
migration transaction then rolled back). Renamed to fit before this was
ever successfully applied anywhere.

Revision: 0012_tts_segments_dropped
Down: 0011_turn_metrics
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_tts_segments_dropped"
down_revision = "0011_turn_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "turn_metrics",
        sa.Column("tts_segments_dropped", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("turn_metrics", "tts_segments_dropped")
