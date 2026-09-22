"""alembic/versions/0026_chat_turn_reply_length.py

Adds ``reply_chars``/``reply_words`` to ``chat_turn_metrics``: the length of
the turn's final, post-guard, customer-visible ``response_text`` in Unicode
codepoints and whitespace-split words respectively (see
``src/agents/chatbot.py``'s ``_measure_reply`` for the exact definition,
including the codepoint-vs-grapheme caveat). Measurement only -- no change to
reply behaviour, no truncation, no retry.

NULLABLE with no server default: NULL means "this row predates the
measurement" -- either an older process during a rolling deploy, or a
WS-layer turn-failure row that never produced a reply at all (see
``chat_turn_metrics.py``'s ``ChatTurnMetric.reply_chars``/``reply_words``
comment). Backfilling either case to 0 would feed a false observation into
any AVG/percentile query over these columns. AVG and percentile_cont both
skip NULLs.

Unlike the neighbouring ``result_chars`` (migration 0023), 0 IS a legitimate,
representable value on these two columns: it means "measured, reply
genuinely empty" -- an alarming observation, not a missing one -- and it
must stay distinguishable from NULL.

Integer, not BigInteger: a single chat reply overflowing a 32-bit int would
require billions of characters in one turn.

Revision: 0026_chat_turn_reply_length
Down: 0025_deposit_verification_order_idx
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0026_chat_turn_reply_length"
down_revision = "0025_deposit_verification_order_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_turn_metrics",
        sa.Column("reply_chars", sa.Integer(), nullable=True),
    )
    op.add_column(
        "chat_turn_metrics",
        sa.Column("reply_words", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_turn_metrics", "reply_words")
    op.drop_column("chat_turn_metrics", "reply_chars")
