"""alembic/versions/0023_chat_tool_result_chars.py

Adds ``result_chars`` to ``chat_tool_metrics``: the character length of a
tool result's JSON as actually sent to the model. Measurement only -- no
change to tool behaviour, limits, or serialisation.

NULLABLE with no server default: NULL means "this row predates the
measurement" (see ``chat_turn_metrics.py``'s ``ChatToolMetricRow.result_chars``
comment); backfilling to 0 would feed a false observation into any AVG/
percentile query over this column, and 0 is never a legitimate value anyway
(``json.dumps`` of a result dict is at least ``"{}"``, 2 chars). AVG and
percentile_cont both skip NULLs.

Integer, not BigInteger: overflowing a 32-bit int would take a single 2GB
tool result.

Revision ID kept under alembic_version.version_num's VARCHAR(32) limit --
see migration 0019 / commit d889604, where an over-length id could never be
recorded and was silently skipped on every deploy.

Revision: 0023_chat_tool_result_chars
Down: 0022_provider_cost_cached_rate
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0023_chat_tool_result_chars"
down_revision = "0022_provider_cost_cached_rate"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_tool_metrics",
        sa.Column("result_chars", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_tool_metrics", "result_chars")
