"""Chat (text) cost tracking — per-token LLM rates + per-turn/session cost.

Voice calls have been cost-tracked since day one (``conversations.cost``,
``provider_costs.cost_per_min``); chat sessions had none. Chat is billed
per-token (not per-minute like voice), so this adds a second rate pair to
``provider_costs`` and cost/token columns to ``chat_messages`` (per agent
turn) and ``chat_sessions`` (running total).

``chat_messages`` cost columns are nullable — only the agent's own turns
have an LLM cost; a customer/human-agent message has none.
``chat_sessions`` cost columns are NOT NULL with a 0 default so the running
total never needs coalescing.

No backfill: historical chat sessions simply show $0 cost — expected.

Revision: 0014_chat_cost
Down: 0013_crm_livekit_secrets
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014_chat_cost"
down_revision = "0013_crm_livekit_secrets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "provider_costs",
        sa.Column("cost_per_1k_input_tokens", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "provider_costs",
        sa.Column("cost_per_1k_output_tokens", sa.Float(), nullable=False, server_default="0"),
    )

    op.add_column("chat_messages", sa.Column("input_tokens", sa.Integer(), nullable=True))
    op.add_column("chat_messages", sa.Column("output_tokens", sa.Integer(), nullable=True))
    op.add_column("chat_messages", sa.Column("cost", sa.Float(), nullable=True))
    op.add_column("chat_messages", sa.Column("llm_provider", sa.String(30), nullable=True))
    op.add_column("chat_messages", sa.Column("llm_model", sa.String(60), nullable=True))

    op.add_column(
        "chat_sessions",
        sa.Column("cost", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "output_tokens")
    op.drop_column("chat_sessions", "input_tokens")
    op.drop_column("chat_sessions", "cost")

    op.drop_column("chat_messages", "llm_model")
    op.drop_column("chat_messages", "llm_provider")
    op.drop_column("chat_messages", "cost")
    op.drop_column("chat_messages", "output_tokens")
    op.drop_column("chat_messages", "input_tokens")

    op.drop_column("provider_costs", "cost_per_1k_output_tokens")
    op.drop_column("provider_costs", "cost_per_1k_input_tokens")
