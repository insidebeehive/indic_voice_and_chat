"""alembic/versions/0022_provider_cost_cached_rate.py

Adds ``cost_per_1k_cached_tokens`` to ``provider_costs`` — the rate a
provider-reported cached input token bills at, separate from
``cost_per_1k_input_tokens``. Until now ``src/api/chat_cost.py`` billed every
input token at the fresh rate, so a heavily-cached prompt (Gemini explicit
caching measured 97.8% of a production-shaped prompt as cached — see
docs/llm-prompt-caching.md) was billed as if none of it were cached.

NULLABLE with no server default, unlike ``cost_per_1k_input_tokens`` /
``cost_per_1k_output_tokens`` (both NOT NULL, default 0.0). This is
deliberate, not an oversight: NULL has to mean "no cached rate configured
for this (provider, model)" so ``compute_chat_turn_cost`` can fall back to
the full input rate — today's exact behaviour — rather than to 0.0. Every
existing row (and every new row that doesn't set this column) ends up NULL
on this migration, so upgrading never changes a single dollar figure the
platform already reports; the rate only takes effect once a (kind='llm',
provider, model) row has it set explicitly (see
config/provider_costs.yaml's ``llm_token_rates`` / ``PUT
/api/v1/providers/{kind}/{provider}``).

Revision: 0022_provider_cost_cached_rate
Down: 0021_chat_turn_metrics_tokens
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0022_provider_cost_cached_rate"
down_revision = "0021_chat_turn_metrics_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "provider_costs",
        sa.Column("cost_per_1k_cached_tokens", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("provider_costs", "cost_per_1k_cached_tokens")
