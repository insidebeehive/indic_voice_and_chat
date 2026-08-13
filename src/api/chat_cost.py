"""Chat (text) turn cost — per-token LLM billing.

Voice calls are billed per-minute (``src/api/call_store.py``); chat is billed
per-token, since a chat turn has no meaningful duration. This mirrors
``call_store.py``'s rate-lookup + cost-compute shape, reading from the same
``provider_costs`` catalog (``cost_per_1k_input_tokens`` /
``cost_per_1k_output_tokens`` columns, added alongside the existing
``cost_per_min``).
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.tenant import ProviderCost

log = logging.getLogger(__name__)


async def token_rates(session: AsyncSession, provider: str, model: str) -> tuple[float, float]:
    """(cost_per_1k_input_tokens, cost_per_1k_output_tokens) for (provider, model);
    falls back to the provider-level ("") row, same pattern as call_store._rate."""
    row = await session.get(ProviderCost, ("llm", provider, model or ""))
    if model and (row is None or (not row.cost_per_1k_input_tokens and not row.cost_per_1k_output_tokens)):
        fallback = await session.get(ProviderCost, ("llm", provider, ""))
        if fallback is not None:
            row = fallback
    if row is None:
        return 0.0, 0.0
    return row.cost_per_1k_input_tokens, row.cost_per_1k_output_tokens


async def compute_chat_turn_cost(
    session: AsyncSession,
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Platform-billed cost for one chat turn = token-rate-weighted sum."""
    if not provider or (not input_tokens and not output_tokens):
        return 0.0
    in_rate, out_rate = await token_rates(session, provider, model)
    if in_rate == 0.0 and out_rate == 0.0:
        log.warning(
            "chat cost resolved to $0 for %s/%s despite %d+%d tokens - "
            "missing ProviderCost token rate row?",
            provider, model, input_tokens, output_tokens,
        )
    cost = (input_tokens / 1000.0 * in_rate) + (output_tokens / 1000.0 * out_rate)
    return round(cost, 6)
