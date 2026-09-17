"""Chat (text) turn cost — per-token LLM billing.

Voice calls are billed per-minute (``src/api/call_store.py``); chat is billed
per-token, since a chat turn has no meaningful duration. This mirrors
``call_store.py``'s rate-lookup + cost-compute shape, reading from the same
``provider_costs`` catalog (``cost_per_1k_input_tokens`` /
``cost_per_1k_output_tokens`` columns, added alongside the existing
``cost_per_min``).

Cached-token awareness (``cost_per_1k_cached_tokens``): a provider-reported
cached input token is billed separately from a fresh one. Gemini's explicit
context caching (behind ``GEMINI_EXPLICIT_CACHE``) measured 97.8% of a
production-shaped prompt as cached (docs/llm-prompt-caching.md) — billing
that at the fresh rate makes every cost figure this platform reports wrong
in the direction of "too expensive", by whatever discount the provider
actually applies to a cache hit.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.tenant import ProviderCost

log = logging.getLogger(__name__)


async def token_rates(
    session: AsyncSession, provider: str, model: str
) -> tuple[float, float, float | None]:
    """(cost_per_1k_input_tokens, cost_per_1k_output_tokens, cost_per_1k_cached_tokens)
    for (provider, model); falls back to the provider-level ("") row, same
    pattern as call_store._rate.

    The cached rate is returned exactly as stored — ``None`` when no cached
    rate is configured on the resolved row, ``0.0`` when one was explicitly
    configured as free. Collapsing that distinction here (e.g. by doing
    ``row.cost_per_1k_cached_tokens or 0.0``) would make it impossible for
    compute_chat_turn_cost to tell "bill cached tokens at the input rate"
    apart from "bill them at $0" — see that function's fallback.
    """
    row = await session.get(ProviderCost, ("llm", provider, model or ""))
    if model and (row is None or (not row.cost_per_1k_input_tokens and not row.cost_per_1k_output_tokens)):
        fallback = await session.get(ProviderCost, ("llm", provider, ""))
        if fallback is not None:
            row = fallback
    if row is None:
        return 0.0, 0.0, None
    return row.cost_per_1k_input_tokens, row.cost_per_1k_output_tokens, row.cost_per_1k_cached_tokens


async def compute_chat_turn_cost(
    session: AsyncSession,
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Platform-billed cost for one chat turn.

    cost = (input_tokens - cached_tokens) * in_rate
         +  cached_tokens               * cached_rate
         +  output_tokens               * out_rate

    ``cached_tokens`` defaults to 0 so a caller with no cached figure (e.g. a
    provider that never reports one) gets today's pre-cache-aware behaviour
    unchanged: every input token bills at ``in_rate``.
    """
    if not provider or (not input_tokens and not output_tokens):
        return 0.0
    in_rate, out_rate, cached_rate = await token_rates(session, provider, model)
    if in_rate == 0.0 and out_rate == 0.0:
        log.warning(
            "chat cost resolved to $0 for %s/%s despite %d+%d tokens - "
            "missing ProviderCost token rate row?",
            provider, model, input_tokens, output_tokens,
        )
    # A provider report is not a trusted input: cached_tokens should always be
    # <= input_tokens (see ChatTurnMetric's own docstring in
    # src/agents/chatbot.py), but that is not enforced anywhere upstream, so
    # an inconsistent report here must never be allowed to make fresh_tokens
    # negative — that would silently make the bill SMALLER as cached_tokens
    # grows past input_tokens, the opposite of what a real cache hit does.
    if cached_tokens < 0:
        cached_tokens = 0
    if cached_tokens > input_tokens:
        log.warning(
            "chat cost: cached_tokens (%d) exceeds input_tokens (%d) for %s/%s - "
            "clamping to input_tokens",
            cached_tokens, input_tokens, provider, model,
        )
        cached_tokens = input_tokens
    fresh_tokens = input_tokens - cached_tokens
    # Safety default: no configured cached rate bills cached tokens at the
    # FULL input rate, never at 0.0. A missing ProviderCost.cost_per_1k_cached_tokens
    # row silently collapsing cached-token cost to near-free would look like a
    # spectacular saving and be a reporting bug, not a real one — see
    # docs/llm-prompt-caching.md's "Cost figures are understated" section,
    # which this fallback exists to not repeat in the other direction.
    effective_cached_rate = cached_rate if cached_rate is not None else in_rate
    cost = (
        (fresh_tokens / 1000.0 * in_rate)
        + (cached_tokens / 1000.0 * effective_cached_rate)
        + (output_tokens / 1000.0 * out_rate)
    )
    return round(cost, 6)
