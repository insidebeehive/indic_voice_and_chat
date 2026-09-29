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
import math
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.tenant import ProviderCost
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

# A missing ProviderCost row bills the turn's LLM tokens at $0.0 -- a
# reporting gap, not a runtime fault, so this only ever warns (never raises).
# token_rates runs per chat turn, so an unpriced provider/model would
# otherwise warn on every single turn; this set makes each distinct
# (kind, provider, model) combo warn once per process -- same idiom as
# call_store._rate's _warned_rate_misses (src/api/call_store.py), which in
# turn mirrors _PushFailureWarner (src/observability/turn_metrics_push.py).
_warned_rate_misses: set[tuple[str, ...]] = set()


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
    used_fallback = False
    if model and (row is None or (not row.cost_per_1k_input_tokens and not row.cost_per_1k_output_tokens)):
        fallback = await session.get(ProviderCost, ("llm", provider, ""))
        if fallback is not None:
            row = fallback
            used_fallback = True
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "chat_cost token_rates resolved",
            provider=provider, model=model, used_fallback=used_fallback,
            row_found=row is not None,
            input_rate=row.cost_per_1k_input_tokens if row is not None else None,
            output_rate=row.cost_per_1k_output_tokens if row is not None else None,
            cached_rate=row.cost_per_1k_cached_tokens if row is not None else None,
        )
    if row is None:
        # No catalog row at all for this (provider, model) -- neither the
        # exact-model row nor its provider-level fallback -- so every token
        # on this turn bills at $0.0. Same shape and same reason as
        # call_store._rate's miss warning: distinguishing "genuinely free"
        # from "nobody priced this yet" needs a trace, not silence.
        key = ("llm", provider, model)
        if key not in _warned_rate_misses:
            _warned_rate_misses.add(key)
            log.warning(
                "no ProviderCost row for kind=llm provider=%s model=%s (nor "
                "its provider-level fallback) -- billing this turn's tokens "
                "at $0.0 until a row is added (further warnings for this "
                "combination suppressed for this process)",
                provider, model,
            )
        return 0.0, 0.0, None
    if not row.cost_per_1k_input_tokens and not row.cost_per_1k_output_tokens:
        # A row exists and prices this at nothing. That is a legitimate state
        # (a genuinely free model) and an equally plausible mistake: the two
        # token-rate columns are NOT NULL with a 0.0 default, so a row added
        # without them looks identical to one priced at zero on purpose.
        # Silence would lose the half-configured case entirely -- it is the
        # one this function cannot distinguish, so it is the one worth
        # surfacing. Deduped per combination like the miss above, because it
        # would otherwise fire on every turn for the lifetime of the row.
        key = ("llm", provider, model, "zero_rated")
        if key not in _warned_rate_misses:
            _warned_rate_misses.add(key)
            log.warning(
                "ProviderCost row for kind=llm provider=%s model=%s prices "
                "input and output tokens at $0.0 -- correct only if this model "
                "is genuinely free; otherwise the row was added without its "
                "rates (further warnings for this combination suppressed for "
                "this process)",
                provider, model,
            )
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
    # No separate "$0 despite N+M tokens" warning here: whenever token_rates
    # resolves both rates to 0.0 it is because no ProviderCost row matched at
    # all (exact or provider-level), and token_rates itself already warns
    # that -- deduped per (kind, provider, model) rather than per turn, and
    # naming kind/provider/model directly. A second, un-deduped warning here
    # for the exact same condition would just be log noise on every turn for
    # an already-known gap. (A row that DOES exist but has both token rates
    # explicitly at 0.0 is indistinguishable from "never configured" on this
    # schema -- cost_per_1k_input_tokens/output_tokens have no NULL state,
    # unlike cost_per_1k_cached_tokens -- so that case is intentionally silent
    # too, consistent with "a lookup that finds a row emits nothing".)
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


# --- Platform-paid voice-note media cost (Phase 1 of the chat-cost-widening
# plan; Phase 2, KB embeddings, is out of scope here) -----------------------
#
# A voice-note reply's TTS synthesis and an inbound voice-note's STT
# transcription are both platform-paid legs with no chat_messages.cost
# entry today -- compute_chat_turn_cost only ever saw LLM tokens. These two
# helpers price them:
#   - TTS (voice-note replies): always per audio minute (``kind="tts"``) —
#     there is no token concept for synthesized audio.
#   - STT (inbound voice-note transcription): per token when the provider
#     reports usage (Gemini does — delegates to compute_chat_turn_cost, the
#     existing token-rate path), else per audio minute (``kind="stt"``) as a
#     fallback for a provider/path that reports no usage at all.


async def minute_rate(session: AsyncSession, kind: str, provider: str, model: str) -> float:
    """``cost_per_min`` for ``(kind, provider, model)``; falls back to the
    provider-level ("") row, same pattern as ``token_rates`` above (and
    ``call_store._rate``). ``kind`` is "tts" or "stt".
    """
    row = await session.get(ProviderCost, (kind, provider, model or ""))
    used_fallback = False
    if model and (row is None or not row.cost_per_min):
        fallback = await session.get(ProviderCost, (kind, provider, ""))
        if fallback is not None:
            row = fallback
            used_fallback = True
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "chat_cost minute_rate resolved",
            kind=kind, provider=provider, model=model, used_fallback=used_fallback,
            row_found=row is not None,
            rate=row.cost_per_min if row is not None else None,
        )
    if row is None:
        key = (kind, provider, model)
        if key not in _warned_rate_misses:
            _warned_rate_misses.add(key)
            log.warning(
                "no ProviderCost row for kind=%s provider=%s model=%s (nor its "
                "provider-level fallback) -- billing this voice-note leg at "
                "$0.0 until a row is added (further warnings for this "
                "combination suppressed for this process)",
                kind, provider, model,
            )
        return 0.0
    if not row.cost_per_min:
        key = (kind, provider, model, "zero_rated")
        if key not in _warned_rate_misses:
            _warned_rate_misses.add(key)
            log.warning(
                "ProviderCost row for kind=%s provider=%s model=%s prices "
                "cost_per_min at $0.0 -- correct only if this model is "
                "genuinely free; otherwise the row was added without its "
                "rate (further warnings for this combination suppressed for "
                "this process)",
                kind, provider, model,
            )
        return 0.0
    return row.cost_per_min


async def compute_chat_tts_cost(
    session: AsyncSession, *, provider: str, model: str, audio_ms: int,
) -> float:
    """Platform-billed cost for one voice-note reply's TTS synthesis."""
    if not provider or not audio_ms or audio_ms <= 0:
        return 0.0
    rate = await minute_rate(session, "tts", provider, model)
    return round(rate * audio_ms / 60_000.0, 6)


async def compute_chat_stt_cost(
    session: AsyncSession,
    *,
    provider: str,
    model: str,
    audio_ms: Optional[int] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
) -> float:
    """Platform-billed cost for one inbound voice-note's STT transcription.

    Prefers token-based pricing (Gemini reports usage on transcribe_audio —
    see ``TranscriptText`` in ``src/interfaces/llm.py``), falling back to
    per-audio-minute pricing when no token usage is available.
    """
    if not provider:
        return 0.0
    if input_tokens or output_tokens:
        return await compute_chat_turn_cost(
            session, provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
    if audio_ms and audio_ms > 0:
        rate = await minute_rate(session, "stt", provider, model)
        return round(rate * audio_ms / 60_000.0, 6)
    key = ("stt", provider, model, "unpriced")
    if key not in _warned_rate_misses:
        _warned_rate_misses.add(key)
        log.warning(
            "inbound voice-note transcription for provider=%s model=%s has "
            "neither token usage nor a known audio duration -- billing this "
            "leg at $0.0 (further warnings for this combination suppressed "
            "for this process)",
            provider, model,
        )
    debug_event(log, "chat_cost stt unpriced", provider=provider, model=model)
    return 0.0


# --- KB embedding cost (Phase 2 of the chat-cost-widening plan) ------------
#
# Gemini's Developer API embed_content is billed per INPUT token, but its
# response reports no token count (token_count/billable_character_count are
# Vertex-only fields -- see GeminiEmbedder in src/rag/embeddings.py), so
# billed tokens are ESTIMATED from input characters. Google's own rule of
# thumb for Gemini models is ~4 chars/token; Devanagari/Hinglish text likely
# tokenizes denser than that, so this estimate can UNDER-count for Hindi
# content.
_EMBED_CHARS_PER_TOKEN = 4.0

# In-process cache for embedding_token_rate, keyed by (provider, model). A KB
# search calls embedding_token_rate on every turn (via compute_embedding_cost
# -> record_embedding_usage) -- caching avoids re-running 1-2 ProviderCost
# lookups per search on the hot path. Short TTL rather than "forever" so a
# rate change (the YAML re-seed on boot per config/provider_costs.yaml, or a
# direct DB edit) is picked up within a bounded window without a restart.
_EMBEDDING_RATE_CACHE_TTL_S = 300.0
_embedding_rate_cache: dict[tuple[str, str], tuple[float, float]] = {}


def reset_embedding_rate_cache() -> None:
    """Test helper: clear the in-process embedding-rate cache
    (``_embedding_rate_cache`` above). Same footgun as ``_warned_rate_misses``
    (see ``_reset_warned_rate_misses`` in tests/unit/test_chat_cost.py) --
    without clearing this between tests, a rate cached by an earlier test
    against the same (provider, model) key would leak into a later one for up
    to ``_EMBEDDING_RATE_CACHE_TTL_S``. Any fixture that seeds or changes a
    ``kind="embedding"`` ``ProviderCost`` row should call this.
    """
    _embedding_rate_cache.clear()


async def embedding_token_rate(session: AsyncSession, provider: str, model: str) -> float:
    """``cost_per_1k_input_tokens`` for ``(kind="embedding", provider, model)``;
    falls back to the provider-level ("") row, same pattern as ``token_rates``
    and ``minute_rate`` above.

    Cached in-process for ``_EMBEDDING_RATE_CACHE_TTL_S`` seconds per
    ``(provider, model)`` -- a cache hit skips the DB lookup (and the
    debug/warning logic below) entirely. See ``reset_embedding_rate_cache``.
    """
    cache_key = (provider, model)
    cached = _embedding_rate_cache.get(cache_key)
    now = time.monotonic()
    if cached is not None and cached[1] > now:
        return cached[0]
    row = await session.get(ProviderCost, ("embedding", provider, model or ""))
    used_fallback = False
    if model and (row is None or not row.cost_per_1k_input_tokens):
        fallback = await session.get(ProviderCost, ("embedding", provider, ""))
        if fallback is not None:
            row = fallback
            used_fallback = True
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "chat_cost embedding_token_rate resolved",
            provider=provider, model=model, used_fallback=used_fallback,
            row_found=row is not None,
            rate=row.cost_per_1k_input_tokens if row is not None else None,
        )
    if row is None:
        # No catalog row at all for this (provider, model) -- neither the
        # exact-model row nor its provider-level fallback -- so every token
        # in this embed batch bills at $0.0. Same shape and same reason as
        # token_rates'/minute_rate's own miss warning.
        key = ("embedding", provider, model)
        if key not in _warned_rate_misses:
            _warned_rate_misses.add(key)
            log.warning(
                "no ProviderCost row for kind=embedding provider=%s model=%s "
                "(nor its provider-level fallback) -- billing this embedding "
                "batch at $0.0 until a row is added (further warnings for "
                "this combination suppressed for this process)",
                provider, model,
            )
        rate = 0.0
    elif not row.cost_per_1k_input_tokens:
        # A row exists and prices this at nothing -- a legitimate state (a
        # genuinely free model) and an equally plausible mistake: the column
        # is NOT NULL with a 0.0 default, so a row added without its rate
        # looks identical to one priced at zero on purpose. Deduped per
        # combination like the miss above, because it would otherwise fire on
        # every embed call for the lifetime of the row.
        key = ("embedding", provider, model, "zero_rated")
        if key not in _warned_rate_misses:
            _warned_rate_misses.add(key)
            log.warning(
                "ProviderCost row for kind=embedding provider=%s model=%s "
                "prices cost_per_1k_input_tokens at $0.0 -- correct only if "
                "this model is genuinely free; otherwise the row was added "
                "without its rate (further warnings for this combination "
                "suppressed for this process)",
                provider, model,
            )
        rate = 0.0
    else:
        rate = row.cost_per_1k_input_tokens
    _embedding_rate_cache[cache_key] = (rate, now + _EMBEDDING_RATE_CACHE_TTL_S)
    return rate


async def compute_embedding_cost(
    session: AsyncSession,
    *,
    provider: str,
    model: str,
    input_chars: int,
    tokens: Optional[int] = None,
) -> float:
    """Platform-billed cost for one KB-embedding batch (ingest or search).

    Prefers a real reported token count; else estimates billed tokens from
    ``input_chars`` at ``_EMBED_CHARS_PER_TOKEN``. Returns 0.0 when there is
    nothing to price (no provider, and neither a token count nor a positive
    char count).
    """
    if not provider:
        return 0.0
    if tokens is not None and tokens > 0:
        billed = tokens
    elif input_chars and input_chars > 0:
        billed = math.ceil(input_chars / _EMBED_CHARS_PER_TOKEN)
    else:
        return 0.0
    rate = await embedding_token_rate(session, provider, model)
    # 9 dp here, not the 6 dp used elsewhere in this module: a single
    # search-query embed batch costs on the order of 1e-6 USD, which 6 dp
    # would round to 0 or distort by a large relative amount.
    return round(billed / 1000.0 * rate, 9)
