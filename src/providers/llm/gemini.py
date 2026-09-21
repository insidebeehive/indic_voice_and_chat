"""Gemini LLM adapter (Google Generative AI).

Wraps the official ``google-genai`` SDK. Both batch and streaming flows
are supported — streaming uses ``aio.generate_content_stream`` and yields
text deltas as they arrive.

Indic-language handling: Gemini understands Hindi/Devanagari natively, so
the adapter doesn't need to do any pre-processing. The agent's system
prompt declares the language directive (see ``build_voicebot_system_prompt``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Any, AsyncIterator, Callable, Optional

import json

from src.interfaces.llm import (
    ContentPart,
    ILLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResult,
    ToolCall,
    is_llm_spending_cap_error,
)
from src.utils.logging import debug_event


log = logging.getLogger(__name__)

# gemini-2.5-flash 404s ("no longer available to new users") on Gemini
# projects created after mid-2026 — the staging key swap broke on it.
DEFAULT_MODEL = "gemini-3.5-flash"

# Gemini intermittently returns transient backend errors — most notably
# ``500 INTERNAL`` ("An internal error has occurred. Please retry...") — even
# for well-formed requests. Google's own error body tells callers to retry, so
# we transparently retry these before any audio is spoken. A live turn must not
# die on a provider blip.
#
# 429 (quota) is handled separately from 5xx: under quota exhaustion, fast
# multi-retry AMPLIFIES the request rate against an already-throttled API (a
# retry storm — 200 concurrent chat sessions × 3 attempts each at 0.4s/0.8s).
# Gemini's dominant quota is TOKENS PER MINUTE, so a saturated window clears on
# a ~minute cadence: retries must ESCALATE across that window, not hammer
# inside it. Verified under a 200-parallel load test: a single short 2-4s
# retry still failed 42/200 turns (both attempts landed in the same saturated
# minute); escalating jittered waits let queued turns land as the window
# rolls. Worst-case cumulative wait ≈ 35s, inside the 60s chat turn ceiling.
_RETRIABLE_STATUS = frozenset({500, 502, 503, 504})
_MAX_RETRIES = 2  # total attempts = 1 + _MAX_RETRIES (5xx only)
_BACKOFF_BASE_S = 0.4
_RATE_LIMIT_BACKOFF_S = ((1.0, 2.0), (4.0, 8.0), (15.0, 25.0))  # jitter window per retry
_RATE_LIMIT_MAX_RETRIES = len(_RATE_LIMIT_BACKOFF_S)
# Gemini's 429 body includes RetryInfo.retryDelay. For the per-MINUTE quota
# it's ~1s and the escalating schedule above bridges it. But the per-DAY
# request quota (e.g. 10K requests/day on Tier 1) answers with retryDelay of
# HOURS ('36016s') — retrying is pure waste that just delays the customer's
# error by ~35s. If Google says the quota won't clear within this bound,
# fail fast instead of retrying.
_RATE_LIMIT_GIVE_UP_S = 60.0
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")

# Below this cumulative (semaphore-wait + backoff-sleep) threshold, a call is
# the common fast case — logging it would just be noise. Above it, something
# (concurrency saturation or a retry) made the call slow, which is exactly
# what production logs need to show.
_SLOW_CALL_LOG_THRESHOLD_S = 1.0


def _suggested_retry_delay_s(exc: Exception) -> Optional[float]:
    """Pull Google's RetryInfo.retryDelay (seconds) out of a 429 error, if present."""
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) if m else None

# Cap on concurrent in-flight Gemini requests per process. Without it, a burst
# of sessions (e.g. a 200-parallel-chat stress test) fans out unbounded and
# blows the per-minute quota, turning every turn into a 429. The cap is at the
# adapter layer so chat, voice, and analysis all share it without call-site
# changes. Streaming holds a slot only until the first chunk arrives (the
# rate-limited unit is the request, not the stream's lifetime).
_DEFAULT_MAX_CONCURRENCY = 24
# Semaphores bind to the running event loop on first await; tests spin up a
# fresh loop per test, so key the semaphore by loop instead of one module global.
_sem_by_loop: dict[int, asyncio.Semaphore] = {}


def _concurrency_sem() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _sem_by_loop.get(id(loop))
    if sem is None:
        limit = int(os.environ.get("GEMINI_MAX_CONCURRENCY", _DEFAULT_MAX_CONCURRENCY))
        sem = asyncio.Semaphore(max(1, limit))
        _sem_by_loop[id(loop)] = sem
    return sem


# --- Explicit context-cache registry -------------------------------------
#
# See docs/llm-prompt-caching.md for the measurements this rests on: Gemini's
# IMPLICIT cache only covers the single contiguous ``system_instruction``
# field (never ``contents``, never ``config.tools``), and even there it's
# capped around 4,024-4,028 tokens against this codebase's real prompts —
# 51% of a ~7,900-token turn. An EXPLICIT cache (``client.caches.create``),
# measured the same session, covered 99.7% of that same prompt including
# tool declarations. But creation took 1.77s — far too slow to sit on a live
# turn — and the Developer API rejects anything under 1,024 tokens. So the
# adapter keeps its own small registry: look up a cache for the exact
# (model, system, tools) a call is about to send, use it if one exists and is
# still fresh, and otherwise let the call go out uncached while a creation is
# kicked off in the background for next time.

_DEFAULT_CACHE_TTL_S = 1800
# Cache creation costs a full extra request (1.77s measured) against the same
# quota as everything else, so a key that just failed to create shouldn't be
# retried on literally the next turn — that would turn one bad key into a
# steady drip of wasted `caches.create` calls. 5 minutes is long enough to
# ride out a transient blip without a human noticing, short enough that a
# real fix (e.g. quota recovering) is picked up again well within the same
# session.
_CACHE_CREATE_COOLDOWN_S = 300
# Server-side expiry is exact; a request landing in the last few hundred ms
# before it would otherwise race the boundary and depend on network timing
# to decide whether it's honoured. Treating the cache as dead 30s early costs
# nothing (the entry just re-arms and recreates) and removes the race.
_CACHE_EXPIRY_MARGIN_S = 30
# Below this, a prompt can't be a cache candidate at all: Gemini's Developer
# API rejects `caches.create` under 1,024 TOKENS outright (measured in
# docs/llm-prompt-caching.md — a 10-token probe came back
# `min_total_token_count=1024`). This uses CHARACTERS as a cheap client-side
# proxy — actual tokenization would need a network round trip just to decide
# whether to try caching — set comfortably above the ~4,100-char equivalent
# of 1,024 tokens so the estimate never undershoots into a guaranteed-reject
# `caches.create` call.
_CACHE_MIN_SYSTEM_CHARS = 5000
# Every distinct (model, system, tools) combination the adapter is ever
# handed gets its own registry slot — voice, chat, and analysis all share one
# adapter instance, and a live deployment can churn through many distinct
# prompts (per-turn-varying voice system text, A/B prompt packs, etc.). Left
# unbounded this dict would grow for the lifetime of the process; capping it
# and evicting the coldest entries keeps memory bounded without needing a
# background sweep.
_CACHE_MAX_ENTRIES = 64

# `asyncio.create_task` only holds a WEAK reference internally — if nothing
# else references the returned Task object, the event loop is free to
# garbage-collect it mid-flight, which silently cancels the background
# `caches.create()` call with no exception and no cache ever appearing. This
# is the one piece of cache state that has to live at module scope rather
# than per-adapter-instance: it exists purely to keep a strong reference
# alive until the task finishes, not to track which adapter it belongs to
# (that's on the entry itself, closed over in the task's coroutine).
_inflight_cache_tasks: set[asyncio.Task] = set()


def explicit_cache_enabled() -> bool:
    """Feature flag for the explicit context-cache registry, default OFF.

    Read live via ``os.environ.get`` on every call — not cached at import
    time — matching ``_concurrency_sem``'s convention above, for the same two
    reasons: ``monkeypatch.setenv`` in tests must take effect without having
    to reset a cached value, and an operator flipping the env var in
    production must not need a process restart to see it take effect.
    """
    return os.environ.get("GEMINI_EXPLICIT_CACHE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _cache_ttl_s() -> int:
    """TTL for a created cache, in seconds. Default 30 minutes.

    Explicit caches bill STORAGE per token-hour, so the TTL is a direct
    dial on wasted spend: creation costs ~1.77s of background work plus
    writing the full token payload (see docs/llm-prompt-caching.md), so it
    only pays for itself if reused across many turns — but once armed, an
    unnecessarily long TTL keeps billing storage for a cache nobody is
    reading if the tenant goes quiet. 30 minutes bounds that idle waste to
    at most half an hour, while comfortably outlasting the gap between turns
    of any tenant who is actually active (chat turns arrive far more often
    than every 30 minutes), so an active tenant's calls keep reusing one
    cache instead of re-paying the creation cost every time the window rolls.
    Floored at 60s so a misconfigured near-zero value can't create a cache
    that expires before anything could ever reuse it.
    """
    raw = os.environ.get("GEMINI_CACHE_TTL_S", str(_DEFAULT_CACHE_TTL_S))
    try:
        ttl = int(raw)
    except ValueError:
        ttl = _DEFAULT_CACHE_TTL_S
    return max(60, ttl)


def _is_stale_cache_error(exc: Exception) -> bool:
    """True if ``exc`` looks like a request was rejected because the
    ``cached_content`` it referenced is gone (expired, deleted, or otherwise
    no longer valid on the server).

    Heuristic, matching the style of ``is_llm_spending_cap_error`` above:
    Google doesn't document one stable error shape for "this CachedContent no
    longer exists", so this matches broadly — either a 403/404 status (the
    resource is gone or we no longer have access to it) or the literal
    substring "cachedcontent" anywhere in the error text (catches "not
    found"/"expired"/"permission denied" phrasings without needing the exact
    wording). This is intentionally over-inclusive: a false positive here
    just costs one wasted uncached retry within the same turn, which is
    cheap and invisible to the customer. It must NEVER be broad enough to
    swallow an unrelated error, so the caller only invokes this at all when
    the failing request actually carried a ``cached_content`` value — a bare
    404 on a request that never referenced a cache is a different failure
    entirely and must propagate normally.
    """
    code = getattr(exc, "code", None)
    if code in (403, 404):
        return True
    return "cachedcontent" in str(exc).lower()


@dataclass
class _CacheEntry:
    """One (model, system, tools) key's state in an adapter's cache registry."""

    sightings: int = 0
    name: Optional[str] = None
    expires_at: float = 0.0
    creating: bool = False
    cooldown_until: float = 0.0
    last_used: float = 0.0


class GeminiLLMAdapter(ILLMProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._default_model = config.get("model") or DEFAULT_MODEL
        # Per-INSTANCE registry, not a module global: the platform-LLM
        # registry hands back a long-lived singleton adapter in production,
        # so this dict lives exactly as long as it needs to (no manual
        # teardown) — and, critically, it can never leak a cache name across
        # two adapters built with different API keys/models (e.g. two
        # tenants on distinct Gemini projects, or a test's fake client next
        # to a real one). Initialized BEFORE the client-injection early
        # return below so an injected-client test adapter gets a working
        # registry too, not just real production ones.
        self._cache_entries: dict[str, _CacheEntry] = {}
        client = config.get("client")
        if client is not None:
            # Tests inject a fake; bypass real SDK construction.
            self._client = client
            return
        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GeminiLLMAdapter requires an API key (config 'api_key' or "
                "GEMINI_API_KEY env var)"
            )
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "GeminiLLMAdapter requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from e
        self._client = genai.Client(api_key=api_key)

    # --- Message conversion ---------------------------------------------

    @staticmethod
    def _to_gemini_contents(messages: list[LLMMessage]) -> tuple[Optional[str], list[dict]]:
        """Split our (role, content) messages into Gemini's shape.

        Gemini takes a separate ``system_instruction`` plus a list of
        ``contents`` entries with ``role`` in {user, model}. We map:
            our "system"    -> system_instruction (concatenated if many)
            our "user"      -> {role: "user", parts: [{text: ...}]}
            our "assistant" -> {role: "model", parts: [{text: ...}]}
        """
        system_parts: list[str] = []
        contents: list[dict] = []
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                continue
            # Tool result (function_response) → a "user"-role function_response part.
            if m.role == "tool":
                try:
                    response = json.loads(m.content) if isinstance(m.content, str) else {}
                except (json.JSONDecodeError, TypeError):
                    response = {"result": m.content}
                if not isinstance(response, dict):
                    response = {"result": response}
                contents.append({"role": "user", "parts": [
                    {"function_response": {"name": m.name or "", "response": response}}]})
                continue
            role = "model" if m.role == "assistant" else "user"
            # Assistant message that emitted tool calls → function_call parts.
            # Gemini 3.x requires the thought_signature captured from the
            # original response to ride along, else 400 INVALID_ARGUMENT.
            if m.tool_calls:
                parts = []
                for tc in m.tool_calls:
                    part: dict[str, Any] = {"function_call": {"name": tc.name, "args": tc.arguments}}
                    if tc.thought_signature is not None:
                        part["thought_signature"] = tc.thought_signature
                    parts.append(part)
                contents.append({"role": role, "parts": parts})
                continue
            contents.append({"role": role, "parts": _content_parts(m.content)})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, contents

    def _build_config(self, config: LLMConfig) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
            # Gemini 2.5 Flash thinking tokens count against max_output_tokens.
            # Disable thinking (budget=0) so all output tokens go to visible text;
            # thinking mode can be re-enabled per-call if deep reasoning is needed.
            "thinking_config": {"thinking_budget": 0},
        }
        if config.tools:
            # Function declarations as plain dicts — the SDK coerces them.
            cfg["tools"] = [{"function_declarations": [{
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }]} for t in config.tools]
            # Gemini rejects response_mime_type=application/json together with
            # tools, so JSON format is suppressed for tool turns (the agent does a
            # separate no-tools JSON finalize turn).
        elif config.response_format == "json":
            cfg["response_mime_type"] = "application/json"
        return cfg

    @staticmethod
    async def _call_with_retry(fn: Callable[[], Any], *, what: str) -> Any:
        """Await ``fn()``, retrying transient errors with backoff.

        5xx errors in ``_RETRIABLE_STATUS`` get up to ``_MAX_RETRIES`` fast
        retries; a 429 gets a single retry after a long jittered backoff (fast
        multi-retry on quota exhaustion amplifies the storm). Everything else
        (and ``CancelledError``, which is a ``BaseException``) propagates
        immediately. Each attempt holds a concurrency-cap slot only while the
        request is in flight — not across backoff sleeps. The caller must
        ensure no side effects have been committed yet — for streaming, that
        means no token has been yielded.
        """
        attempt = 0
        rate_limit_attempt = 0
        total_sem_wait = 0.0
        total_backoff = 0.0
        while True:
            wait_start = perf_counter()
            try:
                async with _concurrency_sem():
                    total_sem_wait += perf_counter() - wait_start
                    result = await fn()
            except Exception as exc:  # noqa: BLE001 - re-raised unless retriable
                code = getattr(exc, "code", None)
                if code == 429 and rate_limit_attempt < _RATE_LIMIT_MAX_RETRIES:
                    if is_llm_spending_cap_error(exc):
                        # A monthly SPENDING CAP is a billing ceiling, not a
                        # quota window — unlike the per-minute/per-day quotas
                        # below, it does not clear on a ~minute cadence, so
                        # the escalating schedule just delays the customer's
                        # error while adding pointless load against an API
                        # that is refusing us outright (not throttling us).
                        # Seen live 2026-09-17: 3 full retries (2.0s/7.7s/
                        # 18.2s) then failure, ~29s per turn, against
                        # ~1,200-2,100 chat turns/day — every turn during the
                        # cap paid that ~29s before the customer saw an
                        # error, and logs showed customers disconnecting
                        # before it arrived. Same predicate as
                        # src.api.chat._classify_turn_error's "llm_billing"
                        # branch (imported, not re-derived) so the two
                        # classifications can't drift apart.
                        log.error(
                            "gemini monthly spending cap exceeded on %s — raise it "
                            "at https://ai.studio/spend to restore service; not retrying",
                            what,
                        )
                        raise
                    if "FreeTier" in str(exc) or "free_tier" in str(exc):
                        # Not load — a misconfigured project. A key swapped to a
                        # Gemini project WITHOUT billing runs on the free tier
                        # (e.g. 20 requests/DAY): retrying is pointless and the
                        # fix is operational. Seen live 2026-07-17 after the
                        # staging key moved to a new, unbilled project.
                        log.error(
                            "gemini 429 on %s is a FREE-TIER quota — the API key's "
                            "project has no billing account attached; enable billing "
                            "on the Gemini project to restore normal quotas",
                            what,
                        )
                        raise
                    suggested = _suggested_retry_delay_s(exc)
                    if suggested is not None and suggested > _RATE_LIMIT_GIVE_UP_S:
                        # e.g. the per-DAY quota: "retry in 10h" — don't make
                        # the customer wait through a doomed retry schedule.
                        log.error(
                            "gemini quota exhausted on %s (provider says retry in %.0fs) — not retrying",
                            what, suggested,
                        )
                        raise
                    delay = random.uniform(*_RATE_LIMIT_BACKOFF_S[rate_limit_attempt])
                    rate_limit_attempt += 1
                    log.warning(
                        "gemini rate-limited (429) on %s; retry %d/%d in %.1fs",
                        what, rate_limit_attempt, _RATE_LIMIT_MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    total_backoff += delay
                    continue
                if code in _RETRIABLE_STATUS and attempt < _MAX_RETRIES:
                    attempt += 1
                    delay = _BACKOFF_BASE_S * attempt
                    log.warning(
                        "gemini transient %s on %s; retry %d/%d",
                        code, what, attempt, _MAX_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    total_backoff += delay
                    continue
                # Every path above either retries (continue) or raises inline
                # with its own log (spend cap / free tier / quota-give-up all
                # log.error before raising). Falling through to here means
                # NONE of those matched -- a non-retriable status (e.g. 400
                # INVALID_ARGUMENT) or a 429/5xx that exhausted its retry
                # budget -- and until now that raised with zero trace: the
                # caller sees the exception, but nothing records which Gemini
                # call it was or how many attempts preceded it.
                debug_event(log, "gemini call failed", what=what, code=code,
                            error=str(exc), attempt=attempt,
                            rate_limit_attempt=rate_limit_attempt)
                raise
            else:
                waited = total_sem_wait + total_backoff
                if waited > _SLOW_CALL_LOG_THRESHOLD_S:
                    log.info(
                        "gemini call waited: sem_wait=%.2fs backoff=%.2fs on %s",
                        total_sem_wait, total_backoff, what,
                    )
                return result

    # --- Explicit context-cache registry --------------------------------

    @staticmethod
    def _cache_key(model: str, system: str, tools: Optional[list[dict]]) -> str:
        """Hash the exact wire shape a call is about to send.

        Hashing ``tools`` via the SAME ``list[dict]`` ``_build_config`` already
        produces for ``cfg["tools"]`` (not a second hand-rolled serialization
        of ``ToolSpec`` objects) matters: if the two ever drifted, a tool
        rename/add/remove could silently keep reusing a stale cache key while
        the actual request body changed underneath it. Each component is
        followed by a NUL separator so e.g. model="ab" + system="c" can never
        collide with model="a" + system="bc". ``tools`` is serialized with
        ``sort_keys=True`` so key ordering inside each dict never perturbs the
        hash — only the actual declared tools do.
        """
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(system.encode("utf-8"))
        h.update(b"\x00")
        h.update(json.dumps(tools or [], sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")
        return h.hexdigest()

    def _evict_cache_entries_if_over_cap(self) -> None:
        """Keep the registry at or under ``_CACHE_MAX_ENTRIES``.

        Preference order for eviction: entries with ``name is None`` first
        (never armed, or armed-then-expired — these hold no live cache, so
        dropping them costs nothing but a re-arm on next sighting), oldest
        ``last_used`` within each group. Only spills into entries that DO
        hold a live cache name if the registry is still over cap after
        clearing every empty one — losing a live cache means the next call
        for that key just falls back to uncached and eventually recreates,
        not a correctness problem, just a wasted future creation.
        """
        entries = self._cache_entries
        if len(entries) <= _CACHE_MAX_ENTRIES:
            return
        before = len(entries)
        ordered = sorted(
            entries.items(),
            key=lambda kv: (kv[1].name is not None, kv[1].last_used),
        )
        for key, _entry in ordered:
            if len(entries) <= _CACHE_MAX_ENTRIES:
                break
            del entries[key]
        debug_event(log, "gemini cache registry evicted", before=before,
                    after=len(entries), cap=_CACHE_MAX_ENTRIES)

    def _cache_lookup_and_arm(
        self, model: str, system: str, tools: Optional[list[dict]],
    ) -> tuple[Optional[str], Optional[str]]:
        """Look up a usable cache for this exact request shape, possibly
        scheduling a background creation as a side effect. Returns
        ``(cache_name_or_None, registry_key_or_None)`` — the key is returned
        even on a miss so the caller can evict the right entry later if the
        call turns out to reference a since-invalidated cache.

        Deliberately does NOT create a cache on the first sighting of a key.
        The adapter is shared by every caller (voice's per-turn-varying
        system text, chat's stable one, analysis'), so treating every novel
        key as a creation candidate would mean callers with unstable prompts
        burn a full ``caches.create`` request on every single call for zero
        future benefit — nothing ever comes back to reuse it. Only creating
        from the SECOND sighting onward means a key that never repeats never
        creates anything, while chat's static system prompt (stable across
        a turn's ~2.47 LLM calls, per docs/llm-prompt-caching.md) gets a
        cache armed within a turn or two of first traffic.
        """
        if not explicit_cache_enabled():
            return None, None
        if len(system) < _CACHE_MIN_SYSTEM_CHARS:
            return None, None

        key = self._cache_key(model, system, tools)
        now = monotonic()
        entry = self._cache_entries.get(key)
        if entry is None:
            entry = _CacheEntry()
            self._cache_entries[key] = entry
        entry.sightings += 1
        entry.last_used = now

        cache_name: Optional[str] = None
        if entry.name is not None:
            if entry.expires_at > now:
                cache_name = entry.name
                debug_event(log, "gemini cache hit", model=model, cache_key=key,
                            cache_name=cache_name, sightings=entry.sightings)
            else:
                # Expired (or past our safety margin before real expiry) —
                # clear the name but keep `sightings` so the very next
                # lookup for this key re-arms creation instead of having to
                # rebuild history from a fresh sightings=1.
                entry.name = None
                entry.expires_at = 0.0

        if (
            cache_name is None
            and entry.sightings >= 2
            and not entry.creating
            and entry.cooldown_until <= now
        ):
            entry.creating = True
            debug_event(log, "gemini cache creation scheduled", model=model,
                        cache_key=key, sightings=entry.sightings,
                        system_chars=len(system))
            task = asyncio.create_task(self._create_cache(key, model, system, tools))
            _inflight_cache_tasks.add(task)
            task.add_done_callback(_inflight_cache_tasks.discard)

        self._evict_cache_entries_if_over_cap()
        return cache_name, key

    async def _create_cache(
        self, key: str, model: str, system: str, tools: Optional[list[dict]],
    ) -> None:
        """Background ``caches.create`` for one registry key. Never awaited
        by a live turn — scheduled via ``asyncio.create_task`` from
        ``_cache_lookup_and_arm`` and left to finish on its own time.

        Re-fetches ``entry`` by key at the top rather than closing over the
        entry object directly: by the time this coroutine actually runs (it
        was only just scheduled, not awaited), ``_evict_cache_entries_if_over_cap``
        may already have dropped this key from the registry (cap eviction —
        see ``_CACHE_MAX_ENTRIES``). If so there is nothing left to update
        and no one waiting on the result, so it just returns.
        """
        entry = self._cache_entries.get(key)
        if entry is None:
            return
        try:
            create_config: dict[str, Any] = {
                "system_instruction": system,
                "ttl": f"{_cache_ttl_s()}s",
                # For identifying the cache in the Gemini console; truncated
                # since the key itself is only useful as a lookup, not a label.
                "display_name": f"chat-{key[:16]}",
            }
            if tools:
                create_config["tools"] = tools
            cached = await self._call_with_retry(
                lambda: self._client.aio.caches.create(model=model, config=create_config),
                what="cache_create",
            )
            entry.name = getattr(cached, "name", None)
            # 30s safety margin (see _CACHE_EXPIRY_MARGIN_S) so a request
            # already in flight never races the server's real expiry boundary.
            entry.expires_at = monotonic() + _cache_ttl_s() - _CACHE_EXPIRY_MARGIN_S
            debug_event(log, "gemini cache armed", model=model, cache_key=key,
                        cache_name=entry.name, ttl_s=_cache_ttl_s(),
                        system_chars=len(system))
        except Exception:  # noqa: BLE001 - a failed cache create must never break a turn
            entry.cooldown_until = monotonic() + _CACHE_CREATE_COOLDOWN_S
            log.warning(
                "gemini context-cache creation failed; continuing uncached",
                exc_info=True,
            )
        finally:
            # Cleared unconditionally (success or failure) so two
            # near-simultaneous `generate()` calls hitting the same
            # still-uncached key don't both schedule a create — the second
            # one to arrive sees `creating=True` and just waits for this one.
            entry.creating = False

    # --- Public API ----------------------------------------------------

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResult:
        system, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(config)
        model = config.model or self._default_model

        # Cache lookup is entirely optional plumbing bolted onto the request
        # this method already builds — `cache_name` stays None (identical to
        # pre-cache behaviour below) whenever caching is off, the prompt is
        # too short to be a candidate, or no armed cache exists yet for this
        # exact key.
        cache_name: Optional[str] = None
        cache_key: Optional[str] = None
        if system:
            cache_name, cache_key = self._cache_lookup_and_arm(
                model, system, gen_config.get("tools"),
            )

        if cache_name:
            # A request carrying `cached_content` must not ALSO carry
            # `system_instruction`/`tools` — verified live against the
            # Developer API this session: combining them 400s with
            # "CachedContent can not be used with GenerateContent request
            # setting system_instruction, tools or tool_config." Whatever the
            # cache holds is implicitly part of the prompt already.
            gen = {k: v for k, v in gen_config.items() if k != "tools"}
            gen["cached_content"] = cache_name
        else:
            # Fresh copy (mirroring the cache-hit branch above), not an alias
            # of gen_config: this dict gets `system_instruction` mutated onto
            # it below, and the caller-visible gen_config must stay pristine
            # for any future retry/fallback logic that reuses it.
            gen = dict(gen_config)
            if system:
                gen["system_instruction"] = system

        # Full request boundary, including the prompt: `gen` and `contents`
        # are the exact wire shapes handed to the SDK (plain dicts already —
        # nothing here needs a separate serialization step), so this is the
        # actual payload sent, not a reconstruction of it.
        debug_event(log, "gemini generate request", model=model, system=system,
                    contents=contents, config=gen, cache_name=cache_name)
        try:
            response = await self._call_with_retry(
                lambda: self._client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen,
                ),
                what="generate",
            )
        except Exception as exc:  # noqa: BLE001 - only a stale-cache hit is special-cased
            if not (cache_name and _is_stale_cache_error(exc)):
                raise
            # The cache we referenced is gone server-side (expired between
            # our TTL bookkeeping and the server's clock, deleted out of
            # band, etc.) — evict it from the registry (keeping `sightings`
            # so it re-arms on the next lookup) and retry ONCE, uncached, in
            # THIS SAME generate() call. A live turn must never fail just
            # because our local cache bookkeeping was stale; the customer
            # never sees this.
            log.warning(
                "gemini cached_content %r rejected as stale (%s); retrying uncached",
                cache_name, exc,
            )
            if cache_key is not None:
                entry = self._cache_entries.get(cache_key)
                if entry is not None:
                    entry.name = None
                    entry.expires_at = 0.0
                    # Same cooldown as a creation FAILURE (_create_cache's
                    # except branch above): without it, a cache that's
                    # REPRODUCIBLY rejected (server-side caching disabled,
                    # a persistent permission issue, ...) re-arms on the
                    # very next lookup — since `sightings` is deliberately
                    # kept, not reset — and the adapter loops
                    # create -> used -> rejected -> evict -> create forever,
                    # burning a real caches.create call on every turn.
                    entry.cooldown_until = monotonic() + _CACHE_CREATE_COOLDOWN_S
            fallback = {k: v for k, v in gen_config.items() if k != "cached_content"}
            if system:
                fallback["system_instruction"] = system
            response = await self._call_with_retry(
                lambda: self._client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=fallback,
                ),
                what="generate",
            )
        text = self._extract_text(response)
        usage = self._extract_usage(response)
        finish_reason = self._extract_finish_reason(response)
        tool_calls = self._extract_tool_calls(response)
        raw = _dump(response)
        # Covers both the cache-hit and the stale-cache-fallback path above —
        # both flow through this one return, so one log line here is the
        # complete response boundary regardless of which branch produced it.
        debug_event(log, "gemini generate response", model=model, text=text,
                    finish_reason=finish_reason, usage=usage,
                    tool_calls=tool_calls, raw_response=raw)
        return LLMResult(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=raw,
            tool_calls=tool_calls,
        )

    async def transcribe_audio(self, audio: bytes, mime_type: str = "audio/mpeg") -> str:
        """Transcribe a complete audio recording (e.g. a call mp3) to text via
        Gemini multimodal — strong on Indian languages + Hindi/English code-switch,
        and handles long, mono recordings the live turn-STT can't. Returns "" on
        failure (analysis then falls back, never crashes)."""
        prompt = (
            "Transcribe this phone call audio verbatim. Preserve the original "
            "languages exactly as spoken (Hindi/English/other code-switching). "
            "Output ONLY the transcript text — no commentary, labels, or translation."
        )
        contents = [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": audio}},
        ]}]
        # The prompt is the diagnostic text; the audio itself is never logged
        # (same rule the STT adapters follow) — it's binary and its size, not
        # its content, is what's useful here.
        debug_event(log, "gemini transcribe_audio request", model=self._default_model,
                    mime_type=mime_type, audio_bytes=len(audio), prompt=prompt)
        try:
            response = await self._call_with_retry(
                lambda: self._client.aio.models.generate_content(
                    model=self._default_model, contents=contents),
                what="transcribe")
            text = self._extract_text(response) or ""
            debug_event(log, "gemini transcribe_audio response", text=text)
            return text
        except Exception:  # noqa: BLE001 - transcription failure must not crash finalize
            log.exception("gemini audio transcription failed")
            return ""

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        # Deliberately NOT wired into the explicit-cache registry used by
        # `generate()`. This is the voice path — latency-critical, and its
        # system prompt itself carries per-turn content (see
        # build_voicebot_system_prompt), so it changes on every call. Every
        # single invocation would therefore be a first sighting of a brand
        # new cache key, and per the "second sighting" rule a key that's
        # never seen twice never creates anything — the cache machinery
        # would sit here doing hash work and dict lookups on every turn for
        # zero hits, ever. Skipping it here is not a missed optimization,
        # it's the correct outcome of the same rule `generate()` uses.
        system, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(config)
        if system:
            gen_config["system_instruction"] = system
        model = config.model or self._default_model
        debug_event(log, "gemini generate_stream request", model=model, system=system,
                    contents=contents, config=gen_config)

        # Transient 5xx can surface either when opening the stream or on the
        # first chunk, so the retry must cover both — but only up to the first
        # yielded token, after which audio may already be playing and a retry
        # would re-speak. Re-opening starts a fresh stream each attempt.
        async def _open_and_first() -> tuple[Any, Any]:
            stream = await self._client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=gen_config,
            )
            agen = stream.__aiter__()
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                first = None
            return agen, first

        agen, first = await self._call_with_retry(
            _open_and_first, what="generate_stream",
        )

        if first is None:
            debug_event(log, "gemini generate_stream response", model=model, text="", chunks=0)
            return

        # The accumulator only exists to feed the debug line below, so it's
        # only built when DEBUG is actually on -- an f-string-free version of
        # the same rule debug_event itself follows (don't pay for a
        # diagnostic nobody is reading). This is the voice path: an
        # unconditional per-token string accumulation would add real
        # per-token cost to every live call for a line that's off by default.
        collecting = log.isEnabledFor(logging.DEBUG)
        chunks: list[str] = []
        text = self._extract_text(first)
        if text:
            if collecting:
                chunks.append(text)
            yield text
        async for chunk in agen:
            text = self._extract_text(chunk)
            if text:
                if collecting:
                    chunks.append(text)
                yield text
        if collecting:
            debug_event(log, "gemini generate_stream response", model=model,
                        text="".join(chunks), chunks=len(chunks))

    # --- Response shape helpers (resilient to SDK changes) -------------

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[ToolCall]:
        """Pull ``function_call`` parts out of the response into ToolCalls.
        Gemini calls carry no id, so we synthesize one from name + index."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []
        content = getattr(candidates[0], "content", None)
        parts = (getattr(content, "parts", None) or []) if content is not None else []
        calls: list[ToolCall] = []
        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc is None:
                continue
            name = getattr(fc, "name", "") or ""
            args = getattr(fc, "args", None) or {}
            calls.append(ToolCall(
                id=getattr(fc, "id", None) or f"{name}-{len(calls)}",
                name=name,
                arguments=dict(args),
                # Signature lives on the PART, not the function_call — must be
                # replayed with the call on the next round (Gemini 3.x).
                thought_signature=getattr(p, "thought_signature", None),
            ))
        return calls

    @staticmethod
    def _extract_text(response: Any) -> str:
        # The SDK exposes a ``.text`` convenience accessor; fall back to
        # walking ``candidates[0].content.parts[*].text`` if that fails. The
        # ``.text`` property RAISES when the only part is a function_call, so
        # guard it.
        try:
            text = getattr(response, "text", None)
        except Exception:  # noqa: BLE001 - function-call-only responses raise here
            text = None
        if text:
            return text
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        content = getattr(candidates[0], "content", None)
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or []
        return "".join((getattr(p, "text", "") or "") for p in parts)

    @staticmethod
    def _extract_usage(response: Any) -> dict:
        u = getattr(response, "usage_metadata", None)
        if u is None:
            return {}
        return {
            "prompt_tokens": getattr(u, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(u, "candidates_token_count", 0) or 0,
            # Optional — absent on older SDK responses or when nothing was cached.
            "cached_tokens": getattr(u, "cached_content_token_count", 0) or 0,
        }

    @staticmethod
    def _extract_finish_reason(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "stop"
        raw = getattr(candidates[0], "finish_reason", None)
        if raw is None:
            return "stop"
        # Gemini returns an enum or a string depending on SDK version.
        name = getattr(raw, "name", None) or str(raw)
        # Map Gemini's vocabulary to ours.
        if "STOP" in name:
            return "stop"
        if "MAX_TOKENS" in name:
            return "length"
        if "SAFETY" in name or "RECITATION" in name:
            return "blocked"
        return name.lower()


def _content_parts(content) -> list[dict]:
    """Turn an ``LLMMessage.content`` (str OR list[ContentPart]) into Gemini parts.
    A plain string → a single text part (unchanged from before). A list →
    text parts + ``inline_data`` parts (the shape proven in ``transcribe_audio``)."""
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict] = []
    for part in content or []:
        if isinstance(part, ContentPart):
            if part.type == "image" and part.inline_data:
                parts.append({"inline_data": part.inline_data})
            elif part.text is not None:
                parts.append({"text": part.text})
        elif isinstance(part, dict):  # tolerate raw dicts
            parts.append(part)
    return parts or [{"text": ""}]


def _dump(response: Any) -> dict:
    """Best-effort serialization so the raw_response can be inspected."""
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return {"text": getattr(response, "text", "")}
