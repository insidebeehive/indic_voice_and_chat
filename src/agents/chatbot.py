"""ChatBot agent (Phase 4).

Text-only counterpart to VoiceBotAgent. One ``handle_message(user_text)``
call per user turn:

1. Retrieve top chunks from the hybrid retriever for the user's question.
2. Build the RAG context block.
3. Compose the system + history + user messages and call the LLM (non-
   streaming — chat clients render the full message at once).
4. Parse the structured ChatBotResponse.
5. Apply the hallucination guard against the retrieved sources.
6. Persist user + agent turns to Redis.

The agent is deliberately stateless about telephony / VAD / audio — those
are voice concerns that the VoiceBotAgent handles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import replace as _replace_cfg
from typing import Any

from src.agents.base import AgentSession, BaseAgent
from src.chatbot.catalog import OPERATOR_TOOLS, PLAYER_TOOLS
from src.chatbot.media import prepare_multimodal_content
from src.chatbot.tools import BUILTIN_TOOLS, ESCALATE, OFFER_CALL, SEARCH_KB, SUBMIT_DEPOSIT_VERIFICATION
from src.dialogue.context import SessionStore
from src.dialogue.prompts import build_chatbot_system_prompt
from src.dialogue.response_parser import (
    ChatBotResponse,
    is_unusable_response,
    parse_chatbot_response,
)
from src.dialogue.slots import SlotFiller, SlotSchema
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult, ToolCall, ToolSpec
from src.rag.context_builder import (
    GuardConfig,
    apply_hallucination_guard,
    apply_no_grounding_guard,
    apply_unverified_data_guard,
    build_rag_context,
    search_combined,
)
from src.rag.retriever import HybridRetriever, RetrievedChunk
from src.utils.trace_id import current_trace_id

log = logging.getLogger(__name__)

# Sliding-window context, mirroring VoiceBotAgent.MAX_HISTORY_TURNS
# (src/agents/voicebot.py): the full transcript lives in ``session.turns``
# (used for the UI and history), but only the last MAX_HISTORY_TURNS
# exchanges are replayed to the LLM each turn. Without this the message list
# grows ~2 messages per turn, so per-turn latency/cost climbs over a long
# chat session.
MAX_HISTORY_TURNS = 10

# The one-shot retry (see _retry_if_unusable) fires only after a turn has
# ALREADY produced an unusable (canned-fallback) response. It gets its own,
# tighter timeout budget rather than sharing the primary turn's — src/api/chat.py
# wraps the whole turn in asyncio.wait_for(..., timeout=_TURN_TIMEOUT_S=90.0),
# tuned against real load-test data; an unbudgeted extra generate() call could
# push a turn past that ceiling under a provider's own retry/backoff (e.g.
# Gemini's escalating 429 backoff, up to ~35s), turning a degraded-but-real
# answer into a hard timeout — worse than not retrying. Mirrors the voicebot's
# _RETRY_HARD_TIMEOUT_S pattern (src/agents/voicebot.py), scaled for a text
# chat turn rather than a spoken one.
_CHAT_RETRY_TIMEOUT_S = 12.0

# Cap on the bumped max_tokens used for a retry attempted after a finish_reason
# of "length" (truncation) — see _retry_if_unusable. Keeps a runaway retry from
# ballooning cost/latency even for a tenant configured with an already-large
# max_tokens.
_CHAT_RETRY_MAX_TOKENS_CAP = 8192

# Cumulative wall-clock ceiling on ALL tool execution within ONE turn, shared
# across every round and every tool call the model emits. Not per-call: the
# model can emit multiple function_call parts in a single response (see
# gemini.py's _extract_tool_calls), so a per-call-only timeout doesn't bound
# a turn with several tool calls in one round.
_TOOL_BUDGET_S = 45.0
# Absolute ceiling for any single tool call, even a lone first one — kept
# below _TOOL_BUDGET_S so one call can never consume the entire budget and
# leave zero chance for a second call the model also asked for.
_TOOL_CALL_CEILING_S = 35.0
# Below this much remaining budget, don't bother starting a call at all —
# return the error result immediately rather than spending a connect
# round-trip on a call that would immediately be cut short anyway.
_TOOL_MIN_SLICE_S = 1.0

# Ceiling on the injected record_metric callback's own wall-clock time (see
# _emit_turn_metric). record_chat_turn_metric never raises, but it has no
# internal timeout of its own -- it does a flush then a commit, and
# src/models/database.py's engine sets no pool_timeout/statement_timeout, so
# an unbounded call there could stall for a long time. _emit_turn_metric runs
# as the LAST statement of handle_message, which sits INSIDE src/api/chat.py's
# 90s asyncio.wait_for(_run_turn(...)) turn-timeout wrapper -- on a turn that
# already spent most of that budget (the CRM tool budget alone is 45s), an
# unbounded metrics write could push the turn's total past 90s and get the
# whole reply (already fully computed) cancelled and discarded by the WS
# layer's timeout handler. A metrics row is never worth an errored reply, so
# this is bounded far below the turn timeout and swallowed like any other
# metrics-write failure.
_RECORD_METRIC_TIMEOUT_S = 2.0

# search_knowledge_base gets its OWN fixed timeout, entirely independent of
# _TOOL_BUDGET_S/_TOOL_CALL_CEILING_S. It is a RAG/embedding lookup, not a
# tenant CRM call — it never draws from, and is never shortened by, the CRM
# tool-call budget above. Previously KB search shared that budget: a slow CRM
# call could exhaust tool_elapsed_s, leaving a same-turn KB search skipped or
# cut to near-zero, which meant retrieved_all stayed empty and the
# hallucination guard (apply_hallucination_guard, gated on `if retrieved_all`
# below) silently never fired — ungrounded, unguarded answers with no signal
# anything went wrong. 15.0s is a generous fixed ceiling for an embedding
# search (RAG quality matters and this call type wasn't the source of
# the incident this budget was added for) while still bounding overall turn
# latency rather than leaving it truly unbounded.
_KB_SEARCH_TIMEOUT_S = 15.0

# Shared "the budget ran out" tool result. Fed back to the model as the tool's
# result content so the turn still produces a real (if degraded) answer
# instead of hanging or erroring out. Always return a fresh dict() copy of
# this (never the module-level object itself) — the result is JSON-serialized
# per-call and callers/tests may hold onto it.
_TOOL_BUDGET_EXHAUSTED = {
    "error": "This data source did not respond in time.",
    "failure": "timeout",
    "message": (
        "You could not verify this — do NOT state any specific number, amount, "
        "status, or date for it, and do not claim you checked or verified it. Tell "
        "the customer honestly that you can't verify this right now and offer to "
        "connect them to a human. That fully satisfies the response-quality bar for "
        "this turn."
    ),
}

# Executes a tenant-registered (CRM) tool call → a JSON-able result dict.
# ``timeout_s`` is this call's share of the turn's cumulative tool budget
# (see _TOOL_BUDGET_S) — keyword-only so a positional-arg executor can't
# silently swallow it as some other parameter.
CrmExecutor = Callable[[ToolCall, float], Awaitable[dict]]

# --- Failure-category classification + directive (Ticket #1762 fix, Steps 2/3) --
#
# Plain-English labels for the catalog CRM tools (src/chatbot/catalog.py) —
# LABELS ONLY, never a tool/endpoint name. The system prompt already has a
# "keep internals internal" rule; this directive text must not violate it
# either, so tool names never appear in anything built from this map. Not
# every tenant registers every one of these (tenants pick a subset via
# POST /chat/tools/from-catalog), but the map covers the full catalog so a
# label is always available without touching tenant config here.
_TOOL_CATEGORY_LABELS: dict[str, str] = {
    "get_player_wallet": "your wallet balance",
    "get_player_transactions": "your transaction history (deposits and withdrawals)",
    "get_player_latest_deposit_order": "your latest deposit order status",
    "get_player_bets": "your bet history",
    "get_player_bonuses": "your bonus and promotion details",
    "get_player_profile": "your account profile",
    "get_player_responsible_gaming": "your responsible-gaming settings",
    "get_payment_config": "your payment/deposit configuration",
    "get_referral_code": "your referral code",
    "get_sports_open_bets": "your open sports bets",
    "get_sports_match_status": "your bet results",
    "get_casino_game_history": "your casino game history",
    "get_matka_bids": "your Matka bid history",
    "get_game": "game availability",
    "get_game_providers": "game provider information",
    "get_operator_games_config": "which game categories are available",
    "get_matka_config": "Matka market configuration",
    "get_matka_result": "the Matka market result",
    "get_operator_promotions": "current promotions",
    "get_operator_platform_config": "platform configuration",
    "get_bet_limit": "the applicable bet limit",
    "get_market_holiday_schedule": "the market holiday schedule",
    "submit_deposit_verification": "your deposit verification request",
}
# Fallback for a custom tenant tool name not in the catalog above (tenants can
# register arbitrary CRM tools, not just catalog ones) — still never leaks the
# literal tool name.
_GENERIC_CATEGORY_LABEL = "some of your account details"


def _category_label(tool_name: str) -> str:
    return _TOOL_CATEGORY_LABELS.get(tool_name, _GENERIC_CATEGORY_LABEL)


def _tool_result_is_failure(result: object) -> bool:
    """True when a tool-result dict represents a failed/degraded call.

    Recognizes tool_executor.py's ``{"error": ..., "failure": ...}`` shape
    (Step 1), the corrected fallback payloads below (which now also carry
    both keys), — defensively — any dict with a genuinely non-empty
    ``"error"`` value even without a ``"failure"`` key, AND
    src/chatbot/deposit_verification.py's own error shape (review Fix 6),
    which uses ``{"status": "error", "message": ...}`` with NEITHER
    ``"error"`` nor ``"failure"`` — without this, a genuinely failed deposit-
    verification submit was invisible to this check and got folded into
    grounded text as if it had succeeded.
    """
    if not isinstance(result, dict):
        return False
    if result.get("failure"):
        return True
    if result.get("error"):
        return True
    return result.get("status") == "error"


def _tool_kind(tool_name: str) -> str:
    """Classify a tool call for metrics grouping (chat_tool_metrics.kind in
    the turn-metrics plan) -- lets zero-I/O local builders (escalate/offer
    call) be excluded from latency stats later, and CRM calls be told apart
    from the deposit-verification submit path and KB search."""
    if tool_name == SEARCH_KB:
        return "kb"
    if tool_name in (ESCALATE, OFFER_CALL):
        return "local"
    if tool_name == SUBMIT_DEPOSIT_VERIFICATION:
        return "deposit_verification"
    return "crm"


def _tool_outcome(result: object) -> str:
    """Classify a tool-result dict into a bounded outcome enum for metrics:
    ``ok`` | ``timeout`` | ``transport_error`` | ``error``. Deliberately a
    fixed enum, never a free-text error string -- a CRM error message can
    embed player data (see the plan's PII section), so there must be no
    column/field for it to land in.

    Does not classify ``skipped_budget`` -- the min-slice skip is a call-site
    decision (the call is never attempted at all), not something derivable
    from a result dict, so callers set that outcome themselves before ever
    reaching this function.

    KNOWN LIMITATION (documented, not fixed here -- a Phase 2 candidate):
    tool_executor.py's ``failure="http_error"`` (a CRM 4xx/5xx response --
    the HTTP call itself succeeded, but the upstream returned an error
    status) currently falls through to the generic ``"error"`` bucket below,
    same as an internal/deposit-verification error with neither an
    ``error``/``failure`` key. This is deliberately safer than a literal
    reading of the plan's derivation (`failure=="timeout"` /
    `failure=="transport_error"` / else "error" for a status=="error"
    shape), which would have missed http_error entirely and misclassified a
    real CRM 500 as ``ok``. But for the CRM-vendor-accountability argument
    this whole metrics pipeline exists to support, folding "the vendor
    returned a 500" and "our own code raised" into one enum value loses
    signal that would matter for a per-tool failure-rate dashboard. Add a
    dedicated ``http_error`` outcome value in Phase 2 rather than now.
    """
    if not _tool_result_is_failure(result):
        return "ok"
    if isinstance(result, dict):
        failure = result.get("failure")
        if failure == "timeout":
            return "timeout"
        if failure == "transport_error":
            return "transport_error"
    return "error"


def _build_failure_directive(labels: list[str], escalate: bool) -> str:
    """Synthetic role=user directive (Step 2). Deliberately role="user", not
    "system" — both the Gemini and Claude adapters hoist system messages into
    the top-level instruction, which would move this away from the tool
    results it must sit next to. Labels only, never a tool/endpoint name.
    """
    joined = "; ".join(labels)
    lines = [
        f"SYSTEM NOTE (not from the customer): the data needed to answer about "
        f"{joined} could NOT be retrieved or verified just now.",
        "Do not state any specific number, amount, status, or date for this, and do "
        "not claim you have checked, verified, or looked this up — you have not.",
        'Saying plainly "I can\'t verify this right now, let me connect you to a '
        'human who can check" fully satisfies the response-quality requirement for '
        "this turn — it is not an incomplete or lesser answer.",
    ]
    if escalate:
        lines.append(
            "This same information has now failed to load on more than one turn in a "
            "row — do not promise to check again or ask the customer to wait further; "
            "proactively offer to connect them to a human right now (the customer "
            "still must confirm before you actually escalate)."
        )
    return "\n".join(lines)

# Unicode block boundaries for common Indic scripts.
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "Hindi"),       # Devanagari (Hindi, Marathi, Sanskrit)
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Punjabi"),     # Gurmukhi
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
]


def _detect_script(text: str) -> str | None:
    """Return an Indic language name when *text* is dominantly written in that
    language's own script (e.g. Devanagari for Hindi). Returns None for
    empty/purely numeric/punctuation text, for mixed script, AND for pure
    Latin/ASCII text.

    Pure-Latin text is deliberately left undetected rather than assumed to be
    English: it's ambiguous between real English and a romanized Indic
    language (Hinglish — "mera balance kya hai"). This used to return
    "English" for any Latin-dominant message, which injected a hard "MUST be
    in English" directive into the prompt that overrode the system prompt's
    own (correct) LANGUAGE rule — reply in Roman Hinglish for Roman-script
    input, never force English/Devanagari onto it. Returning None here lets
    that rule govern instead of contradicting it.
    """
    if not text:
        return None
    indic_lang: str | None = None
    indic_count = 0
    latin_count = 0
    for ch in text:
        cp = ord(ch)
        if ch.isascii() and ch.isalpha():
            latin_count += 1
        else:
            for lo, hi, lang in _SCRIPT_RANGES:
                if lo <= cp <= hi:
                    indic_count += 1
                    indic_lang = lang
                    break
    total = indic_count + latin_count
    if total == 0:
        return None
    if indic_count / total >= 0.6:
        return indic_lang
    return None  # pure/majority Latin, or too mixed to call — the prompt's
    # LANGUAGE section already handles script-matching for these


# Common romanized-Hindi tokens. Deliberately excludes anything that collides
# with English words ("do", "ho", "par", "se", "the", "kar"…) — a false
# positive would flip an English reply into Hinglish. Word-boundary matched,
# lowercase.
_HINGLISH_MARKERS = frozenset({
    "kya", "hai", "hain", "mera", "mere", "meri", "nahi", "nahin", "kaise",
    "kaisa", "karo", "raha", "rahi", "rahe", "aap", "aapka", "aapki", "kab",
    "kyu", "kyun", "kyon", "batao", "bataiye", "chahiye", "hua", "hoga",
    "hogi", "gaya", "gayi", "kitna", "kitni", "kitne", "wala", "wale",
    "mein", "bhai", "bhaiya", "didi", "paisa", "paise", "rupay", "rupaye",
    "jaldi", "madad", "shukriya", "dhanyavaad", "haan", "theek", "thik",
    "accha", "acha", "bolo", "suno", "dekho", "milega", "milegi", "karna",
    "kahan", "kaun", "kaunsa", "toh", "abhi", "kardo", "krdo", "karde",
})


def _latin_language_hint(text: str) -> str | None:
    """Classify a Roman-script message as "Hinglish" or "English" — or None
    when it's too short to carry a signal (bare "ok"/"thanks", which should
    follow the conversation's existing language, not force a switch).

    An advisory "pick whichever fits" directive proved too weak: with a
    Hinglish conversation history and default_language="hi", the model kept
    replying in Hinglish even to plain English ("tell me about this site").
    The classification must be deterministic so the directive can be firm.
    """
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if w]
    if not words:
        return None
    if any(w in _HINGLISH_MARKERS for w in words):
        return "Hinglish"
    if len(words) >= 3:
        return "English"
    return None  # short, marker-free ("ok", "yes") — no signal either way


def _chunk_source(chunk: RetrievedChunk) -> str:
    md = chunk.document.metadata or {}
    fn = md.get("filename") or md.get("document_id") or chunk.document.id
    page = md.get("page", md.get("section"))
    return f"{fn}:{page}" if page is not None else str(fn)


def _usage_tokens(result: LLMResult | None) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) from an LLMResult, defensively —
    ``LLMResult.usage`` is typed as a dict but at least one adapter
    (openai_compat, pre-fix) has been seen to pass ``None``; treat that (and a
    None result, e.g. a failed retry) as ``{}`` rather than raising."""
    if result is None:
        return 0, 0
    usage = result.usage or {}
    return usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0


@dataclass(frozen=True)
class ChatToolMetric:
    """One tool-call's metrics within a turn (chat_tool_metrics grain in the
    turn-metrics plan). ``id``/``turn_id``/``tenant_id``/``created_at`` are
    the future DB row's business (Phase 2), not this dataclass's."""
    tool_name: str
    kind: str            # "crm" | "kb" | "local" | "deposit_verification"
    latency_ms: int
    outcome: str          # "ok" | "timeout" | "transport_error" | "error" | "skipped_budget"
    budget_slice_ms: int
    round_index: int


@dataclass(frozen=True)
class ChatTurnMetrics:
    """Aggregate + per-tool-call timing/outcome data for one agent turn
    (chat_turn_metrics grain in the turn-metrics plan). Phase 1: in-process
    only -- attached to ChatTurnResult, never persisted here. Fields omit
    ``id``/``tenant_id``/``crm_id``/``created_at``, which are the future
    write path's (a factory closure / the DB), not the agent's, business.
    """
    trace_id: str | None
    path: str             # "tools" | "single_shot"
    llm_provider: str
    llm_model: str
    action: str
    total_ms: int
    llm_total_ms: int
    llm_calls: int
    # Budgeted CRM tool time only (tool_elapsed_s * 1000) -- directly
    # comparable to _TOOL_BUDGET_S. KB search is deliberately excluded (see
    # kb_search_ms/kb_searches below) since it draws from its own independent
    # budget (_KB_SEARCH_TIMEOUT_S), not this one.
    tool_total_ms: int
    # tool_calls counts every CRM/deposit-verification call the model made,
    # INCLUDING ones that were skipped_budget (never actually dispatched --
    # see tool_calls_skipped). tool_failures/tool_timeouts count only calls
    # that were actually attempted and failed/timed out -- skipped_budget is
    # excluded from both, since "never tried" isn't the same failure mode as
    # "tried and failed". Consequence, PINNED HERE before anyone builds a
    # dashboard on it: a naive `tool_failures / tool_calls` UNDERSTATES
    # degradation whenever calls were skipped -- those calls didn't fail
    # outright, but they also didn't succeed, and the ratio silently treats
    # them as successes. A true "how bad" number needs
    # `(tool_failures + tool_calls_skipped) / tool_calls`.
    tool_calls: int
    tool_failures: int
    tool_timeouts: int
    tool_calls_skipped: int
    kb_search_ms: int
    kb_searches: int
    retrieved_chunks: int
    rounds: int
    rounds_exhausted: bool
    retry_fired: bool
    failure_directive_fired: bool
    failure_directive_escalated: bool
    guard_hallucination_fired: bool
    guard_no_grounding_fired: bool
    guard_unverified_data_fired: bool
    escalated: bool
    tools: tuple[ChatToolMetric, ...] = ()


@dataclass
class ChatTurnResult:
    response: ChatBotResponse
    retrieved: list[RetrievedChunk]
    rag_context_chars: int
    escalation: dict | None = None
    call_offer: dict | None = None
    # Usage accumulated across every generate() call in this turn (a turn is
    # rarely just one LLM call — see _single_shot's retry and
    # _handle_with_tools' multi-round loop) + the platform LLM identity used,
    # for src/api/chat_cost.py's per-turn cost lookup.
    input_tokens: int = 0
    output_tokens: int = 0
    llm_provider: str = ""
    llm_model: str = ""
    # Phase 1 of the turn-metrics plan: rich in-process timing/outcome data,
    # assembled defensively (never breaks a turn on a metrics-assembly bug —
    # see _single_shot/_handle_with_tools). None only when metrics assembly
    # itself failed; a turn is never blocked on this.
    metrics: ChatTurnMetrics | None = None


class ChatBotAgent(BaseAgent):
    def __init__(
        self,
        session: AgentSession,
        llm: ILLMProvider,
        retriever: HybridRetriever,
        crm_retriever: HybridRetriever | None = None,
        llm_config: LLMConfig | None = None,
        company_name: str = "[Your Company]",
        language_default: str = "en",
        tenant_timezone: str = "Asia/Kolkata",
        prompt_pack: str = "generic",
        store: SessionStore | None = None,
        guard_config: GuardConfig | None = None,
        max_context_chars: int = 2000,
        enable_tools: bool = False,
        crm_tools: list[ToolSpec] | None = None,
        crm_executor: CrmExecutor | None = None,
        deposit_verification_executor: Callable | None = None,
        max_tool_rounds: int = 2,
        llm_provider: str = "",
        llm_model: str = "",
        session_id: str | None = None,
        ticket_id: str | None = None,
        record_metric: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        # ChatBot doesn't need slots — pass an empty schema so BaseAgent is happy.
        super().__init__(
            session=session,
            state_machine=None,  # type: ignore[arg-type] — chatbot doesn't drive a call FSM
            slots=SlotFiller(SlotSchema()),
            store=store,
        )
        self._llm = llm
        self._retriever = retriever
        self._crm_retriever = crm_retriever
        # Ordered list: the linked CRM's shared KB first, then tenant-specific.
        self._retrievers: list[HybridRetriever] = [
            r for r in [crm_retriever, retriever] if r is not None
        ]
        self._llm_config = llm_config or LLMConfig(response_format="json", max_tokens=4096)
        self._company = company_name
        self._language = language_default
        self._tenant_timezone = tenant_timezone
        self._prompt_pack = prompt_pack
        self._guard = guard_config
        self._max_context_chars = max_context_chars
        # Agentic tool-calling (opt-in): builtin tools (search KB / escalate /
        # offer call) + any tenant CRM tools. Off by default so the single-shot
        # RAG path is unchanged.
        self._enable_tools = enable_tools
        self._crm_tools = crm_tools or []
        self._crm_executor = crm_executor
        self._deposit_verification_executor = deposit_verification_executor
        self._max_tool_rounds = max_tool_rounds
        # Identity of the platform LLM this agent was built with — threaded
        # through to ChatTurnResult for per-turn token-cost lookup
        # (src/api/chat_cost.py). Chat always runs the platform LLM (see
        # make_chatbot_factory in src/bootstrap.py), never a per-tenant one.
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        # External-ticket / session identifiers threaded through purely for
        # log correlation (see tool-call-failure logs below) — never used for
        # behavior. Both optional; None when the caller (e.g. bootstrap.py's
        # make_chatbot_factory) has no crm_ticket_id for this session.
        self._session_id = session_id
        self._ticket_id = ticket_id
        # Phase 2 of the turn-metrics plan (docs/superpowers/plans/
        # 2026-09-08-chatbot-turn-metrics.md, §4): injected write-path
        # callback, mirroring VoiceBotAgent's record_metric inversion of
        # control exactly. This module intentionally imports ZERO model
        # modules -- importing one here would make the ~50 unit tests in
        # this file DB-aware. The closure that builds this callable (see
        # make_chatbot_factory in src/bootstrap.py) owns tenant_id/crm_id and
        # the actual DB write (record_chat_turn_metric); this agent only
        # ever hands it a plain dict of ids/enums/numbers -- see the PII
        # docstring in src/models/chat_turn_metrics.py for what that dict may
        # never contain.
        self._record_metric = record_metric
        # Ticket #1762 fix, Step 3: per-agent-instance (== per WS connection)
        # counter of CONSECUTIVE turns where the same category (keyed by its
        # plain-English label, never a tool name) failed. In-memory only —
        # resets on reconnect (explicit, documented tradeoff; durability across
        # reconnects is a follow-up, not part of this fix). Drives the
        # escalation sentence in _build_failure_directive once a category has
        # failed on 2 turns running.
        self._consecutive_category_failures: dict[str, int] = {}

    async def _emit_turn_metric(self, metrics: ChatTurnMetrics | None) -> None:
        """Persist ``metrics`` via the injected ``record_metric`` callback
        (turn-metrics plan §4), if one was supplied and turn-metrics assembly
        actually succeeded this turn. Two layers of protection against a
        metrics-write failure ever reaching a live turn, same as voice's
        record_metric idiom (src/agents/voicebot.py): the callback itself
        (``record_chat_turn_metric``, src/models/chat_turn_metrics.py) never
        raises internally, and this call site catches anyway in case the
        injected callable is something else entirely (e.g. a test double).

        Does nothing when ``metrics`` is None -- that already means
        ChatTurnMetrics construction failed and was logged as a warning at
        the call site above, so there is nothing correct left to persist.
        Only ids, provider/model names, bounded enum strings, integers, and
        booleans ever go into this payload -- see the PII section of
        src/models/chat_turn_metrics.py's module docstring for what must
        never land here.

        The callback is additionally bounded by ``_RECORD_METRIC_TIMEOUT_S``:
        this method runs as the LAST statement of ``handle_message``, which is
        inside the WS layer's 90s per-turn ``asyncio.wait_for`` (see
        ``_TURN_TIMEOUT_S`` in src/api/chat.py) -- an unbounded metrics write
        stalling near that ceiling would otherwise get the turn's already-
        computed reply cancelled and discarded, a failure mode that did not
        exist before this write path (``_persist_turn``'s own DB write runs
        strictly after that wrapper returns).
        """
        if self._record_metric is None or metrics is None:
            return
        try:
            await asyncio.wait_for(self._record_metric({
                "session_id": self._session_id or "",
                "trace_id": metrics.trace_id,
                "path": metrics.path,
                "llm_provider": metrics.llm_provider,
                "llm_model": metrics.llm_model,
                "action": metrics.action,
                "metrics": {
                    "total_ms": metrics.total_ms,
                    "llm_total_ms": metrics.llm_total_ms,
                    "llm_calls": metrics.llm_calls,
                    "tool_total_ms": metrics.tool_total_ms,
                    "tool_calls": metrics.tool_calls,
                    "tool_failures": metrics.tool_failures,
                    "tool_timeouts": metrics.tool_timeouts,
                    "tool_calls_skipped": metrics.tool_calls_skipped,
                    "kb_search_ms": metrics.kb_search_ms,
                    "kb_searches": metrics.kb_searches,
                    "retrieved_chunks": metrics.retrieved_chunks,
                    "rounds": metrics.rounds,
                    "rounds_exhausted": metrics.rounds_exhausted,
                    "retry_fired": metrics.retry_fired,
                    "failure_directive_fired": metrics.failure_directive_fired,
                    "failure_directive_escalated": metrics.failure_directive_escalated,
                    "guard_hallucination_fired": metrics.guard_hallucination_fired,
                    "guard_no_grounding_fired": metrics.guard_no_grounding_fired,
                    "guard_unverified_data_fired": metrics.guard_unverified_data_fired,
                    "escalated": metrics.escalated,
                },
                "tools": [
                    {
                        "tool_name": t.tool_name, "kind": t.kind, "latency_ms": t.latency_ms,
                        "outcome": t.outcome, "budget_slice_ms": t.budget_slice_ms,
                        "round_index": t.round_index,
                    }
                    for t in metrics.tools
                ],
            }), timeout=_RECORD_METRIC_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - never break a live turn on a metrics-write failure
            # Catches asyncio.TimeoutError (from the wait_for above) the same
            # way as any other metrics-write failure -- a slow/stalled DB
            # write degrades to "no row, one WARNING", never an errored turn.
            log.warning("record_metric failed; continuing without persistence", exc_info=True)

    async def handle_message(self, user_text: str) -> ChatTurnResult:
        if not user_text or not user_text.strip():
            return ChatTurnResult(
                response=ChatBotResponse(
                    response_text="",
                    language=self._language,
                    parse_error="empty user input",
                ),
                retrieved=[],
                rag_context_chars=0,
            )
        user_msg = LLMMessage(role="user", content=user_text)
        if self._enable_tools:
            return await self._handle_with_tools(user_msg, query_text=user_text)
        return await self._single_shot(user_msg, query_text=user_text)

    async def handle_image(
        self, media: object, mime: str, text: str = "",
    ) -> ChatTurnResult:
        """Handle an image/video (+ optional caption) from the customer. The
        media is sent to the multimodal LLM; retrieval (if any) uses the caption."""
        parts = prepare_multimodal_content(text, media, mime)
        user_msg = LLMMessage(role="user", content=parts)
        if self._enable_tools:
            return await self._handle_with_tools(user_msg, query_text=text)
        return await self._single_shot(user_msg, query_text=text)

    # --- Single-shot RAG path (no tools) -------------------------------

    async def _single_shot(self, user_msg: LLMMessage, query_text: str) -> ChatTurnResult:
        turn_start = time.perf_counter()
        # 1. Retrieval (on the text part; multimodal-only turns skip it)
        retrieval_start = time.perf_counter()
        retrieved = await search_combined(query_text, self._retrievers) if query_text.strip() else []
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        # 2. Build context
        rag = build_rag_context(retrieved, max_chars=self._max_context_chars)
        # 3. Compose messages
        messages = self._compose(rag.text, user_msg, query_text=query_text)
        # 4. LLM
        llm_start = time.perf_counter()
        result = await self._llm.generate(messages, self._llm_config)
        llm_ms_list = [(time.perf_counter() - llm_start) * 1000]
        in_tok, out_tok = _usage_tokens(result)
        response = parse_chatbot_response(result.text)
        retry_fired = False
        # Only retry a genuinely unusable turn — one where the parser had to
        # fall back to one of its canned lines (empty input, missing
        # response_text, or truncated/malformed JSON — see
        # response_parser.is_unusable_response). A parse_error from
        # legitimate plain (non-JSON) text does NOT count: _fallback_text
        # already returns that verbatim as a perfectly usable response_text,
        # and retrying it too would waste a call on an already-good answer.
        if is_unusable_response(response.response_text):
            # Nothing has been shown to the customer yet (chat is request/
            # response, unlike voice's incremental TTS) — safe to regenerate
            # once before falling back to the canned "couldn't formulate an
            # answer" line.
            retry_fired = True
            retried_result, retry_ms = await self._retry_if_unusable(
                result.finish_reason, messages, self._llm_config,
            )
            llm_ms_list.append(retry_ms)
            retry_in, retry_out = _usage_tokens(retried_result)
            in_tok += retry_in
            out_tok += retry_out
            if retried_result is not None:
                # Adopt whatever the retry produces unconditionally — including
                # legitimate plain (non-JSON) text, which parse_chatbot_response
                # already returns verbatim as a usable response_text despite
                # setting parse_error. If the retry is ALSO genuinely unusable,
                # this parses to the same canned fallback as before — no worse
                # off than not retrying at all.
                response = parse_chatbot_response(retried_result.text)
        # 5. Guard. Skip only for a multimodal turn with no retrieval — that
        # answer is grounded in the image, not the (empty) knowledge base, so the
        # no-retrieval fallback would wrongly clobber it. Text turns are unchanged.
        multimodal = isinstance(user_msg.content, list)
        _guard_before = (response.response_text, response.confidence)
        if retrieved or not multimodal:
            response = apply_hallucination_guard(response, rag, self._guard)
        guard_hallucination_fired = (response.response_text, response.confidence) != _guard_before
        # 6. Persist
        await self._persist(user_msg, query_text, response, len(retrieved))
        total_ms = (time.perf_counter() - turn_start) * 1000
        # Computed outside the try/except below (same reasoning as
        # _handle_with_tools): the log line must show real measurements even
        # in the narrow case where ChatTurnMetrics construction itself fails.
        llm_total_ms_val = round(sum(llm_ms_list))
        escalated_flag = response.action == "escalate"
        metrics: ChatTurnMetrics | None = None
        try:
            # Never let a metrics-assembly bug break a live turn — see
            # src/agents/voicebot.py's record_metric idiom and
            # src/models/turn_metrics.py's never-raises docstring for the
            # established house style this mirrors.
            metrics = ChatTurnMetrics(
                trace_id=current_trace_id(),
                path="single_shot",
                llm_provider=self._llm_provider,
                llm_model=self._llm_model,
                action=response.action,
                total_ms=round(total_ms),
                llm_total_ms=llm_total_ms_val,
                llm_calls=len(llm_ms_list),
                tool_total_ms=0,
                tool_calls=0,
                tool_failures=0,
                tool_timeouts=0,
                tool_calls_skipped=0,
                # DECISION (documented, not a bug): the single-shot path's
                # retrieval (search_combined, above) is measured as
                # retrieval_ms and logged, but deliberately NOT written into
                # kb_search_ms/kb_searches here -- those two fields are
                # scoped specifically to the tool-calling loop's
                # search_knowledge_base TOOL CALL (see _handle_with_tools),
                # which is a different mechanism (explicit model-invoked tool
                # vs. automatic pre-retrieval) even though both do a KB
                # lookup. Consequence: a future cross-path aggregate like
                # "avg_kb_search_ms across all turns" would silently exclude
                # every single-shot turn's real retrieval time rather than
                # averaging it in -- any such consumer must filter/group by
                # `path`, or this field must be revisited to populate
                # retrieval_ms here too.
                kb_search_ms=0,
                kb_searches=0,
                retrieved_chunks=len(retrieved),
                rounds=1,
                rounds_exhausted=False,
                retry_fired=retry_fired,
                failure_directive_fired=False,
                failure_directive_escalated=False,
                # DECISION (documented, not a bug — see the plan's §11.6):
                # this can under-report. The before/after (response_text,
                # confidence) snapshot is the plan-prescribed way to detect a
                # guard firing without changing the guards themselves (which
                # always return a fresh copy, making identity comparison
                # useless). But apply_hallucination_guard's citation-dropping
                # branch (context_builder.py) only mutates sources_used, not
                # response_text/confidence, when confidence was ALREADY
                # "low" going in -- that specific case is a false negative
                # here (guard did something, snapshot shows no change).
                guard_hallucination_fired=guard_hallucination_fired,
                guard_no_grounding_fired=False,
                guard_unverified_data_fired=False,
                escalated=escalated_flag,
            )
        except Exception:  # noqa: BLE001 - metrics assembly must never break a reply
            log.warning("chat turn metrics assembly failed; continuing without metrics",
                        exc_info=True,
                        extra={"ticket_id": self._ticket_id, "session_id": self._session_id})
        log.info(
            "chat turn done (single_shot path)",
            extra={
                "ticket_id": self._ticket_id, "session_id": self._session_id,
                "path": "single_shot",
                "total_ms": round(total_ms),
                "llm_ms_list": [round(ms) for ms in llm_ms_list],
                "llm_calls": len(llm_ms_list),
                "llm_total_ms": llm_total_ms_val,
                "retrieval_ms": round(retrieval_ms),
                "retrieved_count": len(retrieved),
                "action": response.action,
                "retry_fired": retry_fired,
                "guard_hallucination_fired": guard_hallucination_fired,
                "escalated": escalated_flag,
            },
        )
        await self._emit_turn_metric(metrics)
        return ChatTurnResult(
            response=response, retrieved=retrieved, rag_context_chars=len(rag.text),
            input_tokens=in_tok, output_tokens=out_tok,
            llm_provider=self._llm_provider, llm_model=self._llm_model,
            metrics=metrics)

    # --- Tool-calling path (agentic) -----------------------------------

    async def _handle_with_tools(self, user_msg: LLMMessage, query_text: str) -> ChatTurnResult:
        turn_start = time.perf_counter()
        llm_ms_list: list[float] = []
        tool_ms_list: list[tuple[str, float]] = []
        tool_metrics: list[ChatToolMetric] = []
        rounds = 0
        rounds_exhausted = False
        retry_fired = False
        # Turn-scoped flags for the failure directive (ticket #1762 fix) —
        # deliberately NOT derived from failed_category_names/directive_index
        # at the end of the turn. The distinguishing case is SAME-TURN
        # RECOVERY: round 1's tool call fails (directive appended,
        # directive_index set), round 2's retry of the SAME tool succeeds --
        # failed_category_names is cleared and the stale directive message is
        # deleted, resetting directive_index back to None (see the "Keep
        # exactly one synthetic directive message" block below). An
        # end-of-turn derivation (`directive_index is not None`) would then
        # report False for a turn where the directive genuinely fired and
        # reached the model in round 1 -- these flags remember that instead.
        # (A mid-loop `break`, by contrast, does NOT create this discrepancy:
        # it happens before the per-round failed_category_names/directive
        # bookkeeping for that round runs, so directive_index simply retains
        # whatever value the prior round left it at.)
        directive_fired_this_turn = False
        directive_escalated_this_turn = False
        # Cumulative tool time spent so far THIS TURN — persists across both
        # rounds (initialized here, outside the round loop below), which is
        # what makes _TOOL_BUDGET_S a per-turn budget rather than a per-round
        # one that would silently reset and double the real ceiling.
        tool_elapsed_s = 0.0
        tools = list(BUILTIN_TOOLS) + list(self._crm_tools)
        # Tools fetch their own context (search_knowledge_base), so the system
        # prompt starts without a pre-built RAG block.
        messages = self._compose("", user_msg, query_text=query_text)
        retrieved_all: list[RetrievedChunk] = []
        tool_calls_made: list[str] = []
        escalation: dict | None = None
        call_offer: dict | None = None
        text = ""
        in_tok = 0
        out_tok = 0
        # Ticket #1762 fix, Steps 2/3: which CRM-data-tool categories (keyed by
        # tool NAME internally — converted to labels only when building
        # customer-facing/model-facing text) are currently unresolved-failed
        # THIS turn, the index of the single synthetic directive message
        # currently in `messages` (kept in sync every round, never
        # accumulated), and the text this turn actually grounded in real data
        # (successful CRM/deposit-verification tool results only — KB results
        # are already covered by rag_context below) for the Step 5 guard.
        failed_category_names: set[str] = set()
        directive_index: int | None = None
        grounded_tool_texts: list[str] = []
        # Snapshot of the PRIOR turns' consecutive-failure counts, taken before
        # this turn mutates them — used to decide whether THIS turn's directive
        # should escalate (i.e. this is the 2nd turn running with the same
        # failure), never the post-this-turn state.
        _prior_failure_counts = dict(self._consecutive_category_failures)
        # JSON response-format is incompatible with tools (Gemini), so the loop
        # runs in text mode; the structured fields are derived from tool results.
        cfg = LLMConfig(
            model=self._llm_config.model,
            temperature=self._llm_config.temperature,
            max_tokens=self._llm_config.max_tokens,
            response_format="text",
            tools=tools,
        )
        for _ in range(self._max_tool_rounds):
            rounds += 1
            llm_start = time.perf_counter()
            result = await self._llm.generate(messages, cfg)
            llm_ms_list.append((time.perf_counter() - llm_start) * 1000)
            _round_in, _round_out = _usage_tokens(result)
            in_tok += _round_in
            out_tok += _round_out
            log.debug(
                "chatbot llm turn: finish=%s usage=%s text_len=%d tool_calls=%d",
                result.finish_reason, result.usage, len(result.text or ""), len(result.tool_calls),
                extra={"ticket_id": self._ticket_id, "session_id": self._session_id},
            )
            if not result.tool_calls:
                text = result.text
                break
            messages.append(LLMMessage(role="assistant", content="", tool_calls=result.tool_calls))
            round_failed_names: set[str] = set()
            round_succeeded_names: set[str] = set()
            for i, tc in enumerate(result.tool_calls):
                tool_start = time.perf_counter()
                slice_s = 0.0
                if tc.name == SEARCH_KB:
                    # KB search has its own independent budget (see
                    # _KB_SEARCH_TIMEOUT_S) — it must never draw from, or be
                    # starved by, the CRM tool-call budget below.
                    out, chunks, esc, off = await self._exec_kb_tool(tc)
                elif tc.name in (ESCALATE, OFFER_CALL):
                    # Pure local dict builders, zero I/O (see _dispatch_tool) —
                    # they cannot time out, and must never be starved by an
                    # exhausted CRM budget: that would silently swallow a
                    # handoff-to-human the model explicitly requested.
                    out, chunks, esc, off = await self._dispatch_tool(tc, 0.0)
                else:
                    # Fair-share the remaining cumulative CRM budget across the
                    # CRM calls left in THIS response (KB/escalate/offer_call
                    # excluded — budgeted/unbudgeted separately, see above),
                    # capped per-call at _TOOL_CALL_CEILING_S so one call
                    # (especially a lone first one) can never eat the whole
                    # turn budget in a single shot.
                    remaining = _TOOL_BUDGET_S - tool_elapsed_s
                    calls_left = sum(
                        1 for t in result.tool_calls[i:]
                        if t.name not in (SEARCH_KB, ESCALATE, OFFER_CALL)
                    )
                    slice_s = min(_TOOL_CALL_CEILING_S, remaining / calls_left) if calls_left > 0 else 0.0
                    out, chunks, esc, off = await self._exec_tool(tc, slice_s)
                elapsed = time.perf_counter() - tool_start
                if tc.name not in (SEARCH_KB, ESCALATE, OFFER_CALL):
                    tool_elapsed_s += elapsed
                    # Ticket #1762 fix, Steps 2/5: classify this CRM/deposit-
                    # verification call as failed or succeeded (SEARCH_KB is
                    # excluded — its content is already covered by rag_context
                    # below; ESCALATE/OFFER_CALL are zero-I/O local builders
                    # that cannot fail). A successful result's JSON is also
                    # collected as grounded text for the Step 5 guard.
                    if _tool_result_is_failure(out):
                        round_failed_names.add(tc.name)
                    else:
                        round_succeeded_names.add(tc.name)
                        grounded_tool_texts.append(json.dumps(out))
                tool_ms_list.append((tc.name, elapsed * 1000))
                try:
                    # Narrow on purpose — this wraps ONLY the per-tool metric
                    # construction, never the surrounding tool-dispatch logic
                    # above (which must keep running/raising normally). This
                    # code runs on every production chat turn's hot tool loop,
                    # so "metrics must never break a live turn" has to hold
                    # here too, not just at the end-of-turn ChatTurnMetrics
                    # assembly below.
                    kind = _tool_kind(tc.name)
                    # The min-slice skip (_exec_tool's timeout_s <
                    # _TOOL_MIN_SLICE_S branch, "the model asked for a tool
                    # and we never even tried") only applies to the
                    # fair-shared CRM/deposit-verification path — KB search
                    # and the zero-I/O local builders never go through that
                    # branch, so they can never be "skipped_budget".
                    if kind in ("crm", "deposit_verification") and slice_s < _TOOL_MIN_SLICE_S:
                        outcome = "skipped_budget"
                    else:
                        outcome = _tool_outcome(out)
                    # budget_slice_ms: the CRM/deposit-verification fair-share
                    # slice computed above; KB search's own fixed, independent
                    # timeout for a KB call; 0 (unbudgeted) for the zero-I/O
                    # local builders, which never draw from any budget.
                    if kind == "kb":
                        budget_slice_ms = round(_KB_SEARCH_TIMEOUT_S * 1000)
                    elif kind == "local":
                        budget_slice_ms = 0
                    else:
                        budget_slice_ms = round(slice_s * 1000)
                    tool_metrics.append(ChatToolMetric(
                        tool_name=tc.name, kind=kind, latency_ms=round(elapsed * 1000),
                        outcome=outcome, budget_slice_ms=budget_slice_ms,
                        round_index=rounds - 1,
                    ))
                except Exception:  # noqa: BLE001 - per-tool metrics must never break a live turn
                    log.warning(
                        "chat per-tool metrics collection failed; continuing without this entry",
                        exc_info=True,
                        extra={"ticket_id": self._ticket_id, "session_id": self._session_id,
                               "tool": tc.name},
                    )
                retrieved_all.extend(chunks)
                if tc.name not in (ESCALATE, OFFER_CALL):
                    tool_calls_made.append(tc.name)
                escalation = esc or escalation
                call_offer = off or call_offer
                messages.append(LLMMessage(
                    role="tool", name=tc.name, tool_call_id=tc.id, content=json.dumps(out)))
            # A category that recovers this round must be cleared, not flagged
            # (success always wins within the round it happens in).
            failed_category_names -= round_succeeded_names
            failed_category_names |= (round_failed_names - round_succeeded_names)
            # Keep exactly one synthetic directive message in `messages`,
            # always freshest-last (adjacent to the tool results it must sit
            # next to) — remove the stale one before deciding whether to add a
            # new one, so a resolved turn never leaves a stale "can't verify
            # X" message behind after X actually succeeded.
            if directive_index is not None:
                del messages[directive_index]
                directive_index = None
            if failed_category_names:
                labels = list(dict.fromkeys(
                    _category_label(n) for n in sorted(failed_category_names)))
                escalate = any(
                    _prior_failure_counts.get(_category_label(n), 0) >= 1
                    for n in failed_category_names
                )
                directive_fired_this_turn = True
                if escalate:
                    directive_escalated_this_turn = True
                messages.append(LLMMessage(
                    role="user", content=_build_failure_directive(labels, escalate)))
                directive_index = len(messages) - 1
        else:
            # Ran out of rounds still wanting tools — force a final plain answer.
            rounds_exhausted = True
            llm_start = time.perf_counter()
            result = await self._llm.generate(
                messages, LLMConfig(temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                                    response_format="text"))
            llm_ms_list.append((time.perf_counter() - llm_start) * 1000)
            _forced_in, _forced_out = _usage_tokens(result)
            in_tok += _forced_in
            out_tok += _forced_out
            text = result.text

        # Step 3: advance the consecutive-turn-failure counters for the NEXT
        # turn's escalation decision, based on how this turn actually ended.
        # A category not in the final failed set this turn is implicitly
        # reset to 0 by omission (fresh dict, not a merge).
        final_failed_labels = {_category_label(n) for n in failed_category_names}
        self._consecutive_category_failures = {
            label: _prior_failure_counts.get(label, 0) + 1 for label in final_failed_labels
        }
        if failed_category_names:
            log.warning(
                "chat turn ended with unresolved tool-data failures: %s",
                sorted(failed_category_names),
                extra={"ticket_id": self._ticket_id, "session_id": self._session_id},
            )

        rag = build_rag_context(retrieved_all, max_chars=self._max_context_chars)
        # Dedupe tool-retrieved sources, preserving order.
        sources = list(dict.fromkeys(_chunk_source(c) for c in retrieved_all))
        # The model usually emits the structured JSON envelope (per the system
        # prompt) even in the tool loop, so PARSE it — otherwise the customer gets
        # raw JSON. ``raw`` is set only when a real envelope was found; for a plain
        # text answer keep it verbatim (the JSON-fallback would mangle it). Then
        # overlay tool-derived signals (retrieved sources, escalation).
        parsed = parse_chatbot_response(text)
        # Only retry a genuinely unusable turn — one where the parser had to
        # fall back to one of its canned lines. NOT a parse_error from
        # legitimate plain (non-JSON) text, which _fallback_text already
        # returns verbatim as a perfectly usable response_text. Retrying that
        # case too would waste a call on an already-good answer.
        if is_unusable_response(parsed.response_text):
            # Nothing has been shown to the customer yet — safe to regenerate
            # once before falling back. The tool rounds already completed and
            # are reflected in ``messages`` (tool-call + tool-result turns
            # appended), so this only re-attempts the final answer synthesis,
            # not the whole tool loop. tools=None explicitly — the retry can't
            # start another tool round, only synthesize a final plain answer.
            retry_cfg = LLMConfig(
                model=self._llm_config.model,
                temperature=self._llm_config.temperature,
                max_tokens=self._llm_config.max_tokens,
                response_format="text",
                tools=None,
            )
            retry_fired = True
            retried, retry_ms = await self._retry_if_unusable(
                result.finish_reason, messages, retry_cfg,
            )
            llm_ms_list.append(retry_ms)
            _retry_in, _retry_out = _usage_tokens(retried)
            in_tok += _retry_in
            out_tok += _retry_out
            if retried is not None:
                # Adopt whatever the retry produces unconditionally — including
                # legitimate plain (non-JSON) text, which parse_chatbot_response
                # already returns verbatim as a usable response_text despite
                # setting parse_error. If the retry is ALSO genuinely unusable,
                # this parses to the same canned fallback as before — no worse
                # off than not retrying at all.
                parsed = parse_chatbot_response(retried.text)
        if parsed.raw and not parsed.parse_error:
            response = parsed   # a real JSON envelope with a response_text
        else:
            # Parse failed or response_text missing — use the safe fallback the
            # parser already extracted (never expose raw LLM JSON to the customer).
            response = ChatBotResponse(response_text=parsed.response_text, language=self._language)
        if not response.language:
            response.language = self._language
        response.sources_used = list(dict.fromkeys([*sources, *(response.sources_used or [])]))
        if escalation:
            response.action = "escalate"
        # Only guard fully when the agent actually retrieved (search_knowledge_base
        # was called). A no-search turn (greeting, clarifying question, CRM-tool
        # answer) is legitimately ungrounded via RAG — the no-retrieval fallback
        # must not clobber it. But a turn with NEITHER retrieval NOR any tool call
        # at all gets the narrower no-grounding guard instead, since that's a turn
        # where nothing at all was consulted.
        guard_hallucination_fired = False
        guard_no_grounding_fired = False
        _guard_before = (response.response_text, response.confidence)
        if retrieved_all:
            response = apply_hallucination_guard(response, rag, self._guard)
            guard_hallucination_fired = (response.response_text, response.confidence) != _guard_before
        else:
            response = apply_no_grounding_guard(
                response, retrieved_any=bool(retrieved_all), tool_calls_made=tool_calls_made,
            )
            guard_no_grounding_fired = (response.response_text, response.confidence) != _guard_before
        # Step 5 (ticket #1762): deterministic, model-independent check — runs
        # unconditionally, regardless of which branch above fired. Grounded
        # text = successful CRM/deposit-verification tool results + the FULL
        # (untruncated) content of every KB chunk the model actually saw in a
        # search_knowledge_base tool result — NOT rag.text, which
        # build_rag_context separately truncates to max_context_chars (review
        # Fix 3: a real figure in a chunk truncated out of rag.text must still
        # count as grounded, since the model itself read the full chunk in
        # its own tool-result message) — + the customer's current query +
        # EVERY prior turn's content, both customer AND assistant (review Fix
        # 2: a number the bot already legitimately stated in an earlier turn,
        # e.g. "your balance is ₹2,075", must remain sayable in a later
        # no-tool-call turn that just restates it). customer_text stays
        # narrower (query + the customer's OWN prior turns only, no tool/RAG/
        # assistant content) — used only to pick which figure to name in the
        # fallback, so the guard can never end up "confirming" a number back
        # to the customer that came from the model's own unverified claim.
        # At this point session.turns does NOT yet include this turn's own
        # messages (that happens in _persist, right below), so this
        # naturally only sees prior turns, never a same-turn echo.
        prior_turns_text = "\n".join(
            m.content for m in self.session.turns if isinstance(m.content, str)
        )
        customer_turns_text = "\n".join(
            m.content for m in self.session.turns
            if m.role == "user" and isinstance(m.content, str)
        )
        kb_chunk_text = "\n".join(c.document.content for c in retrieved_all)
        grounded_text = "\n".join([
            *grounded_tool_texts, kb_chunk_text, query_text, prior_turns_text,
        ])
        _guard_before_unverified = (response.response_text, response.confidence)
        response = apply_unverified_data_guard(
            response,
            grounded_text=grounded_text,
            customer_text=f"{query_text}\n{customer_turns_text}",
            config=self._guard,
        )
        guard_unverified_data_fired = (
            (response.response_text, response.confidence) != _guard_before_unverified
        )
        await self._persist(user_msg, query_text, response, len(retrieved_all))
        total_ms = (time.perf_counter() - turn_start) * 1000
        # Computed OUTSIDE the ChatTurnMetrics try/except below, and used for
        # BOTH the dataclass and the log line — so the log's per-tool/
        # aggregate numbers are correct even in the (defensive, narrow-guard)
        # case where ChatTurnMetrics construction itself somehow fails; the
        # log must never fall back to a fabricated 0 for a real measurement.
        crm_metrics = [tm for tm in tool_metrics if tm.kind in ("crm", "deposit_verification")]
        kb_metrics = [tm for tm in tool_metrics if tm.kind == "kb"]
        tool_calls_count = len(crm_metrics)
        tool_failures_count = sum(
            1 for tm in crm_metrics if tm.outcome in ("timeout", "transport_error", "error")
        )
        tool_timeouts_count = sum(1 for tm in crm_metrics if tm.outcome == "timeout")
        tool_calls_skipped_count = sum(1 for tm in tool_metrics if tm.outcome == "skipped_budget")
        kb_search_ms_total = round(sum(tm.latency_ms for tm in kb_metrics))
        kb_searches_count = len(kb_metrics)
        llm_total_ms_val = round(sum(llm_ms_list))
        escalated_flag = response.action == "escalate"
        metrics: ChatTurnMetrics | None = None
        try:
            # Never let a metrics-assembly bug break a live turn — see
            # src/agents/voicebot.py's record_metric idiom and
            # src/models/turn_metrics.py's never-raises docstring for the
            # established house style this mirrors.
            metrics = ChatTurnMetrics(
                trace_id=current_trace_id(),
                path="tools",
                llm_provider=self._llm_provider,
                llm_model=self._llm_model,
                action=response.action,
                total_ms=round(total_ms),
                llm_total_ms=llm_total_ms_val,
                llm_calls=len(llm_ms_list),
                # Budgeted CRM total only (tool_elapsed_s * 1000) — directly
                # comparable to _TOOL_BUDGET_S. KB search is excluded (see
                # kb_search_ms below): it draws from its own independent
                # budget, so folding it in would make this incomparable to
                # the 45s CRM budget.
                tool_total_ms=round(tool_elapsed_s * 1000),
                tool_calls=tool_calls_count,
                tool_failures=tool_failures_count,
                tool_timeouts=tool_timeouts_count,
                tool_calls_skipped=tool_calls_skipped_count,
                kb_search_ms=kb_search_ms_total,
                kb_searches=kb_searches_count,
                retrieved_chunks=len(retrieved_all),
                rounds=rounds,
                rounds_exhausted=rounds_exhausted,
                retry_fired=retry_fired,
                failure_directive_fired=directive_fired_this_turn,
                failure_directive_escalated=directive_escalated_this_turn,
                guard_hallucination_fired=guard_hallucination_fired,
                guard_no_grounding_fired=guard_no_grounding_fired,
                guard_unverified_data_fired=guard_unverified_data_fired,
                escalated=escalated_flag,
                tools=tuple(tool_metrics),
            )
        except Exception:  # noqa: BLE001 - metrics assembly must never break a reply
            log.warning("chat turn metrics assembly failed; continuing without metrics",
                        exc_info=True,
                        extra={"ticket_id": self._ticket_id, "session_id": self._session_id})
        log.info(
            "chat turn done (tools path)",
            extra={
                "ticket_id": self._ticket_id, "session_id": self._session_id,
                "path": "tools",
                "total_ms": round(total_ms),
                "llm_ms_list": [round(ms) for ms in llm_ms_list],
                "llm_calls": len(llm_ms_list),
                "llm_total_ms": llm_total_ms_val,
                # Human-readable string, genuinely useful for reading one turn
                # by hand — but NOT the only representation: "tools" below
                # carries the full structured detail (kind/outcome/budget
                # slice/round), which is what makes per-tool diagnosis (e.g.
                # ticket #1762) a Loki field query instead of eyeballing a
                # string. outcome + budget_slice_ms specifically are, per the
                # plan, "the most diagnostic pair in the design."
                "tool_ms_list": [f"{name}:{ms:.0f}ms" for name, ms in tool_ms_list],
                "tools": [
                    {
                        "name": tm.tool_name, "kind": tm.kind, "ms": tm.latency_ms,
                        "outcome": tm.outcome, "slice_ms": tm.budget_slice_ms,
                        "round": tm.round_index,
                    }
                    for tm in tool_metrics
                ],
                "tool_total_ms": round(tool_elapsed_s * 1000),
                "tool_calls": tool_calls_count,
                "tool_failures": tool_failures_count,
                "tool_timeouts": tool_timeouts_count,
                "tool_calls_skipped": tool_calls_skipped_count,
                "kb_search_ms": kb_search_ms_total,
                "kb_searches": kb_searches_count,
                "rounds": rounds,
                "rounds_exhausted": rounds_exhausted,
                "retrieved_count": len(retrieved_all),
                "retry_fired": retry_fired,
                "failure_directive_fired": directive_fired_this_turn,
                "failure_directive_escalated": directive_escalated_this_turn,
                "guard_hallucination_fired": guard_hallucination_fired,
                "guard_no_grounding_fired": guard_no_grounding_fired,
                "guard_unverified_data_fired": guard_unverified_data_fired,
                "escalated": escalated_flag,
                "action": response.action,
            },
        )
        await self._emit_turn_metric(metrics)
        return ChatTurnResult(
            response=response, retrieved=retrieved_all, rag_context_chars=len(rag.text),
            escalation=escalation, call_offer=call_offer,
            input_tokens=in_tok, output_tokens=out_tok,
            llm_provider=self._llm_provider, llm_model=self._llm_model,
            metrics=metrics)

    async def _exec_tool(self, tc: ToolCall, timeout_s: float):
        """Bound a tool call's dispatch to its budget slice.

        ``timeout_s`` is this call's share of the turn's cumulative tool
        budget (see _TOOL_BUDGET_S in _handle_with_tools). Below
        _TOOL_MIN_SLICE_S the call isn't even attempted — the budget is
        already exhausted, so return the degraded result immediately rather
        than spend a connect round-trip on a call that would be cut short
        anyway. Otherwise dispatch is wrapped in asyncio.wait_for as a
        backstop: individual dispatch paths (e.g. the CRM HTTP call) are
        expected to respect timeout_s themselves, but this guarantees the
        slice is never exceeded regardless of tool type.
        """
        if timeout_s < _TOOL_MIN_SLICE_S:
            log.warning("tool budget exhausted; skipping call", extra={
                "ticket_id": self._ticket_id, "session_id": self._session_id, "tool": tc.name,
            })
            return dict(_TOOL_BUDGET_EXHAUSTED), [], None, None
        try:
            return await asyncio.wait_for(self._dispatch_tool(tc, timeout_s), timeout=timeout_s)
        except asyncio.TimeoutError:
            log.warning("tool call exceeded its slice", extra={
                "ticket_id": self._ticket_id, "session_id": self._session_id,
                "tool": tc.name, "slice_s": timeout_s,
            })
            return dict(_TOOL_BUDGET_EXHAUSTED), [], None, None

    async def _exec_kb_tool(self, tc: ToolCall):
        """Dispatch search_knowledge_base under its own fixed timeout
        (_KB_SEARCH_TIMEOUT_S), completely independent of the CRM tool-call
        budget (_TOOL_BUDGET_S/tool_elapsed_s in _handle_with_tools) — a
        slow/exhausted CRM budget must never cause a KB search to be skipped
        or cut short. Mirrors _exec_tool's wait_for backstop shape, minus the
        shared-budget slicing/min-slice-skip logic that doesn't apply here.
        """
        try:
            return await asyncio.wait_for(
                self._dispatch_tool(tc, _KB_SEARCH_TIMEOUT_S), timeout=_KB_SEARCH_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("kb search exceeded its timeout",
                        extra={"ticket_id": self._ticket_id, "session_id": self._session_id,
                               "tool": tc.name, "timeout_s": _KB_SEARCH_TIMEOUT_S})
            return dict(_TOOL_BUDGET_EXHAUSTED), [], None, None

    async def _dispatch_tool(self, tc: ToolCall, timeout_s: float):
        """Dispatch a tool call. Returns (result_dict, chunks, escalation, call_offer)."""
        args = tc.arguments or {}
        if tc.name == SEARCH_KB:
            try:
                chunks = await search_combined(args.get("query", ""), self._retrievers)
            except Exception:  # noqa: BLE001 — a search failure (e.g. embedder
                # unavailable) must not kill the turn; the model answers without RAG.
                log.exception("knowledge search failed", extra={
                    "ticket_id": self._ticket_id, "session_id": self._session_id,
                    "query": args.get("query", ""),
                })
                return {"error": "knowledge search is temporarily unavailable", "results": []}, [], None, None
            return (
                {"results": [{"content": c.document.content, "source": _chunk_source(c),
                              "score": c.score} for c in chunks]},
                chunks, None, None,
            )
        if tc.name == ESCALATE:
            esc = {"reason": args.get("reason", ""), "summary": args.get("summary", "")}
            return {"status": "escalated", **esc}, [], esc, None
        if tc.name == OFFER_CALL:
            off = {"reason": args.get("reason", "")}
            return {"status": "offered", **off}, [], None, off
        if tc.name == SUBMIT_DEPOSIT_VERIFICATION:
            if self._deposit_verification_executor is None:
                return {"error": "verification is not available"}, [], None, None
            try:
                out = await self._deposit_verification_executor(tc, timeout_s=max(0.5, timeout_s - 1.0))
                return (out if isinstance(out, dict) else {"result": out}), [], None, None
            except Exception:  # noqa: BLE001 — a failing submission must not kill the turn
                log.exception("deposit verification submission failed", extra={
                    "ticket_id": self._ticket_id, "session_id": self._session_id, "tool": tc.name,
                })
                return {
                    "status": "error",
                    "message": (
                        "Could not submit the verification request right now. Let the "
                        "customer know you're having trouble and offer to escalate to a "
                        "human agent."
                    ),
                }, [], None, None
        # Tenant CRM tool.
        if self._crm_executor is not None:
            try:
                # -1.0 margin: lets httpx's own read timeout normally fire
                # before the outer wait_for in _exec_tool does, so the
                # existing graceful {"error": ...} degradation path below
                # wins the race in the common case. wait_for is just the
                # backstop for cases where the read timeout doesn't equal
                # true wall-clock time (e.g. a slow connect).
                out = await self._crm_executor(tc, timeout_s=max(0.5, timeout_s - 1.0))
                return (out if isinstance(out, dict) else {"result": out}), [], None, None
            except Exception:  # noqa: BLE001 — a failing tool must not kill the turn
                log.exception("crm tool failed", extra={
                    "ticket_id": self._ticket_id, "session_id": self._session_id, "tool": tc.name,
                })
                return {
                    "status": "error",
                    "error": "the CRM call failed",
                    "failure": "transport_error",
                    "message": (
                        "You could not verify this — do NOT state any specific number, "
                        "amount, status, or date for it, and do not claim you checked or "
                        "verified it. Tell the customer honestly that you can't verify this "
                        "right now and offer to connect them to a human. That fully "
                        "satisfies the response-quality bar for this turn."
                    ),
                }, [], None, None
        return {
            "status": "error",
            "error": "no CRM integration is connected for this tool",
            "failure": "transport_error",
            "message": (
                "You could not verify this — do NOT state any specific number, amount, "
                "status, or date for it, and do not claim you checked or verified it. Tell "
                "the customer honestly that you can't verify this right now and offer to "
                "connect them to a human. That fully satisfies the response-quality bar for "
                "this turn."
            ),
        }, [], None, None

    # --- Shared helpers -------------------------------------------------

    async def _retry_if_unusable(
        self,
        finish_reason: str,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> tuple[LLMResult | None, float]:
        """Retry a generate() call once, bounded by ``_CHAT_RETRY_TIMEOUT_S``.

        Call this only after ``is_unusable_response()`` has confirmed the
        first attempt's parsed response_text was one of the parser's canned
        fallback lines — this helper always retries once when called, it
        doesn't re-check.

        If ``finish_reason`` indicates the first attempt got cut off by the
        token limit ("length"), bump max_tokens for the retry (capped) so it
        isn't just truncated again the same way; otherwise reuse ``config``
        as-is.

        Returns ``(retried_result_or_None, elapsed_ms)``. ``elapsed_ms`` is
        the retry's measured duration even when it failed or timed out — so a
        failed retry still shows up in telemetry instead of silently
        vanishing. Never raises: a failing/timing-out retry just yields
        ``(None, elapsed_ms)`` and the caller falls back to the original
        (pre-retry) response, no worse off than not retrying at all.
        """
        log.warning(
            "chatbot retrying turn: no usable response (finish_reason=%s)", finish_reason,
            extra={"ticket_id": self._ticket_id, "session_id": self._session_id},
        )
        retry_start = time.perf_counter()
        try:
            retry_config = config
            if finish_reason == "length":
                # Truncation is the likely cause — retrying with the same
                # max_tokens would probably just get cut off again the same
                # way. config.max_tokens can be None (provider default), so
                # this must stay inside the try — the whole point of this
                # helper is to never raise regardless of config shape.
                bumped = min(int((config.max_tokens or 1024) * 1.5), _CHAT_RETRY_MAX_TOKENS_CAP)
                retry_config = _replace_cfg(config, max_tokens=bumped)
            result = await asyncio.wait_for(
                self._llm.generate(messages, retry_config), timeout=_CHAT_RETRY_TIMEOUT_S,
            )
            return result, (time.perf_counter() - retry_start) * 1000
        except Exception:  # noqa: BLE001 - incl. asyncio.TimeoutError; the retry
            # itself (and its own bounded timeout) must not crash the turn.
            log.exception("chatbot retry after empty/unparseable LLM response failed",
                          extra={"ticket_id": self._ticket_id, "session_id": self._session_id})
            return None, (time.perf_counter() - retry_start) * 1000

    def _compose(
        self, rag_text: str, user_msg: LLMMessage, query_text: str = "",
    ) -> list[LLMMessage]:
        # Per-turn language directive. History of this logic (three real bugs):
        # 1. All-Latin text was labeled "English" → romanized Hindi got forced
        #    into English replies.
        # 2. Then Latin text got NO signal → the "Default language: hi"
        #    fallback answered plain English in Devanagari.
        # 3. Then an advisory "pick English or Hinglish yourself" directive →
        #    Hinglish history momentum kept answering plain English in
        #    Hinglish. Hence the deterministic marker-based classification:
        #    the directive must NAME the language, firmly, each turn.
        lang = _detect_script(query_text)
        if lang is None:
            lang = _latin_language_hint(query_text)
        if lang == "Hinglish":
            extra = [
                "The user's current message is romanized Hindi (Hinglish). Reply in Roman-"
                "script Hinglish — NEVER Devanagari, regardless of the conversation's "
                "default language or the language of earlier turns."
            ]
        elif lang:
            extra = [
                f"The user's current message is in {lang}. Your response_text MUST be in "
                f"{lang} — regardless of the conversation's default language or the "
                "language of earlier turns."
            ]
        elif not any(m.role == "user" for m in self.session.turns):
            # Opening message, no language signal at all (e.g. a bare "games").
            # There's no established conversation language yet to fall back on,
            # and the configured default (often Devanagari Hindi) risks
            # alienating an English-only user on their very first message.
            # Roman Hinglish is readable by both English and Hindi/Hinglish
            # speakers, so it's the safer opener; a later message with a real
            # signal switches language deterministically from there, same as
            # any other turn. Scoped to the opening message ONLY — a
            # mid-conversation short ack ("ok") must keep following the
            # established conversation's language (see bug #3 above), so this
            # branch must never fire once a user turn already exists.
            extra = [
                "This is the very first message of the conversation and it carries no "
                "clear language signal (e.g. a bare word like 'games'). Reply in "
                "Roman-script Hinglish for this opening turn — readable by English and "
                "Hindi/Hinglish speakers alike — rather than the configured default "
                "language."
            ]
        else:
            extra = None  # no signal mid-conversation — follow the conversation
        system_prompt = build_chatbot_system_prompt(
            company_name=self._company,
            language_default=self._language,
            rag_context=rag_text,
            extra_directives=extra,
            has_player_tools=any(t.name in PLAYER_TOOLS for t in self._crm_tools),
            has_operator_tools=any(t.name in OPERATOR_TOOLS for t in self._crm_tools),
            has_deposit_verification_tool=any(t.name == SUBMIT_DEPOSIT_VERIFICATION for t in self._crm_tools),
            tenant_timezone=self._tenant_timezone,
            prompt_pack=self._prompt_pack,
        )
        messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]
        # Replay the last MAX_HISTORY_TURNS exchanges (system is rebuilt each
        # turn); session.turns itself is kept full for history/UI purposes.
        for m in self.session.turns[-(2 * MAX_HISTORY_TURNS):]:
            if m.role in ("user", "assistant"):
                messages.append(m)
        messages.append(user_msg)
        return messages

    async def _persist(
        self, user_msg: LLMMessage, query_text: str, response: ChatBotResponse, retrieved_count: int,
    ) -> None:
        user_repr = query_text if query_text.strip() else "[media]"
        self.session.turns.append(user_msg)
        self.session.turns.append(LLMMessage(role="assistant", content=response.response_text))
        await self.persist_turn("user", user_repr)
        await self.persist_turn(
            "agent",
            response.response_text,
            metadata={
                "confidence": response.confidence,
                "sources_used": response.sources_used,
                "action": response.action,
                "retrieved_count": retrieved_count,
            },
        )
        if self.store is not None:
            await self.store.set_state(
                self.session.session_id,
                {
                    "agent_type": "chatbot",
                    "last_action": response.action,
                    "last_confidence": response.confidence,
                    "turn_count": sum(1 for m in self.session.turns if m.role == "user"),
                },
            )

    async def summarize_session(self) -> str:
        """One-line LLM summary of the conversation (for session end / handoff).
        Returns "" when there's nothing to summarize or the LLM fails."""
        lines = [
            f"{'Customer' if m.role == 'user' else 'Agent'}: {m.content}"
            for m in self.session.turns
            if m.role in ("user", "assistant") and isinstance(m.content, str) and m.content
        ]
        if not lines:
            return ""
        messages = [
            LLMMessage(role="system",
                       content="Summarize this customer-support chat in one concise sentence "
                               "(the issue + outcome). Reply with only the sentence."),
            LLMMessage(role="user", content="\n".join(lines)),
        ]
        try:
            result = await self._llm.generate(
                messages, LLMConfig(response_format="text", temperature=0.3, max_tokens=120))
            return (result.text or "").strip()
        except Exception:  # noqa: BLE001 — summary is best-effort
            log.exception("chat session summarize failed",
                          extra={"ticket_id": self._ticket_id, "session_id": self._session_id})
            return ""

    async def get_history(self) -> list[dict[str, Any]]:
        if self.store is None:
            return [
                {"role": m.role, "content": m.content}
                for m in self.session.turns
                if m.role in ("user", "assistant")
            ]
        return await self.store.get_history(self.session.session_id)

    # ChatBot doesn't drive a state machine; override BaseAgent's persistence.
    async def persist_state(self, extra: dict | None = None) -> None:  # type: ignore[override]
        if self.store is None:
            return
        payload = {"agent_type": "chatbot"}
        if extra:
            payload.update(extra)
        await self.store.set_state(self.session.session_id, payload)
