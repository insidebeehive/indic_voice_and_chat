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
    build_rag_context,
    search_combined,
)
from src.rag.retriever import HybridRetriever, RetrievedChunk

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

# search_knowledge_base gets its OWN fixed timeout, entirely independent of
# _TOOL_BUDGET_S/_TOOL_CALL_CEILING_S. It is a RAG/embedding lookup, not a
# tenant CRM call — it never draws from, and is never shortened by, the CRM
# tool-call budget above. Previously KB search shared that budget: a slow CRM
# call could exhaust tool_elapsed_s, leaving a same-turn KB search skipped or
# cut to near-zero, which meant retrieved_all stayed empty and the
# hallucination guard (apply_hallucination_guard, gated on `if retrieved_all`
# below) silently never fired — ungrounded, unguarded answers with no signal
# anything went wrong. 15.0s is a generous fixed ceiling for an embedding
# search/rerank (RAG quality matters and this call type wasn't the source of
# the incident this budget was added for) while still bounding overall turn
# latency rather than leaving it truly unbounded.
_KB_SEARCH_TIMEOUT_S = 15.0

# Shared "the budget ran out" tool result. Fed back to the model as the tool's
# result content so the turn still produces a real (if degraded) answer
# instead of hanging or erroring out. Always return a fresh dict() copy of
# this (never the module-level object itself) — the result is JSON-serialized
# per-call and callers/tests may hold onto it.
_TOOL_BUDGET_EXHAUSTED = {
    "error": "timed out",
    "message": ("This data source did not respond in time. Answer now using what you "
                "already have, briefly acknowledge you could not fetch that detail, and "
                "do not ask the customer for any account details."),
}

# Executes a tenant-registered (CRM) tool call → a JSON-able result dict.
# ``timeout_s`` is this call's share of the turn's cumulative tool budget
# (see _TOOL_BUDGET_S) — keyword-only so a positional-arg executor can't
# silently swallow it as some other parameter.
CrmExecutor = Callable[[ToolCall, float], Awaitable[dict]]

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
        if retrieved or not multimodal:
            response = apply_hallucination_guard(response, rag, self._guard)
        # 6. Persist
        await self._persist(user_msg, query_text, response, len(retrieved))
        log.info(
            "chat turn done in %.2fs: llm=%s retrieval=%.0fms retrieved=%d",
            time.perf_counter() - turn_start,
            [f"{ms:.0f}ms" for ms in llm_ms_list], retrieval_ms, len(retrieved),
        )
        return ChatTurnResult(
            response=response, retrieved=retrieved, rag_context_chars=len(rag.text),
            input_tokens=in_tok, output_tokens=out_tok,
            llm_provider=self._llm_provider, llm_model=self._llm_model)

    # --- Tool-calling path (agentic) -----------------------------------

    async def _handle_with_tools(self, user_msg: LLMMessage, query_text: str) -> ChatTurnResult:
        turn_start = time.perf_counter()
        llm_ms_list: list[float] = []
        tool_ms_list: list[tuple[str, float]] = []
        rounds = 0
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
            )
            if not result.tool_calls:
                text = result.text
                break
            messages.append(LLMMessage(role="assistant", content="", tool_calls=result.tool_calls))
            for i, tc in enumerate(result.tool_calls):
                tool_start = time.perf_counter()
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
                tool_ms_list.append((tc.name, elapsed * 1000))
                retrieved_all.extend(chunks)
                if tc.name not in (ESCALATE, OFFER_CALL):
                    tool_calls_made.append(tc.name)
                escalation = esc or escalation
                call_offer = off or call_offer
                messages.append(LLMMessage(
                    role="tool", name=tc.name, tool_call_id=tc.id, content=json.dumps(out)))
        else:
            # Ran out of rounds still wanting tools — force a final plain answer.
            llm_start = time.perf_counter()
            result = await self._llm.generate(
                messages, LLMConfig(temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                                    response_format="text"))
            llm_ms_list.append((time.perf_counter() - llm_start) * 1000)
            _forced_in, _forced_out = _usage_tokens(result)
            in_tok += _forced_in
            out_tok += _forced_out
            text = result.text

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
        if retrieved_all:
            response = apply_hallucination_guard(response, rag, self._guard)
        else:
            response = apply_no_grounding_guard(
                response, retrieved_any=bool(retrieved_all), tool_calls_made=tool_calls_made,
            )
        await self._persist(user_msg, query_text, response, len(retrieved_all))
        log.info(
            "chat turn done in %.2fs: llm=%s tools=%s rounds=%d retrieved=%d",
            time.perf_counter() - turn_start,
            [f"{ms:.0f}ms" for ms in llm_ms_list],
            [f"{name}:{ms:.0f}ms" for name, ms in tool_ms_list],
            rounds, len(retrieved_all),
        )
        return ChatTurnResult(
            response=response, retrieved=retrieved_all, rag_context_chars=len(rag.text),
            escalation=escalation, call_offer=call_offer,
            input_tokens=in_tok, output_tokens=out_tok,
            llm_provider=self._llm_provider, llm_model=self._llm_model)

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
            log.warning("tool budget exhausted; skipping call", extra={"tool": tc.name})
            return dict(_TOOL_BUDGET_EXHAUSTED), [], None, None
        try:
            return await asyncio.wait_for(self._dispatch_tool(tc, timeout_s), timeout=timeout_s)
        except asyncio.TimeoutError:
            log.warning("tool call exceeded its slice", extra={"tool": tc.name, "slice_s": timeout_s})
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
                        extra={"tool": tc.name, "timeout_s": _KB_SEARCH_TIMEOUT_S})
            return dict(_TOOL_BUDGET_EXHAUSTED), [], None, None

    async def _dispatch_tool(self, tc: ToolCall, timeout_s: float):
        """Dispatch a tool call. Returns (result_dict, chunks, escalation, call_offer)."""
        args = tc.arguments or {}
        if tc.name == SEARCH_KB:
            try:
                chunks = await search_combined(args.get("query", ""), self._retrievers)
            except Exception:  # noqa: BLE001 — a search failure (e.g. embedder
                # unavailable) must not kill the turn; the model answers without RAG.
                log.exception("knowledge search failed", extra={"query": args.get("query", "")})
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
                log.exception("deposit verification submission failed", extra={"tool": tc.name})
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
                log.exception("crm tool failed", extra={"tool": tc.name})
                return {
                    "status": "pending",
                    "message": (
                        "This is a player-specific query. The data is being retrieved — "
                        "give the customer a brief holding message and do not ask them for "
                        "any account details."
                    ),
                }, [], None, None
        return {
            "status": "pending",
            "message": (
                "This is a player-specific query. The integration is not yet connected — "
                "give the customer a brief holding message and do not ask them for any "
                "account details."
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
            "chatbot retrying turn: no usable response (finish_reason=%s)", finish_reason
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
            log.exception("chatbot retry after empty/unparseable LLM response failed")
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
            log.exception("chat session summarize failed")
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
