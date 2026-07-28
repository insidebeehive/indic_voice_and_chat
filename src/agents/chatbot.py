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

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from src.agents.base import AgentSession, BaseAgent
from src.chatbot.media import prepare_multimodal_content
from src.chatbot.tools import BUILTIN_TOOLS, ESCALATE, OFFER_CALL, SEARCH_KB
from src.dialogue.context import SessionStore
from src.dialogue.prompts import build_chatbot_system_prompt
from src.dialogue.response_parser import ChatBotResponse, parse_chatbot_response
from src.dialogue.slots import SlotFiller, SlotSchema
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, ToolCall, ToolSpec
from src.rag.context_builder import (
    GuardConfig,
    apply_hallucination_guard,
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

# Executes a tenant-registered (CRM) tool call → a JSON-able result dict.
CrmExecutor = Callable[[ToolCall], Awaitable[dict]]

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


def _detect_script(text: str) -> Optional[str]:
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
    indic_lang: Optional[str] = None
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


def _latin_language_hint(text: str) -> Optional[str]:
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


@dataclass
class ChatTurnResult:
    response: ChatBotResponse
    retrieved: list[RetrievedChunk]
    rag_context_chars: int
    escalation: Optional[dict] = None
    call_offer: Optional[dict] = None


class ChatBotAgent(BaseAgent):
    def __init__(
        self,
        session: AgentSession,
        llm: ILLMProvider,
        retriever: HybridRetriever,
        crm_retriever: Optional[HybridRetriever] = None,
        llm_config: Optional[LLMConfig] = None,
        company_name: str = "[Your Company]",
        language_default: str = "en",
        store: Optional[SessionStore] = None,
        guard_config: Optional[GuardConfig] = None,
        max_context_chars: int = 2000,
        enable_tools: bool = False,
        crm_tools: Optional[list[ToolSpec]] = None,
        crm_executor: Optional[CrmExecutor] = None,
        max_tool_rounds: int = 2,
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
        self._guard = guard_config
        self._max_context_chars = max_context_chars
        # Agentic tool-calling (opt-in): builtin tools (search KB / escalate /
        # offer call) + any tenant CRM tools. Off by default so the single-shot
        # RAG path is unchanged.
        self._enable_tools = enable_tools
        self._crm_tools = crm_tools or []
        self._crm_executor = crm_executor
        self._max_tool_rounds = max_tool_rounds

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
        llm_ms = (time.perf_counter() - llm_start) * 1000
        response = parse_chatbot_response(result.text)
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
            time.perf_counter() - turn_start, [f"{llm_ms:.0f}ms"], retrieval_ms, len(retrieved),
        )
        return ChatTurnResult(
            response=response, retrieved=retrieved, rag_context_chars=len(rag.text))

    # --- Tool-calling path (agentic) -----------------------------------

    async def _handle_with_tools(self, user_msg: LLMMessage, query_text: str) -> ChatTurnResult:
        turn_start = time.perf_counter()
        llm_ms_list: list[float] = []
        tool_ms_list: list[tuple[str, float]] = []
        rounds = 0
        tools = list(BUILTIN_TOOLS) + list(self._crm_tools)
        # Tools fetch their own context (search_knowledge_base), so the system
        # prompt starts without a pre-built RAG block.
        messages = self._compose("", user_msg, query_text=query_text)
        retrieved_all: list[RetrievedChunk] = []
        escalation: Optional[dict] = None
        call_offer: Optional[dict] = None
        text = ""
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
            log.debug(
                "chatbot llm turn: finish=%s usage=%s text_len=%d tool_calls=%d",
                result.finish_reason, result.usage, len(result.text or ""), len(result.tool_calls),
            )
            if not result.tool_calls:
                text = result.text
                break
            messages.append(LLMMessage(role="assistant", content="", tool_calls=result.tool_calls))
            for tc in result.tool_calls:
                tool_start = time.perf_counter()
                out, chunks, esc, off = await self._exec_tool(tc)
                tool_ms_list.append((tc.name, (time.perf_counter() - tool_start) * 1000))
                retrieved_all.extend(chunks)
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
        # Only guard when the agent actually retrieved (search_knowledge_base was
        # called). A no-search turn (greeting, clarifying question, CRM-tool
        # answer) is legitimately ungrounded — the no-retrieval fallback must not
        # clobber it.
        if retrieved_all:
            response = apply_hallucination_guard(response, rag, self._guard)
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
            escalation=escalation, call_offer=call_offer)

    async def _exec_tool(self, tc: ToolCall):
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
        # Tenant CRM tool.
        if self._crm_executor is not None:
            try:
                out = await self._crm_executor(tc)
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
            has_player_tools=bool(self._crm_tools),
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
    async def persist_state(self, extra: Optional[dict] = None) -> None:  # type: ignore[override]
        if self.store is None:
            return
        payload = {"agent_type": "chatbot"}
        if extra:
            payload.update(extra)
        await self.store.set_state(self.session.session_id, payload)
