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

# Executes a tenant-registered (CRM) tool call → a JSON-able result dict.
CrmExecutor = Callable[[ToolCall], Awaitable[dict]]


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
        platform_retriever: Optional[HybridRetriever] = None,
        llm_config: Optional[LLMConfig] = None,
        company_name: str = "[Your Company]",
        language_default: str = "en",
        store: Optional[SessionStore] = None,
        guard_config: Optional[GuardConfig] = None,
        max_context_chars: int = 4000,
        enable_tools: bool = False,
        crm_tools: Optional[list[ToolSpec]] = None,
        crm_executor: Optional[CrmExecutor] = None,
        max_tool_rounds: int = 4,
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
        self._platform_retriever = platform_retriever
        # Ordered list: platform KB first (broadest), then tenant-specific.
        self._retrievers: list[HybridRetriever] = [
            r for r in [platform_retriever, retriever] if r is not None
        ]
        self._llm_config = llm_config or LLMConfig(response_format="json")
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
        # 1. Retrieval (on the text part; multimodal-only turns skip it)
        retrieved = await search_combined(query_text, self._retrievers) if query_text.strip() else []
        # 2. Build context
        rag = build_rag_context(retrieved, max_chars=self._max_context_chars)
        # 3. Compose messages
        messages = self._compose(rag.text, user_msg)
        # 4. LLM
        result = await self._llm.generate(messages, self._llm_config)
        response = parse_chatbot_response(result.text)
        # 5. Guard. Skip only for a multimodal turn with no retrieval — that
        # answer is grounded in the image, not the (empty) knowledge base, so the
        # no-retrieval fallback would wrongly clobber it. Text turns are unchanged.
        multimodal = isinstance(user_msg.content, list)
        if retrieved or not multimodal:
            response = apply_hallucination_guard(response, rag, self._guard)
        # 6. Persist
        await self._persist(user_msg, query_text, response, len(retrieved))
        return ChatTurnResult(
            response=response, retrieved=retrieved, rag_context_chars=len(rag.text))

    # --- Tool-calling path (agentic) -----------------------------------

    async def _handle_with_tools(self, user_msg: LLMMessage, query_text: str) -> ChatTurnResult:
        tools = list(BUILTIN_TOOLS) + list(self._crm_tools)
        # Tools fetch their own context (search_knowledge_base), so the system
        # prompt starts without a pre-built RAG block.
        messages = self._compose("", user_msg)
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
            result = await self._llm.generate(messages, cfg)
            if not result.tool_calls:
                text = result.text
                break
            messages.append(LLMMessage(role="assistant", content="", tool_calls=result.tool_calls))
            for tc in result.tool_calls:
                out, chunks, esc, off = await self._exec_tool(tc)
                retrieved_all.extend(chunks)
                escalation = esc or escalation
                call_offer = off or call_offer
                messages.append(LLMMessage(
                    role="tool", name=tc.name, tool_call_id=tc.id, content=json.dumps(out)))
        else:
            # Ran out of rounds still wanting tools — force a final plain answer.
            result = await self._llm.generate(
                messages, LLMConfig(temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                                    response_format="text"))
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
            response = ChatBotResponse(response_text=(text or "").strip(), language=self._language)
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
                return {"error": f"tool {tc.name} failed"}, [], None, None
        return {"error": f"unknown tool {tc.name}"}, [], None, None

    # --- Shared helpers -------------------------------------------------

    def _compose(self, rag_text: str, user_msg: LLMMessage) -> list[LLMMessage]:
        system_prompt = build_chatbot_system_prompt(
            company_name=self._company,
            language_default=self._language,
            rag_context=rag_text,
        )
        messages: list[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]
        # Replay prior user/assistant turns (system is rebuilt each turn).
        for m in self.session.turns:
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
