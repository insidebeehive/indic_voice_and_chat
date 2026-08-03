"""ChatBotAgent agentic tool loop + multimodal input (Phase 3a)."""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.chatbot.media import prepare_multimodal_content
from src.interfaces.llm import (
    ContentPart,
    ILLMProvider,
    LLMMessage,
    LLMResult,
    ToolCall,
    ToolSpec,
)
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import HybridRetriever, RetrievalConfig


class ScriptedLLM(ILLMProvider):
    """Returns the next pre-canned LLMResult per call; records calls."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[list[LLMMessage], object]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.calls.append((list(messages), config))
        return self._results.pop(0) if self._results else LLMResult(text="ok", finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


@pytest.fixture
async def retriever(tmp_faiss_index: str) -> HybridRetriever:
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    r = HybridRetriever(
        embedder=HashEmbedder(dim=64), vector_store=store, reranker=IdentityReranker(),
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8,
                               reranking=True, similarity_threshold=0.0))
    await r.index([
        Document(id="c1", content="Plan B has 500GB unlimited data.",
                 metadata={"filename": "plans.pdf", "page": 2}),
        Document(id="c2", content="Plan A is 100GB for Rs 199.",
                 metadata={"filename": "plans.pdf", "page": 1}),
        Document(id="c3", content="Biryani recipe.", metadata={"filename": "cookbook.md"}),
    ])
    return r


def _agent(llm, retriever, **kw) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-1"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en", enable_tools=True, **kw)


# --- Tool loop ----------------------------------------------------------


@pytest.mark.asyncio
async def test_search_tool_is_called_then_answer(retriever) -> None:
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "Plan B"})]),
        LLMResult(text="Plan B has 500GB unlimited data.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever)
    result = await agent.handle_message("Tell me about Plan B")

    assert result.response.response_text == "Plan B has 500GB unlimited data."
    # Sources came from the search tool results, not the LLM.
    assert "plans.pdf:2" in result.response.sources_used
    # The first generate was sent tools; the second saw a tool-result message.
    assert llm.calls[0][1].tools is not None
    second = llm.calls[1][0]
    assert any(m.role == "tool" and m.name == "search_knowledge_base" for m in second)


@pytest.mark.asyncio
async def test_tool_path_parses_json_envelope_final_reply(retriever) -> None:
    # The model emits the structured JSON envelope (per the system prompt) even in
    # the tool loop — the agent must PARSE it, not return raw JSON to the customer.
    envelope = ('```json\n{"response_text": "Plan B has 500GB.", "language": "hi",'
                ' "confidence": "high", "sources_used": [], "action": "none"}\n```')
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "Plan B"})]),
        LLMResult(text=envelope, finish_reason="stop"),
    ])
    agent = _agent(llm, retriever)
    result = await agent.handle_message("Plan B?")
    assert result.response.response_text == "Plan B has 500GB."   # parsed, not raw JSON
    assert "```json" not in result.response.response_text
    assert result.response.language == "hi"


@pytest.mark.asyncio
async def test_escalate_tool_sets_action_and_escalation(retriever) -> None:
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="escalate_to_human",
                     arguments={"reason": "angry", "summary": "refund dispute"})]),
        LLMResult(text="Connecting you to a human agent.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever)
    result = await agent.handle_message("I want a human NOW")
    assert result.response.action == "escalate"
    assert result.escalation == {"reason": "angry", "summary": "refund dispute"}


@pytest.mark.asyncio
async def test_offer_call_tool_sets_call_offer(retriever) -> None:
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="offer_voice_call", arguments={"reason": "complex setup"})]),
        LLMResult(text="Would a quick call help?", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever)
    result = await agent.handle_message("this is confusing")
    assert result.call_offer == {"reason": "complex setup"}


@pytest.mark.asyncio
async def test_crm_tool_executed_via_injected_executor(retriever) -> None:
    seen = {}

    async def crm_exec(tc: ToolCall) -> dict:
        seen["call"] = (tc.name, tc.arguments)
        return {"status": "dispatched", "eta": "2 days"}

    crm_tools = [ToolSpec(name="check_order_status", description="check order",
                          parameters={"type": "object", "properties": {"order_id": {"type": "string"}},
                                      "required": ["order_id"]})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="check_order_status", arguments={"order_id": "ORD-9"})]),
        LLMResult(text="Your order ORD-9 ships in 2 days.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await agent.handle_message("where is ORD-9?")
    assert seen["call"] == ("check_order_status", {"order_id": "ORD-9"})
    assert result.response.response_text == "Your order ORD-9 ships in 2 days."
    # The CRM tool spec was offered to the LLM alongside the builtins.
    offered = {t.name for t in llm.calls[0][1].tools}
    assert "check_order_status" in offered and "search_knowledge_base" in offered


@pytest.mark.asyncio
async def test_search_failure_still_replies(retriever) -> None:
    # If the KB search throws (e.g. embedder unavailable), the turn must NOT crash —
    # the error is fed back as a tool result and the model still answers.
    class _BoomRetriever:
        async def search(self, *a, **k):
            raise RuntimeError("embedder unavailable")

    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "plans"})]),
        LLMResult(text="I couldn't check the docs just now, but I can still help.",
                  finish_reason="stop"),
    ])
    agent = ChatBotAgent(
        session=AgentSession(session_id="s"), llm=llm, retriever=_BoomRetriever(),
        company_name="Acme", language_default="en", enable_tools=True)
    result = await agent.handle_message("what plans do you have?")
    assert result.response.response_text == "I couldn't check the docs just now, but I can still help."
    # The error was fed back to the model as the tool result.
    tool_msgs = [m for m in llm.calls[1][0] if m.role == "tool"]
    assert tool_msgs and "error" in json.loads(tool_msgs[0].content)


@pytest.mark.asyncio
async def test_summarize_session(retriever) -> None:
    llm = ScriptedLLM([LLMResult(
        text="Customer asked about Plan B; provided details.", finish_reason="stop")])
    agent = ChatBotAgent(
        session=AgentSession(session_id="s", turns=[
            LLMMessage(role="user", content="tell me about Plan B"),
            LLMMessage(role="assistant", content="Plan B has 500GB."),
        ]),
        llm=llm, retriever=retriever, company_name="Acme", language_default="en")
    summary = await agent.summarize_session()
    assert summary == "Customer asked about Plan B; provided details."
    # The conversation was passed to the summarizer.
    assert "Plan B" in llm.calls[0][0][-1].content


@pytest.mark.asyncio
async def test_summarize_empty_session_returns_blank(retriever) -> None:
    llm = ScriptedLLM([])
    agent = ChatBotAgent(session=AgentSession(session_id="s"), llm=llm, retriever=retriever,
                         company_name="Acme", language_default="en")
    assert await agent.summarize_session() == ""
    assert llm.calls == []


@pytest.mark.asyncio
async def test_no_tool_call_answers_directly(retriever) -> None:
    llm = ScriptedLLM([LLMResult(text="Hello! How can I help?", finish_reason="stop")])
    agent = _agent(llm, retriever)
    result = await agent.handle_message("hi")
    assert result.response.response_text == "Hello! How can I help?"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_tool_loop_retries_once_when_final_answer_is_empty(retriever) -> None:
    """The tool round(s) can complete fine but the final answer-synthesis call
    still comes back empty (e.g. a safety-filter block) — retry that final
    call once (reusing the completed tool-call history in messages) rather
    than showing the customer the canned fallback line immediately."""
    async def crm_exec(tc: ToolCall) -> dict:
        return {"status": "settled", "result": "won"}

    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_matka_bids", arguments={"status": "settled"})]),
        LLMResult(text="", finish_reason="stop"),  # final synthesis: empty
        LLMResult(text="Your last Matka bid was settled as a win.", finish_reason="stop"),
    ])
    crm_tools = [ToolSpec(name="get_matka_bids", description="Get Matka bids",
                          parameters={"type": "object", "properties": {}})]
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await agent.handle_message("meri jeet credit kyu nahi hui?")
    assert result.response.response_text == "Your last Matka bid was settled as a win."
    assert result.response.parse_error is None
    assert len(llm.calls) == 3  # tool round + empty final + one retry
    # The retry reused the completed tool-call history, not a fresh tool round.
    assert any(m.role == "tool" and m.name == "get_matka_bids" for m in llm.calls[2][0])
    # The retry's LLMConfig has no tools — it cannot start another tool round,
    # only synthesize a final plain-text answer from the completed history.
    assert llm.calls[2][1].tools is None


# --- Multimodal prep ----------------------------------------------------


def test_prepare_image_content_passthrough() -> None:
    parts = prepare_multimodal_content("what is this?", b"\xff\xd8imgbytes", "image/jpeg")
    assert parts[0] == ContentPart(type="text", text="what is this?")
    assert parts[1].type == "image"
    assert parts[1].inline_data == {"mime_type": "image/jpeg", "data": b"\xff\xd8imgbytes"}


def test_prepare_image_content_base64_decoded() -> None:
    import base64
    b64 = base64.b64encode(b"hello").decode()
    parts = prepare_multimodal_content("", b64, "image/png")
    assert parts[0].type == "image"
    assert parts[0].inline_data["data"] == b"hello"


def test_prepare_video_without_decoder_degrades_to_text() -> None:
    # No PyAV in the test env → a text note, never a crash.
    parts = prepare_multimodal_content("see this", b"fakevideo", "video/mp4")
    assert parts[0].text == "see this"
    assert any("video" in (p.text or "").lower() for p in parts)


@pytest.mark.asyncio
async def test_handle_image_sends_multimodal_to_llm(retriever) -> None:
    # No caption → no retrieval → the multimodal image reaches the LLM directly.
    llm = ScriptedLLM([LLMResult(
        text=json.dumps({"response_text": "That's an error screen.", "language": "en",
                         "confidence": "high", "action": "none"}),
        finish_reason="stop")])
    # tools OFF → single-shot multimodal path
    agent = ChatBotAgent(session=AgentSession(session_id="cb-img"), llm=llm, retriever=retriever,
                         company_name="Acme", language_default="en")
    result = await agent.handle_image(b"imgbytes", "image/jpeg", text="")
    # The multimodal user message (with an image part) reached the LLM.
    user_msg = llm.calls[0][0][-1]
    assert isinstance(user_msg.content, list)
    assert any(p.type == "image" and p.inline_data["data"] == b"imgbytes" for p in user_msg.content)
    assert result.response.response_text == "That's an error screen."
