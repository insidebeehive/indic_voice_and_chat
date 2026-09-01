"""ChatBotAgent agentic tool loop + multimodal input (Phase 3a)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

import pytest

from src.agents import chatbot as chatbot_mod
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
async def test_tool_loop_sums_usage_across_rounds(retriever) -> None:
    """A turn with a tool round + a final answer round must sum usage from
    BOTH generate() calls — a chat turn is not one LLM call."""
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", usage={
            "prompt_tokens": 200, "completion_tokens": 10}, tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "Plan B"})]),
        LLMResult(text="Plan B has 500GB unlimited data.", finish_reason="stop",
                  usage={"prompt_tokens": 250, "completion_tokens": 20}),
    ])
    agent = _agent(llm, retriever, llm_provider="gemini", llm_model="gemini-3.5-flash")
    result = await agent.handle_message("Tell me about Plan B")
    assert result.input_tokens == 450    # 200 + 250
    assert result.output_tokens == 30    # 10 + 20
    assert result.llm_provider == "gemini"
    assert result.llm_model == "gemini-3.5-flash"


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

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
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
    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
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


# --- Per-turn cumulative tool budget (Fix 1) -----------------------------
#
# `_exec_tool`/`_dispatch_tool` enforce a per-call `timeout_s` slice that the
# `_handle_with_tools` loop fair-shares out of a per-TURN cumulative budget
# (`_TOOL_BUDGET_S`), not a fresh per-call timeout. The CRM executor receives
# `timeout_s - 1.0` (floored at 0.5s) as its own margin (see `_dispatch_tool`),
# so tests that want to observe the fair-share VALUE precisely use large
# patched constants with an instantly-returning stub (no real sleep is
# incurred regardless of the constants' size, since the stub never blocks);
# tests that want to prove REAL cumulative wall-clock enforcement use small
# patched constants with a stub that sleeps far longer than any slice, cut
# short by the surrounding `asyncio.wait_for`.


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_round_share_the_budget(retriever, monkeypatch) -> None:
    """3 tool calls in ONE LLM response must fair-share the cumulative
    per-turn tool budget rather than each getting its own fresh timeout --
    every dispatched call is cut short by the per-call wait_for at its slice,
    the cumulative real time stays bounded near the shared budget (well under
    3 independent per-call ceilings), and the turn still finishes with the
    scripted final answer."""
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 0.3)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 0.2)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.01)

    seen: list[float] = []

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        seen.append(timeout_s)
        await asyncio.sleep(10)  # always cut short by the per-call wait_for
        return {"result": "never reached"}  # pragma: no cover

    crm_tools = [ToolSpec(name="get_balance", description="get balance",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_balance", arguments={}),
            ToolCall(id="t2", name="get_balance", arguments={}),
            ToolCall(id="t3", name="get_balance", arguments={}),
        ]),
        LLMResult(text="Here is what I have so far.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    start = time.perf_counter()
    result = await asyncio.wait_for(agent.handle_message("balance please"), timeout=5.0)
    elapsed = time.perf_counter() - start

    assert len(seen) == 3
    # Bounded near the shared 0.3s budget -- clearly under what 3 independent
    # 0.2s per-call ceilings would allow (0.6s) if the budget weren't shared.
    assert 0.15 <= elapsed <= 0.55
    assert result.response.response_text == "Here is what I have so far."


@pytest.mark.asyncio
async def test_budget_exhausted_calls_are_skipped_without_dispatch(retriever, monkeypatch) -> None:
    """When the fair-shared slice for a call falls below _TOOL_MIN_SLICE_S,
    it must never reach the executor at all -- the turn gets a synthetic
    "timed out" tool result for it instead of spending a connect round-trip
    on a call that would be cut short immediately anyway."""
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 1.0)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 10.0)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.3)

    seen: list[float] = []

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        seen.append(timeout_s)
        return {"result": "ok"}  # returns instantly

    crm_tools = [ToolSpec(name="get_balance", description="get balance",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id=f"t{i}", name="get_balance", arguments={}) for i in range(4)
        ]),
        LLMResult(text="done", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await asyncio.wait_for(agent.handle_message("balance please"), timeout=5.0)

    # budget(1.0)/4 = 0.25 < min_slice(0.3) for the first call -- it must be
    # skipped, so the executor is invoked fewer than 4 times.
    assert len(seen) < 4
    tool_msgs = [m for m in llm.calls[1][0] if m.role == "tool"]
    assert any("error" in json.loads(m.content) for m in tool_msgs)
    assert result.response.response_text == "done"


@pytest.mark.asyncio
async def test_budget_shared_across_rounds_not_reset(retriever, monkeypatch) -> None:
    """The cumulative budget must persist across BOTH tool rounds, not reset
    per round -- round 1's two slow calls consume (approximately) the whole
    budget, so round 2's call must be skipped by the min-slice guard rather
    than getting a fresh full budget."""
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 0.2)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 0.15)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.05)

    seen: list[float] = []

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        seen.append(timeout_s)
        await asyncio.sleep(10)
        return {"result": "never reached"}  # pragma: no cover

    crm_tools = [ToolSpec(name="get_balance", description="get balance",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_balance", arguments={}),
            ToolCall(id="t2", name="get_balance", arguments={}),
        ]),
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t3", name="get_balance", arguments={}),
        ]),
        # _max_tool_rounds (2) is exhausted while still wanting tools -> the
        # loop's `else` branch forces one more plain-text synthesis call.
        LLMResult(text="final answer", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await asyncio.wait_for(agent.handle_message("balance please"), timeout=5.0)

    # Only round 1's two calls ever reached the executor.
    assert len(seen) == 2
    third_round_tool_msgs = [
        m for m in llm.calls[2][0] if m.role == "tool" and m.tool_call_id == "t3"
    ]
    assert third_round_tool_msgs
    assert "error" in json.loads(third_round_tool_msgs[0].content)
    assert result.response.response_text == "final answer"


@pytest.mark.asyncio
async def test_slow_tool_turn_still_returns_a_real_answer_within_turn_timeout(
    retriever, monkeypatch,
) -> None:
    """The key end-to-end regression test: proves the turn completes via the
    cumulative FAIR-SHARE budget specifically, not merely that it completes
    at all (a loose bound can't tell bounded from unbounded behavior apart).

    With budget=0.3s, ceiling=0.2s and 8 sequential slow tool calls: a FIXED
    (fair-share) implementation slices each call as remaining/calls_left,
    which shrinks well below the 0.2s ceiling as calls proceed, so total tool
    time stays bounded near the 0.3s budget regardless of call count --
    comfortably inside the 0.8s wait_for wrapped around the whole turn below.
    A BROKEN implementation that instead gives every call the full
    _TOOL_CALL_CEILING_S regardless of how many calls remain would take
    8 x 0.2s = 1.6s, blowing the 0.8s wait and failing this test with a
    TimeoutError. (Verified by temporarily reintroducing that exact mutation
    and confirming this test fails -- see the fix-round report.)"""
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 0.3)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 0.2)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.01)

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        await asyncio.sleep(10)
        return {"result": "never reached"}  # pragma: no cover

    crm_tools = [ToolSpec(name="get_balance", description="get balance",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id=f"t{i}", name="get_balance", arguments={}) for i in range(8)
        ]),
        LLMResult(text="Final scripted answer.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    # Comfortably above the fixed version's ~0.3-0.4s, clearly below the
    # broken version's ~1.6s -- see docstring.
    result = await asyncio.wait_for(agent.handle_message("balance please"), timeout=0.8)
    assert result.response.response_text == "Final scripted answer."


@pytest.mark.asyncio
async def test_kb_search_gets_its_own_budget_independent_of_exhausted_crm_budget(
    retriever, monkeypatch,
) -> None:
    """A CRM tool call that exhausts the entire CRM tool budget (_TOOL_BUDGET_S)
    must NOT starve a same-round search_knowledge_base call. KB search has its
    own independent timeout (_KB_SEARCH_TIMEOUT_S) -- it must still actually
    execute and return real chunks, not the synthetic budget-exhausted stub it
    would get if it drew from the (already fully-spent) CRM budget."""
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 0.05)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 0.05)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.01)
    monkeypatch.setattr(chatbot_mod, "_KB_SEARCH_TIMEOUT_S", 5.0)

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        await asyncio.sleep(10)  # always cut short at its (tiny) CRM slice
        return {"result": "never reached"}  # pragma: no cover

    class _DelayedRetriever:
        """Wraps the real retriever with a delay well past what a near-zero
        exhausted-budget slice would allow (the CRM call alone consumes the
        whole 0.05s _TOOL_BUDGET_S), but comfortably inside the 5.0s
        _KB_SEARCH_TIMEOUT_S -- proves KB search got its own real budget
        rather than a slice inherited from the exhausted CRM budget."""

        async def search(self, *a, **kw):
            await asyncio.sleep(0.1)
            return await retriever.search(*a, **kw)

    crm_tools = [ToolSpec(name="get_balance", description="get balance",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_balance", arguments={}),
            ToolCall(id="t2", name="search_knowledge_base", arguments={"query": "Plan B"}),
        ]),
        LLMResult(text="Here you go.", finish_reason="stop"),
    ])
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-kb-budget"), llm=llm, retriever=_DelayedRetriever(),
        company_name="Acme", language_default="en", enable_tools=True,
        crm_tools=crm_tools, crm_executor=crm_exec)
    result = await asyncio.wait_for(agent.handle_message("balance and Plan B please"), timeout=5.0)

    # The KB search actually ran and returned real chunks -- not skipped, and
    # not the budget-exhausted stub a near-zero inherited slice would produce.
    tool_msgs = {m.tool_call_id: json.loads(m.content) for m in llm.calls[1][0] if m.role == "tool"}
    assert "results" in tool_msgs["t2"], tool_msgs["t2"]
    assert tool_msgs["t2"]["results"], "KB search must return real results, not be skipped"
    assert "error" not in tool_msgs["t2"]
    assert result.response.response_text == "Here you go."


@pytest.mark.asyncio
async def test_lone_tool_call_gets_the_full_per_call_ceiling(retriever, monkeypatch) -> None:
    """A single tool call in the round gets the full per-call ceiling, NOT
    the (much larger) whole cumulative budget."""
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 100.0)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 5.0)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.01)

    seen: list[float] = []

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        seen.append(timeout_s)
        return {"result": "ok"}

    crm_tools = [ToolSpec(name="get_balance", description="get balance",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_balance", arguments={})]),
        LLMResult(text="ok", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    await agent.handle_message("balance please")

    # Bound by the 5.0s ceiling, not the 100.0s budget -- observed through
    # the executor's timeout_s, which is the ceiling-derived slice minus the
    # 1.0s CRM dispatch margin applied in _dispatch_tool.
    assert seen == [pytest.approx(5.0 - 1.0, abs=0.05)]


@pytest.mark.asyncio
async def test_fast_first_call_returns_unused_slice_to_the_second(retriever, monkeypatch) -> None:
    """Call 1 finishes instantly; its unused share of the budget must flow
    to call 2, which should then get close to the FULL remaining budget --
    not a naive static budget/2 split that would shortchange it."""
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 3.0)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 10.0)  # non-binding
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.01)

    seen: dict[str, float] = {}

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        seen[tc.id] = timeout_s
        return {"result": "ok"}  # both calls return instantly

    crm_tools = [ToolSpec(name="get_balance", description="get balance",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_balance", arguments={}),
            ToolCall(id="t2", name="get_balance", arguments={}),
        ]),
        LLMResult(text="ok", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    await agent.handle_message("balance please")

    # A naive equal split would give call 2 ~1.5s (budget/2) too -- minus the
    # margin, ~0.5s. Since call 1 consumed ~0 real time, call 2 instead gets
    # essentially the whole remaining budget (3.0s, minus the margin) --
    # clearly more than call 1's half-share, and more than a naive split.
    assert seen["t1"] == pytest.approx(0.5, abs=0.1)
    assert seen["t2"] == pytest.approx(2.0, abs=0.1)
    assert seen["t2"] > seen["t1"]


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


# --- Deposit verification tool -------------------------------------------


@pytest.mark.asyncio
async def test_deposit_verification_tool_without_executor_returns_error(retriever) -> None:
    agent = _agent(ScriptedLLM([]), retriever)
    tc = ToolCall(id="t1", name="submit_deposit_verification", arguments={"order_id": "ORD-1"})
    result, chunks, escalation, call_offer = await agent._dispatch_tool(tc, timeout_s=10.0)
    assert result == {"error": "verification is not available"}
    assert chunks == []
    assert escalation is None
    assert call_offer is None


@pytest.mark.asyncio
async def test_deposit_verification_executor_exception_is_swallowed(retriever, caplog) -> None:
    async def boom_exec(tc: ToolCall, *, timeout_s: float = 0.0):
        raise RuntimeError("boom")

    agent = _agent(ScriptedLLM([]), retriever, deposit_verification_executor=boom_exec)
    tc = ToolCall(id="t1", name="submit_deposit_verification", arguments={"order_id": "ORD-1"})
    with caplog.at_level(logging.ERROR, logger="src.agents.chatbot"):
        result, chunks, escalation, call_offer = await agent._dispatch_tool(tc, timeout_s=10.0)
    assert result["status"] == "error"
    assert "message" in result
    assert chunks == []
    assert escalation is None
    assert call_offer is None
    assert any("deposit verification submission failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_deposit_verification_executor_result_is_passed_through_and_budget_is_reduced(
    retriever,
) -> None:
    seen: list[float] = []

    async def deposit_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        seen.append(timeout_s)
        return {"status": "submitted"}

    agent = _agent(ScriptedLLM([]), retriever, deposit_verification_executor=deposit_exec)
    tc = ToolCall(id="t1", name="submit_deposit_verification", arguments={"order_id": "ORD-1"})

    result, chunks, escalation, call_offer = await agent._dispatch_tool(tc, timeout_s=10.0)
    assert result == {"status": "submitted"}
    assert chunks == []
    assert escalation is None
    assert call_offer is None
    # _dispatch_tool gives the executor its own margin: max(0.5, timeout_s - 1.0).
    assert seen[-1] == pytest.approx(9.0)

    # A very small remaining budget must floor at 0.5s, never hit zero/negative.
    await agent._dispatch_tool(tc, timeout_s=0.6)
    assert seen[-1] == pytest.approx(0.5)
    await agent._dispatch_tool(tc, timeout_s=0.0)
    assert seen[-1] == pytest.approx(0.5)

    # A non-dict executor result is wrapped as {"result": <value>}.
    async def deposit_exec_non_dict(tc: ToolCall, *, timeout_s: float = 0.0):
        return "submitted"

    agent_non_dict = _agent(
        ScriptedLLM([]), retriever, deposit_verification_executor=deposit_exec_non_dict)
    non_dict_result, _, _, _ = await agent_non_dict._dispatch_tool(tc, timeout_s=10.0)
    assert non_dict_result == {"result": "submitted"}
