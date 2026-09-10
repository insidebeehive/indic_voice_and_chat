"""Phase 1 of the chat turn-metrics plan: ChatTurnMetrics/ChatToolMetric
assembly on both agent paths (docs/superpowers/plans/2026-09-08-chatbot-turn-metrics.md).

Defines its own ScriptedLLM/retriever/_agent harness (matching
test_chatbot_tools.py's shape) rather than cross-importing fixtures from
another test module -- repo convention (test_chatbot_tools.py,
test_chatbot_agent.py, test_chat_turn_timing.py each define their own
identically-shaped ``retriever`` fixture).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from src.agents import chatbot as chatbot_mod
from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.interfaces.llm import ILLMProvider, LLMMessage, LLMResult, ToolCall, ToolSpec
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import HybridRetriever, RetrievalConfig

pytestmark = pytest.mark.asyncio


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
        embedder=HashEmbedder(dim=64), vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8,
                               similarity_threshold=0.0))
    await r.index([
        Document(id="c1", content="Plan B has 500GB unlimited data.",
                 metadata={"filename": "plans.pdf", "page": 2}),
    ])
    return r


def _agent(llm, retriever, **kw) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-metrics"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en", enable_tools=True, **kw)


# --- Tool loop: rounds, per-tool kind/outcome, KB-vs-CRM budget separation --


async def test_two_round_tool_turn_has_correct_rounds_and_tool_entries(retriever) -> None:
    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"balance": 500}

    crm_tools = [ToolSpec(name="get_player_wallet", description="wallet",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "plans"}),
            ToolCall(id="t2", name="get_player_wallet", arguments={}),
        ]),
        LLMResult(text="Your balance is ₹500.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await agent.handle_message("what's my balance and tell me about Plan B?")

    m = result.metrics
    assert m is not None
    assert m.path == "tools"
    # round 1: both tool calls; round 2: final text, no more tool calls ->
    # loop breaks.
    assert m.rounds == 2
    names_kinds = {(t.tool_name, t.kind) for t in m.tools}
    assert ("search_knowledge_base", "kb") in names_kinds
    assert ("get_player_wallet", "crm") in names_kinds
    assert all(t.outcome == "ok" for t in m.tools)
    # round_index reflects which round each call happened in (0-indexed) --
    # both calls were emitted in round 1 (index 0) here.
    assert {t.round_index for t in m.tools} == {0}
    # A normally-completing turn never exhausts its round budget and never
    # needs the unusable-response retry.
    assert m.rounds_exhausted is False
    assert m.retry_fired is False


async def test_tool_total_ms_excludes_kb_search_which_counts_separately(
    retriever, monkeypatch,
) -> None:
    """Real separation, not a tolerance artifact: the KB search is made
    artificially slow (200ms) so that if its latency were ever folded into
    tool_total_ms (the exact bug the plan warns about in §3), this test
    would fail hard rather than passing by coincidence of both times being
    near-zero."""
    async def slow_search(*a, **kw):
        await asyncio.sleep(0.2)
        return []

    monkeypatch.setattr(chatbot_mod, "search_combined", slow_search)

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"balance": 500}

    crm_tools = [ToolSpec(name="get_player_wallet", description="wallet",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "plans"}),
            ToolCall(id="t2", name="get_player_wallet", arguments={}),
        ]),
        LLMResult(text="Your balance is ₹500.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await asyncio.wait_for(
        agent.handle_message("balance and Plan B please"), timeout=5.0)

    m = result.metrics
    assert m is not None
    assert m.kb_searches == 1
    # The slow (200ms) KB search's time must land in kb_search_ms...
    assert m.kb_search_ms >= 190
    # ...and must NOT be folded into tool_total_ms, which tracks only the
    # (near-instant) CRM call.
    assert m.tool_total_ms < 50
    assert m.tool_calls == 1  # only the CRM call counts toward tool_calls


async def test_timing_out_crm_executor_marks_timeout_and_fires_directive(
    retriever, monkeypatch,
) -> None:
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 0.05)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 0.05)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.01)

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        await asyncio.sleep(10)
        return {"result": "never reached"}  # pragma: no cover

    crm_tools = [ToolSpec(name="get_player_transactions", description="txns",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_player_transactions", arguments={})]),
        LLMResult(text="I can't verify this right now, let me connect you to a human.",
                  finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await asyncio.wait_for(agent.handle_message("where is my withdrawal?"), timeout=5.0)

    m = result.metrics
    assert m is not None
    assert any(t.tool_name == "get_player_transactions" and t.outcome == "timeout" for t in m.tools)
    assert m.failure_directive_fired is True
    assert m.tool_timeouts == 1
    assert m.tool_failures == 1


async def test_second_consecutive_failing_turn_escalates_directive(retriever) -> None:
    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"error": "timed out", "failure": "timeout"}

    crm_tools = [ToolSpec(name="get_player_wallet", description="wallet",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_player_wallet", arguments={})]),
        LLMResult(text="I can't check that right now.", finish_reason="stop"),
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t2", name="get_player_wallet", arguments={})]),
        LLMResult(text="Still can't check that.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    turn1 = await agent.handle_message("what's my balance?")
    turn2 = await agent.handle_message("please check again")

    assert turn1.metrics is not None and turn1.metrics.failure_directive_fired is True
    assert turn1.metrics.failure_directive_escalated is False
    assert turn2.metrics is not None
    assert turn2.metrics.failure_directive_fired is True
    assert turn2.metrics.failure_directive_escalated is True


async def test_directive_fired_flag_survives_same_turn_recovery(retriever) -> None:
    """Distinguishes failure_directive_fired from an end-of-turn derivation
    off directive_index/failed_category_names: round 1's call fails
    (directive appended, directive_index set); round 2's retry of the SAME
    tool succeeds, which clears failed_category_names AND deletes the
    directive message, resetting directive_index back to None. A derivation
    of `directive_index is not None` at the end of the turn would wrongly
    read this as "no directive fired" -- the flag correctly remembers that
    it did, in round 1, and reached the model."""
    calls = {"n": 0}

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"error": "timed out", "failure": "timeout"}
        return {"balance": 500}

    crm_tools = [ToolSpec(name="get_player_wallet", description="wallet",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_player_wallet", arguments={})]),
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t2", name="get_player_wallet", arguments={})]),
        LLMResult(text="Your balance is ₹500.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec, max_tool_rounds=3)
    result = await agent.handle_message("what's my balance?")

    assert result.response.response_text == "Your balance is ₹500."
    assert result.metrics is not None
    assert result.metrics.failure_directive_fired is True


async def test_budget_exhausted_skip_marks_skipped_budget(retriever, monkeypatch) -> None:
    monkeypatch.setattr(chatbot_mod, "_TOOL_BUDGET_S", 1.0)
    monkeypatch.setattr(chatbot_mod, "_TOOL_CALL_CEILING_S", 10.0)
    monkeypatch.setattr(chatbot_mod, "_TOOL_MIN_SLICE_S", 0.3)

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"result": "ok"}

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

    m = result.metrics
    assert m is not None
    # budget(1.0)/4 = 0.25 < min_slice(0.3) for the first call -- it must be
    # skipped without ever reaching the executor.
    assert m.tool_calls_skipped >= 1
    assert any(t.outcome == "skipped_budget" for t in m.tools)


async def test_rounds_exhausted_true_on_forced_final_answer(retriever) -> None:
    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"error": "timed out", "failure": "timeout"}

    crm_tools = [ToolSpec(name="get_player_transactions", description="txns",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_player_transactions", arguments={})]),
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t2", name="get_player_transactions", arguments={})]),
        # max_tool_rounds default is 2 -- the loop's `else` branch fires here.
        LLMResult(text="I can't verify this right now, let me connect you to a human.",
                  finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await agent.handle_message("where is my ₹19,600 withdrawal?")

    assert result.metrics is not None
    assert result.metrics.rounds_exhausted is True


async def test_unusable_final_answer_fires_retry_in_tools_path(retriever) -> None:
    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"status": "settled", "result": "won"}

    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_matka_bids", arguments={"status": "settled"})]),
        LLMResult(text="", finish_reason="stop"),  # final synthesis: empty -> unusable
        LLMResult(text="Your last Matka bid was settled as a win.", finish_reason="stop"),
    ])
    crm_tools = [ToolSpec(name="get_matka_bids", description="Get Matka bids",
                          parameters={"type": "object", "properties": {}})]
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)
    result = await agent.handle_message("meri jeet credit kyu nahi hui?")

    assert result.metrics is not None
    assert result.metrics.retry_fired is True


async def test_retry_fired_false_on_clean_single_shot_turn(retriever) -> None:
    llm = ScriptedLLM([LLMResult(text="Hello! How can I help?", finish_reason="stop")])
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-clean"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en")  # enable_tools defaults False
    result = await agent.handle_message("hi")

    assert result.metrics is not None
    assert result.metrics.retry_fired is False


async def test_deposit_verification_tool_kind_and_outcome(retriever) -> None:
    async def deposit_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"status": "submitted"}

    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="submit_deposit_verification", arguments={"order_id": "ORD-1"})]),
        LLMResult(text="Your verification request has been submitted.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, deposit_verification_executor=deposit_exec)
    result = await agent.handle_message("please verify my deposit")

    assert result.metrics is not None
    entry = next(t for t in result.metrics.tools if t.tool_name == "submit_deposit_verification")
    assert entry.kind == "deposit_verification"
    assert entry.outcome == "ok"
    # deposit_verification is budgeted alongside CRM calls (shares
    # tool_elapsed_s/_TOOL_BUDGET_S -- see _handle_with_tools), so it counts
    # toward tool_calls just like a CRM call would.
    assert result.metrics.tool_calls == 1


async def test_ungrounded_currency_figure_fires_unverified_data_guard(retriever) -> None:
    llm = ScriptedLLM([
        LLMResult(text="Your balance is ₹99,999 exactly.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever)
    result = await agent.handle_message("what's my balance?")

    assert result.metrics is not None
    assert result.metrics.guard_unverified_data_fired is True


async def test_single_shot_path_reports_path_and_zero_kb_searches(retriever) -> None:
    llm = ScriptedLLM([LLMResult(text="Plan B has 500GB unlimited data.", finish_reason="stop")])
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-single"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en")  # enable_tools defaults False
    result = await agent.handle_message("Tell me about Plan B")

    assert result.metrics is not None
    assert result.metrics.path == "single_shot"
    assert result.metrics.kb_searches == 0


# --- Metrics assembly must never break a live turn --------------------------


async def test_metrics_assembly_failure_does_not_break_the_reply(retriever, monkeypatch) -> None:
    """A bug in metrics assembly (e.g. ChatTurnMetrics construction raising)
    must degrade to metrics=None, never break the turn's actual reply."""
    llm = ScriptedLLM([LLMResult(text="Plan B has 500GB unlimited data.", finish_reason="stop")])
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-explode"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en")

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(chatbot_mod, "ChatTurnMetrics", _boom)
    result = await agent.handle_message("Tell me about Plan B")

    assert result.metrics is None
    assert result.response.response_text == "Plan B has 500GB unlimited data."


async def test_metrics_assembly_failure_does_not_break_tools_path_reply(retriever, monkeypatch) -> None:
    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"status": "dispatched"}

    crm_tools = [ToolSpec(name="check_order_status", description="check order",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="check_order_status", arguments={})]),
        LLMResult(text="Your order is dispatched.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(chatbot_mod, "ChatTurnMetrics", _boom)
    result = await agent.handle_message("where is my order?")

    assert result.metrics is None
    assert result.response.response_text == "Your order is dispatched."


async def test_per_tool_metric_failure_does_not_break_the_reply(retriever, monkeypatch) -> None:
    """Same invariant as above, but for the PER-TOOL ChatToolMetric
    construction inside the hot tool loop (chatbot.py, inside the per-call
    try/except) -- a narrower and separately-guarded piece of code from the
    turn-level ChatTurnMetrics assembly tested above. A raise here must
    degrade to "this tool's entry is missing from metrics.tools", never
    break the reply or the rest of metrics assembly."""
    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"status": "dispatched"}

    crm_tools = [ToolSpec(name="check_order_status", description="check order",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="check_order_status", arguments={})]),
        LLMResult(text="Your order is dispatched.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec)

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(chatbot_mod, "ChatToolMetric", _boom)
    result = await agent.handle_message("where is my order?")

    assert result.response.response_text == "Your order is dispatched."
    # The per-tool entry couldn't be constructed, but turn-level metrics
    # assembly (which only ever reads the tool_metrics list, never raises on
    # it being short) must still succeed.
    assert result.metrics is not None
    assert result.metrics.tools == ()
