"""Phase 2 of the chat turn-metrics plan (docs/superpowers/plans/
2026-09-08-chatbot-turn-metrics.md): the ChatBotAgent.record_metric write
path. Reuses ScriptedLLM/_agent()'s shape from test_chatbot_metrics.py (own
copy per repo convention -- test_chatbot_tools.py/test_chatbot_agent.py/
test_chat_turn_timing.py/test_chatbot_metrics.py each define their own
identically-shaped retriever fixture rather than cross-importing)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.agents import chatbot as chatbot_mod
from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.interfaces.llm import ILLMProvider, LLMMessage, LLMResult, ToolCall, ToolSpec
from src.interfaces.vector_store import Document
from src.models.chat_turn_metrics import ChatToolMetricRow, ChatTurnMetric, record_chat_turn_metric
from src.models.database import Base
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
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


class RecordingMetric:
    """Test double for ChatBotAgent's record_metric callback -- records every
    payload it receives verbatim."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, payload: dict) -> None:
        self.calls.append(payload)


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
    ])
    return r


@pytest_asyncio.fixture
async def chat_metrics_db():
    """In-memory sqlite sessionmaker for the real record_chat_turn_metric
    write path -- same shape as test_chat_turn_metrics_model.py's own
    ``sessionmaker`` fixture."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()


def _agent(llm, retriever, **kw) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-metrics-persist"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en", enable_tools=True,
        session_id=kw.pop("session_id", "sess-1"),
        llm_provider=kw.pop("llm_provider", "GeminiLLMAdapter"),
        llm_model=kw.pop("llm_model", "gemini-2.0-flash"),
        **kw)


async def test_two_round_tool_turn_emits_correct_parent_and_child_payload(retriever) -> None:
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
    recorder = RecordingMetric()
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec,
                   record_metric=recorder)
    result = await agent.handle_message("what's my balance and tell me about Plan B?")

    assert len(recorder.calls) == 1
    payload = recorder.calls[0]
    assert payload["session_id"] == "sess-1"
    assert payload["path"] == "tools"
    assert payload["llm_provider"] == "GeminiLLMAdapter"
    assert payload["llm_model"] == "gemini-2.0-flash"
    assert payload["action"] == result.response.action

    m = payload["metrics"]
    assert m["rounds"] == 2
    assert m["tool_calls"] == 1  # only the CRM call counts, not the KB search
    assert m["kb_searches"] == 1
    assert m["rounds_exhausted"] is False
    assert m["retry_fired"] is False

    tools = payload["tools"]
    names_kinds = {(t["tool_name"], t["kind"]) for t in tools}
    assert ("search_knowledge_base", "kb") in names_kinds
    assert ("get_player_wallet", "crm") in names_kinds
    assert all(t["outcome"] == "ok" for t in tools)
    # Every child dict is DB-column-shaped and PII-clean: bounded fields only.
    for t in tools:
        assert set(t.keys()) == {
            "tool_name", "kind", "latency_ms", "outcome", "budget_slice_ms", "round_index",
        }


async def test_timeout_turn_emits_timeout_outcome_and_directive_flag(retriever, monkeypatch) -> None:
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
    recorder = RecordingMetric()
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec,
                   record_metric=recorder)
    await asyncio.wait_for(agent.handle_message("where is my withdrawal?"), timeout=5.0)

    assert len(recorder.calls) == 1
    payload = recorder.calls[0]
    assert payload["metrics"]["failure_directive_fired"] is True
    assert payload["metrics"]["tool_timeouts"] == 1
    tool_entry = next(t for t in payload["tools"] if t["tool_name"] == "get_player_transactions")
    assert tool_entry["outcome"] == "timeout"


async def test_single_shot_path_emits_metric_with_path_single_shot(retriever) -> None:
    llm = ScriptedLLM([LLMResult(text="Plan B has 500GB unlimited data.", finish_reason="stop")])
    recorder = RecordingMetric()
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-single-persist"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en",  # enable_tools defaults False
        session_id="sess-single", llm_provider="GeminiLLMAdapter", llm_model="gemini-2.0-flash",
        record_metric=recorder,
    )
    await agent.handle_message("Tell me about Plan B")

    assert len(recorder.calls) == 1
    payload = recorder.calls[0]
    assert payload["path"] == "single_shot"
    assert payload["metrics"]["kb_searches"] == 0
    assert payload["tools"] == []


async def test_no_record_metric_configured_is_a_silent_noop(retriever) -> None:
    """record_metric defaults to None -- a turn must complete identically
    without ever touching the (absent) callback."""
    llm = ScriptedLLM([LLMResult(text="Hello!", finish_reason="stop")])
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-none"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en")
    result = await agent.handle_message("hi")
    assert result.response.response_text == "Hello!"


async def test_metrics_assembly_failure_skips_persistence_without_raising(retriever, monkeypatch) -> None:
    """When ChatTurnMetrics construction itself fails (metrics=None on the
    result), there is nothing correct to persist -- record_metric must not be
    called at all, and the turn must still complete normally."""
    llm = ScriptedLLM([LLMResult(text="Plan B has 500GB unlimited data.", finish_reason="stop")])
    recorder = RecordingMetric()
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-explode-persist"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en", record_metric=recorder)

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(chatbot_mod, "ChatTurnMetrics", _boom)
    result = await agent.handle_message("Tell me about Plan B")

    assert result.metrics is None
    assert result.response.response_text == "Plan B has 500GB unlimited data."
    assert recorder.calls == []


# --- Review fix 1: a slow/stalled record_metric must never delay or break --
# --- the turn itself (it must not ride inside the WS layer's 90s per-turn --
# --- asyncio.wait_for -- see _RECORD_METRIC_TIMEOUT_S in chatbot.py). ------


async def test_slow_record_metric_does_not_delay_or_break_the_turn(retriever) -> None:
    """record_chat_turn_metric never raises, but it has no timeout of its
    own -- a real DB stall could otherwise run long enough to push an
    already-near-budget turn past the WS layer's 90s ceiling and get an
    already-computed reply cancelled and discarded. _emit_turn_metric bounds
    the callback with _RECORD_METRIC_TIMEOUT_S; this asserts handle_message
    returns promptly (well under the callback's artificial 5s stall) with a
    normal, complete result."""
    async def slow_record_metric(payload: dict) -> None:
        await asyncio.sleep(5.0)  # far longer than _RECORD_METRIC_TIMEOUT_S

    llm = ScriptedLLM([LLMResult(text="Hello!", finish_reason="stop")])
    agent = ChatBotAgent(
        session=AgentSession(session_id="cb-slow-metric"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en", record_metric=slow_record_metric)

    # If the internal timeout regressed (or were removed), this outer
    # wait_for -- well under the callback's 5s stall -- would itself time out
    # and fail the test.
    result = await asyncio.wait_for(agent.handle_message("hi"), timeout=3.5)

    assert result.response.response_text == "Hello!"
    assert result.metrics is not None


async def test_slow_tools_path_record_metric_does_not_delay_or_break_the_turn(retriever) -> None:
    """Same invariant as above, exercised on the tools path (the other
    _emit_turn_metric call site)."""
    async def slow_record_metric(payload: dict) -> None:
        await asyncio.sleep(5.0)

    async def crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
        return {"balance": 500}

    crm_tools = [ToolSpec(name="get_player_wallet", description="wallet",
                          parameters={"type": "object", "properties": {}})]
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="get_player_wallet", arguments={})]),
        LLMResult(text="Your balance is ₹500.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec,
                   record_metric=slow_record_metric)

    result = await asyncio.wait_for(
        agent.handle_message("what's my balance?"), timeout=3.5)

    assert result.response.response_text == "Your balance is ₹500."
    assert result.metrics is not None


# --- Review fix 2: a permanent end-to-end test against the REAL -----------
# --- record_chat_turn_metric (every other test in this file uses the -------
# --- RecordingMetric stub, which only checks the payload's SHAPE, not that --
# --- it actually matches record_chat_turn_metric's real signature). --------


async def test_real_agent_turn_persists_via_record_chat_turn_metric(
    retriever, chat_metrics_db, monkeypatch,
) -> None:
    """Drives a real two-round tool turn with record_metric wired to the
    ACTUAL record_chat_turn_metric (src/models/chat_turn_metrics.py),
    exactly as src/bootstrap.py's make_chatbot_factory wires it in
    production -- ``lambda payload: record_chat_turn_metric(tenant_id=...,
    crm_id=..., **payload)``. This is the one test that would catch a future
    keyword-argument drift between the agent's emitted payload and the
    writer's signature: a mismatch there is a production-only TypeError,
    swallowed into a per-turn WARNING (silently zero rows forever) with an
    otherwise fully green suite that only exercises the RecordingMetric
    stub."""
    monkeypatch.setattr(
        "src.models.chat_turn_metrics.get_sessionmaker", lambda: chat_metrics_db,
    )

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
    agent = _agent(
        llm, retriever, crm_tools=crm_tools, crm_executor=crm_exec,
        session_id="sess-e2e", llm_provider="GeminiLLMAdapter", llm_model="gemini-2.0-flash",
        record_metric=lambda payload: record_chat_turn_metric(
            tenant_id="dev", crm_id="betstudio", **payload),
    )
    result = await agent.handle_message("what's my balance and tell me about Plan B?")

    async with chat_metrics_db() as db:
        turns = (await db.execute(select(ChatTurnMetric))).scalars().all()
        tools = (await db.execute(select(ChatToolMetricRow))).scalars().all()

    assert len(turns) == 1
    row = turns[0]
    assert row.tenant_id == "dev"
    assert row.crm_id == "betstudio"
    assert row.session_id == "sess-e2e"
    assert row.path == "tools"
    assert row.llm_provider == "GeminiLLMAdapter"
    assert row.llm_model == "gemini-2.0-flash"
    assert row.action == result.response.action
    assert row.rounds == 2
    assert row.tool_calls == 1  # only the CRM call, not the KB search
    assert row.kb_searches == 1

    assert len(tools) == 2
    by_name = {t.tool_name: t for t in tools}
    assert by_name["search_knowledge_base"].kind == "kb"
    assert by_name["get_player_wallet"].kind == "crm"
    assert all(t.turn_id == row.id for t in tools)
    assert all(t.tenant_id == "dev" for t in tools)  # denormalized correctly
