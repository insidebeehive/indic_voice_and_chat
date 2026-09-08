"""Turn-metrics plan Phase 2, item 3: an exploding record_metric callback
must never alter a turn's outcome. Drives a full two-round tool turn through
two otherwise-identical agents -- one with record_metric=None, one with an
always-raising callback -- and asserts the resulting ChatTurnResult is
field-for-field identical. (The tracing plan's ExplodingTracer idea, applied
here -- see the turn-metrics plan §9 item 3.)"""

from __future__ import annotations

import dataclasses
import itertools
from typing import AsyncIterator

import pytest

from src.agents import chatbot as chatbot_mod
from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.interfaces.llm import ILLMProvider, LLMMessage, LLMResult, ToolCall, ToolSpec
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import HybridRetriever, RetrievalConfig

pytestmark = pytest.mark.asyncio


class ScriptedLLM(ILLMProvider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)

    async def generate(self, messages, config) -> LLMResult:
        return self._results.pop(0) if self._results else LLMResult(text="ok", finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


async def _exploding_record_metric(payload: dict) -> None:
    raise RuntimeError("record_metric exploded")


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


def _scripted_llm() -> ScriptedLLM:
    return ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "plans"}),
            ToolCall(id="t2", name="get_player_wallet", arguments={}),
        ]),
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t3", name="get_player_wallet", arguments={})]),
        LLMResult(text="Your balance is ₹500.", finish_reason="stop"),
    ])


async def _crm_exec(tc: ToolCall, *, timeout_s: float = 0.0) -> dict:
    return {"balance": 500}


def _build_agent(retriever, *, record_metric) -> ChatBotAgent:
    crm_tools = [ToolSpec(name="get_player_wallet", description="wallet",
                          parameters={"type": "object", "properties": {}})]
    return ChatBotAgent(
        session=AgentSession(session_id="cb-explode-metric"),
        llm=_scripted_llm(), retriever=retriever,
        company_name="Acme", language_default="en", enable_tools=True,
        max_tool_rounds=3, crm_tools=crm_tools, crm_executor=_crm_exec,
        session_id="sess-1", llm_provider="GeminiLLMAdapter", llm_model="gemini-2.0-flash",
        record_metric=record_metric,
    )


def _reset_perf_counter(monkeypatch) -> None:
    """Both runs below must produce identical timing numbers for the
    field-for-field comparison to be meaningful (rather than trivially
    failing on real wall-clock jitter) -- give time.perf_counter() a fresh,
    deterministic, fixed-step sequence before each run."""
    counter = itertools.count(step=0.01)
    monkeypatch.setattr(chatbot_mod.time, "perf_counter", lambda: next(counter))


async def test_exploding_record_metric_does_not_change_turn_result(retriever, monkeypatch) -> None:
    _reset_perf_counter(monkeypatch)
    baseline_agent = _build_agent(retriever, record_metric=None)
    baseline = await baseline_agent.handle_message("what's my balance and tell me about Plan B?")

    _reset_perf_counter(monkeypatch)
    exploding_agent = _build_agent(retriever, record_metric=_exploding_record_metric)
    exploded = await exploding_agent.handle_message("what's my balance and tell me about Plan B?")

    assert dataclasses.asdict(exploded.response) == dataclasses.asdict(baseline.response)
    assert exploded.retrieved == baseline.retrieved
    assert exploded.rag_context_chars == baseline.rag_context_chars
    assert exploded.escalation == baseline.escalation
    assert exploded.call_offer == baseline.call_offer
    assert exploded.input_tokens == baseline.input_tokens
    assert exploded.output_tokens == baseline.output_tokens
    assert exploded.llm_provider == baseline.llm_provider
    assert exploded.llm_model == baseline.llm_model
    assert exploded.metrics == baseline.metrics
