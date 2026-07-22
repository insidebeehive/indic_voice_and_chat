"""Latency instrumentation for the ChatBot turn path (production log visibility).

Covers:
 - chatbot.py: one summary INFO log per turn (`_single_shot` and
   `_handle_with_tools`), with per-LLM-call and per-tool timings.
 - gemini.py: `_call_with_retry` logs when semaphore-wait + backoff sleep
   exceeds 1s, and stays silent on the fast/common path.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult, ToolCall
from src.interfaces.vector_store import Document
from src.providers.llm.gemini import GeminiLLMAdapter
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
    ])
    return r


def _agent(llm, retriever, **kw) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-timing"), llm=llm, retriever=retriever,
        company_name="Acme", language_default="en", **kw)


# --- chatbot.py: single-shot summary log --------------------------------


@pytest.mark.asyncio
async def test_single_shot_logs_turn_summary(retriever, caplog) -> None:
    llm = ScriptedLLM([LLMResult(text="Hello there!", finish_reason="stop")])
    agent = _agent(llm, retriever)
    with caplog.at_level(logging.INFO, logger="src.agents.chatbot"):
        await agent.handle_message("hi")
    records = [r for r in caplog.records if "chat turn done in" in r.getMessage()]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "llm=" in msg
    assert "retrieval=" in msg
    assert "retrieved=" in msg
    # One LLM call was made -> one timing entry rendered in the llm list.
    assert msg.count("ms") >= 1


# --- chatbot.py: tool-path summary log -----------------------------------


@pytest.mark.asyncio
async def test_tool_path_logs_turn_summary_with_tool_name(retriever, caplog) -> None:
    llm = ScriptedLLM([
        LLMResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="t1", name="search_knowledge_base", arguments={"query": "Plan B"})]),
        LLMResult(text="Plan B has 500GB unlimited data.", finish_reason="stop"),
    ])
    agent = _agent(llm, retriever, enable_tools=True)
    with caplog.at_level(logging.INFO, logger="src.agents.chatbot"):
        result = await agent.handle_message("Tell me about Plan B")
    assert result.response.response_text == "Plan B has 500GB unlimited data."
    records = [r for r in caplog.records if "chat turn done in" in r.getMessage()]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "search_knowledge_base" in msg
    assert "rounds=" in msg
    assert "tools=" in msg


# --- gemini.py: sem-wait / backoff logging -------------------------------


class _FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"{code} transient")
        self.code = code


def _response(text: str) -> SimpleNamespace:
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
        finish_reason=SimpleNamespace(name="STOP"),
    )
    return SimpleNamespace(
        text=text, candidates=[candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )


@pytest.mark.asyncio
async def test_gemini_logs_wait_when_backoff_exceeds_threshold(monkeypatch, caplog) -> None:
    # Force a real (but tiny) backoff sleep by shrinking the 429 schedule so the
    # test stays fast, then push the reported threshold down via a small
    # sleep-time inflation isn't needed — instead we assert the log fires when
    # sem_wait+backoff > 1.0s by shrinking the log threshold path itself:
    # patch the module's rate-limit backoff window to something that reliably
    # sleeps >1s once, since that's the real trigger path in production.
    import src.providers.llm.gemini as gemini_mod
    monkeypatch.setattr(gemini_mod, "_RATE_LIMIT_BACKOFF_S", ((1.05, 1.05),))
    monkeypatch.setattr(gemini_mod, "_RATE_LIMIT_MAX_RETRIES", 1)

    calls = {"n": 0}

    async def _quota_once(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _FakeAPIError(429)
        return _response("ok")

    models = SimpleNamespace(
        generate_content=AsyncMock(side_effect=_quota_once),
        generate_content_stream=AsyncMock(),
    )
    adapter = GeminiLLMAdapter({"client": SimpleNamespace(aio=SimpleNamespace(models=models))})
    with caplog.at_level(logging.INFO, logger="src.providers.llm.gemini"):
        result = await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    assert result.text == "ok"
    records = [r for r in caplog.records if "gemini call waited" in r.getMessage()]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "sem_wait=" in msg
    assert "backoff=" in msg


@pytest.mark.asyncio
async def test_gemini_fast_path_does_not_log_wait(caplog) -> None:
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=AsyncMock(return_value=_response("ok")),
        generate_content_stream=AsyncMock(),
    )))
    adapter = GeminiLLMAdapter({"client": client})
    with caplog.at_level(logging.INFO, logger="src.providers.llm.gemini"):
        result = await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    assert result.text == "ok"
    records = [r for r in caplog.records if "gemini call waited" in r.getMessage()]
    assert records == []
