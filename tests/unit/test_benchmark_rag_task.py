from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
import yaml

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.benchmarks.datasets import RAGSample, TaskScenario, TaskTurn
from src.benchmarks.rag_benchmark import (
    run_rag_benchmark,
    score_answer,
    score_retrieval,
)
from src.benchmarks.task_benchmark import run_task_benchmark
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.interfaces.vector_store import Document
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import HybridRetriever, RetrievalConfig


# --- Score functions ----------------------------------------------------


def test_score_retrieval_perfect_match() -> None:
    s = score_retrieval(expected_chunk_ids=["a", "b"], retrieved_chunk_ids=["a", "b", "c"])
    assert s.recall_at_k == 1.0
    assert s.precision_at_k == pytest.approx(2 / 3)
    assert s.reciprocal_rank == 1.0
    assert s.hit is True


def test_score_retrieval_miss() -> None:
    s = score_retrieval(["a"], ["x", "y", "z"])
    assert s.recall_at_k == 0.0
    assert s.precision_at_k == 0.0
    assert s.reciprocal_rank == 0.0
    assert s.hit is False


def test_score_retrieval_mrr() -> None:
    s = score_retrieval(["b"], ["x", "y", "b"])
    assert s.reciprocal_rank == pytest.approx(1 / 3)


def test_score_retrieval_empty_ground_truth_is_trivially_correct() -> None:
    s = score_retrieval([], ["x"])
    assert s.precision_at_k == 1.0
    assert s.recall_at_k == 1.0


def test_score_answer_faithful() -> None:
    a = score_answer(
        expected_answer="plan b has 500gb unlimited data",
        response_text="Plan B has 500GB unlimited data.",
        cited_sources=["plans.pdf:2"],
        retrieved_source_tags=["plans.pdf:2", "plans.pdf:1"],
    )
    assert a.faithful is True
    assert a.answer_recall > 0.5
    assert a.citations_supported == 1
    assert a.citations_total == 1


def test_score_answer_unfaithful_when_citation_invented() -> None:
    a = score_answer(
        expected_answer="plan b 500gb",
        response_text="Plan B has 500GB.",
        cited_sources=["plans.pdf:2", "imaginary.pdf:7"],
        retrieved_source_tags=["plans.pdf:2"],
    )
    assert a.faithful is False
    assert a.citations_supported == 1


def test_score_answer_no_expected_answer() -> None:
    a = score_answer(None, "some text", [], ["plans.pdf:2"])
    assert a.answer_recall == 1.0
    a2 = score_answer(None, "", [], ["plans.pdf:2"])
    assert a2.answer_recall == 0.0


# --- RAG benchmark end-to-end ------------------------------------------


class _CannedLLM(ILLMProvider):
    def __init__(self, payload: dict) -> None:
        self._json = json.dumps(payload)

    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(text=self._json, finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


@pytest.mark.asyncio
async def test_run_rag_benchmark_basic(tmp_faiss_index: str) -> None:
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8),
    )
    await retriever.index([
        Document(id="c1", content="Plan B has 500GB unlimited", metadata={"filename": "plans.md", "page": 0}),
        Document(id="c2", content="Plan A has 100GB", metadata={"filename": "plans.md", "page": 1}),
        Document(id="c3", content="Cookbook for biryani", metadata={"filename": "cookbook.md", "page": 0}),
    ])
    agent = ChatBotAgent(
        session=AgentSession(session_id="rag-bench-1"),
        llm=_CannedLLM({
            "response_text": "Plan B has 500GB unlimited data.",
            "language": "en",
            "sources_used": ["plans.md:0"],
            "confidence": "high",
            "action": "none",
        }),
        retriever=retriever,
        company_name="Acme",
    )
    samples = [
        RAGSample(id="r1", query="What does Plan B include?",
                  expected_chunks=["c1"], expected_answer="500GB unlimited"),
    ]
    result = await run_rag_benchmark(agent, samples)
    assert result.sample_count == 1
    assert result.recall_mean == 1.0
    assert result.faithfulness_rate == 1.0
    assert result.answer_recall_mean > 0.5
    assert result.per_sample[0].retrieved_ids[0] == "c1"


# --- Task-completion benchmark -----------------------------------------


class _RoundRobinLLM(ILLMProvider):
    """Per scenario we'll inject canned responses in order — first the
    interim ``continue`` turns, then a terminal ``close_positive`` to end."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: int = 0

    async def generate(self, messages, config) -> LLMResult:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return LLMResult(text=json.dumps(self._responses[idx]), finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        text = json.dumps(self._responses[idx])
        yield text


class _StaticSTT(ISTTProvider):
    """STT shim that returns the supplied text verbatim (we drive scripted text)."""

    def __init__(self) -> None:
        self.next_text = "placeholder"

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        return STTResult(text=self.next_text, confidence=0.99, language="hi", raw_response={})

    async def transcribe_stream(self, audio_stream, config) -> AsyncIterator[STTResult]:
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi"]


class _NoopTTS(ITTSProvider):
    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        return TTSResult(audio=b"", duration_ms=0.0, sample_rate=16000)

    async def synthesize_stream(self, text_stream, config) -> AsyncIterator[bytes]:
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language: str):
        return []


SCRIPT_YAML = {
    "agent_name": "Priya",
    "agent_role": "Eng",
    "company_name": "Acme",
    "language_default": "hi",
    "opening": "Namaste",
    "talking_points": [],
    "qualifying_questions": [],
    "objection_responses": {},
    "closing": {"positive": "Bye", "negative": "Bye"},
}
SLOT_YAML = """
interest_level: { type: enum, required: true, values: [hot, warm, cold] }
"""


@pytest.mark.asyncio
async def test_run_task_benchmark_completion() -> None:
    """Two-turn scripted scenario; the agent should land on close_positive."""

    scenarios = [TaskScenario(
        id="s1",
        user_turns=[
            TaskTurn(role="user", content="Yes I am interested"),
            TaskTurn(role="user", content="Sign me up"),
        ],
        expected_disposition="close_positive",
        required_slots={"interest_level": "hot"},
    )]

    async def agent_factory(scenario):
        stt = _StaticSTT()
        llm = _RoundRobinLLM([
            {"response_text": "Achha", "language": "hi", "action": "continue",
             "updated_slots": {"interest_level": "hot"}, "sentiment": "positive"},
            {"response_text": "Dhanyavaad!", "language": "hi", "action": "close_positive",
             "sentiment": "positive"},
        ])
        engine = PipelineEngine(
            stt, llm, _NoopTTS(),
            PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
        )
        agent = VoiceBotAgent(
            session=AgentSession(session_id=scenario.id),
            state_machine=AgentStateMachine(),
            slot_schema=SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML)),
            script=VoiceBotScript.from_campaign_yaml(SCRIPT_YAML),
            engine=engine,
        )
        agent._injected_stt = stt  # type: ignore[attr-defined]
        return agent

    async def turn_driver(agent, content):
        # Set the next STT result and drive a turn with a dummy audio buffer.
        agent._injected_stt.next_text = content  # type: ignore[attr-defined]
        return await agent.handle_turn(b"\x00\x00", _drop_sink)

    result = await run_task_benchmark(scenarios, agent_factory=agent_factory, turn_driver=turn_driver)
    assert result.scenario_count == 1
    assert result.disposition_match_rate == 1.0
    assert result.avg_slot_fill_rate == 1.0
    assert result.avg_slot_value_match_rate == 1.0
    assert result.completion_rate == 1.0
    assert result.per_scenario[0].actual_action == "close_positive"


@pytest.mark.asyncio
async def test_run_task_benchmark_partial_failure() -> None:
    """Slot not filled -> fill rate 0; disposition still matches."""

    scenarios = [TaskScenario(
        id="s1",
        user_turns=[TaskTurn(role="user", content="Sign me up")],
        expected_disposition="close_positive",
        required_slots={"interest_level": "hot"},
    )]

    async def agent_factory(scenario):
        stt = _StaticSTT()
        # Note: no updated_slots in the LLM response
        llm = _RoundRobinLLM([
            {"response_text": "Bye!", "language": "hi", "action": "close_positive"},
        ])
        engine = PipelineEngine(
            stt, llm, _NoopTTS(),
            PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
        )
        agent = VoiceBotAgent(
            session=AgentSession(session_id=scenario.id),
            state_machine=AgentStateMachine(),
            slot_schema=SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML)),
            script=VoiceBotScript.from_campaign_yaml(SCRIPT_YAML),
            engine=engine,
        )
        agent._injected_stt = stt  # type: ignore[attr-defined]
        return agent

    async def turn_driver(agent, content):
        agent._injected_stt.next_text = content  # type: ignore[attr-defined]
        return await agent.handle_turn(b"\x00\x00", _drop_sink)

    result = await run_task_benchmark(scenarios, agent_factory=agent_factory, turn_driver=turn_driver)
    assert result.disposition_match_rate == 1.0
    assert result.avg_slot_fill_rate == 0.0
    assert result.completion_rate == 0.0  # slots not filled -> incomplete


async def _drop_sink(_: bytes) -> None:
    pass
