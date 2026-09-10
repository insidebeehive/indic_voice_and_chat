"""End-to-end benchmark suite.

Runs every benchmark slice (STT, TTS, latency, RAG) against mocked
providers, persists the results via ``SuiteRunner``, then exercises the
provider-recommendation matrix and ANOVA over multiple providers.

This is the dress rehearsal for a real run with live keys: we exercise
every export path, every aggregate metric, and the recommendation logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent
from src.benchmarks.datasets import RAGSample, STTSample, TTSSample
from src.benchmarks.export import (
    recommend_providers,
    write_latency_csv,
    write_rag_csv,
    write_stt_csv,
    write_tts_csv,
)
from src.benchmarks.latency_benchmark import run_latency_matrix
from src.benchmarks.rag_benchmark import run_rag_benchmark
from src.benchmarks.runner import SuiteResults, SuiteRunner
from src.benchmarks.stats import one_way_anova
from src.benchmarks.stt_benchmark import run_stt_benchmark
from src.benchmarks.tts_benchmark import join_mos_scores, run_tts_benchmark
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMResult
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import HybridRetriever, RetrievalConfig


# --- Fakes ---------------------------------------------------------------


class AccurateSTT(ISTTProvider):
    """Returns the reference verbatim — useful as 'best provider' control."""

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        return STTResult(text=audio.decode("utf-8"), confidence=0.99, language=config.language)

    async def transcribe_stream(self, audio_stream, config) -> AsyncIterator[STTResult]:
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi-IN", "en-IN"]


class NoisySTT(ISTTProvider):
    """Drops every other word — useful as 'worse provider' for ANOVA."""

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        words = audio.decode("utf-8").split()
        kept = words[::2]
        return STTResult(text=" ".join(kept), confidence=0.6, language=config.language)

    async def transcribe_stream(self, audio_stream, config) -> AsyncIterator[STTResult]:
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi-IN", "en-IN"]


class ConstantTTS(ITTSProvider):
    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        return TTSResult(audio=b"\x00\x00" * 100, duration_ms=80.0 * len(text), sample_rate=config.sample_rate)

    async def synthesize_stream(self, text_stream, config) -> AsyncIterator[bytes]:
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language: str):
        return []


class FastLLM(ILLMProvider):
    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(text='{"response_text":"Plan B has 500GB unlimited data.","language":"en","sources_used":["plans.md:0"],"confidence":"high","action":"none"}', finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        for tok in ('{"response_text":"', 'Plan B has 500GB.","language":"en","action":"continue"}'):
            yield tok


# --- The big test --------------------------------------------------------


@pytest.mark.asyncio
async def test_full_benchmark_suite_end_to_end(tmp_path: Path) -> None:
    # 1. STT — two providers run against the same dataset.
    stt_samples = [
        STTSample(id="s1", transcript="namaste dosto kaise hain", language="hi-IN", audio_bytes=b"namaste dosto kaise hain"),
        STTSample(id="s2", transcript="hello world how are you", language="en-IN", audio_bytes=b"hello world how are you"),
        STTSample(id="s3", transcript="mixed sentence with two languages", language="hi-IN", code_switch=True,
                  audio_bytes=b"mixed sentence with two languages"),
    ]
    stt_good = await run_stt_benchmark("sarvam", AccurateSTT(), stt_samples)
    stt_bad = await run_stt_benchmark("groq", NoisySTT(), stt_samples)

    assert stt_good.overall.wer_mean == 0.0
    assert stt_bad.overall.wer_mean > 0.0
    assert stt_good.code_switch is not None
    assert stt_bad.code_switch is not None
    # Per-sample WERs feed ANOVA — assert the providers are statistically distinct.
    good_wers = [row.wer for row in stt_good.per_sample]
    bad_wers = [row.wer for row in stt_bad.per_sample]
    # Pad with replicas so n is large enough for the F table.
    anova = one_way_anova([good_wers * 6, bad_wers * 6])
    assert anova.significance in ("<0.05", "<0.001")

    # 2. TTS run + join external MOS scores.
    tts_samples = [
        TTSSample(id="t1", text="Namaste dosto", language="hi-IN"),
        TTSSample(id="t2", text="Hello world how are you", language="en-IN"),
    ]
    tts_run = await run_tts_benchmark("sarvam", ConstantTTS(), tts_samples)
    join_mos_scores(tts_run, {"t1": 4.3, "t2": 4.1})
    assert tts_run.mos_mean == pytest.approx(4.2)
    assert tts_run.sample_rate_violations == 0

    # 3. Latency matrix across two LLM choices.
    latency_matrix = await run_latency_matrix(
        combos=[
            {"stt": "sarvam", "llm": "groq", "tts": "sarvam"},
            {"stt": "sarvam", "llm": "gemini", "tts": "sarvam"},
        ],
        make_stt=lambda name: AccurateSTT(),
        make_llm=lambda name: FastLLM(),
        make_tts=lambda name: ConstantTTS(),
        audio_samples=[b"hello world", b"namaste dosto"],
    )
    assert len(latency_matrix.results) == 2

    # 4. RAG benchmark with a wired retriever + chatbot.
    store = FAISSAdapter({"embedding_dim": 64, "index_path": str(tmp_path / "fbench")})
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=8),
    )
    await retriever.index([
        Document(id="c1", content="Plan B has 500GB unlimited data", metadata={"filename": "plans.md", "page": 0}),
        Document(id="c2", content="Plan A is 100GB basic plan", metadata={"filename": "plans.md", "page": 1}),
        Document(id="c3", content="Biryani recipes cookbook", metadata={"filename": "cookbook.md", "page": 0}),
    ])
    agent = ChatBotAgent(
        session=AgentSession(session_id="rag-suite-1"),
        llm=FastLLM(),
        retriever=retriever,
        company_name="Acme",
    )
    rag_samples = [
        RAGSample(id="r1", query="What is plan B?",
                  expected_chunks=["c1"], expected_answer="500GB unlimited"),
    ]
    rag_run = await run_rag_benchmark(agent, rag_samples)
    assert rag_run.recall_mean == 1.0
    assert rag_run.faithfulness_rate == 1.0

    # 5. Persist all of it through SuiteRunner.
    runner = SuiteRunner()
    record = await runner.record(
        name="phase6-smoke",
        description="end-to-end benchmark suite",
        pipeline_config={"stt": "sarvam", "llm": "groq", "tts": "sarvam"},
        language="hi-IN",
        dataset="inline",
        results=SuiteResults(
            stt=[stt_good, stt_bad],
            tts=[tts_run],
            latency=latency_matrix,
            rag=rag_run,
        ),
    )
    assert record.id.startswith("br_")
    payload = record.results
    assert len(payload["stt"]) == 2
    assert payload["latency"]["results"][0]["combo"]["llm"] == "groq"

    # 6. Recommendation matrix prefers the accurate STT + lowest-latency combo.
    rec = recommend_providers(
        [stt_good, stt_bad],
        [tts_run],
        latency=latency_matrix,
    )
    assert rec["stt_recommendation"] == "sarvam"
    assert rec["latency_combo_recommendation"]["stt"] == "sarvam"
    # The two combos shared the same providers under fakes; either is fine,
    # we just check the structure is correct.
    assert "tts_recommendation" in rec

    # 7. CSV exports — all of them, to tmp_path
    out_dir = tmp_path / "out"
    write_stt_csv(out_dir / "stt_good.csv", stt_good)
    write_stt_csv(out_dir / "stt_bad.csv", stt_bad)
    write_tts_csv(out_dir / "tts.csv", tts_run)
    write_latency_csv(out_dir / "latency.csv", latency_matrix)
    write_rag_csv(out_dir / "rag.csv", rag_run)
    for fname in ("stt_good.csv", "stt_bad.csv", "tts.csv", "latency.csv", "rag.csv"):
        assert (out_dir / fname).exists(), f"missing {fname}"
        assert (out_dir / fname).read_text().strip()  # non-empty
