"""Tests for context-level recall -- src/benchmarks/rag_benchmark.py's
score_context_recall and its wiring into run_retrieval_benchmark.

The harness's existing recall_at_k scores every chunk the retriever
returned; the production agent (src/agents/chatbot.py) never sees that --
it sees build_rag_context(retrieved, max_chars=...).text, which stops
admitting whole chunks once the running total would exceed max_chars
(src/rag/context_builder.py, imported here, never modified). These tests
pin the resulting gap between recall_at_k and context_recall, and the
always-admit-the-first-chunk behaviour that makes an oversized chunk alone
still register as 1 chunk in context, not 0.
"""

from __future__ import annotations

from typing import Optional

import pytest

from src.benchmarks.datasets import RAGSample
from src.benchmarks.rag_benchmark import (
    ContextRecallScore,
    score_context_recall,
    score_retrieval_spans,
    run_retrieval_benchmark,
)
from src.interfaces.vector_store import Document
from src.rag.retriever import RetrievedChunk


def _chunk(id_: str, content: str, filename: Optional[str] = None) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(id=id_, content=content, metadata={"filename": filename} if filename else {}),
        score=1.0,
    )


class _FakeRetriever:
    """Minimal HybridRetriever stand-in -- same shape as the one in
    test_rag_span_scoring.py, kept local here so this file has no import
    dependency on that one."""

    def __init__(self, responses: dict[str, list[RetrievedChunk]]) -> None:
        self._responses = responses
        self.config = type("Cfg", (), {
            "similarity_threshold": 0.0, "strategy": "hybrid",
            "bm25_weight": 0.3, "dense_weight": 0.7,
        })()

    async def search(self, query: str, top_k=None, filters=None) -> list[RetrievedChunk]:
        return self._responses.get(query, [])


# --- score_context_recall (unit-level) -------------------------------------


def test_span_at_rank_1_survives_context_recall_is_1() -> None:
    retrieved = [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")]
    result = score_context_recall(
        expected_spans=["credited within 15 minutes"], retrieved=retrieved, max_chars=2000)
    assert result.context_recall == 1.0
    assert result.chunks_in_context == 1


def test_recall_at_k_hides_truncation_but_context_recall_catches_it() -> None:
    """THE central case: chunks 1-2 alone exhaust the char budget, so the
    chunk carrying the expected span (rank 3) is retrieved and scored by
    recall_at_k, but never reaches build_rag_context's output. recall_at_k
    reports a perfect 1.0; context_recall must report 0.0 -- that gap IS
    the finding this metric exists to surface (see the module docstring
    above and rag_benchmark.py's score_context_recall docstring)."""
    retrieved = [
        _chunk("c1", "x" * 40, filename="a"),
        _chunk("c2", "x" * 40, filename="a"),
        _chunk("c3", "the actual expected span text", filename="a"),
    ]
    span = "the actual expected span text"

    # max_chars=100: chunks 1 and 2 (each formatted block is 46 chars) fit
    # (46, then 48+46=94), but chunk 3's block (96+35=131) does not --
    # build_rag_context breaks before admitting it. See the module-level
    # arithmetic check this test's fixture sizes were derived from.
    spans_score = score_retrieval_spans(
        expected_spans=[span], expected_files=["a"], retrieved=retrieved, k=3)
    ctx_score = score_context_recall(expected_spans=[span], retrieved=retrieved, max_chars=100)

    assert spans_score.recall_at_k == 1.0  # rank-3 hit, fully visible to recall_at_k
    assert ctx_score.context_recall == 0.0  # ...but truncated out of the actual context
    assert ctx_score.chunks_in_context == 2  # only chunks 1-2 made it through the budget


def test_oversized_single_chunk_still_admitted_chunks_in_context_is_1() -> None:
    # Pins build_rag_context's always-admit-the-first-chunk rule (the "and
    # parts" guard in its break condition): a chunk far larger than the
    # whole budget is still let through alone, so chunks_in_context is 1,
    # never 0 -- this is what lets a single ~2000-char live-index chunk
    # consume an entire 2000-char budget by itself.
    huge_chunk = [_chunk("c1", "y" * 500, filename="a")]
    result = score_context_recall(expected_spans=[], retrieved=huge_chunk, max_chars=50)
    assert result.chunks_in_context == 1


def test_whitespace_normalisation_matches_span_scorer() -> None:
    # A span differing only by newline-vs-space must still count -- reuses
    # _normalize_ws, the SAME normaliser score_retrieval_spans uses, not a
    # second implementation that could silently diverge.
    retrieved = [_chunk("c1", "Deposits are credited\nwithin 15 minutes.")]
    result = score_context_recall(
        expected_spans=["credited within 15 minutes"], retrieved=retrieved, max_chars=2000)
    assert result.context_recall == 1.0


def test_context_recall_score_dataclass_shape() -> None:
    # Trivial construction check -- pins the field names or a future rename
    # of either field breaks callers silently instead of at import time.
    s = ContextRecallScore(context_recall=0.5, chunks_in_context=2)
    assert s.context_recall == 0.5
    assert s.chunks_in_context == 2


# --- run_retrieval_benchmark wiring (aggregate + per-sample) ---------------


@pytest.mark.asyncio
async def test_run_retrieval_benchmark_reports_context_recall_mean_and_chunks_in_context_mean() -> None:
    samples = [
        RAGSample(
            id="rank1", query="q1",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
        RAGSample(
            id="rank3-truncated", query="q2",
            expected_spans=["the actual expected span text"],
            expected_files=["a"],
        ),
    ]
    retriever = _FakeRetriever(responses={
        "q1": [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")],
        "q2": [
            _chunk("c1", "x" * 40, filename="a"),
            _chunk("c2", "x" * 40, filename="a"),
            _chunk("c3", "the actual expected span text", filename="a"),
        ],
    })
    result = await run_retrieval_benchmark(
        [retriever], samples, top_k=3, max_context_chars=100)

    # Sample 1: span at rank 1, comfortably inside the budget -> 1.0.
    # Sample 2: span at rank 3, truncated out -> 0.0. Mean of [1.0, 0.0].
    assert result.context_recall_mean == pytest.approx(0.5)
    assert result.chunks_in_context_mean == pytest.approx((1 + 2) / 2)
    assert result.max_context_chars == 100

    row1 = next(r for r in result.per_sample if r.sample_id == "rank1")
    row2 = next(r for r in result.per_sample if r.sample_id == "rank3-truncated")
    assert row1.context_recall == 1.0
    assert row1.chunks_in_context == 1
    assert row2.context_recall == 0.0
    assert row2.chunks_in_context == 2
    # recall_at_k stays reported and unaffected -- both metrics visible
    # side by side, which is the whole point.
    assert row2.recall_at_k == 1.0


@pytest.mark.asyncio
async def test_unanswerable_samples_excluded_from_context_recall_same_as_recall_at_k() -> None:
    samples = [
        RAGSample(
            id="ans-1", query="q1",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
        RAGSample(id="unans-1", query="q2", unanswerable=True),
    ]
    retriever = _FakeRetriever(responses={
        "q1": [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")],
        "q2": [_chunk("c2", "some tangential content")],
    })
    result = await run_retrieval_benchmark([retriever], samples, top_k=5, max_context_chars=2000)

    # If the unanswerable sample leaked into the mean (as a 0.0), the mean
    # would be 0.5 instead of the answerable-only 1.0.
    assert result.context_recall_mean == 1.0
    unans_row = next(r for r in result.per_sample if r.sample_id == "unans-1")
    assert unans_row.context_recall == 0.0  # placeholder, not a measurement
    assert unans_row.chunks_in_context == 0
