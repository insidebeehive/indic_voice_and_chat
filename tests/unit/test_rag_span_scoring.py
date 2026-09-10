from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from src.benchmarks.datasets import RAGSample, load_rag_dataset
from src.benchmarks.rag_benchmark import (
    _top_dense_score,
    is_false_positive,
    run_retrieval_benchmark,
    score_retrieval_spans,
)
from src.interfaces.vector_store import Document, SearchResult
from src.rag.retriever import RetrievedChunk, _fuse


def _chunk(
    id_: str,
    content: str,
    score: float = 1.0,
    filename: Optional[str] = None,
    dense_score: Optional[float] = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        document=Document(id=id_, content=content, metadata={"filename": filename} if filename else {}),
        score=score,
        dense_score=dense_score,
    )


# --- score_retrieval_spans ------------------------------------------------


def test_span_found_in_exactly_one_chunk() -> None:
    retrieved = [
        _chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md"),
        _chunk("c2", "Withdrawals take 24 hours.", filename="05-withdrawals.md"),
    ]
    score = score_retrieval_spans(
        expected_spans=["credited within 15 minutes"],
        expected_files=["04-deposits.md"],
        retrieved=retrieved,
    )
    assert score.precision_at_k == pytest.approx(0.5)  # 1 of 2 retrieved chunks is relevant
    assert score.recall_at_k == 1.0
    assert score.reciprocal_rank == 1.0
    assert score.file_hit is True


def test_span_in_two_overlapping_chunks_recall_counts_once() -> None:
    # The chunker's overlap means the same span can legitimately land in two
    # neighbouring chunks -- recall must count the underlying span once, not
    # once per chunk it happens to appear in.
    retrieved = [
        _chunk("c1", "...paise kat gaye par balance nahi aaya turant...", filename="04-deposits.md"),
        _chunk("c2", "balance nahi aaya turant, contact support...", filename="04-deposits.md"),
    ]
    score = score_retrieval_spans(
        expected_spans=["balance nahi aaya turant"],
        expected_files=["04-deposits.md"],
        retrieved=retrieved,
    )
    assert score.recall_at_k == 1.0  # not 2.0 / 1 or otherwise double counted
    assert score.precision_at_k == 1.0  # both chunks are relevant


def test_whitespace_normalisation_newline_vs_space() -> None:
    retrieved = [_chunk("c1", "Deposits are credited\nwithin 15 minutes.")]
    score = score_retrieval_spans(
        expected_spans=["credited within 15 minutes"],
        expected_files=[],
        retrieved=retrieved,
    )
    assert score.recall_at_k == 1.0
    assert score.reciprocal_rank == 1.0


def test_two_spans_only_one_retrieved_recall_half() -> None:
    retrieved = [_chunk("c1", "Deposits are credited within 15 minutes.")]
    score = score_retrieval_spans(
        expected_spans=["credited within 15 minutes", "minimum deposit is 100 rupees"],
        expected_files=[],
        retrieved=retrieved,
    )
    assert score.recall_at_k == pytest.approx(0.5)


def test_file_hit_true_right_document_wrong_passage() -> None:
    retrieved = [_chunk("c1", "Some unrelated sentence.", filename="04-deposits.md")]
    score = score_retrieval_spans(
        expected_spans=["credited within 15 minutes"],
        expected_files=["04-deposits.md"],
        retrieved=retrieved,
    )
    assert score.recall_at_k == 0.0
    assert score.file_hit is True


def test_file_hit_false_wrong_document() -> None:
    retrieved = [_chunk("c1", "Some unrelated sentence.", filename="05-withdrawals.md")]
    score = score_retrieval_spans(
        expected_spans=["credited within 15 minutes"],
        expected_files=["04-deposits.md"],
        retrieved=retrieved,
    )
    assert score.file_hit is False


def test_file_hit_respects_top_k_match_past_k_does_not_count() -> None:
    # Finding 9: file_hit used to scan the FULL retrieved list while
    # precision/recall are top-k-scoped -- a filename match at rank 7 with
    # k=5 must not report file_hit=True, since top-k retrieval never
    # actually surfaced it.
    retrieved = [_chunk(f"c{i}", "irrelevant filler content") for i in range(6)]
    retrieved.append(_chunk("c7", "the actual match", filename="04-deposits.md"))  # rank 7
    score = score_retrieval_spans(
        expected_spans=["credited within 15 minutes"],
        expected_files=["04-deposits.md"],
        retrieved=retrieved,
        k=5,
    )
    assert score.file_hit is False
    # But the same match is visible with no k cap (full list) -- confirms
    # the fixture is doing what it claims, not just always False.
    uncapped = score_retrieval_spans(
        expected_spans=["credited within 15 minutes"],
        expected_files=["04-deposits.md"],
        retrieved=retrieved,
        k=None,
    )
    assert uncapped.file_hit is True


# --- is_false_positive / run_retrieval_benchmark exclusion ---------------


def test_is_false_positive_above_threshold() -> None:
    # is_false_positive reads the RAW dense_score, never the fused .score --
    # score=0.9 alone (no dense_score) must NOT trip it; dense_score=0.9 must.
    retrieved = [_chunk("c1", "irrelevant content", score=0.1, dense_score=0.9)]
    assert is_false_positive(retrieved, dense_score_threshold=0.2) is True


def test_is_false_positive_ignores_fused_score() -> None:
    # A chunk with a high FUSED score but no dense signal at all (BM25-only
    # hit, dense_score=None) must never count -- None is "no dense signal",
    # not "a low score" that happens to fail the threshold.
    retrieved = [_chunk("c1", "irrelevant content", score=0.95, dense_score=None)]
    assert is_false_positive(retrieved, dense_score_threshold=0.2) is False


def test_is_false_positive_empty_retrieval() -> None:
    assert is_false_positive([], dense_score_threshold=0.2) is False


class _FakeRetriever:
    """Minimal stand-in for HybridRetriever -- exposes just what
    search_combined / run_retrieval_benchmark need (async search + a config
    with similarity_threshold), so this test never builds a real FAISS/BM25
    index or embedder.

    Deliberately has NO ``list_all`` unless ``indexed_docs`` is given --
    _collect_indexed_filenames (src/benchmarks/rag_benchmark.py) treats a
    retriever with no list_all as "unindexed detection unavailable" and
    skips the check entirely, so every test that doesn't care about
    unindexed-sample handling keeps working unmodified against this fake.
    """

    def __init__(
        self,
        responses: dict[str, list[RetrievedChunk]],
        similarity_threshold: float = 0.0,
        strategy: str = "hybrid",
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        indexed_docs: Optional[list] = None,
    ) -> None:
        self._responses = responses
        self.config = type("Cfg", (), {
            "similarity_threshold": similarity_threshold,
            "strategy": strategy,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
        })()
        if indexed_docs is not None:
            self._indexed_docs = indexed_docs
            self.list_all = self._list_all  # only exists when opted in

    async def search(self, query: str, top_k=None, filters=None) -> list[RetrievedChunk]:
        return self._responses.get(query, [])

    def _list_all(self, max_chunks: int = 200):
        return self._indexed_docs[:max_chunks]


@pytest.mark.asyncio
async def test_unanswerable_sample_with_retrieval_counts_as_false_positive_and_excluded() -> None:
    samples = [
        RAGSample(
            id="ans-1", query="deposit query",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
        RAGSample(id="unans-1", query="unanswerable query", unanswerable=True),
    ]
    retriever = _FakeRetriever(
        responses={
            "deposit query": [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")],
            "unanswerable query": [_chunk("c2", "some tangential content", dense_score=0.9)],
        },
        similarity_threshold=0.0,
    )
    # fp_threshold must be explicitly supplied -- see Finding 1 tests below
    # for why there's no baked-in default any more.
    result = await run_retrieval_benchmark([retriever], samples, top_k=5, fp_threshold=0.5)

    assert result.answerable_count == 1
    assert result.unanswerable_count == 1
    assert result.false_positive_rate == 1.0
    # The unanswerable sample must not drag the means -- only the answerable
    # sample (a perfect retrieval) should feed precision/recall/mrr.
    assert result.recall_mean == 1.0
    assert result.precision_mean == 1.0
    assert result.mrr_mean == 1.0
    unans_row = next(r for r in result.per_sample if r.sample_id == "unans-1")
    assert unans_row.false_positive is True
    assert unans_row.recall_at_k == 0.0


@pytest.mark.asyncio
async def test_unanswerable_sample_no_retrieval_not_a_false_positive() -> None:
    samples = [RAGSample(id="unans-1", query="unanswerable query", unanswerable=True)]
    retriever = _FakeRetriever(responses={"unanswerable query": []})
    result = await run_retrieval_benchmark([retriever], samples, top_k=5, fp_threshold=0.5)
    assert result.false_positive_rate == 0.0


@pytest.mark.asyncio
async def test_fp_threshold_not_computed_by_default() -> None:
    # No baked-in default any more (Finding 1's follow-up): omitting
    # fp_threshold must leave false_positive_rate/fp_threshold as None
    # ("not computed"), not silently apply some guessed cutoff.
    samples = [RAGSample(id="unans-1", query="unanswerable query", unanswerable=True)]
    retriever = _FakeRetriever(
        responses={"unanswerable query": [_chunk("c1", "weak tangential match", dense_score=0.3)]},
    )
    result = await run_retrieval_benchmark([retriever], samples, top_k=5)
    assert result.false_positive_rate is None
    assert result.fp_threshold is None
    row = result.per_sample[0]
    assert row.false_positive is None


# --- Finding 1: false_positive_rate/fp_threshold must be driven off the RAW
# dense_score, never the fused/normalized score, and the dense-score
# distributions must be informative even when false_positive_rate cannot
# discriminate ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_fp_threshold_independent_of_retriever_similarity_threshold() -> None:
    # Reproduces the exact Finding-1 bug scenario: the retriever's own floor
    # is 0.0 (the shipped default), so every nonempty retrieval trivially
    # satisfies "score >= similarity_threshold". The false-positive check
    # must be driven off dense_score with an explicit, independent bar.
    samples = [RAGSample(id="unans-1", query="unanswerable query", unanswerable=True)]
    retriever = _FakeRetriever(
        responses={"unanswerable query": [_chunk("c1", "weak tangential match", dense_score=0.3)]},
        similarity_threshold=0.0,
    )
    # A bar above the weak match's dense_score doesn't flag it.
    loose_result = await run_retrieval_benchmark(
        [retriever], samples, top_k=5, fp_threshold=0.5)
    assert loose_result.false_positive_rate == 0.0
    assert loose_result.fp_threshold == 0.5

    # A stricter bar below it does.
    strict_result = await run_retrieval_benchmark(
        [retriever], samples, top_k=5, fp_threshold=0.2)
    assert strict_result.false_positive_rate == 1.0
    assert strict_result.fp_threshold == 0.2


@pytest.mark.asyncio
async def test_unanswerable_and_answerable_dense_score_mean_and_max() -> None:
    samples = [
        RAGSample(id="unans-1", query="q1", unanswerable=True),
        RAGSample(id="unans-2", query="q2", unanswerable=True),
        RAGSample(id="unans-3", query="q3", unanswerable=True),
        RAGSample(
            id="ans-1", query="q4",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
    ]
    retriever = _FakeRetriever(responses={
        "q1": [_chunk("c1", "x", dense_score=0.9)],
        "q2": [_chunk("c2", "y", dense_score=0.3)],
        "q3": [],  # empty retrieval -> top dense score treated as 0.0
        "q4": [_chunk(
            "c4", "Deposits are credited within 15 minutes.",
            filename="04-deposits.md", dense_score=0.95,
        )],
    })
    result = await run_retrieval_benchmark([retriever], samples, top_k=5)
    assert result.unanswerable_dense_score_max == pytest.approx(0.9)
    assert result.unanswerable_dense_score_mean == pytest.approx((0.9 + 0.3 + 0.0) / 3)
    assert result.answerable_dense_score_max == pytest.approx(0.95)
    assert result.answerable_dense_score_mean == pytest.approx(0.95)


def test_dense_score_stats_default_to_zero_with_no_samples() -> None:
    # Constructed directly rather than via run_retrieval_benchmark -- just
    # pins the dataclass defaults so a caller that never populates them
    # doesn't get a misleading nonzero value.
    from src.benchmarks.rag_benchmark import RetrievalRunResult
    from src.benchmarks.metrics import latency_stats

    r = RetrievalRunResult(
        sample_count=0, answerable_count=0, unanswerable_count=0,
        precision_mean=0.0, recall_mean=0.0, mrr_mean=0.0,
        file_hit_rate=0.0,
        latency=latency_stats([]),
    )
    assert r.unanswerable_dense_score_mean == 0.0
    assert r.unanswerable_dense_score_max == 0.0
    assert r.answerable_dense_score_mean == 0.0
    assert r.answerable_dense_score_max == 0.0
    assert r.false_positive_rate is None
    assert r.fp_threshold is None


# --- Regression guard for the whole Finding-1 class of bug: prove the
# tautology through the REAL _fuse, not a hand-built RetrievedChunk list. ----


def test_fused_top_score_floors_at_dense_weight_even_for_a_non_matching_query() -> None:
    # A "matching" query: the dense arm's best candidate has a genuinely high
    # raw cosine similarity.
    matching_dense = [SearchResult(document=Document(id="d1", content="a"), score=0.92)]
    matching_fused = _fuse(
        dense_results=matching_dense, bm25_results=[],
        dense_weight=0.7, bm25_weight=0.3,
    )

    # A "non-matching" query: the dense arm still returns its single best
    # candidate (HybridRetriever._dense_search always returns up to top_k
    # results, however weak), but its raw cosine similarity is near zero.
    nonmatching_dense = [SearchResult(document=Document(id="d2", content="b"), score=0.03)]
    nonmatching_fused = _fuse(
        dense_results=nonmatching_dense, bm25_results=[],
        dense_weight=0.7, bm25_weight=0.3,
    )

    # The bug: min-max normalization maps the sole candidate to 1.0 on BOTH
    # queries, so the FUSED top score is identically dense_weight regardless
    # of whether the query matched anything at all.
    assert matching_fused[0].score == pytest.approx(0.7)
    assert nonmatching_fused[0].score == pytest.approx(0.7)
    assert matching_fused[0].score == nonmatching_fused[0].score

    # The fix: the RAW dense_score-based statistic DOES discriminate --
    # exactly what the fused score above cannot do.
    assert _top_dense_score(matching_fused) == pytest.approx(0.92)
    assert _top_dense_score(nonmatching_fused) == pytest.approx(0.03)
    assert _top_dense_score(matching_fused) > _top_dense_score(nonmatching_fused)


def test_import_rag_benchmark_does_not_pull_in_agents_module() -> None:
    # Regression guard: src.benchmarks.rag_benchmark's TYPE_CHECKING-only
    # import of ChatBotAgent must stay TYPE_CHECKING-only. A future
    # module-level `import src.agents.x` here would silently drag the
    # agent/LLM module graph into run_retrieval_benchmark's cheap, LLM-free
    # sweep loop -- exactly what that import was structured to avoid (see
    # the module docstring / TYPE_CHECKING block at the top of
    # src/benchmarks/rag_benchmark.py). Run in a fresh subprocess -- this
    # test suite's own conftest/other tests routinely import src.agents.*
    # before this test would run in-process, which would make an in-process
    # check pass regardless of whether rag_benchmark itself is clean.
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [
            _sys.executable, "-c",
            "import sys; import src.benchmarks.rag_benchmark; "
            "leaked = [m for m in sys.modules if m.startswith('src.agents')]; "
            "print(','.join(leaked))",
        ],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr
    leaked = proc.stdout.strip()
    assert leaked == "", f"importing rag_benchmark pulled in src.agents.* modules: {leaked}"


# --- Finding 3: unindexed-sample detection + recall_by_tier ----------------


@pytest.mark.asyncio
async def test_unindexed_sample_excluded_from_means_and_reported() -> None:
    indexed_docs = [Document(id="d1", content="...", metadata={"filename": "04-deposits.md"})]
    samples = [
        RAGSample(
            id="ans-indexed", query="deposit query",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
        RAGSample(
            id="ans-unindexed", query="layout-only query",
            expected_spans=["some layout-specific text"],
            expected_files=["layout-42.md"],  # never ingested into this index
        ),
    ]
    retriever = _FakeRetriever(
        responses={
            "deposit query": [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")],
            "layout-only query": [_chunk("c2", "irrelevant filler", filename="04-deposits.md")],
        },
        indexed_docs=indexed_docs,
    )
    result = await run_retrieval_benchmark([retriever], samples, top_k=5)

    assert result.unindexed_sample_count == 1
    assert result.unindexed_sample_ids == ["ans-unindexed"]
    # Only the indexed sample (a perfect retrieval) feeds the means -- if the
    # unindexed sample leaked in as a miss, recall_mean would be 0.5, not 1.0.
    assert result.answerable_count == 1
    assert result.recall_mean == 1.0
    assert result.precision_mean == 1.0
    unindexed_row = next(r for r in result.per_sample if r.sample_id == "ans-unindexed")
    assert unindexed_row.unindexed is True
    assert unindexed_row.recall_at_k == 0.0


@pytest.mark.asyncio
async def test_unindexed_detection_skipped_when_retriever_lacks_list_all() -> None:
    # No indexed_docs given -> this fake has no list_all at all -- detection
    # must fail OPEN (skip the check), not treat every sample as unindexed.
    samples = [
        RAGSample(
            id="ans-1", query="deposit query",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
    ]
    retriever = _FakeRetriever(responses={
        "deposit query": [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")],
    })
    result = await run_retrieval_benchmark([retriever], samples, top_k=5)
    assert result.unindexed_sample_count == 0
    assert result.answerable_count == 1
    assert result.recall_mean == 1.0


@pytest.mark.asyncio
async def test_unindexed_detection_truncated_is_flagged_not_trusted_silently() -> None:
    # Finding 3: when list_all(max_chunks) comes back with exactly max_chunks
    # rows, the enumeration behind indexed_filenames may have been truncated
    # -- and with a deterministic ORDER BY, that truncation deterministically
    # excludes the same late-sorting files every run. A sample whose
    # expected_files fall in that excluded tail would otherwise be
    # misclassified "unindexed" and silently vanish from the means. The
    # result must say so rather than report clean numbers.
    indexed_docs = [
        Document(id="d1", content="...", metadata={"filename": "04-deposits.md"}),
    ]
    samples = [
        RAGSample(
            id="ans-1", query="deposit query",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
    ]
    retriever = _FakeRetriever(
        responses={
            "deposit query": [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")],
        },
        indexed_docs=indexed_docs,
    )
    # max_chunks == the exact number of docs the fake's list_all returns ->
    # indistinguishable from "the corpus is bigger and got truncated".
    result = await run_retrieval_benchmark([retriever], samples, top_k=5, max_chunks=1)
    assert result.unindexed_detection_truncated is True


@pytest.mark.asyncio
async def test_unindexed_detection_not_truncated_when_enumeration_is_under_the_cap() -> None:
    indexed_docs = [
        Document(id="d1", content="...", metadata={"filename": "04-deposits.md"}),
    ]
    samples = [
        RAGSample(
            id="ans-1", query="deposit query",
            expected_spans=["credited within 15 minutes"],
            expected_files=["04-deposits.md"],
        ),
    ]
    retriever = _FakeRetriever(
        responses={
            "deposit query": [_chunk("c1", "Deposits are credited within 15 minutes.", filename="04-deposits.md")],
        },
        indexed_docs=indexed_docs,
    )
    result = await run_retrieval_benchmark([retriever], samples, top_k=5, max_chunks=2000)
    assert result.unindexed_detection_truncated is False


@pytest.mark.asyncio
async def test_recall_by_tier_groups_independently() -> None:
    indexed_docs = [
        Document(id="d1", content="...", metadata={"filename": "04-deposits.md"}),
        Document(id="d2", content="...", metadata={"filename": "module-x.md"}),
    ]
    samples = [
        RAGSample(
            id="pack-1", query="pack query", tier="pack",
            expected_spans=["pack content here"],
            expected_files=["04-deposits.md"],
        ),
        RAGSample(
            id="module-1", query="module query", tier="module",
            expected_spans=["module content here"],
            expected_files=["module-x.md"],
        ),
    ]
    retriever = _FakeRetriever(
        responses={
            "pack query": [_chunk("c1", "pack content here", filename="04-deposits.md")],
            "module query": [_chunk("c2", "totally unrelated text", filename="module-x.md")],
        },
        indexed_docs=indexed_docs,
    )
    result = await run_retrieval_benchmark([retriever], samples, top_k=5)
    assert result.recall_by_tier["pack"] == 1.0
    assert result.recall_by_tier["module"] == 0.0


@pytest.mark.asyncio
async def test_tier_filter_restricts_to_matching_samples() -> None:
    indexed_docs = [Document(id="d1", content="...", metadata={"filename": "04-deposits.md"})]
    samples = [
        RAGSample(
            id="pack-1", query="pack query", tier="pack",
            expected_spans=["pack content here"], expected_files=["04-deposits.md"],
        ),
        RAGSample(
            id="module-1", query="module query", tier="module",
            expected_spans=["module content here"], expected_files=["04-deposits.md"],
        ),
    ]
    retriever = _FakeRetriever(
        responses={
            "pack query": [_chunk("c1", "pack content here", filename="04-deposits.md")],
            "module query": [_chunk("c2", "module content here", filename="04-deposits.md")],
        },
        indexed_docs=indexed_docs,
    )
    result = await run_retrieval_benchmark(
        [retriever], samples, top_k=5, tier_filter="pack")
    assert result.sample_count == 1
    assert [r.sample_id for r in result.per_sample] == ["pack-1"]


# --- Finding 5: config provenance -------------------------------------------


@pytest.mark.asyncio
async def test_result_carries_retrieval_config_provenance() -> None:
    samples = [RAGSample(id="unans-1", query="q", unanswerable=True)]
    retriever = _FakeRetriever(
        responses={"q": []},
        similarity_threshold=0.15,
        strategy="hybrid",
        bm25_weight=0.4,
        dense_weight=0.6,
    )
    result = await run_retrieval_benchmark(
        [retriever], samples, top_k=7, fp_threshold=0.33)
    assert result.strategy == "hybrid"
    assert result.top_k == 7
    assert result.bm25_weight == 0.4
    assert result.dense_weight == 0.6
    assert result.similarity_threshold == 0.15
    assert result.fp_threshold == 0.33


# --- load_rag_dataset round-trip -------------------------------------------


def test_load_rag_dataset_round_trips_new_fields() -> None:
    records = [{
        "id": "dep-001",
        "query": "paise kat gaye par balance nahi aaya",
        "lang": "hinglish",
        "intent": "deposit_not_credited",
        "expected_files": ["04-deposits.md"],
        "expected_spans": ["credited within 15 minutes"],
        "unanswerable": False,
    }]
    samples = load_rag_dataset(records)
    assert len(samples) == 1
    s = samples[0]
    assert s.id == "dep-001"
    assert s.lang == "hinglish"
    assert s.intent == "deposit_not_credited"
    assert s.expected_files == ["04-deposits.md"]
    assert s.expected_spans == ["credited within 15 minutes"]
    assert s.unanswerable is False
    assert s.expected_chunks == []  # not present in this record -> default


def test_load_rag_dataset_old_style_record_still_loads() -> None:
    records = [{"id": "r1", "query": "what plans are there?", "expected_chunks": ["c1", "c2"]}]
    samples = load_rag_dataset(records)
    assert len(samples) == 1
    s = samples[0]
    assert s.expected_chunks == ["c1", "c2"]
    assert s.expected_spans == []
    assert s.expected_files == []
    assert s.lang is None
    assert s.intent is None
    assert s.unanswerable is False
    assert s.tier is None


def test_load_rag_dataset_round_trips_tier() -> None:
    records = [{
        "id": "layout-1", "query": "q",
        "expected_files": ["layout-42.md"], "tier": "layout",
    }]
    samples = load_rag_dataset(records)
    assert samples[0].tier == "layout"
