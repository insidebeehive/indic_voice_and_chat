from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pytest

from src.interfaces.vector_store import Document, SearchResult
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import (
    BM25Index,
    HybridRetriever,
    RetrievalConfig,
    _fuse,
    _fuse_rrf,
    _minmax,
    retrieval_config_from_settings,
    validate_retrieval_config,
)


# --- retrieval_config_from_settings --------------------------------------


def test_retrieval_config_from_settings_maps_fields_explicitly() -> None:
    """The bridge from the YAML-sourced settings object to the runtime
    dataclass must carry every mapped field through faithfully -- including
    similarity_threshold, which the bridge itself must NOT hardcode (pinning
    it to 0.0 belongs in config/default.yaml, not in this function)."""
    from src.config import RetrievalSettings

    settings = RetrievalSettings(
        strategy="dense",
        top_k=42,
        bm25_weight=0.11,
        dense_weight=0.89,
        rrf_k=17,
        similarity_threshold=0.4,
    )
    cfg = retrieval_config_from_settings(settings)
    assert cfg.strategy == "dense"
    assert cfg.top_k == 42
    assert cfg.bm25_weight == 0.11
    assert cfg.dense_weight == 0.89
    assert cfg.rrf_k == 17
    assert cfg.similarity_threshold == 0.4  # passed through as configured, not clamped


# --- BM25Index ----------------------------------------------------------


def test_bm25_index_round_trip() -> None:
    idx = BM25Index()
    idx.index([
        Document(id="a", content="plan b has 500GB unlimited data"),
        Document(id="b", content="plan a costs 199 per month"),
        Document(id="c", content="cooking recipes for biryani"),
    ])
    assert idx.count() == 3
    results = idx.search("plan unlimited data", top_k=3)
    ids = [d.id for d, _ in results]
    assert "a" in ids
    # BM25 should rank "a" highest (most overlap)
    assert ids[0] == "a"


def test_bm25_index_replace_existing() -> None:
    # rank_bm25's IDF returns 0 for terms in exactly half the corpus, so we
    # need >=3 docs to get reliable non-zero scores.
    idx = BM25Index()
    idx.index([
        Document(id="a", content="old generic content"),
        Document(id="b", content="cooking recipes for biryani"),
        Document(id="c", content="filler content"),
    ])
    idx.index([Document(id="a", content="new content with plan b unlimited")])
    # The replaced doc should be retrievable by its new term.
    results = idx.search("plan", top_k=3)
    assert results
    assert results[0][0].id == "a"
    assert "plan" in results[0][0].content
    assert idx.count() == 3


def test_bm25_index_delete() -> None:
    idx = BM25Index()
    idx.index([
        Document(id="a", content="alpha"),
        Document(id="b", content="beta"),
    ])
    n = idx.delete(["a", "missing-id"])
    assert n == 1
    assert idx.count() == 1


def test_bm25_index_search_empty() -> None:
    idx = BM25Index()
    assert idx.search("hello", top_k=5) == []
    idx.index([Document(id="a", content="hi")])
    assert idx.search("", top_k=5) == []


# --- score fusion --------------------------------------------------------


def test_minmax_normalizes_to_unit_range() -> None:
    out = _minmax({"a": 1.0, "b": 5.0, "c": 3.0})
    assert out["a"] == 0.0
    assert out["b"] == 1.0
    assert 0.0 < out["c"] < 1.0


def test_minmax_handles_constant_scores() -> None:
    out = _minmax({"a": 2.0, "b": 2.0})
    assert out == {"a": 1.0, "b": 1.0}


def test_minmax_empty_input() -> None:
    assert _minmax({}) == {}


# --- RRF fusion -----------------------------------------------------------


def test_rrf_surfaces_bm25_top_hit_that_minmax_fusion_buries_dep001() -> None:
    """Reproduces the real dep-001 support-query shape: a support query about
    a bank deduction not reflecting in the wallet balance. The dense arm
    clusters a handful of semantically-similar-but-wrong docs in a narrow
    0.63-0.73 score band (an embedding that barely discriminates); the BM25
    arm has the correct doc as a clear rank-1 lexical match.

    _minmax stretches that narrow dense band across the full 0-1 range,
    manufacturing a confident-looking gap that _fuse's 0.7 dense weight then
    multiplies -- burying the correct doc. RRF ignores score magnitude
    entirely and fuses by rank, so it wins here: the dense-top doc
    ("wrong-1") is entirely absent from BM25, while the correct doc
    ("dep-001") is BM25-rank-1 and still present (if dense-last) in the
    dense arm, so its contributions from both arms add up. This is not a
    guarantee against every dense/lexical disagreement -- just a fix for
    this measured shape.
    """
    dense_results = [
        SearchResult(document=Document(id="wrong-1", content="wrong doc 1"), score=0.73),
        SearchResult(document=Document(id="wrong-2", content="wrong doc 2"), score=0.72),
        SearchResult(document=Document(id="wrong-3", content="wrong doc 3"), score=0.71),
        SearchResult(document=Document(id="wrong-4", content="wrong doc 4"), score=0.70),
        SearchResult(document=Document(id="wrong-5", content="wrong doc 5"), score=0.69),
        SearchResult(document=Document(id="wrong-6", content="wrong doc 6"), score=0.68),
        SearchResult(document=Document(id="wrong-7", content="wrong doc 7"), score=0.67),
        SearchResult(
            document=Document(id="dep-001", content="bank deduction not reflected in wallet balance"),
            score=0.63,
        ),
    ]
    bm25_results = [
        (Document(id="dep-001", content="bank deduction not reflected in wallet balance"), 8.2),
        (Document(id="filler-1", content="filler"), 1.2),
        (Document(id="filler-2", content="filler"), 1.0),
    ]

    fused_hybrid = _fuse(dense_results, bm25_results, dense_weight=0.7, bm25_weight=0.3)
    assert fused_hybrid[0].document.id == "wrong-1"  # pins the failure: _fuse itself is unchanged

    fused_rrf = _fuse_rrf(dense_results, bm25_results, dense_weight=0.7, bm25_weight=0.3, rrf_k=60)
    assert fused_rrf[0].document.id == "dep-001"


def test_rrf_uses_rank_position_not_score_magnitude() -> None:
    docs = [Document(id="d1", content="d1"), Document(id="d2", content="d2"), Document(id="d3", content="d3")]
    dense_a = [SearchResult(document=docs[i], score=s) for i, s in enumerate([0.9, 0.8, 0.7])]
    dense_b = [SearchResult(document=docs[i], score=s) for i, s in enumerate([900.0, 2.0, 1.0])]

    bdocs = [Document(id="e1", content="e1"), Document(id="e2", content="e2"), Document(id="e3", content="e3")]
    bm25_a = [(bdocs[i], s) for i, s in enumerate([5.0, 3.0, 1.0])]
    bm25_b = [(bdocs[i], s) for i, s in enumerate([500.0, 3.1, 1.05])]

    fused_a = _fuse_rrf(dense_a, bm25_a, dense_weight=0.7, bm25_weight=0.3, rrf_k=60)
    fused_b = _fuse_rrf(dense_b, bm25_b, dense_weight=0.7, bm25_weight=0.3, rrf_k=60)

    assert [rc.document.id for rc in fused_a] == [rc.document.id for rc in fused_b]

    scores_a = {rc.document.id: rc.score for rc in fused_a}
    scores_b = {rc.document.id: rc.score for rc in fused_b}
    for doc_id in scores_a:
        assert scores_a[doc_id] == pytest.approx(scores_b[doc_id])


def test_rrf_scores_document_present_in_only_one_arm() -> None:
    dense_results = [
        SearchResult(document=Document(id="dense-only", content="x"), score=0.5),
    ]
    bm25_results = [
        (Document(id="bm25-first", content="y"), 5.0),
        (Document(id="bm25-only", content="z"), 3.0),  # bm25 rank 2 (1-based)
    ]
    fused = _fuse_rrf(dense_results, bm25_results, dense_weight=0.7, bm25_weight=0.3, rrf_k=60)
    by_id = {rc.document.id: rc for rc in fused}

    bm25_only = by_id["bm25-only"]
    assert bm25_only.score == pytest.approx(0.3 / (60 + 2))
    assert bm25_only.dense_score is None

    dense_only = by_id["dense-only"]
    assert dense_only.score == pytest.approx(0.7 / (60 + 1))
    assert dense_only.bm25_score is None


def test_rrf_larger_k_flattens_the_rank1_to_rankn_gap() -> None:
    dense_results = [
        SearchResult(document=Document(id="r1", content="r1"), score=0.9),
        SearchResult(document=Document(id="r2", content="r2"), score=0.8),
        SearchResult(document=Document(id="r3", content="r3"), score=0.7),
    ]
    fused_k1 = _fuse_rrf(dense_results, [], dense_weight=1.0, bm25_weight=0.3, rrf_k=1)
    fused_k60 = _fuse_rrf(dense_results, [], dense_weight=1.0, bm25_weight=0.3, rrf_k=60)

    scores_k1 = {rc.document.id: rc.score for rc in fused_k1}
    scores_k60 = {rc.document.id: rc.score for rc in fused_k60}

    ratio_k1 = scores_k1["r1"] / scores_k1["r3"]
    ratio_k60 = scores_k60["r1"] / scores_k60["r3"]
    assert ratio_k1 > ratio_k60

    gap_k1 = scores_k1["r1"] - scores_k1["r3"]
    gap_k60 = scores_k60["r1"] - scores_k60["r3"]
    assert gap_k60 < gap_k1

    assert [rc.document.id for rc in fused_k1] == ["r1", "r2", "r3"]
    assert [rc.document.id for rc in fused_k60] == ["r1", "r2", "r3"]


def test_rrf_populates_raw_arm_scores_not_rrf_partials() -> None:
    dense_results = [SearchResult(document=Document(id="a", content="a"), score=0.73)]
    bm25_results = [(Document(id="a", content="a"), 8.209)]
    fused = _fuse_rrf(dense_results, bm25_results, dense_weight=0.7, bm25_weight=0.3, rrf_k=60)
    rc = fused[0]
    assert rc.dense_score == pytest.approx(0.73)
    assert rc.bm25_score == pytest.approx(8.209)
    assert rc.score != pytest.approx(rc.dense_score)
    assert rc.score != pytest.approx(rc.bm25_score)


def test_rrf_weights_scale_each_arms_rank_contribution() -> None:
    dense_results = [
        SearchResult(document=Document(id="d1", content="d1"), score=0.9),
        SearchResult(document=Document(id="d2", content="d2"), score=0.5),
    ]
    bm25_results = [
        (Document(id="d2", content="d2"), 9.0),
        (Document(id="d1", content="d1"), 1.0),
    ]
    dense_only = _fuse_rrf(dense_results, bm25_results, dense_weight=1.0, bm25_weight=0.0, rrf_k=60)
    assert [rc.document.id for rc in dense_only] == ["d1", "d2"]

    bm25_only = _fuse_rrf(dense_results, bm25_results, dense_weight=0.0, bm25_weight=1.0, rrf_k=60)
    assert [rc.document.id for rc in bm25_only] == ["d2", "d1"]

    equal = _fuse_rrf(dense_results, bm25_results, dense_weight=1.0, bm25_weight=1.0, rrf_k=60)
    by_id = {rc.document.id: rc.score for rc in equal}
    # d1: dense rank 1, bm25 rank 2. d2: dense rank 2, bm25 rank 1.
    assert by_id["d1"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["d2"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_asymmetric_weights_scale_arms_independently() -> None:
    dense_results = [
        SearchResult(document=Document(id="both", content="both"), score=0.9),
        SearchResult(document=Document(id="dense-only", content="dense-only"), score=0.5),
    ]
    bm25_results = [
        (Document(id="both", content="both"), 9.0),
    ]

    asymmetric = _fuse_rrf(dense_results, bm25_results, dense_weight=1.0, bm25_weight=0.5, rrf_k=60)
    asym_by_id = {rc.document.id: rc.score for rc in asymmetric}
    # "both" is dense rank 1 and bm25 rank 1.
    assert asym_by_id["both"] == pytest.approx(1 / 61 + 0.5 / 61)

    equal = _fuse_rrf(dense_results, bm25_results, dense_weight=1.0, bm25_weight=1.0, rrf_k=60)
    equal_by_id = {rc.document.id: rc.score for rc in equal}
    assert asym_by_id["both"] != pytest.approx(equal_by_id["both"])

    # Raising bm25_weight should raise "both"'s score relative to a doc that
    # only ever appears in the dense arm.
    low_bm25 = _fuse_rrf(dense_results, bm25_results, dense_weight=1.0, bm25_weight=0.1, rrf_k=60)
    low_by_id = {rc.document.id: rc.score for rc in low_bm25}
    ratio_low = low_by_id["both"] / low_by_id["dense-only"]
    ratio_high = asym_by_id["both"] / asym_by_id["dense-only"]
    assert ratio_high > ratio_low


@pytest.mark.asyncio
async def test_search_rrf_strategy_returns_fused_chunks(store: FAISSAdapter) -> None:
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="rrf", top_k=3, oversample_k=10, similarity_threshold=0.0),
    )
    await retriever.index([
        Document(id="a", content="plan b has 500GB unlimited data"),
        Document(id="b", content="cooking recipes for biryani"),
        Document(id="c", content="plan a is the basic 100GB plan"),
    ])
    results = await retriever.search("unlimited data plan b", top_k=3)
    assert results
    assert any(r.dense_score is not None and r.bm25_score is not None for r in results), (
        "at least one returned chunk should have been recalled by both arms"
    )


@pytest.mark.asyncio
async def test_lazy_hydration_runs_for_rrf_strategy() -> None:
    docs = [
        Document(id="a", content="plan b has 500GB unlimited data"),
        Document(id="b", content="cooking recipes for biryani"),
        Document(id="c", content="plan a is the basic 100GB plan"),
    ]
    store = _CountingListDocumentsStore(docs)
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="rrf", top_k=3, oversample_k=10, similarity_threshold=0.0),
    )
    assert retriever._bm25.count() == 0

    results = await retriever.search("unlimited data plan b")
    assert results
    assert any(r.bm25_score is not None for r in results), (
        "first rrf search against an empty-BM25/populated-dense retriever "
        "must self-heal and return fused results"
    )
    assert store.list_calls == 1


@pytest.mark.asyncio
async def test_unknown_strategy_still_raises(store: FAISSAdapter) -> None:
    # Constructing RetrievalConfig(strategy="bm25") must NOT raise from
    # __post_init__ -- only rrf+nonzero-threshold and rrf_k<1 raise there.
    cfg = RetrievalConfig(strategy="bm25")
    retriever = HybridRetriever(embedder=HashEmbedder(dim=64), vector_store=store, config=cfg)
    with pytest.raises(ValueError, match="unknown retrieval strategy: bm25"):
        await retriever.search("anything")


def test_rrf_with_nonzero_similarity_threshold_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="similarity_threshold") as exc_info:
        RetrievalConfig(strategy="rrf", similarity_threshold=0.5)
    assert "rrf" in str(exc_info.value)


@pytest.mark.asyncio
async def test_rrf_with_threshold_mutated_after_construction_refused_at_search(store: FAISSAdapter) -> None:
    cfg = RetrievalConfig(strategy="rrf", top_k=3, oversample_k=10, similarity_threshold=0.0)
    retriever = HybridRetriever(embedder=HashEmbedder(dim=64), vector_store=store, config=cfg)
    # Mimics the CLI's pattern of mutating a config's fields after
    # construction (_build_retriever_for_cli), which bypasses __post_init__.
    retriever.config.similarity_threshold = 0.3
    with pytest.raises(ValueError, match="similarity_threshold"):
        await retriever.search("anything")


def test_hybrid_with_nonzero_similarity_threshold_is_still_allowed() -> None:
    cfg = RetrievalConfig(strategy="hybrid", similarity_threshold=0.99)
    assert cfg.similarity_threshold == 0.99


def test_rrf_k_below_one_is_refused() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(rrf_k=0)
    with pytest.raises(ValueError):
        RetrievalConfig(rrf_k=-60)


def test_validate_retrieval_config_is_the_public_entry_point_used_by_the_cli() -> None:
    # scripts/run_benchmark.py's _build_retriever_for_cli calls this directly
    # (after mutating a config's fields, bypassing __post_init__) -- exercise
    # that same call shape here rather than only indirectly via construction.
    valid = RetrievalConfig(strategy="rrf", similarity_threshold=0.0)
    validate_retrieval_config(valid)  # must not raise

    valid.similarity_threshold = 0.4  # CLI-style post-construction mutation
    with pytest.raises(ValueError, match="similarity_threshold"):
        validate_retrieval_config(valid)


def test_fuse_hybrid_minmax_scores_are_unchanged_golden() -> None:
    """Pins _fuse's exact numeric behavior as an explicit regression guard --
    _fuse must not change at all as part of adding RRF."""
    dense_results = [
        SearchResult(document=Document(id="x", content="x"), score=1.0),
        SearchResult(document=Document(id="y", content="y"), score=5.0),
        SearchResult(document=Document(id="z", content="z"), score=3.0),
    ]
    bm25_results = [
        (Document(id="z", content="z"), 10.0),
        (Document(id="w", content="w"), 2.0),
    ]
    fused = _fuse(dense_results, bm25_results, dense_weight=0.7, bm25_weight=0.3)
    scores = {rc.document.id: rc.score for rc in fused}
    assert scores["y"] == pytest.approx(0.7)
    assert scores["z"] == pytest.approx(0.65)
    assert scores["x"] == pytest.approx(0.0)
    assert scores["w"] == pytest.approx(0.0)
    assert [rc.document.id for rc in fused] == ["y", "z", "x", "w"]


# --- HybridRetriever ----------------------------------------------------


@pytest.fixture
def store(tmp_faiss_index: str) -> FAISSAdapter:
    return FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})


@pytest.fixture
def retriever(store: FAISSAdapter) -> HybridRetriever:
    return HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=3,
            oversample_k=10,
            bm25_weight=0.3,
            dense_weight=0.7,
            similarity_threshold=0.0,
        ),
    )


@pytest.mark.asyncio
async def test_index_backfills_embeddings(retriever: HybridRetriever) -> None:
    docs = [
        Document(id="a", content="plan b unlimited data"),
        Document(id="b", content="cooking recipes"),
    ]
    n = await retriever.index(docs)
    assert n == 2
    # Both chunks should have embeddings filled in by the embedder.
    assert all(d.embedding is not None for d in docs)


@pytest.mark.asyncio
async def test_hybrid_search_returns_relevant_first(retriever: HybridRetriever) -> None:
    await retriever.index([
        Document(id="a", content="plan b has 500GB unlimited data"),
        Document(id="b", content="cooking recipes for biryani"),
        Document(id="c", content="plan a is the basic 100GB plan"),
    ])
    results = await retriever.search("unlimited data plan b", top_k=3)
    assert results
    ids = [r.document.id for r in results]
    # The most-relevant doc should rank first.
    assert ids[0] == "a"
    # Cooking recipe must not be the top hit.
    assert ids[0] != "b"


@pytest.mark.asyncio
async def test_dense_only_strategy(store: FAISSAdapter) -> None:
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="dense", top_k=2),
    )
    await retriever.index([
        Document(id="a", content="plan b unlimited"),
        Document(id="b", content="cooking biryani"),
    ])
    results = await retriever.search("plan b", top_k=2)
    assert all(r.dense_score is not None for r in results)
    # In dense-only mode, BM25 score should be None (we never ran BM25).
    assert all(r.bm25_score is None for r in results)


@pytest.mark.asyncio
async def test_similarity_threshold_filters_weak_matches(store: FAISSAdapter) -> None:
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=5,
            oversample_k=10,
            similarity_threshold=0.99,  # nothing should pass
        ),
    )
    await retriever.index([
        Document(id="a", content="plan b unlimited"),
        Document(id="b", content="cooking biryani"),
    ])
    results = await retriever.search("plan b", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_delete_removes_from_both_backends(retriever: HybridRetriever, store: FAISSAdapter) -> None:
    await retriever.index([
        Document(id="a", content="alpha test content"),
        Document(id="b", content="beta test content"),
    ])
    assert await store.count() == 2

    n = await retriever.delete(["a"])
    assert n == 1
    assert await store.count() == 1
    # BM25 should also have lost the doc
    bm25_results = await retriever.search("alpha", top_k=5)
    assert not any(r.document.id == "a" for r in bm25_results)


@pytest.mark.asyncio
async def test_search_empty_index(retriever: HybridRetriever) -> None:
    assert await retriever.search("anything", top_k=3) == []


@pytest.mark.asyncio
async def test_query_embedding_runs_off_the_event_loop(store: FAISSAdapter) -> None:
    # Embedders are sync REST calls (Gemini). The QUERY path must run them in
    # a worker thread like the ingest path already does — a blocking call on
    # the loop freezes every concurrent session on a single-worker deployment
    # (this was the root cause of the 200-parallel-chat stress-test collapse).
    import threading

    main_thread = threading.get_ident()
    embed_threads: list[int] = []

    class _ThreadRecordingEmbedder(HashEmbedder):
        def embed_query(self, text: str):
            embed_threads.append(threading.get_ident())
            return super().embed_query(text)

    r = HybridRetriever(
        embedder=_ThreadRecordingEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10,
                               similarity_threshold=0.0),
    )
    await r.index([Document(id="a", content="plan b unlimited data")])
    await r.search("plan b", top_k=3)
    assert embed_threads, "embed_query was never called"
    assert all(t != main_thread for t in embed_threads), (
        "embed_query ran on the event-loop thread — it must be offloaded via to_thread"
    )


# --- hydrate_sparse_from_persistent --------------------------------------
#
# Reproduces the actual production bug: BM25Index is process-local while the
# dense store (FAISS file, or pgvector) survives a restart. A fresh process
# that reopens the persistent store but never called index() itself starts
# with an empty BM25 arm, so hybrid search silently fuses a populated dense
# result set against nothing and reports dense-only results with no error.


class _NoListDocumentsStore:
    """A dense store with no ``list_documents`` capability at all (not even
    the base class's default) -- e.g. a hand-rolled fake or a store that
    predates the capability. Exercises the ``getattr(..., None)`` branch in
    ``list_all_persistent``."""

    async def index(self, documents: list[Document]) -> int:
        return len(documents)

    async def search(self, query_embedding: list[float], top_k: int = 5, filters=None) -> list[SearchResult]:
        return []

    async def delete(self, doc_ids: list[str]) -> int:
        return 0

    async def count(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_hydrate_fixes_dense_only_bug_after_restart(tmp_faiss_index: str) -> None:
    """Exercises the explicit, manual call to hydrate_sparse_from_persistent
    (e.g. the benchmark CLI's usage) independently of search()'s own lazy
    self-heal, which is covered separately below. The "before" state here is
    read directly off the BM25 index rather than via search() -- calling
    search() first would itself trigger the lazy hydration and make this
    test indistinguishable from the lazy-hydration tests.
    """
    embedder = HashEmbedder(dim=64)
    docs = [
        Document(id="a", content="plan b has 500GB unlimited data"),
        Document(id="b", content="cooking recipes for biryani"),
        Document(id="c", content="plan a is the basic 100GB plan"),
    ]

    # Earlier process: ingest via the normal dual-write path into the
    # persistent FAISS file.
    seeder = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    await seeder.index(docs)

    # Fresh process: reopens the same persistent store. Its BM25 index was
    # never populated -- this is the bug.
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10),
    )
    assert retriever._bm25.count() == 0
    assert retriever._bm25.search("unlimited data plan b", top_k=3) == [], (
        "BM25 is empty pre-hydration -- confirms the bug this method fixes"
    )

    n = await retriever.hydrate_sparse_from_persistent()
    assert n == 3

    after = await retriever.search("unlimited data plan b")
    assert any(r.bm25_score is not None and r.bm25_score > 0 for r in after), (
        "post-hydration search must be genuinely fused, not still dense-only"
    )


@pytest.mark.asyncio
async def test_hydrate_returns_hydrated_chunk_count(tmp_faiss_index: str) -> None:
    embedder = HashEmbedder(dim=64)
    seeder = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    await seeder.index([
        Document(id="a", content="alpha content"),
        Document(id="b", content="beta content"),
        Document(id="c", content="gamma content"),
    ])

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    n = await retriever.hydrate_sparse_from_persistent()
    assert n == 3


@pytest.mark.asyncio
async def test_hydrate_is_idempotent(tmp_faiss_index: str) -> None:
    embedder = HashEmbedder(dim=64)
    seeder = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    await seeder.index([
        Document(id="a", content="alpha content"),
        Document(id="b", content="beta content"),
    ])

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    n1 = await retriever.hydrate_sparse_from_persistent()
    n2 = await retriever.hydrate_sparse_from_persistent()
    assert n1 == n2 == 2
    # The actual invariant: repeated hydration must not duplicate corpus
    # entries in the BM25 index (BM25Index.index() replaces by id).
    assert retriever._bm25.count() == 2


@pytest.mark.asyncio
async def test_hydrate_returns_zero_when_store_cannot_enumerate() -> None:
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=_NoListDocumentsStore(),
    )
    n = await retriever.hydrate_sparse_from_persistent()
    assert n == 0
    assert retriever._bm25.count() == 0


class _CappedListDocumentsStore:
    """A dense store whose ``list_documents`` just slices a fixed list --
    stands in for pgvector's ``LIMIT``-only query, which truncates silently
    once the corpus exceeds the cap (Finding 7)."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    async def index(self, documents: list[Document]) -> int:
        return len(documents)

    async def search(self, query_embedding: list[float], top_k: int = 5, filters=None) -> list[SearchResult]:
        return []

    async def delete(self, doc_ids: list[str]) -> int:
        return 0

    async def count(self) -> int:
        return len(self._docs)

    async def list_documents(self, limit: int = 2000) -> list[Document]:
        return self._docs[:limit]


@pytest.mark.asyncio
async def test_list_all_persistent_warns_when_result_count_equals_cap(caplog: pytest.LogCaptureFixture) -> None:
    docs = [Document(id=str(i), content=f"doc {i}") for i in range(5)]
    store = _CappedListDocumentsStore(docs)
    retriever = HybridRetriever(embedder=HashEmbedder(dim=64), vector_store=store)
    with caplog.at_level(logging.WARNING, logger="src.rag.retriever"):
        result = await retriever.list_all_persistent(max_chunks=5)
    assert len(result) == 5
    assert any("max_chunks" in rec.message for rec in caplog.records), (
        "returning exactly max_chunks results must warn that the corpus may "
        "have been truncated by the LIMIT"
    )


@pytest.mark.asyncio
async def test_list_all_persistent_no_warning_when_under_cap(caplog: pytest.LogCaptureFixture) -> None:
    docs = [Document(id=str(i), content=f"doc {i}") for i in range(3)]
    store = _CappedListDocumentsStore(docs)
    retriever = HybridRetriever(embedder=HashEmbedder(dim=64), vector_store=store)
    with caplog.at_level(logging.WARNING, logger="src.rag.retriever"):
        result = await retriever.list_all_persistent(max_chunks=5)
    assert len(result) == 3
    assert not any("max_chunks" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_hydrated_bm25_index_is_actually_queryable(tmp_faiss_index: str) -> None:
    """Count going non-zero isn't proof the corpus is searchable -- prove a
    term unique to one hydrated document actually scores non-zero."""
    embedder = HashEmbedder(dim=64)
    seeder = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    await seeder.index([
        Document(id="a", content="xylophone unique searchable term appears here"),
        Document(id="b", content="filler content one"),
        Document(id="c", content="filler content two"),
    ])

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    assert retriever._bm25.search("xylophone", top_k=3) == []  # empty pre-hydration

    await retriever.hydrate_sparse_from_persistent()

    results = retriever._bm25.search("xylophone", top_k=3)
    assert results
    assert results[0][0].id == "a"
    assert results[0][1] > 0


# --- lazy hydration on first search ---------------------------------------
#
# search() itself must self-heal an empty BM25 arm the first time it runs a
# hybrid search, without requiring the caller to remember to call
# hydrate_sparse_from_persistent() explicitly (that's the bug: nothing did,
# in production). These tests exercise search() directly.


class _CountingListDocumentsStore:
    """Fake dense store that records how many times list_documents is
    called, so tests can assert lazy hydration enumerates at most once (or
    not at all, when it shouldn't run)."""

    def __init__(self, docs: Optional[list[Document]] = None, raise_on_list: bool = False) -> None:
        self._docs = docs or []
        self.list_calls = 0
        self._raise_on_list = raise_on_list

    async def index(self, documents: list[Document]) -> int:
        return len(documents)

    async def search(self, query_embedding, top_k: int = 5, filters=None) -> list[SearchResult]:
        return [
            SearchResult(document=d, score=1.0 - i * 0.01)
            for i, d in enumerate(self._docs[:top_k])
        ]

    async def delete(self, doc_ids: list[str]) -> int:
        return 0

    async def count(self) -> int:
        return len(self._docs)

    async def list_documents(self, limit: int = 2000) -> list[Document]:
        self.list_calls += 1
        # Actually yield to the event loop -- an async def with no await
        # inside never suspends, so awaiting it lets the calling coroutine
        # run to completion before any concurrent task is scheduled. That
        # made test_lazy_hydration_concurrent_first_searches_enumerate_once
        # vacuous: search A would finish hydration before search B was ever
        # scheduled, so _sparse_hydrate_lock was never actually contended.
        await asyncio.sleep(0)
        if self._raise_on_list:
            raise RuntimeError("boom")
        return self._docs[:limit]


@pytest.mark.asyncio
async def test_lazy_hydration_fixes_dense_only_bug_on_first_search() -> None:
    """The bug, demonstrated end to end: a retriever whose dense store has
    documents but whose BM25 is empty must return genuinely fused results
    (non-None bm25_score) on the very FIRST search -- no explicit hydrate
    call anywhere in this test."""
    # rank_bm25's IDF returns 0 for terms in exactly half the corpus (see
    # test_bm25_index_replace_existing above) -- use >=3 docs for a reliable
    # non-zero score.
    docs = [
        Document(id="a", content="plan b has 500GB unlimited data"),
        Document(id="b", content="cooking recipes for biryani"),
        Document(id="c", content="plan a is the basic 100GB plan"),
    ]
    store = _CountingListDocumentsStore(docs)
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10),
    )
    assert retriever._bm25.count() == 0

    results = await retriever.search("unlimited data plan b")
    assert results
    assert any(r.bm25_score is not None for r in results), (
        "first hybrid search against an empty-BM25/populated-dense "
        "retriever must self-heal and return fused results"
    )
    assert store.list_calls == 1


@pytest.mark.asyncio
async def test_lazy_hydration_happens_at_most_once() -> None:
    store = _CountingListDocumentsStore([Document(id="a", content="alpha content plan")])
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10),
    )
    await retriever.search("alpha")
    await retriever.search("alpha")
    assert store.list_calls == 1


@pytest.mark.asyncio
async def test_lazy_hydration_still_runs_when_already_ingested_in_process() -> None:
    """A non-empty BM25 (from an in-process index() call) is NOT proof that
    BM25 mirrors the persistent corpus -- it may hold only a subset. So
    hydration must still run once on first search, even though count() > 0,
    and must not run again on a second search."""
    store = _CountingListDocumentsStore()
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10),
    )
    await retriever.index([Document(id="a", content="alpha content plan")])
    assert retriever._bm25.count() == 1

    await retriever.search("alpha")
    assert store.list_calls == 1, (
        "a retriever that already ingested in-process must still enumerate "
        "the persistent store once -- its BM25 may only hold a subset of "
        "the full corpus"
    )

    await retriever.search("alpha")
    assert store.list_calls == 1, "a second search must not enumerate again"


@pytest.mark.asyncio
async def test_lazy_hydration_after_partial_in_process_ingest_covers_full_corpus() -> None:
    """The exact bug from the task: a persistent store already holds several
    documents from an earlier process. This process ingests ONE new document
    (BM25 becomes non-empty but incomplete), then searches. The search must
    still hydrate the rest of the persistent corpus into BM25 -- not skip
    hydration just because BM25 was already non-empty -- so a query matching
    only an old, never-ingested-this-process document is still found via
    BM25 fusion."""
    old_docs = [
        Document(id="old-1", content="refund policy for prepaid plans"),
        Document(id="old-2", content="cooking recipes for biryani"),
        Document(id="old-3", content="unrelated filler about weather patterns"),
    ]
    store = _CountingListDocumentsStore(old_docs)
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=5, oversample_k=10),
    )

    # This process ingests one brand-new document -- BM25 is now non-empty
    # but covers only 1 of the 4 documents that will exist once the old,
    # persistent-only documents are counted.
    await retriever.index([Document(id="new-1", content="how to top up mobile balance")])
    assert retriever._bm25.count() == 1, "BM25 holds only what this process ingested so far"

    results = await retriever.search("refund policy prepaid")

    # count() alone could pass while fusion is still broken (e.g. if the new
    # doc's presence made count() > 0 look like "already complete") -- so
    # assert the union, AND that an old, persistent-only document actually
    # comes back with a real bm25_score.
    assert retriever._bm25.count() == len(old_docs) + 1, (
        "hydration must add the persistent documents on top of the "
        "in-process one, not skip because BM25 was already non-empty"
    )
    assert results, "search must return results for a query matching an old document"
    matches = [r for r in results if r.document.id == "old-1"]
    assert matches, "the old, never-ingested-this-process document must be retrievable"
    assert matches[0].bm25_score is not None, (
        "the old document must come back with a real bm25_score -- proof "
        "that BM25 fusion (not just dense recall) found it"
    )


@pytest.mark.asyncio
async def test_lazy_hydration_skipped_for_dense_strategy() -> None:
    store = _CountingListDocumentsStore([Document(id="a", content="alpha content plan")])
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="dense", top_k=3, oversample_k=10),
    )
    await retriever.search("alpha")
    assert store.list_calls == 0, "dense strategy never reads BM25 -- hydrating for it is pure waste"


@pytest.mark.asyncio
async def test_lazy_hydration_empty_store_does_not_retry() -> None:
    store = _CountingListDocumentsStore([])
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10),
    )
    await retriever.search("anything")
    await retriever.search("anything")
    assert store.list_calls == 1, (
        "an empty corpus must not turn into a database round-trip on every query"
    )


@pytest.mark.asyncio
async def test_lazy_hydration_failure_does_not_break_search() -> None:
    docs = [Document(id="a", content="alpha content plan")]
    store = _CountingListDocumentsStore(docs)
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10),
    )

    calls = {"n": 0}

    async def _boom(max_chunks: int = 2000) -> int:
        calls["n"] += 1
        raise RuntimeError("boom")

    retriever.hydrate_sparse_from_persistent = _boom  # type: ignore[method-assign]

    results = await retriever.search("alpha")
    assert results, "dense arm must still return results despite hydration failure"
    assert all(r.bm25_score is None for r in results)
    assert retriever._sparse_hydration_attempted is True

    # Must not retry the failed hydration on a subsequent search.
    await retriever.search("alpha")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_lazy_hydration_concurrent_first_searches_enumerate_once() -> None:
    """Concurrent turns share a retriever instance -- two simultaneous first
    searches must not both enumerate the persistent store."""
    store = _CountingListDocumentsStore([Document(id="a", content="alpha content plan")])
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10),
    )

    results_a, results_b = await asyncio.gather(
        retriever.search("alpha"),
        retriever.search("alpha"),
    )
    assert results_a and results_b
    assert store.list_calls == 1
