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
    _minmax,
    retrieval_config_from_settings,
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
        similarity_threshold=0.4,
    )
    cfg = retrieval_config_from_settings(settings)
    assert cfg.strategy == "dense"
    assert cfg.top_k == 42
    assert cfg.bm25_weight == 0.11
    assert cfg.dense_weight == 0.89
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
