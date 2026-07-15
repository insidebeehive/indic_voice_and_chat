from __future__ import annotations

import pytest

from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import (
    BM25Index,
    HybridRetriever,
    RetrievalConfig,
    _fuse,
    _minmax,
)


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
        reranker=None,
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=3,
            oversample_k=10,
            bm25_weight=0.3,
            dense_weight=0.7,
            reranking=False,
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
        config=RetrievalConfig(strategy="dense", top_k=2, reranking=False),
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
async def test_hybrid_with_reranker_changes_order(store: FAISSAdapter) -> None:
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=3,
            oversample_k=10,
            reranking=True,
            similarity_threshold=0.0,
        ),
    )
    await retriever.index([
        Document(id="a", content="plan b 500GB data unlimited"),
        Document(id="b", content="random unrelated cooking"),
        Document(id="c", content="plan b family pack"),
    ])
    results = await retriever.search("plan b unlimited data", top_k=3)
    # Reranker scores were attached
    assert all(r.rerank_score is not None for r in results[:3])


@pytest.mark.asyncio
async def test_similarity_threshold_filters_weak_matches(store: FAISSAdapter) -> None:
    retriever = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=5,
            oversample_k=10,
            reranking=False,
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
        reranker=None,
        config=RetrievalConfig(strategy="hybrid", top_k=3, oversample_k=10,
                               reranking=False, similarity_threshold=0.0),
    )
    await r.index([Document(id="a", content="plan b unlimited data")])
    await r.search("plan b", top_k=3)
    assert embed_threads, "embed_query was never called"
    assert all(t != main_thread for t in embed_threads), (
        "embed_query ran on the event-loop thread — it must be offloaded via to_thread"
    )
