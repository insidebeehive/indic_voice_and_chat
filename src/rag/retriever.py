"""Hybrid retriever: dense (FAISS) + sparse (BM25) + cross-encoder rerank.

The retriever owns:
- A ``IVectorStore`` for dense recall (FAISS in this phase).
- An in-memory ``BM25Index`` for sparse recall.
- An ``IEmbedder`` for query encoding.
- An optional ``IReranker`` for the final stage.

Indexing is dual-write: every chunk is added to both the dense store and the
BM25 index in lockstep. The retriever exposes a single ``search(query)`` that:

1. Pulls top ``oversample_k`` candidates from each backend.
2. Combines them via weighted score fusion (dense_weight + bm25_weight) when
   ``strategy == "hybrid"``; otherwise uses just the configured backend.
3. Optionally reranks the top fused candidates with the cross-encoder.
4. Drops anything below ``similarity_threshold`` and trims to ``top_k``.

This keeps the API surface small (one ``index``, one ``search``, one
``delete``) while letting individual stages be swapped or mocked in tests.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

from src.interfaces.vector_store import (
    Document,
    IVectorStore,
    SearchResult,
)
from src.rag.embeddings import IEmbedder, IReranker, _tokenize


@dataclass
class RetrievalConfig:
    strategy: str = "hybrid"          # dense | hybrid
    top_k: int = 5
    oversample_k: int = 20            # candidates pulled from each backend before fusion
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    reranking: bool = True
    reranker_top_n: int = 3
    similarity_threshold: float = 0.0  # final post-rerank floor


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


# --- BM25 index ----------------------------------------------------------


class BM25Index:
    """In-memory BM25Okapi wrapper. Lazily rebuilds when the corpus mutates."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._order: list[str] = []  # stable id order for rank lookups
        self._bm25: Optional[BM25Okapi] = None

    def index(self, documents: list[Document]) -> None:
        for d in documents:
            if d.id in self._docs:
                # Replace existing document — drop from order list, then re-append.
                self._order = [i for i in self._order if i != d.id]
            self._docs[d.id] = d
            self._order.append(d.id)
        self._bm25 = None  # invalidate

    def delete(self, doc_ids: list[str]) -> int:
        before = len(self._docs)
        for i in doc_ids:
            self._docs.pop(i, None)
        self._order = [i for i in self._order if i in self._docs]
        if len(self._docs) != before:
            self._bm25 = None
        return before - len(self._docs)

    def count(self) -> int:
        return len(self._docs)

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        if not self._docs:
            return []
        if self._bm25 is None:
            corpus = [_tokenize(self._docs[i].content) for i in self._order]
            self._bm25 = BM25Okapi(corpus)
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(zip(self._order, scores), key=lambda x: x[1], reverse=True)
        out: list[tuple[Document, float]] = []
        for doc_id, score in ranked[:top_k]:
            if score <= 0:
                continue
            out.append((self._docs[doc_id], float(score)))
        return out


# --- Hybrid retriever ----------------------------------------------------


class HybridRetriever:
    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        bm25: Optional[BM25Index] = None,
        reranker: Optional[IReranker] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._embedder = embedder
        self._dense = vector_store
        self._bm25 = bm25 if bm25 is not None else BM25Index()
        self._reranker = reranker
        self._config = config or RetrievalConfig()

    @property
    def config(self) -> RetrievalConfig:
        return self._config

    async def index(self, chunks: list[Document]) -> int:
        """Embed the chunks (if needed) and dual-write into FAISS + BM25."""
        if not chunks:
            return 0

        # Backfill embeddings on chunks that arrive without them.
        # Embedders are sync (Gemini REST call); run in a thread to avoid blocking the event loop.
        missing = [c for c in chunks if c.embedding is None]
        if missing:
            vectors = await asyncio.to_thread(
                self._embedder.embed_documents, [c.content for c in missing]
            )
            for c, v in zip(missing, vectors):
                c.embedding = v

        # Dual-write. BM25 first so a FAISS failure doesn't leave us with
        # half-indexed state we can't roll back. (Both are still in-memory.)
        self._bm25.index(chunks)
        return await self._dense.index(chunks)

    async def delete(self, doc_ids: list[str]) -> int:
        self._bm25.delete(doc_ids)
        return await self._dense.delete(doc_ids)

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        cfg = self._config
        k = top_k or cfg.top_k

        if cfg.strategy == "dense":
            dense_results = await self._dense_search(query, cfg.oversample_k, filters)
            fused = [
                RetrievedChunk(document=r.document, score=r.score, dense_score=r.score)
                for r in dense_results
            ]
        elif cfg.strategy == "hybrid":
            dense_results, bm25_results = await asyncio.gather(
                self._dense_search(query, cfg.oversample_k, filters),
                asyncio.to_thread(self._bm25.search, query, cfg.oversample_k),
            )
            fused = _fuse(
                dense_results=dense_results,
                bm25_results=bm25_results,
                dense_weight=cfg.dense_weight,
                bm25_weight=cfg.bm25_weight,
            )
        else:
            raise ValueError(f"unknown retrieval strategy: {cfg.strategy}")

        # Rerank top candidates with the cross-encoder if available.
        if cfg.reranking and self._reranker is not None and fused:
            top_for_rerank = fused[: cfg.oversample_k]
            scores = await asyncio.to_thread(
                self._reranker.rerank,
                query,
                [c.document.content for c in top_for_rerank],
            )
            for c, s in zip(top_for_rerank, scores):
                c.rerank_score = float(s)
                c.score = float(s)
            top_for_rerank.sort(key=lambda c: c.score, reverse=True)
            fused = top_for_rerank + fused[cfg.oversample_k :]

        # Apply post-fusion threshold then trim.
        passing = [c for c in fused if c.score >= cfg.similarity_threshold]
        return passing[:k]

    async def _dense_search(
        self,
        query: str,
        k: int,
        filters: Optional[dict],
    ) -> list[SearchResult]:
        q_vec = self._embedder.embed_query(query)
        return await self._dense.search(q_vec, top_k=k, filters=filters)


# --- score fusion --------------------------------------------------------


def _fuse(
    dense_results: list[SearchResult],
    bm25_results: list[tuple[Document, float]],
    dense_weight: float,
    bm25_weight: float,
) -> list[RetrievedChunk]:
    """Min-max normalize each backend's scores then combine by weight."""
    dense_norm = _minmax({r.document.id: r.score for r in dense_results})
    bm25_norm = _minmax({d.id: s for d, s in bm25_results})

    by_id: dict[str, RetrievedChunk] = {}
    for r in dense_results:
        rc = RetrievedChunk(
            document=r.document,
            score=0.0,
            dense_score=r.score,
        )
        by_id[r.document.id] = rc
    for d, s in bm25_results:
        rc = by_id.get(d.id)
        if rc is None:
            rc = RetrievedChunk(document=d, score=0.0, bm25_score=s)
            by_id[d.id] = rc
        rc.bm25_score = s

    for doc_id, rc in by_id.items():
        d = dense_norm.get(doc_id, 0.0)
        b = bm25_norm.get(doc_id, 0.0)
        rc.score = d * dense_weight + b * bm25_weight

    fused = list(by_id.values())
    fused.sort(key=lambda rc: rc.score, reverse=True)
    return fused


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if math.isclose(hi, lo):
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}
