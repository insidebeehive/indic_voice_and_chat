"""Hybrid retriever: dense (FAISS) + sparse (BM25) fusion.

The retriever owns:
- A ``IVectorStore`` for dense recall (FAISS in this phase).
- An in-memory ``BM25Index`` for sparse recall.
- An ``IEmbedder`` for query encoding.

Indexing is dual-write: every chunk is added to both the dense store and the
BM25 index in lockstep. The retriever exposes a single ``search(query)`` that:

1. Pulls top ``oversample_k`` candidates from each backend.
2. Combines them via weighted score fusion (dense_weight + bm25_weight) when
   ``strategy == "hybrid"``; otherwise uses just the configured backend.
3. Drops anything below ``similarity_threshold`` and trims to ``top_k``.

This keeps the API surface small (one ``index``, one ``search``, one
``delete``) while letting individual stages be swapped or mocked in tests.

Process-local vs. durable state: BM25Index lives only in this process's
memory. The dense store (FAISS file, pgvector) is durable and shared across
workers/restarts. ``search()`` self-heals the sparse arm the first time a
``hybrid``-strategy retriever runs a search, regardless of whether BM25 is
already non-empty from an in-process ``index()`` call -- an in-process
``index()`` only covers what this process ingested, a subset of (not a
mirror of) the persistent corpus -- see ``HybridRetriever`` below for the
mechanics. ``index()`` remains the eager, dual-write path used by ingestion;
nothing about the self-heal changes it.

There used to be a fourth, cross-encoder reranking stage here. It was removed
(see ``retrieval_config_from_settings`` below and Phase 0 cleanup) because no
production wiring ever instantiated a reranker for it — it was dead code.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from rank_bm25 import BM25Okapi

from src.interfaces.vector_store import (
    Document,
    IVectorStore,
    SearchResult,
)
from src.rag.embeddings import IEmbedder, _tokenize

if TYPE_CHECKING:
    # Type-checking only to keep this module importable without pulling in
    # the settings layer — NOT because a runtime import would be circular.
    # It wouldn't: src/config.py imports no src.* module at all. Consequence
    # to know about: the annotation on retrieval_config_from_settings below
    # is unresolvable at runtime, so typing.get_type_hints() on it (or any
    # decorator that calls it — @validate_call, FastAPI Depends) raises
    # NameError. Move this import to module scope if that day comes.
    from src.config import RetrievalSettings

log = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    strategy: str = "hybrid"          # dense | hybrid
    top_k: int = 5
    oversample_k: int = 20            # candidates pulled from each backend before fusion
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    similarity_threshold: float = 0.0  # final post-fusion floor


def retrieval_config_from_settings(settings: RetrievalSettings) -> RetrievalConfig:
    """Bridge the YAML-sourced ``RetrievalSettings`` (src/config.py, itself
    populated from ``config/default.yaml``'s ``rag.retrieval`` block) onto
    this module's runtime ``RetrievalConfig`` dataclass — the one
    ``HybridRetriever`` actually reads.

    Previously the two wiring sites in src/bootstrap.py constructed
    ``RetrievalConfig()`` with no arguments at all, so the entire YAML block
    was inert. Field mapping is explicit (no ``**settings.model_dump()``
    splat) so adding a field to ``RetrievalSettings`` requires a deliberate
    decision here about whether/how it reaches the runtime dataclass, rather
    than silently flowing through under an assumption the two classes stay
    in lockstep.
    """
    return RetrievalConfig(
        strategy=settings.strategy,
        top_k=settings.top_k,
        bm25_weight=settings.bm25_weight,
        dense_weight=settings.dense_weight,
        similarity_threshold=settings.similarity_threshold,
        # oversample_k has no YAML knob (yet) -- keep the dataclass default.
    )


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None


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
    """Dense (durable) + sparse (process-local) hybrid retriever.

    Invariant that matters in production: ``self._bm25`` is in-memory and
    exists only for the lifetime of this process. ``self._dense`` is durable
    -- it survives restarts and is shared across workers. A freshly
    constructed retriever over an already-populated dense store therefore
    starts with a BM25 arm that is empty even though the corpus is not.

    ``index()`` is the eager path: every ``index()`` call dual-writes into
    both backends in lockstep. That only covers what THIS process ingested,
    though -- it says nothing about chunks a different process wrote to the
    persistent store earlier, so a non-empty BM25 index is not on its own
    proof that BM25 mirrors the full corpus.

    ``search()`` is the self-healing path: on the first ``hybrid`` search
    against an instance that hasn't yet attempted hydration, it hydrates
    BM25 from the persistent store (see ``hydrate_sparse_from_persistent``)
    before running the query -- regardless of whether BM25 is already
    non-empty from an in-process ``index()`` call -- so neither a
    search-only process nor an ingest-then-search process silently degrades
    to dense-only (or partially-fused) forever. The hydration attempt
    happens at most once per instance -- including when it finds zero
    chunks -- so an empty corpus does not pay a database round-trip on every
    subsequent query. See ``_ensure_sparse_hydrated``.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        bm25: Optional[BM25Index] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._embedder = embedder
        self._dense = vector_store
        self._bm25 = bm25 if bm25 is not None else BM25Index()
        self._config = config or RetrievalConfig()
        # Lazy self-heal bookkeeping for search() -- see _ensure_sparse_hydrated.
        # Attempted (not "succeeded"): set True even on failure or a
        # zero-chunk result, so we never retry per-query.
        self._sparse_hydration_attempted = False
        # Created lazily (on first use, inside a running loop) rather than
        # here in __init__ -- __init__ can run outside a running event loop
        # (the bootstrap factories do exactly that), and while asyncio.Lock
        # no longer hard-binds to a loop at construction time on the Python
        # versions this project targets, creating it lazily sidesteps the
        # question entirely rather than relying on that version detail.
        self._sparse_hydrate_lock: Optional[asyncio.Lock] = None

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

    def list_all(self, max_chunks: int = 200) -> list[Document]:
        """Return all indexed chunks (drawn from the in-memory BM25 index)."""
        return list(self._bm25._docs.values())[:max_chunks]

    async def list_all_persistent(self, max_chunks: int = 2000) -> list[Document]:
        """Enumerate chunks from the PERSISTENT store, falling back to the
        process-local BM25 index.

        ``list_all`` only sees chunks indexed by *this* process; the voicebot's
        one-shot KB dump needs everything a tenant ever ingested, in any worker.
        """
        lister = getattr(self._dense, "list_documents", None)
        if lister is not None:
            try:
                docs = await lister(max_chunks)
            except Exception:
                log.exception(
                    "list_all_persistent: persistent enumeration failed; "
                    "falling back to in-memory BM25 index",
                    extra={"store": type(self._dense).__name__},
                )
            else:
                if docs:
                    if len(docs) >= max_chunks:
                        # A returned count equal to the cap is indistinguishable
                        # from "the corpus has exactly max_chunks rows" -- but
                        # far more often it means the corpus is BIGGER than
                        # max_chunks and list_documents' LIMIT silently
                        # truncated it. That matters here specifically because
                        # the BM25 arm built from this enumeration would then
                        # cover less of the corpus than the dense arm (which
                        # queries per-request with no such cap), skewing hybrid
                        # fusion in a way that looks like a config effect.
                        log.warning(
                            "list_all_persistent: returned exactly max_chunks=%d "
                            "chunks -- the corpus may be larger and got truncated "
                            "by the LIMIT. Pass a higher max_chunks if so.",
                            max_chunks,
                            extra={"store": type(self._dense).__name__},
                        )
                    return docs
                log.info(
                    "list_all_persistent: persistent store returned no chunks",
                    extra={"store": type(self._dense).__name__},
                )
        return self.list_all(max_chunks)

    async def hydrate_sparse_from_persistent(self, max_chunks: int = 2000) -> int:
        """Populate the (process-local) BM25 index from the PERSISTENT dense
        store, without re-embedding anything or writing back to the dense
        store.

        Why this exists: BM25Index lives only in this process's memory, while
        the dense store (pgvector, or a FAISS index file shared across
        workers) survives process restarts. A process that only *searches* --
        one that never called ``index()`` itself -- starts with an empty BM25
        arm, so ``search(strategy="hybrid")`` silently fuses a populated dense
        result set against nothing and returns dense-only results with no
        error or warning. Call this once after construction, before serving
        queries, to close that gap.

        Delegates enumeration to ``list_all_persistent`` -- it already
        handles the ``list_documents`` capability check and the in-memory
        fallback (see its docstring for both). From there, only
        ``self._bm25.index(...)`` is called: ``HybridRetriever.index()``
        would be the wrong tool here since it also embeds and dual-writes to
        the dense store, which would be expensive and would rewrite rows
        that are already there.

        Idempotent: ``BM25Index.index()`` replaces existing entries by
        document id, so calling this more than once (e.g. once per CLI
        invocation) does not duplicate corpus entries or skew BM25 term
        statistics.

        Never raises: a store that can't enumerate is handled by
        ``list_all_persistent`` itself (logs and falls back), matching its
        defensive posture -- this method returns 0 rather than propagating.

        NOTE: application startup (src/main.py) does NOT call this eagerly.
        Its seeder only re-ingests the bundled KB pack, which repopulates
        BM25 for those files via the normal index() dual-write. Documents a
        tenant uploaded through /knowledge/ingest in an earlier process live
        in pgvector but would be invisible to a fresh process's BM25 --
        except ``HybridRetriever.search()`` calls this method itself (via
        ``_ensure_sparse_hydrated``) the first time it runs a ``hybrid``
        search against an instance whose BM25 is still empty, so the gap is
        closed lazily on first use instead of requiring a startup hook or an
        explicit call by the caller.
        """
        docs = await self.list_all_persistent(max_chunks)
        if not docs:
            return 0
        self._bm25.index(docs)
        return len(docs)

    async def delete(self, doc_ids: list[str]) -> int:
        self._bm25.delete(doc_ids)
        return await self._dense.delete(doc_ids)

    async def _ensure_sparse_hydrated(self) -> None:
        """Self-heal the process-local BM25 arm before a ``hybrid`` search,
        at most once per instance.

        Called from ``search()`` only on the ``hybrid`` path -- ``dense``
        never reads BM25, so hydrating for it would be pure waste.

        Invariant: hydration runs at most once per instance, gated ONLY by
        ``_sparse_hydration_attempted`` -- NOT by ``self._bm25.count()``. A
        non-empty BM25 does not mean a complete one: an in-process ``index()``
        call (e.g. via ``/knowledge/ingest``) makes BM25 non-empty while it
        still holds only the chunks THIS process wrote, a strict subset of
        the persistent corpus. Skipping hydration in that case would leave
        the older, persistent-only chunks permanently invisible to BM25 --
        exactly the dense-only-fusion bug this method exists to close.
        Hydrating over an already-populated index is safe and cheap: see
        ``hydrate_sparse_from_persistent``'s docstring -- ``BM25Index.index()``
        replaces existing entries by id, so re-adding in-process-ingested
        documents alongside the rest of the corpus does not duplicate them.

        SCALE-OUT CONSTRAINT (not live today, becomes live with more than one
        worker process per dense store -- ``Dockerfile``'s CMD runs a single
        uvicorn with no ``--workers``, so there is exactly one BM25 arm right
        now): this method has no TTL and no cross-process invalidation, so it
        can resurrect a chunk another process deleted. Sequence: worker A
        never hydrates (no hybrid search yet); worker B hydrates the whole
        corpus on its first hybrid search; a document delete then runs on
        worker A (``src/api/knowledge.py``'s delete route calls
        ``HybridRetriever.delete()``, which mutates only worker A's OWN
        process-local BM25 index, plus the shared dense store). Worker B's
        BM25 index still holds the deleted chunk -- indefinitely, since
        hydration never re-runs after the first attempt and nothing tells
        worker B the corpus changed. Fusion min-max-normalizes each arm's raw
        scores before weighting (see ``_fuse``), so a BM25-only hit's fused
        score is its normalized BM25 score times ``bm25_weight`` (0.3
        shipped) -- up to 0.3 for the best-matching BM25 hit, less for
        others, but with the shipped ``similarity_threshold: 0.0`` floor,
        even a near-zero one still clears it -- so that deleted chunk can
        still reach the LLM context via worker B's sparse arm. The moment a
        second process shares a dense store (``--workers`` on uvicorn, or a
        second replica), this needs either cross-process
        BM25 invalidation (e.g. a delete broadcast/pub-sub all workers
        subscribe to) or a hydration TTL that forces periodic re-hydration --
        neither exists today.

        Fast path (no lock, no enumeration): if a prior call already
        attempted hydration, return immediately.

        Slow path: no attempt has been made yet. Acquire the lock and
        double-check ``_sparse_hydration_attempted`` again inside it -- two
        concurrent first searches (a real scenario: concurrent turns share a
        retriever instance) must not both enumerate the persistent store.
        Whichever coroutine wins the lock hydrates once; the loser's
        double-check finds ``_sparse_hydration_attempted`` already True and
        returns without doing anything.

        The attempted flag is set in a ``finally`` so it is marked done even
        when hydration finds zero chunks or raises -- an empty or
        unreachable persistent store must not turn into a per-query
        round-trip. A hydration failure is logged and swallowed: a search
        must never fail because the self-heal did, it just stays dense-only
        (or, if this instance had already ingested some chunks in-process,
        partially-fused) for the rest of this instance's life.
        """
        if self._sparse_hydration_attempted:
            return

        if self._sparse_hydrate_lock is None:
            self._sparse_hydrate_lock = asyncio.Lock()

        async with self._sparse_hydrate_lock:
            # Double-check: a concurrent first search may have already
            # completed hydration while we were waiting for the lock.
            if self._sparse_hydration_attempted:
                return
            try:
                n = await self.hydrate_sparse_from_persistent()
            except Exception:
                log.exception(
                    "lazy sparse hydration failed on first hybrid search; "
                    "continuing with dense-only results for this and all "
                    "subsequent searches on this retriever instance",
                    extra={"store": type(self._dense).__name__},
                )
            else:
                log.info(
                    "lazy sparse hydration: populated BM25 with %d chunk(s) "
                    "from the persistent store on first hybrid search",
                    n,
                    extra={"store": type(self._dense).__name__},
                )
            finally:
                self._sparse_hydration_attempted = True

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
            await self._ensure_sparse_hydrated()
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

        # Apply post-fusion threshold then trim.
        passing = [c for c in fused if c.score >= cfg.similarity_threshold]
        return passing[:k]

    async def _dense_search(
        self,
        query: str,
        k: int,
        filters: Optional[dict],
    ) -> list[SearchResult]:
        # Embedders are sync (Gemini REST call); run in a thread to avoid
        # blocking the event loop — same treatment as the ingest path above.
        # On a single-worker deployment a blocking call here freezes every
        # concurrent session, not just this one.
        q_vec = await asyncio.to_thread(self._embedder.embed_query, query)
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
