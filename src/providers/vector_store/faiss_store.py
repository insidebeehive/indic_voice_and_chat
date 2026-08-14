"""FAISS vector store adapter (file-backed).

Uses ``IndexFlatIP`` over L2-normalized vectors so inner-product = cosine
similarity. A sidecar JSON file holds the ``id -> Document`` mapping plus
the active id list (stable indexing into the FAISS array).

Embedding *generation* is deliberately out of scope here — callers must
supply ``Document.embedding``. Generating embeddings (with
``sentence-transformers``) is Phase 4 work.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from src.interfaces.vector_store import (
    Document,
    IVectorStore,
    SearchResult,
)


class FAISSAdapter(IVectorStore):
    def __init__(self, config: dict[str, Any]) -> None:
        self._dim = int(config.get("embedding_dim", 384))
        index_path = config.get("index_path") or "./data/faiss_index"
        self._index_path = Path(index_path)
        self._faiss_file = self._index_path.with_suffix(".faiss")
        self._meta_file = self._index_path.with_suffix(".meta.json")
        self._lock = asyncio.Lock()

        # In-memory state.
        self._index: faiss.Index = faiss.IndexFlatIP(self._dim)
        self._ids: list[str] = []  # row order matches FAISS array
        self._docs: dict[str, Document] = {}

        if self._faiss_file.exists() and self._meta_file.exists():
            self._load()

    # --- Persistence -----------------------------------------------------

    def _load(self) -> None:
        self._index = faiss.read_index(str(self._faiss_file))
        with self._meta_file.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self._ids = list(meta.get("ids", []))
        self._docs = {
            doc_id: Document(
                id=d["id"],
                content=d["content"],
                metadata=d.get("metadata", {}),
                embedding=None,  # we don't keep the vector outside FAISS
            )
            for doc_id, d in meta.get("docs", {}).items()
        }

    def _persist(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._faiss_file))
        with self._meta_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": self._ids,
                    "docs": {
                        d.id: {"id": d.id, "content": d.content, "metadata": d.metadata}
                        for d in self._docs.values()
                    },
                },
                f,
            )

    # --- IVectorStore ----------------------------------------------------

    async def index(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        vectors: list[list[float]] = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(
                    f"Document {doc.id!r} has no embedding. Embedding generation "
                    "is the caller's responsibility (Phase 4 RAG layer)."
                )
            if len(doc.embedding) != self._dim:
                raise ValueError(
                    f"Document {doc.id!r} embedding dim {len(doc.embedding)} "
                    f"does not match index dim {self._dim}"
                )
            vectors.append(doc.embedding)

        arr = np.array(vectors, dtype="float32")
        _l2_normalize_inplace(arr)

        async with self._lock:
            # Replace existing docs by id: drop + re-add idempotently.
            existing = [doc.id for doc in documents if doc.id in self._docs]
            if existing:
                await self._delete_unlocked(existing)

            self._index.add(arr)
            for doc in documents:
                self._ids.append(doc.id)
                self._docs[doc.id] = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=dict(doc.metadata),
                    embedding=None,
                )
            self._persist()
        return len(documents)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        if len(query_embedding) != self._dim:
            raise ValueError(
                f"query_embedding dim {len(query_embedding)} != index dim {self._dim}"
            )
        if self._index.ntotal == 0:
            return []

        q = np.array([query_embedding], dtype="float32")
        _l2_normalize_inplace(q)

        # Over-fetch when filtering, then trim post-search.
        k = top_k * 4 if filters else top_k
        k = min(k, self._index.ntotal)

        async with self._lock:
            scores, idxs = self._index.search(q, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._ids):
                continue
            doc_id = self._ids[idx]
            doc = self._docs.get(doc_id)
            if doc is None:
                continue
            if filters and not _matches(doc.metadata, filters):
                continue
            results.append(SearchResult(document=doc, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    async def delete(self, doc_ids: list[str]) -> int:
        async with self._lock:
            count = await self._delete_unlocked(doc_ids)
            self._persist()
        return count

    async def _delete_unlocked(self, doc_ids: list[str]) -> int:
        # IndexFlatIP doesn't support remove_ids; rebuild without the targets.
        targets = set(doc_ids)
        keep_ids = [i for i in self._ids if i not in targets]
        removed = len(self._ids) - len(keep_ids)
        if removed == 0:
            return 0

        # We discarded raw vectors after indexing, so rebuilding from scratch
        # isn't possible without re-embedding. Instead, mask removed rows by
        # constructing a fresh IndexFlatIP from the *retained* rows of the
        # current index.
        retained_rows = [
            self._index.reconstruct(self._ids.index(i))
            for i in keep_ids
        ]
        self._index = faiss.IndexFlatIP(self._dim)
        if retained_rows:
            self._index.add(np.array(retained_rows, dtype="float32"))
        self._ids = keep_ids
        for tid in targets:
            self._docs.pop(tid, None)
        return removed

    async def count(self) -> int:
        return int(self._index.ntotal)

    async def list_documents(self, limit: int = 2000) -> list[Document]:
        async with self._lock:
            return [self._docs[i] for i in self._ids[:limit] if i in self._docs]


# --- helpers --------------------------------------------------------------


def _l2_normalize_inplace(arr: np.ndarray) -> None:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    arr /= norms


def _matches(metadata: dict, filters: dict) -> bool:
    for k, v in filters.items():
        if metadata.get(k) != v:
            return False
    return True
