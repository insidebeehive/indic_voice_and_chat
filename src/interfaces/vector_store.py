"""Vector store provider interface (PRD §4.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class SearchResult:
    document: Document
    score: float


@dataclass
class VectorStoreConfig:
    index_path: Optional[str] = None
    collection_name: str = "default"
    embedding_dim: int = 384


class IVectorStore(ABC):
    @abstractmethod
    async def index(self, documents: list[Document]) -> int:
        """Index documents. Returns count indexed."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""

    @abstractmethod
    async def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by ID. Returns count deleted."""

    @abstractmethod
    async def count(self) -> int:
        """Return total document count."""

    async def list_documents(self, limit: int = 2000) -> list[Document]:
        """Enumerate stored documents for this scope (no query). Stores that
        cannot enumerate return []. Order is NOT guaranteed — callers that care
        must sort. Embeddings are not populated."""
        return []
