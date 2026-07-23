"""tests/unit/test_pgvector_crm_scoping.py"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="requires a real Postgres with pgvector (voicebot.knowledge_chunks)",
)


@pytest.mark.asyncio
async def test_crm_scoped_chunk_is_isolated_from_tenant_scoped_chunk() -> None:
    from src.providers.vector_store.pgvector_store import PGVectorAdapter
    from src.interfaces.vector_store import Document

    crm_store = PGVectorAdapter({"embedding_dim": 4, "crm_id": "test-crm-x"})
    tenant_store = PGVectorAdapter({"embedding_dim": 4, "tenant_id": "test-tenant-x"})

    try:
        await crm_store.index([Document(id="crm-doc-1", content="crm content",
                                         metadata={}, embedding=[0.1, 0.2, 0.3, 0.4])])
        await tenant_store.index([Document(id="tenant-doc-1", content="tenant content",
                                            metadata={}, embedding=[0.1, 0.2, 0.3, 0.4])])

        crm_results = await crm_store.search([0.1, 0.2, 0.3, 0.4], top_k=10)
        tenant_results = await tenant_store.search([0.1, 0.2, 0.3, 0.4], top_k=10)

        assert {r.document.id for r in crm_results} == {"crm-doc-1"}
        assert {r.document.id for r in tenant_results} == {"tenant-doc-1"}
        assert await crm_store.count() == 1
        assert await tenant_store.count() == 1
    finally:
        await crm_store.delete(["crm-doc-1"])
        await tenant_store.delete(["tenant-doc-1"])
