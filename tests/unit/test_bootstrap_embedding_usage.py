"""Unit test for src/bootstrap.py's _embedding_usage_recorder -- the small
factory that binds a HybridRetriever-shaped ``record_embedding_usage``
callback (see src/rag/retriever.py's EmbeddingUsageRecorder) to one
retriever's scope (tenant-KB or CRM-KB) and embedder instance, without going
through a live DB (src.models.embedding_usage.record_embedding_usage is
monkeypatched).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.bootstrap as bootstrap


@pytest.mark.asyncio
async def test_embedding_usage_recorder_tenant_scoped_forwards_kwargs(monkeypatch) -> None:
    calls = []

    async def _fake_record_embedding_usage(**kwargs):
        calls.append(kwargs)
        return 42

    monkeypatch.setattr(
        "src.models.embedding_usage.record_embedding_usage", _fake_record_embedding_usage,
    )

    embedder = SimpleNamespace(model_name="gemini-embedding-001")
    record = bootstrap._embedding_usage_recorder(tenant_id="t1", crm_id=None, embedder=embedder)

    result = await record("search", 10)

    assert result == 42
    assert calls == [{
        "tenant_id": "t1", "crm_id": None, "purpose": "search",
        "provider": "gemini", "model": "gemini-embedding-001", "input_chars": 10,
    }]


@pytest.mark.asyncio
async def test_embedding_usage_recorder_crm_scoped_forwards_kwargs(monkeypatch) -> None:
    calls = []

    async def _fake_record_embedding_usage(**kwargs):
        calls.append(kwargs)
        return 7

    monkeypatch.setattr(
        "src.models.embedding_usage.record_embedding_usage", _fake_record_embedding_usage,
    )

    embedder = SimpleNamespace(model_name="gemini-embedding-001")
    record = bootstrap._embedding_usage_recorder(tenant_id=None, crm_id="c1", embedder=embedder)

    result = await record("ingest", 4000)

    assert result == 7
    assert calls == [{
        "tenant_id": None, "crm_id": "c1", "purpose": "ingest",
        "provider": "gemini", "model": "gemini-embedding-001", "input_chars": 4000,
    }]


@pytest.mark.asyncio
async def test_embedding_usage_recorder_falls_back_to_empty_model_name(monkeypatch) -> None:
    """An embedder with no model_name attribute (or an empty one) must not
    crash the recorder -- falls back to ""."""
    calls = []

    async def _fake_record_embedding_usage(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "src.models.embedding_usage.record_embedding_usage", _fake_record_embedding_usage,
    )

    embedder = SimpleNamespace()  # no model_name attribute at all
    record = bootstrap._embedding_usage_recorder(tenant_id="t1", crm_id=None, embedder=embedder)

    await record("search", 5)

    assert calls[0]["model"] == ""


# --- Wiring: both retriever-building call sites actually pass a working
# recorder into HybridRetriever, not just that _embedding_usage_recorder
# works in isolation (above). Mirrors test_bootstrap_retrieval_config.py's
# real-wiring-not-just-the-helper approach.


@pytest.mark.asyncio
async def test_build_crm_retriever_wires_crm_scoped_recorder(monkeypatch) -> None:
    import src.config as config_module
    import src.providers as providers_module

    calls = []

    async def _fake_record_embedding_usage(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "src.models.embedding_usage.record_embedding_usage", _fake_record_embedding_usage,
    )
    fake_retrieval_settings = SimpleNamespace(
        strategy="hybrid", top_k=5, bm25_weight=0.3, dense_weight=0.7,
        rrf_k=60, similarity_threshold=0.0,
    )
    fake_settings = SimpleNamespace(rag=SimpleNamespace(retrieval=fake_retrieval_settings))
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(providers_module, "get_vector_store", lambda cfg: object())

    retriever = bootstrap.build_crm_retriever("crm-1", {"vector_store": {"provider": "pgvector"}})

    assert retriever is not None
    assert retriever._record_embedding_usage is not None
    await retriever._record_embedding_usage("search", 5)
    assert calls == [{
        "tenant_id": None, "crm_id": "crm-1", "purpose": "search",
        "provider": "gemini", "model": "gemini-embedding-001", "input_chars": 5,
    }]


@pytest.mark.asyncio
async def test_build_runtime_registry_wires_tenant_scoped_recorder(monkeypatch) -> None:
    import src.config as config_module

    calls = []

    async def _fake_record_embedding_usage(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "src.models.embedding_usage.record_embedding_usage", _fake_record_embedding_usage,
    )
    fake_retrieval_settings = SimpleNamespace(
        strategy="hybrid", top_k=5, bm25_weight=0.3, dense_weight=0.7,
        rrf_k=60, similarity_threshold=0.0,
    )
    fake_settings = SimpleNamespace(rag=SimpleNamespace(retrieval=fake_retrieval_settings))
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)

    tenant = SimpleNamespace(id="tenant-1", settings=SimpleNamespace(
        compliance=None, max_concurrent_calls=None,
    ))
    providers = SimpleNamespace(get_vector_store=lambda t: object())
    base_session_store = SimpleNamespace(redis=None, ttl=60)

    registry = bootstrap.build_runtime_registry(providers, base_session_store)
    retriever = registry.retrievers.get(tenant)

    assert retriever._record_embedding_usage is not None
    await retriever._record_embedding_usage("ingest", 4000)
    assert calls == [{
        "tenant_id": "tenant-1", "crm_id": None, "purpose": "ingest",
        "provider": "gemini", "model": "gemini-embedding-001", "input_chars": 4000,
    }]
