"""Regression test for the RetrievalConfig/RetrievalConfig name collision.

There used to be two unrelated classes both named ``RetrievalConfig``: a
Pydantic settings model (src/config.py, populated from config/default.yaml's
``rag.retrieval`` block) and the dataclass HybridRetriever actually reads
(src/rag/retriever.py). Nothing bridged them, and both bootstrap.py wiring
sites constructed the dataclass with no arguments at all -- so the entire
YAML block was silently inert.

This test builds a retriever via the real bootstrap wiring (build_crm_retriever)
with a non-default rag.retrieval setting and asserts it actually lands on the
constructed HybridRetriever. It must fail again if someone reintroduces a
no-arg ``RetrievalConfig()`` at either wiring site.
"""
from __future__ import annotations

from types import SimpleNamespace

import src.bootstrap as bootstrap
import src.config as config_module
import src.providers as providers_module


def test_build_crm_retriever_bridges_non_default_top_k(monkeypatch) -> None:
    # top_k and bm25_weight (not similarity_threshold — that one is
    # deliberately pinned to 0.0 in config/default.yaml pending a
    # retrieval-eval measurement) are values nothing else could plausibly
    # produce by coincidence: 17 isn't a default anywhere, and 0.6 isn't the
    # RetrievalConfig dataclass default (0.3), so a no-arg RetrievalConfig()
    # regression would fail this assertion instead of passing it vacuously.
    fake_retrieval_settings = SimpleNamespace(
        strategy="hybrid",
        top_k=17,
        bm25_weight=0.6,
        dense_weight=0.7,
        similarity_threshold=0.0,
    )
    fake_settings = SimpleNamespace(rag=SimpleNamespace(retrieval=fake_retrieval_settings))
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)

    dummy_vector_store = object()
    monkeypatch.setattr(providers_module, "get_vector_store", lambda cfg: dummy_vector_store)

    retriever = bootstrap.build_crm_retriever(
        "crm-1", {"vector_store": {"provider": "pgvector"}}
    )

    assert retriever is not None
    assert retriever.config.top_k == 17
    assert retriever.config.bm25_weight == 0.6


def test_build_runtime_registry_bridges_non_default_top_k(monkeypatch) -> None:
    # Second wiring site (src/bootstrap.py's build_runtime_registry ->
    # _retriever), covering the docstring's claim that both sites are
    # protected against the no-arg RetrievalConfig() regression. A
    # SimpleNamespace stands in for TenantContext — _PerTenantRegistry.get
    # only ever reads tenant.id.
    fake_retrieval_settings = SimpleNamespace(
        strategy="hybrid",
        top_k=23,
        bm25_weight=0.6,
        dense_weight=0.7,
        similarity_threshold=0.0,
    )
    fake_settings = SimpleNamespace(rag=SimpleNamespace(retrieval=fake_retrieval_settings))
    monkeypatch.setattr(config_module, "get_settings", lambda: fake_settings)

    tenant = SimpleNamespace(id="tenant-1", settings=SimpleNamespace(
        compliance=None, max_concurrent_calls=None,
    ))

    dummy_vector_store = object()
    providers = SimpleNamespace(get_vector_store=lambda t: dummy_vector_store)
    base_session_store = SimpleNamespace(redis=None, ttl=60)

    registry = bootstrap.build_runtime_registry(providers, base_session_store)
    retriever = registry.retrievers.get(tenant)

    assert retriever.config.top_k == 23
    assert retriever.config.bm25_weight == 0.6
