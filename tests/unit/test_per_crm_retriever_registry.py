"""tests/unit/test_per_crm_retriever_registry.py"""
from __future__ import annotations

from src.bootstrap import PerCrmRetrieverRegistry, build_crm_retriever


def test_build_crm_retriever_returns_none_for_non_pgvector_provider() -> None:
    r = build_crm_retriever("betstudio", global_defaults={"vector_store": {"provider": "faiss"}})
    assert r is None


def test_registry_caches_per_crm_id(monkeypatch) -> None:
    calls = []

    def fake_build(crm_id, global_defaults=None):
        calls.append(crm_id)
        return object()

    monkeypatch.setattr("src.bootstrap.build_crm_retriever", fake_build)
    registry = PerCrmRetrieverRegistry(global_defaults={})

    first = registry.get("betstudio")
    second = registry.get("betstudio")
    other = registry.get("other-crm")

    assert first is second           # cached, same instance on second call
    assert first is not other        # different crm_id -> different instance
    assert calls == ["betstudio", "other-crm"]  # built exactly once per crm_id
