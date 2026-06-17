"""Tests for the per-tenant TenantRuntimeRegistry instantiation (Phase 3)."""

from __future__ import annotations

from types import SimpleNamespace

from src.auth.context import TenantContext
from src.bootstrap import TenantDnd, build_runtime_registry
from src.config_tenant import TenantPipelineConfig, TenantSettings


def _base_store():
    return SimpleNamespace(redis=None, ttl=1800)


def _tenant():
    return TenantContext(settings=TenantSettings(
        id="t1", slug="t1", name="T1", pipeline=TenantPipelineConfig()))


def test_registry_builds_real_per_tenant_instances():
    from src.campaign.scheduler import CallScheduler
    from src.dialogue.context import SessionStore
    from src.rag.retriever import HybridRetriever

    providers = SimpleNamespace(get_vector_store=lambda t: object(),
                                evict=lambda tid=None: None)
    reg = build_runtime_registry(providers, _base_store())
    t = _tenant()
    assert isinstance(reg.dnd.get(t), TenantDnd)
    assert isinstance(reg.schedulers.get(t), CallScheduler)
    assert isinstance(reg.session_stores.get(t), SessionStore)
    assert isinstance(reg.retrievers.get(t), HybridRetriever)
    # one instance per tenant (cached)
    assert reg.dnd.get(t) is reg.dnd.get(t)
    assert reg.session_stores.get(t).tenant_id == "t1"


def test_registry_evict_all_clears_providers_and_subregistries():
    flags = {"providers_evicted": False}
    providers = SimpleNamespace(
        get_vector_store=lambda t: object(),
        evict=lambda tid=None: flags.update(providers_evicted=True))
    reg = build_runtime_registry(providers, _base_store())
    t = _tenant()
    reg.dnd.get(t)
    reg.session_stores.get(t)

    reg.evict_all()
    assert flags["providers_evicted"] is True
    assert reg.dnd.items() == {}
    assert reg.session_stores.items() == {}
