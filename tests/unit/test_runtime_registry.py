"""Tests for the per-tenant TenantRuntimeRegistry instantiation (Phase 3)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.auth.context import TenantContext
from src.auth.registry import _AsyncPerTenantRegistry
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


def test_registry_crm_tools_defaults_to_none_and_is_evict_safe():
    # build_runtime_registry doesn't know how to load CRM tools (that lives in
    # make_chatbot_factory) — crm_tools starts unset, and evict_all/evict_tenant
    # must tolerate that instead of blowing up on a None sub-registry.
    providers = SimpleNamespace(get_vector_store=lambda t: object(), evict=lambda tid=None: None)
    reg = build_runtime_registry(providers, _base_store())
    assert reg.crm_tools is None
    reg.evict_all()          # must not raise
    reg.evict_tenant("t1")   # must not raise


@pytest.mark.asyncio
async def test_async_per_tenant_registry_caches_and_evicts():
    calls: list[str] = []

    async def factory(tenant: TenantContext):
        calls.append(tenant.id)
        return f"loaded-for-{tenant.id}"

    reg = _AsyncPerTenantRegistry(factory)
    t = _tenant()

    assert await reg.get(t) == "loaded-for-t1"
    assert await reg.get(t) == "loaded-for-t1"
    assert calls == ["t1"]  # second call was a cache hit, factory ran once

    reg.evict("t1")
    assert await reg.get(t) == "loaded-for-t1"
    assert calls == ["t1", "t1"]  # eviction forces a reload


@pytest.mark.asyncio
async def test_async_per_tenant_registry_dedupes_concurrent_cold_calls():
    # The "thundering herd" case this exists for: many concurrent first-time
    # callers for the same tenant (e.g. a burst of new chat sessions) must
    # await one shared load, not each trigger their own DB round-trip.
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_factory(tenant: TenantContext):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "loaded"

    reg = _AsyncPerTenantRegistry(slow_factory)
    t = _tenant()

    task1 = asyncio.create_task(reg.get(t))
    task2 = asyncio.create_task(reg.get(t))
    await started.wait()
    release.set()
    results = await asyncio.gather(task1, task2)

    assert results == ["loaded", "loaded"]
    assert calls == 1
