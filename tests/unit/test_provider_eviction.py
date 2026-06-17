"""Provider-cache eviction on tenant reload (stale-creds fix)."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth.db_resolver import DbTenantResolver
from src.auth.registry import TenantProviders
from src.models.database import Base
from src.models.tenant import Tenant


def _providers() -> TenantProviders:
    return TenantProviders(
        global_defaults={}, stt_factory=lambda c: object(), llm_factory=lambda c: object(),
        tts_factory=lambda c: object(), telephony_factory=lambda c: object(),
        vector_store_factory=lambda c: object())


def test_evict_none_clears_all_tenants():
    p = _providers()
    p._cache[("t1", "stt")] = object()
    p._cache[("t2", "llm")] = object()
    p.evict("t1")
    assert ("t1", "stt") not in p._cache and ("t2", "llm") in p._cache   # one tenant
    p.evict(None)
    assert p._cache == {}                                                # all


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        await s.commit()
    yield maker
    await engine.dispose()


async def test_resolver_reload_fires_on_reload(sm):
    fired = []
    r = DbTenantResolver(sm)
    r.on_reload = lambda: fired.append(True)
    await r.reload()
    assert fired == [True]          # so providers.evict runs on a config reload
