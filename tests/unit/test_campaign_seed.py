"""Unit tests for the per-tenant default-campaign seed."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth.seed import seed_campaigns_if_empty
from src.models.campaign import Campaign
from src.models.database import Base
from src.models.tenant import Tenant

_YAML = """
campaign:
  name: Bharat Matka
  agent:
    name: Anaaya
    company: Bharat Matka
"""


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add_all([Tenant(id="t1", slug="t1", name="T1"),
                   Tenant(id="t2", slug="t2", name="T2")])
        await s.commit()
    yield maker
    await engine.dispose()


async def test_seeds_a_campaign_for_each_tenant(sm, tmp_path):
    # active_campaign_slug() defaults to "bharat_matka" when VOX_CAMPAIGN is unset.
    (tmp_path / "bharat_matka.yaml").write_text(_YAML)

    n = await seed_campaigns_if_empty(sm, campaigns_dir=tmp_path)
    assert n == 2                                  # one per tenant

    async with sm() as s:
        rows = (await s.execute(select(Campaign))).scalars().all()
    assert {r.tenant_id for r in rows} == {"t1", "t2"}
    assert all(r.status == "active" for r in rows)
    assert all(r.name == "Bharat Matka" for r in rows)


async def test_idempotent_skips_tenants_with_a_campaign(sm, tmp_path):
    (tmp_path / "bharat_matka.yaml").write_text(_YAML)
    await seed_campaigns_if_empty(sm, campaigns_dir=tmp_path)
    # Second run adds nothing (every tenant already has a campaign).
    assert await seed_campaigns_if_empty(sm, campaigns_dir=tmp_path) == 0


async def test_missing_campaign_file_seeds_nothing(sm, tmp_path):
    # No <slug>.yaml present → skip cleanly (don't fabricate empty campaigns).
    assert await seed_campaigns_if_empty(sm, campaigns_dir=tmp_path) == 0
