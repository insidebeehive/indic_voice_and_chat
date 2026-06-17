"""Unit tests for per-tenant DB-backed campaign resolution."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.dialogue.campaign_loader import LoadedCampaign, parse_campaign_yaml
from src.dialogue.campaign_resolver import DbCampaignResolver
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.models.campaign import Campaign
from src.models.database import Base
from src.models.tenant import Tenant

_YAML_A = """
campaign:
  agent:
    name: Riya
    company: Acme
    role: sales
  script:
    greeting: Hello from A
  slots:
    interest:
      type: enum
      values: [hot, warm, cold]
"""

_YAML_B = """
campaign:
  agent:
    name: Maya
    company: Globex
"""

# Explicit fallback so tests don't depend on the on-disk VOX_CAMPAIGN file.
FALLBACK = LoadedCampaign(
    VoiceBotScript.from_campaign_yaml({"name": "Fallbk", "company": "Plat"}),
    SlotSchema())


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add_all([Tenant(id="t1", slug="t1", name="T1"),
                   Tenant(id="t2", slug="t2", name="T2")])
        s.add_all([
            Campaign(id="c_active", tenant_id="t1", name="A", status="active", config_yaml=_YAML_A),
            Campaign(id="c_other", tenant_id="t1", name="B", status="active", config_yaml=_YAML_B),
            Campaign(id="c_t2", tenant_id="t2", name="X", status="active", config_yaml=_YAML_B),
        ])
        await s.commit()
    yield maker
    await engine.dispose()


def test_parse_campaign_yaml_builds_script_and_slots():
    lc = parse_campaign_yaml(_YAML_A)
    assert lc.script.agent_name == "Riya"
    assert lc.script.company_name == "Acme"
    assert "interest" in lc.slots.specs


async def test_resolve_by_campaign_id_returns_that_campaign(sm):
    r = DbCampaignResolver(sm, fallback=FALLBACK)
    lc = await r.resolve("t1", "c_other")
    assert lc.script.agent_name == "Maya"          # the requested one, not the default


async def test_resolve_cross_tenant_guard(sm):
    # c_t2 belongs to t2; t1 must NOT receive it — falls through to a t1 campaign.
    r = DbCampaignResolver(sm, fallback=FALLBACK)
    lc = await r.resolve("t1", "c_t2")
    assert lc.script.company_name in ("Acme", "Globex")   # a t1 campaign, never Globex-as-t2
    assert lc.script.agent_name != "Fallbk"               # t1 has active campaigns


async def test_resolve_active_when_no_campaign_id(sm):
    r = DbCampaignResolver(sm, fallback=FALLBACK)
    lc = await r.resolve("t1")
    assert lc.script.agent_name in ("Riya", "Maya")       # a t1 active campaign


async def test_resolve_falls_back_when_tenant_has_no_campaign(sm):
    r = DbCampaignResolver(sm, fallback=FALLBACK)
    lc = await r.resolve("t_none")
    assert lc.script.agent_name == "Fallbk"


async def test_resolve_parse_error_falls_back(sm):
    async with sm() as s:
        s.add(Tenant(id="t3", slug="t3", name="T3"))
        s.add(Campaign(id="c_bad", tenant_id="t3", name="bad", status="active",
                       config_yaml="just a plain string, not a campaign mapping"))
        await s.commit()
    r = DbCampaignResolver(sm, fallback=FALLBACK)
    lc = await r.resolve("t3", "c_bad")
    assert lc.script.agent_name == "Fallbk"               # unparseable → safe fallback
