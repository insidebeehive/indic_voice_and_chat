"""Unit tests for the Crm/CrmTool models and Tenant.crm_id."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.crm import Crm, CrmTool
from src.models.database import Base
from src.models.tenant import Tenant


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def test_crm_and_crm_tool_round_trip(sm):
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://apistage.betstudio.io/api",
                    events_webhook_url_template="https://bostage.betstudio.io/webhooks/crm/softphone-events/{operator_id}",
                    auth_type="api_key"))
        db.add(CrmTool(crm_id="betstudio", name="get_player_wallet",
                        description="Get wallet", endpoint="/players/{user_id}/wallet",
                        method="GET", parameters={"user_id": {"type": "string", "source": "session"}}))
        await db.commit()

    async with sm() as db:
        crm = await db.get(Crm, "betstudio")
        assert crm.base_url == "https://apistage.betstudio.io/api"
        assert crm.events_webhook_url_template.endswith("{operator_id}")
        assert crm.auth_type == "api_key"


async def test_crm_tool_unique_name_per_crm(sm):
    from sqlalchemy.exc import IntegrityError
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x", auth_type="api_key"))
        db.add(CrmTool(crm_id="betstudio", name="get_player_wallet", description="d",
                        endpoint="/e", method="GET", parameters={}))
        await db.commit()
    async with sm() as db:
        db.add(CrmTool(crm_id="betstudio", name="get_player_wallet", description="dup",
                        endpoint="/e2", method="GET", parameters={}))
        try:
            await db.commit()
            assert False, "expected IntegrityError on duplicate (crm_id, name)"
        except IntegrityError:
            pass


async def test_tenant_crm_id_nullable_and_settable(sm):
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x", auth_type="api_key"))
        db.add(Tenant(id="t1", slug="t1", name="T1"))
        await db.commit()

    async with sm() as db:
        t = await db.get(Tenant, "t1")
        assert t.crm_id is None
        t.crm_id = "betstudio"
        await db.commit()

    async with sm() as db:
        t = await db.get(Tenant, "t1")
        assert t.crm_id == "betstudio"
