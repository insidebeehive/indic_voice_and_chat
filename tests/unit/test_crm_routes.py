"""Route tests for CRM entity CRUD (admin-only)."""

from __future__ import annotations

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import crms
from src.api.deps import get_db_session
from src.auth.middleware import set_admin_tokens
from src.models.database import Base

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async def _session_override():
        async with sm() as session:
            yield session

    set_admin_tokens(["admin-token"])
    app = FastAPI()
    app.include_router(crms.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_admin_tokens([])
    await engine.dispose()


async def test_create_and_get_crm(client: AsyncClient) -> None:
    resp = await client.post("/crms", json={
        "id": "betstudio", "name": "BetStudio", "base_url": "https://apistage.betstudio.io/api",
        "auth_type": "api_key",
        "tools": [{"name": "get_player_wallet", "description": "d",
                   "endpoint": "/players/{user_id}/wallet", "method": "GET", "parameters": {}}],
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text

    resp = await client.get("/crms/betstudio", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["base_url"] == "https://apistage.betstudio.io/api"
    assert len(body["tools"]) == 1


async def test_list_crms_requires_admin(client: AsyncClient) -> None:
    assert (await client.get("/crms")).status_code == 401


async def test_patch_crm_replaces_tool_list(client: AsyncClient) -> None:
    await client.post("/crms", json={
        "id": "betstudio", "name": "BetStudio", "base_url": "https://x",
        "tools": [{"name": "a", "description": "d", "endpoint": "/a", "method": "GET", "parameters": {}}],
    }, headers=ADMIN_HEADERS)

    resp = await client.patch("/crms/betstudio", json={
        "tools": [{"name": "b", "description": "d2", "endpoint": "/b", "method": "GET", "parameters": {}}],
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 200

    detail = (await client.get("/crms/betstudio", headers=ADMIN_HEADERS)).json()
    names = {t["name"] for t in detail["tools"]}
    assert names == {"b"}
