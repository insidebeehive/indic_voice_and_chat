"""Route tests for /api/v1/chat/tools (CRM tool registration, Phase 3b)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat_tools
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test, secrets as crypto
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.chat import ChatTool
from src.models.database import Base
from src.models.tenant import Tenant, TenantSecret

HEADERS = {"Authorization": "Bearer test-token"}

_TOOL = {
    "name": "check_order_status",
    "description": "Check delivery status of an order by id",
    "endpoint": "https://crm.example.com/api/orders/{order_id}/status",
    "method": "GET",
    "auth_type": "bearer",
    "auth_token": "super-secret-crm-token",
    "parameters": {"order_id": {"type": "string", "source": "llm"}},
}


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    monkeypatch.setenv("VOX_SECRET_KEY", crypto.generate_key())
    crypto.reset_cache_for_tests()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(TenantSettings(id="t1", slug="t1", name="T1"),
                             plaintext_tokens=["test-token"])
    app = FastAPI()
    app.include_router(chat_tools.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=HEADERS) as c:
        yield c, sm
    set_tenant_resolver(None)
    await engine.dispose()


async def test_register_persists_tool_and_encrypts_token(ctx) -> None:
    client, sm = ctx
    resp = await client.post("/chat/tools", json={"tools": [_TOOL]})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"registered": 1, "tool_names": ["check_order_status"]}

    async with sm() as s:
        row = (await s.execute(select(ChatTool).where(ChatTool.name == "check_order_status"))).scalar_one()
        assert row.endpoint.endswith("/orders/{order_id}/status")
        assert row.auth_config["token_secret_name"] == "chat_tool:check_order_status:token"
        sec = (await s.execute(select(TenantSecret).where(
            TenantSecret.name == "chat_tool:check_order_status:token"))).scalar_one()
        # Stored encrypted, decryptable back to the original.
        assert sec.value_encrypted != "super-secret-crm-token"
        assert crypto.decrypt(sec.value_encrypted) == "super-secret-crm-token"


async def test_list_tools_hides_token(ctx) -> None:
    client, _ = ctx
    await client.post("/chat/tools", json={"tools": [_TOOL]})
    resp = await client.get("/chat/tools")
    assert resp.status_code == 200
    tools = resp.json()["tools"]
    assert len(tools) == 1
    t = tools[0]
    assert t["name"] == "check_order_status"
    assert t["auth_configured"] is True
    assert "auth_token" not in t and "token" not in t


async def test_register_is_idempotent_upsert(ctx) -> None:
    client, sm = ctx
    await client.post("/chat/tools", json={"tools": [_TOOL]})
    updated = {**_TOOL, "description": "updated desc", "auth_token": None}
    await client.post("/chat/tools", json={"tools": [updated]})
    resp = await client.get("/chat/tools")
    tools = resp.json()["tools"]
    assert len(tools) == 1  # upserted, not duplicated
    assert tools[0]["description"] == "updated desc"


async def test_delete_removes_tool_and_secret(ctx) -> None:
    client, sm = ctx
    await client.post("/chat/tools", json={"tools": [_TOOL]})
    resp = await client.delete("/chat/tools/check_order_status")
    assert resp.status_code == 200
    async with sm() as s:
        assert (await s.execute(select(ChatTool))).scalars().all() == []
        assert (await s.execute(select(TenantSecret))).scalars().all() == []


async def test_delete_unknown_404(ctx) -> None:
    client, _ = ctx
    assert (await client.delete("/chat/tools/nope")).status_code == 404


async def test_register_requires_auth(ctx) -> None:
    client, _ = ctx
    resp = await client.post("/chat/tools", json={"tools": [_TOOL]},
                             headers={"Authorization": "Bearer wrong"})
    assert resp.status_code in (401, 403)


async def test_factory_loads_crm_tools_and_enables_loop(monkeypatch) -> None:
    # The bootstrap factory loads a tenant's registered tools into the agent
    # (as ToolSpecs) and turns the agentic loop on.
    from types import SimpleNamespace

    from src.auth import TenantContext
    from src.bootstrap import make_chatbot_factory

    monkeypatch.setenv("VOX_SECRET_KEY", crypto.generate_key())
    crypto.reset_cache_for_tests()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        s.add(ChatTool(
            tenant_id="t1", name="check_order_status", description="check order",
            endpoint="https://crm/api/orders/{order_id}", method="GET", auth_type="bearer",
            auth_config={"token_secret_name": "chat_tool:check_order_status:token"},
            parameters={"order_id": {"type": "string", "source": "llm"}}))
        s.add(TenantSecret(tenant_id="t1", name="chat_tool:check_order_status:token",
                           value_encrypted=crypto.encrypt("tok-123")))
        await s.commit()

    registry = SimpleNamespace(
        providers=SimpleNamespace(get_llm=lambda t: object()),
        retrievers=SimpleNamespace(get=lambda t: object()),
        session_stores=SimpleNamespace(get=lambda t: None),
    )
    factory = make_chatbot_factory(registry, sm)
    tenant = TenantContext(settings=TenantSettings(id="t1", slug="t1", name="T1"))
    agent = await factory(tenant, "s1")
    assert agent._enable_tools is True
    assert any(t.name == "check_order_status" for t in agent._crm_tools)
    await engine.dispose()
