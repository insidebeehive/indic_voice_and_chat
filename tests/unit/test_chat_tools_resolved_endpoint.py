"""Tests for GET /chat/tools/resolved — the fresh, cache-bypassing view of a
tenant's ACTUALLY-resolved CRM tools (src/bootstrap.py resolve_crm_tools()).

Unlike GET /chat/tools (which only queries the chat_tools DB table), this
endpoint reports the platform-fallback path too, since that's what a tenant
with zero chat_tools rows actually gets served on its next chat turn.
"""

from __future__ import annotations

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat_tools
from src.api.deps import get_db_session
from src.auth import TenantContext, register_tenant_for_test
from src.auth import secrets as crypto
from src.auth.middleware import set_tenant_resolver
from src.bootstrap import resolve_crm_tools
from src.chatbot.catalog import ALL_TOOLS
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


def _clean_platform_env(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_CRM_BASE_URL", raising=False)
    monkeypatch.delenv("PLATFORM_CRM_API_TOKEN", raising=False)
    monkeypatch.delenv("PLATFORM_CRM_AUTH_TYPE", raising=False)


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    _clean_platform_env(monkeypatch)
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
    # The resolved-tools endpoint needs the raw sessionmaker (to open its own
    # sessions for the batched ChatTool + TenantSecret queries), not a single
    # per-request AsyncSession — patch the module-level import chat_tools.py
    # uses, same pattern as tests/unit/test_dev_console.py.
    monkeypatch.setattr(chat_tools, "get_sessionmaker", lambda: sm)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=HEADERS) as c:
        yield c, sm
    set_tenant_resolver(None)
    await engine.dispose()


async def test_tenant_registered_tools_reported_as_source_tenant(ctx) -> None:
    client, _sm = ctx
    await client.post("/chat/tools", json={"tools": [_TOOL]})

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "tenant"
    assert len(body["tools"]) == 1
    t = body["tools"][0]
    assert t["name"] == "check_order_status"
    assert t["endpoint"] == _TOOL["endpoint"]
    assert t["auth_type"] == "bearer"
    assert t["token_configured"] is True


async def test_platform_fallback_ignores_configured_platform_token(ctx, monkeypatch) -> None:
    # The shared PLATFORM_CRM_API_TOKEN must never be used for auth, even
    # though it's configured in the environment — this CRM authorizes by the
    # token itself, so a shared token would grant cross-tenant CRM access.
    client, _sm = ctx
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    monkeypatch.setenv("PLATFORM_CRM_API_TOKEN", "platform-token-abc")

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "platform_fallback"
    assert len(body["tools"]) == len(ALL_TOOLS)
    assert {t["name"] for t in body["tools"]} == set(ALL_TOOLS)
    assert all(t["token_configured"] is False for t in body["tools"])


async def test_platform_fallback_without_token_reports_not_configured(ctx, monkeypatch) -> None:
    client, _sm = ctx
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    # No PLATFORM_CRM_API_TOKEN set.

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "platform_fallback"
    assert len(body["tools"]) == len(ALL_TOOLS)
    assert all(t["token_configured"] is False for t in body["tools"])


async def test_x_api_key_configured_reported_independently_of_token(ctx, monkeypatch) -> None:
    client, sm = ctx
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    # The in-memory test resolver doesn't re-read TenantSecret rows, so seed
    # the tenant's secrets_resolved directly (same pattern as the other
    # platform-fallback tests that construct TenantContext with secrets_resolved).
    fresh_ctx = register_tenant_for_test(TenantSettings(id="t1", slug="t1", name="T1"),
                                          plaintext_tokens=["test-token"])
    fresh_ctx.secrets_resolved["crm:x_api_key"] = "the-x-api-key"

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "platform_fallback"
    # token_configured stays False (no crm:api_token set) while
    # x_api_key_configured is True — the two are independent.
    assert all(t["token_configured"] is False for t in body["tools"])
    assert all(t["x_api_key_configured"] is True for t in body["tools"])


async def test_nothing_configured_gives_empty_none_source(ctx) -> None:
    client, _sm = ctx
    # No chat_tools rows, no PLATFORM_CRM_* env, no tenant crm:* secrets.
    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "none"
    assert body["tools"] == []


async def test_resolved_endpoint_requires_auth(ctx) -> None:
    client, _sm = ctx
    resp = await client.get("/chat/tools/resolved", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code in (401, 403)


async def test_resolve_crm_tools_matches_cached_path_shape(monkeypatch) -> None:
    # Direct unit test of resolve_crm_tools(): same (specs, execs) shape the
    # cached _load_crm_tools_uncached path used to produce directly, proving
    # the extraction didn't change the cached path's behavior.
    _clean_platform_env(monkeypatch)
    monkeypatch.setenv("VOX_SECRET_KEY", crypto.generate_key())
    crypto.reset_cache_for_tests()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    try:
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

        tenant = TenantContext(settings=TenantSettings(id="t1", slug="t1", name="T1"))
        specs, execs, source = await resolve_crm_tools(tenant, sm)
        assert source == "tenant"
        assert len(specs) == 1
        assert specs[0].name == "check_order_status"
        assert execs["check_order_status"]["token"] == "tok-123"
        assert execs["check_order_status"]["endpoint"] == "https://crm/api/orders/{order_id}"
    finally:
        await engine.dispose()
