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
        c._sm = sm  # stashed for tests that need the exact same DB the route used
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


async def test_patch_crm_livekit_secrets_not_leaked_but_resolvable(
    client: AsyncClient, monkeypatch,
) -> None:
    """PATCH sets livekit_url + secrets; GET returns the (non-secret) URL and
    a livekit_configured flag but never the decrypted key/secret; a
    subsequent resolve_livekit_creds call for a tenant under that CRM
    resolves the real values (integration through the resolver, not just the
    HTTP round-trip)."""
    from src.auth import secrets as crypto
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())

    await client.post("/crms", json={
        "id": "betstudio", "name": "BetStudio", "base_url": "https://x",
    }, headers=ADMIN_HEADERS)

    resp = await client.patch("/crms/betstudio", json={
        "livekit_url": "wss://lk.betstudio.example/rtc",
        "livekit_api_key": "super-secret-key",
        "livekit_api_secret": "super-secret-secret",
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["livekit_url"] == "wss://lk.betstudio.example/rtc"
    assert body["livekit_configured"] is True
    assert "super-secret-key" not in resp.text
    assert "super-secret-secret" not in resp.text

    detail = (await client.get("/crms/betstudio", headers=ADMIN_HEADERS)).json()
    assert detail["livekit_url"] == "wss://lk.betstudio.example/rtc"
    assert detail["livekit_configured"] is True
    assert "super-secret-key" not in str(detail)
    assert "super-secret-secret" not in str(detail)

    # Integration: resolve_livekit_creds for a tenant under this CRM gets the
    # real decrypted values, via the exact same DB (sessionmaker) the route
    # itself wrote to (stashed on the client by the fixture).
    from src.auth.context import TenantContext
    from src.config_tenant import (
        TenantPipelineConfig, TenantSettings, TenantTelephonyConfig, resolve_livekit_creds,
    )

    tenant = TenantContext(
        settings=TenantSettings(
            id="t1", slug="t1", name="T1", crm_id="betstudio",
            pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig()),
        ),
    )
    async with client._sm() as session:
        result = await resolve_livekit_creds(session, tenant)
    assert result == (
        "wss://lk.betstudio.example/rtc", "super-secret-key", "super-secret-secret")


async def test_patch_crm_livekit_secrets_without_vox_secret_key_503(
    client: AsyncClient, monkeypatch,
) -> None:
    """Mirrors the 4 crypto.has_key() guards in src/api/tenants.py: attempting
    to write a LiveKit secret with no VOX_SECRET_KEY configured must 503, not
    500 on an unhandled SecretsError. A PATCH that doesn't touch secrets at
    all must still succeed with no key configured."""
    from src.auth import secrets as crypto
    monkeypatch.delenv(crypto.VOX_SECRET_KEY_ENV, raising=False)

    await client.post("/crms", json={
        "id": "betstudio", "name": "BetStudio", "base_url": "https://x",
    }, headers=ADMIN_HEADERS)

    # No secrets touched -> no key required, still 200.
    resp = await client.patch("/crms/betstudio", json={
        "livekit_url": "wss://lk.betstudio.example/rtc",
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text

    # A secret provided with no VOX_SECRET_KEY set -> 503, not 500.
    resp = await client.patch("/crms/betstudio", json={
        "livekit_api_key": "some-key",
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 503
    assert "VOX_SECRET_KEY" in resp.text


async def test_crm_detail_livekit_configured_false_when_url_missing(
    client: AsyncClient, monkeypatch,
) -> None:
    """Both secret halves present but no livekit_url -> livekit_configured
    must be False, matching what resolve_livekit_creds actually requires
    (url + both secrets) rather than just "the secrets happen to be set"."""
    from src.auth import secrets as crypto
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())

    await client.post("/crms", json={
        "id": "betstudio", "name": "BetStudio", "base_url": "https://x",
    }, headers=ADMIN_HEADERS)

    resp = await client.patch("/crms/betstudio", json={
        "livekit_api_key": "k", "livekit_api_secret": "s",
    }, headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    assert resp.json()["livekit_url"] is None
    assert resp.json()["livekit_configured"] is False


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
