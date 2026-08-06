"""Route-level tests for the CRM-facing call registration endpoint.

    POST /telephony/register-call

This is the endpoint a CRM partner calls to pre-register a call it is about
to place, so a ``conversations`` row exists before Twilio/Exotel fires our
slug-scoped answer webhook.
"""

from __future__ import annotations

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import telephony_crm
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.campaign import Campaign
from src.models.database import Base
from src.models.tenant import Tenant

HEADERS = {"Authorization": "Bearer test-token"}


@pytest_asyncio.fixture
async def client(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        await s.commit()
        s.add(Campaign(id="c1", tenant_id="t1", name="C1", config_yaml=""))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"), plaintext_tokens=["test-token"]
    )

    app = FastAPI()
    app.include_router(telephony_crm.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=HEADERS) as c:
        yield c

    set_tenant_resolver(None)
    await engine.dispose()


# --- Pattern A: /register-call ------------------------------------------


async def test_register_call_creates_conversation(client: AsyncClient) -> None:
    resp = await client.post(
        "/telephony/register-call",
        json={"provider": "twilio", "provider_call_sid": "CA123",
              "campaign_id": "c1"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["call_id"].startswith("call_")
    assert body["status"] == "in_progress"


async def test_register_call_idempotent_retry_returns_200(client: AsyncClient) -> None:
    first = await client.post(
        "/telephony/register-call",
        json={"provider": "twilio", "provider_call_sid": "CA123"},
    )
    assert first.status_code == 201

    retry = await client.post(
        "/telephony/register-call",
        json={"provider": "twilio", "provider_call_sid": "CA123"},
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["call_id"] == first.json()["call_id"]


async def test_register_call_livekit_provider_creates_conversation(client: AsyncClient) -> None:
    resp = await client.post(
        "/telephony/register-call",
        json={"provider": "livekit", "provider_call_sid": "room-abc123",
              "campaign_id": "c1"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["call_id"].startswith("call_")
    assert body["status"] == "in_progress"


async def test_register_call_unsupported_provider_400(client: AsyncClient) -> None:
    resp = await client.post(
        "/telephony/register-call",
        json={"provider": "stringee", "provider_call_sid": "s1"},
    )
    assert resp.status_code == 400


async def test_register_call_unknown_campaign_id_returns_400(client: AsyncClient) -> None:
    resp = await client.post(
        "/telephony/register-call",
        json={"provider": "twilio", "provider_call_sid": "CA999",
              "campaign_id": "does-not-exist"},
    )
    assert resp.status_code == 400
