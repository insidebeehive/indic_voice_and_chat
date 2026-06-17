"""Route tests for Call Lead + GET /calls/{id}."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import calls
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import (
    TenantLLMConfig,
    TenantPipelineConfig,
    TenantSettings,
    TenantSTTConfig,
    TenantTelephonyConfig,
    TenantTTSConfig,
)
from src.models.campaign import Campaign as DbCampaign
from src.models.conversation import Conversation
from src.models.database import Base
from src.models.tenant import Tenant

HEADERS = {"Authorization": "Bearer test-token"}


class _FakeSession:
    def __init__(self, sid: str) -> None:
        self.session_id = sid


class _FakeAdapter:
    def __init__(self, sid: str = "SID-CALL-1") -> None:
        self._sid = sid

    async def initiate_call(self, cfg):  # noqa: ANN001
        return _FakeSession(self._sid)


def _tenant(max_concurrent: int = 2) -> TenantSettings:
    return TenantSettings(
        id="t1", slug="t1", name="T1", max_concurrent_calls=max_concurrent,
        pipeline=TenantPipelineConfig(
            mode="layered",
            stt=TenantSTTConfig(provider="groq"),
            llm=TenantLLMConfig(provider="gemini"),
            tts=TenantTTSConfig(provider="sarvam", voice_id="anushka"),
            telephony=TenantTelephonyConfig(
                provider="twilio", from_number="+15705255679",
                webhook_base_url="https://x.example/api/v1/telephony",
            ),
        ),
    )


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    monkeypatch.setattr(calls, "get_telephony_provider", lambda cfg: _FakeAdapter())

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        s.add(DbCampaign(id="c1", tenant_id="t1", name="C1", status="active", config_yaml=""))
        s.add(DbCampaign(id="c-ended", tenant_id="t1", name="Old", status="ended", config_yaml=""))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(_tenant(), plaintext_tokens=["test-token"])

    app = FastAPI()
    app.include_router(calls.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=HEADERS) as c:
        yield c, sm
    set_tenant_resolver(None)
    await engine.dispose()


async def test_call_lead_places_and_records(ctx) -> None:
    client, sm = ctx
    resp = await client.post("/campaigns/c1/calls", json={"to_number": "+918618795697"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["call_id"].startswith("call_")
    assert body["provider_call_sid"] == "SID-CALL-1"
    assert body["status"] == "in_progress"

    # The conversation row records the config used.
    async with sm() as s:
        row = await s.get(Conversation, body["call_id"])
    assert row.status == "in_progress"
    assert row.provider_call_sid == "SID-CALL-1"
    assert row.mode == "layered"
    assert row.stt_provider == "groq"
    assert row.llm_provider == "gemini"
    assert row.tts_provider == "sarvam"
    assert row.telephony_provider == "twilio"
    assert row.voice == "anushka"
    assert row.campaign_id == "c1"


async def test_call_lead_dials_with_tenant_creds(monkeypatch) -> None:
    # The outbound adapter must be built with the TENANT's telephony creds (so the
    # call bills/identifies as the tenant), not the platform env.
    monkeypatch.setenv("TENANT_T1_STRINGEE_SID", "ACTUAL-SID")
    monkeypatch.setenv("TENANT_T1_STRINGEE_SECRET", "ACTUAL-SECRET")
    captured: dict = {}
    monkeypatch.setattr(calls, "get_telephony_provider",
                        lambda cfg: captured.update(cfg) or _FakeAdapter())

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        s.add(DbCampaign(id="c1", tenant_id="t1", name="C1", status="active", config_yaml=""))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    tenant = TenantSettings(
        id="t1", slug="t1", name="T1", max_concurrent_calls=2,
        pipeline=TenantPipelineConfig(
            mode="layered", stt=TenantSTTConfig(provider="groq"),
            llm=TenantLLMConfig(provider="gemini"), tts=TenantTTSConfig(provider="sarvam"),
            telephony=TenantTelephonyConfig(
                provider="stringee", from_number="918204268005",
                webhook_base_url="https://x.example/api/v1/telephony",
                account_sid_env="TENANT_T1_STRINGEE_SID",
                auth_token_env="TENANT_T1_STRINGEE_SECRET")))
    set_tenant_resolver(None)
    register_tenant_for_test(tenant, plaintext_tokens=["test-token"])
    app = FastAPI()
    app.include_router(calls.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test", headers=HEADERS) as c:
            resp = await c.post("/campaigns/c1/calls", json={"to_number": "+918618795697"})
    finally:
        set_tenant_resolver(None)
        await engine.dispose()

    assert resp.status_code == 202
    assert captured["account_sid"] == "ACTUAL-SID"
    assert captured["auth_token"] == "ACTUAL-SECRET"
    # Mapped onto the keys the Stringee server adapter reads (not STRINGEE_* env).
    assert captured["api_key_sid"] == "ACTUAL-SID"
    assert captured["api_key_secret"] == "ACTUAL-SECRET"


async def test_call_lead_inactive_campaign_409(ctx) -> None:
    client, _ = ctx
    resp = await client.post("/campaigns/c-ended/calls", json={"to_number": "+9118"})
    assert resp.status_code == 409


async def test_call_lead_unknown_campaign_404(ctx) -> None:
    client, _ = ctx
    resp = await client.post("/campaigns/nope/calls", json={"to_number": "+9118"})
    assert resp.status_code == 404


async def test_call_lead_concurrency_cap_429(ctx) -> None:
    client, sm = ctx
    # Cap is 2; pre-load 2 in-progress calls so the next is rejected.
    async with sm() as s:
        for i in range(2):
            s.add(Conversation(
                id=f"pre{i}", tenant_id="t1", agent_type="voicebot", channel="voice",
                status="in_progress", pipeline_config={}, provider_call_sid=f"pre-sid-{i}"))
        await s.commit()
    resp = await client.post("/campaigns/c1/calls", json={"to_number": "+9118"})
    assert resp.status_code == 429


async def test_get_call_returns_status(ctx) -> None:
    client, _ = ctx
    call_id = (await client.post(
        "/campaigns/c1/calls", json={"to_number": "+918618795697"})).json()["call_id"]
    resp = await client.get(f"/calls/{call_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["call_id"] == call_id
    assert body["status"] == "in_progress"
    assert body["outcome"] is None


async def test_get_call_unknown_404(ctx) -> None:
    client, _ = ctx
    assert (await client.get("/calls/missing")).status_code == 404


async def test_get_call_cross_tenant_404(ctx) -> None:
    client, sm = ctx
    # A call owned by another tenant must 404 for t1.
    async with sm() as s:
        s.add(Tenant(id="t2", slug="t2", name="T2"))
        s.add(Conversation(
            id="other-call", tenant_id="t2", agent_type="voicebot", channel="voice",
            status="in_progress", pipeline_config={}, provider_call_sid="x"))
        await s.commit()
    assert (await client.get("/calls/other-call")).status_code == 404


async def test_call_lead_webconsole_returns_helpful_409(ctx) -> None:
    client, sm = ctx
    # A tenant whose telephony is 'webconsole' has no outbound dialing.
    register_tenant_for_test(
        TenantSettings(
            id="t_wc", slug="wc", name="WC", max_concurrent_calls=2,
            pipeline=TenantPipelineConfig(
                stt=TenantSTTConfig(provider="groq"), llm=TenantLLMConfig(provider="gemini"),
                tts=TenantTTSConfig(provider="sarvam", voice_id="anushka"),
                telephony=TenantTelephonyConfig(provider="webconsole"),
            ),
        ),
        plaintext_tokens=["wc-token"],
    )
    async with sm() as s:
        s.add(Tenant(id="t_wc", slug="wc", name="WC"))
        s.add(DbCampaign(id="wc1", tenant_id="t_wc", name="WC", status="active", config_yaml=""))
        await s.commit()
    resp = await client.post(
        "/campaigns/wc1/calls", json={"to_number": "+9118"},
        headers={"Authorization": "Bearer wc-token"})
    assert resp.status_code == 409
    assert "webconsole" in resp.json()["detail"]
    assert "/console" in resp.json()["detail"]


async def test_call_lead_requires_auth(ctx) -> None:
    client, _ = ctx
    resp = await client.post(
        "/campaigns/c1/calls", json={"to_number": "+9118"}, headers={"Authorization": ""})
    assert resp.status_code == 401
