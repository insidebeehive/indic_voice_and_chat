"""Route tests for Call Lead + GET /calls/{id}."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import calls
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.audit import reset_suppression_state
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


@pytest.fixture(autouse=True)
def _reset_audit_suppression_state():
    """log_denied's per-(reason, client_ip) suppression window is module-level
    state — without resetting it, one test's rejection log gets silently
    suppressed by a prior test's rejections for the same reason."""
    reset_suppression_state()
    yield
    reset_suppression_state()


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


async def test_call_lead_twilio_answer_url_is_slug_scoped(ctx, monkeypatch) -> None:
    # Campaign outbound (like the dev console) must slug-scope the answer URL so
    # the bridge resolves the tenant by slug — not by reverse-looking-up the
    # caller-ID (which isn't in tenant_phone_numbers). The tenant here is twilio.
    client, sm = ctx
    captured: dict = {}

    class _Cap(_FakeAdapter):
        async def initiate_call(self, cfg):  # noqa: ANN001
            captured["webhook_url"] = cfg.webhook_url
            return await super().initiate_call(cfg)

    monkeypatch.setattr(calls, "get_telephony_provider", lambda cfg: _Cap())
    resp = await client.post("/campaigns/c1/calls", json={"to_number": "+918618795697"})
    assert resp.status_code == 202, resp.text
    assert captured["webhook_url"].endswith("/twilio/voice/t1")


async def test_call_lead_dials_with_tenant_creds(monkeypatch) -> None:
    # The outbound adapter must be built with the TENANT's telephony creds (so the
    # call bills/identifies as the tenant), not the platform env.
    monkeypatch.setenv("TENANT_T1_STRINGEE_SID", "ACTUAL-SID")
    monkeypatch.setenv("TENANT_T1_STRINGEE_SECRET", "ACTUAL-SECRET")
    monkeypatch.setenv("TENANT_T1_STRINGEE_USER", "dev")
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
                account_sid_env="TENANT_T1_STRINGEE_SID",
                auth_token_env="TENANT_T1_STRINGEE_SECRET",
                user_id_env="TENANT_T1_STRINGEE_USER")))
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
    # Stringee userId — without it the callout degrades to phone->phone external
    # and the Answer URL/SCCO never runs (silent bot).
    assert captured["user_id"] == "dev"


async def _app_with_registry(registry):
    """A call_lead app with a given runtime registry wired (for the compliance gate)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        s.add(DbCampaign(id="c1", tenant_id="t1", name="C1", status="active", config_yaml=""))
        await s.commit()

    async def _override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(_tenant(), plaintext_tokens=["test-token"])
    app = FastAPI()
    app.include_router(calls.router)
    app.dependency_overrides[get_db_session] = _override
    app.state.registry = registry
    return app, engine


def _dnd_registry(*, can_call, blocked):
    from src.bootstrap import TenantDnd
    return SimpleNamespace(dnd=SimpleNamespace(get=lambda t: TenantDnd(
        filter=SimpleNamespace(is_blocked=lambda n: blocked),
        hours=SimpleNamespace(can_call_now=lambda: can_call))))


async def test_call_lead_blocked_outside_calling_hours() -> None:
    app, engine = await _app_with_registry(_dnd_registry(can_call=False, blocked=False))
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test", headers=HEADERS) as c:
            resp = await c.post("/campaigns/c1/calls", json={"to_number": "+918618795697"})
    finally:
        set_tenant_resolver(None)
        await engine.dispose()
    assert resp.status_code == 403
    assert "calling hours" in resp.json()["detail"]


async def test_call_lead_blocked_dnd_number() -> None:
    app, engine = await _app_with_registry(_dnd_registry(can_call=True, blocked=True))
    try:
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test", headers=HEADERS) as c:
            resp = await c.post("/campaigns/c1/calls", json={"to_number": "+918618795697"})
    finally:
        set_tenant_resolver(None)
        await engine.dispose()
    assert resp.status_code == 403
    assert "DND" in resp.json()["detail"]


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


# --- POST /calls/{sid}/transfer-result — H2 cross-tenant fix -------------


async def test_transfer_result_cross_tenant_blocked_and_future_stays_pending(ctx) -> None:
    """Tenant B must not be able to resolve tenant A's pending transfer via the
    HTTP route — and, critically, A's future must still be pending/undone
    afterwards (proves the hijack is actually blocked, not just that the HTTP
    response looks like a 404)."""
    from src.api import transfer_store

    client, _ = ctx
    fut = transfer_store.register("t1", "SID-HIJACK")
    try:
        register_tenant_for_test(
            TenantSettings(
                id="t2", slug="t2", name="T2", max_concurrent_calls=2,
                pipeline=TenantPipelineConfig(
                    mode="layered",
                    stt=TenantSTTConfig(provider="groq"),
                    llm=TenantLLMConfig(provider="gemini"),
                    tts=TenantTTSConfig(provider="sarvam"),
                    telephony=TenantTelephonyConfig(provider="twilio", from_number="+1"),
                ),
            ),
            plaintext_tokens=["t2-token"],
        )
        resp = await client.post(
            "/calls/SID-HIJACK/transfer-result", json={"status": "success"},
            headers={"Authorization": "Bearer t2-token"})
        assert resp.status_code == 404
        assert "for this tenant" in resp.json()["detail"]
        assert not fut.done()
    finally:
        transfer_store.cancel_pending("t1", "SID-HIJACK")


async def test_transfer_result_own_tenant_resolves(ctx) -> None:
    from src.api import transfer_store

    client, _ = ctx
    fut = transfer_store.register("t1", "SID-OWN")
    resp = await client.post(
        "/calls/SID-OWN/transfer-result", json={"status": "success"}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved"] is True
    assert body["call_sid"] == "SID-OWN"
    assert body["status"] == "success"
    assert fut.done()
    assert fut.result() == "success"


async def test_transfer_result_unknown_call_sid_404(ctx) -> None:
    client, _ = ctx
    resp = await client.post(
        "/calls/does-not-exist/transfer-result", json={"status": "success"}, headers=HEADERS)
    assert resp.status_code == 404
    assert "for this tenant" in resp.json()["detail"]


# --- cross-tenant access-attempt audit logging ----------------------------


async def test_transfer_result_cross_tenant_logs_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    # The store is keyed by (tenant_id, call_sid), so a miss is structurally
    # indistinguishable between "no such SID" and "SID belongs to another
    # tenant" — found is always False here, single reason.
    from src.api import transfer_store

    client, _ = ctx
    fut = transfer_store.register("t1", "SID-HIJACK-2")
    try:
        register_tenant_for_test(
            TenantSettings(id="t2", slug="t2", name="T2"), plaintext_tokens=["t2-token-2"]
        )
        with caplog.at_level(logging.INFO):
            resp = await client.post(
                "/calls/SID-HIJACK-2/transfer-result", json={"status": "success"},
                headers={"Authorization": "Bearer t2-token-2"})
        assert resp.status_code == 404
        assert "for this tenant" in resp.json()["detail"]
        assert not fut.done()
        denied = [r for r in caplog.records
                  if getattr(r, "event", None) == "cross_tenant_access_denied"]
        assert len(denied) == 1, denied
        rec = denied[0]
        assert rec.reason == "transfer_not_pending_for_tenant"
        assert rec.levelno == logging.WARNING
        assert rec.found is False
        assert rec.resource == "transfer"
        assert rec.resource_id == "SID-HIJACK-2"
        assert rec.tenant == "t2"
    finally:
        transfer_store.cancel_pending("t1", "SID-HIJACK-2")


async def test_transfer_result_unknown_sid_logs_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    # No such SID at all — same reason/found=False (no found=True case is
    # possible for this route: the store's composite key makes "found but
    # wrong tenant" and "not found" indistinguishable, see comment above).
    client, _ = ctx
    with caplog.at_level(logging.INFO):
        resp = await client.post(
            "/calls/does-not-exist-2/transfer-result", json={"status": "success"},
            headers=HEADERS)
    assert resp.status_code == 404
    assert "for this tenant" in resp.json()["detail"]
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, denied
    assert denied[0].reason == "transfer_not_pending_for_tenant"
    assert denied[0].found is False


async def test_transfer_result_own_tenant_emits_no_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    from src.api import transfer_store

    client, _ = ctx
    transfer_store.register("t1", "SID-OWN-2")
    with caplog.at_level(logging.INFO):
        resp = await client.post(
            "/calls/SID-OWN-2/transfer-result", json={"status": "success"}, headers=HEADERS)
    assert resp.status_code == 200
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert denied == []


async def test_call_lead_cross_tenant_campaign_logs_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    # t1 owns c1; t2 probing it must 404 (unchanged) and log found=True.
    client, _ = ctx
    register_tenant_for_test(
        TenantSettings(id="t2", slug="t2", name="T2"), plaintext_tokens=["t2-token-3"]
    )
    with caplog.at_level(logging.INFO):
        resp = await client.post(
            "/campaigns/c1/calls", json={"to_number": "+9118"},
            headers={"Authorization": "Bearer t2-token-3"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "campaign not found"
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, denied
    rec = denied[0]
    assert rec.reason == "campaign_not_owned"
    assert rec.levelno == logging.WARNING
    assert rec.found is True
    assert rec.owner_tenant_id == "t1"
    assert rec.resource == "campaign"
    assert rec.resource_id == "c1"


async def test_call_lead_unknown_campaign_logs_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    client, _ = ctx
    with caplog.at_level(logging.INFO):
        resp = await client.post("/campaigns/nope/calls", json={"to_number": "+9118"})
    assert resp.status_code == 404
    assert resp.json()["detail"] == "campaign not found"
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, denied
    assert denied[0].reason == "campaign_not_found"
    assert denied[0].found is False
    assert denied[0].owner_tenant_id is None


async def test_call_lead_same_tenant_success_emits_no_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    client, _ = ctx
    with caplog.at_level(logging.INFO):
        resp = await client.post(
            "/campaigns/c1/calls", json={"to_number": "+918618795697"})
    assert resp.status_code == 202
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert denied == []


async def test_get_call_cross_tenant_logs_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    client, sm = ctx
    async with sm() as s:
        s.add(Tenant(id="t2b", slug="t2b", name="T2B"))
        s.add(Conversation(
            id="other-call-2", tenant_id="t2b", agent_type="voicebot", channel="voice",
            status="in_progress", pipeline_config={}, provider_call_sid="x2"))
        await s.commit()
    with caplog.at_level(logging.INFO):
        resp = await client.get("/calls/other-call-2")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "call not found"
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, denied
    rec = denied[0]
    assert rec.reason == "call_not_owned"
    assert rec.levelno == logging.WARNING
    assert rec.found is True
    assert rec.owner_tenant_id == "t2b"
    assert rec.resource == "conversation"
    assert rec.resource_id == "other-call-2"


async def test_get_call_unknown_logs_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    client, _ = ctx
    with caplog.at_level(logging.INFO):
        resp = await client.get("/calls/missing-2")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "call not found"
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert len(denied) == 1, denied
    assert denied[0].reason == "call_not_found"
    assert denied[0].found is False
    assert denied[0].owner_tenant_id is None


async def test_get_call_same_tenant_success_emits_no_denial(
    ctx, caplog: pytest.LogCaptureFixture
) -> None:
    client, _ = ctx
    call_id = (await client.post(
        "/campaigns/c1/calls", json={"to_number": "+918618795697"})).json()["call_id"]
    with caplog.at_level(logging.INFO):
        resp = await client.get(f"/calls/{call_id}")
    assert resp.status_code == 200
    denied = [r for r in caplog.records
              if getattr(r, "event", None) == "cross_tenant_access_denied"]
    assert denied == []
