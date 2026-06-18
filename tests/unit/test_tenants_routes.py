"""Route tests for Register Tenant (POST /api/v1/tenants)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import tenants
from src.api.deps import get_db_session
from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.db_resolver import DbTenantResolver
from src.auth.middleware import set_admin_tokens, set_tenant_resolver
from src.models.database import Base
from src.models.tenant import TenantSecret

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    monkeypatch.setenv("VOX_SECRET_KEY", crypto.generate_key())
    crypto.reset_cache_for_tests()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async def _session_override():
        async with sm() as session:
            yield session

    resolver = DbTenantResolver(sm)
    await resolver.reload()
    set_tenant_resolver(resolver)
    set_admin_tokens(["admin-token"])

    app = FastAPI()
    app.state.tenant_resolver = resolver
    app.state.tenants = resolver.loaded_settings()
    app.include_router(tenants.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, resolver, sm
    set_tenant_resolver(None)
    set_admin_tokens([])
    crypto.reset_cache_for_tests()
    await engine.dispose()


def _body(**over):
    base = {
        "name": "Acme Telecom",
        "max_concurrent_calls": 3,
        "stt": {"provider": "groq", "model": "whisper-large-v3"},
        "llm": {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
        "tts": {"provider": "sarvam", "model": "bulbul:v3", "voice_id": "anushka", "language": "hi-IN"},
        "telephony": {
            "provider": "twilio",
            "from_number": "+15705255679",
            "keys": {"account_sid": "ACxxx", "auth_token": "tok-secret"},
            "phone_numbers": ["+15705255679"],
        },
    }
    base.update(over)
    return base


async def test_register_returns_token_and_id(ctx) -> None:
    client, _, _ = ctx
    resp = await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)
    assert resp.status_code == 201
    body = resp.json()
    assert body["tenant_id"].startswith("t_")
    assert body["slug"] == "acme-telecom"
    assert body["api_token"].startswith("vox_")


async def test_register_requires_admin(ctx) -> None:
    client, _, _ = ctx
    assert (await client.post("/tenants", json=_body())).status_code == 401


async def test_update_tenant_fixes_stringee_keys_and_resolver(ctx) -> None:
    """PATCH updates telephony creds, re-encrypts secrets, and refreshes the
    resolver so a corrected Stringee API Key SID takes effect immediately."""
    import jwt

    from src.auth.context import hash_api_token
    from src.providers.telephony.softphone import mint_browser_credentials

    client, resolver, _ = ctx
    # register a Stringee tenant with a WRONG api key sid
    body = _body(slug="dev")
    body["telephony"] = {
        "provider": "stringee",
        "from_number": "+15705255679",
        "keys": {"account_sid": "WRONG_SID", "auth_token": "old-secret"},
    }
    reg = (await client.post("/tenants", json=body, headers=ADMIN_HEADERS)).json()
    token = reg["api_token"]
    tid = reg["tenant_id"]

    # the wrong sid is what the client token would carry
    tctx = await resolver.resolve_by_token(hash_api_token(token))
    creds = mint_browser_credentials(tctx, "agent-1")
    assert jwt.decode(creds.token, options={"verify_signature": False})["iss"] == "WRONG_SID"

    # PATCH to the correct Stringee API Key SID
    resp = await client.patch(
        f"/tenants/{tid}",
        json={"telephony": {"keys": {"account_sid": "RIGHT_SID", "auth_token": "new-secret"}}},
        headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    out = resp.json()
    assert out["telephony_provider"] == "stringee"
    assert "account_sid_env" in out["telephony_creds_configured"]

    # resolver now serves the corrected key → minted token carries RIGHT_SID
    tctx2 = await resolver.resolve_by_token(hash_api_token(token))
    creds2 = mint_browser_credentials(tctx2, "agent-1")
    claims = jwt.decode(creds2.token, "new-secret", algorithms=["HS256"],
                        options={"verify_exp": False})
    assert claims["iss"] == "RIGHT_SID"


async def test_update_tenant_stores_stringee_user_id(ctx) -> None:
    """The backoffice Telephony tab can set the Stringee outbound ``user_id``
    (stored as user_id_env). Without it the callout degrades to phone->phone
    external and the Answer URL/SCCO never runs."""
    client, _, _ = ctx
    body = _body(slug="dev")
    body["telephony"] = {
        "provider": "stringee", "from_number": "+15705255679",
        "keys": {"account_sid": "SID", "auth_token": "secret"},
    }
    reg = (await client.post("/tenants", json=body, headers=ADMIN_HEADERS)).json()
    tid = reg["tenant_id"]

    resp = await client.patch(
        f"/tenants/{tid}",
        json={"telephony": {"keys": {"user_id": "dev"}}},
        headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    assert "user_id_env" in resp.json()["telephony_creds_configured"]


async def test_update_tenant_status_and_404(ctx) -> None:
    client, _, _ = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]
    resp = await client.patch(
        f"/tenants/{tid}", json={"status": "suspended"}, headers=ADMIN_HEADERS)
    assert resp.status_code == 200 and resp.json()["status"] == "suspended"
    # unknown tenant
    assert (await client.patch(
        "/tenants/nope", json={"status": "active"}, headers=ADMIN_HEADERS)).status_code == 404
    # requires admin
    assert (await client.patch(f"/tenants/{tid}", json={"status": "active"})).status_code == 401


async def test_register_softphone_keys_mint_browser_token(ctx) -> None:
    """Registering Twilio softphone keys persists the *_env references and the
    resolver decrypts them so a browser AccessToken can be minted end-to-end."""
    import jwt

    from src.providers.telephony.softphone import mint_browser_credentials

    client, resolver, _ = ctx
    body = _body(slug="acme")
    body["telephony"]["keys"].update({
        "api_key_sid": "SKxxx", "api_key_secret": "sk-secret", "twiml_app_sid": "APxxx",
    })
    resp = await client.post("/tenants", json=body, headers=ADMIN_HEADERS)
    assert resp.status_code == 201
    token = resp.json()["api_token"]

    tctx = await resolver.resolve_by_token(hash_api_token(token))
    tel = tctx.settings.pipeline.telephony
    assert tel.api_key_sid_env and tel.api_key_secret_env and tel.twiml_app_sid_env

    creds = mint_browser_credentials(tctx, "agent-1")
    claims = jwt.decode(creds.token, options={"verify_signature": False})
    assert claims["iss"] == "SKxxx"
    assert claims["grants"]["voice"]["outgoing"]["application_sid"] == "APxxx"


async def test_register_issued_token_resolves(ctx) -> None:
    client, resolver, _ = ctx
    token = (await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)).json()["api_token"]
    new_ctx = await resolver.resolve_by_token(hash_api_token(token))
    assert new_ctx is not None
    assert new_ctx.settings.max_concurrent_calls == 3


async def test_register_telephony_keys_encrypted_only(ctx) -> None:
    client, _, sm = ctx
    resp = await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)
    tenant_id = resp.json()["tenant_id"]
    async with sm() as s:
        rows = (await s.execute(
            select(TenantSecret).where(TenantSecret.tenant_id == tenant_id)
        )).scalars().all()
    assert len(rows) == 2
    assert all("tok-secret" not in r.value_encrypted and "ACxxx" not in r.value_encrypted
               for r in rows)
    token_row = next(r for r in rows if r.name.endswith("AUTH_TOKEN"))
    assert crypto.decrypt(token_row.value_encrypted) == "tok-secret"


async def test_register_telephony_secret_resolves_via_context(ctx) -> None:
    client, resolver, _ = ctx
    token = (await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)).json()["api_token"]
    new_ctx = await resolver.resolve_by_token(hash_api_token(token))
    tel = new_ctx.settings.pipeline.telephony
    assert new_ctx.secret(tel.auth_token_env) == "tok-secret"
    # STT api_key_env points at the shared master env var name, not stored per tenant.
    assert new_ctx.settings.pipeline.stt.api_key_env == "GROQ_API_KEY"


async def test_register_persists_model_choices(ctx) -> None:
    client, resolver, _ = ctx
    token = (await client.post("/tenants", json=_body(), headers=ADMIN_HEADERS)).json()["api_token"]
    new_ctx = await resolver.resolve_by_token(hash_api_token(token))
    p = new_ctx.settings.pipeline
    assert p.stt.model == "whisper-large-v3"
    assert p.llm.model == "gemini-2.5-flash-lite"   # a non-default variant survives
    assert p.tts.model == "bulbul:v3"               # TTS model now persisted too


async def test_register_duplicate_slug_409(ctx) -> None:
    client, _, _ = ctx
    await client.post("/tenants", json=_body(slug="dup"), headers=ADMIN_HEADERS)
    resp = await client.post("/tenants", json=_body(slug="dup"), headers=ADMIN_HEADERS)
    assert resp.status_code == 409


async def test_list_tenants_shows_mode_and_models(ctx) -> None:
    client, _, _ = ctx
    await client.post("/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)
    resp = await client.get("/tenants", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    acme = next(t for t in body["tenants"] if t["slug"] == "acme")
    assert acme["mode"] == "layered"
    assert acme["llm"]["provider"] == "gemini"
    assert acme["llm"]["model"] == "gemini-2.5-flash-lite"
    assert acme["tts"]["model"] == "bulbul:v3"
    assert acme["telephony_provider"] == "twilio"
    # Non-secret telephony config surfaces so the backoffice can prefill it, and
    # the names (NOT values) of configured creds show what's set.
    assert acme["telephony_from_number"] == "+15705255679"
    assert "account_sid_env" in acme["telephony_creds_configured"]
    assert "auth_token_env" in acme["telephony_creds_configured"]


async def test_list_tenants_requires_admin(ctx) -> None:
    client, _, _ = ctx
    assert (await client.get("/tenants")).status_code == 401


async def test_tenant_analytics_and_billing(ctx) -> None:
    client, _, sm = ctx
    tid = (await client.post("/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]

    # seed a couple of finished conversations + a telephony rate
    from src.models.conversation import Conversation
    from src.models.tenant import ProviderCost
    async with sm() as s:
        s.add(ProviderCost(kind="telephony", provider="twilio", model="", cost_per_min=0.10))
        s.add(Conversation(
            id="c1", tenant_id=tid, agent_type="voicebot", channel="voice", status="ended",
            outcome="interested", pipeline_config={}, provider_call_sid="s1",
            telephony_provider="twilio", cost=0.06, duration_ms=120_000))
        s.add(Conversation(
            id="c2", tenant_id=tid, agent_type="voicebot", channel="voice", status="ended",
            outcome="not_interested", pipeline_config={}, provider_call_sid="s2",
            telephony_provider="twilio", cost=0.03, duration_ms=60_000))
        # a call with no outcome (ended before analysis) — must show as no_outcome
        s.add(Conversation(
            id="c3", tenant_id=tid, agent_type="voicebot", channel="webconsole", status="ended",
            outcome=None, pipeline_config={}, provider_call_sid="s3",
            cost=0.02, duration_ms=30_000))
        # a manual (human softphone) call — must be counted under by_agent_type
        s.add(Conversation(
            id="c4", tenant_id=tid, agent_type="human", channel="softphone", status="ended",
            outcome="callback_requested", pipeline_config={}, provider_call_sid="s4",
            telephony_provider="twilio", cost=0.01, duration_ms=45_000))
        await s.commit()

    an = (await client.get(f"/tenants/{tid}/analytics", headers=ADMIN_HEADERS)).json()
    assert an["total_calls"] == 4
    assert an["by_status"]["ended"] == 4
    assert an["by_outcome"]["interested"] == 1
    assert an["by_outcome"]["no_outcome"] == 1            # null outcome counted
    # manual vs AI breakdown
    assert an["by_agent_type"]["voicebot"] == 3
    assert an["by_agent_type"]["human"] == 1
    # all breakdowns total to total_calls (the bug: they used to mismatch)
    assert sum(an["by_status"].values()) == an["total_calls"]
    assert sum(an["by_outcome"].values()) == an["total_calls"]
    assert sum(an["by_agent_type"].values()) == an["total_calls"]
    assert an["total_duration_ms"] == 255_000

    bill = (await client.get(f"/tenants/{tid}/billing", headers=ADMIN_HEADERS)).json()
    assert bill["total_calls"] == 4
    assert bill["platform_cost"] == pytest.approx(0.12)        # 0.06 + 0.03 + 0.02 + 0.01, telephony excluded
    assert bill["billable_minutes"] == pytest.approx(4.25)
    # tentative telephony: 0.10/min * (2 + 1 + 0.75) min = 0.375 (c3 has no telephony)
    assert bill["tentative_telephony_cost"] == pytest.approx(0.375)


async def test_tenant_analytics_unknown_404(ctx) -> None:
    client, _, _ = ctx
    assert (await client.get("/tenants/nope/analytics", headers=ADMIN_HEADERS)).status_code == 404


async def test_register_s2s_mode(ctx) -> None:
    client, resolver, _ = ctx
    body = _body(
        mode="s2s",
        realtime={"provider": "gemini_live", "model": "gemini-3.1-flash-live-preview",
                  "voice": "Aoede", "language_code": "hi-IN"},
    )
    resp = await client.post("/tenants", json=body, headers=ADMIN_HEADERS)
    assert resp.status_code == 201
    new_ctx = await resolver.resolve_by_slug(resp.json()["slug"])
    assert new_ctx.settings.pipeline.mode == "s2s"
    assert new_ctx.settings.pipeline.realtime.api_key_env == "GEMINI_API_KEY"
