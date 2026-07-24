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
from src.models.tenant import Tenant, TenantSecret

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


async def test_update_tenant_stores_events_webhook_secret(ctx) -> None:
    """The events-webhook signing secret is entered as a VALUE and stored encrypted
    (not an env-var name), so we can sign outbound call-event POSTs with it."""
    client, resolver, _ = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]

    resp = await client.patch(
        f"/tenants/{tid}",
        json={"events_webhook_url": "https://crm.example/events",
              "telephony": {"keys": {"events_webhook_secret": "s3cret-value"}}},
        headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["events_webhook_url"] == "https://crm.example/events"
    assert body["events_webhook_secret_set"] is True

    ctx2 = await resolver.resolve_by_slug("acme")
    assert ctx2.settings.events_webhook_url == "https://crm.example/events"
    assert ctx2.settings.events_webhook_secret_env           # a derived NAME, set via keys
    assert ctx2.secret_optional(ctx2.settings.events_webhook_secret_env) == "s3cret-value"


_CAMP_YAML = """
campaign:
  id: c1
  name: "Old Name"
  status: active
  agent:
    name: "Anaaya"
    company: "OldCo"
    role: "Sales"
  script:
    greeting: "Hi from OldCo"
    talking_points:
      - "point one"
    knowledge:
      safety: "safe"
  slots:
    interest_level:
      type: enum
      values: [hot, cold]
"""


async def _seed_campaign(sm, tenant_id, cid="c1"):
    from src.models.campaign import Campaign
    async with sm() as s:
        s.add(Campaign(id=cid, tenant_id=tenant_id, name="Old Name",
                       status="active", config_yaml=_CAMP_YAML))
        await s.commit()


async def test_list_tenant_campaigns_returns_structured_script(ctx) -> None:
    client, _, sm = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]
    await _seed_campaign(sm, tid)

    r = await client.get(f"/tenants/{tid}/campaigns", headers=ADMIN_HEADERS)
    assert r.status_code == 200
    camp = r.json()["campaigns"][0]
    assert camp["id"] == "c1" and camp["status"] == "active"
    sc = camp["script"]
    assert sc["company"] == "OldCo"
    assert sc["greeting"] == "Hi from OldCo"
    assert sc["talking_points"] == ["point one"]
    assert sc["knowledge"]["safety"] == "safe"


async def test_update_campaign_script_preserves_slots_and_persists(ctx) -> None:
    import yaml
    from src.models.campaign import Campaign

    client, _, sm = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]
    await _seed_campaign(sm, tid)

    r = await client.put(
        f"/tenants/{tid}/campaigns/c1",
        json={"company": "XYZ", "greeting": "Hi from {company_name}",
              "talking_points": ["p1", "p2"]},
        headers=ADMIN_HEADERS)
    assert r.status_code == 200, r.text

    g = (await client.get(f"/tenants/{tid}/campaigns", headers=ADMIN_HEADERS)).json()["campaigns"][0]
    assert g["script"]["company"] == "XYZ"
    assert g["script"]["greeting"] == "Hi from {company_name}"
    assert g["script"]["talking_points"] == ["p1", "p2"]

    # unmodeled keys (slots) and untouched fields (agent.name) are preserved
    async with sm() as s:
        row = (await s.execute(select(Campaign).where(Campaign.id == "c1"))).scalar_one()
    camp = yaml.safe_load(row.config_yaml)["campaign"]
    assert "interest_level" in camp["slots"]
    assert camp["agent"]["name"] == "Anaaya"


async def test_update_campaign_cross_tenant_404(ctx) -> None:
    client, _, sm = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]
    await _seed_campaign(sm, "t_other")   # campaign belongs to a different tenant
    r = await client.put(f"/tenants/{tid}/campaigns/c1",
                         json={"company": "X"}, headers=ADMIN_HEADERS)
    assert r.status_code == 404


async def test_update_and_list_tenant_stringee_base_url(ctx) -> None:
    """Per-tenant regional Stringee REST host is settable + surfaced for prefill."""
    client, _, _ = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]
    r = await client.patch(
        f"/tenants/{tid}",
        json={"telephony": {"stringee_base_url": "https://asia-2.api.stringee.com"}},
        headers=ADMIN_HEADERS)
    assert r.status_code == 200
    acme = next(t for t in (await client.get("/tenants", headers=ADMIN_HEADERS)).json()["tenants"]
                if t["tenant_id"] == tid)
    assert acme["telephony_stringee_base_url"] == "https://asia-2.api.stringee.com"


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
    # STT always resolves its key from the platform master env var at runtime
    # (never per-tenant), so register_tenant leaves api_key_env unset.
    assert new_ctx.settings.pipeline.stt.api_key_env is None


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


async def test_register_with_crm_id_links_tenant_to_crm(ctx) -> None:
    """Registration can link the new tenant to a shared Crm row in one step
    (previously only possible via a follow-up PATCH .../tenants/{id})."""
    client, _, sm = ctx
    from src.models.crm import Crm

    async with sm() as s:
        s.add(Crm(id="betstudio", name="BetStudio", base_url="https://crm.example.com"))
        await s.commit()

    resp = await client.post(
        "/tenants", json=_body(slug="acme", crm_id="betstudio"), headers=ADMIN_HEADERS)
    assert resp.status_code == 201
    tid = resp.json()["tenant_id"]

    async with sm() as s:
        t = await s.get(Tenant, tid)
    assert t.crm_id == "betstudio"


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
    # Per-tenant Stringee webhook URLs (slug in the path) to paste into the
    # tenant's Stringee project so calls attribute to the right tenant.
    assert acme["stringee_softphone_answer_url"].endswith("/stringee/softphone-answer/acme")
    assert acme["stringee_answer_url"].endswith("/stringee/answer/acme")


async def test_list_tenants_requires_admin(ctx) -> None:
    client, _, _ = ctx
    assert (await client.get("/tenants")).status_code == 401


async def test_tenant_analytics_and_billing(ctx) -> None:
    client, _, sm = ctx
    tid = (await client.post("/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]

    # seed a couple of finished conversations + a telephony rate
    from src.models.campaign import Campaign
    from src.models.conversation import Conversation
    from src.models.tenant import ProviderCost
    async with sm() as s:
        s.add(ProviderCost(kind="telephony", provider="twilio", model="", cost_per_min=0.10))
        s.add(Campaign(id="camp1", tenant_id=tid, name="Promo", status="active", config_yaml=""))
        s.add(Conversation(
            id="c1", tenant_id=tid, agent_type="voicebot", channel="voice", status="ended",
            outcome="interested", pipeline_config={}, provider_call_sid="s1", campaign_id="camp1",
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
    # channel / provider / campaign breakdowns
    assert an["by_channel"] == {"voice": 2, "webconsole": 1, "softphone": 1}
    assert an["by_provider"]["twilio"] == 3 and an["by_provider"]["none"] == 1
    assert an["by_campaign"]["Promo"] == 1 and an["by_campaign"]["none"] == 3  # campaign_id→name
    # all breakdowns total to total_calls (the bug: they used to mismatch)
    for key in ("by_status", "by_outcome", "by_agent_type", "by_channel", "by_provider", "by_campaign"):
        assert sum(an[key].values()) == an["total_calls"], key
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


async def test_update_and_read_back_crm_x_api_key_masked(ctx) -> None:
    """The new, independent crm:x_api_key secret can be set via PATCH and is
    read back masked (never the raw value) via GET .../chat-config."""
    client, _, sm = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]

    resp = await client.patch(
        f"/tenants/{tid}",
        json={"crm": {"auth_type": "bearer", "api_token": "bearer-tok",
                       "x_api_key": "the-x-api-key"}},
        headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text

    cfg = (await client.get(f"/tenants/{tid}/chat-config", headers=ADMIN_HEADERS)).json()
    assert cfg["crm"]["x_api_key"] == "..."
    assert cfg["crm"]["api_token"] == "..."

    async with sm() as s:
        rows = (await s.execute(
            select(TenantSecret).where(TenantSecret.tenant_id == tid)
        )).scalars().all()
    x_row = next(r for r in rows if r.name == "crm:x_api_key")
    assert crypto.decrypt(x_row.value_encrypted) == "the-x-api-key"
    assert "the-x-api-key" not in x_row.value_encrypted


async def test_crm_x_api_key_not_set_reports_empty_string(ctx) -> None:
    client, _, _ = ctx
    tid = (await client.post(
        "/tenants", json=_body(slug="acme"), headers=ADMIN_HEADERS)).json()["tenant_id"]
    cfg = (await client.get(f"/tenants/{tid}/chat-config", headers=ADMIN_HEADERS)).json()
    assert cfg["crm"]["x_api_key"] == ""


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
    # Realtime always resolves its key from the platform master env var at
    # runtime (never per-tenant), so register_tenant leaves api_key_env unset.
    assert new_ctx.settings.pipeline.realtime.api_key_env is None
