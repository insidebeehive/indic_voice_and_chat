"""Tests for GET /chat/tools/resolved — the fresh, cache-bypassing view of a
tenant's ACTUALLY-resolved CRM tools (src/bootstrap.py resolve_crm_tools()).

Unlike GET /chat/tools (which only queries the chat_tools DB table), this
endpoint reports the crm_catalog path too (a tenant linked to a Crm entity via
``tenant.settings.crm_id`` — see docs/superpowers/plans/2026-07-23-crm-entity.md,
Task 3), since that's what a tenant with zero chat_tools rows actually gets
served on its next chat turn.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat_tools
from src.api.deps import get_db_session
from src.auth import TenantContext, register_tenant_for_test
from src.auth import secrets as crypto
from src.auth.middleware import set_tenant_resolver
from src.bootstrap import make_chatbot_factory, resolve_crm_tools
from src.chatbot.catalog import ALL_TOOLS, OPERATOR_TOOLS, PLAYER_TOOLS
from src.chatbot.tools import SUBMIT_DEPOSIT_VERIFICATION
from src.config_tenant import DepositVerificationConfig, TenantSettings
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


async def _seed_crm(sm, crm_id: str = "betstudio") -> None:
    """Seed one Crm + its full CrmTool catalog into ``sm`` — the DB-backed
    catalog a tenant's ``crm_id`` links to (tier 2 of resolve_crm_tools)."""
    from src.models.crm import Crm, CrmTool

    async with sm() as s:
        s.add(Crm(id=crm_id, name="BetStudio",
                   base_url="https://apistage.betstudio.io/api", auth_type="api_key"))
        for name, spec in ALL_TOOLS.items():
            s.add(CrmTool(crm_id=crm_id, name=name, description=spec["description"],
                           endpoint=spec["default_path"], method=spec.get("method", "GET"),
                           parameters=spec.get("parameters", {})))
        await s.commit()


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
    crm_tools = [t for t in body["tools"] if t["kind"] == "crm"]
    assert len(crm_tools) == 1
    t = crm_tools[0]
    assert t["name"] == "check_order_status"
    assert t["endpoint"] == _TOOL["endpoint"]
    assert t["auth_type"] == "bearer"
    assert t["token_configured"] is True


async def test_builtin_tools_always_present_even_with_no_crm_source(ctx) -> None:
    # Asserted against the literal name set, not BUILTIN_TOOLS itself — a
    # comparison against a copy of the same source the endpoint reads from
    # would pass even if both were wrong together (e.g. a tool silently
    # dropped from BUILTIN_TOOLS). These three are unconditionally added by
    # the chatbot factory (src/agents/chatbot.py) regardless of CRM
    # resolution — the resolved endpoint must report them even when nothing
    # else resolves (source == "none").
    client, _sm = ctx
    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "none"
    builtin = [t for t in body["tools"] if t["kind"] == "builtin"]
    assert {t["name"] for t in builtin} == {
        "search_knowledge_base", "escalate_to_human", "offer_voice_call",
    }
    assert all(t["endpoint"] == "" for t in builtin)
    assert all(t["auth_type"] is None for t in builtin)
    assert all(t["token_configured"] is False for t in builtin)
    assert all(t["x_api_key_configured"] is False for t in builtin)


async def test_crm_tools_reported_with_kind_crm(ctx) -> None:
    client, _sm = ctx
    await client.post("/chat/tools", json={"tools": [_TOOL]})

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    crm_tools = [t for t in body["tools"] if t["name"] == "check_order_status"]
    assert len(crm_tools) == 1
    assert crm_tools[0]["kind"] == "crm"


_DV_WEBHOOK_URL = "https://vendor.example.com/verify"
_DV_SECRET_ENV = "DV_WEBHOOK_SECRET"


def _dv_config(**overrides) -> DepositVerificationConfig:
    defaults = dict(
        enabled=True, webhook_url=_DV_WEBHOOK_URL,
        webhook_secret_env=_DV_SECRET_ENV, timeout_minutes=5,
    )
    defaults.update(overrides)
    return DepositVerificationConfig(**defaults)


async def test_deposit_verification_tool_present_when_registrable(ctx) -> None:
    client, _sm = ctx
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1", deposit_verification=_dv_config()),
        plaintext_tokens=["test-token"], secrets={_DV_SECRET_ENV: "s3cr3t"},
    )

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    dv = [t for t in body["tools"] if t["kind"] == "deposit_verification"]
    assert len(dv) == 1
    assert dv[0]["name"] == SUBMIT_DEPOSIT_VERIFICATION
    assert dv[0]["endpoint"] == _DV_WEBHOOK_URL
    assert dv[0]["token_configured"] is True
    assert dv[0]["x_api_key_configured"] is False


async def test_deposit_verification_tool_absent_when_secret_unresolvable(ctx) -> None:
    client, _sm = ctx
    # webhook_secret_env is set but no such secret was ever provisioned for
    # this tenant — mirrors the factory's own registration gate, which must
    # never offer the LLM a flow whose verdict callback can't be verified.
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1", deposit_verification=_dv_config()),
        plaintext_tokens=["test-token"],
    )

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert not any(t["kind"] == "deposit_verification" for t in body["tools"])


async def test_deposit_verification_tool_absent_when_disabled(ctx) -> None:
    client, _sm = ctx
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1",
                        deposit_verification=_dv_config(enabled=False)),
        plaintext_tokens=["test-token"], secrets={_DV_SECRET_ENV: "s3cr3t"},
    )

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert not any(t["kind"] == "deposit_verification" for t in body["tools"])


async def test_deposit_verification_tool_absent_when_no_webhook_url(ctx) -> None:
    # Enabled with a resolvable secret but no webhook_url — the callback has
    # nowhere to be verified against a vendor for, so the tool must stay
    # unregistered just like the disabled/unresolvable-secret cases.
    client, _sm = ctx
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1",
                        deposit_verification=_dv_config(webhook_url=None)),
        plaintext_tokens=["test-token"], secrets={_DV_SECRET_ENV: "s3cr3t"},
    )

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert not any(t["kind"] == "deposit_verification" for t in body["tools"])


async def test_resolved_tools_ordered_builtin_then_crm_then_dv(ctx) -> None:
    client, _sm = ctx
    await client.post("/chat/tools", json={"tools": [_TOOL]})
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1", deposit_verification=_dv_config()),
        plaintext_tokens=["test-token"], secrets={_DV_SECRET_ENV: "s3cr3t"},
    )

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    kinds = [t["kind"] for t in body["tools"]]
    assert kinds == ["builtin", "builtin", "builtin", "crm", "deposit_verification"]


async def test_endpoint_and_factory_agree_on_non_builtin_tool_names(ctx) -> None:
    # The endpoint must report exactly the non-builtin tools
    # make_chatbot_factory would actually hand the LLM this turn — resolved
    # independently here (by driving the factory itself, the same way
    # tests/unit/test_deposit_verification_executor.py does), not by
    # re-deriving the expected set from the endpoint's own code path, so a
    # bug in the endpoint's assembly (wrong gate, missed tool, stale cache)
    # would actually be caught instead of validated against itself.
    client, sm = ctx
    await client.post("/chat/tools", json={"tools": [_TOOL]})
    tenant_ctx = register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1", deposit_verification=_dv_config()),
        plaintext_tokens=["test-token"], secrets={_DV_SECRET_ENV: "s3cr3t"},
    )

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    endpoint_non_builtin = {t["name"] for t in body["tools"] if t["kind"] != "builtin"}

    registry = SimpleNamespace(
        providers=SimpleNamespace(get_llm=lambda t: object(), get_platform_llm=lambda: object()),
        retrievers=SimpleNamespace(get=lambda t: object()),
        session_stores=SimpleNamespace(get=lambda t: None),
        crm_tools=None,
    )
    factory = make_chatbot_factory(registry, sm)
    agent = await factory(tenant_ctx, "s1")
    factory_non_builtin = {t.name for t in agent._crm_tools}

    assert endpoint_non_builtin == factory_non_builtin
    assert endpoint_non_builtin == {"check_order_status", SUBMIT_DEPOSIT_VERIFICATION}


async def test_platform_fallback_ignores_configured_platform_token(ctx, monkeypatch) -> None:
    # The shared, legacy PLATFORM_CRM_API_TOKEN env var must never be used for
    # auth, even though it's configured in the environment — it isn't even
    # read anymore (the crm_catalog branch only ever uses the tenant's own
    # crm:api_token secret). This CRM authorizes by the token itself, so a
    # shared token would grant cross-tenant CRM access.
    client, sm = ctx
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    monkeypatch.setenv("PLATFORM_CRM_API_TOKEN", "platform-token-abc")
    await _seed_crm(sm)
    register_tenant_for_test(TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio"),
                              plaintext_tokens=["test-token"])

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "crm_catalog"
    assert body["crm_id"] == "betstudio"
    crm_tools = [t for t in body["tools"] if t["kind"] == "crm"]
    assert len(crm_tools) == len(ALL_TOOLS)
    assert {t["name"] for t in crm_tools} == set(ALL_TOOLS)
    assert all(t["token_configured"] is False for t in body["tools"])


async def test_platform_fallback_without_token_reports_not_configured(ctx, monkeypatch) -> None:
    client, sm = ctx
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    # No PLATFORM_CRM_API_TOKEN set (and no crm:api_token tenant secret).
    await _seed_crm(sm)
    register_tenant_for_test(TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio"),
                              plaintext_tokens=["test-token"])

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "crm_catalog"
    crm_tools = [t for t in body["tools"] if t["kind"] == "crm"]
    assert len(crm_tools) == len(ALL_TOOLS)
    assert all(t["token_configured"] is False for t in body["tools"])


async def test_x_api_key_configured_reported_independently_of_token(ctx, monkeypatch) -> None:
    client, sm = ctx
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    await _seed_crm(sm)
    # The in-memory test resolver doesn't re-read TenantSecret rows, so seed
    # the tenant's secrets_resolved directly (same pattern as the other
    # crm_catalog tests that construct TenantContext with secrets_resolved).
    fresh_ctx = register_tenant_for_test(TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio"),
                                          plaintext_tokens=["test-token"])
    fresh_ctx.secrets_resolved["crm:x_api_key"] = "the-x-api-key"

    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "crm_catalog"
    crm_tools = [t for t in body["tools"] if t["kind"] == "crm"]
    assert len(crm_tools) == len(ALL_TOOLS)
    # token_configured stays False (no crm:api_token set) while
    # x_api_key_configured is True — the two are independent.
    assert all(t["token_configured"] is False for t in body["tools"])
    assert all(t["x_api_key_configured"] is True for t in crm_tools)


async def test_nothing_configured_gives_empty_none_source(ctx) -> None:
    client, _sm = ctx
    # No chat_tools rows, no PLATFORM_CRM_* env, no tenant crm:* secrets —
    # BUILTIN_TOOLS are still always present (see
    # test_builtin_tools_always_present_even_with_no_crm_source), so "nothing
    # configured" means no CRM tools, not an empty tools list.
    resp = await client.get("/chat/tools/resolved")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "none"
    assert [t for t in body["tools"] if t["kind"] == "crm"] == []


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


# --- Catalog shape sanity ----------------------------------------------------


def test_all_tools_catalog_entries_are_well_formed() -> None:
    """Every ALL_TOOLS entry must have the shape resolve_crm_tools()/the
    catalog-onboarding endpoint expect: description/parameters/default_path/
    method, well-formed param dicts, and default_path placeholders that all
    correspond to declared params."""
    for name, entry in ALL_TOOLS.items():
        for key in ("description", "parameters", "default_path", "method"):
            assert key in entry, f"{name} is missing '{key}'"
        assert isinstance(entry["description"], str) and entry["description"]
        assert isinstance(entry["default_path"], str) and entry["default_path"]
        assert entry["method"] in {"GET", "POST", "PUT", "PATCH", "DELETE"}, name

        params = entry["parameters"]
        assert isinstance(params, dict)
        for param_name, param in params.items():
            for key in ("type", "source", "description"):
                assert key in param, f"{name}.{param_name} is missing '{key}'"
            assert param["source"] in {"session", "llm"}, f"{name}.{param_name}"

        placeholders = set(re.findall(r"\{(\w+)\}", entry["default_path"]))
        assert placeholders <= set(params), (
            f"{name}: default_path placeholders {placeholders} not all declared "
            f"as params {set(params)}"
        )


def test_get_player_latest_deposit_order_is_registered_in_catalog() -> None:
    entry = PLAYER_TOOLS["get_player_latest_deposit_order"]
    assert entry["default_path"] == "/players/{user_id}/latest-deposit-order"
    assert entry["method"] == "GET"
    assert set(entry["parameters"]) == {"user_id"}
    assert entry["parameters"]["user_id"]["source"] == "session"


def test_get_bet_limit_matches_crm_pr_3963_shipped_path() -> None:
    """CRM PR #3963 shipped get_bet_limit at
    GET /operators/{operator_id}/players/{user_id}/bet-limit?name=<partial>
    -- our catalog previously registered /casino/{operator_id}/players/
    {user_id}/games/{game_name}/bet-limit, a path the CRM never exposes.
    'name' is player-scoped (user_id in the path) and a required query
    param, not a path segment -- see the query-landing test in
    test_chat_tool_executor.py for proof it doesn't get path-substituted."""
    entry = OPERATOR_TOOLS["get_bet_limit"]
    assert entry["default_path"] == "/operators/{operator_id}/players/{user_id}/bet-limit"
    assert entry["method"] == "GET"
    assert set(entry["parameters"]) == {"operator_id", "user_id", "name"}
    assert entry["parameters"]["name"]["source"] == "llm"
    # default_path has no {name} placeholder -- pins the mechanism this
    # relies on (tool_executor.py only path-substitutes placeholders present
    # in the endpoint; anything else becomes a query param).
    assert "{name}" not in entry["default_path"]


def test_get_market_holiday_schedule_matches_crm_pr_3963_shipped_path() -> None:
    """CRM PR #3963 shipped get_market_holiday_schedule at
    GET /operators/{operator_id}/matka/holiday-schedule?market=<name>&date=<YYYY-MM-DD>
    -- our catalog previously registered /matka/{operator_id}/markets/
    {market_name}/holiday-schedule, a path the CRM never exposes. 'market'
    (renamed from market_name) and the new optional 'date' are both query
    params, not path segments."""
    entry = OPERATOR_TOOLS["get_market_holiday_schedule"]
    assert entry["default_path"] == "/operators/{operator_id}/matka/holiday-schedule"
    assert entry["method"] == "GET"
    assert set(entry["parameters"]) == {"operator_id", "market", "date"}
    assert entry["parameters"]["market"]["source"] == "llm"
    assert entry["parameters"]["date"]["source"] == "llm"
    assert "{market}" not in entry["default_path"]
    assert "{date}" not in entry["default_path"]


def test_matka_market_params_tell_the_model_to_strip_the_session() -> None:
    """Matka markets run Open and Close sessions, and customers name the one
    they want ("Milan Morning Close"). The session is never part of the market
    name -- per docs/crm-api-contract.md the CRM returns a `sessions` array and
    the caller picks the entry it needs -- so every market-name parameter has
    to say so explicitly.

    Without it the model has nowhere to put the session word and puts the whole
    phrase in the market field, which matches nothing; the turn then spends its
    remaining tool round guessing and dies on the forced plain answer. Observed
    in production ticket 7525, where five consecutive turns ended
    rounds_exhausted with "no usable response (finish_reason=stop)".

    Asserting on the market PARAMETER description, not the tool description:
    the parameter is what the model reads when it decides what to put in that
    field.
    """
    for tool_name, param in (
        ("get_matka_result", "market"),
        ("get_matka_bids", "market"),
        ("get_matka_config", "market_name"),
    ):
        desc = ALL_TOOLS[tool_name]["parameters"][param]["description"].lower()
        assert "close" in desc, f"{tool_name}.{param} never mentions the Close session"
        assert "strip" in desc, (
            f"{tool_name}.{param} does not tell the model to strip the session word"
        )
        # The concrete case that failed, so the guidance stays worked-through
        # rather than becoming an abstract instruction nobody can apply.
        assert "milan morning" in desc, (
            f"{tool_name}.{param} dropped the worked example"
        )


def test_max_tool_rounds_leaves_a_recovery_round() -> None:
    """Three rounds, not two. At two the model gets one lookup and one
    correction: a first call that returns nothing useful leaves it out of
    rounds before it can act on what it learned, and the forced plain-answer
    call that follows routinely produces nothing usable.

    Pinned because this is a cost/behaviour tradeoff someone may be tempted to
    reverse while trimming tokens -- each round re-sends the whole prompt.
    Lowering it back to 2 should be a deliberate decision, not a silent one.
    """
    import inspect

    from src.agents.chatbot import ChatBotAgent

    default = inspect.signature(ChatBotAgent.__init__).parameters["max_tool_rounds"].default
    assert default == 3, f"max_tool_rounds default is {default}, expected 3"
