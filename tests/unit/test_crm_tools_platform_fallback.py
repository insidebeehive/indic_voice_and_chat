"""Tier-2 CRM tool resolution (src/bootstrap.py resolve_crm_tools /
_load_crm_tools_uncached).

Originally covered the gap described in
docs/superpowers/specs/2026-07-22-platform-level-crm-tools-design.md (the
platform catalog fallback via PLATFORM_CRM_BASE_URL + catalog.ALL_TOOLS had no
platform fallback for api_token/auth_type). That env-var-driven mechanism has
since been replaced (docs/superpowers/plans/2026-07-23-crm-entity.md, Task 3)
by a tenant's link to a DB-backed ``Crm``/``CrmTool`` catalog
(``tenant.settings.crm_id``) — these tests now pin the same precedence/token/
x_api_key/extra_headers scenarios against that mechanism instead. The legacy
PLATFORM_CRM_* env vars are still asserted as inert/ignored where a test
previously depended on them, so any accidental resurrection of the old
env-var path would be caught here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth import TenantContext
from src.auth import secrets as crypto
from src.bootstrap import make_chatbot_factory
from src.chatbot.catalog import ALL_TOOLS
from src.config_tenant import TenantCRMConfig, TenantSettings
from src.models.chat import ChatTool
from src.models.database import Base
from src.models.tenant import Tenant, TenantSecret


def _registry():
    return SimpleNamespace(
        providers=SimpleNamespace(get_llm=lambda t: object(), get_platform_llm=lambda: object()),
        retrievers=SimpleNamespace(get=lambda t: object()),
        session_stores=SimpleNamespace(get=lambda t: None),
        crm_tools=None,
    )


async def _make_sessionmaker(tenant_id: str = "t1"):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id=tenant_id, slug=tenant_id, name="T1"))
        await s.commit()
    return engine, sm


async def _make_sessionmaker_with_crm(tenant_id: str = "t1", crm_id: str = "betstudio"):
    """Same as ``_make_sessionmaker`` but also seeds a ``Crm`` + its full
    ``CrmTool`` catalog — the tier-2 fixture for tests exercising a tenant
    linked to a Crm entity via ``TenantSettings(crm_id=...)``."""
    from src.models.crm import Crm, CrmTool

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id=tenant_id, slug=tenant_id, name="T1"))
        s.add(Crm(id=crm_id, name="BetStudio",
                   base_url="https://apistage.betstudio.io/api", auth_type="api_key"))
        for name, spec in ALL_TOOLS.items():
            s.add(CrmTool(crm_id=crm_id, name=name, description=spec["description"],
                           endpoint=spec["default_path"], method=spec.get("method", "GET"),
                           parameters=spec.get("parameters", {})))
        await s.commit()
    return engine, sm


@pytest_asyncio.fixture
async def sm_with_crm_seed():
    """Sessionmaker-only fixture (no Tenant row) for tests that call
    ``resolve_crm_tools`` directly with a hand-built ``TenantContext``."""
    engine, sm = await _make_sessionmaker_with_crm()
    yield sm
    await engine.dispose()


def _clean_platform_env(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_CRM_BASE_URL", raising=False)
    monkeypatch.delenv("PLATFORM_CRM_API_TOKEN", raising=False)
    monkeypatch.delenv("PLATFORM_CRM_AUTH_TYPE", raising=False)


async def test_platform_token_ignored_even_when_configured(monkeypatch) -> None:
    # Scenario 1 (corrected): no chat_tools rows, no crm:* secrets, tenant
    # linked to a Crm entity -> full 18-tool catalog is still returned, but
    # the legacy shared PLATFORM_CRM_API_TOKEN env var must NEVER be used as
    # the resolved token, even though it is present in the environment (it's
    # not even read anymore — the crm_catalog branch only ever uses the
    # tenant's own crm:api_token secret). This platform's CRM authorizes by
    # the token itself (not a request parameter), so a shared token would let
    # every tenant's session act with one tenant's CRM authorization — a real
    # cross-tenant access issue.
    _clean_platform_env(monkeypatch)
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    monkeypatch.setenv("PLATFORM_CRM_API_TOKEN", "platform-token-abc")

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        # crm.operator_id must be configured or resolve_crm_tools now
        # registers zero crm_catalog tools (C1/W1) -- set it here since this
        # test's actual subject is the token fallback, not operator_id.
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio",
                                    crm=TenantCRMConfig(operator_id="op-123")),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert len(agent._crm_tools) == len(ALL_TOOLS)
        assert agent._crm_tools  # non-empty sanity check
        for name in ALL_TOOLS:
            exec_spec = registry.crm_tools._items["t1"][1][name]
            assert exec_spec["token"] is None
    finally:
        await engine.dispose()


async def test_tenant_own_token_wins_over_platform(monkeypatch) -> None:
    # Scenario 2: tenant has its own crm:api_token secret -> tenant token wins
    # even though the (now-inert) legacy platform token env var is also set
    # (existing precedence).
    _clean_platform_env(monkeypatch)
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    monkeypatch.setenv("PLATFORM_CRM_API_TOKEN", "platform-token-abc")

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        # crm.operator_id required (C1/W1) -- see comment above.
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio",
                                    crm=TenantCRMConfig(operator_id="op-123")),
            secrets_resolved={"crm:api_token": "tenant-own-token"},
        )
        agent = await factory(tenant, "s1")
        assert len(agent._crm_tools) == len(ALL_TOOLS)
        any_name = next(iter(ALL_TOOLS))
        exec_spec = registry.crm_tools._items["t1"][1][any_name]
        assert exec_spec["token"] == "tenant-own-token"
    finally:
        await engine.dispose()


async def test_tenant_registered_tools_take_precedence_over_platform_fallback(monkeypatch) -> None:
    # Scenario 3: tenant has its own chat_tools rows -> those are returned
    # unchanged, and the platform fallback path (token resolution) is never
    # reached even though the platform token is configured via env.
    _clean_platform_env(monkeypatch)
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    monkeypatch.setenv("PLATFORM_CRM_API_TOKEN", "platform-token-abc")
    monkeypatch.setenv("VOX_SECRET_KEY", crypto.generate_key())
    crypto.reset_cache_for_tests()

    engine, sm = await _make_sessionmaker()
    try:
        async with sm() as s:
            s.add(ChatTool(
                tenant_id="t1", name="check_order_status", description="check order",
                endpoint="https://crm/api/orders/{order_id}", method="GET", auth_type="bearer",
                auth_config={"token_secret_name": "chat_tool:check_order_status:token"},
                parameters={"order_id": {"type": "string", "source": "llm"}}))
            s.add(TenantSecret(tenant_id="t1", name="chat_tool:check_order_status:token",
                               value_encrypted=crypto.encrypt("tenant-registered-token")))
            await s.commit()

        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1"),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        # Exactly the tenant-registered tool, not the 18-tool catalog.
        names = {t.name for t in agent._crm_tools}
        assert names == {"check_order_status"}
        assert len(agent._crm_tools) != len(ALL_TOOLS)
        exec_spec = registry.crm_tools._items["t1"][1]["check_order_status"]
        assert exec_spec["token"] == "tenant-registered-token"
    finally:
        await engine.dispose()


async def test_tenant_registered_branch_warns_on_header_path_operator_id_mismatch(monkeypatch, caplog) -> None:
    # A tenant registered via POST /chat/tools/from-catalog with a real
    # operator_id (written into this tool's auth_config.extra_headers) but a
    # DIFFERENT tenant.settings.crm.operator_id configured: the header
    # "operatorid" sent on every call and the path operator_id substituted
    # into the same call diverge. Per the CRM's shipped contract this 403s
    # before auth is even checked -- must be logged so it's diagnosable
    # instead of read as a bad credential.
    #
    # Follow-up to C1/W1: this mismatch is only detectable/meaningful when
    # crm.operator_id IS actually configured -- when it's unset, crm_executor
    # now falls back to each row's OWN header value for the path too, so
    # path and header can never disagree in that case (see the negative test
    # below and test_no_operator_id_configured_falls_back_to_row_header_for_path).
    import logging

    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker()
    try:
        async with sm() as s:
            s.add(ChatTool(
                tenant_id="t1", name="check_order_status", description="check order",
                endpoint="https://crm/api/orders/{order_id}", method="GET", auth_type="bearer",
                auth_config={"extra_headers": {"operatorid": "real-operator-uuid"}},
                parameters={"order_id": {"type": "string", "source": "llm"}}))
            await s.commit()

        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1",
                                    crm=TenantCRMConfig(operator_id="configured-operator-id")),
            secrets_resolved={},
        )
        with caplog.at_level(logging.WARNING, logger="src.bootstrap"):
            await factory(tenant, "s1")

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "operatorid" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert warnings[0].__dict__.get("tenant_id") == "t1"
        assert "real-operator-uuid" in warnings[0].getMessage()
        assert "configured-operator-id" in warnings[0].getMessage()
    finally:
        await engine.dispose()


async def test_tenant_registered_branch_no_warning_when_no_operator_id_configured(monkeypatch, caplog) -> None:
    # Negative case: crm.operator_id is unset entirely -- the mismatch check
    # doesn't even apply (crm_executor falls back to each row's own header
    # for the path instead of a tenant.id guess), so no warning fires no
    # matter what the row's header value is.
    import logging

    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker()
    try:
        async with sm() as s:
            s.add(ChatTool(
                tenant_id="t1", name="check_order_status", description="check order",
                endpoint="https://crm/api/orders/{order_id}", method="GET", auth_type="bearer",
                auth_config={"extra_headers": {"operatorid": "real-operator-uuid"}},
                parameters={"order_id": {"type": "string", "source": "llm"}}))
            await s.commit()

        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1"),
            secrets_resolved={},
        )
        with caplog.at_level(logging.WARNING, logger="src.bootstrap"):
            await factory(tenant, "s1")

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "operatorid" in r.getMessage()
        ]
        assert warnings == []
    finally:
        await engine.dispose()


async def test_no_operator_id_configured_falls_back_to_row_header_for_path(monkeypatch) -> None:
    # Follow-up to C1/W1: when crm.operator_id is unset, a tenant-registered
    # tool whose endpoint template declares a session-sourced {operator_id}
    # path segment must still resolve it -- from the row's OWN "operatorid"
    # header (baked in at registration time via POST /chat/tools/from-catalog),
    # not tenant.id (removed) and not a literal unresolved "{operator_id}".
    _clean_platform_env(monkeypatch)

    captured = {}

    async def _fake_execute_crm_tool(*, endpoint, method, parameters, auth_type,
                                       token, args, context, **kwargs):
        captured["endpoint"] = endpoint
        captured["context"] = context
        return {"ok": True}

    monkeypatch.setattr(
        "src.chatbot.tool_executor.execute_crm_tool", _fake_execute_crm_tool,
    )

    engine, sm = await _make_sessionmaker()
    try:
        async with sm() as s:
            s.add(ChatTool(
                tenant_id="t1", name="get_bet_limit", description="bet limit",
                endpoint="https://crm/api/operators/{operator_id}/players/{user_id}/bet-limit",
                method="GET", auth_type="bearer",
                auth_config={"extra_headers": {"operatorid": "row-operator-uuid"}},
                parameters={
                    "operator_id": {"type": "string", "source": "session"},
                    "user_id": {"type": "string", "source": "session"},
                }))
            await s.commit()

        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1"),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1", customer_id="player-1")

        from src.interfaces.llm import ToolCall
        result = await agent._crm_executor(
            ToolCall(id="tc1", name="get_bet_limit", arguments={}), timeout_s=5.0)

        assert result == {"ok": True}
        assert captured["context"]["operator_id"] == "row-operator-uuid"
    finally:
        await engine.dispose()


async def test_x_api_key_secret_populated_for_every_platform_catalog_tool(monkeypatch) -> None:
    # The new, independent crm:x_api_key secret is tenant-level (like
    # operator_id) and must be attached to every tool's exec spec in the
    # crm_catalog branch, not just some.
    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        # crm.operator_id required (C1/W1) -- see comment on
        # test_platform_token_ignored_even_when_configured above.
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio",
                                    crm=TenantCRMConfig(operator_id="op-123")),
            secrets_resolved={"crm:x_api_key": "the-x-api-key"},
        )
        agent = await factory(tenant, "s1")
        assert len(agent._crm_tools) == len(ALL_TOOLS)
        for name in ALL_TOOLS:
            exec_spec = registry.crm_tools._items["t1"][1][name]
            assert exec_spec["x_api_key"] == "the-x-api-key"
    finally:
        await engine.dispose()


async def test_extra_headers_carries_operator_id_from_tenant_crm_config(monkeypatch) -> None:
    # Regression: the platform-fallback branch used to hardcode
    # extra_headers=None, so it NEVER sent the "operatorid" header the
    # downstream CRM needs (only the tenant-registered-tools branch did, via
    # chat_tools.auth_config). When the tenant has its own crm.operator_id
    # configured, every crm_catalog tool's extra_headers must carry it.
    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(
                id="t1", slug="t1", name="T1", crm_id="betstudio",
                crm=TenantCRMConfig(operator_id="operator-uuid-123")),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert len(agent._crm_tools) == len(ALL_TOOLS)
        for name in ALL_TOOLS:
            exec_spec = registry.crm_tools._items["t1"][1][name]
            assert exec_spec["extra_headers"] == {"operatorid": "operator-uuid-123"}
    finally:
        await engine.dispose()


async def test_no_operator_id_configured_registers_zero_crm_catalog_tools(monkeypatch) -> None:
    # C1/W1 (was test_extra_headers_falls_back_to_tenant_id_when_no_operator_id_
    # configured, which pinned the OLD behavior: falling back to tenant.id as
    # the "operatorid" header/path value). tenant.id is not a real CRM
    # operator uuid, so every such call used to 403 under the CRM's shipped
    # guard -- now resolve_crm_tools registers zero crm_catalog tools instead
    # of handing the LLM tools guaranteed to fail, and the chatbot must still
    # build fine with an empty crm tool list.
    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio"),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert agent._crm_tools == []
    finally:
        await engine.dispose()


async def test_operator_id_fallback_logs_a_warning_naming_the_consequence(monkeypatch, caplog) -> None:
    # Against CRM PR #3963's shipped guard, tenant.id is not a real
    # registered operator uuid -- a tenant with no crm.operator_id configured
    # now gets zero crm_catalog tools (see the test above) rather than tools
    # guaranteed to 403, but that's still a real gap worth surfacing in logs
    # rather than a silently-empty tool list.
    import logging

    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio"),
            secrets_resolved={},
        )
        with caplog.at_level(logging.WARNING, logger="src.bootstrap"):
            agent = await factory(tenant, "s1")

        assert agent._crm_tools == []
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "operatorid" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert warnings[0].__dict__.get("tenant_id") == "t1"
    finally:
        await engine.dispose()


async def test_operator_id_configured_does_not_log_the_fallback_warning(monkeypatch, caplog) -> None:
    # Negative case: when crm.operator_id IS configured, no warning should
    # fire -- pins that the log is conditional on the fallback actually
    # being taken, not unconditional noise on every crm-catalog resolution.
    import logging

    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(
                id="t1", slug="t1", name="T1", crm_id="betstudio",
                crm=TenantCRMConfig(operator_id="operator-uuid-123")),
            secrets_resolved={},
        )
        with caplog.at_level(logging.WARNING, logger="src.bootstrap"):
            await factory(tenant, "s1")

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "operatorid" in r.getMessage()
        ]
        assert warnings == []
    finally:
        await engine.dispose()


async def test_no_platform_token_and_no_tenant_secret_gives_none_token(monkeypatch) -> None:
    # Scenario 4: nothing configured at all (no chat_tools rows, no crm:*
    # secrets) but the tenant is linked to a Crm entity AND has crm.operator_id
    # configured (required per C1/W1, else zero tools) -> token is None,
    # matching today's existing "nothing configured" behavior.
    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker_with_crm()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1", crm_id="betstudio",
                                    crm=TenantCRMConfig(operator_id="op-123")),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert len(agent._crm_tools) == len(ALL_TOOLS)
        any_name = next(iter(ALL_TOOLS))
        exec_spec = registry.crm_tools._items["t1"][1][any_name]
        assert exec_spec["token"] is None
    finally:
        await engine.dispose()


async def test_crm_linked_tenant_gets_crm_catalog_tools(sm_with_crm_seed) -> None:
    """A tenant with tenant.crm_id set gets that CRM's DB-backed tool catalog,
    using the CRM's base_url/auth_type and the tenant's OWN api_token/x_api_key
    (unchanged per-tenant resolution)."""
    from src.bootstrap import resolve_crm_tools

    tenant = TenantContext(
        settings=TenantSettings(
            id="t1", slug="t1", name="T1", crm_id="betstudio",
            crm=TenantCRMConfig(operator_id="op-123"),
        ),
        secrets_resolved={"crm:api_token": "tok-abc", "crm:x_api_key": "key-xyz"},
    )

    specs, execs, source = await resolve_crm_tools(tenant, sm_with_crm_seed)

    assert source == "crm_catalog"
    assert len(specs) == len(ALL_TOOLS)  # matches the seeded catalog's tool count
    sample = execs["get_player_wallet"]
    assert sample["endpoint"] == "https://apistage.betstudio.io/api/players/{user_id}/wallet"
    assert sample["auth_type"] == "api_key"
    assert sample["token"] == "tok-abc"
    assert sample["x_api_key"] == "key-xyz"
    assert sample["extra_headers"] == {"operatorid": "op-123"}


async def test_tenant_without_crm_link_and_no_chat_tools_gets_none(sm_with_crm_seed) -> None:
    from src.bootstrap import resolve_crm_tools

    tenant = TenantContext(settings=TenantSettings(id="t2", slug="t2", name="T2"), secrets_resolved={})

    specs, execs, source = await resolve_crm_tools(tenant, sm_with_crm_seed)
    assert source == "none"
    assert specs == []


async def test_crm_linked_tenant_with_no_sessionmaker_degrades_to_none() -> None:
    """A crm_id-linked tenant resolved with sessionmaker=None must degrade to
    the same ([], {}, "none") result as every other "nothing configured" case
    in this function, not raise TypeError('NoneType' object is not callable)."""
    from src.bootstrap import resolve_crm_tools

    tenant = TenantContext(
        settings=TenantSettings(
            id="t1", slug="t1", name="T1", crm_id="betstudio",
            crm=TenantCRMConfig(operator_id="op-123"),
        ),
        secrets_resolved={"crm:api_token": "tok-abc", "crm:x_api_key": "key-xyz"},
    )

    specs, execs, source = await resolve_crm_tools(tenant, None)

    assert (specs, execs, source) == ([], {}, "none")


# --- B8: chat default language follows tenant.settings.default_language -----

async def test_chatbot_factory_uses_tenant_default_language(monkeypatch) -> None:
    # B8: was `or "en"` here vs `or "hi"` in src/api/external_chat.py -- two
    # different fallbacks for the same tenant field. A tenant with an explicit
    # default_language must get exactly that language, not a hardcoded "en".
    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1", default_language="ta"),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert agent._language == "ta"
    finally:
        await engine.dispose()


async def test_chatbot_factory_defaults_to_hi_matching_model_default(monkeypatch) -> None:
    # TenantSettings.default_language's own model default is "hi" -- a tenant
    # that never set it explicitly (bare TenantSettings still carries "hi") or
    # a settings-like stub missing the attribute entirely must land on "hi",
    # not the "en" this used to fall back to.
    _clean_platform_env(monkeypatch)

    engine, sm = await _make_sessionmaker()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1"),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert agent._language == "hi"
    finally:
        await engine.dispose()
