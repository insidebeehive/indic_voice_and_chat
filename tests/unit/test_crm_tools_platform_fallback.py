"""Platform-level CRM auth fallback (src/bootstrap.py _load_crm_tools_uncached).

Covers the gap described in
docs/superpowers/specs/2026-07-22-platform-level-crm-tools-design.md: the
platform catalog fallback already supported PLATFORM_CRM_BASE_URL, but had no
platform fallback for api_token/auth_type — any tenant without its own
crm:api_token secret got zero tools. These tests pin the 4 spec scenarios.
"""

from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth import TenantContext
from src.auth import secrets as crypto
from src.bootstrap import make_chatbot_factory
from src.chatbot.catalog import ALL_TOOLS
from src.config_tenant import TenantSettings
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


def _clean_platform_env(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_CRM_BASE_URL", raising=False)
    monkeypatch.delenv("PLATFORM_CRM_API_TOKEN", raising=False)
    monkeypatch.delenv("PLATFORM_CRM_AUTH_TYPE", raising=False)


async def test_platform_fallback_used_when_no_tenant_secrets(monkeypatch) -> None:
    # Scenario 1: no chat_tools rows, no crm:* secrets, platform base_url +
    # token set via env -> full 18-tool catalog, every exec token == platform token.
    _clean_platform_env(monkeypatch)
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    monkeypatch.setenv("PLATFORM_CRM_API_TOKEN", "platform-token-abc")

    engine, sm = await _make_sessionmaker()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1"),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert len(agent._crm_tools) == len(ALL_TOOLS)
        assert agent._crm_tools  # non-empty sanity check
        for name in ALL_TOOLS:
            exec_spec = registry.crm_tools._items["t1"][1][name]
            assert exec_spec["token"] == "platform-token-abc"
    finally:
        await engine.dispose()


async def test_tenant_own_token_wins_over_platform(monkeypatch) -> None:
    # Scenario 2: tenant has its own crm:api_token secret -> tenant token wins
    # even though the platform token is also configured (existing precedence).
    _clean_platform_env(monkeypatch)
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")
    monkeypatch.setenv("PLATFORM_CRM_API_TOKEN", "platform-token-abc")

    engine, sm = await _make_sessionmaker()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1"),
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


async def test_no_platform_token_and_no_tenant_secret_gives_none_token(monkeypatch) -> None:
    # Scenario 4: nothing configured at all (no chat_tools rows, no crm:*
    # secrets, no PLATFORM_CRM_API_TOKEN) but PLATFORM_CRM_BASE_URL is set (so
    # the fallback activates and returns tools) -> token is None, matching
    # today's existing "nothing configured" behavior, just now distinguished
    # from "platform token configured".
    _clean_platform_env(monkeypatch)
    monkeypatch.setenv("PLATFORM_CRM_BASE_URL", "https://platform-crm.example.com")

    engine, sm = await _make_sessionmaker()
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = TenantContext(
            settings=TenantSettings(id="t1", slug="t1", name="T1"),
            secrets_resolved={},
        )
        agent = await factory(tenant, "s1")
        assert len(agent._crm_tools) == len(ALL_TOOLS)
        any_name = next(iter(ALL_TOOLS))
        exec_spec = registry.crm_tools._items["t1"][1][any_name]
        assert exec_spec["token"] is None
    finally:
        await engine.dispose()
