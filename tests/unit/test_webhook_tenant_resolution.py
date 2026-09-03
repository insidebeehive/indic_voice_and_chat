"""Tests for the per-tenant webhook-credential resolver indexes.

Covers the ``resolve_by_stringee_webhook_token`` / ``resolve_by_chatwoot_webhook_id``
(and ``resolve_by_chatwoot_inbox``) lookups on both ``DbTenantResolver`` and
``InMemoryTenantResolver``, plus the ``external_chat.py`` Chatwoot webhook
handler's fallback from ``app.state.tenant_resolver`` to the module-level
``middleware._resolver`` global.

These indexes are infrastructure only in this PR — not yet wired into any
live route's credential-verification logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat as chat_module
from src.api import external_chat
from src.api.chat import ChatMessageResponse
from src.api.deps import get_db_session
from src.auth import secrets as crypto
from src.auth.db_resolver import DbTenantResolver
from src.auth.middleware import (
    InMemoryTenantResolver,
    register_tenant_for_test,
    set_tenant_resolver,
)
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import Tenant, TenantSecret

# ---------------------------------------------------------------------------
# DbTenantResolver
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_ctx(monkeypatch):
    monkeypatch.setenv("VOX_SECRET_KEY", crypto.generate_key())
    crypto.reset_cache_for_tests()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        s.add(TenantSecret(
            tenant_id="t1", name="webhook:stringee_path_token",
            value_encrypted=crypto.encrypt("stringee-tok-abc"),
        ))
        s.add(TenantSecret(
            tenant_id="t1", name="chatwoot:webhook_id",
            value_encrypted=crypto.encrypt("wh-xyz-123"),
        ))
        await s.commit()

    resolver = DbTenantResolver(sm)
    await resolver.reload()
    try:
        yield resolver
    finally:
        crypto.reset_cache_for_tests()
        await engine.dispose()


async def test_db_resolver_resolves_by_stringee_webhook_token(db_ctx: DbTenantResolver) -> None:
    ctx = await db_ctx.resolve_by_stringee_webhook_token("stringee-tok-abc")
    assert ctx is not None
    assert ctx.slug == "t1"


async def test_db_resolver_resolves_by_chatwoot_webhook_id(db_ctx: DbTenantResolver) -> None:
    ctx = await db_ctx.resolve_by_chatwoot_webhook_id("wh-xyz-123")
    assert ctx is not None
    assert ctx.slug == "t1"


async def test_db_resolver_returns_none_for_unknown_keys(db_ctx: DbTenantResolver) -> None:
    assert await db_ctx.resolve_by_stringee_webhook_token("no-such-token") is None
    assert await db_ctx.resolve_by_chatwoot_webhook_id("no-such-webhook-id") is None


# ---------------------------------------------------------------------------
# InMemoryTenantResolver
# ---------------------------------------------------------------------------


def _settings(slug: str = "t1") -> TenantSettings:
    return TenantSettings(id=slug, slug=slug, name="Acme")


def test_in_memory_resolver_register_with_secrets() -> None:
    resolver = InMemoryTenantResolver()
    ctx = resolver.register(
        _settings(),
        secrets={"webhook:stringee_path_token": "tok123", "chatwoot:webhook_id": "wh456"},
    )
    assert ctx.secrets_resolved == {
        "webhook:stringee_path_token": "tok123",
        "chatwoot:webhook_id": "wh456",
    }

    import asyncio

    async def _run():
        stringee_ctx = await resolver.resolve_by_stringee_webhook_token("tok123")
        webhook_ctx = await resolver.resolve_by_chatwoot_webhook_id("wh456")
        return stringee_ctx, webhook_ctx

    stringee_ctx, webhook_ctx = asyncio.run(_run())
    assert stringee_ctx is ctx
    assert webhook_ctx is ctx


def test_in_memory_resolver_clear_does_not_leak_state() -> None:
    resolver = InMemoryTenantResolver()
    resolver.register(
        _settings("t1"),
        secrets={"webhook:stringee_path_token": "tok-old", "chatwoot:webhook_id": "wh-old"},
    )
    resolver.clear()
    resolver.register(
        _settings("t2"),
        secrets={"webhook:stringee_path_token": "tok-new", "chatwoot:webhook_id": "wh-new"},
    )

    import asyncio

    async def _run():
        return (
            await resolver.resolve_by_stringee_webhook_token("tok-old"),
            await resolver.resolve_by_chatwoot_webhook_id("wh-old"),
            await resolver.resolve_by_stringee_webhook_token("tok-new"),
            await resolver.resolve_by_chatwoot_webhook_id("wh-new"),
        )

    old_tok, old_wh, new_tok, new_wh = asyncio.run(_run())
    assert old_tok is None
    assert old_wh is None
    assert new_tok is not None
    assert new_wh is not None


def test_in_memory_resolver_register_without_secrets_kwarg_still_works() -> None:
    resolver = InMemoryTenantResolver()
    ctx = resolver.register(_settings(), plaintext_tokens=["plain-tok"])
    assert ctx.secrets_resolved == {}
    assert ctx.slug == "t1"


# ---------------------------------------------------------------------------
# register_tenant_for_test
# ---------------------------------------------------------------------------


async def test_register_tenant_for_test_forwards_secrets() -> None:
    set_tenant_resolver(None)
    try:
        register_tenant_for_test(
            _settings("t-fwd"),
            secrets={"webhook:stringee_path_token": "fwd-tok", "chatwoot:webhook_id": "fwd-wh"},
        )
        from src.auth import middleware as auth_middleware

        assert isinstance(auth_middleware._resolver, InMemoryTenantResolver)
        stringee_ctx = await auth_middleware._resolver.resolve_by_stringee_webhook_token("fwd-tok")
        webhook_ctx = await auth_middleware._resolver.resolve_by_chatwoot_webhook_id("fwd-wh")
        assert stringee_ctx is not None
        assert stringee_ctx.slug == "t-fwd"
        assert webhook_ctx is not None
        assert webhook_ctx.slug == "t-fwd"
    finally:
        set_tenant_resolver(None)


# ---------------------------------------------------------------------------
# external_chat.py: app.state.tenant_resolver -> middleware._resolver fallback
# ---------------------------------------------------------------------------


async def _fake_process_message(tenant, session_id, text) -> ChatMessageResponse:
    return ChatMessageResponse(
        session_id=session_id,
        response_text="mocked response text",
        language="en",
        confidence="high",
        sources_used=[],
        action="none",
        suggested_followups=[],
    )


@pytest_asyncio.fixture
async def fallback_app(monkeypatch) -> AsyncIterator[FastAPI]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t-cw", slug="t-cw", name="Acme"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    monkeypatch.setattr(chat_module, "process_message", _fake_process_message)

    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t-cw", slug="t-cw", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-fallback-1"},
    )
    set_tenant_resolver(resolver)

    a = FastAPI()
    a.include_router(external_chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    # Deliberately do NOT set a.state.tenant_resolver — this exercises the
    # module-level middleware._resolver fallback added in external_chat.py.

    try:
        yield a
    finally:
        set_tenant_resolver(None)
        await engine.dispose()


def test_chatwoot_webhook_falls_back_to_module_level_resolver(fallback_app: FastAPI) -> None:
    """Verifies the app.state.tenant_resolver -> middleware._resolver
    fallback on the primary webhook_id-based route (the route this fallback
    actually matters for long-term). The legacy tokenless route's
    resolve_by_chatwoot_inbox / log_only behavior has its own explicit
    coverage in tests/unit/test_external_chat.py."""
    payload = {
        "event": "message_created",
        "message_type": 0,
        "content": "hello",
        "conversation": {"id": 456},
        "sender": {"name": "Ravi", "identifier": "ext-id-1", "type": "contact"},
    }

    client = TestClient(fallback_app)
    resp = client.post("/integrations/chatwoot/webhook/wh-fallback-1", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"accepted": True}
    assert body.get("ignored") is not True
