"""Unit tests for the outbound per-tenant call-event webhook dispatcher."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.integration import tenant_events as te
from src.models.database import Base


def test_sign_body_is_hmac_sha256():
    raw = b'{"event_type":"call.initiated"}'
    expect = "sha256=" + hmac.new(b"s3cr3t", raw, hashlib.sha256).hexdigest()
    assert te.sign_body("s3cr3t", raw) == expect


def test_build_envelope_shape():
    e = te.build_envelope(event_type="call.completed", call_id="c1",
                          tenant_id="t1", channel="voicebot", data={"outcome": "x"})
    assert e["event_type"] == "call.completed"
    assert (e["call_id"], e["tenant_id"], e["channel"]) == ("c1", "t1", "voicebot")
    assert e["data"] == {"outcome": "x"}
    assert e["event_id"] and e["occurred_at"]          # always stamped


def test_channel_label_maps_human_to_softphone():
    assert te.channel_label("human") == "softphone"
    assert te.channel_label("voicebot") == "voicebot"
    assert te.channel_label(None) == "voicebot"


async def test_deliver_posts_signed_body_and_returns_true():
    calls = []

    async def poster(url, raw, headers):
        calls.append((url, raw, headers))
        return 200

    env = te.build_envelope(event_type="call.initiated", call_id="c1",
                            tenant_id="t1", channel="voicebot")
    ok = await te.deliver("https://crm/hook", env, "secret", http_post=poster)
    assert ok is True
    url, raw, headers = calls[0]
    assert url == "https://crm/hook"
    # The signature must be over the EXACT bytes posted (so the tenant can verify
    # by HMAC-ing the raw request body).
    assert headers["X-Signature"] == te.sign_body("secret", raw)
    assert json.loads(raw)["event_type"] == "call.initiated"


async def test_deliver_no_secret_omits_signature():
    seen = {}

    async def poster(url, raw, headers):
        seen.update(headers)
        return 204

    assert await te.deliver("https://crm/hook", {"x": 1}, None, http_post=poster) is True
    assert "X-Signature" not in seen


async def test_deliver_retries_then_gives_up(monkeypatch):
    monkeypatch.setattr(te, "_BACKOFF_BASE_S", 0.0)   # don't actually sleep
    attempts = []

    async def poster(url, raw, headers):
        attempts.append(1)
        return 500

    ok = await te.deliver("https://crm/hook", {"x": 1}, None, http_post=poster)
    assert ok is False
    assert len(attempts) == te._MAX_ATTEMPTS


async def test_deliver_swallows_poster_exception(monkeypatch):
    monkeypatch.setattr(te, "_BACKOFF_BASE_S", 0.0)

    async def poster(url, raw, headers):
        raise RuntimeError("boom")

    assert await te.deliver("https://crm/hook", {"x": 1}, None, http_post=poster) is False


# --- src.main._resolve_tenant_event_secret -----------------------------------
# Both call sites (main.py's _notify_tenant_event closure and chat_webhooks'
# send_bo_webhook) resolve a signing secret the same way — per-tenant secret
# first, else platform EVENTS_WEBHOOK_SECRET, else None — and must WARN loudly
# (never block delivery) when no secret resolves at all, since an unsigned
# webhook means the tenant's CRM can't verify events are really from us.


async def test_resolve_tenant_event_secret_logs_warning_when_unsigned(monkeypatch, caplog):
    from src import main as main_module

    monkeypatch.delenv("EVENTS_WEBHOOK_SECRET", raising=False)
    settings = SimpleNamespace(slug="acme")

    with caplog.at_level(logging.WARNING, logger="src.main"):
        secret = await main_module._resolve_tenant_event_secret(
            settings, None, None, tenant_id="t_acme", event_type="call.completed",
        )

    assert secret is None
    assert any(
        r.levelno == logging.WARNING and "UNSIGNED" in r.message
        for r in caplog.records
    )


async def test_resolve_tenant_event_secret_no_warning_when_signed(monkeypatch, caplog):
    from src import main as main_module

    monkeypatch.setenv("EVENTS_WEBHOOK_SECRET", "platform-secret")
    settings = SimpleNamespace(slug="acme")

    with caplog.at_level(logging.WARNING, logger="src.main"):
        secret = await main_module._resolve_tenant_event_secret(
            settings, None, None, tenant_id="t_acme", event_type="call.completed",
        )

    assert secret == "platform-secret"
    assert not any(
        r.levelno == logging.WARNING and "UNSIGNED" in r.message
        for r in caplog.records
    )


async def test_resolve_tenant_event_secret_per_tenant_secret_no_warning(monkeypatch, caplog):
    from src import main as main_module

    monkeypatch.delenv("EVENTS_WEBHOOK_SECRET", raising=False)
    settings = SimpleNamespace(slug="acme")

    class _Ctx:
        def secret_optional(self, env_var):
            return "tenant-secret"

    class _Resolver:
        async def resolve_by_slug(self, slug):
            return _Ctx()

    with caplog.at_level(logging.WARNING, logger="src.main"):
        secret = await main_module._resolve_tenant_event_secret(
            settings, "TENANT_WEBHOOK_SECRET", _Resolver(),
            tenant_id="t_acme", event_type="call.completed",
        )

    assert secret == "tenant-secret"
    assert not any(
        r.levelno == logging.WARNING and "UNSIGNED" in r.message
        for r in caplog.records
    )


# --- src.integration.tenant_events.resolve_events_webhook_url ----------------
# The tenant's own explicit events_webhook_url wins (an escape hatch for a
# tenant needing a different shape than its CRM's default); otherwise the
# tenant's linked Crm's events_webhook_url_template with {operator_id}
# substituted; otherwise None.


@pytest_asyncio.fixture
async def sm_with_crm_seed():
    """Sessionmaker seeded with a single Crm row (id='betstudio') — the
    fixture for tests exercising a tenant linked via TenantSettings(crm_id=...).
    Mirrors tests/unit/test_crm_tools_platform_fallback.py's fixture of the
    same name (kept local here since that module's fixture isn't shared via a
    conftest)."""
    from src.models.crm import Crm

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Crm(id="betstudio", name="BetStudio",
                   base_url="https://apistage.betstudio.io/api", auth_type="api_key"))
        await s.commit()
    yield sm
    await engine.dispose()


async def test_resolve_events_webhook_url_uses_tenant_override_when_set():
    from src.auth.context import TenantContext
    from src.config_tenant import TenantSettings
    from src.integration.tenant_events import resolve_events_webhook_url

    tenant = TenantContext(settings=TenantSettings(
        id="t1", slug="t1", name="T1", events_webhook_url="https://explicit.example.com/hook"))
    url = await resolve_events_webhook_url(tenant, sessionmaker=None)
    assert url == "https://explicit.example.com/hook"


async def test_resolve_events_webhook_url_uses_crm_template_with_operator_id(sm_with_crm_seed):
    from src.auth.context import TenantContext
    from src.config_tenant import TenantCRMConfig, TenantSettings
    from src.integration.tenant_events import resolve_events_webhook_url
    from src.models.crm import Crm

    async with sm_with_crm_seed() as db:
        crm = await db.get(Crm, "betstudio")
        crm.events_webhook_url_template = "https://bostage.betstudio.io/webhooks/crm/softphone-events/{operator_id}"
        await db.commit()

    # NOTE: TenantCRMConfig is a top-level field of TenantSettings (`crm=`),
    # not nested under `pipeline` — see src/config_tenant.py TenantSettings.crm.
    tenant = TenantContext(settings=TenantSettings(
        id="t1", slug="t1", name="T1", crm_id="betstudio",
        crm=TenantCRMConfig(operator_id="ab858a8c-7ad4-47d2-a0b7-05ee93f8f134")))
    url = await resolve_events_webhook_url(tenant, sm_with_crm_seed)
    assert url == "https://bostage.betstudio.io/webhooks/crm/softphone-events/ab858a8c-7ad4-47d2-a0b7-05ee93f8f134"


async def test_resolve_events_webhook_url_none_when_nothing_configured():
    from src.auth.context import TenantContext
    from src.config_tenant import TenantSettings
    from src.integration.tenant_events import resolve_events_webhook_url

    tenant = TenantContext(settings=TenantSettings(id="t1", slug="t1", name="T1"))
    url = await resolve_events_webhook_url(tenant, sessionmaker=None)
    assert url is None
