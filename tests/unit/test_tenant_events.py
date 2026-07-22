"""Unit tests for the outbound per-tenant call-event webhook dispatcher."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from types import SimpleNamespace

from src.integration import tenant_events as te


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
