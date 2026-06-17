"""Unit tests for the outbound per-tenant call-event webhook dispatcher."""

from __future__ import annotations

import hashlib
import hmac
import json

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
