"""Unit tests for the BO chat-lifecycle webhook sender."""

from __future__ import annotations

import logging

from src.api import chat_webhooks


class _Tenant:
    """Minimal stand-in for a TenantSettings-like object."""

    def __init__(self, *, events_webhook_url="https://crm.example/hook",
                 events_webhook_secret_env=None, secret=None):
        self.events_webhook_url = events_webhook_url
        self.events_webhook_secret_env = events_webhook_secret_env
        self._secret = secret

    def secret_optional(self, env_var):
        return self._secret


async def test_send_bo_webhook_logs_warning_when_unsigned(monkeypatch, caplog):
    monkeypatch.delenv("EVENTS_WEBHOOK_SECRET", raising=False)

    async def fake_deliver(url, body, secret):
        return True

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    tenant = _Tenant(events_webhook_secret_env=None, secret=None)

    with caplog.at_level(logging.WARNING):
        ok = await chat_webhooks.send_bo_webhook(tenant, "session_started", {})

    assert ok is True
    assert any(
        r.levelno == logging.WARNING and "UNSIGNED" in r.message
        for r in caplog.records
    )


async def test_send_bo_webhook_no_warning_when_signed(monkeypatch, caplog):
    monkeypatch.setenv("EVENTS_WEBHOOK_SECRET", "platform-secret")

    async def fake_deliver(url, body, secret):
        return True

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    tenant = _Tenant(events_webhook_secret_env=None, secret=None)

    with caplog.at_level(logging.WARNING):
        ok = await chat_webhooks.send_bo_webhook(tenant, "session_started", {})

    assert ok is True
    assert not any(
        r.levelno == logging.WARNING and "UNSIGNED" in r.message
        for r in caplog.records
    )


# --- enqueue_on_failure (durable retry for session_closed only) --------------
# escalation_requested call sites never pass enqueue_on_failure — see
# src/api/chat.py's _escalate_session, which reverts the session to bot mode
# on failure, making a later durable delivery actively wrong. These tests
# exercise send_bo_webhook's own contract directly (whether it enqueues),
# which is what both call sites rely on.


async def test_send_bo_webhook_enqueues_on_failure_when_durable(monkeypatch):
    async def fake_deliver(url, body, secret):
        return False

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    enqueued = []

    async def fake_enqueue(**kwargs):
        enqueued.append(kwargs)

    monkeypatch.setattr(chat_webhooks, "enqueue_webhook_outbox", fake_enqueue)

    tenant = _Tenant()
    tenant.id = "t1"

    ok = await chat_webhooks.send_bo_webhook(
        tenant, "session_closed", {"session_id": "cs_1", "summary": "done"},
        enqueue_on_failure=True,
    )

    assert ok is False
    assert len(enqueued) == 1
    call = enqueued[0]
    assert call["tenant_id"] == "t1"
    assert call["session_id"] == "cs_1"
    assert call["event_type"] == "session_closed"
    assert call["url"] == "https://crm.example/hook"
    # N5: byte-identical to the pre-durability-work shape -- nothing added.
    assert call["body"] == {"event": "session_closed", "session_id": "cs_1", "summary": "done"}


async def test_send_bo_webhook_does_not_enqueue_without_durable_flag(monkeypatch):
    """escalation_requested (and every other call site that doesn't opt in)
    must never enqueue on failure — a durable retry there would re-open a CRM
    handoff for a session already reverted to bot mode."""
    async def fake_deliver(url, body, secret):
        return False

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    enqueued = []

    async def fake_enqueue(**kwargs):
        enqueued.append(kwargs)

    monkeypatch.setattr(chat_webhooks, "enqueue_webhook_outbox", fake_enqueue)

    tenant = _Tenant()
    tenant.id = "t1"

    ok = await chat_webhooks.send_bo_webhook(
        tenant, "escalation_requested", {"session_id": "cs_1"},
    )

    assert ok is False
    assert enqueued == []


async def test_send_bo_webhook_does_not_enqueue_when_no_url_configured(monkeypatch):
    """No webhook configured at all is a real no-op, not a delivery failure to
    retry — nothing should be queued."""
    enqueued = []

    async def fake_enqueue(**kwargs):
        enqueued.append(kwargs)

    monkeypatch.setattr(chat_webhooks, "enqueue_webhook_outbox", fake_enqueue)

    tenant = _Tenant(events_webhook_url=None)
    tenant.id = "t1"
    tenant.crm_id = None

    ok = await chat_webhooks.send_bo_webhook(
        tenant, "session_closed", {"session_id": "cs_1"}, enqueue_on_failure=True,
    )

    assert ok is False
    assert enqueued == []


# --- N5: no event_id (or any other field) added to the wire payload --------
# An earlier revision stamped a new event_id into session_closed bodies for
# CRM-side dedup; reverted -- the CRM hasn't agreed to a new field, and a
# strict endpoint could reject an unknown key. The body must be
# byte-identical to what it always was, for every event type.


async def test_session_closed_body_has_no_event_id(monkeypatch):
    seen = []

    async def fake_deliver(url, body, secret):
        seen.append(body)
        return True

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    tenant = _Tenant()
    tenant.id = "t1"

    await chat_webhooks.send_bo_webhook(
        tenant, "session_closed", {"session_id": "cs_1", "summary": "done"},
        enqueue_on_failure=True,
    )

    assert seen[0] == {"event": "session_closed", "session_id": "cs_1", "summary": "done"}
    assert "event_id" not in seen[0]


async def test_no_event_type_gets_an_event_id(monkeypatch):
    seen = []

    async def fake_deliver(url, body, secret):
        seen.append(body)
        return True

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    tenant = _Tenant()
    tenant.id = "t1"

    await chat_webhooks.send_bo_webhook(tenant, "session_started", {"session_id": "cs_1"})
    await chat_webhooks.send_bo_webhook(tenant, "escalation_requested", {"session_id": "cs_1"})

    assert "event_id" not in seen[0]
    assert "event_id" not in seen[1]


# --- L2: enqueue_webhook_outbox is bounded + never raises on the ------------
# customer-facing close path.


async def test_enqueue_timeout_is_bounded_and_never_raises(monkeypatch, caplog):
    async def fake_deliver(url, body, secret):
        return False

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    import asyncio

    async def fake_enqueue_hangs(**kwargs):
        await asyncio.sleep(10)  # much longer than _ENQUEUE_TIMEOUT_S

    monkeypatch.setattr(chat_webhooks, "enqueue_webhook_outbox", fake_enqueue_hangs)
    monkeypatch.setattr(chat_webhooks, "_ENQUEUE_TIMEOUT_S", 0.05)

    tenant = _Tenant()
    tenant.id = "t1"

    with caplog.at_level(logging.WARNING):
        ok = await chat_webhooks.send_bo_webhook(
            tenant, "session_closed", {"session_id": "cs_1"}, enqueue_on_failure=True,
        )

    assert ok is False  # send_bo_webhook itself still returns normally
    assert any(
        r.levelno == logging.WARNING and "timed out" in r.message
        for r in caplog.records
    )


async def test_enqueue_exception_never_propagates(monkeypatch):
    async def fake_deliver(url, body, secret):
        return False

    monkeypatch.setattr(chat_webhooks, "deliver", fake_deliver)

    async def fake_enqueue_raises(**kwargs):
        raise RuntimeError("db is on fire")

    monkeypatch.setattr(chat_webhooks, "enqueue_webhook_outbox", fake_enqueue_raises)

    tenant = _Tenant()
    tenant.id = "t1"

    # Must not raise -- this is the customer-facing close path.
    ok = await chat_webhooks.send_bo_webhook(
        tenant, "session_closed", {"session_id": "cs_1"}, enqueue_on_failure=True,
    )
    assert ok is False
