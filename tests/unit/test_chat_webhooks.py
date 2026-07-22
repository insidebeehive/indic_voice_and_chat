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
