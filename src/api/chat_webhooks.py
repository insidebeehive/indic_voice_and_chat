"""BO webhook sender for chat lifecycle events.

Posts JSON events to the tenant's webhook URL. Resolution order:
  1. ``chat_support.bo_webhook_url`` — explicit override for tenants who want
     chat events on a different endpoint than call events.
  2. ``pipeline.telephony.events_webhook_url`` — the existing CRM webhook URL
     reused for both call and chat events (no extra config needed).

Delivery uses the same retry + HMAC-SHA256 signing as call events
(``pipeline.telephony.events_webhook_secret_env``).

Event types emitted:
  session_started      — on POST /chat/sessions
  escalation_requested — when the AI escalates; includes claim + agent-ws URLs
  session_closed       — on session end; includes transcript + summary
"""
from __future__ import annotations

import logging
from typing import Any

from src.integration.tenant_events import deliver

log = logging.getLogger(__name__)


def _resolve(tenant) -> tuple[str | None, str | None]:
    """Return (url, secret) for chat webhook delivery."""
    settings = getattr(tenant, "settings", tenant)

    tel = getattr(getattr(settings, "pipeline", None), "telephony", None)
    base_url = getattr(tel, "events_webhook_url", None) if tel else None
    secret_env = getattr(tel, "events_webhook_secret_env", None) if tel else None

    chat_support = getattr(settings, "chat_support", None)
    override_url = getattr(chat_support, "bo_webhook_url", None) if chat_support else None

    url = override_url or base_url
    if not url:
        return None, None

    secret = None
    if secret_env and hasattr(tenant, "secret_optional"):
        secret = tenant.secret_optional(secret_env)
    return url, secret


async def send_bo_webhook(tenant, event_type: str, payload: dict) -> None:
    """POST event_type + payload to the tenant's webhook URL. Fire-and-forget."""
    url, secret = _resolve(tenant)
    if not url:
        return
    body: dict[str, Any] = {"event": event_type, **payload}
    await deliver(url, body, secret)
