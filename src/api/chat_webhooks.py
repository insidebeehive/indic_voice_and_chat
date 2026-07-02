"""BO webhook sender for chat lifecycle events.

Posts JSON events to the tenant's ``events_webhook_url`` (top-level tenant
setting) with HMAC-SHA256 signing when ``events_webhook_secret_env`` is set —
the same URL and secret used for call lifecycle events.

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


async def send_bo_webhook(tenant, event_type: str, payload: dict) -> bool:
    """POST event_type + payload to the tenant's webhook URL.
    Returns True if the CRM acknowledged (2xx), False on failure or no URL configured."""
    settings = getattr(tenant, "settings", tenant)
    url = getattr(settings, "events_webhook_url", None)
    if not url:
        return False
    secret_env = getattr(settings, "events_webhook_secret_env", None)
    secret = tenant.secret_optional(secret_env) if secret_env and hasattr(tenant, "secret_optional") else None
    body: dict[str, Any] = {"event": event_type, **payload}
    ok = await deliver(url, body, secret)
    if not ok:
        log.warning("bo webhook delivery failed", extra={"event_type": event_type})
    return ok
