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

from src.integration.tenant_events import deliver, resolve_events_webhook_url
from src.models.database import get_sessionmaker
from src.utils.logging import debug_event
from src.utils.redact import redact_url

log = logging.getLogger(__name__)


async def send_bo_webhook(tenant, event_type: str, payload: dict) -> bool:
    """POST event_type + payload to the tenant's webhook URL.
    Returns True if the CRM acknowledged (2xx), False on failure or no URL configured."""
    settings = getattr(tenant, "settings", tenant)
    tenant_id = getattr(tenant, "id", None)
    url = await resolve_events_webhook_url(tenant, get_sessionmaker())
    if not url:
        # Silent no-op previously: a tenant with no events_webhook_url
        # configured never gets a trace anywhere that a lifecycle event
        # (session_started/escalation_requested/session_closed) was even
        # attempted, let alone dropped.
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "chat_webhooks bo_webhook skipped", event_type=event_type,
                tenant_id=tenant_id, reason="no_webhook_url_configured",
            )
        return False
    secret_env = getattr(settings, "events_webhook_secret_env", None)
    secret = tenant.secret_optional(secret_env) if secret_env and hasattr(tenant, "secret_optional") else None
    if not secret:
        import os
        secret = os.environ.get("EVENTS_WEBHOOK_SECRET") or None
    if not secret:
        log.warning(
            "bo webhook sending UNSIGNED (no events_webhook_secret_env or "
            "platform EVENTS_WEBHOOK_SECRET configured) — configure a webhook secret; "
            "see docs/integrations/chat-widget-backend-integration.md#4-webhook-events",
            extra={"event_type": event_type},
        )
    body: dict[str, Any] = {"event": event_type, **payload}
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            # redact_url: this is the tenant's own configured webhook url, so
            # it can carry an api key as a query param or basic-auth userinfo.
            # src/main.py's _notify_tenant_event redacts the same category for
            # the same reason; these two were inconsistent.
            log, "chat_webhooks bo_webhook request", event_type=event_type,
            tenant_id=tenant_id, url=redact_url(url), payload=payload,
            signed=bool(secret),
        )
    ok = await deliver(url, body, secret)
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "chat_webhooks bo_webhook response", event_type=event_type,
            tenant_id=tenant_id, url=redact_url(url), ok=ok,
        )
    if not ok:
        log.warning("bo webhook delivery failed", extra={"event_type": event_type})
    return ok
