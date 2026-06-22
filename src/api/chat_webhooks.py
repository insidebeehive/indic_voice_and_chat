"""BO webhook sender for chat lifecycle events.

Posts JSON events to the tenant's configured bo_webhook_url (from chat_support
in TenantSettings). All sends are best-effort — a failed webhook never breaks
the conversation.

Event types emitted:
  session_started      — on POST /chat/sessions
  escalation_requested — when the AI escalates; includes claim + agent-ws URLs
  session_closed       — on session end; includes transcript + summary
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = 8.0


async def send_bo_webhook(tenant, event_type: str, payload: dict) -> None:
    """POST event_type + payload to the tenant's BO webhook URL. Fire-and-forget."""
    settings = getattr(tenant, "settings", tenant)
    chat_support = getattr(settings, "chat_support", None)
    if chat_support is None:
        return
    url = getattr(chat_support, "bo_webhook_url", None)
    if not url:
        return
    body: dict[str, Any] = {"event": event_type, **payload}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(url, json=body)
            if r.status_code >= 400:
                log.warning(
                    "bo_webhook %s returned %s", event_type, r.status_code,
                    extra={"event": event_type, "status": r.status_code},
                )
    except Exception:  # noqa: BLE001
        log.warning(
            "bo_webhook %s failed (network/timeout)", event_type,
            extra={"event": event_type},
        )
