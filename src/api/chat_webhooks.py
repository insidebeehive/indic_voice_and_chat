"""BO webhook sender for chat lifecycle events.

Posts JSON events to the tenant's ``events_webhook_url`` (top-level tenant
setting) with HMAC-SHA256 signing when ``events_webhook_secret_env`` is set —
the same URL and secret used for call lifecycle events.

Event types emitted:
  session_started      — on POST /chat/sessions
  escalation_requested — when the AI escalates; includes claim + agent-ws URLs
  session_closed       — on session end; includes transcript + summary

``session_closed`` deliveries may additionally be made durable via
``enqueue_on_failure`` (see ``send_bo_webhook`` below and
``src/models/webhook_outbox.py``): when the in-line attempt exhausts its
retry budget, a row is queued so ``src/main.py``'s background loop keeps
retrying long after this request/session is gone. Deliberately NOT used for
``escalation_requested`` -- see ``src/api/chat.py``'s ``_escalate_session``,
which reverts the session to bot mode on failure, making a later durable
delivery actively wrong (it would open a CRM handoff for a session no longer
awaiting one).

The BO body shape is, and must stay, byte-identical to what it was before the
durable outbox existed: ``{"event": event_type, **payload}``, nothing added.
An earlier revision of this durability work stamped a new ``event_id`` field
into ``session_closed`` bodies for CRM-side dedup -- reverted, because the
CRM has not agreed to a new field, a strict endpoint could reject an unknown
key (a 4xx there would, under the outbox's permanent-failure handling, risk
giving up on a close event instead of helping deliver it), and
``docs/integrations/coordination-service-prd.md`` already has the
coordination service stamping its own ``event_id`` downstream. Dedup, if
ever needed, is a ``session_id`` + ``event`` question to raise with the CRM
team directly, not something this module unilaterally adds to the wire
payload.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.integration.tenant_events import deliver, resolve_events_webhook_url
from src.models.database import get_sessionmaker
from src.models.webhook_outbox import enqueue_webhook_outbox
from src.utils.logging import debug_event
from src.utils.redact import redact_url

log = logging.getLogger(__name__)

# L2: cap on enqueue_webhook_outbox's own DB work when it runs inline on a
# customer-facing close path (a WS "end" frame, the idle-timeout sweep) --
# enqueue_webhook_outbox already never raises on its own, but a wedged DB
# connection could otherwise hang that request indefinitely rather than
# failing fast. 3s is generous for a single insert + commit; losing this one
# durability enqueue on a timeout is an acceptable, logged trade-off against
# stalling the customer-facing path that triggered it.
_ENQUEUE_TIMEOUT_S = 3.0


async def send_bo_webhook(
    tenant, event_type: str, payload: dict, *, enqueue_on_failure: bool = False,
) -> bool:
    """POST event_type + payload to the tenant's webhook URL.
    Returns True if the CRM acknowledged (2xx), False on failure or no URL configured.

    ``enqueue_on_failure`` (default False, opt-in per call site): when the
    in-line delivery attempt fails AFTER a URL was actually resolved (i.e.
    not the "no webhook configured at all" case, which is a real no-op, not
    a failure to retry), enqueue a ``webhook_outbox`` row so a background
    loop keeps retrying. Only ``session_closed`` call sites pass this — see
    the module docstring."""
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
    # Byte-identical to the pre-durability-work shape -- see the module
    # docstring on why nothing (e.g. an event_id) is added here.
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
        if enqueue_on_failure:
            # No special-casing for a PERMANENT 4xx here: `deliver` above
            # always retries the full in-line budget regardless of status
            # (it never passes stop_on_permanent), so a permanent failure
            # looks identical to a transient one at this point -- `ok` is
            # just False either way, and this enqueues exactly the same as
            # it always has. The outbox's own 3-consecutive-attempt rule
            # (see src/main.py's _finalize_webhook_outbox_row) is what
            # eventually decides whether a permanent failure is actually
            # undeliverable; nothing here needs to know the difference.
            #
            # Note the session_id passed through by the two session_closed
            # call sites (src/api/chat.py's human-mode end block and
            # _send_close_webhook) always carries "session_id" in payload —
            # .get with a default is defensive only, not expected to fire.
            #
            # L2: bounded + never raises on this customer-facing path.
            # enqueue_webhook_outbox already swallows its own exceptions;
            # wait_for additionally bounds how long a wedged DB connection
            # can hold up whatever triggered this send (a WS "end" frame, the
            # idle-timeout sweep) -- losing the enqueue on timeout is an
            # acceptable, logged trade-off against stalling that caller.
            try:
                await asyncio.wait_for(
                    enqueue_webhook_outbox(
                        tenant_id=tenant_id, session_id=payload.get("session_id", ""),
                        event_type=event_type, url=url, body=body,
                    ),
                    timeout=_ENQUEUE_TIMEOUT_S,
                )
            except Exception:  # noqa: BLE001 - must never break the customer-facing close path
                log.warning(
                    "webhook_outbox enqueue timed out; session_closed event will not be retried",
                    exc_info=True,
                    extra={"tenant_id": tenant_id, "session_id": payload.get("session_id", ""),
                           "event_type": event_type},
                )
    return ok
