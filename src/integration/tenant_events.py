"""Outbound per-tenant call-event webhooks.

When a call starts or ends we POST a signed JSON envelope to the tenant's
configured ``events_webhook_url`` so their CRM receives call lifecycle events and
the final LLM outcome (the ``call.completed`` event doubles as the end-call
signal). The same envelope is used for BOTH the human-agent "softphone" path and
the AI "voice-bot" path — both funnel through ``call_store`` (insert_call /
record_outcome), which is where the events are emitted.

Delivery is best-effort with bounded retry + exponential backoff and NEVER
raises — webhook delivery must not block or break call handling. The caller
schedules ``deliver(...)`` fire-and-forget (e.g. ``asyncio.create_task``).

Signing: when a secret is configured, the body is signed with HMAC-SHA256 and
sent as ``X-Signature: sha256=<hex>`` over the EXACT bytes posted, so the tenant
can verify the call is genuinely from us (compute the same HMAC over the raw
request body).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Optional

import httpx

from src.utils.logging import debug_event
from src.utils.redact import redact_url

log = logging.getLogger(__name__)

# ``http_post(url, raw_body, headers) -> status_code``. ``-1`` means no response.
HTTPPoster = Callable[[str, bytes, dict], Awaitable[int]]

_TIMEOUT_S = 5.0
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 0.3  # 0.3, 0.6, 1.2 ...


def channel_label(agent_type: Optional[str]) -> str:
    """Envelope ``channel``: a human softphone agent vs our AI voice-bot."""
    return "softphone" if agent_type == "human" else "voicebot"


def sign_body(secret: str, raw: bytes) -> str:
    """HMAC-SHA256 of the raw request body, as ``sha256=<hex>``."""
    return "sha256=" + hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def verify_signature(secret: str, raw_body: bytes, header_value: Optional[str]) -> bool:
    """Verify an INBOUND ``X-Signature``-style header against ``raw_body``,
    using the same ``sha256=<hex>`` format ``sign_body`` produces.

    Never raises: a missing/malformed header or missing secret is treated as
    a verification failure (``False``), not an error — callers (webhook
    handlers) turn that into a 401 rather than a 500."""
    if not secret or not header_value:
        return False
    try:
        expected = sign_body(secret, raw_body)
        return hmac.compare_digest(expected, header_value)
    except Exception:  # noqa: BLE001 — signature verification must never raise
        return False


def sign_body_hex(secret: str, raw: bytes) -> str:
    """Bare-hex HMAC-SHA256, no ``sha256=`` prefix — for vendors whose contract
    specifies ``hex(HMAC-SHA256(raw_body, salt))`` verbatim (unlike
    ``sign_body``'s ``sha256=<hex>`` form used by our own outbound webhooks)."""
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def verify_signature_hex(secret: str, raw_body: bytes, header_value: Optional[str]) -> bool:
    """Verify an INBOUND bare-hex signature header against ``raw_body``, using
    the same unprefixed ``hex(HMAC-SHA256(...))`` format ``sign_body_hex``
    produces.

    Strict: a ``sha256=``-prefixed value (``verify_signature``'s format) is
    rejected, not silently accepted — the two signature schemes are kept
    distinct on purpose.

    Never raises: a missing/malformed header or missing secret is treated as
    a verification failure (``False``), not an error — callers (webhook
    handlers) turn that into a 401 rather than a 500."""
    if not secret or not header_value:
        return False
    try:
        expected = sign_body_hex(secret, raw_body)
        # .lower() on the INCOMING header only — sign_body_hex's own output
        # (both `expected` here and whatever we send on outbound requests)
        # is already lowercase via hexdigest(), so this doesn't change what
        # we send; it just tolerates a vendor that sends uppercase hex.
        return hmac.compare_digest(expected, header_value.lower())
    except Exception:  # noqa: BLE001 — signature verification must never raise
        return False


def build_envelope(
    *,
    event_type: str,
    call_id: str,
    tenant_id: str,
    channel: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """A unified event envelope, identical across softphone + voice-bot."""
    return {
        "event_type": event_type,
        "event_id": uuid.uuid4().hex,
        "call_id": call_id,
        "tenant_id": tenant_id,
        "channel": channel,
        "occurred_at": datetime.now(UTC).isoformat(),
        "data": data or {},
    }


async def _httpx_post(url: str, raw: bytes, headers: dict) -> int:
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        try:
            resp = await client.post(url, content=raw, headers=headers)
            return resp.status_code
        except httpx.HTTPError as e:  # noqa: BLE001 - a delivery error must not raise
            log.warning("tenant event POST failed", extra={"url": url, "error": str(e)})
            return -1


async def resolve_events_webhook_url(tenant, sessionmaker) -> Optional[str]:
    """The URL to POST tenant lifecycle events to.

    Priority: the tenant's own explicit ``events_webhook_url`` (an escape
    hatch for a tenant needing a different shape than its CRM's default),
    else the tenant's linked CRM's ``events_webhook_url_template`` with
    ``{operator_id}`` substituted from the tenant's own operator_id, else
    None (no webhook configured at all).

    ``tenant`` may be a ``TenantContext`` (real call sites) or a bare
    ``TenantSettings``-like object exposing the same attributes directly
    (some call sites/tests pass either) — unwrapped the same way
    ``src.api.chat_webhooks.send_bo_webhook`` already does.
    """
    settings = getattr(tenant, "settings", tenant)

    explicit = getattr(settings, "events_webhook_url", None)
    if explicit:
        debug_event(
            log, "tenant_events url_resolve resolved",
            source="tenant_override", url=redact_url(explicit),
        )
        return explicit

    crm_id = getattr(settings, "crm_id", None)
    if not crm_id or sessionmaker is None:
        # Both collapse to "no URL" for the caller, but for different
        # reasons -- a tenant with no CRM linked at all vs. this call site
        # not having DB access to look one up -- and the caller only ever
        # sees the coarse "no_webhook_url_configured" outcome, never which.
        debug_event(
            log, "tenant_events url_resolve skipped",
            reason="no_crm_linked" if not crm_id else "no_sessionmaker",
        )
        return None

    from src.models.crm import Crm

    async with sessionmaker() as db:
        crm = await db.get(Crm, crm_id)
    if crm is None or not crm.events_webhook_url_template:
        debug_event(
            log, "tenant_events url_resolve skipped",
            reason="crm_not_found" if crm is None else "crm_missing_template",
            crm_id=crm_id,
        )
        return None

    crm_config = getattr(settings, "crm", None)
    operator_id = (
        getattr(crm_config, "operator_id", None)
        or getattr(settings, "id", None)
        or getattr(tenant, "id", None)
    )
    url = crm.events_webhook_url_template.replace("{operator_id}", operator_id)
    debug_event(
        log, "tenant_events url_resolve resolved",
        source="crm_template", url=redact_url(url), crm_id=crm_id,
    )
    return url


async def deliver(
    url: str,
    envelope: dict[str, Any],
    secret: Optional[str] = None,
    *,
    http_post: Optional[HTTPPoster] = None,
) -> bool:
    """POST a signed envelope to ``url`` with bounded retry. Returns True on a
    2xx within the retry budget. Never raises."""
    raw = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Signature"] = sign_body(secret, raw)
    poster = http_post or _httpx_post
    event_type = envelope.get("event_type")
    for attempt in range(_MAX_ATTEMPTS):
        debug_event(
            log, "tenant_events deliver request",
            url=redact_url(url), event_type=event_type,
            attempt=attempt + 1, max_attempts=_MAX_ATTEMPTS, signed=bool(secret),
        )
        try:
            status = await poster(url, raw, headers)
        except Exception:  # noqa: BLE001 - a custom poster must not break the call
            log.exception("tenant event poster raised", extra={"url": url})
            status = -1
        # Only the final outcome reaches the caller (see call_store.py's/
        # chat_webhooks.py's "response" events) -- without this, a delivery
        # that fails twice and succeeds on the third attempt looks identical
        # to one that succeeded immediately.
        debug_event(
            log, "tenant_events deliver response",
            url=redact_url(url), event_type=event_type,
            attempt=attempt + 1, status=status,
        )
        if 200 <= status < 300:
            return True
        if attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(_BACKOFF_BASE_S * (2**attempt))
    log.warning("tenant event delivery exhausted retries",
                extra={"url": url, "event_type": envelope.get("event_type")})
    return False
