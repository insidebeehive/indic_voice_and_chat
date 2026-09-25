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
from dataclasses import dataclass
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

# 4xx codes that mean "try again later" rather than "this will never work" --
# every OTHER 4xx (400, 401, 403, 404, ...) is classified PERMANENT on the
# DeliveryResult (see deliver_detailed below), though whether that
# classification actually cuts a delivery's retries short is a separate,
# opt-in choice (``stop_on_permanent``) — only src/main.py's webhook_outbox
# loop uses it, and even there a row isn't given up on until 3 delivery
# attempts have accumulated (see _finalize_webhook_outbox_row's "3-strike"
# rule) — a transient 403/404 during a brief CRM deploy must not kill a row.
_RETRYABLE_4XX = frozenset({408, 425, 429})


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
            # error_type alongside error: str(httpx.ReadTimeout) (and several
            # other httpx exceptions) is "" -- without error_type, a timeout
            # and a connect-refused both log as `"error": ""`, indistinguishable
            # from each other or from "nothing was even attempted".
            log.warning(
                "tenant event POST failed",
                extra={"url": url, "error": str(e), "error_type": type(e).__name__},
            )
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


@dataclass(frozen=True)
class DeliveryResult:
    """Full outcome of a ``deliver_detailed`` call — ``deliver`` below
    collapses this to just ``.ok`` for callers that never needed more.

    ``final_status`` is the last HTTP status code received, or ``-1`` if the
    last attempt was a transport-level failure (see ``_httpx_post``'s own
    ``error_type`` log for which exception). ``permanent`` is True when a
    non-retryable 4xx (see ``_RETRYABLE_4XX``) stopped retries early — a
    caller that persists failures (e.g. ``src/main.py``'s webhook_outbox
    loop) should NOT keep retrying a permanent failure the way it would a
    transient 5xx/timeout.
    """

    ok: bool
    final_status: int
    permanent: bool = False


def _envelope_event_type(envelope: dict[str, Any]) -> Optional[str]:
    """The event-type-ish field to log, whichever this envelope actually
    carries: call-event envelopes (``build_envelope`` above) use
    ``event_type``; BO chat-lifecycle bodies (``src.api.chat_webhooks.
    send_bo_webhook``) use ``event``. Logging only ever read ``event_type``,
    so a BO delivery's logs always carried ``event_type=None``."""
    return envelope.get("event_type") or envelope.get("event")


async def deliver_detailed(
    url: str,
    envelope: dict[str, Any],
    secret: Optional[str] = None,
    *,
    http_post: Optional[HTTPPoster] = None,
    stop_on_permanent: bool = False,
) -> DeliveryResult:
    """POST a signed envelope to ``url`` with bounded retry, returning full
    delivery detail (``DeliveryResult``) rather than collapsing to a bool —
    see ``deliver`` below for the bool-only convenience wrapper most callers
    still use. Never raises.

    ``stop_on_permanent`` (default False): when True, a non-retryable 4xx
    (anything but 408/425/429 — see ``_RETRYABLE_4XX``) stops retrying after
    just ONE attempt instead of exhausting ``_MAX_ATTEMPTS`` — that status
    means the request itself is wrong (bad payload, expired signature,
    unknown endpoint), and repeating it verbatim can only ever get the same
    answer. This is opt-in and used ONLY by ``src/main.py``'s webhook_outbox
    loop, which needs a fast within-one-attempt signal to feed its own
    cross-attempt 3-strike dead rule. Every other caller — including
    ``deliver`` below, and so ``src/api/call_store.py``'s call-event
    webhooks and ``src/api/chat_webhooks.py``'s in-line ``session_closed``
    attempt — gets the default ``False``, i.e. the SAME unconditional
    retry-``_MAX_ATTEMPTS``-times-regardless-of-status behavior this
    function has always had. ``DeliveryResult.permanent`` on the final
    result still reflects whether the LAST status was a non-retryable 4xx
    either way — only the retry-loop's own attempt count is gated by this
    flag, not the classification itself.
    """
    raw = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Signature"] = sign_body(secret, raw)
    poster = http_post or _httpx_post
    event_type = _envelope_event_type(envelope)
    status = -1
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
            return DeliveryResult(ok=True, final_status=status)
        is_permanent = 400 <= status < 500 and status not in _RETRYABLE_4XX
        if is_permanent and stop_on_permanent:
            log.warning(
                "tenant event delivery permanent failure — not retrying",
                extra={"url": url, "event_type": event_type, "status": status},
            )
            return DeliveryResult(ok=False, final_status=status, permanent=True)
        if attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(_BACKOFF_BASE_S * (2**attempt))
    final_permanent = 400 <= status < 500 and status not in _RETRYABLE_4XX
    log.warning(
        "tenant event delivery exhausted retries",
        # final_status: -1 means the last attempt was a transport-level
        # failure (see _httpx_post's own error_type log for which exception);
        # any other value is the last non-2xx HTTP status code received.
        extra={"url": url, "event_type": event_type, "final_status": status},
    )
    return DeliveryResult(ok=False, final_status=status, permanent=final_permanent)


async def deliver(
    url: str,
    envelope: dict[str, Any],
    secret: Optional[str] = None,
    *,
    http_post: Optional[HTTPPoster] = None,
) -> bool:
    """POST a signed envelope to ``url`` with bounded retry. Returns True on a
    2xx within the retry budget. Never raises. Retries the full
    ``_MAX_ATTEMPTS`` budget regardless of status code (4xx included) —
    unconditionally, this wrapper never passes ``stop_on_permanent``.

    Thin bool-only wrapper around ``deliver_detailed`` — kept as-is for
    existing callers (``src/api/call_store.py``'s call-event webhooks,
    ``src/api/chat_webhooks.py``'s inline attempt) that only ever needed
    success/failure, not the full ``DeliveryResult``, and whose behavior
    must not change just because ``deliver_detailed`` gained an opt-in
    fast-fail mode for the durable outbox queue."""
    result = await deliver_detailed(url, envelope, secret, http_post=http_post)
    return result.ok
