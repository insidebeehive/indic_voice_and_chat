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
    for attempt in range(_MAX_ATTEMPTS):
        try:
            status = await poster(url, raw, headers)
        except Exception:  # noqa: BLE001 - a custom poster must not break the call
            log.exception("tenant event poster raised", extra={"url": url})
            status = -1
        if 200 <= status < 300:
            return True
        if attempt < _MAX_ATTEMPTS - 1:
            await asyncio.sleep(_BACKOFF_BASE_S * (2**attempt))
    log.warning("tenant event delivery exhausted retries",
                extra={"url": url, "event_type": envelope.get("event_type")})
    return False
