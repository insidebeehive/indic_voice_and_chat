"""Inbound webhook: deposit dispute screenshot verification callback.

The customer disputes a failed deposit and uploads a screenshot (handled by a
chatbot tool, elsewhere); that tool forwards the screenshot to a per-tenant
verification vendor and gets back a fast synchronous ack — NOT the verdict.
The verdict itself arrives later, out of band, via this endpoint: the vendor
POSTs the outcome for a given ``request_id`` (a ``DepositVerificationRequest``
row id), we verify it's genuinely from the vendor (HMAC over the raw body,
same scheme/header ``send_bo_webhook`` uses outbound), persist the verdict,
and push it into the live chat conversation if one is still connected.

No vendor exists yet — the request/response contract here (``status``/
``order_id``/``detail``) is ours to define.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth.middleware import tenant_from_id
from src.integration.tenant_events import verify_signature
from src.models.deposit_verification import DepositVerificationRequest

log = logging.getLogger(__name__)
router = APIRouter(prefix="/deposit-verification", tags=["deposit-verification"])

# Identical header name to the one `send_bo_webhook` (src/api/chat_webhooks.py,
# via src/integration/tenant_events.py's `deliver`) sends on the OUTBOUND side —
# the vendor is expected to sign its callback the same way we sign our own
# outbound webhooks, so verification here mirrors that convention exactly.
_SIGNATURE_HEADER = "X-Signature"


class DepositVerificationCallbackBody(BaseModel):
    """Contract for the vendor's verdict callback."""
    status: Literal["verified", "rejected"]
    order_id: str
    detail: str = ""


_VERDICT_MESSAGES = {
    "verified": (
        "Good news — we've verified your deposit and it's been credited to "
        "your account. Thanks for your patience!"
    ),
    "rejected": (
        "We've reviewed the screenshot you shared, but we weren't able to "
        "verify this deposit. If you believe this is a mistake, please reach "
        "out to our support team with your payment reference."
    ),
}


@router.post("/callback/{request_id}")
async def deposit_verification_callback(
    request_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Vendor calls this with the verdict for a previously-submitted deposit
    dispute screenshot. Signed with HMAC-SHA256 over the raw request body
    (see ``_SIGNATURE_HEADER`` / ``verify_signature``) — an unsigned or
    incorrectly-signed callback is rejected with 401 rather than silently
    accepted, since there is no other authentication on this endpoint (the
    vendor has no tenant bearer token, only the opaque ``request_id``)."""
    raw_body = await request.body()
    try:
        body = DepositVerificationCallbackBody.model_validate_json(raw_body)
    except Exception as exc:  # noqa: BLE001 — malformed body is a client error
        raise HTTPException(status_code=422, detail=f"invalid callback body: {exc}") from None

    row = await db.get(DepositVerificationRequest, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="deposit verification request not found")

    tenant = await tenant_from_id(row.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=503, detail="tenant unavailable")

    if not getattr(tenant.settings.deposit_verification, "enabled", False):
        # The row genuinely exists (checked above), so 403 is more honest
        # here than a 404 — the tenant has simply turned the feature off
        # since the request was created (or it was created before an
        # inconsistent state), matching the config-based 403s elsewhere in
        # this codebase (see src/api/calls.py) rather than masquerading as
        # "not found".
        raise HTTPException(status_code=403, detail="deposit verification is not enabled for this tenant")

    webhook_secret_env = getattr(tenant.settings.deposit_verification, "webhook_secret_env", None)
    secret: Optional[str] = tenant.secret_optional(webhook_secret_env)
    signature_header = request.headers.get(_SIGNATURE_HEADER)
    # No secret configured is treated as a verification failure, not an
    # implicit "unsigned callbacks are fine" — this endpoint has no other
    # authentication, so an unconfigured secret must not silently accept
    # arbitrary callbacks for the tenant.
    if not secret or not verify_signature(secret, raw_body, signature_header):
        log.warning(
            "deposit verification callback signature check failed",
            extra={"request_id": request_id, "tenant_id": row.tenant_id},
        )
        raise HTTPException(status_code=401, detail="invalid signature")

    if body.order_id != row.order_id:
        # Reconcile the callback body's order_id against the request's own
        # record rather than trusting the vendor's value outright — a
        # vendor-side bug/confusion here would otherwise resolve the wrong
        # dispute. Checked only after signature verification succeeds, so
        # this can't be used as an enumeration oracle by an unsigned caller.
        log.warning(
            "deposit verification callback order_id mismatch",
            extra={
                "request_id": request_id, "tenant_id": row.tenant_id,
                "expected_order_id": row.order_id, "received_order_id": body.order_id,
            },
        )
        raise HTTPException(status_code=400, detail="order_id does not match this verification request")

    if row.status != "pending":
        # Already resolved (verdict or timeout) — idempotent no-op so a
        # retried/duplicate vendor callback doesn't clobber state or double-push.
        return {"status": "already processed"}

    row.status = body.status
    row.verdict_payload = body.model_dump()
    row.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()

    from src.api.chat import push_async_message

    await push_async_message(
        row.session_id,
        _VERDICT_MESSAGES[body.status],
        role="system",
    )

    return {"status": "ok"}
