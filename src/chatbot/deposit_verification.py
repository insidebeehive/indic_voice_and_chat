"""Outbound side of deposit dispute screenshot verification: submits the
customer's most recently uploaded screenshot + order_id to the tenant's
configured verification vendor webhook and returns a fast synchronous ack —
NOT the verdict. The verdict arrives later via the inbound callback endpoint
(``src/api/deposit_verification.py``).

Called from ``ChatBotAgent._dispatch_tool``'s ``SUBMIT_DEPOSIT_VERIFICATION``
branch via the executor closure built in ``src/bootstrap.py``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select

from src.auth.context import TenantContext
from src.config_tenant import platform_webhook_base_url
from src.integration.tenant_events import sign_body
from src.interfaces.media_storage import IMediaStorage
from src.models.chat import ChatMessage
from src.models.deposit_verification import DepositVerificationRequest

log = logging.getLogger(__name__)

# Bounded ceiling for the outbound vendor POST, independent of the caller's
# own per-tool-call budget — this call carries a multipart file upload and
# must not be allowed to eat the whole turn budget.
_MAX_TIMEOUT_S = 15.0

# Same header name the inbound callback (src/api/deposit_verification.py)
# expects and that outbound tenant-event webhooks already send.
_SIGNATURE_HEADER = "X-Signature"


async def submit_deposit_verification(
    *,
    tenant: TenantContext,
    session_id: str,
    order_id: str,
    sessionmaker,
    media_store: IMediaStorage,
    timeout_s: float,
    ticket_id: str | None = None,
) -> dict:
    dv_config = tenant.settings.deposit_verification
    secret = tenant.secret_optional(dv_config.webhook_secret_env)
    if not dv_config.enabled or not dv_config.webhook_url or media_store is None or not secret:
        # Defensive: the tool should only be registered when this is true
        # (see src/bootstrap.py), but guard here too in case it's ever
        # invoked without that gate. The secret is part of the gate because
        # the inbound verdict callback (src/api/deposit_verification.py)
        # 401s anything it can't HMAC-verify — submitting without one would
        # create a request whose verdict can never be accepted.
        return {"status": "error", "message": "Verification is not available for this account."}

    order_id = (order_id or "").strip()
    if not order_id:
        # The callback handler cross-checks the vendor's order_id against the
        # value stored on this row with strict equality, so an empty/missing
        # order_id here would guarantee a 400 on the verdict callback and
        # leave the request to time out. Fail fast and tell the LLM instead.
        return {
            "status": "missing_order_id",
            "message": (
                "No order id was provided. Look up the customer's failed deposit with the "
                "deposit-status tool first and call this again with the order id from "
                "that response."
            ),
        }

    async with sessionmaker() as db:
        screenshot_row = (await db.execute(
            select(ChatMessage)
            .where(
                ChatMessage.session_id == session_id,
                ChatMessage.type == "image",
                ChatMessage.media_url.isnot(None),
            )
            .order_by(ChatMessage.id.desc())
            .limit(1)
        )).scalars().first()
        if screenshot_row is None:
            return {
                "status": "no_screenshot",
                "message": (
                    "No screenshot has been uploaded in this conversation yet. Ask the "
                    "customer to upload a screenshot of the successful transaction before "
                    "calling this tool again."
                ),
            }

        existing_pending = (await db.execute(
            select(DepositVerificationRequest).where(
                DepositVerificationRequest.session_id == session_id,
                DepositVerificationRequest.status == "pending",
            )
        )).scalars().first()
        if existing_pending is not None:
            return {
                "status": "already_pending",
                "message": (
                    "A verification is already in progress for this conversation; do not "
                    "resubmit — tell the customer we're still checking and will update "
                    "them here."
                ),
            }

        try:
            data, mime = await media_store.download(screenshot_row.media_url)
        except FileNotFoundError:
            log.warning(
                "deposit verification: screenshot missing from media store",
                extra={
                    "ticket_id": ticket_id, "session_id": session_id,
                    "media_url": screenshot_row.media_url,
                },
            )
            return {
                "status": "no_screenshot",
                "message": (
                    "No screenshot has been uploaded in this conversation yet. Ask the "
                    "customer to upload a screenshot of the successful transaction before "
                    "calling this tool again."
                ),
            }

        request_id = f"dvr_{uuid.uuid4().hex}"
        timeout_minutes = dv_config.timeout_minutes
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        timeout_at = now + timedelta(minutes=timeout_minutes)

        row = DepositVerificationRequest(
            id=request_id,
            tenant_id=tenant.id,
            session_id=session_id,
            order_id=order_id,
            screenshot_message_id=screenshot_row.id,
            status="pending",
            timeout_at=timeout_at,
        )
        db.add(row)
        await db.commit()

    base_url = platform_webhook_base_url()
    if base_url:
        parsed = urlsplit(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        callback_url = f"{origin}/api/v1/deposit-verification/callback/{request_id}"
    else:
        # Gap: no platform base URL is configured (WEBHOOK_BASE_URL unset), so
        # this falls back to a relative path — a vendor calling back over the
        # public internet needs an ABSOLUTE URL, so this callback_url is not
        # actually usable by an external vendor until WEBHOOK_BASE_URL is set.
        log.warning(
            "deposit verification: WEBHOOK_BASE_URL is not configured — callback_url "
            "sent to the vendor is a relative path and unusable by an external caller",
            extra={"ticket_id": ticket_id, "session_id": session_id, "request_id": request_id},
        )
        callback_url = f"/api/v1/deposit-verification/callback/{request_id}"

    metadata = {
        "request_id": request_id,
        "order_id": order_id,
        "tenant_id": tenant.id,
        "callback_url": callback_url,
    }
    # The metadata is a multipart sidecar field, not the whole request body,
    # so there's no single "raw body" to sign the way sign_body's other
    # callers do — the deviation is in what's signed (this canonical JSON),
    # not the algorithm, which is sign_body's unchanged HMAC-SHA256 scheme.
    canonical_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    # `secret` is resolved (and required) by the guard at the top of this
    # function, so this is always a real signature — never an empty header.
    signature = sign_body(secret, canonical_bytes)

    bounded_timeout = min(timeout_s, _MAX_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=bounded_timeout) as client:
            resp = await client.post(
                dv_config.webhook_url,
                data={"metadata": canonical_bytes.decode("utf-8")},
                files={"screenshot": (f"{request_id}.bin", data, mime)},
                headers={_SIGNATURE_HEADER: signature},
            )
        ok = 200 <= resp.status_code < 300
    except Exception:  # noqa: BLE001 — a failing vendor call must not kill the turn
        log.exception("deposit verification vendor POST failed", extra={
            "ticket_id": ticket_id, "session_id": session_id, "request_id": request_id,
        })
        ok = False

    if not ok:
        await _mark_error(sessionmaker, request_id)
        return {
            "status": "error",
            "message": (
                "Could not submit the verification request right now. Let the customer "
                "know you're having trouble and offer to escalate to a human agent."
            ),
        }

    from src.api.chat import schedule_verification_timeout

    schedule_verification_timeout(request_id, session_id, timeout_minutes)

    return {
        "status": "submitted",
        "message": (
            "Verification submitted successfully. Tell the customer this can take a few "
            "minutes and you'll update them right here in this chat as soon as it's back "
            "— they don't need to ask again in the meantime."
        ),
    }


async def _mark_error(sessionmaker, request_id: str) -> None:
    """Best-effort status update to 'error' — wrapped so a DB failure here
    doesn't mask the original submission failure being reported to the LLM."""
    try:
        async with sessionmaker() as db:
            row = await db.get(DepositVerificationRequest, request_id)
            if row is not None and row.status == "pending":
                row.status = "error"
                await db.commit()
    except Exception:  # noqa: BLE001 — best-effort only
        log.exception("deposit verification: failed to mark request as error", extra={"request_id": request_id})
