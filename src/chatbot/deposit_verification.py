"""Outbound side of deposit dispute screenshot verification: submits the
customer's most recently uploaded screenshot + order_id to the tenant's
configured verification vendor webhook and returns a fast synchronous ack —
NOT the verdict. The verdict arrives later via the inbound callback endpoint
(``src/api/deposit_verification.py``).

Two vendor contracts, selected by ``DepositVerificationConfig.contract``:

- ``multipart_verdict`` (default, original) — POSTs the screenshot bytes as a
  multipart file upload + a JSON metadata sidecar, signed via ``sign_body``
  (``X-Signature: sha256=<hex>``). See ``_post_multipart_vendor``.
- ``json_ticket_relay`` (newer) — POSTs a plain JSON body
  ``{"order_id", "screenshot_url", "mobile"?}`` with ``Content-Type:
  application/json`` and a bare-hex HMAC in ``X-Signature`` (``sign_body_hex``,
  no ``sha256=`` prefix). ``screenshot_url`` is a time-limited signed URL from
  the tenant's media store (``IMediaStorage.signed_url``), not the raw bytes —
  so it requires a media store that can produce a real, publicly-fetchable
  URL. A relative/unsigned URL (e.g. from ``LocalMediaStorage``) is refused
  before any vendor call is attempted. See ``_post_json_ticket_vendor``.

  Caveat: once the signed URL's TTL (``screenshot_url_ttl_seconds``) expires,
  the vendor gets an opaque failure (e.g. an S3 403) with no re-issue channel
  in this vendor's protocol — there is no way to hand it a fresh URL after
  the fact. Operators should configure a generous TTL and rely on the
  existing timeout-to-human-escalation (``schedule_verification_timeout``) as
  the safety net for a stale/expired link, not treat this as retryable.

Called from ``ChatBotAgent._dispatch_tool``'s ``SUBMIT_DEPOSIT_VERIFICATION``
branch via the executor closure built in ``src/bootstrap.py``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import httpx
from sqlalchemy import select

from src.auth.context import TenantContext
from src.campaign.dnd_filter import normalize_phone
from src.config_tenant import DepositVerificationConfig, platform_webhook_base_url
from src.integration.tenant_events import sign_body, sign_body_hex
from src.interfaces.media_storage import IMediaStorage
from src.models.chat import ChatMessage, ChatSession
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

    if dv_config.contract == "json_ticket_relay":
        # The raw screenshot bytes fetched above were only an existence probe
        # for this contract (no_screenshot detection) — this vendor wants a
        # fetchable URL, not the bytes, so `data`/`mime` are discarded here.
        try:
            url = await media_store.signed_url(
                screenshot_row.media_url, dv_config.screenshot_url_ttl_seconds
            )
        except Exception:  # noqa: BLE001 — a failing vendor call must not kill the turn
            log.exception(
                "deposit verification: signed_url failed",
                extra={"ticket_id": ticket_id, "session_id": session_id, "request_id": request_id},
            )
            await _mark_error(sessionmaker, request_id)
            return {
                "status": "error",
                "message": (
                    "Could not submit the verification request right now. Let the customer "
                    "know you're having trouble and offer to escalate to a human agent."
                ),
            }
        if urlsplit(url).scheme != "https":
            # Critical security gate: LocalMediaStorage.signed_url() happily
            # returns a relative, unsigned, non-expiring path — that must
            # never be handed to an external vendor as if it were a real
            # time-limited public URL. A bare substring check for "://" would
            # still pass a non-HTTPS absolute URL (e.g. "http://..." or an
            # internal/RFC1918 host reachable over plain HTTP), so this
            # requires an explicit "https" scheme instead. Refuse rather than
            # attempt the POST.
            log.error(
                "deposit verification: signed_url did not return an https URL "
                "(local media fallback or misconfigured store?) — refusing to send it "
                "to an external vendor",
                extra={"ticket_id": ticket_id, "session_id": session_id, "request_id": request_id},
            )
            await _mark_error(sessionmaker, request_id)
            return {
                "status": "error",
                "message": (
                    "Could not submit the verification request right now. Let the customer "
                    "know you're having trouble and offer to escalate to a human agent."
                ),
            }

        if urlsplit(dv_config.webhook_url).scheme != "https":
            # Same gate as the signed-URL check above, applied to the vendor
            # endpoint itself: the outbound POST body carries that signed
            # HTTPS screenshot URL, so a misconfigured "http://" webhook_url
            # would ship it in cleartext over the network to whoever's on
            # the wire, defeating the point of signing/expiring it. Refuse
            # rather than attempt the POST.
            log.error(
                "deposit verification: webhook_url is not https — refusing to send the "
                "signed screenshot URL to a non-HTTPS vendor endpoint",
                extra={"ticket_id": ticket_id, "session_id": session_id, "request_id": request_id},
            )
            await _mark_error(sessionmaker, request_id)
            return {
                "status": "error",
                "message": (
                    "Could not submit the verification request right now. Let the customer "
                    "know you're having trouble and offer to escalate to a human agent."
                ),
            }

        async with sessionmaker() as db:
            chat_session = await db.get(ChatSession, session_id)
        mobile = _resolve_mobile(chat_session, dv_config) if chat_session is not None else None

        ok = await _post_json_ticket_vendor(
            dv_config=dv_config,
            secret=secret,
            order_id=order_id,
            screenshot_url=url,
            mobile=mobile,
            timeout_s=timeout_s,
            ticket_id=ticket_id,
            session_id=session_id,
        )
    else:
        ok = await _post_multipart_vendor(
            dv_config=dv_config,
            secret=secret,
            request_id=request_id,
            order_id=order_id,
            tenant_id=tenant.id,
            data=data,
            mime=mime,
            timeout_s=timeout_s,
            ticket_id=ticket_id,
            session_id=session_id,
        )

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


async def _post_multipart_vendor(
    *,
    dv_config: DepositVerificationConfig,
    secret: str,
    request_id: str,
    order_id: str,
    tenant_id: str,
    data: bytes,
    mime: str,
    timeout_s: float,
    ticket_id: str | None,
    session_id: str,
) -> bool:
    """``multipart_verdict`` contract: POST the screenshot bytes as a
    multipart file upload + a signed JSON metadata sidecar. Extracted
    unchanged from the original single-contract implementation — behavior is
    byte-for-byte identical to before the ``json_ticket_relay`` contract was
    added. Returns True on any 2xx response, False otherwise (never raises)."""
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
        "tenant_id": tenant_id,
        "callback_url": callback_url,
    }
    # The metadata is a multipart sidecar field, not the whole request body,
    # so there's no single "raw body" to sign the way sign_body's other
    # callers do — the deviation is in what's signed (this canonical JSON),
    # not the algorithm, which is sign_body's unchanged HMAC-SHA256 scheme.
    canonical_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    # `secret` is resolved (and required) by the guard at the top of the
    # caller, so this is always a real signature — never an empty header.
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
        return 200 <= resp.status_code < 300
    except Exception:  # noqa: BLE001 — a failing vendor call must not kill the turn
        log.exception("deposit verification vendor POST failed", extra={
            "ticket_id": ticket_id, "session_id": session_id, "request_id": request_id,
        })
        return False


async def _post_json_ticket_vendor(
    *,
    dv_config: DepositVerificationConfig,
    secret: str,
    order_id: str,
    screenshot_url: str,
    mobile: str | None,
    timeout_s: float,
    ticket_id: str | None,
    session_id: str,
) -> bool:
    """``json_ticket_relay`` contract: POST a plain JSON body signed with a
    bare-hex HMAC (``sign_body_hex`` — no ``sha256=`` prefix, unlike
    ``sign_body``'s multipart-sidecar signature). Uses ``content=raw`` rather
    than httpx's own ``json=`` serializer so the bytes that get signed are
    guaranteed to be exactly the bytes that get sent. Returns True on any 2xx
    response (including a "duplicate ignored" verdict — both are vendor-side
    success), False otherwise (never raises)."""
    payload: dict = {"order_id": order_id, "screenshot_url": screenshot_url}
    if mobile:
        payload["mobile"] = mobile
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", _SIGNATURE_HEADER: sign_body_hex(secret, raw)}

    bounded_timeout = min(timeout_s, _MAX_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=bounded_timeout) as client:
            resp = await client.post(dv_config.webhook_url, content=raw, headers=headers)
        ok = 200 <= resp.status_code < 300
        if ok:
            try:
                body = resp.json()
            except Exception:  # noqa: BLE001 — best-effort only, a parse miss must not fail the turn
                body = None
            if isinstance(body, dict) and body.get("result") == "duplicate ignored":
                # Distinct from a normal "ok": this row will likely never get
                # a reply, since someone else's ticket already exists for
                # this order_id — it will fall through to the existing
                # timeout-to-human-escalation instead.
                log.info(
                    "deposit verification: vendor reported 'duplicate ignored' for this "
                    "order_id — a reply may never arrive; the timeout escalation is the "
                    "safety net here",
                    extra={"ticket_id": ticket_id, "session_id": session_id, "order_id": order_id},
                )
        return ok
    except Exception:  # noqa: BLE001 — a failing vendor call must not kill the turn
        log.exception("deposit verification vendor POST failed", extra={
            "ticket_id": ticket_id, "session_id": session_id, "order_id": order_id,
        })
        return False


# ASCII-only digit check — deliberately NOT `str.isdigit()`, which is
# Unicode-aware and returns True for non-ASCII digit scripts (Arabic-Indic,
# Devanagari, superscripts, ...). The real defense against those already
# happened in `normalize_phone()`, which strips everything except `[0-9+]`
# (Unicode digit scripts included) before this regex ever sees the value;
# this regex is a secondary validation on that already-normalized value —
# it just rejects anything that doesn't reduce to a plausible 10-15 digit
# number (e.g. a candidate that normalized down to nothing, or to something
# too short/long), it does not itself do the stripping.
_PHONE_SHAPE_RE = re.compile(r"\+?[0-9]{10,15}")


def _looks_like_phone_number(value: str) -> bool:
    """True if ``value`` is a plausible phone number: ASCII digits only, with
    at most one leading ``+``, 10-15 digits long. Shared by every candidate
    source (``extra_data`` keys and the ``customer_id`` fallback) so none of
    them can forward arbitrary-length arbitrary text to the external vendor.
    Callers must run ``normalize_phone()`` on the candidate first — this only
    validates shape, it does not strip punctuation/spacing itself."""
    return bool(_PHONE_SHAPE_RE.fullmatch(value))


def _resolve_mobile(chat_session: ChatSession, dv_config: DepositVerificationConfig) -> str | None:
    """Best-effort mobile number for the ``json_ticket_relay`` payload's
    optional ``mobile`` field. Tries each configured ``extra_data`` key in
    order, then falls back to ``customer_id`` — every candidate is first
    normalized with ``normalize_phone()`` (``src/campaign/dnd_filter.py``,
    e.g. ``"+91 (999) 999-9999" -> "+919999999999"``) so real-world
    CRM-supplied punctuation/spacing (``"+91-98765-43210"``, ``"(998)
    887-7700"``, ...) doesn't get mistaken for a malformed value, and only
    then validated with ``_looks_like_phone_number`` before being accepted.
    Never sends a garbage value, so callers should omit the ``mobile`` key
    entirely when this returns None.

    ``extra_data`` is populated straight from client-supplied request
    metadata (``req.metadata`` on session creation — see ``src/api/chat.py``),
    so a candidate failing validation here falls through to the next
    configured key rather than being forwarded verbatim to the vendor."""
    extra = chat_session.extra_data if isinstance(chat_session.extra_data, dict) else {}
    for key in dv_config.mobile_metadata_keys:
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            candidate = normalize_phone(value.strip())
            if _looks_like_phone_number(candidate):
                return candidate
    customer_id = normalize_phone((chat_session.customer_id or "").strip())
    if _looks_like_phone_number(customer_id):
        return customer_id
    return None


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
