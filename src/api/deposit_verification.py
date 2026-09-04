"""Inbound webhooks: deposit dispute screenshot verification.

Two independent vendor contracts are handled here, for two different vendors:

1. ``POST /deposit-verification/callback/{request_id}`` — the original
   ("multipart_verdict") vendor. The customer disputes a failed deposit and
   uploads a screenshot (handled by a chatbot tool, elsewhere); that tool
   forwards the screenshot to this vendor and gets back a fast synchronous
   ack — NOT the verdict. The verdict itself arrives later, out of band, via
   this endpoint: the vendor POSTs a single terminal ``verified``/``rejected``
   outcome for a given ``request_id`` (a ``DepositVerificationRequest`` row
   id), we verify it's genuinely from the vendor (HMAC over the raw body,
   same scheme/header ``send_bo_webhook`` uses outbound), persist the verdict,
   and push it into the live chat conversation if one is still connected.

2. ``POST /deposit-verification/reply/{token}`` — a second ("json_ticket_relay")
   vendor with a fundamentally different contract: it only ever knows our
   ``order_id`` (never our internal ``request_id``), it may send multiple
   non-terminal messages per order (an "agent_reply" progress trail plus an
   "auto" holding message) instead of one terminal verdict, and tenant
   identity comes from an unguessable capability token in the URL path
   (mirroring the Chatwoot webhook's ``webhook_id`` pattern) rather than the
   request body. Every relayed message resets ("slides") the request's
   timeout window, since silence — not a single verdict — is this vendor's
   only implicit "done" signal.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import middleware as auth_middleware
from src.auth.audit import log_denied, token_fingerprint
from src.auth.context import TenantContext
from src.auth.middleware import tenant_from_id
from src.integration.tenant_events import verify_signature, verify_signature_hex
from src.models.chat import ChatSession
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


# --- json_ticket_relay vendor: multi-message ticket reply relay -----------

# C0 control characters (0x00-0x1F) EXCEPT '\n' (0x0A) — this vendor's
# `message` field is free text that lands directly in a `role="system"`
# `ChatMessage`, which `_hydrate_agent_history` (src/api/chat.py) replays as
# an "assistant" turn in the LLM's message list on the customer's next turn
# (role="system" here is our own DB/display label, NOT an LLM system-
# instruction role — see _HYDRATE_ROLES / the system->assistant mapping in
# that function). That's still a genuinely new trust surface vs. the
# callback endpoint above (which only ever emits two fixed internal
# strings): unbounded vendor-supplied text landing in what the model treats
# as its own prior turn, so it's defensively stripped of control characters
# before it's relayed.
_C0_CONTROL_RE = re.compile(r"[\x00-\x09\x0b-\x1f]")

_MAX_RELAY_MESSAGE_LEN = 2000
# Display/debugging history cap — kept small since it's only ever read by a
# human/tooling, not used for correctness. Deliberately separate from
# _MAX_RELAY_SIG_HISTORY (replay-dedupe signatures), which needs a much
# larger cap — see the dedupe check below for why.
_MAX_RELAY_HISTORY = 50
_MAX_RELAY_SIG_HISTORY = 500

# Single identical 401 body for every pre-row-lookup failure on the reply
# route below (unknown/invalid token, disabled feature, wrong contract,
# missing/bad signature) — see that route's docstring for why the body, not
# just the status code, must be indistinguishable across these cases.
_UNAUTHORIZED_DETAIL = "unauthorized"


def _reject_unauthorized(message: str, **extra: Any) -> HTTPException:
    """Log a ``/reply/{token}`` auth failure and return the uniform 401 to
    raise.

    Mirrors ``_reject_unauthorized`` in ``src/api/external_chat.py``: every
    distinct auth-failure reason (unknown token, feature disabled, wrong
    contract, bad signature) gets the IDENTICAL status+detail on the wire
    (``_UNAUTHORIZED_DETAIL``) — an operator debugging "the vendor is
    getting 401'd" can otherwise not tell stale-token vs. feature-disabled
    vs. contract-misconfigured apart, even though the response is correctly
    uniform. Differentiated only in our own logs, via a distinct ``message``
    plus a structured ``reason=`` field in ``extra`` (present on every call
    site, so all four log lines are structurally comparable). Callers use
    ``raise _reject_unauthorized(...)``. Never pass the raw token, secret,
    or request body in ``extra`` — fingerprint a token first (see
    ``token_fingerprint``) if it needs to appear at all.

    Logged via ``log_denied`` (``src/auth/audit.py``), not a plain
    ``log.warning`` — this route is unauthenticated and token-probeable, so
    an attacker grinding tokens could otherwise drive unbounded WARN log
    volume. ``log_denied`` rate-limits per ``(reason, client_ip)`` and emits
    a suppression summary instead once that limit is hit within its window.
    """
    log_denied(
        logging.WARNING, message,
        event="auth_rejected", route="/deposit-verification/reply/{token}",
        **extra,
    )
    return HTTPException(status_code=401, detail=_UNAUTHORIZED_DETAIL)


class DepositTicketReplyBody(BaseModel):
    """Contract for the json_ticket_relay vendor's ticket-reply callback.

    ``type`` is deliberately a bare ``str``, not a ``Literal`` — an
    unrecognized future type value must still be accepted (200), not
    rejected (400), since this vendor may introduce new message types
    without notice and rejecting them would drop the message on the floor
    with no retry.
    """
    order_id: str = Field(min_length=1)
    message: str = ""
    type: str = "agent_reply"


def _clean_relay_message(text: str) -> str:
    """Strip C0 control chars (keeping '\\n') and truncate to the bounded
    history/message length this vendor's relay allows."""
    cleaned = _C0_CONTROL_RE.sub("", text)
    return cleaned[:_MAX_RELAY_MESSAGE_LEN]


def _session_is_live(session: Optional[ChatSession]) -> bool:
    if session is None:
        return False
    if session.status == "ended" or session.mode == "closed":
        return False
    return True


@router.post("/reply/{token}")
async def deposit_ticket_reply(
    token: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> Response:
    """json_ticket_relay vendor's ticket-reply callback.

    Tenant identity comes from the unguessable ``token`` capability token in
    the URL path (mirrors the Chatwoot webhook's ``webhook_id`` pattern —
    see ``src/api/external_chat.py``), never from the request body. Every
    failure mode before the row lookup returns a uniform 401 with the exact
    same body (``_UNAUTHORIZED_DETAIL``) — this route has no other
    authentication than the token + signature, so a differentiated status
    code OR response body (invalid token vs. disabled vs. wrong contract vs.
    bad signature) would let an attacker probing tokens learn something from
    the response. The vendor gets a 200 ack on every successfully-
    authenticated callback for a known ``order_id``, even ones that end up
    being a no-op (closed session, duplicate replay), since there is no
    per-message retry semantics to preserve on this vendor's side — an
    authenticated callback for an unknown ``order_id`` still 404s (see the
    row lookup below), since there is nothing to relay to or dedupe against.
    """
    # Resolve tenant from the path token BEFORE reading the request body at
    # all (mirrors src/api/external_chat.py's chatwoot_webhook) — an
    # unauthenticated caller must not be able to make the process buffer an
    # arbitrarily large body pre-auth. `raw_body` is only actually needed
    # once signature verification runs, below.
    resolver = getattr(request.app.state, "tenant_resolver", None) or auth_middleware._resolver
    tenant: Optional[TenantContext] = None
    if resolver is not None and hasattr(resolver, "resolve_by_deposit_verification_reply_token"):
        tenant = await resolver.resolve_by_deposit_verification_reply_token(token)
    if tenant is None:
        raise _reject_unauthorized(
            "deposit ticket reply: unknown token",
            reason="unknown_token", token_fp=token_fingerprint(token),
        )

    dv_config = tenant.settings.deposit_verification
    if not dv_config.enabled:
        # Uniform 401 body, NOT the 403 the callback endpoint above uses for
        # this same case — there, the caller already proved it knows the
        # row's request_id AND the HMAC secret before this check runs; here,
        # the token/path is the ONLY proof of legitimacy so far (signature is
        # checked below), so a differentiated response would leak
        # tenant-config state to a caller we haven't authenticated yet.
        raise _reject_unauthorized(
            "deposit ticket reply: feature disabled for tenant",
            reason="disabled", tenant_id=tenant.id,
        )

    if dv_config.contract != "json_ticket_relay":
        # Prevents a multipart_verdict tenant's token (if one somehow
        # existed) from being used against this route.
        raise _reject_unauthorized(
            "deposit ticket reply: wrong contract configured for tenant",
            reason="wrong_contract", tenant_id=tenant.id, contract=dv_config.contract,
        )

    raw_body = await request.body()

    secret: Optional[str] = tenant.secret_optional(dv_config.webhook_secret_env)
    signature_header = request.headers.get(_SIGNATURE_HEADER)
    if not secret or not verify_signature_hex(secret, raw_body, signature_header):
        raise _reject_unauthorized(
            "deposit ticket reply: signature check failed",
            reason="bad_signature", tenant_id=tenant.id,
        )

    try:
        body = DepositTicketReplyBody.model_validate_json(raw_body)
    except Exception as exc:  # noqa: BLE001 — malformed body is a client error
        raise HTTPException(status_code=400, detail=f"invalid reply body: {exc}") from None

    result = await db.execute(
        select(DepositVerificationRequest)
        .where(
            DepositVerificationRequest.tenant_id == tenant.id,
            DepositVerificationRequest.order_id == body.order_id,
        )
        # Deliberately NO status filter — this vendor never sends a terminal
        # status, so a row that already timed out (e.g. the existing
        # sliding-window timeout fired while a genuine late reply was still
        # in flight) must still be found and relayed to. `id DESC` breaks
        # ties for same-second `created_at` deterministically (repeatable
        # across retries/pagination) — NOT by recency: `id` is
        # `f"dvr_{uuid.uuid4().hex}"`, a random value, not a monotonic one.
        # A real same-second tie is only practically possible on
        # second-granularity `created_at` backends (e.g. some SQLite
        # configs); Postgres's microsecond-precision timestamps make this a
        # non-issue in production.
        .order_by(DepositVerificationRequest.created_at.desc(), DepositVerificationRequest.id.desc())
        .limit(1)
    )
    row = result.scalars().first()
    if row is None:
        # Safe post-signature: not an enumeration oracle since the caller
        # already proved it holds a valid tenant token + signing secret.
        return JSONResponse(status_code=404, content={"status": "unknown order_id"})

    session = await db.get(ChatSession, row.session_id)
    if not _session_is_live(session):
        return {"status": "session closed"}

    # Known, accepted race (not fixed here): this whole block is a
    # read-modify-write on `row.verdict_payload`, not a CAS — two concurrent
    # relays for the same order could both read the same starting payload
    # and each write back a version missing the other's entry. Worst case is
    # one duplicate/missing chat line; a real fix would need row-level
    # locking, which risks SQLite/Postgres compatibility issues in tests, so
    # it's left as-is.
    body_hash = hashlib.sha256(raw_body).hexdigest()
    payload = dict(row.verdict_payload or {})
    # Replay/dedupe signatures are kept in their own list, separate from
    # `replies` (the display/debugging history capped at _MAX_RELAY_HISTORY).
    # These two lists used to be the same list — but with a single 50-entry
    # cap, an exact replay of an old signed body would stop being recognized
    # as a duplicate (and get re-relayed into the live chat) as soon as 50
    # further messages had landed on the same order. `reply_sigs` gets a much
    # larger (but still bounded) cap instead, well beyond any realistic
    # per-order message count, so dedupe protection doesn't silently expire
    # while the display history keeps rotating.
    reply_sigs = list(payload.get("reply_sigs") or [])
    if not reply_sigs and payload.get("replies"):
        # Migration path: a row written before the reply_sigs split existed
        # only has `replies`, each entry carrying its own "sig" key. Seed
        # reply_sigs from it on first read so a replay of a pre-split
        # message is still caught, instead of silently starting dedupe over
        # from an empty list.
        reply_sigs = [
            r.get("sig") for r in payload["replies"]
            if isinstance(r, dict) and r.get("sig")
        ]
    if body_hash in reply_sigs:
        # Replay/dedupe: this vendor's contract carries no nonce/timestamp to
        # distinguish a genuine retry of the same signed body from a replay
        # attack, so an identical body is always treated as "already
        # relayed" — a deliberate simplification that also collapses two
        # genuinely-distinct-but-textually-identical messages into one hit.
        return {"status": "duplicate ignored"}

    # Stripped/truncated ONCE, upstream of both destinations — the stored
    # `replies` entry and the value actually relayed via push_async_message
    # must never diverge (they used to: the stored copy kept the vendor's
    # raw, only length-truncated text while the relayed copy was stripped
    # first, so a display/debug read of verdict_payload could show control
    # characters that were never actually shown to the customer).
    cleaned_message = _clean_relay_message(body.message)

    replies = list(payload.get("replies") or [])
    replies.append({
        "sig": body_hash,
        "type": body.type,
        "message": cleaned_message,
        "received_at": datetime.now(timezone.utc).isoformat(),
    })
    reply_sigs.append(body_hash)
    payload["replies"] = replies[-_MAX_RELAY_HISTORY:]
    payload["reply_sigs"] = reply_sigs[-_MAX_RELAY_SIG_HISTORY:]
    row.verdict_payload = payload  # reassign whole dict for JSON-column change tracking
    row.timeout_at = (
        datetime.now(timezone.utc).replace(tzinfo=None)
        + timedelta(minutes=dv_config.timeout_minutes)
    )
    await db.commit()

    from src.api.chat import push_async_message, schedule_verification_timeout

    if cleaned_message:
        await push_async_message(row.session_id, cleaned_message, role="system")

    # Re-arm the timeout for the new (slid) deadline. `_check_and_timeout_
    # verification`'s scheduled sleep does not reschedule itself if it wakes
    # up before `timeout_at` — see that function's docstring — so a fresh
    # timer must be armed here on every relay.
    schedule_verification_timeout(row.id, row.session_id, dv_config.timeout_minutes)

    return {"status": "ok"}
