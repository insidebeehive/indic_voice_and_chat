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
  no ``sha256=`` prefix). ``screenshot_url`` is either the CRM's own inbound
  URL (``ChatMessage.source_media_url``, when the message carried one) or,
  failing that, a time-limited signed URL from the tenant's media store
  (``IMediaStorage.signed_url``) — never the raw bytes. Either way it must be
  a real, publicly-fetchable https URL: a relative/unsigned URL (e.g. from
  ``LocalMediaStorage``) or a plain-http CRM URL is refused before any vendor
  call is attempted. See ``_post_json_ticket_vendor``.

  Caveat: once a signed URL's TTL (``screenshot_url_ttl_seconds``) expires —
  or, on the source-URL path, once the CRM's own URL expires or is revoked —
  the vendor gets an opaque failure (e.g. an S3 403) with no re-issue channel
  in this vendor's protocol — there is no way to hand it a fresh URL after
  the fact. Operators should configure a generous TTL and rely on the
  existing timeout-to-human-escalation (``schedule_verification_timeout``) as
  the safety net for a stale/expired link, not treat this as retryable.

Called from ``ChatBotAgent._dispatch_tool``'s ``SUBMIT_DEPOSIT_VERIFICATION``
branch via the executor closure built in ``src/bootstrap.py``.

Pre-submission cross-check: before any of the above (vendor call, DB row),
``_cross_check_screenshot`` reads the screenshot with a separate vision call
(``src/chatbot/screenshot_extract.py`` — the chat model itself is not trusted
for this) and compares the amount/date it reports against the tenant's own
CRM (``get_player_transactions`` + ``get_player_latest_deposit_order``,
called via the ``crm_lookup`` callable wired in from ``src/bootstrap.py``,
the same path tenant-registered CRM tools use). A mismatch, an unreadable
screenshot, or a CRM lookup that can't be trusted all short-circuit before
the vendor is ever called — see that function's docstring for the full
decision table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Awaitable, Callable, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from sqlalchemy import and_, or_, select

from src.api.chat_cost import compute_chat_turn_cost
from src.auth.context import TenantContext
from src.campaign.dnd_filter import normalize_phone
from src.chatbot.screenshot_extract import ScreenshotExtractor
from src.chatbot.tool_executor import REDACTED_PLACEHOLDER
from src.config_tenant import DepositVerificationConfig, platform_webhook_base_url
from src.integration.tenant_events import sign_body, sign_body_hex
from src.interfaces.media_storage import IMediaStorage
from src.models.chat import ChatMessage, ChatSession
from src.models.deposit_verification import DepositVerificationRequest
from src.utils.http_fetch import fetch_capped
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

# Bounded ceiling for the outbound vendor POST, independent of the caller's
# own per-tool-call budget — this call carries a multipart file upload and
# must not be allowed to eat the whole turn budget.
_MAX_TIMEOUT_S = 15.0

# Same header name the inbound callback (src/api/deposit_verification.py)
# expects and that outbound tenant-event webhooks already send.
_SIGNATURE_HEADER = "X-Signature"

# CrmLookup(tool_name, args, *, timeout_s) -> the tenant's CRM tool result,
# same shape execute_crm_tool returns (src/chatbot/tool_executor.py): a
# successful call is {"status_code": int, "data": {...}}; a failure (unknown
# tool name, HTTP error, timeout, transport error) always carries an "error"
# key -- that is the single signal _cross_check_screenshot treats as "could
# not check". `timeout_s` is the REMAINING time budget at the point of the
# call (see the time-budget constants above), not a fixed per-call value --
# callers (src/bootstrap.py) must thread it through to whatever bounds the
# underlying HTTP call, not hardcode one.
CrmLookup = Callable[..., Awaitable[dict]]

# Pre-submission cross-check: fetching the screenshot bytes over https (for
# the json_ticket_relay source-URL path -- see _cross_check_screenshot) is
# bounded by whatever's left of submit_deposit_verification's own `timeout_s`
# budget at that point (see the time-budget constants below), capped at this
# ceiling regardless -- a screenshot fetch has no business taking longer than
# this even when the budget would otherwise allow it.
_CROSS_CHECK_FETCH_TIMEOUT_S = 10.0
_CROSS_CHECK_FETCH_MAX_BYTES = 10 * 1024 * 1024

# Time-budget management: `submit_deposit_verification` receives `timeout_s`
# as its share of the turn's cumulative tool budget (src/agents/chatbot.py's
# _exec_tool already wraps the whole call in asyncio.wait_for(timeout_s) as a
# backstop) -- everything below is this module being a good citizen WITHIN
# that budget rather than relying solely on the outer backstop, so a
# cross-check step that's about to blow the budget fails closed as
# could_not_check instead of being cut off mid-vendor-call by the outer
# wait_for (which would leave a "pending" row with no vendor call ever
# having been attempted, stuck until schedule_verification_timeout fires).
#
# Minimum time that must remain AFTER the cross-check passes before the
# vendor POST is even attempted -- below this, the row is never created and
# the vendor is never called; the submission fails closed as could_not_check
# instead of attempting a POST that's unlikely to complete in time.
_MIN_VENDOR_BUDGET_S = 8.0
# Margin subtracted from the remaining budget when computing the vendor
# POST's own timeout, so the POST itself doesn't get handed the exact full
# remaining slice with zero slack for the bookkeeping around it (building
# the callback URL, resolving `mobile`, the final DB round trip).
_VENDOR_TIMEOUT_MARGIN_S = 1.0

# Floor checked IMMEDIATELY before each vendor POST (after the row has
# already been created, and after whatever's left of the budget was spent on
# signed_url/ChatSession lookups/etc. since the _MIN_VENDOR_BUDGET_S gate
# ran) -- below this, the POST is not attempted at all, since a timeout this
# small wouldn't plausibly complete a real HTTP call. Deliberately smaller
# than _MIN_VENDOR_BUDGET_S: that gate runs BEFORE the row exists and before
# any of the per-contract bookkeeping below it has had a chance to consume
# more of the budget, so re-applying the same 8s floor here would rarely
# trigger on its own merits -- this is a last-resort check right at the POST
# call site, not a restatement of the earlier gate.
_MIN_POST_BUDGET_S = 3.0

SCREENSHOT_UNREADABLE_MESSAGE = (
    "The deposit amount and date could not be read from the screenshot. Do NOT "
    "submit. Ask the customer for a clearer screenshot showing the payment amount "
    "and date."
)

COULD_NOT_CHECK_MESSAGE = (
    "The customer's deposits could not be looked up to check against the "
    "screenshot. Do NOT submit. Apologise and offer to connect them to a human agent."
)

_NO_MATCHING_TRANSACTION_MESSAGE_TEMPLATE = (
    "The screenshot shows {amount} on {date}, but no deposit on the customer's "
    "account matches that amount and date. Do NOT submit for verification. Tell the "
    "customer we could not find a matching transaction for verification, mention the "
    "amount and date we read, and ask them to check they sent the right screenshot."
)


def _format_amount_for_message(value) -> str:
    """Human-friendly rendering of an extracted amount for the
    no_matching_transaction message -- ``1000.0`` reads as "1000", not
    "1000.0", while a genuinely fractional amount keeps its decimals."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# A comma is only treated as a thousands-separator (and stripped) when it's
# immediately followed by exactly three digits and then a non-digit or the
# string's end -- "1,000.50" strips to "1000.50", but "1.000,50" (comma used
# as a DECIMAL separator, European-style) is left alone, so its one dot +
# leftover comma combination fails _AMOUNT_SHAPE_RE below instead of being
# silently misread as 1.00050.
_THOUSANDS_COMMA_RE = re.compile(r",(?=\d{3}(?:\D|$))")
# A leading currency token this module accepts on a string amount, matched
# case-insensitively with an optional following space, and stripped before
# the shape check below. Deliberately a closed set -- an unrecognized prefix
# (any other currency, or stray text) is left in place so it fails the shape
# check and normalizes to None rather than being silently chopped off.
_CURRENCY_PREFIX_RE = re.compile(r"^(?:₹|rs\.?|inr|\$|usd)\s*", re.IGNORECASE)
_AMOUNT_SHAPE_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _normalize_amount(value: object) -> Optional[Decimal]:
    """Parse an amount from any of the shapes this module sees it in
    (extraction JSON numbers, CRM JSON numbers, or -- defensively -- a
    CRM/extraction-supplied string like ``"Rs. 500"``, ``"₹1,000.50"``,
    or ``"INR 500"``) into a ``Decimal``, or ``None`` if it can't be parsed
    as one. ``bool`` is rejected even though ``isinstance(True, int)`` is
    true in Python -- a flag is never an amount.

    Numbers (``int``/``float``) go through ``Decimal(str(x))`` unchanged.
    Strings are stripped of whitespace, thousands-separator commas, and one
    recognized leading currency token, then the remainder must match
    ``^-?\\d+(\\.\\d+)?$`` exactly -- anything else (an unrecognized currency
    token, stray text, an ambiguous format like ``"1.000,50"``) normalizes to
    ``None`` rather than being guessed at.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    if isinstance(value, str):
        cleaned = _THOUSANDS_COMMA_RE.sub("", value.strip())
        cleaned = _CURRENCY_PREFIX_RE.sub("", cleaned).strip()
        if not _AMOUNT_SHAPE_RE.fullmatch(cleaned):
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def _amounts_match(a: Decimal, b: Decimal) -> bool:
    # Deposits are positive; a transaction row's sign convention is not part
    # of the contract this compares against (see docs/crm-api-contract.md
    # §2 -- only casino/sports debits are documented as negative), so compare
    # magnitudes rather than trust either side's sign.
    return abs(a) == abs(b)


def _dates_within_one_day(a: date, b: date) -> bool:
    return abs((a - b).days) <= 1


def _parse_screenshot_date(value: object) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _resolve_tenant_zone(tenant: TenantContext) -> ZoneInfo:
    tz_name = getattr(tenant.settings, "timezone", None) or "UTC"
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


# Above this, a numeric CRM timestamp is treated as epoch MILLIseconds
# rather than seconds -- a seconds-since-epoch value for any date in this
# system's lifetime is comfortably below this (year ~33658 in seconds), so
# this cleanly separates the two without needing a units flag from the CRM.
_EPOCH_MS_THRESHOLD = 1e12


def _crm_timestamp_to_tenant_date(value: object, tz: ZoneInfo) -> Optional[date]:
    """Parse a CRM timestamp and return its calendar date IN THE TENANT'S
    TIMEZONE -- the whole point of the ±1-day window is to compare against
    the date the customer actually saw on their screen, which a server
    rendering a UTC timestamp's raw date would get wrong by a day near
    midnight IST (see the module's tests for the boundary case this guards).

    Accepts:
    - A numeric epoch timestamp (``int``/``float``) -- always UTC, seconds
      unless it exceeds ``_EPOCH_MS_THRESHOLD`` (then milliseconds).
    - An ISO-8601 string with an explicit offset/``Z`` -- parsed as that
      instant and converted to the tenant's timezone for the date.
    - An ISO-8601 string with NO timezone info (a naive CRM timestamp, e.g.
      ``"2026-06-20T10:30:00"``) -- CRM naive times are local, so this is
      interpreted as already being in the TENANT's timezone, not UTC.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            epoch_seconds = value / 1000.0 if abs(value) > _EPOCH_MS_THRESHOLD else value
            dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
        return dt.astimezone(tz).date()
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Naive CRM timestamp (including a date-only value, which parses as
        # midnight naive): interpret it as already being in the tenant's own
        # timezone rather than UTC.
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz).date()


def _deposit_transaction_matches(
    txn: object, shot_amount: Decimal, shot_date: date, tz: ZoneInfo,
) -> bool:
    if not isinstance(txn, dict) or (txn.get("type") or "").strip().lower() != "deposit":
        return False
    amount = _normalize_amount(txn.get("amount"))
    txn_date = _crm_timestamp_to_tenant_date(txn.get("timestamp"), tz)
    if amount is None or txn_date is None:
        return False
    return _amounts_match(amount, shot_amount) and _dates_within_one_day(shot_date, txn_date)


def _latest_order_matches(
    order: object, shot_amount: Decimal, shot_date: date, tz: ZoneInfo,
) -> bool:
    if not isinstance(order, dict):
        return False
    amount = _normalize_amount(order.get("amount"))
    order_date = _crm_timestamp_to_tenant_date(order.get("created_at"), tz)
    if amount is None or order_date is None:
        return False
    return _amounts_match(amount, shot_amount) and _dates_within_one_day(shot_date, order_date)


def _crm_call_ok(result: object) -> bool:
    """True for a CRM lookup that actually succeeded. Every failure shape
    execute_crm_tool / the crm_lookup wrapper can produce -- an unknown tool
    name, an HTTP error, a timeout, a transport error, a non-dict return --
    carries an "error" key (see CrmLookup's docstring above); this is the
    single check that covers all of them."""
    return isinstance(result, dict) and "error" not in result


async def _fetch_source_media_for_cross_check(url: str, *, timeout_s: float) -> tuple[bytes, str]:
    """https-only, size-capped, image-only fetch of a CRM-supplied
    ``source_media_url`` for the vision extraction call. Delegates to
    ``src.utils.http_fetch.fetch_capped``, the same SSRF-safe helper
    ``src/api/chat.py`` uses for caller-supplied media URLs: no redirects are
    followed, the host must resolve to a public address, and a non-https URL
    is rejected outright (``fetch_capped``'s ``assert_safe_url`` always
    requires https).

    ``timeout_s`` is the caller's remaining time budget -- this fetch never
    gets more than ``_CROSS_CHECK_FETCH_TIMEOUT_S`` regardless (its own
    independent ceiling, sized for a screenshot fetch, not the whole tool
    call), but it can get less when the budget is already tight."""
    bounded = max(0.1, min(_CROSS_CHECK_FETCH_TIMEOUT_S, timeout_s))
    data, content_type = await fetch_capped(
        url,
        max_bytes=_CROSS_CHECK_FETCH_MAX_BYTES,
        accept_content_types=("image/",),
        timeout=httpx.Timeout(bounded, connect=min(5.0, bounded)),
    )
    return data, content_type


async def _record_screenshot_extraction_cost(
    sessionmaker, session_id: str, extraction: dict, *, ticket_id: Optional[str],
) -> None:
    """Best-effort: bills the vision-extraction call's tokens onto the
    session's running cost total, the same mechanism src/api/chat.py's
    _persist_turn uses for the main chat turn's LLM cost. Runs on its own
    DB session (never the caller's) and never raises -- a billing
    bookkeeping failure must not block, or fail, the cross-check it is
    reporting the cost of."""
    usage = extraction.get("usage") or {}
    provider = extraction.get("provider")
    model = extraction.get("model")
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(usage.get("cached_tokens") or 0)
    if not provider or (not input_tokens and not output_tokens):
        return
    try:
        async with sessionmaker() as cost_db:
            cost = await compute_chat_turn_cost(
                cost_db, provider=provider, model=model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
            if not cost:
                return
            chat_session = await cost_db.get(ChatSession, session_id)
            if chat_session is None:
                return
            chat_session.cost = (chat_session.cost or 0.0) + cost
            chat_session.input_tokens = (chat_session.input_tokens or 0) + input_tokens
            chat_session.output_tokens = (chat_session.output_tokens or 0) + output_tokens
            await cost_db.commit()
    except Exception:  # noqa: BLE001 — billing bookkeeping must never break the cross-check
        log.exception(
            "deposit verification: failed to record screenshot extraction cost",
            extra={"ticket_id": ticket_id, "session_id": session_id},
        )


async def _cross_check_screenshot(
    *,
    tenant: TenantContext,
    sessionmaker,
    session_id: str,
    data: bytes,
    mime: str,
    extractor: Optional[ScreenshotExtractor],
    crm_lookup: Optional[CrmLookup],
    ticket_id: Optional[str],
    remaining: Callable[[], float],
) -> Optional[dict]:
    """Pre-submission cross-check. Returns a result dict (status != "submitted")
    when the caller must stop and return that to the LLM instead of
    proceeding; ``None`` when the screenshot and the CRM agree and the
    caller should continue exactly as before (insert the row, call the
    vendor).

    Decision table:
      - no extractor wired, the extraction call raises (including an
        unparseable/empty/safety-blocked LLM response -- see
        ``screenshot_extract.UnparseableExtractionResponse``), or it times
        out against ``remaining()`` -> could_not_check (fail CLOSED: all of
        these are system/infrastructure failures, not a judgement about the
        screenshot's readability, so none of them get the
        screenshot_unreadable message)
      - extraction SUCCEEDED (parsed) but reports unreadable, or missing
        amount/date -> screenshot_unreadable (the only path that reaches
        this status)
      - no crm_lookup wired, either CRM call errors/times out/returns
        non-JSON, or the latest-order lookup itself failed
        (status=lookup_unavailable) -> could_not_check (fail CLOSED: no
        vendor call either way)
      - CRM reachable but nothing matches (no deposit transaction row AND
        the latest order both matching on amount + date within ±1 calendar
        day in the tenant's timezone, including latest order
        status=no_recent_deposit) -> no_matching_transaction
      - both match -> None (proceed)

    ``remaining`` is the caller's time-budget callable (see the module's
    time-budget constants) -- the extraction call and the two CRM calls are
    each bounded by it so a slow extractor/CRM can't eat the whole tool
    budget; exhausting it is just another "could not check" failure.

    ``extractor``/``crm_lookup`` are optional only so unit tests that never
    reach past an earlier check (no_screenshot, already_pending,
    missing_order_id, ...) don't need to stub them — src/bootstrap.py's
    production wiring always supplies both.
    """
    if extractor is None:
        log.error(
            "deposit verification cross-check: no screenshot extractor wired — "
            "failing closed instead of submitting unchecked",
            extra={"ticket_id": ticket_id, "session_id": session_id},
        )
        return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

    try:
        extraction = await asyncio.wait_for(extractor(data, mime), timeout=remaining())
    except Exception:  # noqa: BLE001 — an extraction-call failure (incl. a budget
        # timeout, or an unparseable LLM response) must not kill the turn,
        # and is an infrastructure failure, not a judgement about the
        # screenshot -- so it fails closed as could_not_check, never
        # screenshot_unreadable.
        log.exception(
            "deposit verification: screenshot extraction call failed",
            extra={"ticket_id": ticket_id, "session_id": session_id},
        )
        return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

    await _record_screenshot_extraction_cost(sessionmaker, session_id, extraction, ticket_id=ticket_id)

    shot_amount = _normalize_amount(extraction.get("amount")) if extraction.get("readable") else None
    shot_date = _parse_screenshot_date(extraction.get("date")) if extraction.get("readable") else None
    if not extraction.get("readable") or shot_amount is None or shot_date is None:
        log.info(
            "deposit verification cross-check: screenshot_unreadable",
            extra={
                "ticket_id": ticket_id, "session_id": session_id,
                "reason": "extraction_unreadable_or_missing_fields",
            },
        )
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "deposit_verification_tool cross_check screenshot_unreadable",
                ticket_id=ticket_id, session_id=session_id, extraction=extraction,
            )
        return {"status": "screenshot_unreadable", "message": SCREENSHOT_UNREADABLE_MESSAGE}

    if crm_lookup is None:
        log.info(
            "deposit verification cross-check: could_not_check",
            extra={"ticket_id": ticket_id, "session_id": session_id, "reason": "no_crm_lookup"},
        )
        return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

    try:
        # Both CRM calls run concurrently (not back-to-back) since they're
        # independent reads needed for the same decision -- and the whole
        # pair is bounded by the remaining time budget as a backstop, on top
        # of the per-call `timeout_s` threaded through `crm_lookup` itself
        # (src/bootstrap.py passes it to execute_crm_tool's own httpx
        # timeout), in case a lookup implementation doesn't fully respect
        # that parameter.
        txns_result, order_result = await asyncio.wait_for(
            asyncio.gather(
                crm_lookup("get_player_transactions", {"type": "deposit", "limit": 20}, timeout_s=remaining()),
                crm_lookup("get_player_latest_deposit_order", {}, timeout_s=remaining()),
            ),
            timeout=remaining(),
        )
    except Exception:  # noqa: BLE001 — a raising or budget-exhausting CRM call must not kill the turn
        log.exception(
            "deposit verification cross-check: crm lookup raised",
            extra={"ticket_id": ticket_id, "session_id": session_id},
        )
        return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

    if not _crm_call_ok(txns_result) or not _crm_call_ok(order_result):
        log.info(
            "deposit verification cross-check: could_not_check",
            extra={
                "ticket_id": ticket_id, "session_id": session_id,
                "reason": "crm_call_failed",
                "transactions_ok": _crm_call_ok(txns_result),
                "order_ok": _crm_call_ok(order_result),
            },
        )
        return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

    txns_data = txns_result.get("data")
    order_data = order_result.get("data")
    transactions = txns_data.get("transactions") if isinstance(txns_data, dict) else None
    order_status = order_data.get("status") if isinstance(order_data, dict) else None

    if (
        not isinstance(transactions, list)
        or order_status not in ("found", "no_recent_deposit", "lookup_unavailable")
    ):
        log.info(
            "deposit verification cross-check: could_not_check",
            extra={
                "ticket_id": ticket_id, "session_id": session_id,
                "reason": "unexpected_crm_response_shape",
            },
        )
        return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

    if order_status == "lookup_unavailable":
        log.info(
            "deposit verification cross-check: could_not_check",
            extra={"ticket_id": ticket_id, "session_id": session_id, "reason": "lookup_unavailable"},
        )
        return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

    tz = _resolve_tenant_zone(tenant)
    transactions_match = any(
        _deposit_transaction_matches(t, shot_amount, shot_date, tz) for t in transactions
    )
    order = order_data.get("order") if order_status == "found" else None
    order_match = _latest_order_matches(order, shot_amount, shot_date, tz)

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "deposit_verification_tool cross_check evaluated",
            ticket_id=ticket_id, session_id=session_id,
            shot_amount=str(shot_amount), shot_date=shot_date.isoformat(),
            transactions_match=transactions_match, order_match=order_match,
            order_status=order_status, tenant_timezone=str(tz),
        )

    if transactions_match and order_match:
        log.info(
            "deposit verification cross-check: matched",
            extra={"ticket_id": ticket_id, "session_id": session_id},
        )
        return None

    log.info(
        "deposit verification cross-check: no_matching_transaction",
        extra={
            "ticket_id": ticket_id, "session_id": session_id,
            "reason": (
                "transactions_mismatch" if not transactions_match else "latest_order_mismatch"
            ),
        },
    )
    message = _NO_MATCHING_TRANSACTION_MESSAGE_TEMPLATE.format(
        amount=_format_amount_for_message(extraction.get("amount")),
        date=extraction.get("date"),
    )
    return {"status": "no_matching_transaction", "message": message}


_SUBMISSION_FAILED_RESULT = {
    "status": "error",
    "message": (
        "Could not submit the verification request right now. Let the customer "
        "know you're having trouble and offer to escalate to a human agent."
    ),
}


async def _vendor_timeout_or_mark_error(
    remaining: Callable[[], float], sessionmaker, request_id: str, *, ticket_id: Optional[str], session_id: str,
) -> Optional[float]:
    """Computed immediately before a vendor POST (never earlier -- see the
    module's time-budget constants): the POST's own timeout, derived from
    whatever's ACTUALLY left of the budget at THIS point, not a value
    computed before the signed_url/ChatSession lookups that precede it.

    Returns None -- after marking the already-created row as 'error' -- when
    less than _MIN_POST_BUDGET_S remains, in which case the caller must not
    attempt the POST at all. Otherwise returns the POST's timeout (remaining
    minus the bookkeeping margin; still capped at _MAX_TIMEOUT_S inside
    _post_json_ticket_vendor/_post_multipart_vendor, exactly as before)."""
    remaining_s = remaining()
    if remaining_s < _MIN_POST_BUDGET_S:
        log.warning(
            "deposit verification: insufficient time budget remaining immediately "
            "before the vendor POST — failing the submission without attempting it",
            extra={
                "ticket_id": ticket_id, "session_id": session_id, "request_id": request_id,
                "remaining_s": remaining_s, "min_post_budget_s": _MIN_POST_BUDGET_S,
            },
        )
        await _mark_error(sessionmaker, request_id)
        return None
    return max(0.0, remaining_s - _VENDOR_TIMEOUT_MARGIN_S)


async def submit_deposit_verification(
    *,
    tenant: TenantContext,
    session_id: str,
    order_id: str,
    sessionmaker,
    media_store: IMediaStorage,
    timeout_s: float,
    ticket_id: str | None = None,
    extractor: ScreenshotExtractor | None = None,
    crm_lookup: CrmLookup | None = None,
) -> dict:
    # Time-budget deadline, computed at entry from the caller's `timeout_s`
    # (this call's share of the turn's cumulative tool budget -- see the
    # module's time-budget constants above). `_remaining()` is threaded
    # through every cross-check step and the final vendor POST below so each
    # one gets a timeout derived from what's ACTUALLY left, not a fixed
    # guess, and the vendor step can be skipped outright (see
    # _MIN_VENDOR_BUDGET_S below) rather than attempted with too little time
    # to plausibly complete.
    _loop = asyncio.get_running_loop()
    _deadline = _loop.time() + timeout_s

    def _remaining() -> float:
        return max(0.0, _deadline - _loop.time())

    dv_config = tenant.settings.deposit_verification
    secret = tenant.secret_optional(dv_config.webhook_secret_env)
    if not dv_config.enabled or not dv_config.webhook_url or media_store is None or not secret:
        # Defensive: the tool should only be registered when this is true
        # (see src/bootstrap.py), but guard here too in case it's ever
        # invoked without that gate. The secret is part of the gate because
        # the inbound verdict callback (src/api/deposit_verification.py)
        # 401s anything it can't HMAC-verify — submitting without one would
        # create a request whose verdict can never be accepted.
        #
        # This should be unreachable (make_chatbot_factory only registers the
        # tool when all four hold), so hitting it at all means the
        # registration-time check and this one have drifted apart -- the
        # discriminating value is WHICH condition failed, not just that one did.
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "deposit_verification_tool submit gate_rejected",
                tool_name="submit_deposit_verification", ticket_id=ticket_id, session_id=session_id,
                dv_enabled=dv_config.enabled, has_webhook_url=bool(dv_config.webhook_url),
                has_media_store=media_store is not None, has_secret=bool(secret),
            )
        return {"status": "error", "message": "Verification is not available for this account."}

    order_id = (order_id or "").strip()
    if not order_id or order_id == REDACTED_PLACEHOLDER:
        # The callback handler cross-checks the vendor's order_id against the
        # value stored on this row with strict equality, so an empty/missing
        # order_id here would guarantee a 400 on the verdict callback and
        # leave the request to time out. Fail fast and tell the LLM instead.
        # REDACTED_PLACEHOLDER is what tool_executor._redact_internal_ids
        # substitutes for a scrubbed internal-id-shaped value; treat it the
        # same as a missing order_id rather than forwarding that literal
        # string to the vendor as a real order id.
        return {
            "status": "missing_order_id",
            "message": (
                "No order id was provided. Call get_player_latest_deposit_order first "
                "and pass its order_id value back here as order_id. If no deposit "
                "order can be found for this customer, do not invent or substitute "
                "an order id — escalate to a human agent instead."
            ),
        }

    async with sessionmaker() as db:
        # The row-selection predicate depends on the contract: `json_ticket_relay`
        # can use EITHER our own `media_url` (download() + signed_url()) OR the
        # CRM's `source_media_url` forwarded as-is (see module docstring and
        # `use_source_url` below), so a row need only have one of the two to be
        # usable. `multipart_verdict` genuinely needs the bytes in our own
        # store, so it must keep requiring `media_url` exactly as before.
        # Without this, a row persisted with `media_url=NULL` (object storage
        # was unavailable at upload time — see src/api/chat.py) but a usable
        # `source_media_url` would be invisible to this query, and the tool
        # would report `no_screenshot` even though a usable URL exists — this
        # is precisely the outage this feature exists to survive. An empty
        # string `source_media_url` does not count as usable (falls back to
        # the signed-URL path instead), matching the truthiness check used
        # everywhere else this column is tested.
        if dv_config.contract == "json_ticket_relay":
            screenshot_predicate = or_(
                ChatMessage.media_url.isnot(None),
                and_(
                    ChatMessage.source_media_url.isnot(None),
                    ChatMessage.source_media_url != "",
                ),
            )
        else:
            screenshot_predicate = ChatMessage.media_url.isnot(None)

        screenshot_row = (await db.execute(
            select(ChatMessage)
            .where(
                ChatMessage.session_id == session_id,
                ChatMessage.type == "image",
                screenshot_predicate,
            )
            .order_by(ChatMessage.id.desc())
            .limit(1)
        )).scalars().first()
        if screenshot_row is None:
            if log.isEnabledFor(logging.DEBUG):
                debug_event(
                    log, "deposit_verification_tool submit no_screenshot",
                    tool_name="submit_deposit_verification", ticket_id=ticket_id,
                    session_id=session_id, order_id=order_id, contract=dv_config.contract,
                )
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
            if log.isEnabledFor(logging.DEBUG):
                debug_event(
                    log, "deposit_verification_tool submit already_pending",
                    tool_name="submit_deposit_verification", ticket_id=ticket_id,
                    session_id=session_id, order_id=order_id,
                    existing_request_id=existing_pending.id,
                    existing_order_id=existing_pending.order_id,
                )
            return {
                "status": "already_pending",
                "message": (
                    "A verification is already in progress for this conversation; do not "
                    "resubmit — tell the customer we're still checking and will update "
                    "them here."
                ),
            }

        # Early budget gate: if there isn't even enough of the budget left to
        # plausibly clear _MIN_VENDOR_BUDGET_S by the time the (much heavier)
        # post-cross-check gate below would run, fail closed right here —
        # before the media-store round trip, the extraction (Gemini) call, or
        # either CRM lookup are ever attempted. The gate below (after the
        # cross-check) still exists and still matters: it catches the budget
        # being exhausted BY the cross-check, which this entry check can't see.
        if _remaining() < _MIN_VENDOR_BUDGET_S:
            log.warning(
                "deposit verification: insufficient time budget remaining at entry "
                "for the cross-check and vendor submission — failing closed as "
                "could_not_check without reading the screenshot or calling the CRM",
                extra={
                    "ticket_id": ticket_id, "session_id": session_id,
                    "remaining_s": _remaining(), "min_vendor_budget_s": _MIN_VENDOR_BUDGET_S,
                },
            )
            return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

        # `json_ticket_relay` rows that carry the CRM's own inbound URL
        # (`source_media_url` — see src/api/chat.py's _persist_turn) never
        # need the screenshot bytes: that URL gets forwarded to the vendor
        # as-is (below), so skip the media-store round trip entirely for
        # this case. This is also the case that no longer depends on our
        # own object storage being reachable — everywhere else, `download()`
        # doubles as both the existence probe (no_screenshot detection) and
        # the byte source `_post_multipart_vendor` needs.
        # Single source of truth for "does this row have a usable source URL",
        # reused below at both the download-skip site and the URL-selection
        # site (Finding: these two used to independently re-derive the same
        # condition and only agreed because the contract string was
        # hard-coded in both places). Gate on truthiness, not `is not None`:
        # an empty string must fall back to the signed-URL path below rather
        # than being treated as a usable URL (and, symmetrically, the query
        # predicate above never selects a row on the strength of an empty
        # `source_media_url` alone).
        use_source_url = (
            dv_config.contract == "json_ticket_relay"
            and bool(screenshot_row.source_media_url)
        )
        data: bytes | None = None
        mime: str | None = None
        if not use_source_url:
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

        # Fix: when this row ALSO has a `media_url` (the cross-check's
        # media-store-first fallback below would use THOSE bytes, reach the
        # extractor, and only fail the https gate on `source_media_url` much
        # later, after the vendor-call section's own `urlsplit(url).scheme`
        # check), run that same https check on `source_media_url` here,
        # before the cross-check -- a submission that's doomed either way
        # shouldn't pay for a screenshot vision call first. Scoped to this
        # shape specifically: without a `media_url` fallback, the cross-check
        # already fails fast on its own https-only fetch of `source_media_url`
        # (`_fetch_source_media_for_cross_check` / `fetch_capped`) before ever
        # reaching the extractor, so there is nothing to pre-empt there.
        if (
            use_source_url
            and screenshot_row.media_url
            and urlsplit(screenshot_row.source_media_url).scheme != "https"
        ):
            log.error(
                "deposit verification: source_media_url is not https — refusing to send "
                "it to an external vendor (checked before the cross-check so a doomed "
                "submission doesn't pay for a screenshot vision call)",
                extra={"ticket_id": ticket_id, "session_id": session_id},
            )
            return _SUBMISSION_FAILED_RESULT

        # Pre-submission cross-check (see _cross_check_screenshot's docstring
        # for the full decision table): runs before the DepositVerificationRequest
        # row is ever created and before any vendor call. Needs the actual
        # screenshot bytes regardless of contract, unlike the vendor-call
        # logic below -- `data`/`mime` above are already populated except on
        # the json_ticket_relay source-URL path (`use_source_url`), which
        # skipped that download on purpose (see its comment) to avoid a
        # dependency on our own object storage.
        #
        # Deliberately still inside this `async with sessionmaker() as db:`
        # block (holding the DB session open across the extraction + CRM
        # network calls below) rather than splitting into two sessions --
        # the simpler shape, at the cost of a connection held a bit longer
        # per submission. Revisit if that becomes a measured pool pressure
        # problem.
        if use_source_url:
            cross_check_data = None
            cross_check_mime = None
            # Prefer our own media store's bytes when this row ALSO has a
            # `media_url` (object storage was reachable at upload time -- see
            # src/api/chat.py's _persist_turn, which can populate both
            # columns on the same row) -- only fall back to fetching the
            # CRM's `source_media_url` over the network when that's
            # unavailable or fails. This keeps the cross-check off an extra
            # external dependency whenever a known-good copy already exists
            # in our own store. Does NOT change which URL is sent to the
            # VENDOR below -- `use_source_url` still forwards
            # `source_media_url` as-is for that; this only affects which
            # bytes the cross-check reads.
            if screenshot_row.media_url:
                try:
                    cross_check_data, cross_check_mime = await asyncio.wait_for(
                        media_store.download(screenshot_row.media_url), timeout=_remaining(),
                    )
                except Exception as e:  # noqa: BLE001 — fall back to source_media_url below
                    log.warning(
                        "deposit verification cross-check: media_store download failed for "
                        "cross-check bytes, falling back to source_media_url",
                        extra={
                            "ticket_id": ticket_id, "session_id": session_id,
                            "exc_type": type(e).__name__,
                        },
                    )
                    log.debug(
                        "deposit verification cross-check: media_store download failure detail",
                        exc_info=True,
                        extra={"ticket_id": ticket_id, "session_id": session_id},
                    )
                    cross_check_data = None
            if cross_check_data is None:
                try:
                    cross_check_data, cross_check_mime = await asyncio.wait_for(
                        _fetch_source_media_for_cross_check(
                            screenshot_row.source_media_url, timeout_s=_remaining(),
                        ),
                        timeout=_remaining(),
                    )
                except Exception as e:  # noqa: BLE001 — an unreachable/invalid source URL, or an
                    # exhausted time budget, must not kill the turn -- this is
                    # an infrastructure failure (ours or the CRM's), not a
                    # judgement about the screenshot's readability, so it
                    # fails closed as could_not_check rather than
                    # screenshot_unreadable. Logged at WARNING by exception
                    # type only (not str(e)), which may embed the signed
                    # screenshot URL -- full detail goes to DEBUG only.
                    log.warning(
                        "deposit verification cross-check: source_media_url fetch failed",
                        extra={
                            "ticket_id": ticket_id, "session_id": session_id,
                            "exc_type": type(e).__name__,
                        },
                    )
                    log.debug(
                        "deposit verification cross-check: source_media_url fetch failure detail",
                        exc_info=True,
                        extra={"ticket_id": ticket_id, "session_id": session_id},
                    )
                    return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}
        else:
            cross_check_data, cross_check_mime = data, mime

        cross_check_result = await _cross_check_screenshot(
            tenant=tenant, sessionmaker=sessionmaker, session_id=session_id,
            data=cross_check_data, mime=cross_check_mime,
            extractor=extractor, crm_lookup=crm_lookup, ticket_id=ticket_id,
            remaining=_remaining,
        )
        if cross_check_result is not None:
            return cross_check_result

        if _remaining() < _MIN_VENDOR_BUDGET_S:
            # Not enough of the tool-call budget is left to plausibly
            # complete the vendor POST -- fail closed as could_not_check
            # WITHOUT creating a request row at all, rather than racing the
            # outer per-tool asyncio.wait_for (src/agents/chatbot.py's
            # _exec_tool) to a row that would be stuck "pending" with no
            # vendor call ever attempted, until the timeout-to-human
            # escalation eventually catches it.
            log.warning(
                "deposit verification: insufficient time budget remaining after cross-check "
                "for the vendor submission — failing closed as could_not_check without "
                "creating a request row",
                extra={
                    "ticket_id": ticket_id, "session_id": session_id,
                    "remaining_s": _remaining(), "min_vendor_budget_s": _MIN_VENDOR_BUDGET_S,
                },
            )
            return {"status": "could_not_check", "message": COULD_NOT_CHECK_MESSAGE}

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

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "deposit_verification_tool submit initiated",
            tool_name="submit_deposit_verification", ticket_id=ticket_id, session_id=session_id,
            request_id=request_id, order_id=order_id, contract=dv_config.contract,
            screenshot_message_id=screenshot_row.id, use_source_url=use_source_url,
            timeout_at=timeout_at.isoformat(),
        )

    # The vendor POST's own timeout is no longer computed once, up front,
    # here -- it's computed immediately before each `_post_*_vendor` call
    # below (via `_vendor_timeout_or_mark_error`), from whatever's ACTUALLY
    # left of the budget at that point. Computing it here (as before) would
    # count time the signed_url/ChatSession lookups below are about to spend
    # as if it were still available to the POST.
    if dv_config.contract == "json_ticket_relay":
        if use_source_url:
            # Forward the CRM's own URL straight through instead of minting a
            # signed one from our media store — this is the row for which the
            # `download()` existence probe above was skipped, since the bytes
            # were never needed on this path, only the URL.
            #
            # Tradeoff being encoded here: this removes our dependency on
            # object storage for this contract (the thing that's actually
            # down on this deploy), but the URL's lifetime and access rules
            # now belong to the CRM instead of us, and — same as the
            # signed-URL path below — this vendor's protocol has no channel
            # to hand over a fresh URL later if it expires or is revoked. An
            # expired/inaccessible CRM link falls through to the existing
            # timeout-to-human escalation (`schedule_verification_timeout`)
            # exactly like an expired signed URL would.
            url = screenshot_row.source_media_url
        else:
            # The raw screenshot bytes fetched above were only an existence
            # probe for this contract (no_screenshot detection) — this vendor
            # wants a fetchable URL, not the bytes, so `data`/`mime` are
            # discarded here.
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
                return _SUBMISSION_FAILED_RESULT
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
            return _SUBMISSION_FAILED_RESULT

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
            return _SUBMISSION_FAILED_RESULT

        async with sessionmaker() as db:
            chat_session = await db.get(ChatSession, session_id)
        mobile = _resolve_mobile(chat_session, dv_config) if chat_session is not None else None

        vendor_timeout_s = await _vendor_timeout_or_mark_error(
            _remaining, sessionmaker, request_id, ticket_id=ticket_id, session_id=session_id,
        )
        if vendor_timeout_s is None:
            return _SUBMISSION_FAILED_RESULT

        ok = await _post_json_ticket_vendor(
            dv_config=dv_config,
            secret=secret,
            order_id=order_id,
            screenshot_url=url,
            mobile=mobile,
            timeout_s=vendor_timeout_s,
            ticket_id=ticket_id,
            session_id=session_id,
        )
    else:
        # `use_source_url` is only ever True for `json_ticket_relay` (see its
        # definition above), so this branch — reached only when the contract
        # is NOT `json_ticket_relay` — always took the `if not use_source_url`
        # download above, meaning `data`/`mime` are guaranteed populated here.
        # Assert rather than trust the duplicated contract check: this makes
        # the non-None-ness structural instead of something that only holds
        # because two branches happen to test the same condition.
        assert data is not None and mime is not None, (
            "unreachable: multipart_verdict never sets use_source_url, so the "
            "download() above always ran for this branch"
        )
        vendor_timeout_s = await _vendor_timeout_or_mark_error(
            _remaining, sessionmaker, request_id, ticket_id=ticket_id, session_id=session_id,
        )
        if vendor_timeout_s is None:
            return _SUBMISSION_FAILED_RESULT

        ok = await _post_multipart_vendor(
            dv_config=dv_config,
            secret=secret,
            request_id=request_id,
            order_id=order_id,
            tenant_id=tenant.id,
            data=data,
            mime=mime,
            timeout_s=vendor_timeout_s,
            ticket_id=ticket_id,
            session_id=session_id,
        )

    if not ok:
        await _mark_error(sessionmaker, request_id)
        return _SUBMISSION_FAILED_RESULT

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

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "deposit_verification_tool vendor_post request",
            tool_name="submit_deposit_verification", contract="multipart_verdict",
            ticket_id=ticket_id, session_id=session_id, request_id=request_id, order_id=order_id,
            webhook_url=dv_config.webhook_url, callback_url=callback_url, mime=mime,
            data_len=len(data),
        )

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
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "deposit_verification_tool vendor_post response",
                tool_name="submit_deposit_verification", contract="multipart_verdict",
                ticket_id=ticket_id, session_id=session_id, request_id=request_id, order_id=order_id,
                status_code=resp.status_code, ok=ok,
            )
        return ok
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

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "deposit_verification_tool vendor_post request",
            tool_name="submit_deposit_verification", contract="json_ticket_relay",
            ticket_id=ticket_id, session_id=session_id, order_id=order_id,
            webhook_url=dv_config.webhook_url, screenshot_url=screenshot_url, mobile=mobile,
        )

    bounded_timeout = min(timeout_s, _MAX_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=bounded_timeout) as client:
            resp = await client.post(dv_config.webhook_url, content=raw, headers=headers)
        ok = 200 <= resp.status_code < 300
        body = None
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
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "deposit_verification_tool vendor_post response",
                tool_name="submit_deposit_verification", contract="json_ticket_relay",
                ticket_id=ticket_id, session_id=session_id, order_id=order_id,
                status_code=resp.status_code, ok=ok, response_body=body,
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
