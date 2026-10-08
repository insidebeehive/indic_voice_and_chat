"""Pre-submission cross-check (``src/chatbot/deposit_verification.py``'s
``_cross_check_screenshot``) and its vision-extraction helper
(``src/chatbot/screenshot_extract.py``).

See ``test_deposit_verification_executor.py`` (multipart_verdict) and
``test_deposit_ticket_outbound.py`` (json_ticket_relay) for the existing
screenshot-resolution/dedup/vendor-call coverage — those files stub a
matching extractor + crm_lookup so the cross-check itself isn't what's under
test there. This file is the reverse: the vendor call is incidental, the
cross-check's own decision table is what's under test.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx
import pytest
import pytest_asyncio
import respx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import src.api.chat as chat_api
from src.auth.context import TenantContext
from src.chatbot.deposit_verification import (
    _amounts_match,
    _crm_timestamp_to_tenant_date,
    _normalize_amount,
    submit_deposit_verification,
)
from src.chatbot.screenshot_extract import (
    EXTRACTION_PROMPT,
    make_gemini_screenshot_extractor,
    parse_extraction_response,
)
from src.config_tenant import DepositVerificationConfig, TenantSettings
from src.interfaces.llm import LLMResult
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base
from src.models.deposit_verification import DepositVerificationRequest
from src.models.tenant import ProviderCost
from src.utils import http_fetch

WEBHOOK_URL = "https://vendor.example.com/verify"
WEBHOOK_SECRET_ENV = "DV_WEBHOOK_SECRET"


def _dv_config(**overrides) -> DepositVerificationConfig:
    defaults = dict(
        enabled=True, webhook_url=WEBHOOK_URL, webhook_secret_env=WEBHOOK_SECRET_ENV,
        timeout_minutes=5,
    )
    defaults.update(overrides)
    return DepositVerificationConfig(**defaults)


def _tenant(*, timezone: str = "Asia/Kolkata", dv_config: DepositVerificationConfig | None = None) -> TenantContext:
    settings = TenantSettings(
        id="t1", slug="t1", name="T1", timezone=timezone,
        deposit_verification=dv_config if dv_config is not None else _dv_config(),
    )
    return TenantContext(settings=settings, secrets_resolved={WEBHOOK_SECRET_ENV: "s3cr3t"})


class _FakeMediaStore:
    def __init__(self, *, data: bytes = b"img-bytes", mime: str = "image/png"):
        self._data = data
        self._mime = mime
        self.downloads: list[str] = []

    async def upload(self, data, key, content_type):  # pragma: no cover - unused here
        raise NotImplementedError

    async def signed_url(self, key, ttl_seconds):  # pragma: no cover - unused here
        raise NotImplementedError

    async def download(self, key: str):
        self.downloads.append(key)
        return self._data, self._mime


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    yield sessionmaker
    await engine.dispose()


async def _add_image_message(sessionmaker, session_id: str, media_url: str = "media/key-1") -> int:
    async with sessionmaker() as db:
        msg = ChatMessage(session_id=session_id, role="customer", type="image",
                           content="", media_url=media_url)
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        return msg.id


async def _add_chat_session(sessionmaker, session_id: str, *, tenant_id: str = "t1") -> None:
    async with sessionmaker() as db:
        db.add(ChatSession(id=session_id, tenant_id=tenant_id))
        await db.commit()


async def _rows(sessionmaker):
    async with sessionmaker() as db:
        return (await db.execute(select(DepositVerificationRequest))).scalars().all()


async def _chat_session(sessionmaker, session_id: str) -> ChatSession:
    async with sessionmaker() as db:
        return await db.get(ChatSession, session_id)


# --- Extractor / crm_lookup stubs -------------------------------------------

_AMOUNT = 1000.0
_DATE = "2026-06-20"
_TIMESTAMP = "2026-06-20T10:30:00Z"


def _extractor(*, readable=True, amount=_AMOUNT, date=_DATE, currency="INR", usage=None):
    async def extract(data: bytes, mime: str) -> dict:
        return {
            "readable": readable, "amount": amount, "currency": currency, "date": date,
            "usage": usage or {}, "provider": "gemini", "model": "gemini-3.8-flash",
        }
    return extract


def _raising_extractor(exc: Exception):
    async def extract(data: bytes, mime: str) -> dict:
        raise exc
    return extract


def _crm_lookup(*, transactions=None, order_status="found", order_amount=_AMOUNT,
                order_created_at=_TIMESTAMP):
    if transactions is None:
        transactions = [
            {"id": "txn_1", "type": "deposit", "amount": _AMOUNT, "status": "success",
             "timestamp": _TIMESTAMP},
        ]
    order = None
    if order_status == "found":
        order = {
            "order_id": "pgs-1", "external_transaction_id": None,
            "amount": order_amount, "currency": "INR",
            "pgs_status": "PGS_SUCCESS", "status_bucket": "pending",
            "created_at": order_created_at,
        }

    async def lookup(tool_name: str, args: dict, *, timeout_s: float = 10.0) -> dict:
        if tool_name == "get_player_transactions":
            assert args == {"type": "deposit", "limit": 20}
            return {"status_code": 200, "data": {"transactions": transactions, "total": len(transactions)}}
        if tool_name == "get_player_latest_deposit_order":
            assert args == {}
            return {"status_code": 200, "data": {"status": order_status, "order": order}}
        raise AssertionError(f"unexpected crm tool {tool_name!r}")
    return lookup


def _raising_crm_lookup(exc: Exception):
    async def lookup(tool_name: str, args: dict, *, timeout_s: float = 10.0) -> dict:
        raise exc
    return lookup


def _unknown_tool_crm_lookup():
    async def lookup(tool_name: str, args: dict, *, timeout_s: float = 10.0) -> dict:
        return {"error": f"unknown tool {tool_name}"}
    return lookup


async def _submit(sm, *, session_id="s1", order_id="ORD-1", extractor=None, crm_lookup=None,
                   tenant=None, timeout_s=10.0):
    return await submit_deposit_verification(
        tenant=tenant or _tenant(), session_id=session_id, order_id=order_id,
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=timeout_s,
        extractor=extractor, crm_lookup=crm_lookup,
    )


# --- Match -> submitted ------------------------------------------------------


@respx.mock
async def test_match_on_both_submits_and_calls_vendor_once(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-match"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(sm, session_id=session_id, extractor=_extractor(), crm_lookup=_crm_lookup())
    assert out["status"] == "submitted"
    assert route.call_count == 1
    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "pending"


# --- Mismatches -> no_matching_transaction ----------------------------------


@respx.mock
async def test_transaction_amount_mismatch_returns_no_matching_transaction(sm) -> None:
    session_id = "s-amount-mismatch"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    txns = [{"id": "txn_1", "type": "deposit", "amount": 500.0, "status": "success",
             "timestamp": _TIMESTAMP}]
    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(transactions=txns, order_amount=500.0))
    assert out["status"] == "no_matching_transaction"
    assert "1000" in out["message"]
    assert _DATE in out["message"]
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_transactions_match_but_latest_order_amount_differs(sm) -> None:
    session_id = "s-order-amount-mismatch"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(order_amount=1500.0))
    assert out["status"] == "no_matching_transaction"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_transactions_match_but_latest_order_date_differs(sm) -> None:
    session_id = "s-order-date-mismatch"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(order_created_at="2026-07-01T10:30:00Z"))
    assert out["status"] == "no_matching_transaction"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_latest_order_no_recent_deposit_returns_no_matching_transaction(sm) -> None:
    session_id = "s-no-recent-deposit"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(order_status="no_recent_deposit"))
    assert out["status"] == "no_matching_transaction"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_withdrawal_row_with_matching_amount_is_ignored(sm) -> None:
    """A withdrawal row that happens to match the screenshot's amount/date
    must not count as a matching deposit -- only type == "deposit" rows are
    considered."""
    session_id = "s-withdrawal-only"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    txns = [{"id": "txn_1", "type": "withdrawal", "amount": _AMOUNT, "status": "processing",
             "timestamp": _TIMESTAMP}]
    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(transactions=txns, order_status="no_recent_deposit"))
    assert out["status"] == "no_matching_transaction"
    assert route.call_count == 0
    assert await _rows(sm) == []


# --- Transactions-leg-only failures (latest order DOES match) ---------------
#
# The tests above (no_recent_deposit, withdrawal-only) all have the latest
# order ALSO failing to match, so they can't tell a transactions-leg bug
# apart from an order-leg bug -- if the `and` in
# `transactions_match and order_match` were loosened to an `or`, or the
# `type == "deposit"` filter were removed, those tests would still pass
# (the order leg already fails on its own). These give the order a matching
# amount/date so only the transactions leg is being exercised: removing the
# transactions check (or the type filter, for the withdrawal case) would
# flip the result from no_matching_transaction to submitted.


@respx.mock
async def test_order_matches_but_transaction_amount_differs(sm) -> None:
    session_id = "s-order-matches-txn-amount-mismatch"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    txns = [{"id": "txn_1", "type": "deposit", "amount": 500.0, "status": "success",
             "timestamp": _TIMESTAMP}]
    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(transactions=txns, order_amount=_AMOUNT, order_created_at=_TIMESTAMP))
    assert out["status"] == "no_matching_transaction"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_order_matches_but_transaction_date_is_two_days_off(sm) -> None:
    session_id = "s-order-matches-txn-date-mismatch"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    txns = [{"id": "txn_1", "type": "deposit", "amount": _AMOUNT, "status": "success",
             "timestamp": "2026-06-22T10:30:00Z"}]  # 2 days after _DATE -- outside ±1 day
    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(transactions=txns, order_amount=_AMOUNT, order_created_at=_TIMESTAMP))
    assert out["status"] == "no_matching_transaction"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_order_matches_but_only_a_withdrawal_row_matches_amount_and_date(sm) -> None:
    """Same scenario as test_withdrawal_row_with_matching_amount_is_ignored,
    but with the latest order matching too -- if the `type == "deposit"`
    filter in _deposit_transaction_matches were removed, this would flip to
    "submitted" purely because of a withdrawal row with the right numbers."""
    session_id = "s-order-matches-withdrawal-only"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    txns = [{"id": "txn_1", "type": "withdrawal", "amount": _AMOUNT, "status": "processing",
             "timestamp": _TIMESTAMP}]
    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(transactions=txns, order_amount=_AMOUNT, order_created_at=_TIMESTAMP))
    assert out["status"] == "no_matching_transaction"
    assert route.call_count == 0
    assert await _rows(sm) == []


# --- Date boundary (±1 calendar day, tenant timezone) -----------------------


@pytest.mark.parametrize("screenshot_date,expect_match", [
    ("2026-10-02", True),   # +1 day from Oct 1 IST
    ("2026-09-30", True),   # -1 day from Oct 1 IST
    ("2026-10-01", True),   # exact day
    ("2026-10-03", False),  # +2 days -- outside the window
    # Reverse direction: under UTC, the CRM timestamp's raw date is
    # 2026-09-30, which would put 2026-09-29 just -1 day and matching. In the
    # tenant's actual timezone (IST, +5:30) it's 2026-10-01, so 2026-09-29 is
    # -2 days and must NOT match -- this is the same bug the other direction
    # guards against, approached from the other side of the boundary.
    ("2026-09-29", False),
])
@respx.mock
async def test_date_boundary_in_tenant_timezone(sm, screenshot_date, expect_match) -> None:
    """CRM timestamp 2026-09-30T19:00:00+00:00 is 2026-10-01 00:30 IST --
    the date comparison must use the TENANT's timezone, not UTC's 2026-09-30,
    or this boundary would be off by a day."""
    session_id = f"s-boundary-{screenshot_date}"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
    crm_ts = "2026-09-30T19:00:00+00:00"

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(date=screenshot_date),
        crm_lookup=_crm_lookup(order_created_at=crm_ts,
                                transactions=[{"id": "txn_1", "type": "deposit", "amount": _AMOUNT,
                                               "status": "success", "timestamp": crm_ts}]))
    if expect_match:
        assert out["status"] == "submitted"
        assert route.call_count == 1
    else:
        assert out["status"] == "no_matching_transaction"
        assert route.call_count == 0


# --- Unreadable screenshot ---------------------------------------------------


@respx.mock
async def test_unreadable_extraction_returns_screenshot_unreadable(sm) -> None:
    session_id = "s-unreadable"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(sm, session_id=session_id, extractor=_extractor(readable=False, amount=None, date=None))
    assert out["status"] == "screenshot_unreadable"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_missing_date_returns_screenshot_unreadable_without_crm_calls(sm) -> None:
    session_id = "s-missing-date"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    called: list[str] = []

    async def _lookup_should_not_be_called(tool_name, args, *, timeout_s: float = 10.0):
        called.append(tool_name)
        raise AssertionError("crm_lookup must not be called when the screenshot is unreadable")

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(date=None),
        crm_lookup=_lookup_should_not_be_called)
    assert out["status"] == "screenshot_unreadable"
    assert called == []
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_extractor_raising_fails_closed_as_could_not_check(sm) -> None:
    """Fix: an extractor-call failure is an infrastructure failure, not a
    judgement about the screenshot -- it must fail closed as could_not_check
    (same bucket as a CRM lookup failure), not a generic "error"."""
    session_id = "s-extractor-raises"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(sm, session_id=session_id, extractor=_raising_extractor(RuntimeError("boom")))
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_no_extractor_wired_fails_closed_as_could_not_check(sm) -> None:
    """Fix: no extractor wired (e.g. the platform LLM provider guard in
    src/bootstrap.py declined to wire one) fails closed as could_not_check,
    not a generic "error" -- same bucket as every other infrastructure gap."""
    session_id = "s-no-extractor"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(sm, session_id=session_id, extractor=None, crm_lookup=_crm_lookup())
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


# --- CRM lookup failures -> could_not_check ----------------------------------


@respx.mock
async def test_no_crm_lookup_wired_returns_could_not_check(sm) -> None:
    session_id = "s-no-crm-lookup"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(sm, session_id=session_id, extractor=_extractor(), crm_lookup=None)
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_crm_lookup_raising_returns_could_not_check(sm) -> None:
    session_id = "s-crm-raises"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_raising_crm_lookup(httpx.TimeoutException("timed out")))
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_crm_tool_not_in_tenant_tool_set_returns_could_not_check(sm) -> None:
    """A tenant missing get_player_transactions/get_player_latest_deposit_order
    from its registered tool set -- the crm_lookup wrapper's "unknown tool"
    error (same shape bootstrap.py's crm_executor returns) must fail closed,
    not be treated as a pass-through match."""
    session_id = "s-crm-unknown-tool"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(), crm_lookup=_unknown_tool_crm_lookup())
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_latest_order_lookup_unavailable_returns_could_not_check(sm) -> None:
    session_id = "s-lookup-unavailable"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(),
        crm_lookup=_crm_lookup(order_status="lookup_unavailable"))
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


# --- source_media_url fetch guards -------------------------------------------


@respx.mock
async def test_source_media_url_http_scheme_is_rejected(sm) -> None:
    """Fix: a source_media_url fetch failure (http scheme, in this case) is
    an infrastructure failure -- ours or the CRM's -- not a judgement about
    the screenshot's readability, so it fails closed as could_not_check, not
    screenshot_unreadable. (media_url is None on this row, so the
    media-store-first fallback never applies here.)"""
    from src.config_tenant import DepositVerificationConfig as _Cfg
    session_id = "s-fetch-http"
    async with sm() as db:
        db.add(ChatMessage(session_id=session_id, role="customer", type="image", content="",
                            media_url=None, source_media_url="http://crm.example.com/shot.jpg"))
        await db.commit()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(), crm_lookup=_crm_lookup(),
        tenant=_tenant(dv_config=_Cfg(enabled=True, webhook_url=WEBHOOK_URL,
                                       webhook_secret_env=WEBHOOK_SECRET_ENV, timeout_minutes=5,
                                       contract="json_ticket_relay")))
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_source_media_url_non_image_content_type_is_rejected(sm) -> None:
    """Fix: same bucket as the http-scheme case above -- a non-image
    content-type on the fetched source_media_url is an infrastructure/data
    mismatch, not a readability judgement, so it's could_not_check."""
    from src.config_tenant import DepositVerificationConfig as _Cfg
    session_id = "s-fetch-nonimage"
    crm_url = "https://crm.example.com/not-an-image.pdf"
    async with sm() as db:
        db.add(ChatMessage(session_id=session_id, role="customer", type="image", content="",
                            media_url=None, source_media_url=crm_url))
        await db.commit()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
    respx.get(crm_url).mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}))

    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await _submit(
            sm, session_id=session_id, extractor=_extractor(), crm_lookup=_crm_lookup(),
            tenant=_tenant(dv_config=_Cfg(enabled=True, webhook_url=WEBHOOK_URL,
                                           webhook_secret_env=WEBHOOK_SECRET_ENV, timeout_minutes=5,
                                           contract="json_ticket_relay")))
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


# --- Cost recording -----------------------------------------------------------


@respx.mock
async def test_extraction_usage_is_billed_onto_chat_session_cost(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-cost"
    await _add_chat_session(sm, session_id)
    await _add_image_message(sm, session_id)
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    async with sm() as db:
        db.add(ProviderCost(kind="llm", provider="gemini", model="gemini-3.8-flash",
                             cost_per_1k_input_tokens=0.001, cost_per_1k_output_tokens=0.002))
        await db.commit()

    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 0}
    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(usage=usage), crm_lookup=_crm_lookup())
    assert out["status"] == "submitted"

    chat_session = await _chat_session(sm, session_id)
    expected_cost = round(1000 / 1000.0 * 0.001 + 100 / 1000.0 * 0.002, 6)
    assert chat_session.cost == pytest.approx(expected_cost)
    assert chat_session.input_tokens == 1000
    assert chat_session.output_tokens == 100


@respx.mock
async def test_cost_recording_failure_does_not_block_submission(sm, monkeypatch, caplog) -> None:
    """Billing bookkeeping is best-effort -- a missing ChatSession row (no
    row persisted for this session_id) must not stop the cross-check or the
    vendor call it's reporting the cost of."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-cost-no-session-row"
    await _add_image_message(sm, session_id)
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(usage=usage), crm_lookup=_crm_lookup())
    assert out["status"] == "submitted"


# --- Tool spec description ----------------------------------------------------


def test_tool_spec_description_mentions_cross_check_statuses() -> None:
    from src.chatbot.tools import SUBMIT_DEPOSIT_VERIFICATION_TOOL_SPEC
    desc = SUBMIT_DEPOSIT_VERIFICATION_TOOL_SPEC.description
    assert "no_matching_transaction" in desc
    assert "screenshot_unreadable" in desc
    assert "could_not_check" in desc


# --- Extractor response parsing (src/chatbot/screenshot_extract.py) --------


def test_parse_extraction_response_plain_json() -> None:
    out = parse_extraction_response(
        '{"readable": true, "amount": 1000, "currency": "INR", "date": "2026-06-20"}')
    assert out == {"readable": True, "amount": 1000, "currency": "INR", "date": "2026-06-20"}


def test_parse_extraction_response_strips_code_fences() -> None:
    out = parse_extraction_response(
        '```json\n{"readable": true, "amount": 500.5, "currency": "INR", "date": "2026-01-01"}\n```')
    assert out["readable"] is True
    assert out["amount"] == 500.5
    assert out["date"] == "2026-01-01"


def test_parse_extraction_response_garbage_text_could_not_parse() -> None:
    """Fix: non-JSON text is a parse FAILURE, not a parsed "unreadable"
    judgement -- the model never actually reported anything, so this must be
    distinguishable from a successful readable=false reading (see
    UnparseableExtractionResponse's docstring)."""
    out = parse_extraction_response("I cannot read this image clearly, sorry!")
    assert out is None


def test_parse_extraction_response_empty_string_could_not_parse() -> None:
    assert parse_extraction_response("") is None
    assert parse_extraction_response("   ") is None


def test_parse_extraction_response_null_fields_is_unreadable() -> None:
    out = parse_extraction_response('{"readable": false, "amount": null, "currency": null, "date": null}')
    assert out == {"readable": False, "amount": None, "currency": None, "date": None}


def test_parse_extraction_response_readable_true_but_missing_amount_forces_unreadable() -> None:
    """The model claiming readable=true with a null amount/date must not be
    trusted -- the cross-check can't use a reading it can't trust the shape
    of."""
    out = parse_extraction_response('{"readable": true, "amount": null, "currency": "INR", "date": "2026-06-20"}')
    assert out["readable"] is False


def test_parse_extraction_response_non_json_object_could_not_parse() -> None:
    """A JSON value that parses but isn't an object (a list, here) is also a
    parse failure -- the expected shape was never obtained -- not a parsed
    "unreadable" judgement."""
    out = parse_extraction_response("[1, 2, 3]")
    assert out is None


# --- Amount normalisation (src/chatbot/deposit_verification.py) ------------


def test_comma_formatted_string_amount_matches_plain_number() -> None:
    assert _amounts_match(_normalize_amount("1,000.00"), _normalize_amount(1000))


def test_differing_amounts_do_not_match() -> None:
    assert not _amounts_match(_normalize_amount(1000), _normalize_amount(1000.5))


def test_negative_transaction_amount_matches_positive_screenshot_amount() -> None:
    # Sign is ignored for the comparison itself (the type=="deposit" filter
    # upstream is what actually guards against a debit row being compared at
    # all -- see test_withdrawal_row_with_matching_amount_is_ignored).
    assert _amounts_match(_normalize_amount(-1000), _normalize_amount(1000))


def test_unparseable_amount_normalizes_to_none() -> None:
    assert _normalize_amount("not a number") is None
    assert _normalize_amount(None) is None
    assert _normalize_amount(True) is None


@pytest.mark.parametrize("raw,expected", [
    ("Rs. 500", "500"),
    ("₹1,000.50", "1000.50"),
    ("INR 500", "500"),
    ("999.99", "999.99"),
])
def test_currency_prefixed_string_amounts_normalize(raw, expected) -> None:
    assert _normalize_amount(raw) == _normalize_amount(expected)


@pytest.mark.parametrize("raw", [
    # European-style decimal comma ("1.000,50" meaning one-thousand-point-five)
    # must NOT be silently misread as 1.00050 by naively stripping all commas.
    "1.000,50",
    "abc",
])
def test_ambiguous_or_garbage_string_amounts_normalize_to_none(raw) -> None:
    assert _normalize_amount(raw) is None


def test_float_amount_normalizes_exactly() -> None:
    assert _normalize_amount(999.99) == _normalize_amount("999.99")


# --- Timestamp normalisation (src/chatbot/deposit_verification.py) ---------


_IST = ZoneInfo("Asia/Kolkata")


def test_epoch_seconds_timestamp_parses_as_utc() -> None:
    # 2026-06-20T10:30:00Z
    epoch_s = datetime(2026, 6, 20, 10, 30, tzinfo=ZoneInfo("UTC")).timestamp()
    assert _crm_timestamp_to_tenant_date(epoch_s, ZoneInfo("UTC")) == date(2026, 6, 20)


def test_epoch_milliseconds_timestamp_parses_as_utc() -> None:
    epoch_s = datetime(2026, 6, 20, 10, 30, tzinfo=ZoneInfo("UTC")).timestamp()
    epoch_ms = epoch_s * 1000.0
    assert _crm_timestamp_to_tenant_date(epoch_ms, ZoneInfo("UTC")) == date(2026, 6, 20)


def test_naive_iso_string_is_interpreted_in_tenant_timezone() -> None:
    """A naive CRM timestamp (no offset/Z) is local to the CRM, i.e. the
    TENANT's timezone -- not UTC. "2026-06-20T23:30:00" naive, interpreted as
    IST, is still calendar date 2026-06-20 in IST -- interpreting it as UTC
    and then converting to IST would push it to 2026-06-21."""
    assert _crm_timestamp_to_tenant_date("2026-06-20T23:30:00", _IST) == date(2026, 6, 20)


def test_naive_iso_string_near_midnight_in_tenant_timezone() -> None:
    """A naive timestamp just after local midnight stays on that same local
    date when interpreted directly as tenant-local -- confirms this isn't
    accidentally still going through a UTC interpretation step."""
    assert _crm_timestamp_to_tenant_date("2026-06-21T00:15:00", _IST) == date(2026, 6, 21)


def test_offset_aware_timestamp_still_converts_through_utc_to_tenant_tz() -> None:
    # Unchanged behavior: an explicit-offset/Z timestamp is a real instant,
    # converted to the tenant's timezone for its calendar date -- this is
    # the reverse-direction case also covered in
    # test_date_boundary_in_tenant_timezone's ("2026-09-29", False) row.
    assert _crm_timestamp_to_tenant_date("2026-09-30T19:00:00+00:00", _IST) == date(2026, 10, 1)


# --- make_gemini_screenshot_extractor (src/chatbot/screenshot_extract.py) --


class _FakeLLM:
    def __init__(self, result: LLMResult):
        self._result = result
        self.calls: list = []

    async def generate(self, messages, config):
        self.calls.append((messages, config))
        return self._result


async def test_make_gemini_screenshot_extractor_sends_image_and_approved_prompt() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0}
    fake_llm = _FakeLLM(LLMResult(
        text='{"readable": true, "amount": 500, "currency": "INR", "date": "2026-06-20"}',
        finish_reason="stop", usage=usage,
    ))
    extract = make_gemini_screenshot_extractor(fake_llm, provider_name="gemini", model="gemini-3.8-flash")

    image_bytes = b"\x89PNG\r\n\x1a\nfake-bytes"
    out = await extract(image_bytes, "image/png")

    assert len(fake_llm.calls) == 1
    messages, config = fake_llm.calls[0]
    assert len(messages) == 1
    assert messages[0].role == "user"
    parts = messages[0].content
    text_parts = [p for p in parts if p.type == "text"]
    image_parts = [p for p in parts if p.type == "image"]
    assert len(text_parts) == 1
    assert text_parts[0].text == EXTRACTION_PROMPT
    assert len(image_parts) == 1
    assert image_parts[0].inline_data == {"mime_type": "image/png", "data": image_bytes}

    assert config.temperature == 0.0
    assert config.response_format == "json"

    assert out["readable"] is True
    assert out["amount"] == 500
    assert out["date"] == "2026-06-20"
    assert out["usage"] == usage
    assert out["provider"] == "gemini"
    assert out["model"] == "gemini-3.8-flash"


async def test_make_gemini_screenshot_extractor_raises_on_unparseable_response() -> None:
    from src.chatbot.screenshot_extract import UnparseableExtractionResponse

    fake_llm = _FakeLLM(LLMResult(text="not json at all", finish_reason="stop", usage={}))
    extract = make_gemini_screenshot_extractor(fake_llm, provider_name="gemini", model="gemini-3.8-flash")

    with pytest.raises(UnparseableExtractionResponse):
        await extract(b"bytes", "image/png")


# --- Time budget (src/chatbot/deposit_verification.py) ----------------------


@respx.mock
async def test_slow_extractor_exhausts_time_budget_returns_could_not_check(sm) -> None:
    """A cross-check step that blows the remaining time budget fails closed
    as could_not_check -- never partially submits, never creates a row."""
    async def _slow_extract(data: bytes, mime: str) -> dict:
        await asyncio.sleep(2.0)
        return {"readable": True, "amount": _AMOUNT, "currency": "INR", "date": _DATE}

    session_id = "s-slow-extractor"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_slow_extract, crm_lookup=_crm_lookup(),
        timeout_s=0.1)
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_slow_crm_lookup_exhausts_time_budget_returns_could_not_check(sm) -> None:
    async def _slow_lookup(tool_name: str, args: dict, *, timeout_s: float = 10.0) -> dict:
        await asyncio.sleep(2.0)
        raise AssertionError("unreachable -- should have timed out first")

    session_id = "s-slow-crm"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await _submit(
        sm, session_id=session_id, extractor=_extractor(), crm_lookup=_slow_lookup,
        timeout_s=0.3)
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []
