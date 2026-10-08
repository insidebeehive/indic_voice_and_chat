"""Deposit dispute screenshot verification: outbound executor
(src/chatbot/deposit_verification.py) and bootstrap-level tool-registration
gating (src/bootstrap.py's make_chatbot_factory)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
import respx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import src.api.chat as chat_api
import src.config
from src.auth.context import TenantContext
from src.bootstrap import make_chatbot_factory
from src.chatbot.deposit_verification import (
    _MAX_TIMEOUT_S,
    _mark_error,
    submit_deposit_verification,
)
from src.chatbot.tools import (
    BUILTIN_TOOL_NAMES,
    BUILTIN_TOOLS,
    SUBMIT_DEPOSIT_VERIFICATION,
)
from src.chatbot.tool_executor import REDACTED_PLACEHOLDER
from src.config_tenant import DepositVerificationConfig, TenantSettings
from src.integration.tenant_events import sign_body
from src.interfaces.llm import ToolCall
from src.models.chat import ChatMessage
from src.models.database import Base
from src.models.deposit_verification import DepositVerificationRequest

WEBHOOK_URL = "https://vendor.example.com/verify"
WEBHOOK_SECRET_ENV = "DV_WEBHOOK_SECRET"


def _dv_config(**overrides) -> DepositVerificationConfig:
    defaults = dict(
        enabled=True,
        webhook_url=WEBHOOK_URL,
        webhook_secret_env=WEBHOOK_SECRET_ENV,
        timeout_minutes=5,
    )
    defaults.update(overrides)
    return DepositVerificationConfig(**defaults)


def _tenant(dv_config: DepositVerificationConfig | None = None, secret: str | None = "s3cr3t") -> TenantContext:
    settings = TenantSettings(
        id="t1", slug="t1", name="T1",
        deposit_verification=dv_config if dv_config is not None else _dv_config(),
    )
    secrets_resolved = {WEBHOOK_SECRET_ENV: secret} if secret is not None else {}
    return TenantContext(settings=settings, secrets_resolved=secrets_resolved)


def _registry(llm_provider: str = "gemini", llm_model: str = "gemini-3.8-flash"):
    # `global_defaults["llm"]` defaults to gemini since that's the real
    # platform default (src/main.py) -- the screenshot cross-check extractor
    # is only wired when this is gemini (src/bootstrap.py's provider guard),
    # so most tests here want it wired. Pass llm_provider="something-else"
    # to exercise the not-wired branch.
    return SimpleNamespace(
        providers=SimpleNamespace(
            get_llm=lambda t: object(), get_platform_llm=lambda: object(),
            global_defaults={"llm": {"provider": llm_provider, "model": llm_model}},
        ),
        retrievers=SimpleNamespace(get=lambda t: object()),
        session_stores=SimpleNamespace(get=lambda t: None),
        crm_tools=None,
    )


# --- Pre-submission cross-check stubs ---------------------------------------
#
# Everything below submits to the vendor must now clear the cross-check
# (src.chatbot.deposit_verification._cross_check_screenshot) first. These
# tests are about the pre-existing screenshot/dedup/vendor-call/timeout
# behavior, not the cross-check itself (see test_deposit_verification_cross_check.py
# for that) -- so every call site that reaches the vendor is given a stub
# extractor + crm_lookup that AGREE with each other by construction, never a
# real Gemini/CRM call.

_MATCH_AMOUNT = 1000.0
_MATCH_DATE = "2026-06-20"
_MATCH_TIMESTAMP = "2026-06-20T10:30:00Z"


def _matching_extractor(amount: float = _MATCH_AMOUNT, date: str = _MATCH_DATE):
    async def extract(data: bytes, mime: str) -> dict:
        return {"readable": True, "amount": amount, "currency": "INR", "date": date}
    return extract


def _matching_crm_lookup(amount: float = _MATCH_AMOUNT, timestamp: str = _MATCH_TIMESTAMP):
    async def lookup(tool_name: str, args: dict, *, timeout_s: float = 10.0) -> dict:
        if tool_name == "get_player_transactions":
            return {
                "status_code": 200,
                "data": {
                    "transactions": [
                        {"id": "txn_1", "type": "deposit", "amount": amount,
                         "status": "success", "timestamp": timestamp},
                    ],
                    "total": 1,
                },
            }
        if tool_name == "get_player_latest_deposit_order":
            return {
                "status_code": 200,
                "data": {
                    "status": "found",
                    "order": {
                        "order_id": "pgs-1", "external_transaction_id": None,
                        "amount": amount, "currency": "INR",
                        "pgs_status": "PGS_SUCCESS", "status_bucket": "pending",
                        "created_at": timestamp,
                    },
                },
            }
        raise AssertionError(f"unexpected crm tool {tool_name!r}")
    return lookup


class _FakeMediaStore:
    def __init__(self, *, data: bytes = b"img-bytes", mime: str = "image/png", raise_missing: bool = False):
        self._data = data
        self._mime = mime
        self._raise_missing = raise_missing
        self.downloads: list[str] = []

    async def upload(self, data, key, content_type):  # pragma: no cover - unused here
        raise NotImplementedError

    async def signed_url(self, key, ttl_seconds):  # pragma: no cover - unused here
        raise NotImplementedError

    async def download(self, key: str):
        self.downloads.append(key)
        if self._raise_missing:
            raise FileNotFoundError(key)
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


async def _add_null_media_image_message(sessionmaker, session_id: str) -> None:
    async with sessionmaker() as db:
        db.add(ChatMessage(session_id=session_id, role="customer", type="image",
                            content="", media_url=None))
        await db.commit()


async def _add_text_message(sessionmaker, session_id: str) -> None:
    async with sessionmaker() as db:
        db.add(ChatMessage(session_id=session_id, role="customer", type="text", content="hi"))
        await db.commit()


async def _add_pending_request(sessionmaker, *, session_id: str, tenant_id: str = "t1",
                                status: str = "pending", order_id: str = "ORD-OLD") -> str:
    row_id = f"dvr_{uuid.uuid4().hex}"
    async with sessionmaker() as db:
        db.add(DepositVerificationRequest(
            id=row_id, tenant_id=tenant_id, session_id=session_id, order_id=order_id,
            status=status, timeout_at=datetime.utcnow() + timedelta(minutes=5)))
        await db.commit()
    return row_id


async def _rows(sessionmaker):
    async with sessionmaker() as db:
        return (await db.execute(select(DepositVerificationRequest))).scalars().all()


# --- Screenshot resolution --------------------------------------------------


@respx.mock
async def test_no_screenshot_returns_no_screenshot_and_writes_no_row(sm) -> None:
    store = _FakeMediaStore()
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id="s1", order_id="ORD-1",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "no_screenshot"
    assert "upload" in out["message"]
    assert await _rows(sm) == []
    assert store.downloads == []


@respx.mock
async def test_picks_most_recent_screenshot_message(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-multi"
    await _add_image_message(sm, session_id, media_url="media/old")
    await _add_text_message(sm, session_id)
    await _add_null_media_image_message(sm, session_id)
    newest_id = await _add_image_message(sm, session_id, media_url="media/newest")

    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
    store = _FakeMediaStore()
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-7",
        sessionmaker=sm, media_store=store, timeout_s=10.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "submitted"
    rows = await _rows(sm)
    assert rows[0].screenshot_message_id == newest_id
    assert store.downloads == ["media/newest"]


@respx.mock
async def test_image_message_without_media_url_is_not_treated_as_screenshot(sm) -> None:
    session_id = "s-nullmedia"
    await _add_null_media_image_message(sm, session_id)
    store = _FakeMediaStore()
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-8",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "no_screenshot"
    assert await _rows(sm) == []
    assert store.downloads == []


@respx.mock
async def test_missing_screenshot_bytes_returns_no_screenshot_and_no_row(sm) -> None:
    session_id = "s-missingbytes"
    await _add_image_message(sm, session_id, media_url="media/missing")
    store = _FakeMediaStore(raise_missing=True)
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-9",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "no_screenshot"
    assert await _rows(sm) == []
    assert store.downloads == ["media/missing"]


# --- Happy path --------------------------------------------------------------


@respx.mock
async def test_happy_path_persists_pending_row_and_posts_signed_multipart(sm, monkeypatch) -> None:
    monkeypatch.setattr(
        src.config, "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(telephony=SimpleNamespace(
            webhook_base_url="https://platform.example.com/api/v1/telephony"))))
    recorder: list = []
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: recorder.append(a))

    session_id = "s-happy"
    screenshot_id = await _add_image_message(sm, session_id, media_url="media/key-1")
    store = _FakeMediaStore(data=b"screenshot-bytes", mime="image/png")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    tenant = _tenant()
    before = datetime.utcnow()
    out = await submit_deposit_verification(
        tenant=tenant, session_id=session_id, order_id="ORD-9",
        sessionmaker=sm, media_store=store, timeout_s=30.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "submitted"

    rows = await _rows(sm)
    assert len(rows) == 1
    row = rows[0]
    assert row.id.startswith("dvr_")
    assert row.tenant_id == "t1"
    assert row.session_id == session_id
    assert row.order_id == "ORD-9"
    assert row.screenshot_message_id == screenshot_id
    assert row.status == "pending"
    assert row.verdict_payload is None
    assert row.resolved_at is None
    expected_timeout = before + timedelta(minutes=tenant.settings.deposit_verification.timeout_minutes)
    assert abs((row.timeout_at - expected_timeout).total_seconds()) < 5

    assert store.downloads == ["media/key-1"]

    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["content-type"].startswith("multipart/form-data")

    # Fixed: platform_webhook_base_url() already includes an /api/v1/telephony
    # path segment, which this call site now discards (via urlsplit) before
    # building the callback URL, so it matches the real registered route
    # (/api/v1/deposit-verification/callback/<id>, mounted directly under
    # api_router's /api/v1 prefix — see src/api/__init__.py) instead of
    # doubling the /api/v1/... path.
    callback_url = (
        "https://platform.example.com"
        f"/api/v1/deposit-verification/callback/{row.id}"
    )
    metadata = {
        "request_id": row.id, "order_id": "ORD-9", "tenant_id": "t1",
        "callback_url": callback_url,
    }
    canonical_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    expected_sig = sign_body("s3cr3t", canonical_bytes)
    assert request.headers["X-Signature"] == expected_sig
    assert canonical_bytes in request.content
    assert b"screenshot-bytes" in request.content
    assert b'name="metadata"' in request.content
    assert b'name="screenshot"' in request.content

    assert recorder == [(row.id, session_id, 5)]


@respx.mock
async def test_no_platform_base_url_falls_back_to_relative_callback_url(sm, monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        src.config, "get_settings",
        lambda: SimpleNamespace(pipeline=SimpleNamespace(telephony=SimpleNamespace(webhook_base_url=None))))
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    caplog.set_level(logging.WARNING, logger="src.chatbot.deposit_verification")

    session_id = "s-relative"
    await _add_image_message(sm, session_id, media_url="media/key-2")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-2",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "submitted"

    rows = await _rows(sm)
    row = rows[0]
    metadata = {
        "request_id": row.id, "order_id": "ORD-2", "tenant_id": "t1",
        "callback_url": f"/api/v1/deposit-verification/callback/{row.id}",
    }
    canonical_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    assert canonical_bytes in route.calls.last.request.content
    assert any("WEBHOOK_BASE_URL is not configured" in r.message for r in caplog.records)


# --- Dedup ---------------------------------------------------------------


@respx.mock
async def test_already_pending_request_is_not_resubmitted(sm) -> None:
    session_id = "s-dedup"
    await _add_image_message(sm, session_id)
    await _add_pending_request(sm, session_id=session_id, status="pending")

    store = _FakeMediaStore()
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-3",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "already_pending"
    assert len(await _rows(sm)) == 1
    assert store.downloads == []


@pytest.mark.parametrize("prior_status", ["verified", "rejected", "timed_out", "error"])
@respx.mock
async def test_previously_resolved_request_does_not_block_a_new_submission(sm, monkeypatch, prior_status) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-resolved"
    await _add_image_message(sm, session_id)
    await _add_pending_request(sm, session_id=session_id, status=prior_status)
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-4",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "submitted"
    assert len(await _rows(sm)) == 2


# --- Empty/missing order_id ------------------------------------------------


@pytest.mark.parametrize("order_id", ["", "   ", None])
@respx.mock
async def test_empty_order_id_is_rejected_before_any_db_write_or_vendor_call(sm, order_id) -> None:
    session_id = "s-empty-oid"
    await _add_image_message(sm, session_id)
    store = _FakeMediaStore()
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id=order_id,
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "missing_order_id"
    assert await _rows(sm) == []
    assert store.downloads == []


@respx.mock
async def test_empty_order_id_rejected_even_when_no_screenshot_exists(sm) -> None:
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id="s-none", order_id="",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    assert out["status"] == "missing_order_id"
    assert await _rows(sm) == []


@respx.mock
async def test_redacted_placeholder_order_id_is_rejected_like_empty(sm) -> None:
    # REDACTED_PLACEHOLDER is what tool_executor._redact_internal_ids
    # substitutes for a scrubbed internal-id-shaped value. If an order_id
    # argument ever comes back as that literal string (e.g. the LLM echoed a
    # redacted field from an earlier tool response), it must be treated
    # exactly like a missing/empty order_id -- never forwarded to the vendor
    # as real.
    session_id = "s-redacted-oid"
    await _add_image_message(sm, session_id)
    store = _FakeMediaStore()
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id=REDACTED_PLACEHOLDER,
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "missing_order_id"
    assert await _rows(sm) == []
    assert store.downloads == []


@respx.mock
async def test_order_id_is_stripped_before_persisting(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-strip"
    await _add_image_message(sm, session_id)
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="  ORD-9  ",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "submitted"
    rows = await _rows(sm)
    assert rows[0].order_id == "ORD-9"


# --- Vendor failure ----------------------------------------------------------


@respx.mock
async def test_vendor_non_2xx_marks_row_error_and_returns_error(sm, monkeypatch) -> None:
    recorder: list = []
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: recorder.append(a))
    session_id = "s-502"
    await _add_image_message(sm, session_id)
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(502))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-5",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "error"
    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "error"
    assert rows[0].resolved_at is None
    assert recorder == []


@respx.mock
async def test_vendor_transport_exception_marks_row_error(sm, monkeypatch, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="src.chatbot.deposit_verification")
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-connerr"
    await _add_image_message(sm, session_id)
    respx.post(WEBHOOK_URL).mock(side_effect=httpx.ConnectError("boom"))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-6",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "error"
    rows = await _rows(sm)
    assert rows[0].status == "error"
    assert rows[0].resolved_at is None
    assert any("vendor POST failed" in r.message for r in caplog.records)


async def test_mark_error_leaves_a_non_pending_row_alone(sm) -> None:
    row_id = await _add_pending_request(sm, session_id="s-mark", status="verified")
    await _mark_error(sm, row_id)
    rows = await _rows(sm)
    assert rows[0].status == "verified"


# --- Config guards -----------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["disabled", "no_webhook_url", "no_media_store", "enabled_url_no_secret", "no_secret_env_name"],
)
@respx.mock
async def test_executor_refuses_when_not_available(sm, monkeypatch, case, caplog) -> None:
    """This defensive gate should be unreachable in production (bootstrap.py
    only registers the tool when all four conditions hold) -- so hitting it
    at all is exactly the situation this repo's audit exists to make
    diagnosable rather than silent. Asserts the DEBUG line fires AND that its
    four boolean fields (the discriminating values, not just "not available")
    actually reflect which condition(s) failed for each case."""
    monkeypatch.delenv(WEBHOOK_SECRET_ENV, raising=False)
    store: _FakeMediaStore | None = _FakeMediaStore()

    if case == "disabled":
        tenant = _tenant(_dv_config(enabled=False))
        expect = {"dv_enabled": False, "has_webhook_url": True, "has_media_store": True, "has_secret": True}
    elif case == "no_webhook_url":
        tenant = _tenant(_dv_config(webhook_url=None))
        expect = {"dv_enabled": True, "has_webhook_url": False, "has_media_store": True, "has_secret": True}
    elif case == "no_media_store":
        tenant = _tenant()
        store = None
        expect = {"dv_enabled": True, "has_webhook_url": True, "has_media_store": False, "has_secret": True}
    elif case == "enabled_url_no_secret":
        tenant = _tenant(secret=None)
        expect = {"dv_enabled": True, "has_webhook_url": True, "has_media_store": True, "has_secret": False}
    else:  # no_secret_env_name
        tenant = _tenant(_dv_config(webhook_secret_env=None))
        expect = {"dv_enabled": True, "has_webhook_url": True, "has_media_store": True, "has_secret": False}

    with caplog.at_level(logging.DEBUG, logger="src.chatbot.deposit_verification"):
        out = await submit_deposit_verification(
            tenant=tenant, session_id="s1", order_id="ORD-1",
            sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out == {"status": "error", "message": "Verification is not available for this account."}
    assert await _rows(sm) == []

    gate_logs = [
        r for r in caplog.records
        if r.message == "deposit_verification_tool submit gate_rejected"
    ]
    assert len(gate_logs) == 1
    assert gate_logs[0].levelname == "DEBUG"
    for field, value in expect.items():
        assert getattr(gate_logs[0], field) == value, f"{case}: {field}"


# --- Timeout clamping --------------------------------------------------------


def test_max_timeout_constant_is_15s() -> None:
    assert _MAX_TIMEOUT_S == 15.0


class _FakeAsyncClient:
    """Captures the `timeout` httpx.AsyncClient was constructed with,
    standing in for the real vendor POST client."""

    def __init__(self, timeout=None):
        _CAPTURED_CLIENT_TIMEOUT["timeout"] = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, data=None, files=None, headers=None):
        return SimpleNamespace(status_code=200)


_CAPTURED_CLIENT_TIMEOUT: dict = {}


async def test_vendor_post_timeout_is_clamped_to_max_timeout(sm, monkeypatch) -> None:
    """A generous budget (60s) still clamps the vendor POST's own timeout to
    _MAX_TIMEOUT_S -- unaffected by the time-budget fix below, since the cap
    is well below anything the remaining-time computation would produce for
    a budget this large."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    _CAPTURED_CLIENT_TIMEOUT.clear()
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    session_id = "s-timeout-cap"
    await _add_image_message(sm, session_id)
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-1",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=60.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "submitted"
    assert _CAPTURED_CLIENT_TIMEOUT["timeout"] == _MAX_TIMEOUT_S


async def test_vendor_post_timeout_uses_remaining_time_minus_margin(sm, monkeypatch) -> None:
    """Fix (time budget): the vendor POST no longer gets the raw `timeout_s`
    the caller passed in -- it gets what's actually LEFT of that budget
    (after the cross-check ran) minus a small margin, still capped at
    _MAX_TIMEOUT_S. With a 10.0s budget and a near-instant stub
    extractor/crm_lookup, ~9.0s should remain for the vendor POST."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    _CAPTURED_CLIENT_TIMEOUT.clear()
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    session_id = "s-timeout-remaining"
    await _add_image_message(sm, session_id)
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-2",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "submitted"
    assert _CAPTURED_CLIENT_TIMEOUT["timeout"] == pytest.approx(9.0, abs=0.5)
    assert _CAPTURED_CLIENT_TIMEOUT["timeout"] < _MAX_TIMEOUT_S


@respx.mock
async def test_insufficient_remaining_budget_before_vendor_step_returns_could_not_check_without_row(sm) -> None:
    """Fix (time budget): a `timeout_s` too small to plausibly leave
    _MIN_VENDOR_BUDGET_S (8.0s) after the cross-check fails closed as
    could_not_check WITHOUT ever creating a request row or calling the
    vendor -- rather than racing the vendor POST against an almost-certain
    outer timeout."""
    session_id = "s-budget-too-small"
    await _add_image_message(sm, session_id)
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-3",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=5.0,
        extractor=_matching_extractor(), crm_lookup=_matching_crm_lookup())
    assert out["status"] == "could_not_check"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_insufficient_budget_at_entry_never_calls_extractor_or_crm_lookup(sm) -> None:
    """Fix (nit): the early entry-point budget gate -- checked right after
    the cheap missing_order_id/screenshot/already_pending checks, before the
    cross-check even starts -- must fail closed as could_not_check WITHOUT
    ever calling the extractor or crm_lookup, not just without creating a
    row. A `timeout_s` of 5.0 is already below `_MIN_VENDOR_BUDGET_S` (8.0),
    so there's no point paying for a Gemini call or a CRM round trip that
    can never lead to a vendor submission anyway."""
    session_id = "s-budget-entry"
    await _add_image_message(sm, session_id)

    extractor_calls: list = []
    crm_calls: list = []

    async def _tracking_extractor(data: bytes, mime: str) -> dict:
        extractor_calls.append((data, mime))
        return {"readable": True, "amount": _MATCH_AMOUNT, "currency": "INR", "date": _MATCH_DATE}

    async def _tracking_crm_lookup(tool_name: str, args: dict, *, timeout_s: float = 10.0) -> dict:
        crm_calls.append(tool_name)
        raise AssertionError("crm_lookup must not be called when the entry budget gate fires")

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-entry",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=5.0,
        extractor=_tracking_extractor, crm_lookup=_tracking_crm_lookup)
    assert out["status"] == "could_not_check"
    assert extractor_calls == []
    assert crm_calls == []
    assert await _rows(sm) == []


# --- Bootstrap-level gating ---------------------------------------------------


async def test_tool_not_registered_when_enabled_with_url_but_no_resolvable_secret(sm, monkeypatch, caplog) -> None:
    monkeypatch.delenv(WEBHOOK_SECRET_ENV, raising=False)
    caplog.set_level(logging.WARNING, logger="src.bootstrap")
    tenant = _tenant(secret=None)
    registry = _registry()
    factory = make_chatbot_factory(registry, sm)
    agent = await factory(tenant, "s1")
    names = {t.name for t in agent._crm_tools}
    assert SUBMIT_DEPOSIT_VERIFICATION not in names
    assert agent._deposit_verification_executor is None
    assert any("NOT being registered" in r.message for r in caplog.records)


async def test_tool_registered_when_enabled_url_and_secret_all_present(sm) -> None:
    tenant = _tenant()
    registry = _registry()
    factory = make_chatbot_factory(registry, sm)
    agent = await factory(tenant, "s1")
    tools_by_name = {t.name: t for t in agent._crm_tools}
    assert SUBMIT_DEPOSIT_VERIFICATION in tools_by_name
    assert agent._deposit_verification_executor is not None
    assert "order_id" in tools_by_name[SUBMIT_DEPOSIT_VERIFICATION].parameters["required"]


@pytest.mark.parametrize("case", ["disabled", "no_webhook_url", "no_sessionmaker"])
async def test_tool_not_registered_when_misconfigured(sm, caplog, case) -> None:
    caplog.set_level(logging.WARNING, logger="src.bootstrap")
    sessionmaker = sm
    if case == "disabled":
        tenant = _tenant(_dv_config(enabled=False))
    elif case == "no_webhook_url":
        tenant = _tenant(_dv_config(webhook_url=None))
    else:
        tenant = _tenant()
        sessionmaker = None

    registry = _registry()
    factory = make_chatbot_factory(registry, sessionmaker)
    agent = await factory(tenant, "s1")
    names = {t.name for t in agent._crm_tools}
    assert SUBMIT_DEPOSIT_VERIFICATION not in names
    assert agent._deposit_verification_executor is None
    assert not any("NOT being registered" in r.message for r in caplog.records)


async def test_registered_executor_passes_bare_session_id_and_current_media_store(sm, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_submit(*, tenant, session_id, order_id, sessionmaker, media_store, timeout_s,
                            ticket_id=None, extractor=None, crm_lookup=None):
        captured.update(tenant=tenant, session_id=session_id, order_id=order_id,
                         sessionmaker=sessionmaker, media_store=media_store, timeout_s=timeout_s,
                         ticket_id=ticket_id, extractor=extractor, crm_lookup=crm_lookup)
        return {"status": "submitted", "message": "ok"}

    import src.chatbot.deposit_verification as dv_module
    monkeypatch.setattr(dv_module, "submit_deposit_verification", _fake_submit)

    fake_store = _FakeMediaStore()
    chat_api.set_media_store(fake_store)
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = _tenant()
        agent = await factory(tenant, "t1:cs_1")
        tc = ToolCall(id="call_1", name=SUBMIT_DEPOSIT_VERIFICATION, arguments={"order_id": "ORD-9"})
        out = await agent._deposit_verification_executor(tc, timeout_s=10.0)
    finally:
        chat_api.set_media_store(None)

    assert out == {"status": "submitted", "message": "ok"}
    assert captured["session_id"] == "cs_1"
    assert captured["order_id"] == "ORD-9"
    assert captured["tenant"] is tenant
    assert captured["sessionmaker"] is sm
    assert captured["media_store"] is fake_store
    assert captured["timeout_s"] == 10.0
    assert callable(captured["extractor"])
    assert callable(captured["crm_lookup"])


async def test_registered_executor_defaults_missing_order_id_argument_to_empty_string(sm, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_submit(*, tenant, session_id, order_id, sessionmaker, media_store, timeout_s,
                            ticket_id=None, extractor=None, crm_lookup=None):
        captured["order_id"] = order_id
        return {"status": "missing_order_id"}

    import src.chatbot.deposit_verification as dv_module
    monkeypatch.setattr(dv_module, "submit_deposit_verification", _fake_submit)

    chat_api.set_media_store(_FakeMediaStore())
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = _tenant()
        agent = await factory(tenant, "t1:cs_2")
        tc = ToolCall(id="call_2", name=SUBMIT_DEPOSIT_VERIFICATION, arguments={})
        await agent._deposit_verification_executor(tc, timeout_s=10.0)
    finally:
        chat_api.set_media_store(None)

    assert captured["order_id"] == ""


# --- Provider guard (src/bootstrap.py) ---------------------------------------


async def test_screenshot_extractor_not_wired_when_platform_llm_is_not_gemini(sm, monkeypatch, caplog) -> None:
    """Fix: the Gemini-shaped screenshot extractor must not be wired when
    the platform default LLM is some other provider -- wiring it anyway
    would send a Gemini-shaped multimodal request to whatever client
    `platform_llm` actually is. `extractor=None` makes the cross-check fail
    closed as could_not_check (visibly, via the WARNING asserted below)."""
    caplog.set_level(logging.WARNING, logger="src.bootstrap")
    captured: dict = {}

    async def _fake_submit(*, tenant, session_id, order_id, sessionmaker, media_store, timeout_s,
                            ticket_id=None, extractor=None, crm_lookup=None):
        captured["extractor"] = extractor
        return {"status": "could_not_check", "message": "x"}

    import src.bootstrap as bootstrap_module
    import src.chatbot.deposit_verification as dv_module
    monkeypatch.setattr(dv_module, "submit_deposit_verification", _fake_submit)
    # Fix (log dedup): the WARNING below is only logged once per (tenant_id,
    # provider) per PROCESS now (see bootstrap.py's `_dv_extractor_not_wired_warned`)
    # -- reset that module-level set so this test's outcome doesn't depend on
    # whether some earlier test already warned for this exact (tenant, provider).
    monkeypatch.setattr(bootstrap_module, "_dv_extractor_not_wired_warned", set())

    chat_api.set_media_store(_FakeMediaStore())
    try:
        registry = _registry(llm_provider="groq")
        factory = make_chatbot_factory(registry, sm)
        tenant = _tenant()
        agent = await factory(tenant, "t1:cs_notgemini")
        tc = ToolCall(id="call_1", name=SUBMIT_DEPOSIT_VERIFICATION, arguments={"order_id": "ORD-1"})
        await agent._deposit_verification_executor(tc, timeout_s=10.0)
    finally:
        chat_api.set_media_store(None)

    assert captured["extractor"] is None
    warnings = [r for r in caplog.records if "screenshot cross-check extractor is NOT wired" in r.message]
    assert len(warnings) == 1
    assert warnings[0].tenant_id == "t1"
    assert warnings[0].provider == "groq"


async def test_screenshot_extractor_not_wired_warning_fires_once_per_tenant_and_provider(
    sm, monkeypatch, caplog,
) -> None:
    """Fix (nit): make_chatbot_factory's inner factory runs on every agent
    build -- i.e. every chat turn for a tenant stuck on a non-gemini platform
    default -- so the WARNING must fire at most once per (tenant_id,
    provider) per process, not once per turn. A different tenant (or a
    different provider for the same tenant) still gets its own warning."""
    caplog.set_level(logging.WARNING, logger="src.bootstrap")

    async def _fake_submit(*, tenant, session_id, order_id, sessionmaker, media_store, timeout_s,
                            ticket_id=None, extractor=None, crm_lookup=None):
        return {"status": "could_not_check", "message": "x"}

    import src.bootstrap as bootstrap_module
    import src.chatbot.deposit_verification as dv_module
    monkeypatch.setattr(dv_module, "submit_deposit_verification", _fake_submit)
    monkeypatch.setattr(bootstrap_module, "_dv_extractor_not_wired_warned", set())

    def _count_warnings():
        return len([
            r for r in caplog.records if "screenshot cross-check extractor is NOT wired" in r.message
        ])

    chat_api.set_media_store(_FakeMediaStore())
    try:
        registry = _registry(llm_provider="groq")
        factory = make_chatbot_factory(registry, sm)
        tenant = _tenant()
        tc = ToolCall(id="call_1", name=SUBMIT_DEPOSIT_VERIFICATION, arguments={"order_id": "ORD-1"})

        agent1 = await factory(tenant, "t1:cs_a")
        await agent1._deposit_verification_executor(tc, timeout_s=10.0)
        assert _count_warnings() == 1

        # Same tenant, same provider, a second turn (a fresh agent build,
        # same as every chat turn produces) -- no additional warning.
        agent2 = await factory(tenant, "t1:cs_b")
        await agent2._deposit_verification_executor(tc, timeout_s=10.0)
        assert _count_warnings() == 1

        # A different tenant gets its own warning.
        other_tenant = _tenant()
        other_tenant.settings.id = "t2"
        agent3 = await factory(other_tenant, "t2:cs_c")
        await agent3._deposit_verification_executor(tc, timeout_s=10.0)
        assert _count_warnings() == 2
    finally:
        chat_api.set_media_store(None)


async def test_screenshot_extractor_wired_when_platform_llm_is_gemini(sm, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_submit(*, tenant, session_id, order_id, sessionmaker, media_store, timeout_s,
                            ticket_id=None, extractor=None, crm_lookup=None):
        captured["extractor"] = extractor
        return {"status": "could_not_check", "message": "x"}

    import src.chatbot.deposit_verification as dv_module
    monkeypatch.setattr(dv_module, "submit_deposit_verification", _fake_submit)

    chat_api.set_media_store(_FakeMediaStore())
    try:
        registry = _registry(llm_provider="gemini")
        factory = make_chatbot_factory(registry, sm)
        tenant = _tenant()
        agent = await factory(tenant, "t1:cs_gemini")
        tc = ToolCall(id="call_1", name=SUBMIT_DEPOSIT_VERIFICATION, arguments={"order_id": "ORD-1"})
        await agent._deposit_verification_executor(tc, timeout_s=10.0)
    finally:
        chat_api.set_media_store(None)

    assert callable(captured["extractor"])


# --- Pre-submission cross-check CRM routing (src/bootstrap.py) --------------


async def test_deposit_verification_crm_lookup_routes_to_crm_tool_executors(sm, monkeypatch) -> None:
    """The cross-check's crm_lookup (wired in make_chatbot_factory as
    `_deposit_verification_crm_lookup`) must route through the SAME
    per-tenant CRM tool executors (`crm_execs`) every other CRM tool call
    uses, with the exact args the cross-check's decision table depends on:
    {"type": "deposit", "limit": 20} for transactions, {} for the latest
    order."""
    import src.bootstrap as bootstrap_module
    import src.chatbot.deposit_verification as dv_module
    import src.chatbot.tool_executor as tool_executor_module
    from src.interfaces.llm import ToolSpec

    calls: list = []

    async def _fake_execute_crm_tool(*, endpoint, method, parameters, auth_type, token, args,
                                      context, x_api_key, extra_headers, session_id, ticket_id,
                                      timeout_s, result_max_chars):
        calls.append({"endpoint": endpoint, "args": args, "timeout_s": timeout_s})
        return {"status_code": 200, "data": {}}

    # `make_chatbot_factory` does `from src.chatbot.tool_executor import
    # execute_crm_tool` INSIDE itself (not at bootstrap's module level), so
    # this must patch the function at its source -- it's re-imported fresh
    # every time `make_chatbot_factory(...)` runs, which happens below.
    monkeypatch.setattr(tool_executor_module, "execute_crm_tool", _fake_execute_crm_tool)

    async def _fake_resolve_crm_tools(tenant, sessionmaker):
        specs = [
            ToolSpec(name="get_player_transactions", description="", parameters={}),
            ToolSpec(name="get_player_latest_deposit_order", description="", parameters={}),
        ]
        execs = {
            "get_player_transactions": {
                "endpoint": "https://crm.example/transactions", "method": "GET",
                "parameters": [], "auth_type": "none", "token": None,
            },
            "get_player_latest_deposit_order": {
                "endpoint": "https://crm.example/latest-order", "method": "GET",
                "parameters": [], "auth_type": "none", "token": None,
            },
        }
        return specs, execs, "tenant"

    monkeypatch.setattr(bootstrap_module, "resolve_crm_tools", _fake_resolve_crm_tools)

    captured_crm_lookup: dict = {}

    async def _fake_submit(*, tenant, session_id, order_id, sessionmaker, media_store, timeout_s,
                            ticket_id=None, extractor=None, crm_lookup=None):
        captured_crm_lookup["crm_lookup"] = crm_lookup
        return {"status": "could_not_check", "message": "x"}

    monkeypatch.setattr(dv_module, "submit_deposit_verification", _fake_submit)

    chat_api.set_media_store(_FakeMediaStore())
    try:
        registry = _registry()
        factory = make_chatbot_factory(registry, sm)
        tenant = _tenant()
        agent = await factory(tenant, "t1:cs_routing")
        tc = ToolCall(id="call_1", name=SUBMIT_DEPOSIT_VERIFICATION, arguments={"order_id": "ORD-1"})
        await agent._deposit_verification_executor(tc, timeout_s=10.0)

        crm_lookup = captured_crm_lookup["crm_lookup"]
        txns_result = await crm_lookup("get_player_transactions", {"type": "deposit", "limit": 20}, timeout_s=5.0)
        order_result = await crm_lookup("get_player_latest_deposit_order", {}, timeout_s=5.0)
    finally:
        chat_api.set_media_store(None)

    assert txns_result == {"status_code": 200, "data": {}}
    assert order_result == {"status_code": 200, "data": {}}
    assert len(calls) == 2
    txns_call = next(c for c in calls if c["endpoint"] == "https://crm.example/transactions")
    order_call = next(c for c in calls if c["endpoint"] == "https://crm.example/latest-order")
    assert txns_call["args"] == {"type": "deposit", "limit": 20}
    assert txns_call["timeout_s"] == 5.0
    assert order_call["args"] == {}
    assert order_call["timeout_s"] == 5.0


def test_submit_deposit_verification_is_not_a_builtin_tool() -> None:
    assert SUBMIT_DEPOSIT_VERIFICATION not in BUILTIN_TOOL_NAMES
    assert SUBMIT_DEPOSIT_VERIFICATION not in {t.name for t in BUILTIN_TOOLS}
