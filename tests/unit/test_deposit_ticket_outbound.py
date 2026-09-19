"""Deposit dispute screenshot verification: the ``json_ticket_relay`` outbound
contract added to ``src/chatbot/deposit_verification.py`` alongside the
original ``multipart_verdict`` contract (see ``test_deposit_verification_executor.py``
for the multipart-contract tests, which must keep passing unmodified — that's
the proof the multipart extraction was behavior-preserving)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import pytest_asyncio
import respx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import src.api.chat as chat_api
from src.auth.context import TenantContext
from src.chatbot.deposit_verification import submit_deposit_verification
from src.config_tenant import DepositVerificationConfig, TenantSettings
from src.integration.tenant_events import sign_body_hex
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base
from src.models.deposit_verification import DepositVerificationRequest

WEBHOOK_URL = "https://vendor.example.com/ticket"
WEBHOOK_SECRET_ENV = "DV_JSON_WEBHOOK_SECRET"


def _dv_config(**overrides) -> DepositVerificationConfig:
    defaults = dict(
        enabled=True,
        webhook_url=WEBHOOK_URL,
        webhook_secret_env=WEBHOOK_SECRET_ENV,
        timeout_minutes=5,
        contract="json_ticket_relay",
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


class _FakeMediaStore:
    """Like the multipart test file's fake store, but ``signed_url`` is
    actually implemented (that file's fake raises NotImplementedError for it,
    since the multipart contract never calls it)."""

    def __init__(
        self,
        *,
        data: bytes = b"img-bytes",
        mime: str = "image/png",
        raise_missing: bool = False,
        signed_url_value: str = "https://cdn.example.com/shots/abc?sig=xyz",
        signed_url_error: Exception | None = None,
    ):
        self._data = data
        self._mime = mime
        self._raise_missing = raise_missing
        self._signed_url_value = signed_url_value
        self._signed_url_error = signed_url_error
        self.downloads: list[str] = []
        self.signed_url_calls: list[tuple[str, int]] = []

    async def upload(self, data, key, content_type):  # pragma: no cover - unused here
        raise NotImplementedError

    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        self.signed_url_calls.append((key, ttl_seconds))
        if self._signed_url_error is not None:
            raise self._signed_url_error
        return self._signed_url_value

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


async def _add_image_message(
    sessionmaker, session_id: str, media_url: str = "media/key-1",
    *, source_media_url: str | None = None,
) -> int:
    async with sessionmaker() as db:
        msg = ChatMessage(session_id=session_id, role="customer", type="image",
                           content="", media_url=media_url, source_media_url=source_media_url)
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        return msg.id


async def _add_chat_session(
    sessionmaker, session_id: str, *, tenant_id: str = "t1",
    customer_id: str | None = None, extra_data: dict | None = None,
) -> None:
    async with sessionmaker() as db:
        db.add(ChatSession(
            id=session_id, tenant_id=tenant_id, customer_id=customer_id,
            extra_data=extra_data or {},
        ))
        await db.commit()


async def _rows(sessionmaker):
    async with sessionmaker() as db:
        return (await db.execute(select(DepositVerificationRequest))).scalars().all()


# --- Happy path ----------------------------------------------------------


@respx.mock
async def test_happy_path_posts_plain_json_with_bare_hex_signature(sm, monkeypatch) -> None:
    recorder: list = []
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: recorder.append(a))
    session_id = "s-happy"
    await _add_image_message(sm, session_id, media_url="media/key-1")
    await _add_chat_session(sm, session_id, extra_data={"mobile": "9876543210"})
    store = _FakeMediaStore(signed_url_value="https://cdn.example.com/shots/abc?sig=xyz")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200, json={"result": "ok"}))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-1",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "submitted"

    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["content-type"] == "application/json"

    body = json.loads(request.content)
    assert body == {
        "order_id": "ORD-1",
        "screenshot_url": "https://cdn.example.com/shots/abc?sig=xyz",
        "mobile": "9876543210",
    }

    # Signature is recomputed from the exact bytes respx captured on the wire,
    # not by re-serializing the payload dict — this is what proves content=raw
    # (not json=payload) was used.
    expected_sig = sign_body_hex("s3cr3t", request.content)
    assert request.headers["X-Signature"] == expected_sig
    assert not request.headers["X-Signature"].startswith("sha256=")

    assert store.signed_url_calls == [("media/key-1", 3600)]

    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "pending"
    assert recorder == [(rows[0].id, session_id, 5)]


@respx.mock
async def test_happy_path_omits_mobile_key_when_it_cannot_be_resolved(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-happy-nomobile"
    await _add_image_message(sm, session_id, media_url="media/key-2")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200, json={"result": "ok"}))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-2",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    assert out["status"] == "submitted"

    body = json.loads(route.calls.last.request.content)
    assert set(body.keys()) == {"order_id", "screenshot_url"}


@respx.mock
async def test_request_is_not_multipart(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-not-multipart"
    await _add_image_message(sm, session_id, media_url="media/key-3")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-3",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    request = route.calls.last.request
    assert "multipart" not in request.headers["content-type"]


@respx.mock
async def test_signed_url_called_with_configured_ttl(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-ttl"
    await _add_image_message(sm, session_id, media_url="media/key-4")
    store = _FakeMediaStore()
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(_dv_config(screenshot_url_ttl_seconds=7200)), session_id=session_id,
        order_id="ORD-4", sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert store.signed_url_calls == [("media/key-4", 7200)]


# --- Signed-URL security gate ---------------------------------------------


@respx.mock
async def test_relative_signed_url_is_refused_before_any_vendor_call(sm) -> None:
    session_id = "s-relative"
    await _add_image_message(sm, session_id, media_url="media/key-5")
    store = _FakeMediaStore(signed_url_value="/media/key-5")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-5",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "error"
    assert route.call_count == 0
    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "error"


@respx.mock
async def test_signed_url_exception_is_refused_before_any_vendor_call(sm) -> None:
    session_id = "s-signed-url-raises"
    await _add_image_message(sm, session_id, media_url="media/key-6")
    store = _FakeMediaStore(signed_url_error=RuntimeError("boom"))
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-6",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "error"
    assert route.call_count == 0
    rows = await _rows(sm)
    assert rows[0].status == "error"


@respx.mock
async def test_http_webhook_url_is_refused_before_any_vendor_call(sm) -> None:
    """Fix (N11): the outbound POST body carries the signed HTTPS screenshot
    URL — a misconfigured ``http://`` vendor webhook_url would ship that
    signed URL in cleartext over the network. Must be refused the same way
    the signed-URL scheme gate above is, before any HTTP call is attempted."""
    session_id = "s-http-webhook"
    await _add_image_message(sm, session_id, media_url="media/key-http")
    http_url = "http://vendor.example.com/ticket"
    route = respx.post(http_url).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(_dv_config(webhook_url=http_url)), session_id=session_id,
        order_id="ORD-http", sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    assert out["status"] == "error"
    assert route.call_count == 0
    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "error"


# --- Screenshot resolution (must still hold on this new branch) -----------


@respx.mock
async def test_no_screenshot_returns_no_screenshot_and_makes_no_vendor_call(sm) -> None:
    store = _FakeMediaStore()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id="s-none", order_id="ORD-7",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "no_screenshot"
    assert await _rows(sm) == []
    assert route.call_count == 0
    assert store.signed_url_calls == []


# --- Vendor response handling ----------------------------------------------


@respx.mock
async def test_vendor_duplicate_ignored_is_still_treated_as_submitted(sm, monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="src.chatbot.deposit_verification")
    recorder: list = []
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: recorder.append(a))
    session_id = "s-dup"
    await _add_image_message(sm, session_id, media_url="media/key-8")
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200, json={"result": "duplicate ignored"}))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-8",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    assert out["status"] == "submitted"
    rows = await _rows(sm)
    assert rows[0].status == "pending"
    assert recorder != []
    assert any("duplicate ignored" in r.message for r in caplog.records)


@pytest.mark.parametrize("status_code", [401, 400, 500])
@respx.mock
async def test_vendor_error_status_marks_row_error_and_skips_timeout_scheduling(sm, monkeypatch, status_code) -> None:
    recorder: list = []
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: recorder.append(a))
    session_id = f"s-err-{status_code}"
    await _add_image_message(sm, session_id, media_url="media/key-9")
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(status_code))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-9",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    assert out["status"] == "error"
    rows = await _rows(sm)
    assert rows[0].status == "error"
    assert recorder == []


@respx.mock
async def test_vendor_transport_exception_marks_row_error_and_skips_timeout_scheduling(sm, monkeypatch, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="src.chatbot.deposit_verification")
    recorder: list = []
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: recorder.append(a))
    session_id = "s-connerr"
    await _add_image_message(sm, session_id, media_url="media/key-10")
    respx.post(WEBHOOK_URL).mock(side_effect=httpx.ConnectError("boom"))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-10",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    assert out["status"] == "error"
    rows = await _rows(sm)
    assert rows[0].status == "error"
    assert recorder == []
    assert any("vendor POST failed" in r.message for r in caplog.records)


# --- Mobile resolution -------------------------------------------------------


@respx.mock
async def test_mobile_resolved_from_extra_data_mobile_key(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-1"
    await _add_image_message(sm, session_id, media_url="media/key-11")
    await _add_chat_session(sm, session_id, extra_data={"mobile": "9998887770"})
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-11",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert body["mobile"] == "9998887770"


@respx.mock
async def test_mobile_resolved_from_extra_data_phone_key_when_mobile_absent(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-2"
    await _add_image_message(sm, session_id, media_url="media/key-12")
    await _add_chat_session(sm, session_id, extra_data={"phone": "9998887771"})
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-12",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert body["mobile"] == "9998887771"


@respx.mock
async def test_mobile_resolved_from_all_digit_customer_id_when_extra_data_empty(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-3"
    await _add_image_message(sm, session_id, media_url="media/key-13")
    await _add_chat_session(sm, session_id, customer_id="919998887772")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-13",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert body["mobile"] == "919998887772"


@respx.mock
async def test_mobile_key_absent_when_customer_id_is_not_all_digits(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-nondigit"
    await _add_image_message(sm, session_id, media_url="media/key-14")
    await _add_chat_session(sm, session_id, customer_id="cust-abc")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-14",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert "mobile" not in body


@respx.mock
async def test_non_numeric_extra_data_mobile_is_rejected_not_forwarded(sm, monkeypatch) -> None:
    """Fix: the `extra_data` branch used to only `.strip()` its candidate —
    no format/length validation — even though `extra_data` is populated
    straight from client-supplied request metadata (`req.metadata` on
    session creation, src/api/chat.py). Arbitrary text there must never be
    forwarded verbatim to the external vendor."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-nonnumeric"
    await _add_image_message(sm, session_id, media_url="media/key-17")
    await _add_chat_session(sm, session_id, extra_data={"mobile": "not-a-phone-number!"})
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-17",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert "mobile" not in body


@respx.mock
async def test_overly_long_extra_data_mobile_is_rejected_not_forwarded(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-toolong"
    await _add_image_message(sm, session_id, media_url="media/key-18")
    # All-digit but far longer than any real phone number.
    await _add_chat_session(sm, session_id, extra_data={"mobile": "9" * 40})
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-18",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert "mobile" not in body


@respx.mock
async def test_invalid_extra_data_mobile_falls_through_to_valid_customer_id(sm, monkeypatch) -> None:
    """An invalid extra_data candidate must fall through to the next
    resolution source (here, a valid all-digit customer_id), not just fail
    the whole lookup."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-fallthrough"
    await _add_image_message(sm, session_id, media_url="media/key-19")
    await _add_chat_session(
        sm, session_id, customer_id="9998887773", extra_data={"mobile": "garbage"})
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-19",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert body["mobile"] == "9998887773"


@respx.mock
@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("+91-98765-43210", "+919876543210"),
        ("+91 98765 43210", "+919876543210"),
        ("(998) 887-7700", "9988877700"),
    ],
)
async def test_punctuated_extra_data_mobile_is_normalized_then_forwarded(
    sm, monkeypatch, raw_value, expected,
) -> None:
    """Fix: real-world CRM-supplied `metadata: {"mobile": ...}` values often
    carry dashes/spaces/parens (e.g. "+91-98765-43210", "(998) 887-7700") —
    these are entirely plausible phone numbers, but used to be rejected
    outright by `_looks_like_phone_number` for not being digits-only, so
    legitimate numbers were silently dropped. `_resolve_mobile` now runs
    `normalize_phone()` on each candidate before validating, so punctuation/
    spacing is stripped first."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = f"s-mobile-punct-{abs(hash(raw_value))}"
    await _add_image_message(sm, session_id, media_url="media/key-punct")
    await _add_chat_session(sm, session_id, extra_data={"mobile": raw_value})
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id=f"ORD-punct-{expected}",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert body["mobile"] == expected


@respx.mock
async def test_unicode_digit_mobile_is_rejected_not_forwarded(sm, monkeypatch) -> None:
    """Fix (nit C): `str.isdigit()` is Unicode-aware and returns True for
    non-ASCII digit scripts (Arabic-Indic, Devanagari, superscripts, ...),
    which would then pass validation and get forwarded to the external
    vendor verbatim. The validation must use an explicit ASCII-only digit
    check instead."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-unicode-digits"
    await _add_image_message(sm, session_id, media_url="media/key-unicode")
    # Arabic-Indic digits for "9999999999" (10 characters).
    await _add_chat_session(sm, session_id, extra_data={"mobile": "٩" * 10})
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-unicode",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert "mobile" not in body


@respx.mock
async def test_mobile_key_absent_when_no_chat_session_row_exists(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-mobile-nosession"
    await _add_image_message(sm, session_id, media_url="media/key-15")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-15",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
    body = json.loads(route.calls.last.request.content)
    assert "mobile" not in body


# --- Regression guard: unset/default contract keeps using multipart --------


@respx.mock
async def test_default_contract_unset_still_uses_old_multipart_path(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-default-contract"
    await _add_image_message(sm, session_id, media_url="media/key-16")
    store = _FakeMediaStore(data=b"screenshot-bytes", mime="image/png")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    default_config = DepositVerificationConfig(
        enabled=True, webhook_url=WEBHOOK_URL, webhook_secret_env=WEBHOOK_SECRET_ENV,
        timeout_minutes=5,
    )
    assert default_config.contract == "multipart_verdict"

    out = await submit_deposit_verification(
        tenant=_tenant(default_config), session_id=session_id, order_id="ORD-16",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "submitted"

    request = route.calls.last.request
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b"screenshot-bytes" in request.content
    # signed_url must never be called for the multipart_verdict contract.
    assert store.signed_url_calls == []


# --- source_media_url passthrough (json_ticket_relay only) -----------------


@respx.mock
async def test_source_media_url_is_forwarded_and_media_store_is_untouched(sm, monkeypatch) -> None:
    """The whole point of `source_media_url`: when the CRM's own inbound URL
    was persisted on the screenshot row, `json_ticket_relay` forwards it
    directly and never touches the media store — this is what keeps the
    contract working even when our object storage is down (the stage bug
    this change fixes)."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-source-url"
    crm_url = "https://crm.example.com/uploads/shot-1.jpg?sig=abc"
    await _add_image_message(sm, session_id, media_url="media/key-src-1", source_media_url=crm_url)
    store = _FakeMediaStore()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-src-1",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "submitted"

    body = json.loads(route.calls.last.request.content)
    assert body["screenshot_url"] == crm_url

    # Neither the existence-probe download nor signed_url minting happens on
    # this path — the CRM's URL is used as-is.
    assert store.downloads == []
    assert store.signed_url_calls == []

    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "pending"


@respx.mock
async def test_http_source_media_url_is_refused_before_any_vendor_call(sm) -> None:
    """A non-https CRM URL must be refused by the same scheme gate that
    guards a bad signed URL — the gate runs against whichever URL ends up
    selected, source or signed."""
    session_id = "s-source-url-http"
    crm_url = "http://crm.example.com/uploads/shot-2.jpg"
    await _add_image_message(sm, session_id, media_url="media/key-src-2", source_media_url=crm_url)
    store = _FakeMediaStore()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-src-2",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "error"
    assert route.call_count == 0
    assert store.downloads == []
    assert store.signed_url_calls == []

    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "error"


@respx.mock
async def test_null_source_media_url_keeps_using_signed_url_path(sm, monkeypatch) -> None:
    """No `source_media_url` on the row (old rows, or a base64 upload) must
    fall back to exactly today's behaviour: download as the existence probe,
    then mint a signed URL from the media store."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-source-url-null"
    await _add_image_message(sm, session_id, media_url="media/key-src-3", source_media_url=None)
    store = _FakeMediaStore(signed_url_value="https://cdn.example.com/shots/src-3?sig=xyz")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-src-3",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "submitted"

    body = json.loads(route.calls.last.request.content)
    assert body["screenshot_url"] == "https://cdn.example.com/shots/src-3?sig=xyz"
    assert store.downloads == ["media/key-src-3"]
    assert store.signed_url_calls == [("media/key-src-3", 3600)]


@respx.mock
async def test_multipart_contract_ignores_source_media_url_and_still_downloads(sm, monkeypatch) -> None:
    """`source_media_url` is a `json_ticket_relay`-only optimization —
    `multipart_verdict` must keep downloading bytes from the media store and
    posting them multipart, completely ignoring the column, even when it's
    populated on the row."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-source-url-multipart"
    crm_url = "https://crm.example.com/uploads/shot-4.jpg"
    await _add_image_message(sm, session_id, media_url="media/key-src-4", source_media_url=crm_url)
    store = _FakeMediaStore(data=b"screenshot-bytes-4", mime="image/png")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(_dv_config(contract="multipart_verdict")), session_id=session_id,
        order_id="ORD-src-4", sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "submitted"

    request = route.calls.last.request
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b"screenshot-bytes-4" in request.content
    assert crm_url.encode() not in request.content
    assert store.downloads == ["media/key-src-4"]
    assert store.signed_url_calls == []


# --- Row selection when media_url is NULL (object-storage-outage regression) -


@respx.mock
async def test_null_media_url_with_source_url_is_still_selected_and_forwarded(sm, monkeypatch) -> None:
    """Regression test for the bug this whole change exists to fix: a row
    persisted with `media_url=NULL` (object storage was unavailable at upload
    time — see src/api/chat.py) but a usable `source_media_url` must still be
    picked up by the screenshot lookup for `json_ticket_relay`, and that URL
    forwarded without ever touching the media store. Before the fix, the
    `media_url.isnot(None)` predicate filtered this row out entirely and the
    tool returned `no_screenshot`."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-null-media-url"
    crm_url = "https://crm.example.com/uploads/shot-5.jpg?sig=abc"
    await _add_image_message(sm, session_id, media_url=None, source_media_url=crm_url)
    store = _FakeMediaStore()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-src-5",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] != "no_screenshot"
    assert out["status"] == "submitted"

    body = json.loads(route.calls.last.request.content)
    assert body["screenshot_url"] == crm_url
    assert store.downloads == []
    assert store.signed_url_calls == []

    rows = await _rows(sm)
    assert len(rows) == 1
    assert rows[0].status == "pending"


@respx.mock
async def test_multipart_contract_ignores_source_only_row_and_reports_no_screenshot(sm) -> None:
    """`multipart_verdict` needs the bytes in our own store — a row with
    `media_url=None` and only a `source_media_url` must NOT be picked up by
    that contract's screenshot lookup, even though `json_ticket_relay` would
    happily use it."""
    session_id = "s-null-media-url-multipart"
    crm_url = "https://crm.example.com/uploads/shot-6.jpg"
    await _add_image_message(sm, session_id, media_url=None, source_media_url=crm_url)
    store = _FakeMediaStore()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(_dv_config(contract="multipart_verdict")), session_id=session_id,
        order_id="ORD-src-6", sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "no_screenshot"
    assert route.call_count == 0
    assert await _rows(sm) == []


@respx.mock
async def test_empty_string_source_media_url_falls_back_to_signed_url_path(sm, monkeypatch) -> None:
    """An empty-string `source_media_url` (e.g. from a future call site
    missing the `or None` idiom, or a backfill) must not be treated as a
    usable source URL — it must fall back to the signed-URL path, not be
    forwarded as-is (which would fail the https scheme gate) and not cause
    the row to be excluded from selection either, since `media_url` here is
    still valid."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-empty-source-url"
    await _add_image_message(
        sm, session_id, media_url="media/key-src-7", source_media_url="")
    store = _FakeMediaStore(signed_url_value="https://cdn.example.com/shots/src-7?sig=xyz")
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-src-7",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "submitted"

    body = json.loads(route.calls.last.request.content)
    assert body["screenshot_url"] == "https://cdn.example.com/shots/src-7?sig=xyz"
    assert store.downloads == ["media/key-src-7"]
    assert store.signed_url_calls == [("media/key-src-7", 3600)]


@respx.mock
async def test_newer_source_only_row_wins_over_older_media_url_row(sm, monkeypatch) -> None:
    """Ordering: with two image rows in the session for `json_ticket_relay`
    — an older one with `media_url` set and a newer one with only
    `source_media_url` — the newest qualifying row must win (order_by(id
    desc()).limit(1)), even though it's the source-only row."""
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-ordering"
    await _add_image_message(sm, session_id, media_url="media/key-old")
    newer_crm_url = "https://crm.example.com/uploads/shot-newer.jpg"
    await _add_image_message(
        sm, session_id, media_url=None, source_media_url=newer_crm_url)
    store = _FakeMediaStore()
    route = respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-ordering",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out["status"] == "submitted"

    body = json.loads(route.calls.last.request.content)
    assert body["screenshot_url"] == newer_crm_url
    assert store.downloads == []
    assert store.signed_url_calls == []
