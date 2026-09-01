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


def _registry():
    return SimpleNamespace(
        providers=SimpleNamespace(get_llm=lambda t: object(), get_platform_llm=lambda: object()),
        retrievers=SimpleNamespace(get=lambda t: object()),
        session_stores=SimpleNamespace(get=lambda t: None),
        crm_tools=None,
    )


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
        sessionmaker=sm, media_store=store, timeout_s=10.0)
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
        sessionmaker=sm, media_store=store, timeout_s=30.0)
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
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
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
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
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
async def test_order_id_is_stripped_before_persisting(sm, monkeypatch) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    session_id = "s-strip"
    await _add_image_message(sm, session_id)
    respx.post(WEBHOOK_URL).mock(return_value=httpx.Response(200))

    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="  ORD-9  ",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
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
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
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
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=10.0)
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
async def test_executor_refuses_when_not_available(sm, monkeypatch, case) -> None:
    monkeypatch.delenv(WEBHOOK_SECRET_ENV, raising=False)
    store: _FakeMediaStore | None = _FakeMediaStore()

    if case == "disabled":
        tenant = _tenant(_dv_config(enabled=False))
    elif case == "no_webhook_url":
        tenant = _tenant(_dv_config(webhook_url=None))
    elif case == "no_media_store":
        tenant = _tenant()
        store = None
    elif case == "enabled_url_no_secret":
        tenant = _tenant(secret=None)
    else:  # no_secret_env_name
        tenant = _tenant(_dv_config(webhook_secret_env=None))

    out = await submit_deposit_verification(
        tenant=tenant, session_id="s1", order_id="ORD-1",
        sessionmaker=sm, media_store=store, timeout_s=10.0)
    assert out == {"status": "error", "message": "Verification is not available for this account."}
    assert await _rows(sm) == []


# --- Timeout clamping --------------------------------------------------------


def test_max_timeout_constant_is_15s() -> None:
    assert _MAX_TIMEOUT_S == 15.0


@pytest.mark.parametrize("timeout_s, expected", [(60.0, 15.0), (3.0, 3.0)])
async def test_vendor_post_timeout_is_clamped_to_max_timeout(sm, monkeypatch, timeout_s, expected) -> None:
    monkeypatch.setattr(chat_api, "schedule_verification_timeout", lambda *a: None)
    captured: dict = {}

    class _FakeAsyncClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, data=None, files=None, headers=None):
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    session_id = "s-timeout"
    await _add_image_message(sm, session_id)
    out = await submit_deposit_verification(
        tenant=_tenant(), session_id=session_id, order_id="ORD-1",
        sessionmaker=sm, media_store=_FakeMediaStore(), timeout_s=timeout_s)
    assert out["status"] == "submitted"
    assert captured["timeout"] == expected
    assert expected <= _MAX_TIMEOUT_S


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
                            ticket_id=None):
        captured.update(tenant=tenant, session_id=session_id, order_id=order_id,
                         sessionmaker=sessionmaker, media_store=media_store, timeout_s=timeout_s,
                         ticket_id=ticket_id)
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


async def test_registered_executor_defaults_missing_order_id_argument_to_empty_string(sm, monkeypatch) -> None:
    captured: dict = {}

    async def _fake_submit(*, tenant, session_id, order_id, sessionmaker, media_store, timeout_s,
                            ticket_id=None):
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


def test_submit_deposit_verification_is_not_a_builtin_tool() -> None:
    assert SUBMIT_DEPOSIT_VERIFICATION not in BUILTIN_TOOL_NAMES
    assert SUBMIT_DEPOSIT_VERIFICATION not in {t.name for t in BUILTIN_TOOLS}
