"""Route-level tests for POST /api/v1/deposit-verification/callback/{request_id}."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat
from src.api import deposit_verification as dv
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import DepositVerificationConfig, TenantSettings
from src.integration.tenant_events import sign_body
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base
from src.models.deposit_verification import DepositVerificationRequest
from src.models.tenant import Tenant

SECRET = "s3cr3t"


def _future_timeout() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)


@pytest.fixture
async def dv_app() -> AsyncIterator[tuple]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        s.add(ChatSession(id="sess-1", tenant_id="t1", mode="ai", extra_data={}))
        s.add(DepositVerificationRequest(
            id="req-1", tenant_id="t1", session_id="sess-1", order_id="ORD-9",
            status="pending", timeout_at=_future_timeout(),
        ))
        await s.commit()

    chat.set_chat_sessionmaker(sm)
    tenant_ctx = register_tenant_for_test(
        TenantSettings(
            id="t1", slug="t1", name="Acme",
            deposit_verification=DepositVerificationConfig(
                enabled=True, webhook_secret_env="DV_SECRET",
            ),
        ),
    )
    tenant_ctx.secrets_resolved["DV_SECRET"] = SECRET

    async def _session_override():
        async with sm() as session:
            yield session

    app = FastAPI()
    app.include_router(dv.router, prefix="/api/v1")
    app.dependency_overrides[get_db_session] = _session_override

    yield app, sm, tenant_ctx

    chat.set_chat_sessionmaker(None)
    set_tenant_resolver(None)
    await engine.dispose()


def _post(client, request_id, body, *, secret=SECRET, sign_over=None, header=True):
    raw = json.dumps(body).encode()
    sig = sign_body(secret, sign_over if sign_over is not None else raw)
    headers = {"X-Signature": sig} if header else {}
    return client.post(f"/api/v1/deposit-verification/callback/{request_id}", content=raw, headers=headers)


async def _row(sm, request_id="req-1") -> DepositVerificationRequest:
    async with sm() as db:
        return await db.get(DepositVerificationRequest, request_id)


async def _messages(sm, session_id="sess-1") -> list[ChatMessage]:
    async with sm() as db:
        result = await db.execute(select(ChatMessage).where(ChatMessage.session_id == session_id))
        return list(result.scalars().all())


async def test_valid_signature_verified_verdict_resolves_row_and_pushes_message(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-9", "detail": "matched"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    row = await _row(sm)
    assert row.status == "verified"
    assert row.verdict_payload == {"status": "verified", "order_id": "ORD-9", "detail": "matched"}
    assert row.resolved_at is not None
    assert row.resolved_at.tzinfo is None

    messages = await _messages(sm)
    assert len(messages) == 1
    assert messages[0].role == "system"
    assert messages[0].content == dv._VERDICT_MESSAGES["verified"]

    async with sm() as db:
        session = await db.get(ChatSession, "sess-1")
    assert session.message_count == 1


async def test_valid_signature_rejected_verdict_uses_rejected_message(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "rejected", "order_id": "ORD-9", "detail": "no match"})
    assert resp.status_code == 200

    row = await _row(sm)
    assert row.status == "rejected"

    messages = await _messages(sm)
    assert len(messages) == 1
    assert messages[0].content == dv._VERDICT_MESSAGES["rejected"]


async def test_detail_defaults_to_empty_string(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-9"})
    assert resp.status_code == 200

    row = await _row(sm)
    assert row.verdict_payload["detail"] == ""


async def test_invalid_signature_returns_401_and_leaves_row_pending(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-9"}, secret="wrong-secret")
    assert resp.status_code == 401

    row = await _row(sm)
    assert row.status == "pending"
    assert row.verdict_payload is None
    assert row.resolved_at is None
    assert await _messages(sm) == []


async def test_missing_signature_header_returns_401_and_leaves_row_pending(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-9"}, header=False)
    assert resp.status_code == 401

    row = await _row(sm)
    assert row.status == "pending"


@pytest.mark.parametrize("bad_sig", ["garbage", "sha256=", "sha256=zz"])
async def test_malformed_signature_header_returns_401_not_500(dv_app, bad_sig):
    app, sm, _ = dv_app
    client = TestClient(app)
    raw = json.dumps({"status": "verified", "order_id": "ORD-9"}).encode()
    resp = client.post(
        "/api/v1/deposit-verification/callback/req-1",
        content=raw,
        headers={"X-Signature": bad_sig},
    )
    assert resp.status_code == 401


async def test_malformed_signature_header_percent_encoded_form_returns_401(dv_app):
    # httpx/starlette header values must be latin-1-representable; the literal
    # "café" can't be sent as a raw header value, so this sends its
    # percent-encoded form instead. Note that's pure ASCII bytes on the wire —
    # it does NOT exercise the hmac.compare_digest TypeError path a real
    # non-ASCII byte sequence would hit; this is just a third malformed-format
    # 401 case (alongside test_malformed_signature_header_returns_401_not_500
    # above). Genuine non-ASCII header coverage (verify_signature's own
    # unit-level handling of e.g. "sha256=café") lives in
    # test_tenant_events.py::test_verify_signature_rejects_non_ascii_header_without_raising.
    app, sm, _ = dv_app
    client = TestClient(app)
    raw = json.dumps({"status": "verified", "order_id": "ORD-9"}).encode()
    resp = client.post(
        "/api/v1/deposit-verification/callback/req-1",
        content=raw,
        headers={"X-Signature": "sha256=%E2%82%AC"},
    )
    assert resp.status_code == 401


async def test_tampered_body_returns_401(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    original = json.dumps({"status": "verified", "order_id": "ORD-9", "detail": "matched"}).encode()
    resp = _post(
        client, "req-1",
        {"status": "verified", "order_id": "ORD-9", "detail": "different"},
        sign_over=original,
    )
    assert resp.status_code == 401

    row = await _row(sm)
    assert row.status == "pending"


async def test_unknown_request_id_returns_404(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "does-not-exist", {"status": "verified", "order_id": "ORD-9"})
    assert resp.status_code == 404


async def test_unknown_request_id_returns_404_even_without_a_signature(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "does-not-exist", {"status": "verified", "order_id": "ORD-9"}, header=False)
    assert resp.status_code == 404


async def test_malformed_json_body_returns_422(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    raw = b"not json"
    sig = sign_body(SECRET, raw)
    resp = client.post(
        "/api/v1/deposit-verification/callback/req-1",
        content=raw,
        headers={"X-Signature": sig},
    )
    assert resp.status_code == 422


async def test_unknown_status_value_returns_422(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "maybe", "order_id": "ORD-9"})
    assert resp.status_code == 422


async def test_missing_order_id_field_returns_422(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified"})
    assert resp.status_code == 422


async def test_order_id_mismatch_returns_400_and_leaves_row_pending(dv_app):
    app, sm, _ = dv_app
    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-OTHER"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "order_id does not match this verification request"

    row = await _row(sm)
    assert row.status == "pending"
    assert await _messages(sm) == []


async def test_already_resolved_row_is_idempotent_no_op(dv_app):
    app, sm, _ = dv_app
    resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)

    async with sm() as db:
        db.add(DepositVerificationRequest(
            id="req-2", tenant_id="t1", session_id="sess-1", order_id="ORD-9",
            status="verified", verdict_payload={"status": "verified", "order_id": "ORD-9", "detail": "x"},
            resolved_at=resolved_at, timeout_at=_future_timeout(),
        ))
        await db.commit()

    client = TestClient(app)
    # order_id must match the stored row ("ORD-9") — the mismatch check runs
    # before the idempotency check, so a non-matching order_id here would hit
    # 400 instead of exercising the idempotent no-op path.
    resp = _post(client, "req-2", {"status": "rejected", "order_id": "ORD-9"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "already processed"}

    row = await _row(sm, "req-2")
    assert row.status == "verified"
    assert row.verdict_payload == {"status": "verified", "order_id": "ORD-9", "detail": "x"}
    assert row.resolved_at == resolved_at
    assert await _messages(sm) == []


async def test_timed_out_row_callback_is_idempotent_no_op(dv_app):
    app, sm, _ = dv_app

    async with sm() as db:
        db.add(DepositVerificationRequest(
            id="req-3", tenant_id="t1", session_id="sess-1", order_id="ORD-9",
            status="timed_out", verdict_payload=None, resolved_at=None,
            timeout_at=_future_timeout(),
        ))
        await db.commit()

    client = TestClient(app)
    resp = _post(client, "req-3", {"status": "verified", "order_id": "ORD-9"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "already processed"}

    row = await _row(sm, "req-3")
    assert row.status == "timed_out"
    assert row.verdict_payload is None
    assert await _messages(sm) == []


async def test_error_row_callback_is_idempotent_no_op(dv_app):
    app, sm, _ = dv_app

    async with sm() as db:
        db.add(DepositVerificationRequest(
            id="req-4", tenant_id="t1", session_id="sess-1", order_id="ORD-9",
            status="error", verdict_payload=None, resolved_at=None,
            timeout_at=_future_timeout(),
        ))
        await db.commit()

    client = TestClient(app)
    resp = _post(client, "req-4", {"status": "verified", "order_id": "ORD-9"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "already processed"}

    row = await _row(sm, "req-4")
    assert row.status == "error"
    assert await _messages(sm) == []


async def test_disabled_tenant_returns_403(dv_app):
    app, sm, tenant_ctx = dv_app
    tenant_ctx.settings.deposit_verification.enabled = False
    try:
        client = TestClient(app)
        resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-9"})
        assert resp.status_code == 403
        row = await _row(sm)
        assert row.status == "pending"
    finally:
        tenant_ctx.settings.deposit_verification.enabled = True


async def test_unresolvable_tenant_returns_503(dv_app):
    app, sm, _ = dv_app

    async with sm() as db:
        db.add(DepositVerificationRequest(
            id="req-5", tenant_id="unknown-tenant", session_id="sess-1", order_id="ORD-9",
            status="pending", timeout_at=_future_timeout(),
        ))
        await db.commit()

    client = TestClient(app)
    resp = _post(client, "req-5", {"status": "verified", "order_id": "ORD-9"})
    assert resp.status_code == 503

    row = await _row(sm, "req-5")
    assert row.status == "pending"


async def test_no_resolvable_secret_returns_401_not_200(dv_app, monkeypatch):
    app, sm, tenant_ctx = dv_app
    tenant_ctx.secrets_resolved.pop("DV_SECRET", None)
    monkeypatch.delenv("DV_SECRET", raising=False)

    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-9"}, secret=SECRET)
    assert resp.status_code == 401

    row = await _row(sm)
    assert row.status == "pending"


async def test_row_is_committed_before_the_async_push(dv_app, monkeypatch):
    app, sm, _ = dv_app
    observed = {}

    async def _recorder(session_id, text, *, role="system", frame_type="async_message"):
        async with sm() as db:
            row = await db.get(DepositVerificationRequest, "req-1")
            observed["status"] = row.status
            observed["resolved_at"] = row.resolved_at
        observed["call"] = (session_id, text, role)
        return 1

    monkeypatch.setattr("src.api.chat.push_async_message", _recorder)

    client = TestClient(app)
    resp = _post(client, "req-1", {"status": "verified", "order_id": "ORD-9", "detail": "matched"})
    assert resp.status_code == 200

    assert observed["status"] == "verified"
    assert observed["resolved_at"] is not None
    assert observed["call"] == ("sess-1", dv._VERDICT_MESSAGES["verified"], "system")
