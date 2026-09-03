"""Sliding-timeout coverage for the json_ticket_relay vendor contract:
POST /api/v1/deposit-verification/reply/{token} re-arms
``DepositVerificationRequest.timeout_at`` on every relayed message (instead
of a fixed deadline from ticket creation), and re-arms the background
timeout check (``schedule_verification_timeout``) so a stale, pre-slide
timer doesn't leave the row stuck without a live watcher.

Also covers the concurrent-double-escalation race in
``_check_and_timeout_verification`` (src/api/chat.py): the sliding window
means MULTIPLE timers can legitimately become due for the same request at
once, so the transition to ``timed_out`` must be a single atomic claim
(``UPDATE ... WHERE status = 'pending'``), not a read-then-write.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat
from src.api import deposit_verification as dv
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import DepositVerificationConfig, TenantSettings
from src.integration.tenant_events import sign_body_hex
from src.models.chat import ChatSession
from src.models.database import Base
from src.models.deposit_verification import DepositVerificationRequest
from src.models.tenant import Tenant

SECRET = "s3cr3t"
TOKEN = "reply-token-abc"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture
async def slide_app() -> AsyncIterator[tuple]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        s.add(ChatSession(id="sess-1", tenant_id="t1", mode="ai", status="active", extra_data={}))
        s.add(DepositVerificationRequest(
            id="req-1", tenant_id="t1", session_id="sess-1", order_id="ORD-9",
            status="pending", timeout_at=_now() + timedelta(minutes=5),
        ))
        await s.commit()

    chat.set_chat_sessionmaker(sm)
    tenant_ctx = register_tenant_for_test(
        TenantSettings(
            id="t1", slug="t1", name="Acme",
            deposit_verification=DepositVerificationConfig(
                enabled=True, webhook_secret_env="DV_SECRET",
                contract="json_ticket_relay", timeout_minutes=5,
            ),
        ),
        secrets={"deposit_verification:reply_token": TOKEN},
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


def _raw(order_id: str, message: str = "checking", type_: str = "agent_reply") -> bytes:
    return json.dumps({"order_id": order_id, "message": message, "type": type_}).encode()


def _post(client, raw: bytes, *, secret=SECRET):
    sig = sign_body_hex(secret, raw)
    return client.post(
        f"/api/v1/deposit-verification/reply/{TOKEN}", content=raw, headers={"X-Signature": sig},
    )


async def _row(sm, request_id="req-1") -> DepositVerificationRequest:
    async with sm() as db:
        return await db.get(DepositVerificationRequest, request_id)


async def test_relay_slides_timeout_forward_to_approximately_now_plus_timeout(slide_app):
    app, sm, _ = slide_app
    client = TestClient(app)
    before = _now()
    resp = _post(client, _raw("ORD-9", "checking"))
    assert resp.status_code == 200

    row = await _row(sm)
    expected = before + timedelta(minutes=5)
    assert abs((row.timeout_at - expected).total_seconds()) < 5


async def test_schedule_verification_timeout_called_once_per_relay(slide_app, monkeypatch):
    app, sm, _ = slide_app
    calls = []

    def _spy(request_id, session_id, timeout_minutes):
        calls.append((request_id, session_id, timeout_minutes))

    monkeypatch.setattr("src.api.chat.schedule_verification_timeout", _spy)
    monkeypatch.setattr("src.api.chat.push_async_message", lambda *a, **k: _noop())

    client = TestClient(app)
    _post(client, _raw("ORD-9", "one"))
    _post(client, _raw("ORD-9", "two"))
    _post(client, _raw("ORD-9", "three"))

    assert len(calls) == 3
    assert all(c == ("req-1", "sess-1", 5) for c in calls)


async def _noop():
    return None


async def test_slid_window_check_is_a_no_op_if_it_fires_early(slide_app):
    app, sm, _ = slide_app
    client = TestClient(app)
    resp = _post(client, _raw("ORD-9", "checking"))
    assert resp.status_code == 200

    # The relay slid timeout_at forward; a check firing "now" (immediately
    # after) must see it as not yet due and no-op, leaving status pending.
    await chat._check_and_timeout_verification("req-1")

    row = await _row(sm)
    assert row.status == "pending"


async def test_genuine_silence_past_slid_deadline_still_escalates(slide_app, monkeypatch):
    app, sm, tenant_ctx = slide_app
    from unittest.mock import AsyncMock
    monkeypatch.setattr("src.api.chat_webhooks.send_bo_webhook", AsyncMock(return_value=True))

    client = TestClient(app)
    resp = _post(client, _raw("ORD-9", "checking"))
    assert resp.status_code == 200

    # Simulate genuine silence past the (slid) deadline by moving timeout_at
    # into the past directly, rather than sleeping minutes_at real time.
    async with sm() as db:
        row = await db.get(DepositVerificationRequest, "req-1")
        row.timeout_at = _now() - timedelta(seconds=1)
        await db.commit()

    await chat._check_and_timeout_verification("req-1")

    row = await _row(sm)
    assert row.status == "timed_out"


async def test_duplicate_relay_does_not_slide_timeout(slide_app):
    app, sm, _ = slide_app
    client = TestClient(app)
    raw = _raw("ORD-9", "checking")

    resp1 = _post(client, raw)
    assert resp1.status_code == 200
    row1 = await _row(sm)
    timeout_after_first = row1.timeout_at

    # Move time forward conceptually by mutating timeout_at won't help here —
    # instead assert the second (duplicate) call leaves timeout_at exactly
    # where the first call set it (no re-slide).
    resp2 = _post(client, raw)
    assert resp2.status_code == 200
    assert resp2.json() == {"status": "duplicate ignored"}

    row2 = await _row(sm)
    assert row2.timeout_at == timeout_after_first


async def test_concurrent_double_escalation_is_prevented_by_atomic_claim(slide_app, monkeypatch):
    """Regression test for the race identified in _check_and_timeout_verification:
    with a sliding timeout window, multiple background timers can become due
    for the same request_id at once. Simulate two concurrent callers racing
    to escalate the same overdue row and assert only one wins (only one BO
    webhook call, row ends up 'timed_out' exactly once)."""
    app, sm, tenant_ctx = slide_app
    from unittest.mock import AsyncMock
    webhook = AsyncMock(return_value=True)
    monkeypatch.setattr("src.api.chat_webhooks.send_bo_webhook", webhook)

    async with sm() as db:
        row = await db.get(DepositVerificationRequest, "req-1")
        row.timeout_at = _now() - timedelta(seconds=1)
        await db.commit()

    # Fire two concurrent checks for the same overdue, still-pending row —
    # both should observe it "due", but only one may win the atomic
    # UPDATE ... WHERE status = 'pending' claim and proceed to escalate.
    await asyncio.gather(
        chat._check_and_timeout_verification("req-1"),
        chat._check_and_timeout_verification("req-1"),
    )

    row = await _row(sm)
    assert row.status == "timed_out"
    webhook.assert_awaited_once()


async def test_slide_between_due_check_and_atomic_claim_prevents_stale_escalation(
    slide_app, monkeypatch,
):
    """Regression for the S1 race: `now` and `row.timeout_at` are read early
    in `_check_and_timeout_verification`, but the function then awaits
    `db.get(ChatSession)` and `tenant_from_id` before attempting the atomic
    claim. If a concurrent relay (POST /reply/{token}) slides `timeout_at`
    forward into the future during exactly that window, the timer must NOT
    still win the claim and escalate right after the vendor replied — the
    atomic UPDATE's WHERE clause must re-check the deadline, not just
    `status == 'pending'`.

    Simulated by monkeypatching `tenant_from_id` (awaited after the initial
    due-check, before the atomic claim) to slide `timeout_at` into the
    future as a side effect, mimicking a relay landing in that exact window.
    """
    app, sm, tenant_ctx = slide_app
    from unittest.mock import AsyncMock
    webhook = AsyncMock(return_value=True)
    monkeypatch.setattr("src.api.chat_webhooks.send_bo_webhook", webhook)

    # Seed the row as already overdue as of the initial due-check.
    async with sm() as db:
        row = await db.get(DepositVerificationRequest, "req-1")
        row.timeout_at = _now() - timedelta(seconds=1)
        await db.commit()

    from src.auth.middleware import tenant_from_id as real_tenant_from_id

    async def _slide_then_resolve(tenant_id):
        # Simulate a concurrent relay sliding the deadline forward while
        # this timer is still awaiting tenant resolution — i.e. AFTER the
        # initial `now < row.timeout_at` due-check but BEFORE the atomic
        # `UPDATE ... WHERE ... timeout_at <= now` claim below it.
        async with sm() as db:
            r = await db.get(DepositVerificationRequest, "req-1")
            r.timeout_at = _now() + timedelta(minutes=5)
            await db.commit()
        return await real_tenant_from_id(tenant_id)

    monkeypatch.setattr("src.auth.middleware.tenant_from_id", _slide_then_resolve)

    await chat._check_and_timeout_verification("req-1")

    row = await _row(sm)
    assert row.status == "pending"
    webhook.assert_not_awaited()
