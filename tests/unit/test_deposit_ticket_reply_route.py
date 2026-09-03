"""Route-level tests for POST /api/v1/deposit-verification/reply/{token} —
the json_ticket_relay vendor contract (multi-message relay, sliding timeout,
token-based tenant identity), independent of the older
/deposit-verification/callback/{request_id} multipart_verdict contract."""

from __future__ import annotations

import hashlib
import json
import logging
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
from src.auth.audit import reset_suppression_state
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import DepositVerificationConfig, TenantSettings
from src.integration.tenant_events import sign_body_hex
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base
from src.models.deposit_verification import DepositVerificationRequest
from src.models.tenant import Tenant

SECRET = "s3cr3t"
TOKEN = "reply-token-abc"


@pytest.fixture(autouse=True)
def _reset_audit_suppression_state():
    """log_denied's per-(reason, client_ip) suppression window is module-level
    state — without resetting it, one test's rejection log gets silently
    suppressed by a prior test's rejections for the same reason (all these
    401s share client_ip == TestClient's fixed address)."""
    reset_suppression_state()
    yield
    reset_suppression_state()


def _future_timeout(minutes: int = 5) -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=minutes)


@pytest.fixture
async def reply_app() -> AsyncIterator[tuple]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        s.add(ChatSession(id="sess-1", tenant_id="t1", mode="ai", status="active", extra_data={}))
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


def _post(client, raw: bytes, *, token=TOKEN, secret=SECRET, header=True, sig_override=None):
    sig = sig_override if sig_override is not None else sign_body_hex(secret, raw)
    headers = {"X-Signature": sig} if header else {}
    return client.post(f"/api/v1/deposit-verification/reply/{token}", content=raw, headers=headers)


async def _row(sm, request_id="req-1") -> DepositVerificationRequest:
    async with sm() as db:
        return await db.get(DepositVerificationRequest, request_id)


async def _messages(sm, session_id="sess-1") -> list[ChatMessage]:
    async with sm() as db:
        result = await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
        )
        return list(result.scalars().all())


async def test_valid_agent_reply_relays_and_leaves_row_pending(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    raw = _raw("ORD-9", "we are checking your deposit")
    resp = _post(client, raw)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    row = await _row(sm)
    assert row.status == "pending"
    assert row.resolved_at is None
    assert row.verdict_payload["replies"][0]["message"] == "we are checking your deposit"
    assert row.verdict_payload["replies"][0]["type"] == "agent_reply"

    messages = await _messages(sm)
    assert len(messages) == 1
    assert messages[0].role == "system"
    assert messages[0].content == "we are checking your deposit"


async def test_multi_message_relay_creates_ordered_messages_and_replies(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)

    for msg in ["checking", "escalated to PG team", "success"]:
        resp = _post(client, _raw("ORD-9", msg))
        assert resp.status_code == 200

    row = await _row(sm)
    assert row.status == "pending"
    replies = row.verdict_payload["replies"]
    assert [r["message"] for r in replies] == ["checking", "escalated to PG team", "success"]

    messages = await _messages(sm)
    assert [m.content for m in messages] == ["checking", "escalated to PG team", "success"]


async def test_auto_holding_message_relayed_status_stays_pending(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    holding = "Have sent to PG team, will take time as they will check thoroughly."
    resp = _post(client, _raw("ORD-9", holding, "auto"))
    assert resp.status_code == 200

    row = await _row(sm)
    assert row.status == "pending"
    messages = await _messages(sm)
    assert messages[0].content == holding


async def test_unrecognized_type_still_relayed_and_200(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    resp = _post(client, _raw("ORD-9", "some future step", "future_type_v2"))
    assert resp.status_code == 200

    row = await _row(sm)
    assert row.verdict_payload["replies"][0]["type"] == "future_type_v2"
    messages = await _messages(sm)
    assert len(messages) == 1


async def test_signature_mismatch_returns_401_no_message_row_untouched(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    resp = _post(client, _raw("ORD-9"), secret="wrong-secret")
    assert resp.status_code == 401

    row = await _row(sm)
    assert row.status == "pending"
    assert row.verdict_payload is None
    assert await _messages(sm) == []


async def test_missing_signature_header_returns_401(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    resp = _post(client, _raw("ORD-9"), header=False)
    assert resp.status_code == 401


async def test_sha256_prefixed_signature_rejected_as_wrong_format(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    raw = _raw("ORD-9")
    from src.integration.tenant_events import sign_body
    prefixed = sign_body(SECRET, raw)  # "sha256=<hex>" — old endpoint's format
    resp = _post(client, raw, sig_override=prefixed)
    assert resp.status_code == 401


@pytest.mark.parametrize("bad_sig", ["garbage", "zz" * 32, "0" * 63])
async def test_malformed_signature_returns_401_not_500(reply_app, bad_sig):
    app, sm, _ = reply_app
    client = TestClient(app)
    resp = _post(client, _raw("ORD-9"), sig_override=bad_sig)
    assert resp.status_code == 401


async def test_non_ascii_signature_percent_encoded_form_returns_401_not_500(reply_app):
    # Mirrors test_deposit_verification_routes.py's equivalent case: httpx/
    # starlette header values must be latin-1-representable, so this sends
    # the percent-encoded form of a non-ASCII value rather than the literal.
    app, sm, _ = reply_app
    client = TestClient(app)
    raw = _raw("ORD-9")
    resp = client.post(
        f"/api/v1/deposit-verification/reply/{TOKEN}",
        content=raw,
        headers={"X-Signature": "%E2%82%AC" * 8},
    )
    assert resp.status_code == 401


async def test_unknown_token_returns_401_not_404(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    raw = _raw("ORD-9")
    sig = sign_body_hex(SECRET, raw)
    resp = client.post(
        "/api/v1/deposit-verification/reply/not-a-real-token",
        content=raw, headers={"X-Signature": sig},
    )
    assert resp.status_code == 401


async def test_disabled_tenant_returns_401(reply_app):
    app, sm, tenant_ctx = reply_app
    tenant_ctx.settings.deposit_verification.enabled = False
    try:
        client = TestClient(app)
        resp = _post(client, _raw("ORD-9"))
        assert resp.status_code == 401
    finally:
        tenant_ctx.settings.deposit_verification.enabled = True


async def test_no_resolvable_secret_returns_401(reply_app, monkeypatch):
    app, sm, tenant_ctx = reply_app
    tenant_ctx.secrets_resolved.pop("DV_SECRET", None)
    monkeypatch.delenv("DV_SECRET", raising=False)
    client = TestClient(app)
    resp = _post(client, _raw("ORD-9"))
    assert resp.status_code == 401


async def test_multipart_verdict_contract_tenant_returns_401(reply_app):
    app, sm, tenant_ctx = reply_app
    tenant_ctx.settings.deposit_verification.contract = "multipart_verdict"
    try:
        client = TestClient(app)
        resp = _post(client, _raw("ORD-9"))
        assert resp.status_code == 401
    finally:
        tenant_ctx.settings.deposit_verification.contract = "json_ticket_relay"


async def test_all_pre_row_lookup_401s_share_an_identical_body(reply_app, caplog):
    """Fix: unknown token, bad signature, disabled tenant, and wrong contract
    used to each 401 with a DIFFERENT body (e.g. 'invalid token' vs 'invalid
    signature'), which let an attacker probing tokens learn which case they
    hit without ever knowing the signing secret. All four must now return
    byte-identical response bodies.

    Also proves the follow-up fix: even though the HTTP response is uniform,
    each of the four cases must log a distinct, informative `reason` — an
    operator debugging "the vendor is getting 401'd" needs to be able to
    tell stale-token vs. feature-disabled vs. contract-misconfigured vs.
    bad-signature apart from the logs alone."""
    app, sm, tenant_ctx = reply_app
    client = TestClient(app)
    raw = _raw("ORD-9")

    def _unknown_token():
        sig = sign_body_hex(SECRET, raw)
        return client.post(
            "/api/v1/deposit-verification/reply/not-a-real-token",
            content=raw, headers={"X-Signature": sig},
        )

    def _bad_signature():
        return _post(client, raw, secret="wrong-secret")

    def _disabled_tenant():
        tenant_ctx.settings.deposit_verification.enabled = False
        try:
            return _post(client, raw)
        finally:
            tenant_ctx.settings.deposit_verification.enabled = True

    def _wrong_contract():
        tenant_ctx.settings.deposit_verification.contract = "multipart_verdict"
        try:
            return _post(client, raw)
        finally:
            tenant_ctx.settings.deposit_verification.contract = "json_ticket_relay"

    # log_denied (src/auth/audit.py) is the logger that actually emits these
    # records now (_reject_unauthorized delegates to it for rate-limited
    # auth-rejection logging), not src.api.deposit_verification directly.
    with caplog.at_level(logging.WARNING, logger="src.auth.audit"):
        responses = [
            _unknown_token(), _bad_signature(), _disabled_tenant(), _wrong_contract(),
        ]
    for resp in responses:
        assert resp.status_code == 401
    bodies = [resp.json() for resp in responses]
    assert all(body == bodies[0] for body in bodies), bodies

    # The HTTP response is uniform (asserted above), but the logs must not
    # be — each case gets its own `reason`, in call order.
    reasons = [r.reason for r in caplog.records if hasattr(r, "reason")]
    assert reasons == ["unknown_token", "bad_signature", "disabled", "wrong_contract"]
    assert len(set(reasons)) == 4  # all four are distinct

    # Log messages themselves are also distinct (not just the reason field).
    messages = [r.getMessage() for r in caplog.records if hasattr(r, "reason")]
    assert len(set(messages)) == 4

    # The raw token must never appear in the logs — only a fingerprint.
    assert "not-a-real-token" not in caplog.text


async def test_unknown_order_id_returns_404_after_valid_signature(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    resp = _post(client, _raw("ORD-DOES-NOT-EXIST"))
    assert resp.status_code == 404
    assert resp.json() == {"status": "unknown order_id"}


async def test_duplicate_body_returns_200_and_is_a_no_op(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    raw = _raw("ORD-9", "checking")

    resp1 = _post(client, raw)
    assert resp1.status_code == 200
    assert resp1.json() == {"status": "ok"}

    row_after_first = await _row(sm)
    timeout_after_first = row_after_first.timeout_at

    resp2 = _post(client, raw)
    assert resp2.status_code == 200
    assert resp2.json() == {"status": "duplicate ignored"}

    messages = await _messages(sm)
    assert len(messages) == 1

    row = await _row(sm)
    assert len(row.verdict_payload["replies"]) == 1
    assert row.timeout_at == timeout_after_first


async def test_replay_of_first_message_detected_after_exceeding_display_cap(reply_app):
    """Fix: replay-dedupe signatures used to live in the SAME 50-entry-capped
    list as the display/debugging history, so once 50 further relays landed
    on the same order, a replay of message #1's exact raw body would no
    longer be recognized as a duplicate and would be re-relayed into the live
    chat. Dedupe signatures now get their own much larger cap
    (_MAX_RELAY_SIG_HISTORY), independent of the 50-entry display cap, so a
    replay of the very first message is still caught after 60 relays."""
    app, sm, _ = reply_app
    client = TestClient(app)

    first_raw = _raw("ORD-9", "message-0")
    resp = _post(client, first_raw)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    # 59 more distinct relays -> 60 total, comfortably past the old 50-cap.
    for i in range(1, 60):
        resp = _post(client, _raw("ORD-9", f"message-{i}"))
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    row = await _row(sm)
    assert len(row.verdict_payload["replies"]) == 50  # display history still capped at 50
    assert len(row.verdict_payload["reply_sigs"]) == 60  # dedupe signatures are not

    messages_before = await _messages(sm)
    assert len(messages_before) == 60

    replay_resp = _post(client, first_raw)
    assert replay_resp.status_code == 200
    assert replay_resp.json() == {"status": "duplicate ignored"}

    messages_after = await _messages(sm)
    assert len(messages_after) == 60  # no new ChatMessage was created for the replay


async def test_replay_detected_when_row_only_has_old_replies_shape_no_reply_sigs(reply_app):
    """Fix: a row written before the reply_sigs split existed only has
    `replies`, each entry carrying its own "sig" key — `reply_sigs` is
    absent entirely (not merely empty), which used to make `reply_sigs = list
    (payload.get("reply_sigs") or [])` start empty and miss a genuine
    replay. `reply_sigs` must now be seeded from `replies` on first read when
    it's absent/empty but `replies` isn't, so a replay of a pre-split message
    is still caught."""
    app, sm, _ = reply_app
    client = TestClient(app)

    old_raw = _raw("ORD-9", "old-message")
    old_sig = hashlib.sha256(old_raw).hexdigest()
    async with sm() as db:
        row = await db.get(DepositVerificationRequest, "req-1")
        row.verdict_payload = {
            "replies": [
                {"sig": old_sig, "type": "agent_reply", "message": "old-message",
                 "received_at": "2020-01-01T00:00:00+00:00"},
            ],
            # Deliberately NO "reply_sigs" key — the pre-split row shape.
        }
        await db.commit()

    resp = _post(client, old_raw)
    assert resp.status_code == 200
    assert resp.json() == {"status": "duplicate ignored"}

    # No new ChatMessage was created for the replay.
    assert await _messages(sm) == []


async def test_newest_row_for_tenant_order_id_pair_receives_the_relay(reply_app):
    app, sm, _ = reply_app
    older = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
    async with sm() as db:
        db.add(ChatSession(id="sess-2", tenant_id="t1", mode="ai", status="active", extra_data={}))
        db.add(DepositVerificationRequest(
            id="req-old", tenant_id="t1", session_id="sess-2", order_id="ORD-9",
            status="pending", timeout_at=_future_timeout(), created_at=older,
        ))
        await db.commit()

    client = TestClient(app)
    resp = _post(client, _raw("ORD-9", "hello"))
    assert resp.status_code == 200

    new_row = await _row(sm, "req-1")
    old_row = await _row(sm, "req-old")
    assert new_row.verdict_payload is not None
    assert old_row.verdict_payload is None

    new_messages = await _messages(sm, "sess-1")
    old_messages = await _messages(sm, "sess-2")
    assert len(new_messages) == 1
    assert old_messages == []


async def test_timed_out_row_still_receives_relay(reply_app):
    app, sm, _ = reply_app
    async with sm() as db:
        row = await db.get(DepositVerificationRequest, "req-1")
        row.status = "timed_out"
        await db.commit()

    client = TestClient(app)
    resp = _post(client, _raw("ORD-9", "late reply"))
    assert resp.status_code == 200

    row = await _row(sm)
    assert row.status == "timed_out"  # status untouched by this route
    messages = await _messages(sm)
    assert len(messages) == 1
    assert messages[0].content == "late reply"


async def test_ended_session_returns_200_no_message_row_untouched(reply_app):
    app, sm, _ = reply_app
    async with sm() as db:
        session = await db.get(ChatSession, "sess-1")
        session.status = "ended"
        session.mode = "closed"
        await db.commit()

    client = TestClient(app)
    resp = _post(client, _raw("ORD-9"))
    assert resp.status_code == 200
    assert resp.json() == {"status": "session closed"}

    row = await _row(sm)
    assert row.verdict_payload is None
    assert await _messages(sm) == []


async def test_empty_message_returns_200_no_message_written(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    resp = _post(client, _raw("ORD-9", ""))
    assert resp.status_code == 200

    assert await _messages(sm) == []
    row = await _row(sm)
    # row is still updated (replies recorded) even though nothing was pushed
    assert row.verdict_payload["replies"][0]["message"] == ""


async def test_long_message_is_truncated(reply_app):
    app, sm, _ = reply_app
    client = TestClient(app)
    long_msg = "a" * 3000
    resp = _post(client, _raw("ORD-9", long_msg))
    assert resp.status_code == 200

    row = await _row(sm)
    assert len(row.verdict_payload["replies"][0]["message"]) == 2000

    messages = await _messages(sm)
    assert len(messages[0].content) == 2000


async def test_control_characters_stripped_but_newline_preserved(reply_app):
    # Control-char stripping happens ONCE, upstream of both destinations —
    # the text actually pushed into the live conversation and the text
    # stored in verdict_payload for display/debugging must not diverge.
    app, sm, _ = reply_app
    client = TestClient(app)
    msg = "line1\nline2\x00\x1bdanger"
    resp = _post(client, _raw("ORD-9", msg))
    assert resp.status_code == 200

    messages = await _messages(sm)
    cleaned = messages[0].content
    assert cleaned == "line1\nline2danger"
    assert "\x00" not in cleaned

    row = await _row(sm)
    assert row.verdict_payload["replies"][0]["message"] == cleaned
    assert "\x1b" not in cleaned
    assert "\n" in cleaned


async def test_row_is_committed_before_the_async_push(reply_app, monkeypatch):
    app, sm, _ = reply_app
    observed = {}

    async def _recorder(session_id, text, *, role="system", frame_type="async_message"):
        async with sm() as db:
            row = await db.get(DepositVerificationRequest, "req-1")
            observed["replies_len"] = len(row.verdict_payload["replies"])
        observed["call"] = (session_id, text, role)
        return 1

    def _fake_schedule(request_id, session_id, timeout_minutes):
        pass

    monkeypatch.setattr("src.api.chat.push_async_message", _recorder)
    monkeypatch.setattr("src.api.chat.schedule_verification_timeout", _fake_schedule)

    client = TestClient(app)
    resp = _post(client, _raw("ORD-9", "committed-before-push"))
    assert resp.status_code == 200

    assert observed["replies_len"] == 1
    assert observed["call"] == ("sess-1", "committed-before-push", "system")


async def test_two_tenants_sharing_an_order_id_are_isolated(reply_app):
    """Fix: DepositVerificationRequest lookups in this route filter by
    ``tenant_id == tenant.id`` (tenant resolved from the URL token) as well
    as ``order_id`` — this proves that filter actually isolates two tenants
    that happen to use the identically-named order_id, rather than relying
    on order_id uniqueness across tenants."""
    app, sm, t1_ctx = reply_app

    t2_secret = "t2-s3cr3t"
    t2_token = "reply-token-t2"
    async with sm() as s:
        s.add(Tenant(id="t2", slug="t2", name="Beta"))
        s.add(ChatSession(id="sess-t2", tenant_id="t2", mode="ai", status="active", extra_data={}))
        s.add(DepositVerificationRequest(
            id="req-t2", tenant_id="t2", session_id="sess-t2", order_id="ORD-9",
            status="pending", timeout_at=_future_timeout(),
        ))
        await s.commit()

    t2_ctx = register_tenant_for_test(
        TenantSettings(
            id="t2", slug="t2", name="Beta",
            deposit_verification=DepositVerificationConfig(
                enabled=True, webhook_secret_env="DV_SECRET_T2",
                contract="json_ticket_relay", timeout_minutes=5,
            ),
        ),
        secrets={"deposit_verification:reply_token": t2_token},
    )
    t2_ctx.secrets_resolved["DV_SECRET_T2"] = t2_secret

    client = TestClient(app)

    # (a) A message signed with t1's secret, sent to t1's token, referencing
    # the shared order_id, must land only in t1's session/row.
    raw_t1 = _raw("ORD-9", "for-t1-only")
    resp1 = _post(client, raw_t1, token=TOKEN, secret=SECRET)
    assert resp1.status_code == 200
    assert resp1.json() == {"status": "ok"}

    t1_row = await _row(sm, "req-1")
    t2_row = await _row(sm, "req-t2")
    assert t1_row.verdict_payload is not None
    assert t1_row.verdict_payload["replies"][0]["message"] == "for-t1-only"
    assert t2_row.verdict_payload is None  # t2's row untouched

    t1_messages = await _messages(sm, "sess-1")
    t2_messages = await _messages(sm, "sess-t2")
    assert [m.content for m in t1_messages] == ["for-t1-only"]
    assert t2_messages == []

    # (b) A message signed with t2's secret, sent to t2's token, referencing
    # the same order_id, must land only in t2's session/row — and must NOT
    # touch t1's row, even though it carries the identical order_id.
    raw_t2 = _raw("ORD-9", "for-t2-only")
    resp2 = _post(client, raw_t2, token=t2_token, secret=t2_secret)
    assert resp2.status_code == 200
    assert resp2.json() == {"status": "ok"}

    t1_row_after = await _row(sm, "req-1")
    t2_row_after = await _row(sm, "req-t2")
    assert len(t1_row_after.verdict_payload["replies"]) == 1  # still just the t1 message
    assert t2_row_after.verdict_payload["replies"][0]["message"] == "for-t2-only"

    t1_messages_after = await _messages(sm, "sess-1")
    t2_messages_after = await _messages(sm, "sess-t2")
    assert [m.content for m in t1_messages_after] == ["for-t1-only"]
    assert [m.content for m in t2_messages_after] == ["for-t2-only"]
