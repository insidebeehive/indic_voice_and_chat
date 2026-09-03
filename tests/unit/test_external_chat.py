"""Tests for src/api/external_chat.py.

Verifies that raw PII (player UUID, message content) never appears in log
records emitted along the external chat paths -- the Chatwoot webhook
(``/integrations/chatwoot/webhook``) and the generic REST session-creation
path (``/integrations/message``) -- and that the fingerprinted ``user_fp``
field is present instead wherever the raw ``user_id`` used to be logged.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_lib
import json
import logging
import re
import time
import types
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat as chat_module
from src.api import external_chat
from src.api.chat import ChatMessageResponse
from src.api.deps import get_db_session
from src.auth import TenantContext, register_tenant_for_test
from src.auth.middleware import InMemoryTenantResolver, set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import Tenant

HEADERS = {"Authorization": "Bearer test-token"}
_HEX12 = re.compile(r"^[0-9a-f]{12}$")


def _capture_create_task(monkeypatch):
    """Rebind the `asyncio` name inside external_chat's own module namespace
    to a stand-in whose create_task() just records + closes the coroutine
    instead of scheduling it -- lets tests assert whether a background
    chatwoot turn was fired, deterministically (no race with the real event
    loop), without touching the real process-wide asyncio.create_task."""
    calls: list = []

    def _fake_create_task(coro, *a, **kw):
        calls.append(coro)
        coro.close()
        return None

    fake_asyncio = types.SimpleNamespace(create_task=_fake_create_task)
    monkeypatch.setattr(external_chat, "asyncio", fake_asyncio)
    return calls


def _chatwoot_signature(secret: str, raw_body: bytes, ts: str) -> str:
    """Build the X-Chatwoot-Signature header exactly as verify_chatwoot expects."""
    payload = f"{ts}.{raw_body.decode('utf-8')}".encode()
    return "sha256=" + hmac_lib.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _make_webhook_app(resolver) -> FastAPI:
    a = FastAPI()
    a.include_router(external_chat.router)
    a.state.tenant_resolver = resolver
    return a


class _ChatwootResolverStub:
    """Minimal stand-in for the DB-backed tenant resolver's
    resolve_by_chatwoot_inbox / resolve_by_chatwoot_webhook_id -- the
    Chatwoot webhook path resolves its tenant from app.state.tenant_resolver,
    not the bearer-token resolver used by the other endpoints."""

    def __init__(self, tctx: TenantContext, webhook_id: str | None = None) -> None:
        self._tctx = tctx
        self._webhook_id = webhook_id

    async def resolve_by_chatwoot_inbox(self, inbox_id: str) -> TenantContext | None:
        return self._tctx

    async def resolve_by_chatwoot_webhook_id(self, webhook_id: str) -> TenantContext | None:
        if self._webhook_id is not None and webhook_id == self._webhook_id:
            return self._tctx
        return None


@pytest.fixture
async def app() -> AsyncIterator[tuple[FastAPI, TenantContext]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="t1", slug="t1", name="Acme"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    tctx = register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        plaintext_tokens=["test-token"],
    )

    a = FastAPI()
    a.include_router(external_chat.router)
    a.dependency_overrides[get_db_session] = _session_override
    a.state.tenant_resolver = _ChatwootResolverStub(tctx, webhook_id="wh-test-1")

    yield a, tctx

    set_tenant_resolver(None)
    await engine.dispose()


def _no_record_contains(records, needle: str) -> bool:
    """True iff `needle` appears nowhere in any record's rendered message or
    any of its extra attribute values (each stringified)."""
    for record in records:
        if needle in record.getMessage():
            return False
        for key, value in record.__dict__.items():
            if key in ("msg", "message", "args"):
                continue  # already covered by getMessage() above
            if needle in str(value):
                return False
    return True


def _find_user_fp(records) -> str | None:
    for record in records:
        fp = getattr(record, "user_fp", None)
        if fp is not None:
            return fp
    return None


async def _fake_process_message(tenant, session_id, text) -> ChatMessageResponse:
    return ChatMessageResponse(
        session_id=session_id,
        response_text="mocked response text",
        language="en",
        confidence="high",
        sources_used=[],
        action="none",
        suggested_followups=[],
    )


def test_chatwoot_webhook_does_not_log_raw_pii(app, monkeypatch, caplog) -> None:
    """Legacy tokenless route only still processes in log_only mode (in the
    default enforce mode it now hard-rejects with 401 -- see H3 remediation).
    Set explicitly here, never in a shared fixture, so this test's original
    PII-redaction intent still exercises real processing."""
    monkeypatch.setenv("VOX_WEBHOOK_SIGNATURE_MODE", "log_only")
    a, _tctx = app
    monkeypatch.setattr(chat_module, "process_message", _fake_process_message)
    # Scope DEBUG capture to our own logger, not the root logger -- at root,
    # DEBUG also pulls in aiosqlite's SQL-echo logging, which legitimately
    # includes bind parameters (i.e. the raw user_id/content going into the
    # INSERT) and would make this assertion fail for a reason unrelated to
    # our own log calls.
    caplog.set_level(logging.DEBUG, logger="src.api.external_chat")

    raw_uuid = "pu-12345-secret-uuid"
    raw_content = "my secret message content xyz"
    payload = {
        "event": "message_created",
        "message_type": 0,
        "content": raw_content,
        "conversation": {"id": 456, "inbox_id": 999},
        "sender": {"name": "Ravi", "identifier": raw_uuid, "type": "contact"},
    }

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}

    # The "chatwoot message received" log happens synchronously in the
    # request handler, before asyncio.create_task hands the turn off to the
    # background task -- so it's already captured here.
    assert _no_record_contains(caplog.records, raw_uuid)
    assert _no_record_contains(caplog.records, raw_content)

    fp = _find_user_fp(caplog.records)
    assert fp is not None
    assert _HEX12.match(fp), fp
    assert fp != raw_uuid


def test_chatwoot_webhook_id_route_does_not_log_raw_pii(app, monkeypatch, caplog) -> None:
    """Same PII-redaction guarantee, verified on the actual primary path
    (the webhook_id-based route) in default enforce mode -- not just on the
    deprecated legacy route."""
    a, _tctx = app
    monkeypatch.setattr(chat_module, "process_message", _fake_process_message)
    caplog.set_level(logging.DEBUG, logger="src.api.external_chat")

    raw_uuid = "pu-12345-secret-uuid"
    raw_content = "my secret message content xyz"
    payload = {
        "event": "message_created",
        "message_type": 0,
        "content": raw_content,
        "conversation": {"id": 456},
        "sender": {"name": "Ravi", "identifier": raw_uuid, "type": "contact"},
    }

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook/wh-test-1", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}

    assert _no_record_contains(caplog.records, raw_uuid)
    assert _no_record_contains(caplog.records, raw_content)

    fp = _find_user_fp(caplog.records)
    assert fp is not None
    assert _HEX12.match(fp), fp
    assert fp != raw_uuid


async def test_external_message_session_creation_does_not_log_raw_user_id(app, monkeypatch, caplog) -> None:
    a, _tctx = app
    monkeypatch.setattr(chat_module, "process_message", _fake_process_message)
    # See comment in test_chatwoot_webhook_does_not_log_raw_pii: scope to our
    # own logger so aiosqlite's SQL-echo (which legitimately logs the raw
    # user_id as a bind parameter) doesn't produce a false failure here.
    caplog.set_level(logging.DEBUG, logger="src.api.external_chat")

    raw_user_id = "pu-99999-secret-uuid"

    client = TestClient(a)
    resp = client.post(
        "/integrations/message",
        json={
            "conversation_id": "conv-brand-new",
            "text": "hello there",
            "user_id": raw_user_id,
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "mocked response text"

    assert _no_record_contains(caplog.records, raw_user_id)

    fp = _find_user_fp(caplog.records)
    assert fp is not None
    assert _HEX12.match(fp), fp
    assert fp != raw_user_id


# ---------------------------------------------------------------------------
# H3 remediation: POST /integrations/chatwoot/webhook/{webhook_id}
# ---------------------------------------------------------------------------


def _basic_payload(inbox_id: int | None = None) -> dict:
    payload = {
        "event": "message_created",
        "message_type": 0,
        "content": "hi",
        "conversation": {"id": 1},
        "sender": {"name": "Ravi", "identifier": "u1", "type": "contact"},
    }
    if inbox_id is not None:
        payload["inbox"] = {"id": inbox_id}
    return payload


def test_chatwoot_webhook_id_unknown_rejects(monkeypatch, caplog) -> None:
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-known"},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.WARNING, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook/wh-unknown", json=_basic_payload())

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert calls == []
    assert any(getattr(r, "reason", None) == "unknown_webhook_id" for r in caplog.records)


def test_chatwoot_webhook_id_no_hmac_configured_processes_correct_tenant(monkeypatch, caplog) -> None:
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1"},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.INFO, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook/wh-1", json=_basic_payload())

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}
    assert len(calls) == 1
    assert not any("HMAC" in r.getMessage() for r in caplog.records)
    # Confirm it's really the correct tenant that got processed.
    assert any(getattr(r, "tenant", None) == "t1" for r in caplog.records)


def test_chatwoot_webhook_id_inbox_mismatch_rejects_regression(monkeypatch, caplog) -> None:
    """Regression guard for H3: tenant identity comes from the webhook_id
    path segment (capability token), never from the body. A correct
    webhook_id for tenant A whose payload claims tenant B's inbox_id must
    resolve tenant A (not B) and then be rejected by the consistency
    cross-check -- no background turn fires for either tenant."""
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="tA", slug="tA", name="TenantA"),
        secrets={"chatwoot:webhook_id": "wh-a", "chatwoot:inbox_id": "111"},
    )
    resolver.register(
        TenantSettings(id="tB", slug="tB", name="TenantB"),
        secrets={"chatwoot:webhook_id": "wh-b", "chatwoot:inbox_id": "222"},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.WARNING, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook/wh-a", json=_basic_payload(inbox_id=222))

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert calls == []
    record = next(r for r in caplog.records if getattr(r, "reason", None) == "inbox_id_mismatch")
    # Identity resolved from the path (tenant A), never from the body.
    assert record.tenant == "tA"


def test_chatwoot_webhook_id_no_inbox_id_configured_skips_crosscheck(monkeypatch, caplog) -> None:
    """Register TWO tenants -- t1 (no chatwoot:inbox_id configured, so the
    cross-check is skipped) and t2 (a different webhook_id AND a configured
    chatwoot:inbox_id) -- so the test can actually prove identity came from
    the path segment rather than merely observing "some tenant got
    processed". With only one tenant registered there'd be no way to
    distinguish "processed as t1" from "processed as some other tenant"."""
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1"},  # no chatwoot:inbox_id configured
    )
    resolver.register(
        TenantSettings(id="t2", slug="t2", name="Other"),
        secrets={"chatwoot:webhook_id": "wh-2", "chatwoot:inbox_id": "42"},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.INFO, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook/wh-1", json=_basic_payload(inbox_id=999))

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}
    assert len(calls) == 1
    # Processed AS t1 (the tenant whose webhook_id was in the path) -- never t2.
    tenant_values = {getattr(r, "tenant", None) for r in caplog.records if getattr(r, "tenant", None)}
    assert "t1" in tenant_values
    assert "t2" not in tenant_values


def test_chatwoot_webhook_id_hmac_missing_signature_rejects(monkeypatch) -> None:
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1", "chatwoot:webhook_hmac_secret": "topsecret"},
    )
    a = _make_webhook_app(resolver)

    raw_body = json.dumps(_basic_payload()).encode()
    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-1",
        content=raw_body,
        headers={"X-Chatwoot-Timestamp": str(int(time.time()))},
    )

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert calls == []


def test_chatwoot_webhook_id_hmac_wrong_signature_rejects(monkeypatch) -> None:
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1", "chatwoot:webhook_hmac_secret": "topsecret"},
    )
    a = _make_webhook_app(resolver)

    raw_body = json.dumps(_basic_payload()).encode()
    ts = str(int(time.time()))
    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-1",
        content=raw_body,
        headers={"X-Chatwoot-Timestamp": ts, "X-Chatwoot-Signature": "sha256=" + "0" * 64},
    )

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert calls == []


def test_chatwoot_webhook_id_hmac_correct_signature_processes(monkeypatch) -> None:
    secret = "topsecret"
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1", "chatwoot:webhook_hmac_secret": secret},
    )
    a = _make_webhook_app(resolver)

    raw_body = json.dumps(_basic_payload()).encode()
    ts = str(int(time.time()))
    sig = _chatwoot_signature(secret, raw_body, ts)
    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-1",
        content=raw_body,
        headers={"X-Chatwoot-Timestamp": ts, "X-Chatwoot-Signature": sig},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}
    assert len(calls) == 1


def test_chatwoot_webhook_id_hmac_stale_timestamp_rejects(monkeypatch) -> None:
    secret = "topsecret"
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1", "chatwoot:webhook_hmac_secret": secret},
    )
    a = _make_webhook_app(resolver)

    raw_body = json.dumps(_basic_payload()).encode()
    ts = str(int(time.time()) - 999)
    sig = _chatwoot_signature(secret, raw_body, ts)
    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-1",
        content=raw_body,
        headers={"X-Chatwoot-Timestamp": ts, "X-Chatwoot-Signature": sig},
    )

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert calls == []


def test_chatwoot_webhook_id_hmac_log_only_mode_proceeds_on_failure(monkeypatch, caplog) -> None:
    monkeypatch.setenv("VOX_WEBHOOK_SIGNATURE_MODE", "log_only")
    secret = "topsecret"
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1", "chatwoot:webhook_hmac_secret": secret},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.WARNING, logger="src.api.external_chat")

    raw_body = json.dumps(_basic_payload()).encode()
    ts = str(int(time.time()))
    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-1",
        content=raw_body,
        headers={"X-Chatwoot-Timestamp": ts, "X-Chatwoot-Signature": "sha256=" + "0" * 64},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}
    assert len(calls) == 1
    assert any(
        r.getMessage().startswith("chatwoot webhook: HMAC check would have rejected")
        and getattr(r, "mode", None) == "log_only"
        for r in caplog.records
    )


def test_chatwoot_webhook_id_hmac_verifies_exact_raw_bytes_not_reserialized(monkeypatch) -> None:
    """Sign the exact raw wire bytes (irregular spacing + trailing newline
    that a naive json.loads -> json.dumps round-trip would NOT reproduce)
    to prove the handler HMAC-checks against await request.body(), never a
    re-serialization of the parsed dict."""
    secret = "topsecret"
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-1", "chatwoot:webhook_hmac_secret": secret},
    )
    a = _make_webhook_app(resolver)

    raw_body = (
        b'{"event": "message_created",  "message_type":0, "content": "hi", '
        b'"conversation": {"id": 1}, '
        b'"sender": {"name": "Ravi", "identifier": "u1", "type": "contact"}}\n'
    )
    ts = str(int(time.time()))
    sig = _chatwoot_signature(secret, raw_body, ts)
    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-1",
        content=raw_body,
        headers={"X-Chatwoot-Timestamp": ts, "X-Chatwoot-Signature": sig},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}
    assert len(calls) == 1


def test_chatwoot_webhook_legacy_route_disabled_in_enforce_mode(caplog) -> None:
    resolver = InMemoryTenantResolver()
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.WARNING, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook", json=_basic_payload())

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert any(getattr(r, "reason", None) == "legacy_route_disabled" for r in caplog.records)


def test_chatwoot_webhook_legacy_route_log_only_mode_processes_and_warns(monkeypatch, caplog) -> None:
    monkeypatch.setenv("VOX_WEBHOOK_SIGNATURE_MODE", "log_only")
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:inbox_id": "42"},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.WARNING, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post("/integrations/chatwoot/webhook", json=_basic_payload(inbox_id=42))

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"accepted": True}
    assert len(calls) == 1
    assert any(getattr(r, "reason", None) == "legacy_route_deprecated" for r in caplog.records)


# ---------------------------------------------------------------------------
# Fix round: malformed / non-dict JSON bodies must not create a webhook_id
# enumeration oracle (400 vs 401, or 500 vs 401) -- every failure mode on the
# webhook_id route must return the IDENTICAL uniform 401.
# ---------------------------------------------------------------------------


def test_chatwoot_webhook_id_malformed_json_rejects_uniformly(monkeypatch, caplog) -> None:
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-known"},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.WARNING, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-known",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 401
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert calls == []
    assert any(getattr(r, "reason", None) == "invalid_json" for r in caplog.records)


def test_chatwoot_webhook_id_malformed_json_matches_unknown_webhook_id_response(monkeypatch) -> None:
    """The actual oracle-closure proof: a KNOWN webhook_id with a malformed
    body and an UNKNOWN webhook_id with the same malformed body must produce
    byte-for-byte identical responses -- otherwise an attacker can use the
    response to learn whether a given webhook_id is real."""
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-known"},
    )
    a = _make_webhook_app(resolver)
    client = TestClient(a)

    malformed = b"{not json"
    resp_known = client.post(
        "/integrations/chatwoot/webhook/wh-known",
        content=malformed,
        headers={"Content-Type": "application/json"},
    )
    resp_unknown = client.post(
        "/integrations/chatwoot/webhook/wh-does-not-exist",
        content=malformed,
        headers={"Content-Type": "application/json"},
    )

    assert resp_known.status_code == resp_unknown.status_code == 401
    assert resp_known.json() == resp_unknown.json() == {"detail": "invalid webhook signature"}


@pytest.mark.parametrize("body", [b"[1,2,3]", b"null", b'"x"', b"true", b"42"])
def test_chatwoot_webhook_id_non_dict_json_rejects_not_500(monkeypatch, body: bytes) -> None:
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:webhook_id": "wh-known"},
    )
    a = _make_webhook_app(resolver)

    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook/wh-known",
        content=body,
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 401, resp.text
    assert resp.json() == {"detail": "invalid webhook signature"}
    assert calls == []


def test_chatwoot_webhook_legacy_malformed_json_log_only_mode_does_not_crash(monkeypatch, caplog) -> None:
    """Legacy route in log_only mode with a malformed body must behave like
    "no inbox_id extracted" -- the existing not-mapped 200 response -- not
    crash and not invent a new response shape."""
    monkeypatch.setenv("VOX_WEBHOOK_SIGNATURE_MODE", "log_only")
    calls = _capture_create_task(monkeypatch)
    resolver = InMemoryTenantResolver()
    resolver.register(
        TenantSettings(id="t1", slug="t1", name="Acme"),
        secrets={"chatwoot:inbox_id": "42"},
    )
    a = _make_webhook_app(resolver)
    caplog.set_level(logging.WARNING, logger="src.api.external_chat")

    client = TestClient(a)
    resp = client.post(
        "/integrations/chatwoot/webhook",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ignored": True, "reason": "inbox_id not mapped to any tenant"}
    assert calls == []
