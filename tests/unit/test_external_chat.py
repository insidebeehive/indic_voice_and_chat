"""Tests for src/api/external_chat.py.

Verifies that raw PII (player UUID, message content) never appears in log
records emitted along the external chat paths -- the Chatwoot webhook
(``/integrations/chatwoot/webhook``) and the generic REST session-creation
path (``/integrations/message``) -- and that the fingerprinted ``user_fp``
field is present instead wherever the raw ``user_id`` used to be logged.
"""

from __future__ import annotations

import logging
import re
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
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import Tenant

HEADERS = {"Authorization": "Bearer test-token"}
_HEX12 = re.compile(r"^[0-9a-f]{12}$")


class _ChatwootResolverStub:
    """Minimal stand-in for the DB-backed tenant resolver's
    resolve_by_chatwoot_inbox -- the Chatwoot webhook path resolves its
    tenant from app.state.tenant_resolver, not the bearer-token resolver
    used by the other endpoints."""

    def __init__(self, tctx: TenantContext) -> None:
        self._tctx = tctx

    async def resolve_by_chatwoot_inbox(self, inbox_id: str) -> TenantContext | None:
        return self._tctx


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
    a.state.tenant_resolver = _ChatwootResolverStub(tctx)

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
