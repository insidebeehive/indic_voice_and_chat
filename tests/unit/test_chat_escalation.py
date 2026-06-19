"""Chat escalation → signed tenant event (Phase 4)."""

from __future__ import annotations

import pytest

from src.agents.chatbot import ChatTurnResult
from src.api import chat
from src.api.call_store import set_tenant_event_notifier
from src.dialogue.response_parser import ChatBotResponse


def _result(escalation=None) -> ChatTurnResult:
    return ChatTurnResult(
        response=ChatBotResponse(response_text="ok", language="en"),
        retrieved=[], rag_context_chars=0, escalation=escalation)


@pytest.mark.asyncio
async def test_emit_escalation_sends_signed_chat_event() -> None:
    captured: list[dict] = []

    async def _notifier(env: dict) -> None:
        captured.append(env)

    set_tenant_event_notifier(_notifier)
    try:
        await chat._emit_escalation(
            "t1", "cs_1", _result({"reason": "frustrated", "summary": "refund dispute"}))
    finally:
        set_tenant_event_notifier(None)

    assert len(captured) == 1
    env = captured[0]
    assert env["event_type"] == "chat.escalated"
    assert env["channel"] == "chat"
    assert env["call_id"] == "cs_1"
    assert env["tenant_id"] == "t1"
    assert env["data"] == {"reason": "frustrated", "summary": "refund dispute"}


@pytest.mark.asyncio
async def test_no_escalation_emits_nothing() -> None:
    captured: list[dict] = []

    async def _notifier(env: dict) -> None:
        captured.append(env)

    set_tenant_event_notifier(_notifier)
    try:
        await chat._emit_escalation("t1", "cs_1", _result(escalation=None))
    finally:
        set_tenant_event_notifier(None)
    assert captured == []
