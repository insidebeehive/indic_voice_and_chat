"""Tests for _persist_turn returning customer message id."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat as chat_api
from src.models.database import Base
from src.models.chat import ChatMessage, ChatSession


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    chat_api.set_chat_sessionmaker(sm)
    async with sm() as session:
        session.add(ChatSession(
            id="s1", tenant_id="t1", language="hi", status="active",
            mode="ai", extra_data={},
        ))
        await session.commit()
    yield sm
    chat_api.set_chat_sessionmaker(None)
    await engine.dispose()


class _FakeResult:
    class _Resp:
        response_text = "hi"
        sources_used = []
        suggested_followups = []
        action = "none"
    response = _Resp()
    escalation = None
    call_offer = None


@pytest.mark.asyncio
async def test_persist_turn_returns_customer_msg_id(db_session):
    msg_id = await chat_api._persist_turn("s1", "hello", _FakeResult())
    assert isinstance(msg_id, int)
    assert msg_id > 0


@pytest.mark.asyncio
async def test_persist_turn_with_media_url(db_session):
    from sqlalchemy import select
    from src.models.chat import ChatMessage
    msg_id = await chat_api._persist_turn(
        "s1", "[audio]", _FakeResult(),
        user_type="audio", media_mime="audio/webm",
        media_url="chat/t1/s1/abc.webm",
    )
    async with db_session() as db:
        row = await db.get(ChatMessage, msg_id)
    assert row.media_url == "chat/t1/s1/abc.webm"
    assert row.media_mime == "audio/webm"
    assert row.type == "audio"
