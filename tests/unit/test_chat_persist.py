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
    input_tokens = 0
    output_tokens = 0
    llm_provider = ""
    llm_model = ""


@pytest.mark.asyncio
async def test_persist_turn_returns_customer_msg_id(db_session):
    persisted = await chat_api._persist_turn("s1", "hello", _FakeResult())
    assert isinstance(persisted, chat_api.PersistedTurnIds)
    assert isinstance(persisted.customer_message_id, int)
    assert persisted.customer_message_id > 0
    assert isinstance(persisted.agent_message_id, int)
    assert persisted.agent_message_id > 0


@pytest.mark.asyncio
async def test_persist_turn_missing_session_logs_debug_and_writes_nothing(db_session, caplog):
    """A vanished/nonexistent chat_sessions row used to make an entire turn's
    transcript (both the customer's message and the reply) disappear with
    only a generic all-None PersistedTurnIds -- indistinguishable from any
    other persistence failure. Mutation proof: an existing session id (see
    test_persist_turn_returns_customer_msg_id above) does not log this line
    at all."""
    with caplog.at_level("DEBUG", logger="src.api.chat"):
        persisted = await chat_api._persist_turn("does-not-exist", "hello", _FakeResult())
    assert persisted == chat_api.PersistedTurnIds()
    missing_logs = [r for r in caplog.records if "chat session row missing" in r.message]
    assert len(missing_logs) == 1
    assert missing_logs[0].levelname == "DEBUG"
    assert missing_logs[0].session_id == "does-not-exist"

    from sqlalchemy import select
    async with db_session() as db:
        rows = (await db.execute(select(ChatMessage))).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_persist_turn_with_media_url(db_session):
    from sqlalchemy import select
    from src.models.chat import ChatMessage
    persisted = await chat_api._persist_turn(
        "s1", "[audio]", _FakeResult(),
        user_type="audio", media_mime="audio/webm",
        media_url="chat/t1/s1/abc.webm",
    )
    async with db_session() as db:
        row = await db.get(ChatMessage, persisted.customer_message_id)
    assert row.media_url == "chat/t1/s1/abc.webm"
    assert row.media_mime == "audio/webm"
    assert row.type == "audio"


@pytest.mark.asyncio
async def test_persist_turn_with_reply_media_extends_agent_row(db_session):
    """`reply_media_mime`/`reply_media_url` (the synthesized voice-note reply's
    audio) land on the AGENT's own row, not a parallel record — same media
    columns the customer's inbound attachment already uses."""
    persisted = await chat_api._persist_turn(
        "s1", "hello", _FakeResult(),
        reply_media_mime="audio/mpeg", reply_media_url="chat/t1/s1/reply.mp3",
    )
    async with db_session() as db:
        agent_row = await db.get(ChatMessage, persisted.agent_message_id)
        customer_row = await db.get(ChatMessage, persisted.customer_message_id)
    assert agent_row.role == "agent"
    assert agent_row.media_url == "chat/t1/s1/reply.mp3"
    assert agent_row.media_mime == "audio/mpeg"
    assert agent_row.type == "audio"
    # The customer's own row is untouched by the reply's media fields.
    assert customer_row.media_url is None


@pytest.mark.asyncio
async def test_persist_turn_with_source_media_url_sets_customer_row_only(db_session):
    """`source_media_url` (the client's own original URL, e.g. the CRM's
    inbound `media_url`) belongs on the CUSTOMER's row only — the agent's
    reply row has no client-supplied URL of its own and must stay NULL."""
    persisted = await chat_api._persist_turn(
        "s1", "[image]", _FakeResult(),
        user_type="image", media_mime="image/png", media_url="chat/t1/s1/shot.png",
        source_media_url="https://crm.example.com/uploads/shot.png",
    )
    async with db_session() as db:
        customer_row = await db.get(ChatMessage, persisted.customer_message_id)
        agent_row = await db.get(ChatMessage, persisted.agent_message_id)
    assert customer_row.source_media_url == "https://crm.example.com/uploads/shot.png"
    # The object-key media_url keeps being stored exactly as before, alongside it.
    assert customer_row.media_url == "chat/t1/s1/shot.png"
    assert agent_row.source_media_url is None


@pytest.mark.asyncio
async def test_persist_turn_source_media_url_defaults_to_none(db_session):
    """No `source_media_url` argument (the overwhelming majority of calls,
    e.g. plain text turns) must leave the column NULL, not an empty string."""
    persisted = await chat_api._persist_turn("s1", "hello", _FakeResult())
    async with db_session() as db:
        customer_row = await db.get(ChatMessage, persisted.customer_message_id)
    assert customer_row.source_media_url is None


@pytest.mark.asyncio
async def test_persist_inbound_media_message_writes_one_committed_row(db_session):
    """The pre-persist helper writes exactly one customer row with the media
    fields set, commits it (readable from a fresh session), and bumps
    message_count by 1 — see its docstring on why the commit matters:
    submit_deposit_verification reads through a separate DB session."""
    from sqlalchemy import select

    msg_id = await chat_api._persist_inbound_media_message(
        "s1", "[image]", user_type="image", media_mime="image/png",
        media_url="chat/t1/s1/shot.png",
        source_media_url="https://crm.example.com/uploads/shot.png",
    )
    assert isinstance(msg_id, int)

    async with db_session() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "s1")
        )).scalars().all()
        session_row = await db.get(ChatSession, "s1")
    assert len(rows) == 1
    row = rows[0]
    assert row.id == msg_id
    assert row.role == "customer"
    assert row.type == "image"
    assert row.media_mime == "image/png"
    assert row.media_url == "chat/t1/s1/shot.png"
    assert row.source_media_url == "https://crm.example.com/uploads/shot.png"
    assert session_row.message_count == 1


@pytest.mark.asyncio
async def test_persist_inbound_media_message_returns_id_despite_close_failure(db_session):
    """Regression test: if session teardown (`close()` / pool reset-on-return)
    raises AFTER the commit has already gone through, the helper must still
    return the committed row's id, not `None` — otherwise `_persist_turn`'s
    fallback path would write a second customer row for the same image. See
    the helper's docstring: teardown is deliberately split out of the
    `except` that guards the write itself, so a close-time failure can't
    masquerade as a write failure and flip a successful commit into `None`."""
    from sqlalchemy.ext.asyncio import AsyncSession

    class _FlakyCloseSession(AsyncSession):
        async def close(self):
            await super().close()
            raise RuntimeError("pool reset-on-return failed")

    engine = db_session.kw["bind"]
    flaky_sm = async_sessionmaker(engine, class_=_FlakyCloseSession, expire_on_commit=False)
    chat_api.set_chat_sessionmaker(flaky_sm)
    try:
        msg_id = await chat_api._persist_inbound_media_message(
            "s1", "[image]", user_type="image", media_mime="image/png",
            media_url="chat/t1/s1/shot.png",
        )
    finally:
        chat_api.set_chat_sessionmaker(db_session)

    assert isinstance(msg_id, int)

    async with db_session() as db:
        session_row = await db.get(ChatSession, "s1")
        row = await db.get(ChatMessage, msg_id)
    assert session_row.message_count == 1
    assert row is not None
    assert row.role == "customer"


@pytest.mark.asyncio
async def test_persist_turn_with_customer_message_id_writes_only_agent_row(db_session):
    """When `customer_message_id` is passed, _persist_turn must not construct
    or add a second customer row — only the agent row — and must bump
    message_count by 1 (not 2), returning the passed-through customer id."""
    from sqlalchemy import select

    pre_id = await chat_api._persist_inbound_media_message(
        "s1", "[image]", user_type="image", media_mime="image/png",
        media_url="chat/t1/s1/shot.png",
    )
    persisted = await chat_api._persist_turn(
        "s1", "[image]", _FakeResult(), customer_message_id=pre_id,
    )
    assert persisted.customer_message_id == pre_id

    async with db_session() as db:
        customer_rows = (await db.execute(
            select(ChatMessage).where(
                ChatMessage.session_id == "s1", ChatMessage.role == "customer")
        )).scalars().all()
        agent_row = await db.get(ChatMessage, persisted.agent_message_id)
        session_row = await db.get(ChatSession, "s1")
    # Exactly the one customer row from the pre-persist step — _persist_turn
    # added none of its own.
    assert len(customer_rows) == 1
    assert customer_rows[0].id == pre_id
    assert agent_row is not None
    assert agent_row.role == "agent"
    # +1 from the pre-persist, +1 from this call's agent-only write == 2.
    assert session_row.message_count == 2


@pytest.mark.asyncio
async def test_persist_turn_without_customer_message_id_unchanged(db_session):
    """No `customer_message_id` (every existing call site) must keep behaving
    exactly as today: two rows written by this single call, message_count +2."""
    from sqlalchemy import select

    persisted = await chat_api._persist_turn("s1", "hello", _FakeResult())
    assert persisted.customer_message_id is not None
    assert persisted.agent_message_id is not None

    async with db_session() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "s1")
        )).scalars().all()
        session_row = await db.get(ChatSession, "s1")
    assert len(rows) == 2
    assert {r.role for r in rows} == {"customer", "agent"}
    assert session_row.message_count == 2


@pytest.mark.asyncio
async def test_persist_turn_falls_back_to_full_customer_row_when_pre_persist_fails(db_session):
    """Direct unit check of `_persist_turn`'s own fallback branch (the
    `if customer_message_id is None:` construction of the customer
    `ChatMessage`): given `customer_message_id=None` plus a full set of media
    kwargs, it must build a real type="image" row with all of them, not a
    corrupt type="text"/no-media row.

    This calls `_persist_turn` directly, so it CANNOT see whether the real
    WS image/video call site (src/api/chat.py's image/video branch) actually
    passes those kwargs through — passing `customer_message_id=None`
    explicitly here hits the exact same branch as
    `test_persist_turn_without_customer_message_id_unchanged` above (not
    passing it at all), just with non-default arguments. For coverage of the
    real call site — which is what matters when
    `_persist_inbound_media_message` fails in production — see
    `tests/unit/test_chat_image_s3.py::test_image_ws_falls_back_to_full_customer_row_when_pre_persist_returns_none`,
    which drives an actual WS frame through it end-to-end."""
    from sqlalchemy import select

    persisted = await chat_api._persist_turn(
        "s1", "[image]", _FakeResult(),
        user_type="image", media_mime="image/png",
        media_url="chat/t1/s1/shot.png",
        source_media_url="https://crm.example.com/uploads/shot.png",
        customer_message_id=None,  # simulates the pre-persist helper returning None
    )
    async with db_session() as db:
        customer_rows = (await db.execute(
            select(ChatMessage).where(
                ChatMessage.session_id == "s1", ChatMessage.role == "customer")
        )).scalars().all()
    assert len(customer_rows) == 1
    row = customer_rows[0]
    assert row.id == persisted.customer_message_id
    assert row.type == "image"
    assert row.media_mime == "image/png"
    assert row.media_url == "chat/t1/s1/shot.png"
    assert row.source_media_url == "https://crm.example.com/uploads/shot.png"

# NOTE: this module used to also have
# `test_pre_persist_helper_row_survives_a_raising_turn` here. It was removed
# as theatre: its "raising turn" was a locally-defined coroutine raised and
# immediately caught by `pytest.raises` right there in the test body — it
# never called `_persist_turn` or drove any real turn-failure path, so minus
# those three lines it was assertion-for-assertion identical to
# `test_persist_inbound_media_message_writes_one_committed_row` above. Real
# coverage of "the pre-persisted customer row survives a raising turn" now
# lives at the WS level, where a raising turn is actually possible:
# `tests/unit/test_chat_image_s3.py::test_image_ws_pre_persisted_customer_row_survives_a_raising_turn`.
