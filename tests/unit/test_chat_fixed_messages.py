"""Localization of the four fixed customer-facing chat messages that live
OUTSIDE the LLM turn loop -- so, unlike a normal bot reply, nothing about
them naturally follows the customer's language on its own:

1. ``_revert_to_bot`` -- CRM declined the handoff, or no claim within
   ``_AWAIT_HUMAN_TIMEOUT_S``.
2. ``_handle_escalation``'s "support team currently unavailable" message
   (outside support hours).
3. ``_handle_escalation``'s "wasn't able to connect you" message (handoff
   webhook failed).
4. The idle-timeout farewell in the main customer WS loop.

Covers:
- ``_customer_language_for_session``: the DB-backed language lookup used at
  call sites (1, 2 above) that have no live per-connection ``interim_lang``
  in hand, only a ``session_id``.
- ``_fixed_text``: language selection (en/hi/hinglish, English fallback for
  every other language code) across all four ``_FIXED_MESSAGES`` keys.
- No speaker-gendered first-person Hindi/Hinglish verb forms in any of the
  localized strings -- same rule ``_INTERIM_WAIT_MESSAGES`` documents.
- End-to-end over a real websocket: the idle-timeout farewell (4 above), in
  Hindi/Hinglish/English. The revert/decline path's (1 above) end-to-end
  coverage lives in test_chat_routes.py, which already has the fixture
  wiring a real CRM-escalation flow end to end.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat as chat_api
from src.models.database import Base
from src.models.chat import ChatMessage, ChatSession


FIXED_KEYS = [
    "revert_to_bot", "escalation_unavailable", "escalation_failed", "idle_farewell",
]


# --- _fixed_text: language selection --------------------------------------


@pytest.mark.parametrize("key", FIXED_KEYS)
def test_fixed_text_english(key: str) -> None:
    assert chat_api._fixed_text(key, "en") == chat_api._FIXED_MESSAGES[key]["en"]


@pytest.mark.parametrize("key", FIXED_KEYS)
def test_fixed_text_hindi(key: str) -> None:
    assert chat_api._fixed_text(key, "hi") == chat_api._FIXED_MESSAGES[key]["hi"]


@pytest.mark.parametrize("key", FIXED_KEYS)
def test_fixed_text_hinglish(key: str) -> None:
    assert chat_api._fixed_text(key, "hinglish") == chat_api._FIXED_MESSAGES[key]["hinglish"]


@pytest.mark.parametrize("key", FIXED_KEYS)
def test_fixed_text_unmapped_language_falls_back_to_english(key: str) -> None:
    """Other languages (e.g. Tamil) aren't covered here -- fall back to
    English, same convention as ``_greeting``/``_interim_wait_text``."""
    assert chat_api._fixed_text(key, "ta") == chat_api._FIXED_MESSAGES[key]["en"]


def test_escalation_unavailable_keeps_next_slot_placeholder() -> None:
    for lang in ("en", "hi", "hinglish"):
        text = chat_api._fixed_text("escalation_unavailable", lang).format(
            next_slot="tomorrow at 10:00")
        assert "tomorrow at 10:00" in text


# --- Speaker-gender neutrality ---------------------------------------------


def test_no_speaker_gendered_first_person_forms() -> None:
    """All four originals speak in first person ("I'm sorry", "I'll close
    this chat"); the Hindi/Hinglish variants must avoid marking the SPEAKER's
    (the bot's) gender, per the convention _INTERIM_WAIT_MESSAGES documents."""
    bad_substrings = [
        "रहा हूँ", "रही हूँ", "raha hoon", "rahi hoon",
        "करता हूँ", "करती हूँ", "karta hoon", "karti hoon",
        "करूँगा", "करूँगी",  # gendered first-person future ("I will do")
        "सकता हूँ", "सकती हूँ", "sakta hoon", "sakti hoon",
        "रहूँगा", "रहूँगी", "rahunga", "rahungi",
    ]
    for key, variants in chat_api._FIXED_MESSAGES.items():
        for lang in ("hi", "hinglish"):
            text = variants[lang]
            for bad in bad_substrings:
                assert bad not in text, f"{key}/{lang} has gendered form {bad!r}: {text}"


# --- _customer_language_for_session -----------------------------------


@pytest_asyncio.fixture
async def db_sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    chat_api.set_chat_sessionmaker(sm)
    yield sm
    chat_api.set_chat_sessionmaker(None)
    await engine.dispose()


async def _seed_session(sm, session_id: str = "s1", language: str = "en") -> None:
    async with sm() as db:
        db.add(ChatSession(id=session_id, tenant_id="t1", language=language,
                            status="active", mode="ai", extra_data={}))
        await db.commit()


async def _add_customer_message(sm, session_id: str, content: str) -> None:
    async with sm() as db:
        db.add(ChatMessage(session_id=session_id, role="customer", type="text", content=content))
        await db.commit()


@pytest.mark.asyncio
async def test_customer_language_devanagari_message(db_sm) -> None:
    await _seed_session(db_sm, "s1", "en")
    await _add_customer_message(db_sm, "s1", "मुझे मदद चाहिए")
    assert await chat_api._customer_language_for_session("s1", "en") == "hi"


@pytest.mark.asyncio
async def test_customer_language_hinglish_message(db_sm) -> None:
    await _seed_session(db_sm, "s1", "en")
    await _add_customer_message(db_sm, "s1", "mujhe madad chahiye please")
    assert await chat_api._customer_language_for_session("s1", "en") == "hinglish"


@pytest.mark.asyncio
async def test_customer_language_signal_less_message_falls_back(db_sm) -> None:
    """"ok"/digits carry no language signal -- keep the fallback."""
    await _seed_session(db_sm, "s1", "ta")
    await _add_customer_message(db_sm, "s1", "ok")
    await _add_customer_message(db_sm, "s1", "12345")
    assert await chat_api._customer_language_for_session("s1", "ta") == "ta"


@pytest.mark.asyncio
async def test_customer_language_walks_back_past_signal_less_messages(db_sm) -> None:
    """Newest message ("ok") carries no signal; the Hindi one just before it
    is still within the lookback window and should be used."""
    await _seed_session(db_sm, "s1", "en")
    await _add_customer_message(db_sm, "s1", "मुझे मदद चाहिए")
    await _add_customer_message(db_sm, "s1", "ok")
    assert await chat_api._customer_language_for_session("s1", "en") == "hi"


@pytest.mark.asyncio
async def test_customer_language_no_customer_messages_falls_back(db_sm) -> None:
    await _seed_session(db_sm, "s1", "bn")
    assert await chat_api._customer_language_for_session("s1", "bn") == "bn"


@pytest.mark.asyncio
async def test_customer_language_db_error_falls_back() -> None:
    class _BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("db unavailable")

    chat_api.set_chat_sessionmaker(_BrokenSessionmaker())
    try:
        assert await chat_api._customer_language_for_session("missing", "en") == "en"
    finally:
        chat_api.set_chat_sessionmaker(None)


# --- End-to-end: idle-timeout farewell over a real websocket -------------


class _FakeTurnResult:
    class _Resp:
        response_text = "noted"
        sources_used = []
        suggested_followups = []
        action = "none"
        language = "en"
        confidence = "high"

    response = _Resp()
    escalation = None
    call_offer = None
    input_tokens = 0
    output_tokens = 0
    llm_provider = None
    llm_model = None


@pytest_asyncio.fixture
async def ws_ctx():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(ChatSession(id="sess1", tenant_id="t1", language="en",
                            status="active", mode="ai", extra_data={}))
        await db.commit()

    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent._llm = None
    fake_agent.llm = MagicMock()
    fake_agent.session = MagicMock()

    async def _handle(text):
        return _FakeTurnResult()

    fake_agent.handle_message = _handle
    fake_agent.summarize_session = AsyncMock(return_value="")

    async def fake_factory(tenant, scoped_id, *, customer_id=None, ticket_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, fake_agent

    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    await engine.dispose()


def _make_tenant(idle_timeout: float):
    fake_tenant = MagicMock()
    fake_tenant.id = "t1"
    fake_tenant.slug = "demo"
    fake_tenant.settings.chat_support.chat_idle_timeout_seconds = idle_timeout
    fake_tenant.settings.events_webhook_url = None  # skip the best-effort webhook call
    return fake_tenant


def _connect(fake_tenant):
    import src.auth.middleware as mw

    patcher = patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant))
    patcher.start()
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    client = TestClient(app)
    return patcher, client


@pytest.mark.asyncio
async def test_idle_farewell_hindi_for_devanagari_conversation(ws_ctx) -> None:
    sm, fake_agent = ws_ctx
    patcher, client = _connect(_make_tenant(idle_timeout=0.05))
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "मुझे मदद चाहिए"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")

            farewell = json.loads(ws.receive_text())
            assert farewell["type"] == "message"
            assert farewell["action"] == "end"
            assert farewell["text"] == chat_api._FIXED_MESSAGES["idle_farewell"]["hi"]

            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended" and ended["reason"] == "idle_timeout"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_idle_farewell_hinglish_for_hinglish_conversation(ws_ctx) -> None:
    sm, fake_agent = ws_ctx
    patcher, client = _connect(_make_tenant(idle_timeout=0.05))
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "mujhe madad chahiye please"}))
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "typing"
            frame = json.loads(ws.receive_text())
            assert frame["type"] == "message" and not frame.get("interim")

            farewell = json.loads(ws.receive_text())
            assert farewell["type"] == "message"
            assert farewell["text"] == chat_api._FIXED_MESSAGES["idle_farewell"]["hinglish"]

            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended" and ended["reason"] == "idle_timeout"
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_idle_farewell_english_when_no_signal_ever_seen(ws_ctx) -> None:
    """No customer message at all before the idle timeout fires -- falls
    back to the session's own (English) language, same as row.language."""
    sm, fake_agent = ws_ctx
    patcher, client = _connect(_make_tenant(idle_timeout=0.05))
    try:
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            farewell = json.loads(ws.receive_text())
            assert farewell["type"] == "message"
            assert farewell["text"] == chat_api._FIXED_MESSAGES["idle_farewell"]["en"]

            ended = json.loads(ws.receive_text())
            assert ended["type"] == "ended" and ended["reason"] == "idle_timeout"
    finally:
        patcher.stop()


def test_fixed_messages_are_in_the_script_their_key_claims() -> None:
    """Independent of the table's own contents: a "hi" entry must be Devanagari
    and the "hinglish"/"en" entries must not be, so a swapped or untranslated
    entry fails here rather than passing a lookup against itself."""
    def has_devanagari(text: str) -> bool:
        return any(0x0900 <= ord(c) <= 0x097F for c in text)
    for key, variants in chat_api._FIXED_MESSAGES.items():
        assert has_devanagari(variants["hi"]), f"{key}/hi is not Devanagari"
        assert not has_devanagari(variants["hinglish"]), f"{key}/hinglish has Devanagari"
        assert not has_devanagari(variants["en"]), f"{key}/en has Devanagari"
        assert variants["hinglish"] != variants["en"], f"{key}/hinglish is untranslated"
