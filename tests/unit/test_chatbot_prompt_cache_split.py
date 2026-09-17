"""ChatBotAgent's cache_split_prompt wiring (see src/agents/chatbot.py's
_compose, _fold_turn_context, and src/bootstrap.py's
_prompt_cache_split_enabled).

These tests exist to prove the piece that makes GeminiLLMAdapter's explicit-
cache registry (src/providers/llm/gemini.py) and build_chatbot_system_prompt's
include_variable_tail split (src/dialogue/prompts.py) actually useful
TOGETHER: with cache_split_prompt=True, the system prompt handed to the LLM
must be byte-identical across turns (so the provider's cache actually gets
reused instead of missing every call), and the per-turn variable tail
(retrieved sources / current date-time / language directive) must instead
ride inside the user-turn message — framed so the model doesn't mistake
platform-supplied context for something the customer said — without ever
mutating the object that gets persisted into session history.

Default (cache_split_prompt=False, unset) behavior is pinned first: every
existing call site of ChatBotAgent must be completely unaffected by this
parameter's existence.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from src.agents.base import AgentSession
from src.agents.chatbot import (
    TURN_CONTEXT_CLOSE,
    TURN_CONTEXT_OPEN,
    ChatBotAgent,
    _fold_turn_context,
)
from src.interfaces.llm import ContentPart, ILLMProvider, LLMMessage, LLMResult
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder
from src.rag.retriever import HybridRetriever, RetrievalConfig

from tests.unit.test_prompts import _freeze_clock


class FakeLLM(ILLMProvider):
    """Records every messages list it's called with; always returns the same
    canned JSON envelope (mirrors test_chatbot_agent.py's FakeLLM)."""

    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {
            "response_text": "ok",
            "language": "en",
            "action": "none",
        }
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(text=json.dumps(self._payload), finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


@pytest.fixture
async def retriever(tmp_faiss_index: str) -> HybridRetriever:
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    r = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        config=RetrievalConfig(strategy="hybrid", top_k=2, oversample_k=8, similarity_threshold=0.0),
    )
    await r.index([
        Document(id="c1", content="Plan B has 500GB unlimited data.",
                 metadata={"filename": "plans.pdf", "page": 2}),
    ])
    return r


def _make_agent(llm, retriever, **kw) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-cache-split"),
        llm=llm,
        retriever=retriever,
        company_name="Acme",
        language_default="en",
        **kw,
    )


# --- 1. Default behavior is unaffected ------------------------------------


@pytest.mark.asyncio
async def test_split_off_by_default_composes_exactly_as_before(retriever) -> None:
    llm = FakeLLM()
    agent = _make_agent(llm, retriever)  # cache_split_prompt left at its default (False)

    user_msg = LLMMessage(role="user", content="Tell me about Plan B")
    messages = agent._compose("Doc: Plan B has 500GB.", user_msg, query_text="Tell me about Plan B")

    # The tail is still inline in the system message — unsplit.
    assert messages[0].role == "system"
    assert "Current date (UTC)" in messages[0].content

    # The user-turn message is the ORIGINAL object, not folded/framed.
    assert messages[-1] is user_msg
    assert messages[-1].content == "Tell me about Plan B"
    assert TURN_CONTEXT_OPEN not in messages[-1].content


# --- 2. Split moves the tail into the user turn ----------------------------


def test_split_moves_tail_out_of_system_into_the_user_turn(retriever) -> None:
    llm = FakeLLM()
    agent = _make_agent(llm, retriever, cache_split_prompt=True)

    rag_sentinel = "SENTINEL_RAG_TEXT_PLAN_B_500GB"
    # "mera balance kya hai" reliably triggers the Hinglish language directive
    # via _latin_language_hint (word "mera"/"kya"/"hai" are all in
    # _HINGLISH_MARKERS) — _detect_script returns None for pure Latin script.
    query_text = "mera balance kya hai"
    user_msg = LLMMessage(role="user", content=query_text)

    messages = agent._compose(rag_sentinel, user_msg, query_text=query_text)

    system_content = messages[0].content
    assert messages[0].role == "system"
    assert "Current date (UTC)" not in system_content
    assert rag_sentinel not in system_content
    assert "Additional directives:" not in system_content

    tail_msg = messages[-1]
    assert tail_msg.content.startswith(TURN_CONTEXT_OPEN)
    assert rag_sentinel in tail_msg.content
    assert "Current date (UTC)" in tail_msg.content
    assert "Hinglish" in tail_msg.content
    close_idx = tail_msg.content.index(TURN_CONTEXT_CLOSE)
    customer_idx = tail_msg.content.index(query_text)
    assert close_idx < customer_idx, "TURN_CONTEXT_CLOSE must precede the customer's own text"
    assert tail_msg.content.endswith(query_text)


# --- 3. Never mutates what gets persisted ----------------------------------


def test_split_never_mutates_the_original_user_msg_object(retriever) -> None:
    llm = FakeLLM()
    agent = _make_agent(llm, retriever, cache_split_prompt=True)

    user_msg = LLMMessage(role="user", content="mera balance kya hai")
    messages = agent._compose("some rag context", user_msg, query_text="mera balance kya hai")

    # The original object handed in is untouched...
    assert user_msg.content == "mera balance kya hai"
    # ...and a NEW object was returned as the last message.
    assert messages[-1] is not user_msg


@pytest.mark.asyncio
async def test_split_never_persists_the_framed_tail_into_session_history(retriever) -> None:
    llm = FakeLLM({"response_text": "Your balance is safe.", "language": "en", "action": "none"})
    agent = _make_agent(llm, retriever, cache_split_prompt=True)

    await agent.handle_message("mera balance kya hai")

    # _persist appends the CALLER's original user_msg (plain customer text)
    # into session.turns, never the framed/tail-prefixed replacement that
    # _compose builds for the LLM call — otherwise every later turn's history
    # replay would carry this turn's frozen clock value and retrieved sources
    # forward forever, and the stored transcript would show internal framing
    # instead of what the customer actually typed.
    for m in agent.session.turns:
        if isinstance(m.content, str):
            assert TURN_CONTEXT_OPEN not in m.content
            assert "Current date (UTC)" not in m.content
    user_turns = [m for m in agent.session.turns if m.role == "user"]
    assert any(m.content == "mera balance kya hai" for m in user_turns)


# --- 4. System prompt is byte-identical across a clock tick ----------------


def test_split_system_prompt_is_byte_identical_across_a_clock_tick(monkeypatch, retriever) -> None:
    import datetime as _dt

    llm = FakeLLM()
    agent = _make_agent(llm, retriever, cache_split_prompt=True)

    _freeze_clock(monkeypatch, _dt.datetime(2026, 1, 15, 10, 30, tzinfo=_dt.UTC))
    messages_t1 = agent._compose("rag text", LLMMessage(role="user", content="hi"), query_text="hi")

    _freeze_clock(monkeypatch, _dt.datetime(2026, 1, 15, 10, 31, tzinfo=_dt.UTC))
    messages_t2 = agent._compose("rag text", LLMMessage(role="user", content="hi"), query_text="hi")

    assert messages_t1[0].content == messages_t2[0].content

    # Also across a calendar-day boundary, mirroring test_prompts.py's
    # analogous clock-tick test for build_chatbot_system_prompt itself.
    _freeze_clock(monkeypatch, _dt.datetime(2026, 3, 2, 23, 59, tzinfo=_dt.UTC))
    messages_t3 = agent._compose("rag text", LLMMessage(role="user", content="hi"), query_text="hi")
    assert messages_t1[0].content == messages_t3[0].content


# --- 5. Multimodal: preserves parts, no in-place mutation -------------------


def test_split_multimodal_prepends_a_text_part_and_preserves_the_image() -> None:
    caption_part = ContentPart(type="text", text="caption")
    image_part = ContentPart(type="image", inline_data={"mime_type": "image/png", "data": b"fake"})
    user_msg = LLMMessage(role="user", content=[caption_part, image_part])

    result = _fold_turn_context(user_msg, "some tail text")

    assert isinstance(result.content, list)
    assert len(result.content) == 3  # frame + original caption + image
    assert result.content[0].text.startswith(TURN_CONTEXT_OPEN)
    assert "some tail text" in result.content[0].text
    assert result.content[1] is caption_part
    assert result.content[2] is image_part

    # The original list object is untouched — no in-place mutation.
    assert len(user_msg.content) == 2
    assert user_msg.content[0] is caption_part
    assert user_msg.content[1] is image_part


# --- 6. bootstrap._prompt_cache_split_enabled -------------------------------


def test_prompt_cache_split_enabled_helper(monkeypatch) -> None:
    from src.bootstrap import _prompt_cache_split_enabled
    from src.providers.llm.gemini import GeminiLLMAdapter

    gemini = GeminiLLMAdapter({"client": object()})

    monkeypatch.delenv("GEMINI_EXPLICIT_CACHE", raising=False)
    assert _prompt_cache_split_enabled(gemini) is False

    monkeypatch.setenv("GEMINI_EXPLICIT_CACHE", "1")
    assert _prompt_cache_split_enabled(gemini) is True

    # Proves the isinstance check matters, not just the env var: a non-Gemini
    # provider must never split the prompt even with the flag on, since
    # splitting only pays for itself against an adapter that will actually
    # create and reuse an explicit cache from the static body.
    not_gemini = object()
    assert _prompt_cache_split_enabled(not_gemini) is False
