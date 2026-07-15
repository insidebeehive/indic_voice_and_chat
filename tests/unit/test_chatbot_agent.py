from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from src.agents.base import AgentSession
from src.agents.chatbot import ChatBotAgent, _detect_script, _latin_language_hint
from src.dialogue.context import SessionStore
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.rag.embeddings import HashEmbedder, IdentityReranker
from src.rag.retriever import HybridRetriever, RetrievalConfig


# --- Fakes ---------------------------------------------------------------


class FakeLLM(ILLMProvider):
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(text=json.dumps(self._payload), finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
async def retriever(tmp_faiss_index: str) -> HybridRetriever:
    store = FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index})
    r = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=store,
        reranker=IdentityReranker(),
        config=RetrievalConfig(
            strategy="hybrid",
            top_k=2,
            oversample_k=8,
            reranking=True,
            similarity_threshold=0.0,
        ),
    )
    await r.index([
        Document(id="c1", content="Plan B has 500GB unlimited data.", metadata={"filename": "plans.pdf", "page": 2}),
        Document(id="c2", content="Plan A is the basic 100GB plan for Rs 199.", metadata={"filename": "plans.pdf", "page": 1}),
        Document(id="c3", content="Cooking recipes for biryani and other dishes.", metadata={"filename": "cookbook.md"}),
    ])
    return r


def _make_agent(llm, retriever, store=None) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-1"),
        llm=llm,
        retriever=retriever,
        company_name="Acme",
        language_default="en",
        store=store,
    )


# --- Tests --------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_full_happy_path(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB unlimited data.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
        "suggested_followups": ["What's the price?"],
    })
    agent = _make_agent(llm, retriever)

    result = await agent.handle_message("Tell me about Plan B")
    assert result.response.response_text == "Plan B has 500GB unlimited data."
    assert result.response.confidence == "high"
    assert "plans.pdf:2" in result.response.sources_used
    assert len(result.retrieved) >= 1
    # System prompt was built with retrieved context
    sent = llm.calls[0]
    assert sent[0].role == "system"
    assert "Plan B" in sent[0].content


@pytest.mark.asyncio
async def test_handle_message_empty_input_returns_early(retriever) -> None:
    llm = FakeLLM({})
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("   ")
    assert result.response.parse_error == "empty user input"
    assert llm.calls == []


def test_detect_script_leaves_romanized_indic_undetected() -> None:
    # Romanized Hindi ("mera withdrawal kahan hai") is pure Latin script but
    # not English — this used to return "English" and force a "MUST be in
    # English" directive, overriding the prompt's own correct Roman-Hinglish
    # rule and causing the bot to ignore customers writing in Hinglish.
    assert _detect_script("mera withdrawal kahan hai") is None
    assert _detect_script("kya bol rahi ho madam") is None
    assert _detect_script("What is my balance?") is None  # ambiguous too — same script


def test_detect_script_still_detects_native_script() -> None:
    assert _detect_script("मेरा बैलेंस क्या है") == "Hindi"
    assert _detect_script("আমার ব্যালেন্স কত") == "Bengali"


def test_latin_language_hint_classifies_deterministically() -> None:
    # Hinglish markers → Hinglish; marker-free with 3+ words → English;
    # short marker-free acks → None (follow the conversation, don't force).
    assert _latin_language_hint("mera withdrawal kahan hai") == "Hinglish"
    assert _latin_language_hint("kya bol rahi ho madam") == "Hinglish"
    assert _latin_language_hint("balance check karo please") == "Hinglish"
    assert _latin_language_hint("tell me about this site") == "English"
    assert _latin_language_hint("whats my balance") == "English"
    assert _latin_language_hint("ok") is None
    assert _latin_language_hint("thanks") is None
    assert _latin_language_hint("") is None


@pytest.mark.asyncio
async def test_romanized_hindi_message_gets_firm_hinglish_directive(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Aapka koi pending withdrawal nahi hai abhi.",
        "language": "hi",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("mera withdrawal kahan hai")
    system_prompt = llm.calls[0][0].content
    assert "MUST be in English" not in system_prompt
    assert "romanized Hindi (Hinglish)" in system_prompt
    assert "NEVER Devanagari" in system_prompt


@pytest.mark.asyncio
async def test_english_message_gets_firm_english_directive(retriever) -> None:
    # Regression chain: (1) romanized Hindi was forced into English; (2) the
    # fix left Latin text signal-less, so default_language="hi" answered
    # English in Devanagari; (3) the advisory Roman-script directive let
    # Hinglish history momentum answer plain English ("tell me about this
    # site") in Hinglish. The directive must now NAME English firmly and
    # explicitly override earlier turns' language.
    llm = FakeLLM({
        "response_text": "Your balance is 100.",
        "language": "en",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("tell me about this site")
    system_prompt = llm.calls[0][0].content
    assert "MUST be in English" in system_prompt
    assert "language of earlier turns" in system_prompt


@pytest.mark.asyncio
async def test_short_ack_gets_no_language_directive(retriever) -> None:
    # A bare "ok" carries no language signal — forcing English would flip a
    # Hinglish conversation mid-stream. No directive: history governs.
    llm = FakeLLM({
        "response_text": "Theek hai!",
        "language": "hi",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("ok")
    system_prompt = llm.calls[0][0].content
    assert "MUST be in" not in system_prompt
    assert "romanized Hindi (Hinglish)" not in system_prompt


@pytest.mark.asyncio
async def test_hallucination_guard_strips_invented_citations(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB and free 5G.",
        "language": "en",
        "sources_used": ["plans.pdf:2", "wireless.pdf:99"],  # second is invented
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Plan B details?")
    assert result.response.sources_used == ["plans.pdf:2"]
    assert result.response.confidence == "low"


@pytest.mark.asyncio
async def test_followup_turn_includes_prior_history_in_prompt(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B is Rs 699 per month.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("Tell me about Plan B")
    await agent.handle_message("What's the price?")
    last_call = llm.calls[-1]
    # First message is system, then prior user/assistant pairs, then current user.
    roles = [m.role for m in last_call]
    assert roles[0] == "system"
    # The prior user message is in history
    assert any("Tell me about Plan B" in m.content for m in last_call if m.role == "user")
    assert last_call[-1].role == "user"
    assert last_call[-1].content == "What's the price?"


@pytest.mark.asyncio
async def test_persists_to_redis(retriever, fake_redis) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
    })
    store = SessionStore(fake_redis, ttl_seconds=300)
    agent = _make_agent(llm, retriever, store=store)
    await agent.handle_message("Plan B?")

    history = await store.get_history("cb-1")
    roles = [t["role"] for t in history]
    assert roles == ["user", "agent"]
    assert "Plan B" in history[0]["content"]
    state = await store.get_state("cb-1")
    assert state["agent_type"] == "chatbot"
    assert state["last_confidence"] == "high"
    assert state["turn_count"] == 1


@pytest.mark.asyncio
async def test_get_history_in_memory_when_no_store(retriever) -> None:
    llm = FakeLLM({
        "response_text": "Plan B has 500GB.",
        "language": "en",
        "sources_used": ["plans.pdf:2"],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("Plan B?")
    history = await agent.get_history()
    assert any(h["role"] == "user" for h in history)
    assert any(h["role"] == "assistant" for h in history)
