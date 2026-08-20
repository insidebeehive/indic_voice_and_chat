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


class QueuedLLM(ILLMProvider):
    """Returns each queued LLMResult in order, one per call; records calls."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.calls.append(list(messages))
        return self._results.pop(0) if self._results else LLMResult(text="ok", finish_reason="stop")

    async def generate_stream(self, messages, config) -> AsyncIterator[str]:
        if False:
            yield  # pragma: no cover


class FailOnSecondCallLLM(ILLMProvider):
    """First call returns the given result; every call after that raises —
    used to prove a failing/timing-out retry degrades gracefully rather than
    crashing the turn."""

    def __init__(self, first: LLMResult) -> None:
        self._first = first
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            return self._first
        raise RuntimeError("provider unavailable")

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


def _make_agent(llm, retriever, store=None, **kw) -> ChatBotAgent:
    return ChatBotAgent(
        session=AgentSession(session_id="cb-1"),
        llm=llm,
        retriever=retriever,
        company_name="Acme",
        language_default="en",
        store=store,
        **kw,
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
async def test_single_shot_reports_usage_and_llm_identity(retriever) -> None:
    """ChatTurnResult carries the token usage from the (single) generate()
    call plus the llm_provider/model the agent was constructed with — the
    identity src/api/chat_cost.py needs to look up a rate."""
    llm = QueuedLLM([
        LLMResult(text=json.dumps({
            "response_text": "Plan B has 500GB unlimited data.",
            "language": "en", "action": "none",
        }), finish_reason="stop", usage={"prompt_tokens": 120, "completion_tokens": 40}),
    ])
    agent = _make_agent(llm, retriever, llm_provider="gemini", llm_model="gemini-3.5-flash")
    result = await agent.handle_message("Tell me about Plan B")
    assert result.input_tokens == 120
    assert result.output_tokens == 40
    assert result.llm_provider == "gemini"
    assert result.llm_model == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_single_shot_sums_usage_across_retry(retriever) -> None:
    """A retried turn (empty first response) must sum usage from BOTH calls,
    not just the first or just the last."""
    llm = QueuedLLM([
        LLMResult(text="", finish_reason="stop", usage={"prompt_tokens": 50, "completion_tokens": 0}),
        LLMResult(text=json.dumps({
            "response_text": "Plan B has 500GB unlimited data.",
            "language": "en", "action": "none",
        }), finish_reason="stop", usage={"prompt_tokens": 55, "completion_tokens": 30}),
    ])
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Tell me about Plan B")
    assert len(llm.calls) == 2
    assert result.input_tokens == 105   # 50 + 55
    assert result.output_tokens == 30   # 0 + 30


@pytest.mark.asyncio
async def test_single_shot_usage_none_treated_as_zero(retriever) -> None:
    """A defensive case: an LLMResult with usage=None (e.g. an adapter that
    doesn't populate it) must not crash accumulation — treated as 0 tokens."""
    llm = QueuedLLM([
        LLMResult(text=json.dumps({
            "response_text": "ok", "language": "en", "action": "none",
        }), finish_reason="stop", usage=None),
    ])
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("hi")
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_handle_message_empty_input_returns_early(retriever) -> None:
    llm = FakeLLM({})
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("   ")
    assert result.response.parse_error == "empty user input"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_single_shot_retries_once_on_empty_llm_response(retriever) -> None:
    """An empty first response (e.g. a safety-filter block) must not be shown
    to the customer as the canned fallback if a retry would succeed — nothing
    has been displayed yet at this point, unlike voice's incremental TTS, so a
    plain retry is safe."""
    llm = QueuedLLM([
        LLMResult(text="", finish_reason="stop"),
        LLMResult(text=json.dumps({
            "response_text": "Plan B has 500GB unlimited data.",
            "language": "en", "action": "none",
        }), finish_reason="stop"),
    ])
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Tell me about Plan B")
    assert result.response.response_text == "Plan B has 500GB unlimited data."
    assert result.response.parse_error is None
    assert len(llm.calls) == 2  # the original call + one retry


@pytest.mark.asyncio
async def test_single_shot_falls_back_gracefully_when_retry_also_fails(retriever) -> None:
    llm = QueuedLLM([
        LLMResult(text="", finish_reason="stop"),
        LLMResult(text="", finish_reason="stop"),
    ])
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Tell me about Plan B")
    assert result.response.parse_error == "empty response"
    assert len(llm.calls) == 2  # the original call + exactly one retry, then give up


@pytest.mark.asyncio
async def test_single_shot_retries_on_truncated_malformed_json(retriever) -> None:
    """A truncated/malformed JSON envelope (e.g. max_tokens cutting a response
    mid-string) hits response_parser._fallback_text's THIRD branch — still
    looks like an unparsed JSON envelope — which returns the canned chatbot
    fallback line. This must trigger a retry same as an empty response;
    matching on parse_error strings alone used to miss this case entirely."""
    llm = QueuedLLM([
        LLMResult(
            text='{"response_text": "partial answer that got cut off mid',
            finish_reason="length",
        ),
        LLMResult(text=json.dumps({
            "response_text": "Plan B has 500GB unlimited data.",
            "language": "en", "action": "none",
        }), finish_reason="stop"),
    ])
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Tell me about Plan B")
    assert result.response.response_text == "Plan B has 500GB unlimited data."
    assert result.response.parse_error is None
    assert len(llm.calls) == 2  # the original call + one retry


@pytest.mark.asyncio
async def test_single_shot_retries_on_missing_response_text_field(retriever) -> None:
    """Valid JSON but no response_text field at all — the parse_error ==
    'missing response_text' trigger that existed before this round, now
    routed through is_unusable_response() instead of a direct string match."""
    llm = QueuedLLM([
        LLMResult(text=json.dumps({"language": "en"}), finish_reason="stop"),
        LLMResult(text=json.dumps({
            "response_text": "Plan B has 500GB unlimited data.",
            "language": "en", "action": "none",
        }), finish_reason="stop"),
    ])
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Tell me about Plan B")
    assert result.response.response_text == "Plan B has 500GB unlimited data."
    assert result.response.parse_error is None
    assert len(llm.calls) == 2  # the original call + one retry


@pytest.mark.asyncio
async def test_single_shot_retry_exception_degrades_to_original_fallback(retriever) -> None:
    """The retry call itself raising (e.g. a hard provider outage, or the
    retry's own bounded timeout) must not crash the turn — it should degrade
    to the original (pre-retry) fallback response."""
    llm = FailOnSecondCallLLM(LLMResult(text="", finish_reason="stop"))
    agent = _make_agent(llm, retriever)
    result = await agent.handle_message("Tell me about Plan B")
    assert result.response.parse_error == "empty response"
    assert len(llm.calls) == 2  # original call + the retry attempt that raised


@pytest.mark.asyncio
async def test_retry_if_unusable_does_not_crash_when_max_tokens_is_none(retriever) -> None:
    """_retry_if_unusable's docstring promises it never raises. The
    finish_reason == "length" branch bumps max_tokens via
    ``config.max_tokens * 1.5`` — config.max_tokens can legitimately be None
    (provider-default), which would previously raise TypeError from outside
    the try block, breaking that contract."""
    llm = FakeLLM({"response_text": "ok"})
    agent = _make_agent(llm, retriever)
    result, elapsed_ms = await agent._retry_if_unusable(
        "length", [LLMMessage(role="user", content="hi")],
        LLMConfig(max_tokens=None),
    )
    assert result is not None
    assert elapsed_ms >= 0


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
    # A bare "ok" mid-conversation carries no language signal — forcing a
    # language would flip an established conversation mid-stream. No
    # directive: history governs. Seed a real prior turn first (via an
    # actual handle_message call, the same way session.turns gets populated
    # in production) so this genuinely exercises the MID-conversation path,
    # not the opening-message one (which now gets a Hinglish-opener
    # directive — see test_opening_message_with_no_signal_gets_hinglish_directive).
    llm = FakeLLM({
        "response_text": "Your balance is 100.",
        "language": "en",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("tell me about this site")  # establishes English
    await agent.handle_message("ok")
    system_prompt = llm.calls[-1][0].content
    assert "MUST be in" not in system_prompt
    assert "romanized Hindi (Hinglish)" not in system_prompt
    assert "very first message" not in system_prompt


@pytest.mark.asyncio
async def test_opening_message_with_no_signal_gets_hinglish_directive(retriever) -> None:
    # A bare, ambiguous single word ("games") as the FIRST message of a fresh
    # session has no established conversation language to fall back on, and
    # the configured default (often Devanagari Hindi) risks alienating an
    # English-only user on their very first message. Default to Roman
    # Hinglish instead — readable by both English and Hindi/Hinglish
    # speakers — rather than the configured default language.
    llm = FakeLLM({
        "response_text": "Yaha kai games available hain!",
        "language": "hi",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("games")
    system_prompt = llm.calls[0][0].content
    assert "very first message" in system_prompt
    assert "Roman-script Hinglish" in system_prompt
    assert "MUST be in" not in system_prompt  # not the named-language branch


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


def test_has_operator_tools_computed_from_crm_tool_names(retriever) -> None:
    from src.interfaces.llm import ToolSpec

    agent = _make_agent(
        FakeLLM({"response_text": "ok", "language": "en", "confidence": "high", "action": "none"}),
        retriever,
        crm_tools=[ToolSpec(name="get_matka_config", description="", parameters={})],
    )
    messages = agent._compose("", LLMMessage(role="user", content="hi"), query_text="hi")
    system_prompt = messages[0].content
    assert "call the operator tool" in system_prompt


def test_has_operator_tools_false_when_no_operator_tools_registered(retriever) -> None:
    agent = _make_agent(
        FakeLLM({"response_text": "ok", "language": "en", "confidence": "high", "action": "none"}),
        retriever,
    )
    messages = agent._compose("", LLMMessage(role="user", content="hi"), query_text="hi")
    system_prompt = messages[0].content
    assert "no real-time operator lookup tools" in system_prompt


def test_tenant_timezone_threaded_into_prompt(retriever) -> None:
    agent = _make_agent(
        FakeLLM({"response_text": "ok", "language": "en", "confidence": "high", "action": "none"}),
        retriever,
        tenant_timezone="Asia/Kolkata",
    )
    messages = agent._compose("", LLMMessage(role="user", content="hi"), query_text="hi")
    assert "Asia/Kolkata" in messages[0].content


def test_has_player_tools_false_when_only_operator_tools_registered(retriever) -> None:
    # A Matka-only tenant registers ONLY an operator tool (get_matka_config is
    # in OPERATOR_TOOLS, not PLAYER_TOOLS). has_player_tools must be computed
    # from per-category membership, not bare `bool(self._crm_tools)` — the
    # latter would wrongly report True just because *some* tool is registered,
    # picking the "call the relevant tool" player_scope branch even though no
    # player tool exists for this tenant.
    from src.interfaces.llm import ToolSpec

    agent = _make_agent(
        FakeLLM({"response_text": "ok", "language": "en", "confidence": "high", "action": "none"}),
        retriever,
        crm_tools=[ToolSpec(name="get_matka_config", description="", parameters={})],
    )
    messages = agent._compose("", LLMMessage(role="user", content="hi"), query_text="hi")
    system_prompt = messages[0].content
    # player_scope must be the NO-tools branch: no PLAYER_TOOLS are registered.
    assert "you have no real-time lookup tools" in system_prompt
    # operator_scope must be the has-tools branch: get_matka_config IS an operator tool.
    assert "call the operator tool" in system_prompt


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
