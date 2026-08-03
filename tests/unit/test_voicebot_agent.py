from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import yaml

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent, _recover_fields_from_raw
from src.dialogue.context import SessionStore
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine, TurnMetrics, TurnResult


# --- Fakes (smaller versions of the pipeline test fakes) -----------------


class _FakeSTT(ISTTProvider):
    def __init__(self, text: str = "Aap kaise hain?") -> None:
        self._text = text

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        return STTResult(text=self._text, confidence=0.9, language="hi", raw_response={})

    async def transcribe_stream(self, audio_stream, config):
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi"]


class _FakeLLM(ILLMProvider):
    def __init__(self, payload: dict) -> None:
        self._json = json.dumps(payload)
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, config):
        self.calls.append(list(messages))
        return LLMResult(text=self._json, finish_reason="stop")

    async def generate_stream(self, messages, config):
        self.calls.append(list(messages))
        # Yield in 2 chunks to exercise streaming
        mid = len(self._json) // 2
        yield self._json[:mid]
        yield self._json[mid:]


class _RaisingLLM(ILLMProvider):
    """LLM whose stream blows up — simulates a provider outage (e.g. a 404)."""

    async def generate(self, messages, config):
        raise RuntimeError("LLM down")

    async def generate_stream(self, messages, config):
        raise RuntimeError("LLM 404: model retired")
        yield  # pragma: no cover - unreachable, makes this an async generator


class _TailCorruptedOnceLLM(ILLMProvider):
    """Streams valid JSON with a real, extractable ``response_text`` — but
    with the trailing metadata corrupted (a stray quote before the closing
    brace), reproducing the real Gemini JSON-mode quirk observed on Stage.
    Used to test that already-spoken text is reused instead of the canned
    fallback line."""

    def __init__(self, response_text: str) -> None:
        self.calls: list[list[LLMMessage]] = []
        self._response_text = response_text

    async def generate(self, messages, config):
        raise NotImplementedError

    async def generate_stream(self, messages, config):
        self.calls.append(list(messages))
        text = (
            '{\n  "response_text": "' + self._response_text + '",\n'
            '  "action": "continue",\n'
            '  "internal_notes": "some notes."\n'
            '"\n}'
        )
        mid = len(text) // 2
        yield text[:mid]
        yield text[mid:]


class _NoResponseTextThenValidLLM(ILLMProvider):
    """First call: no ``response_text`` field at all — nothing extractable or
    speakable. Second call (the retry): valid JSON. Used to test the
    retry-on-empty-output path."""

    def __init__(self, valid_payload: dict) -> None:
        self.calls: list[list[LLMMessage]] = []
        self._valid = json.dumps(valid_payload)

    async def generate(self, messages, config):
        raise NotImplementedError

    async def generate_stream(self, messages, config):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            yield "not json at all, garbled output with no fields"
        else:
            mid = len(self._valid) // 2
            yield self._valid[:mid]
            yield self._valid[mid:]


class _SlowFakeSTT(ISTTProvider):
    """Like _FakeSTT but with a small artificial delay so stt_latency_ms is
    measurably nonzero, and a configurable language/confidence — used to
    prove the retry path in _finish_turn carries over the ORIGINAL attempt's
    STT metrics/language/confidence (fix 4) rather than reporting the
    retry's own zeroed stt_latency_ms and a dropped user_language."""

    def __init__(self, text: str, language: str, confidence: float, delay_s: float = 0.02) -> None:
        self._text = text
        self._language = language
        self._confidence = confidence
        self._delay_s = delay_s

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        await asyncio.sleep(self._delay_s)
        return STTResult(
            text=self._text, confidence=self._confidence,
            language=self._language, raw_response={},
        )

    async def transcribe_stream(self, audio_stream, config):
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return [self._language]


class _TailCorruptedWithActionLLM(ILLMProvider):
    """Same corruption pattern as _TailCorruptedOnceLLM, but with a
    caller-supplied, non-default ``action`` (e.g. "close_positive") — used to
    prove fix 5 recovers the real action from the raw text instead of
    hard-overriding it to "continue" whenever the reuse-already-spoken-text
    branch fires."""

    def __init__(self, response_text: str, action: str) -> None:
        self.calls: list[list[LLMMessage]] = []
        self._response_text = response_text
        self._action = action

    async def generate(self, messages, config):
        raise NotImplementedError

    async def generate_stream(self, messages, config):
        self.calls.append(list(messages))
        text = (
            '{\n  "response_text": "' + self._response_text + '",\n'
            '  "action": "' + self._action + '",\n'
            '  "internal_notes": "some notes."\n'
            '"\n}'
        )
        mid = len(text) // 2
        yield text[:mid]
        yield text[mid:]


class _AlwaysGarbageLLM(ILLMProvider):
    """Always produces unparseable output with nothing extractable — used to
    confirm the graceful canned-fallback still applies after one failed retry."""

    def __init__(self) -> None:
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, config):
        raise NotImplementedError

    async def generate_stream(self, messages, config):
        self.calls.append(list(messages))
        yield "still not valid json, no fields at all"


class _FakeTTS(ITTSProvider):
    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self.configs: list[TTSConfig] = []

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        self.synthesized.append(text)
        self.configs.append(config)
        return TTSResult(audio=text.encode(), duration_ms=10.0, sample_rate=16000)

    async def synthesize_stream(self, text_stream, config):
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language: str):
        return []


# --- Fixtures ------------------------------------------------------------


SCRIPT_YAML = {
    "agent_name": "Priya",
    "agent_role": "Engagement",
    "company_name": "Acme",
    "language_default": "hi",
    "opening": "Namaste",
    "talking_points": ["Plan B"],
    "qualifying_questions": [],
    "objection_responses": {},
    "closing": {"positive": "Bye", "negative": "Bye"},
}

SLOT_YAML = """
lead_name:        { type: string,   required: true }
interest_level:   { type: enum,     required: true,  values: [hot, warm, cold] }
"""


def _make_engine(llm_payload: dict, stt_text: str = "Aap kaise hain?"):
    stt = _FakeSTT(text=stt_text)
    llm = _FakeLLM(llm_payload)
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    return engine, llm, tts


def _make_agent(engine, store=None) -> VoiceBotAgent:
    session = AgentSession(session_id="s1", lead_data={"lead_name": "Manoj"})
    sm = AgentStateMachine()
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    script = VoiceBotScript.from_campaign_yaml(SCRIPT_YAML)
    return VoiceBotAgent(
        session=session,
        state_machine=sm,
        slot_schema=schema,
        script=script,
        engine=engine,
        store=store,
    )


# --- Tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_opening_uses_configured_voice_not_provider_default() -> None:
    """The opening line must be spoken with the SELECTED/configured TTS voice
    (reusing the engine's TTS config), not a bare config that falls back to the
    provider default — otherwise the greeting is in the wrong voice."""
    stt = _FakeSTT(text="x")
    llm = _FakeLLM({"response_text": "x", "language": "hi", "action": "continue"})
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(),
                       tts=TTSConfig(voice_id="karun", language="hi-IN")),
    )
    agent = _make_agent(engine)

    async def sink(_audio):
        pass

    await agent.play_opening(sink)   # script opening is "Namaste"
    assert tts.synthesized == ["Namaste"]
    assert tts.configs[0].voice_id == "karun"        # the configured voice, not the default
    assert tts.configs[0].language == "hi-IN"        # active language preserved


@pytest.mark.asyncio
async def test_start_transitions_to_listening() -> None:
    engine, _, _ = _make_engine({"response_text": "x", "language": "hi", "action": "continue"})
    agent = _make_agent(engine)
    assert agent.state.state is State.IDLE
    await agent.start()
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_runs_pipeline_and_returns_to_listening() -> None:
    engine, llm, tts = _make_engine({
        "response_text": "Theek hoon.",
        "language": "hi",
        "action": "continue",
        "updated_slots": {"interest_level": "warm"},
    })
    agent = _make_agent(engine)
    await agent.start()

    sink_buf: list[bytes] = []

    async def sink(b: bytes) -> None:
        sink_buf.append(b)

    outcome = await agent.handle_turn(b"\x00\x00", sink)

    assert agent.state.state is State.LISTENING
    assert outcome.response.response_text == "Theek hoon."
    assert agent.slots.get("interest_level") == "warm"
    # System + user + assistant turns (the agent appended both)
    roles = [m.role for m in agent.session.turns]
    assert roles == ["system", "user", "assistant"]
    # Sentiment defaulted to neutral and was tracked
    assert agent.session.sentiment_history == ["neutral"]
    # TTS was driven
    assert tts.synthesized != []


@pytest.mark.asyncio
async def test_handle_turn_with_close_positive_terminates() -> None:
    engine, _, _ = _make_engine({
        "response_text": "Bahut accha! Dhanyavaad!",
        "language": "hi",
        "action": "close_positive",
    })
    agent = _make_agent(engine)
    await agent.start()
    await agent.handle_turn(b"\x00", _drop_sink)
    assert agent.state.state is State.ENDED
    assert agent.state.is_terminal


@pytest.mark.asyncio
async def test_handle_turn_with_transfer_action_escalates() -> None:
    engine, _, _ = _make_engine({
        "response_text": "Ek minute, transfer kar rahi hoon.",
        "language": "hi",
        "action": "transfer",
    })
    agent = _make_agent(engine)
    await agent.start()
    await agent.handle_turn(b"\x00", _drop_sink)
    # Escalation completes to ENDED (terminal) — the agent's role is done and
    # leaving it in ESCALATING would crash the next turn dispatch.
    assert agent.state.state is State.ENDED
    assert agent.state.is_terminal is True


@pytest.mark.asyncio
async def test_handle_turn_empty_stt_keeps_listening() -> None:
    engine, _, _ = _make_engine({}, stt_text="")
    agent = _make_agent(engine)
    await agent.start()
    out = await agent.handle_turn(b"\x00", _drop_sink)
    assert agent.state.state is State.LISTENING
    assert out.response.parse_error == "empty STT"


@pytest.mark.asyncio
async def test_handle_turn_called_outside_listening_raises() -> None:
    engine, _, _ = _make_engine({"response_text": "x", "language": "hi", "action": "continue"})
    agent = _make_agent(engine)
    with pytest.raises(RuntimeError):
        await agent.handle_turn(b"\x00", _drop_sink)  # still in IDLE


@pytest.mark.asyncio
async def test_handle_silence_timeout_advances_and_returns() -> None:
    engine, _, _ = _make_engine({"response_text": "x", "language": "hi", "action": "continue"})
    agent = _make_agent(engine)
    await agent.start()
    await agent.handle_silence_timeout(_drop_sink)
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_extended_silence_terminates() -> None:
    engine, _, _ = _make_engine({"response_text": "x", "language": "hi", "action": "continue"})
    agent = _make_agent(engine)
    await agent.start()
    await agent.handle_extended_silence()
    assert agent.state.state is State.ENDED


@pytest.mark.asyncio
async def test_handle_hangup_terminates_from_any_active_state() -> None:
    engine, _, _ = _make_engine({"response_text": "x", "language": "hi", "action": "continue"})
    agent = _make_agent(engine)
    await agent.start()
    await agent.handle_hangup()
    assert agent.state.state is State.ENDED


@pytest.mark.asyncio
async def test_session_store_persists_turns(fake_redis) -> None:
    store = SessionStore(fake_redis, ttl_seconds=300)
    engine, _, _ = _make_engine({
        "response_text": "Theek hoon.",
        "language": "hi",
        "action": "continue",
        "updated_slots": {"interest_level": "warm"},
    })
    agent = _make_agent(engine, store=store)
    await agent.start()
    await agent.handle_turn(b"\x00", _drop_sink)

    history = await store.get_history("s1")
    roles = [t["role"] for t in history]
    assert "user" in roles
    assert "agent" in roles
    state = await store.get_state("s1")
    assert state["state"] == "listening"
    assert state["slots"]["interest_level"] == "warm"


async def _drop_sink(b: bytes) -> None:
    pass


@pytest.mark.asyncio
async def test_handle_turn_recovers_from_pipeline_error() -> None:
    """A provider failure mid-turn must NOT propagate (which would drop the
    call). The agent recovers to LISTENING and reports the error."""
    stt = _FakeSTT(text="haan ji ek minute hai")
    engine = PipelineEngine(
        stt, _RaisingLLM(), _FakeTTS(),
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    outcome = await agent.handle_turn(b"\x00\x00", _drop_sink)

    # Did not raise; call survives and is ready for the next turn.
    assert agent.state.state is State.LISTENING
    assert outcome.response.response_text == ""
    assert outcome.response.parse_error and "error" in outcome.response.parse_error.lower()

    # A second failing turn also recovers (the call keeps going).
    await agent.handle_turn(b"\x00\x00", _drop_sink)
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_reuses_already_spoken_text_on_tail_corrupted_json() -> None:
    """response_text was already extracted and spoken (sentence-by-sentence)
    before the JSON's corrupted tail surfaces as a parse failure — the
    already-spoken text must be reused instead of the canned fallback, and
    the turn must NOT retry (audio has already gone out)."""
    stt = _FakeSTT(text="haan ji bataiye")
    llm = _TailCorruptedOnceLLM("Bahut accha, dhanyavaad aapke samay ke liye!")
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    outcome = await agent.handle_turn(b"\x00\x00", _drop_sink)

    assert outcome.response.parse_error  # still recorded, for logging
    assert outcome.response.response_text == "Bahut accha, dhanyavaad aapke samay ke liye!"
    assert outcome.response.action == "continue"
    assert agent.state.state is State.LISTENING
    assert len(llm.calls) == 1  # no retry — something was already spoken
    assert tts.synthesized != []


@pytest.mark.asyncio
async def test_handle_turn_retries_once_when_nothing_was_spoken() -> None:
    """No response_text was extractable at all — nothing has been spoken yet,
    so it's safe to regenerate once. The retry succeeds here."""
    stt = _FakeSTT(text="haan ji bataiye")
    llm = _NoResponseTextThenValidLLM({
        "response_text": "Theek hai, bataata hoon.",
        "language": "hi",
        "action": "continue",
    })
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    outcome = await agent.handle_turn(b"\x00\x00", _drop_sink)

    assert outcome.response.response_text == "Theek hai, bataata hoon."
    assert outcome.response.action == "continue"
    assert outcome.response.parse_error is None
    assert agent.state.state is State.LISTENING
    assert len(llm.calls) == 2  # the original call + one retry
    # The retry reused the already-transcribed text, not a re-run of STT.
    assert llm.calls[1][-1].content == "haan ji bataiye"


@pytest.mark.asyncio
async def test_handle_turn_falls_back_gracefully_when_retry_also_fails() -> None:
    """Nothing spoken on the first attempt or the retry — falls back to the
    canned line exactly as before this change, after trying once."""
    stt = _FakeSTT(text="haan ji bataiye")
    llm = _AlwaysGarbageLLM()
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    outcome = await agent.handle_turn(b"\x00\x00", _drop_sink)

    assert outcome.response.parse_error
    assert outcome.response.action == "clarify"
    assert agent.state.state is State.LISTENING
    assert len(llm.calls) == 2  # the original call + exactly one retry, then give up


@pytest.mark.asyncio
async def test_handle_turn_retry_cancelled_but_partially_spoken_is_reused() -> None:
    """If the RETRY itself gets cancelled partway through (LLM budget
    exceeded, barge-in, or MAX_CONSECUTIVE_TTS_FAILURES — all set
    cancelled=True in engine.py) but real content was already spoken before
    the cancellation fired, that content must be reused — not discarded for
    the canned fallback, which would recreate the exact transcript/audio
    divergence bug this whole recovery path exists to fix, just relocated to
    the retry.

    But cancellation of a retry can be caused by a barge-in — the caller
    interrupting mid-reply, never hearing the full JSON envelope (including
    its `action`) announced. This codebase has a deliberate policy elsewhere
    (the two parse_error="barge-in" sites in handle_turn/handle_turn_text) of
    forcing action="continue" whenever the caller interrupted before hearing
    the full reply, precisely so a terminal decision (close_positive/
    transfer) the caller never actually heard isn't acted on. This branch
    must follow the same policy: even though the raw, truncated JSON still
    has an intact `action: close_positive`, that must NOT be recovered/acted
    on here — the state machine must NOT terminate or escalate."""
    stt = _FakeSTT(text="haan ji bataiye")
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, _AlwaysGarbageLLM(), tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    # The original attempt: nothing extractable, nothing spoken, not cancelled
    # — this is what makes _finish_turn take the retry branch.
    first_attempt = TurnResult(
        user_text="haan ji bataiye",
        user_language="hi",
        user_confidence=0.9,
        agent_text="garbled, unparseable, nothing extractable",
        audio_bytes_sent=0,
        metrics=TurnMetrics(),
        cancelled=False,
        sentences_spoken=[],
    )
    # The retry: got cancelled partway through (simulating e.g. a
    # MAX_CONSECUTIVE_TTS_FAILURES abort), but one real sentence was already
    # spoken before that happened, and the raw (truncated) JSON still has an
    # intact `action` field.
    retried_but_cancelled = TurnResult(
        user_text="haan ji bataiye",
        user_language="hi",
        user_confidence=0.9,
        agent_text='{"response_text": "Kuch bol diya maine.", "action": "close_positive"',
        audio_bytes_sent=10,
        metrics=TurnMetrics(),
        cancelled=True,
        sentences_spoken=["Kuch bol diya maine."],
    )
    # PipelineEngine.run_turn delegates to run_turn_text internally, so
    # mocking run_turn_text at the instance level intercepts BOTH the
    # original attempt (via run_turn) and the direct retry call in
    # _finish_turn, in that order.
    engine.run_turn_text = AsyncMock(side_effect=[first_attempt, retried_but_cancelled])

    outcome = await agent.handle_turn(b"\x00\x00", _drop_sink)

    assert engine.run_turn_text.await_count == 2  # original (via run_turn) + one retry
    assert outcome.response.response_text == "Kuch bol diya maine."
    # Forced to "continue" — NOT recovered from the raw JSON — because the
    # cancellation means the caller may never have heard the full reply.
    assert outcome.response.action == "continue"
    # Must NOT terminate or escalate on an action the caller never heard.
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_retry_carries_over_original_stt_metrics_and_language() -> None:
    """Fix 4: after a successful retry, stt_latency_ms/user_language/
    user_confidence must be carried over from the ORIGINAL attempt (the
    retry itself skips STT and would otherwise report stt_latency_ms=0 and
    drop a real language-switch signal STT detected on the original turn)."""
    stt = _SlowFakeSTT("haan ji bataiye", language="mr", confidence=0.87, delay_s=0.02)
    llm = _NoResponseTextThenValidLLM({
        "response_text": "Theek hai, bataata hoon.",
        "language": "hi",
        "action": "continue",
    })
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    outcome = await agent.handle_turn(b"\x00\x00", _drop_sink)

    assert len(llm.calls) == 2  # the original call + one retry
    assert outcome.response.response_text == "Theek hai, bataata hoon."
    # Carried over from the ORIGINAL attempt, not zeroed/dropped by the retry.
    assert outcome.pipeline.metrics.stt_latency_ms > 0
    assert outcome.pipeline.user_language == "mr"
    assert outcome.pipeline.user_confidence == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_handle_turn_reuse_branch_recovers_real_action_not_hardcoded_continue() -> None:
    """Fix 5: when the reuse-already-spoken-text branch fires (the initial
    parse failure, not a retry), a real action (e.g. close_positive) still
    intact in the raw text must be recovered and used instead of
    unconditionally hard-overriding action to "continue" — otherwise a real
    close/transfer decision from the model is silently dropped."""
    stt = _FakeSTT(text="haan ji bataiye")
    llm = _TailCorruptedWithActionLLM(
        "Bahut accha, dhanyavaad aapke samay ke liye!", action="close_positive"
    )
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    outcome = await agent.handle_turn(b"\x00\x00", _drop_sink)

    assert outcome.response.parse_error  # still recorded, for logging
    assert outcome.response.response_text == "Bahut accha, dhanyavaad aapke samay ke liye!"
    assert outcome.response.action == "close_positive"  # recovered, not hardcoded "continue"
    assert len(llm.calls) == 1  # no retry — something was already spoken
    # A close_positive action must actually end the call, not be discarded in
    # favour of a forced "continue" that would keep it alive indefinitely.
    assert agent.state.state is State.ENDED


def test_recover_fields_from_raw_ignores_action_sentiment_shadowed_by_slot_keys() -> None:
    """Fix 2: the real envelope's field order is response_text, language,
    conversation_phase, updated_slots, action, action_reason, sentiment,
    internal_notes — updated_slots comes BEFORE action/sentiment. A plain
    first-match .search() over the whole raw text would incorrectly match a
    slot key literally named "action"/"sentiment" (e.g. a badly-named
    campaign slot) whose value happens to be a valid enum member, instead of
    the REAL action/sentiment fields that appear later in the envelope. The
    recovery must skip past the updated_slots object and only match fields
    after it."""
    raw = (
        '{"response_text":"hi","updated_slots":{"a":"b","action":"transfer",'
        '"sentiment":"frustrated"},"action":"close_negative","sentiment":"positive"}'
    )
    recovered = _recover_fields_from_raw(raw)
    assert recovered["action"] == "close_negative"
    assert recovered["sentiment"] == "positive"
    assert recovered["updated_slots"] == {
        "a": "b", "action": "transfer", "sentiment": "frustrated",
    }


def test_recover_fields_from_raw_still_recovers_conversation_phase_before_slots() -> None:
    """Regression guard: conversation_phase comes BEFORE updated_slots in the
    real envelope, so — unlike action/sentiment — its search must NOT be
    restricted to text after updated_slots (that would break recovering the
    real field). A realistic envelope with conversation_phase before
    updated_slots, and a corrupted/truncated tail, must still recover the
    phase correctly."""
    raw = (
        '{"response_text":"Theek hai","language":"hi",'
        '"conversation_phase":"qualification","updated_slots":{"interest_level":"hot"},'
        '"action":"continue"'
    )
    recovered = _recover_fields_from_raw(raw)
    assert recovered["conversation_phase"] == "qualification"
    assert recovered["action"] == "continue"
    assert recovered["updated_slots"] == {"interest_level": "hot"}


def test_recover_fields_from_raw_handles_real_prompt_field_order() -> None:
    """The field order the model is ACTUALLY instructed to emit (see
    build_voicebot_system_prompt's field spec in prompts.py) is response_text,
    language, action, conversation_phase, sentiment, updated_slots,
    action_reason, internal_notes — action/sentiment/conversation_phase all
    come BEFORE updated_slots here, the opposite of VoiceBotResponse's
    dataclass field order. Recovery must not depend on which side of
    updated_slots the real fields land on: it must recover the REAL fields
    here too, not just in the dataclass-order shape the other tests use."""
    raw = (
        '{"response_text":"hi","language":"hi","action":"close_positive",'
        '"conversation_phase":"closing","sentiment":"positive",'
        '"updated_slots":{"a":"b","action":"transfer","sentiment":"frustrated"}'
    )
    recovered = _recover_fields_from_raw(raw)
    assert recovered["action"] == "close_positive"
    assert recovered["sentiment"] == "positive"
    assert recovered["conversation_phase"] == "closing"
    assert recovered["updated_slots"] == {
        "a": "b", "action": "transfer", "sentiment": "frustrated",
    }


def test_find_updated_slots_span_ignores_braces_inside_slot_string_values() -> None:
    """A literal '}' inside a slot's string value must not be miscounted as
    the object's closing brace — that would truncate the span early, silently
    drop updated_slots recovery, AND reopen the decoy-shadowing hole by
    leaving a bogus "action"/"sentiment" key outside the excised span."""
    raw = (
        '{"response_text":"x","updated_slots":{"note":"we discussed } stuff",'
        '"action":"end"},"action":"clarify","sentiment":"positive"}'
    )
    recovered = _recover_fields_from_raw(raw)
    assert recovered["action"] == "clarify"
    assert recovered["sentiment"] == "positive"
    assert recovered["updated_slots"] == {
        "note": "we discussed } stuff", "action": "end",
    }


@pytest.mark.asyncio
async def test_handle_turn_retry_reuses_same_cancel_event_object() -> None:
    """Regression guard (unaddressed in the prior review round): the retry
    call dispatched from _finish_turn must reuse the turn's ORIGINAL
    cancel_event object — not a fresh one — so a barge-in during the retry
    can still interrupt it, and so the backstop's own expiry mechanism
    (cancel_event.set()) keeps working."""
    stt = _FakeSTT(text="haan ji bataiye")
    tts = _FakeTTS()
    engine = PipelineEngine(
        stt, _AlwaysGarbageLLM(), tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )
    agent = _make_agent(engine)
    await agent.start()

    first_attempt = TurnResult(
        user_text="haan ji bataiye",
        user_language="hi",
        user_confidence=0.9,
        agent_text="garbled, unparseable, nothing extractable",
        audio_bytes_sent=0,
        metrics=TurnMetrics(),
        cancelled=False,
        sentences_spoken=[],
    )
    retried = TurnResult(
        user_text="haan ji bataiye",
        user_language="hi",
        user_confidence=0.9,
        agent_text='{"response_text": "Theek hai.", "action": "continue"}',
        audio_bytes_sent=10,
        metrics=TurnMetrics(),
        cancelled=False,
        sentences_spoken=["Theek hai."],
    )
    # PipelineEngine.run_turn delegates to run_turn_text internally, so
    # mocking run_turn_text at the instance level intercepts BOTH the
    # original attempt (via run_turn) and the direct retry call in
    # _finish_turn, in that order.
    engine.run_turn_text = AsyncMock(side_effect=[first_attempt, retried])

    await agent.handle_turn(b"\x00\x00", _drop_sink)

    assert engine.run_turn_text.await_count == 2  # original (via run_turn) + one retry
    # cancel_event is the 4th positional arg (after user_text, history,
    # audio_sink) in both run_turn's delegation call and _finish_turn's
    # direct retry call — see PipelineEngine.run_turn_text's signature.
    original_cancel_event = engine.run_turn_text.await_args_list[0].args[3]
    retry_cancel_event = engine.run_turn_text.await_args_list[1].args[3]
    assert retry_cancel_event is original_cancel_event
