from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
import yaml

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent
from src.dialogue.context import SessionStore
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine


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
