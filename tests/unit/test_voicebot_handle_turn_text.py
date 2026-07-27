from __future__ import annotations

import asyncio

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import LLMConfig
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine, TurnMetrics, TurnResult


class _FakeEngine:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
        self.calls.append(user_text)
        return self._result


def _agent(engine):
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema(),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=engine,
        store=None,
    )


@pytest.mark.asyncio
async def test_active_language_switches_and_threads_to_engine():
    """The agent starts in the campaign default, switches when the LLM reports a
    new language, and passes the (BCP-47) active language to the engine each turn."""
    captured = {"langs": []}

    class _LangEngine:
        agent_text = '{"response_text": "namaste", "action": "continue", "language": "hi"}'

        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            captured["langs"].append(kw.get("language"))
            return TurnResult(
                user_text=user_text, user_language=None, user_confidence=1.0,
                agent_text=self.agent_text, audio_bytes_sent=1, metrics=TurnMetrics(),
            )

    engine = _LangEngine()
    agent = _agent(engine)
    await agent.start()

    async def sink(a):
        pass

    # Turn 1: default Hindi → engine gets hi-IN, stays hi.
    await agent.handle_turn_text("hello", sink)
    assert captured["langs"][0] == "hi-IN"
    assert agent.active_language == "hi"

    # Turn 2: the LLM now answers in Marathi → active language switches.
    engine.agent_text = '{"response_text": "namaskar", "action": "continue", "language": "mr"}'
    await agent.handle_turn_text("marathi madhe bola", sink)
    assert agent.active_language == "mr"

    # Turn 3: the switched language now reaches the engine.
    await agent.handle_turn_text("kasa aahes", sink)
    assert captured["langs"][-1] == "mr-IN"


@pytest.mark.asyncio
async def test_history_window_bounds_llm_context():
    """The LLM gets system prompt + last MAX_HISTORY_TURNS exchanges, not the
    whole transcript — so per-turn latency doesn't grow with call length. The
    full transcript still lives in session.turns."""
    from src.agents.voicebot import MAX_HISTORY_TURNS
    from src.interfaces.llm import LLMMessage

    captured = {}

    class _CaptureEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            captured["history"] = list(history)
            return TurnResult(
                user_text=user_text, user_language="hi", user_confidence=1.0,
                agent_text='{"response_text": "ok", "action": "continue"}',
                audio_bytes_sent=1, metrics=TurnMetrics(),
            )

    agent = _agent(_CaptureEngine())
    await agent.start()
    # Simulate a long call: system prompt (added in __init__) + 20 prior messages.
    for i in range(10):
        agent.session.turns.append(LLMMessage(role="user", content=f"u{i}"))
        agent.session.turns.append(LLMMessage(role="assistant", content=f"a{i}"))

    async def sink(a):
        pass

    await agent.handle_turn_text("latest", sink)

    hist = captured["history"]
    # system prompt + 2*MAX_HISTORY_TURNS recent messages.
    assert hist[0].role == "system"
    assert len(hist) == 1 + 2 * MAX_HISTORY_TURNS
    # Oldest recent message is u4 (turns 0-3 dropped), newest is a9.
    assert hist[1].content == f"u{10 - MAX_HISTORY_TURNS}"
    assert hist[-1].content == "a9"
    # Full transcript is preserved (system + 20 prior + new user + new assistant).
    assert len(agent.session.turns) == 1 + 20 + 2


@pytest.mark.asyncio
async def test_handle_turn_text_records_and_advances():
    result = TurnResult(
        user_text="और कुछ benefits हैं?",
        user_language="hi",
        user_confidence=1.0,
        agent_text='{"response_text": "जी हाँ!", "action": "continue", "updated_slots": {}}',
        audio_bytes_sent=10,
        metrics=TurnMetrics(),
    )
    engine = _FakeEngine(result)
    agent = _agent(engine)
    await agent.start()

    sink_calls = []

    async def sink(a):
        sink_calls.append(a)

    outcome = await agent.handle_turn_text("और कुछ benefits हैं?", sink)
    assert engine.calls == ["और कुछ benefits हैं?"]
    assert outcome.response.response_text == "जी हाँ!"
    roles = [t.role for t in agent.session.turns]
    assert roles[-2:] == ["user", "assistant"]
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_text_empty_is_noop():
    result = TurnResult(
        user_text="", user_language=None, user_confidence=0.0,
        agent_text="", audio_bytes_sent=0, metrics=TurnMetrics(),
    )
    agent = _agent(_FakeEngine(result))
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("", sink)
    assert outcome.response.parse_error == "empty STT"
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_text_recovers_on_provider_hang(monkeypatch):
    """A hung provider call that never checks cancel_event must still not wedge
    the agent forever: the backstop's hard-cap + grace period force-cancel it,
    landing in the same cancelled-turn recovery path as a barge-in."""
    import src.agents.voicebot as vb
    monkeypatch.setattr(vb, "HARD_TURN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(vb, "_BACKSTOP_GRACE_S", 0.05)

    class _HangingEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            await asyncio.sleep(5)  # never returns, never checks cancel_event
            raise AssertionError("should have been force-cancelled")

    agent = _agent(_HangingEngine())
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("कुछ", sink)
    assert agent.state.state is State.LISTENING
    assert outcome.response.parse_error == "barge-in"


@pytest.mark.asyncio
async def test_handle_turn_text_barge_in_drops_agent_reply():
    """A cancelled (barge-in) turn keeps the user turn but drops the agent
    reply and returns to LISTENING."""
    result = TurnResult(
        user_text="और कुछ benefits हैं?",
        user_language="hi",
        user_confidence=1.0,
        agent_text='{"response_text": "जी हाँ, सुन',  # partial / abandoned
        audio_bytes_sent=4,
        metrics=TurnMetrics(),
        cancelled=True,
    )
    engine = _FakeEngine(result)
    agent = _agent(engine)
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("और कुछ benefits हैं?", sink)
    assert outcome.response.parse_error == "barge-in"
    assert outcome.response.response_text == ""
    roles = [t.role for t in agent.session.turns]
    assert roles[-1] == "user"          # user turn kept, no trailing assistant turn
    assert agent.state.state is State.LISTENING


@pytest.mark.asyncio
async def test_handle_turn_text_passes_cancel_event_to_engine():
    captured = {}

    class _CaptureEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            captured["cancel_event"] = cancel_event
            return TurnResult("u", "hi", 1.0,
                              '{"response_text": "ok", "action": "continue"}', 0, TurnMetrics())

    agent = _agent(_CaptureEngine())
    await agent.start()
    sentinel = object()

    async def sink(a):
        pass

    await agent.handle_turn_text("hi", sink, cancel_event=sentinel)
    assert captured["cancel_event"] is sentinel


# --- apply_signal (shared by the cascade + the S2S Live bridge) -------------

def _agent_with_slots(schema, record_metric=None):
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=schema,
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=_FakeEngine(None),
        store=None,
        record_metric=record_metric,
    )


@pytest.mark.asyncio
async def test_apply_signal_records_transcript_slots_sentiment():
    from src.agents.state_machine import Event, State
    schema = SlotSchema.from_campaign_yaml(
        {"interest_level": {"type": "enum", "values": ["hot", "warm", "cold"]}})
    agent = _agent_with_slots(schema)
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)  # LISTENING -> PROCESSING
    applied = await agent.apply_signal(
        user_text="haan bhej do", agent_text="ज़रूर!",
        action="send_info", updated_slots={"interest_level": "hot"}, sentiment="positive")
    assert [(t.role, t.content) for t in agent.session.turns][-2:] == \
        [("user", "haan bhej do"), ("assistant", "ज़रूर!")]
    assert applied.get("interest_level") == "hot"
    assert agent.slots.values.get("interest_level") == "hot"
    assert agent.session.sentiment_history == ["positive"]
    assert agent.state.state is State.LISTENING  # send_info -> back to listening


@pytest.mark.asyncio
async def test_apply_signal_end_action_transitions_to_ended():
    from src.agents.state_machine import Event, State
    agent = _agent_with_slots(SlotSchema())
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)
    await agent.apply_signal(user_text="bye", agent_text="dhanyavaad jee", action="close_positive")
    assert agent.state.state is State.ENDED


@pytest.mark.asyncio
async def test_apply_signal_escalation_action_transitions_to_ended():
    from src.agents.state_machine import Event, State
    agent = _agent_with_slots(SlotSchema())
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)
    await agent.apply_signal(user_text="transfer me", agent_text="jee, transfer kar rahi hoon",
                             action="transfer")
    assert agent.state.state is State.ENDED


# --- apply_signal: structured per-turn metrics logging ----------------------


class _StubSTT:
    pass


class _StubLLM:
    pass


class _StubTTS:
    pass


@pytest.mark.asyncio
async def test_apply_signal_logs_metrics_when_metrics_dict_present(caplog):
    """A real metrics_dict (as passed by _finish_turn from the pipeline) must
    produce exactly one durable structured log line naming the provider
    classes and carrying the raw metrics — this is the only durable record
    once the Redis-persisted turn TTLs out."""
    from src.agents.state_machine import Event

    engine = _FakeEngine(None)
    engine._stt = _StubSTT()
    engine._llm = _StubLLM()
    engine._tts = _StubTTS()
    agent = VoiceBotAgent(
        session=AgentSession(session_id="t-metrics", campaign_id="camp-1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema(),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=engine,
        store=None,
    )
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)

    metrics_dict = {
        "stt_latency_ms": 120,
        "llm_ttft_ms": 300,
        "llm_total_ms": 900,
        "tts_first_chunk_ms": 150,
        "tts_total_ms": 700,
        "total_latency_ms": 1400,
    }
    with caplog.at_level("INFO", logger="src.agents.voicebot"):
        await agent.apply_signal(
            user_text="haan bhej do", agent_text="ज़रूर!",
            action="continue", metrics_dict=metrics_dict,
        )

    records = [r for r in caplog.records if r.getMessage() == "voice turn metrics"]
    assert len(records) == 1
    rec = records[0]
    assert rec.session_id == "t-metrics"
    assert rec.campaign_id == "camp-1"
    assert rec.stt_provider == "_StubSTT"
    assert rec.llm_provider == "_StubLLM"
    assert rec.tts_provider == "_StubTTS"
    assert rec.metrics == metrics_dict


@pytest.mark.asyncio
async def test_apply_signal_does_not_log_metrics_when_absent(caplog):
    """Callers that don't carry real STT/LLM/TTS timing (e.g. the S2S Live
    bridge's apply_signal call, which omits metrics_dict entirely) must not
    trigger the metrics log line on every turn."""
    from src.agents.state_machine import Event

    agent = _agent_with_slots(SlotSchema())
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)

    with caplog.at_level("INFO", logger="src.agents.voicebot"):
        await agent.apply_signal(
            user_text="bye", agent_text="dhanyavaad jee", action="continue",
        )

    records = [r for r in caplog.records if r.getMessage() == "voice turn metrics"]
    assert records == []


# --- apply_signal: optional record_metric callback --------------------------


@pytest.mark.asyncio
async def test_apply_signal_calls_record_metric_when_present():
    from src.agents.state_machine import Event

    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    agent = _agent_with_slots(SlotSchema(), record_metric=_record_metric)
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)

    metrics_dict = {
        "stt_latency_ms": 300, "llm_ttft_ms": 1200, "llm_total_ms": 4000,
        "tts_first_chunk_ms": 2000, "tts_total_ms": 2500, "total_latency_ms": 4300,
    }

    await agent.apply_signal(
        user_text="hi", agent_text="hello", action="continue",
        metrics_dict=metrics_dict,
    )

    assert len(calls) == 1
    payload = calls[0]
    assert payload["session_id"] == agent.session.session_id
    assert payload["mode"] == "layered"
    assert payload["action"] == "continue"
    assert payload["metrics"] == metrics_dict
    assert payload["llm_provider"]  # non-empty class name string


@pytest.mark.asyncio
async def test_apply_signal_record_metric_failure_does_not_break_turn():
    from src.agents.state_machine import Event

    async def _record_metric(payload):
        raise RuntimeError("db unavailable")

    agent = _agent_with_slots(SlotSchema(), record_metric=_record_metric)
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)

    metrics_dict = {
        "stt_latency_ms": 0, "llm_ttft_ms": 0, "llm_total_ms": 0,
        "tts_first_chunk_ms": 0, "tts_total_ms": 0, "total_latency_ms": 0,
    }

    # Must not raise.
    await agent.apply_signal(
        user_text="hi", agent_text="hello", action="continue",
        metrics_dict=metrics_dict,
    )


@pytest.mark.asyncio
async def test_apply_signal_no_record_metric_is_a_no_op():
    from src.agents.state_machine import Event

    agent = _agent_with_slots(SlotSchema(), record_metric=None)
    await agent.start()
    await agent.state.fire(Event.UTTERANCE_COMPLETE)

    metrics_dict = {
        "stt_latency_ms": 0, "llm_ttft_ms": 0, "llm_total_ms": 0,
        "tts_first_chunk_ms": 0, "tts_total_ms": 0, "total_latency_ms": 0,
    }
    # Must not raise (default behavior, unchanged from before this task).
    await agent.apply_signal(
        user_text="hi", agent_text="hello", action="continue",
        metrics_dict=metrics_dict,
    )


# --- campaign pronunciations threaded into TTS (fix A) ----------------------

@pytest.mark.asyncio
async def test_play_opening_passes_script_pronunciations_to_tts():
    captured = {}

    class _CapturingTTS:
        async def synthesize(self, text, config):
            captured["extra_pronunciations"] = config.extra_pronunciations
            return TTSResult(audio=b"\x00\x00", duration_ms=1.0, sample_rate=16000)

    engine = PipelineEngine(
        stt=None, llm=None, tts=_CapturingTTS(),
        config=PipelineConfig(
            stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(language="hi-IN"),
        ),
    )
    script = VoiceBotScript(
        agent_name="Priya", agent_role="sales", company_name="XYZ",
        opening="Hello from {company_name}",
        pronunciations={"XYZ": "एक्स वाय ज़ेड"},
    )
    agent = VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(), slot_schema=SlotSchema(),
        script=script, engine=engine, store=None,
    )

    async def sink(audio: bytes):
        pass

    await agent.play_opening(sink)
    assert captured["extra_pronunciations"] == {"XYZ": "एक्स वाय ज़ेड"}


@pytest.mark.asyncio
async def test_voicebot_agent_threads_script_pronunciations_onto_engine_config():
    engine = PipelineEngine(
        stt=None, llm=None, tts=None,
        config=PipelineConfig(
            stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(language="hi-IN"),
        ),
    )
    script = VoiceBotScript(
        agent_name="Priya", agent_role="sales", company_name="XYZ",
        pronunciations={"XYZ": "एक्स वाय ज़ेड"},
    )
    VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(), slot_schema=SlotSchema(),
        script=script, engine=engine, store=None,
    )
    assert engine._config.tts.extra_pronunciations == {"XYZ": "एक्स वाय ज़ेड"}


@pytest.mark.asyncio
async def test_voicebot_agent_leaves_engine_config_alone_when_no_pronunciations():
    original_tts = TTSConfig(language="hi-IN")
    engine = PipelineEngine(
        stt=None, llm=None, tts=None,
        config=PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=original_tts),
    )
    script = VoiceBotScript(agent_name="Priya", agent_role="sales", company_name="XYZ")
    VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(), slot_schema=SlotSchema(),
        script=script, engine=engine, store=None,
    )
    assert engine._config.tts is original_tts
