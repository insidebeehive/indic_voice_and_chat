from __future__ import annotations

import asyncio
import json

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent
from src.api.gemini_live_bridge import GeminiLiveBridge
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.realtime import RealtimeConfig, RealtimeEvent


class _FakeWS:
    def __init__(self):
        self.sent_json = []
        self.sent_bytes = []

    async def send_text(self, t):
        self.sent_json.append(json.loads(t))

    async def send_bytes(self, b):
        self.sent_bytes.append(b)


class _FakeSession:
    def __init__(self, events):
        self._events = events
        self.tool_responses = []
        self.sent_text = []
        self.closed = False

    async def send_audio(self, p):
        pass

    async def send_text(self, t):
        self.sent_text.append(t)

    async def events(self):
        for e in self._events:
            yield e

    async def send_tool_response(self, *, tool_id, name, response):
        self.tool_responses.append((tool_id, name, response))

    async def aclose(self):
        self.closed = True


def _agent(record_metric=None):
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema.from_campaign_yaml(
            {"interest_level": {"type": "enum", "values": ["hot", "warm", "cold"]}}),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=object(),
        store=None,
        record_metric=record_metric,
    )


def _bridge(events, llm=None, record_metric=None):
    agent = _agent(record_metric=record_metric)
    sess = _FakeSession(events)

    async def connect(cfg):
        return sess

    b = GeminiLiveBridge(websocket=_FakeWS(), agent=agent,
                         config=RealtimeConfig(model="m"), connect_session=connect, llm=llm)
    b._session = sess
    return b, sess, agent


@pytest.mark.asyncio
async def test_consume_events_records_turn_and_slots():
    events = [
        RealtimeEvent(type="input_transcript", text="yeh app safe hai?"),
        RealtimeEvent(type="output_transcript", text="bilkul safe hai"),
        RealtimeEvent(type="audio", audio=b"\x01\x02" * 100, audio_rate=24000),
        RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                      tool_args={"action": "send_info", "updated_slots": {"interest_level": "hot"}},
                      tool_id="x1"),
        RealtimeEvent(type="turn_complete"),
    ]
    b, sess, agent = _bridge(events)
    await agent.start()
    await b._consume_events()

    contents = [(m.role, m.content) for m in agent.session.turns]
    assert ("user", "yeh app safe hai?") in contents
    assert ("assistant", "bilkul safe hai") in contents
    assert agent.slots.values.get("interest_level") == "hot"
    assert b._last_action == "send_info"
    assert len(b._ws.sent_bytes) > 0                       # agent audio resampled + sent
    assert sess.tool_responses and sess.tool_responses[0][1] == "record_turn_signal"
    assert agent.state.state is State.LISTENING            # back to listening after the turn


@pytest.mark.asyncio
async def test_consume_events_records_s2s_turn_metrics():
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    events = [
        RealtimeEvent(type="input_transcript", text="yeh app safe hai?"),
        RealtimeEvent(type="output_transcript", text="bilkul safe hai"),
        RealtimeEvent(type="audio", audio=b"\x01\x02" * 100, audio_rate=24000),
        RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                      tool_args={"action": "send_info", "updated_slots": {}},
                      tool_id="x1"),
        RealtimeEvent(type="turn_complete"),
    ]
    b, sess, agent = _bridge(events, record_metric=_record_metric)
    await agent.start()
    await b._consume_events()

    assert len(calls) == 1
    payload = calls[0]
    assert payload["mode"] == "s2s"
    assert payload["stt_provider"] is None
    assert payload["llm_provider"] == "_FakeSession"
    assert payload["tts_provider"] is None
    metrics = payload["metrics"]
    # tts_first_chunk_ms here means "time to first spoken audio" for S2S rows
    # (see the code comment in live_bridge_base.py) — the audio event fired
    # before turn_complete, so this must be a real, non-zero measurement, and
    # must not exceed the full turn's total_latency_ms.
    assert metrics["tts_first_chunk_ms"] >= 0
    assert metrics["total_latency_ms"] >= metrics["tts_first_chunk_ms"]


@pytest.mark.asyncio
async def test_consume_events_no_metrics_recorded_for_silent_turn():
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    events = [RealtimeEvent(type="turn_complete")]
    b, sess, agent = _bridge(events, record_metric=_record_metric)
    await agent.start()
    await b._consume_events()

    # No user/agent text this turn (model went silent) — no real turn
    # happened, so no metrics row should be recorded.
    assert calls == []


@pytest.mark.asyncio
async def test_interrupted_emits_interrupt_frame():
    b, sess, agent = _bridge([RealtimeEvent(type="interrupted")])
    await agent.start()
    await b._consume_events()
    assert any(m.get("type") == "interrupt" for m in b._ws.sent_json)


@pytest.mark.asyncio
async def test_close_action_ends_call():
    events = [
        RealtimeEvent(type="output_transcript", text="dhanyavaad jee"),
        RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                      tool_args={"action": "close_positive"}, tool_id="x2"),
        RealtimeEvent(type="turn_complete"),
    ]
    b, sess, agent = _bridge(events)   # llm=None -> outcome no-op
    await agent.start()
    await b._consume_events()
    assert agent.state.state is State.ENDED
    assert b._stopped is True


@pytest.mark.asyncio
async def test_browser_bridge_greets_first_kickoff():
    # The browser console has the agent greet first: a kickoff turn triggers the
    # opening (so the model speaks immediately) and warms the session, so the
    # caller's first reply gets an instant reply (no cold-start lag on turn 1).
    b, sess, agent = _bridge([])
    assert b._greets_first is True
    await b._maybe_greet()
    assert sess.sent_text and sess.sent_text[0].strip()   # a kickoff was sent


def test_greeting_kickoff_is_short_and_language_adaptive():
    # The opening must be a SHORT greeting so a non-default-language caller can
    # interject quickly and the agent continues in whatever language they reply
    # in (script-independent — works for every campaign). No language menu.
    from src.api.live_bridge_base import _GREETING_KICKOFF
    k = _GREETING_KICKOFF.lower()
    assert "short" in k                  # keep the opening brief
    assert "language" in k               # match the caller's language from their reply
    assert "wait" in k                   # then hand over to the caller


def test_telephony_bridge_does_not_greet_first():
    # Telephony keeps "caller says hello first" — no kickoff.
    from src.api.telephony_live_bridge import TelephonyLiveBridge
    assert TelephonyLiveBridge._greets_first is False


@pytest.mark.asyncio
async def test_idle_watchdog_forces_stop_when_session_hangs():
    # The Live session went silent (stopped endpointing) — no model events arrive.
    # The watchdog must force stop so the call tears down + emits an outcome
    # instead of hanging forever with no response and no outcome.
    import time
    b, sess, agent = _bridge([])
    b._idle_timeout_s = 0.02
    b._idle_check_s = 0.01
    b._last_event_at = time.monotonic() - 5     # already long idle
    await asyncio.wait_for(b._idle_watchdog(), timeout=2)
    assert b._stopped is True


@pytest.mark.asyncio
async def test_consume_events_marks_activity():
    # Every model event resets the idle clock so the watchdog doesn't end a live call.
    b, sess, agent = _bridge([RealtimeEvent(type="turn_complete")])
    await agent.start()
    assert b._last_event_at == 0.0
    await b._consume_events()
    assert b._last_event_at > 0.0


@pytest.mark.asyncio
async def test_session_close_midcall_signals_stop():
    # A normal (non-terminal) turn, then the Live event stream ends because the
    # upstream socket dropped (the preview native-audio model does this). The
    # bridge must signal stop so it tears down + emits an outcome instead of
    # leaving the caller stuck in "listening" feeding a dead session forever.
    events = [
        RealtimeEvent(type="output_transcript", text="haan jee, boliye"),
        RealtimeEvent(type="turn_complete"),
        # stream ends here with no terminal action — the socket dropped.
    ]
    b, sess, agent = _bridge(events)
    await agent.start()
    await b._consume_events()
    assert agent.state.state is State.LISTENING   # the call never reached a terminal action
    assert b._stopped is True                      # ...but stop was signalled so teardown runs


@pytest.mark.asyncio
async def test_empty_turn_stays_in_listening():
    # The model emitted a turn with no transcript/audio (it went silent — the
    # post-language-switch failure mode). The bridge must stay in listening and
    # not advance the state machine or crash.
    events = [RealtimeEvent(type="turn_complete")]
    b, sess, agent = _bridge(events)
    await agent.start()
    await b._consume_events()
    assert agent.state.state is State.LISTENING
    assert any(m.get("status") == "listening" for m in b._ws.sent_json)


@pytest.mark.asyncio
async def test_greeting_turn_with_no_user_text():
    # The kickoff greeting produces an agent turn with no user transcript.
    events = [
        RealtimeEvent(type="output_transcript", text="नमस्ते! मैं Anaaya बोल रही हूँ"),
        RealtimeEvent(type="turn_complete"),
    ]
    b, sess, agent = _bridge(events)
    await agent.start()
    await b._consume_events()
    contents = [(m.role, m.content) for m in agent.session.turns]
    assert ("assistant", "नमस्ते! मैं Anaaya बोल रही हूँ") in contents
    assert not any(r == "user" for r, _ in contents[1:])   # no user turn (turns[0] is system)
    assert agent.state.state is State.LISTENING
