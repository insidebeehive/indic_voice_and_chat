from __future__ import annotations

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
        self.closed = False

    async def send_audio(self, p):
        pass

    async def send_text(self, t):
        pass

    async def events(self):
        for e in self._events:
            yield e

    async def send_tool_response(self, *, tool_id, name, response):
        self.tool_responses.append((tool_id, name, response))

    async def aclose(self):
        self.closed = True


def _agent():
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema.from_campaign_yaml(
            {"interest_level": {"type": "enum", "values": ["hot", "warm", "cold"]}}),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=object(),
        store=None,
    )


def _bridge(events, llm=None):
    agent = _agent()
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
