from __future__ import annotations

import asyncio
import base64
import json

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent
from src.api.telephony_live_bridge import TelephonyLiveBridge
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.realtime import RealtimeConfig, RealtimeEvent


class _FakeWS:
    def __init__(self):
        self.sent_text = []

    async def send_text(self, t):
        self.sent_text.append(t)


class _FakeSession:
    def __init__(self, events=()):
        self._events = list(events)
        self.audio = []
        self.tool_responses = []

    async def send_audio(self, pcm):
        self.audio.append(pcm)

    async def events(self):
        for e in self._events:
            yield e

    async def send_tool_response(self, *, tool_id, name, response):
        self.tool_responses.append((tool_id, name, response))

    async def aclose(self):
        pass


def _agent():
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema.from_campaign_yaml(
            {"interest_level": {"type": "enum", "values": ["hot", "warm", "cold"]}}),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=object(), store=None)


def _bridge(encoding="mulaw", events=(), llm=None):
    agent = _agent()
    sess = _FakeSession(events)

    async def connect(cfg):
        return sess

    b = TelephonyLiveBridge(
        websocket=_FakeWS(), agent=agent, config=RealtimeConfig(model="m"),
        connect_session=connect, encoding=encoding,
        sid_field="streamSid" if encoding == "mulaw" else "stream_sid",
        supports_clear=(encoding == "mulaw"), llm=llm)
    b._session = sess
    return b, sess, agent


def _media(text_list):
    return [json.loads(t) for t in text_list]


@pytest.mark.asyncio
async def test_inbound_mulaw_decoded_and_forwarded():
    b, sess, _ = _bridge("mulaw")
    b._stream_sid = "S1"
    mulaw = b"\xff" * 160                       # 20ms μ-law
    await b._on_media({"track": "inbound", "payload": base64.b64encode(mulaw).decode()})
    assert sess.audio and len(sess.audio[0]) > 320   # decoded + upsampled 8k->16k


@pytest.mark.asyncio
async def test_inbound_pcm_forwarded_exotel():
    b, sess, _ = _bridge("pcm")
    b._stream_sid = "S1"
    pcm = b"\x00\x01" * 160                      # 320 bytes = 20ms PCM16@8k
    await b._on_media({"payload": base64.b64encode(pcm).decode()})
    assert sess.audio and len(sess.audio[0]) > len(pcm)   # upsampled 8k->16k


@pytest.mark.asyncio
async def test_outbound_dropped_before_stream_sid():
    b, _, _ = _bridge("mulaw")                   # no stream_sid yet
    await b._send_audio_out(b"\x00\x01" * 480, 24000)
    assert b._audio_q.empty()


@pytest.mark.asyncio
async def test_outbound_audio_paced_to_media_frames():
    b, _, _ = _bridge("mulaw")
    b._stream_sid = "S1"
    await b._send_audio_out(b"\x00\x01" * 480, 24000)   # 24k -> 8k, enqueued
    assert not b._audio_q.empty()
    task = asyncio.create_task(b._sender_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    frames = _media(b._ws.sent_text)
    assert frames and frames[0]["event"] == "media"
    assert frames[0]["streamSid"] == "S1"
    assert len(base64.b64decode(frames[0]["media"]["payload"])) <= 160  # μ-law chunk


@pytest.mark.asyncio
async def test_interrupt_clears_queue_and_sends_clear():
    b, _, _ = _bridge("mulaw")
    b._stream_sid = "S1"
    b._audio_q.put_nowait(b"x" * 160)
    await b._send_interrupt()
    assert b._audio_q.empty()
    clears = [m for m in _media(b._ws.sent_text) if m.get("event") == "clear"]
    assert clears and clears[0]["streamSid"] == "S1"


@pytest.mark.asyncio
async def test_exotel_interrupt_no_clear():
    b, _, _ = _bridge("pcm")                     # supports_clear=False
    b._stream_sid = "S1"
    b._audio_q.put_nowait(b"x" * 320)
    await b._send_interrupt()
    assert b._audio_q.empty()                    # queue still drained
    assert not any(m.get("event") == "clear" for m in _media(b._ws.sent_text))


def _one_frame_then_disconnect(frame: dict):
    """A fake receive_text that yields one frame then ends the WS (caller hangs up)."""
    from starlette.websockets import WebSocketDisconnect

    state = {"n": 0}

    async def _receive():
        state["n"] += 1
        if state["n"] == 1:
            return json.dumps(frame)
        raise WebSocketDisconnect()

    return _receive


@pytest.mark.asyncio
async def test_start_event_captures_call_sid_and_marks_answered():
    from src.api import dev_call_control

    b, _, _ = _bridge("mulaw")
    b._ws.receive_text = _one_frame_then_disconnect(
        {"event": "start", "start": {"streamSid": "S1", "callSid": "CAabc"}})
    await b._inbound_loop()
    assert b._call_sid == "CAabc"
    assert dev_call_control.monitor.get("CAabc")["status"] == "answered"


@pytest.mark.asyncio
async def test_teardown_and_outcome_publish_to_monitor():
    from src.api import dev_call_control

    b, _, _ = _bridge("mulaw")
    b._call_sid = "CAxyz"
    await b._on_teardown()
    assert dev_call_control.monitor.get("CAxyz")["status"] == "ended"
    await b._deliver_outcome({"outcome": "interested", "summary": "s"})
    got = dev_call_control.monitor.get("CAxyz")
    assert got["status"] == "ended"
    assert got["outcome"]["outcome"] == "interested"


@pytest.mark.asyncio
async def test_deliver_outcome_merges_turns_for_persister_without_mutating_payload():
    """Pins change #4: the persister gets turns merged into a NEW dict, while
    the payload object handed to the monitor (stored by reference, later
    JSON-serialized by GET /dev/call-status/{call_sid}) is left untouched."""
    from src.api import call_store, dev_call_control
    from src.interfaces.llm import LLMMessage

    b, sess, agent = _bridge("mulaw")
    b._call_sid = "CA-MERGE"
    agent.session.turns.extend([
        LLMMessage(role="user", content="yeh app safe hai?"),
        LLMMessage(role="assistant", content="bilkul safe hai"),
    ])
    payload = {"type": "outcome", "outcome": "interested", "summary": "s"}

    delivered = {}

    async def _persister(call_sid, p):
        delivered["call_sid"] = call_sid
        delivered["payload"] = p

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._deliver_outcome(payload)
    finally:
        call_store.set_call_outcome_persister(None)

    # Persister got the turns (session.turns always leads with a system prompt;
    # only the conversational turns we added matter here).
    assert delivered["call_sid"] == "CA-MERGE"
    turns = delivered["payload"]["turns"]
    assert [(t.role, t.content) for t in turns if t.role != "system"] == [
        ("user", "yeh app safe hai?"), ("assistant", "bilkul safe hai")]

    # The original payload object (also handed to the monitor) is untouched.
    assert "turns" not in payload
    stored = dev_call_control.monitor.get("CA-MERGE")["outcome"]
    assert "turns" not in stored
    assert stored is payload   # same object — confirms no copy-then-mutate either


@pytest.mark.asyncio
async def test_deliver_outcome_failed_stamps_notes_for_persister_not_monitor():
    """Fix #3: an outcome_failed delivery must be distinguishable, downstream in
    the conversations row / tenant webhook, from a genuine no-outcome call —
    the persister payload gets notes="outcome analysis failed" stamped in.
    This must not leak into the monitor's stored copy (browser-facing), and
    must not override notes already present on a normal "outcome" payload."""
    from src.api import call_store, dev_call_control

    b, sess, agent = _bridge("mulaw")
    b._call_sid = "CA-NOTES-FAIL"

    delivered = {}

    async def _persister(call_sid, p):
        delivered["payload"] = p

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._deliver_outcome({"type": "outcome_failed"})
    finally:
        call_store.set_call_outcome_persister(None)

    assert delivered["payload"]["notes"] == "outcome analysis failed"
    # The monitor copy (served to the browser) is the original, unstamped payload.
    stored = dev_call_control.monitor.get("CA-NOTES-FAIL")["outcome"]
    assert "notes" not in stored


@pytest.mark.asyncio
async def test_deliver_outcome_success_notes_untouched():
    """Fix #3 must not change behavior for the normal "outcome" payload type —
    an existing notes value (or its absence) passes through unstamped."""
    from src.api import call_store

    b, sess, agent = _bridge("mulaw")
    b._call_sid = "CA-NOTES-OK"

    delivered = {}

    async def _persister(call_sid, p):
        delivered["payload"] = p

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._deliver_outcome({"type": "outcome", "outcome": "interested",
                                   "summary": "s", "notes": "genuine notes"})
    finally:
        call_store.set_call_outcome_persister(None)

    assert delivered["payload"]["notes"] == "genuine notes"


@pytest.mark.asyncio
async def test_emit_outcome_success_delivers_turns_to_persister():
    """Change #2/#3 end-to-end: a successful analysis still gets the transcript
    to the persister via TelephonyLiveBridge._deliver_outcome, and the shared
    _emit_outcome payload itself carries no turns key (change #5)."""
    from src.api import call_store, dev_call_control
    from src.interfaces.llm import LLMMessage, LLMResult

    class _FakeLLM:
        async def generate(self, messages, config):
            return LLMResult(
                text=('{"outcome": "interested", "summary": "Wants info", '
                      '"notes": "n", "callback_datetime": null, "callback_phrase": null}'),
                finish_reason="stop")

    b, sess, agent = _bridge("mulaw", llm=_FakeLLM())
    b._call_sid = "CA-SUCCESS"
    agent.session.turns.extend([
        LLMMessage(role="user", content="yeh app safe hai?"),
        LLMMessage(role="assistant", content="bilkul safe hai"),
    ])

    delivered = {}

    async def _persister(call_sid, payload):
        delivered["call_sid"] = call_sid
        delivered["payload"] = payload

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._emit_outcome()
    finally:
        call_store.set_call_outcome_persister(None)

    assert delivered["call_sid"] == "CA-SUCCESS"
    assert delivered["payload"]["outcome"] == "interested"
    turns = delivered["payload"]["turns"]
    assert [(t.role, t.content) for t in turns if t.role != "system"] == [
        ("user", "yeh app safe hai?"), ("assistant", "bilkul safe hai")]

    # Non-mutation pinned again via the full _emit_outcome path.
    stored = dev_call_control.monitor.get("CA-SUCCESS")["outcome"]
    assert "turns" not in stored
    assert stored["outcome"] == "interested"


@pytest.mark.asyncio
async def test_emit_outcome_analysis_failure_still_delivers_turns(monkeypatch):
    """Change #6: when analyze_call raises, the transcript must still reach the
    persister instead of silently being lost alongside the outcome analysis."""
    from src.api import call_store
    from src.api import live_bridge_base
    from src.interfaces.llm import LLMMessage

    async def _boom(**kwargs):
        raise RuntimeError("llm quota exceeded")

    monkeypatch.setattr(live_bridge_base, "analyze_call", _boom)

    b, sess, agent = _bridge("mulaw", llm=object())   # just needs to be non-None
    b._call_sid = "CA-FAIL"
    agent.session.turns.extend([
        LLMMessage(role="user", content="hello"),
        LLMMessage(role="assistant", content="hi there"),
    ])

    delivered = {}

    async def _persister(call_sid, payload):
        delivered["call_sid"] = call_sid
        delivered["payload"] = payload

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._emit_outcome()
    finally:
        call_store.set_call_outcome_persister(None)

    assert delivered["call_sid"] == "CA-FAIL"
    assert delivered["payload"]["type"] == "outcome_failed"
    turns = delivered["payload"]["turns"]
    assert [(t.role, t.content) for t in turns if t.role != "system"] == [
        ("user", "hello"), ("assistant", "hi there")]

    # The monitor-stored copy (served verbatim by GET /dev/call-status) must
    # stay turn-free even on the failure path — pins the non-mutation fix
    # (change #4) for this path too, not just the success path.
    from src.api import dev_call_control
    stored = dev_call_control.monitor.get("CA-FAIL")["outcome"]
    assert "turns" not in stored


@pytest.mark.asyncio
async def test_call_status_endpoint_serializes_after_analysis_failure_outcome():
    """GET /dev/call-status/{call_sid} must still serialize successfully after
    an outcome_failed payload (the fixed _emit_outcome except-path shape:
    {"type": "outcome_failed"}, no "turns" key) has been delivered — catches a
    FastAPI jsonable_encoder regression directly rather than only asserting on
    the stored dict, and exercises the actual failure path instead of a
    success-shaped payload that never contained turns to begin with."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.dev_console import dev_router
    from src.interfaces.llm import LLMMessage

    b, sess, agent = _bridge("mulaw")
    b._call_sid = "CA-STATUS"
    agent.session.turns.extend([
        LLMMessage(role="user", content="hello"),
        LLMMessage(role="assistant", content="hi"),
    ])
    # Matches the fixed live_bridge_base._emit_outcome except-path payload.
    await b._deliver_outcome({"type": "outcome_failed"})

    app = FastAPI()
    app.include_router(dev_router)
    client = TestClient(app)
    resp = client.get("/dev/call-status/CA-STATUS")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ended"
    assert body["outcome"]["type"] == "outcome_failed"
    assert "turns" not in body["outcome"]
    assert "turns" not in body


@pytest.mark.asyncio
async def test_exotel_start_reads_snake_case_call_sid():
    from src.api import dev_call_control

    b, _, _ = _bridge("pcm")
    b._call_sid_field = "call_sid"    # Exotel uses snake_case
    b._ws.receive_text = _one_frame_then_disconnect(
        {"event": "start", "stream_sid": "S2", "call_sid": "EXcall"})
    await b._inbound_loop()
    assert b._call_sid == "EXcall"
    assert dev_call_control.monitor.get("EXcall")["status"] == "answered"


@pytest.mark.asyncio
async def test_consume_events_commits_turn_and_slots():
    events = [
        RealtimeEvent(type="input_transcript", text="yeh app safe hai?"),
        RealtimeEvent(type="output_transcript", text="bilkul safe hai"),
        RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                      tool_args={"action": "send_info", "updated_slots": {"interest_level": "hot"}},
                      tool_id="x"),
        RealtimeEvent(type="turn_complete"),
    ]
    b, sess, agent = _bridge("mulaw", events)
    b._stream_sid = "S1"
    await agent.start()
    await b._consume_events()
    contents = [(m.role, m.content) for m in agent.session.turns]
    assert ("user", "yeh app safe hai?") in contents
    assert ("assistant", "bilkul safe hai") in contents
    assert agent.slots.values.get("interest_level") == "hot"
    assert b._last_action == "send_info"
    assert sess.tool_responses and sess.tool_responses[0][1] == "record_turn_signal"
    assert agent.state.state is State.LISTENING


def test_bootstrap_builds_s2s_telephony_bridge():
    """The bootstrap factory helper returns a TelephonyLiveBridge wired for the
    provider's encoding/sid when the tenant is in s2s mode."""
    from types import SimpleNamespace

    from src.bootstrap import _build_s2s_telephony_bridge

    rt = SimpleNamespace(model="gemini-3.1-flash-live-preview", voice="Aoede",
                         language_code="hi-IN", api_key_env="K")
    tenant = SimpleNamespace(
        id="t1", slug="dev",
        settings=SimpleNamespace(pipeline=SimpleNamespace(mode="s2s", realtime=rt),
                                 timezone="Asia/Kolkata"),
        secret=lambda env: "fake-key", secret_optional=lambda env: "fake-key")
    providers = SimpleNamespace(get_stt=lambda t: None, get_llm=lambda t: object(),
                                get_tts=lambda t: None)

    bridge = _build_s2s_telephony_bridge(
        providers, tenant, VoiceBotScript(agent_name="A", agent_role="s", company_name="X"),
        SlotSchema(), websocket=object(), session_store=None,
        encoding="mulaw", sid_field="streamSid", supports_clear=True)
    assert isinstance(bridge, TelephonyLiveBridge)
    assert bridge._encoding == "mulaw" and bridge._sid_field == "streamSid"
    assert bridge._config.model == "gemini-3.1-flash-live-preview"
    assert "record_turn_signal" in bridge._config.tools[0].name


def test_build_s2s_telephony_bridge_applies_voice_and_lead_override():
    """A dev-console override threads voice + lead_name + the provider's
    call_sid field into the S2S telephony bridge."""
    from types import SimpleNamespace

    from src.bootstrap import _build_s2s_telephony_bridge

    rt = SimpleNamespace(model="m", voice="Aoede", language_code="hi-IN",
                         api_key_env="K", allowed_voices=["Aoede", "Kore", "Leda"])
    tenant = SimpleNamespace(
        id="t1", slug="dev",
        settings=SimpleNamespace(pipeline=SimpleNamespace(mode="s2s", realtime=rt),
                                 timezone="Asia/Kolkata"),
        secret=lambda env: "fake-key", secret_optional=lambda env: "fake-key")
    providers = SimpleNamespace(get_stt=lambda t: None, get_llm=lambda t: object(),
                                get_tts=lambda t: None)

    bridge = _build_s2s_telephony_bridge(
        providers, tenant, VoiceBotScript(agent_name="A", agent_role="s", company_name="X"),
        SlotSchema(), websocket=object(), session_store=None,
        encoding="pcm", sid_field="stream_sid", supports_clear=False,
        call_sid_field="call_sid", voice_override="Kore", lead_data={"lead_name": "Raju"})
    assert bridge._config.voice == "Kore"          # override applied + allowed
    assert bridge._call_sid_field == "call_sid"
    assert bridge._agent.session.lead_data.get("lead_name") == "Raju"


def test_build_s2s_telephony_bridge_rejects_disallowed_voice():
    from types import SimpleNamespace

    from src.bootstrap import _build_s2s_telephony_bridge

    rt = SimpleNamespace(model="m", voice="Aoede", language_code="hi-IN",
                         api_key_env="K", allowed_voices=["Aoede", "Kore"])
    tenant = SimpleNamespace(
        id="t1", slug="dev",
        settings=SimpleNamespace(pipeline=SimpleNamespace(mode="s2s", realtime=rt),
                                 timezone="Asia/Kolkata"),
        secret=lambda env: "k", secret_optional=lambda env: "k")
    providers = SimpleNamespace(get_stt=lambda t: None, get_llm=lambda t: object(),
                                get_tts=lambda t: None)
    bridge = _build_s2s_telephony_bridge(
        providers, tenant, VoiceBotScript(agent_name="A", agent_role="s", company_name="X"),
        SlotSchema(), websocket=object(), session_store=None,
        encoding="mulaw", sid_field="streamSid", supports_clear=True,
        voice_override="Charon")                   # not in allowed_voices
    assert bridge._config.voice == "Aoede"         # falls back to config default
