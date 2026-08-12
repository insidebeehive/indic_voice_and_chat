from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api import dev_call_control
from src.api.livekit_bridge import LiveKitBridge
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.realtime import RealtimeConfig, RealtimeEvent


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


class _FakeAudioFrame:
    """Mirrors livekit.rtc.AudioFrame's public shape: .data is a memoryview
    (real SDK casts it to int16 "h"; here it's just bytes wrapped in a
    memoryview so .tobytes() round-trips exactly for assertions)."""

    def __init__(self, pcm16: bytes, sample_rate: int, num_channels: int = 1):
        self.data = memoryview(pcm16)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = len(pcm16) // (2 * num_channels)


class _FakeAudioFrameEvent:
    def __init__(self, frame):
        self.frame = frame


class _FakeAudioStream:
    """Async-iterable yielding a preset sequence of _FakeAudioFrameEvents, then
    stopping. If constructed with ``hold=True``, blocks after exhausting the
    preset events until ``.stop()`` is called — lets a test control exactly
    when the inbound loop exits (used by the full run() smoke test)."""

    def __init__(self, events=(), hold: bool = False):
        self._events = list(events)
        self._idx = 0
        self._hold = hold
        self._stop_event = asyncio.Event()

    def stop(self):
        self._stop_event.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx < len(self._events):
            ev = self._events[self._idx]
            self._idx += 1
            return ev
        if self._hold:
            await self._stop_event.wait()
        raise StopAsyncIteration


class _FakeAudioSource:
    def __init__(self):
        self.captured = []          # frame objects passed to capture_frame
        self.clear_queue_called = False

    async def capture_frame(self, frame):
        self.captured.append(frame)

    def clear_queue(self):
        # Sync, matching the real SDK (livekit/rtc/audio_source.py: `def
        # clear_queue(self) -> None`, not async).
        self.clear_queue_called = True


def _fake_frame_factory(pcm, rate, channels):
    return SimpleNamespace(data=pcm, sample_rate=rate, num_channels=channels)


def _bridge(events=(), llm=None, audio_stream=None, audio_source=None,
            room_name="ROOM1", on_hangup=None, output_rate=24000, tenant_id=None):
    agent = _agent()
    sess = _FakeSession(events)

    async def connect(cfg):
        return sess

    stream = audio_stream if audio_stream is not None else _FakeAudioStream()
    source = audio_source if audio_source is not None else _FakeAudioSource()

    b = LiveKitBridge(
        agent=agent, config=RealtimeConfig(model="m"), connect_session=connect,
        llm=llm, audio_stream=stream, audio_source=source,
        frame_factory=_fake_frame_factory, room_name=room_name,
        on_hangup=on_hangup, output_rate=output_rate, tenant_id=tenant_id)
    b._session = sess
    return b, sess, agent


# --- inbound audio ---------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_16k_forwarded_unchanged():
    pcm = b"\x01\x02" * 160
    frame = _FakeAudioFrame(pcm, sample_rate=16000)
    stream = _FakeAudioStream(events=[_FakeAudioFrameEvent(frame)])
    b, sess, _ = _bridge(audio_stream=stream)
    await b._inbound_loop()
    assert sess.audio == [pcm]   # matching rate -> no resample, forwarded as-is


@pytest.mark.asyncio
async def test_inbound_non16k_resampled():
    pcm = b"\x00\x01" * 160      # 320 bytes = 20ms PCM16 @8kHz
    frame = _FakeAudioFrame(pcm, sample_rate=8000)
    stream = _FakeAudioStream(events=[_FakeAudioFrameEvent(frame)])
    b, sess, _ = _bridge(audio_stream=stream)
    await b._inbound_loop()
    assert sess.audio
    assert sess.audio[0] != pcm                # not a raw passthrough
    assert len(sess.audio[0]) > len(pcm)        # upsampled 8k -> 16k


# --- outbound audio ---------------------------------------------------------


@pytest.mark.asyncio
async def test_outbound_chunked_at_output_rate():
    b, _, _ = _bridge(output_rate=24000)
    pcm = b"\x00\x01" * 2400     # 4800 bytes = 100ms @24kHz mono 16-bit
    await b._send_audio_out(pcm, 24000)   # already at output rate, no resample
    assert not b._audio_q.empty()
    task = asyncio.create_task(b._sender_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    captured = b._audio_source.captured
    expected_chunk = int(24000 * 0.02) * 2   # 960 bytes per 20ms frame
    assert captured
    assert sum(len(f.data) for f in captured) == len(pcm)
    for f in captured:
        assert len(f.data) <= expected_chunk
        assert f.sample_rate == 24000
        assert f.num_channels == 1


# --- barge-in ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_interrupt_drains_queue_and_clears_source():
    b, _, _ = _bridge()
    b._audio_q.put_nowait(b"x" * 100)
    b._audio_q.put_nowait(b"y" * 100)
    await b._send_interrupt()
    assert b._audio_q.empty()
    assert b._audio_source.clear_queue_called


# --- start/teardown ----------------------------------------------------------


@pytest.mark.asyncio
async def test_on_start_marks_answered():
    b, _, _ = _bridge(room_name="ROOM-START")
    await b._on_start()
    assert dev_call_control.monitor.get("ROOM-START")["status"] == "answered"
    # cleanup the sender task started by _on_start
    await b._on_teardown()


@pytest.mark.asyncio
async def test_on_teardown_marks_ended_and_calls_hangup():
    hangup_calls = []

    async def on_hangup():
        hangup_calls.append(True)

    b, _, _ = _bridge(room_name="ROOM-END", on_hangup=on_hangup)
    await b._on_start()
    await b._on_teardown()
    assert dev_call_control.monitor.get("ROOM-END")["status"] == "ended"
    assert hangup_calls == [True]


@pytest.mark.asyncio
async def test_on_teardown_swallows_hangup_failure():
    async def bad_hangup():
        raise RuntimeError("hangup boom")

    b, _, _ = _bridge(room_name="ROOM-BAD-HANGUP", on_hangup=bad_hangup)
    await b._on_teardown()   # must not raise
    assert dev_call_control.monitor.get("ROOM-BAD-HANGUP")["status"] == "ended"


@pytest.mark.asyncio
async def test_on_teardown_no_hangup_is_noop():
    b, _, _ = _bridge(room_name="ROOM-NO-HANGUP", on_hangup=None)
    await b._on_teardown()   # must not raise
    assert dev_call_control.monitor.get("ROOM-NO-HANGUP")["status"] == "ended"


# --- outcome delivery contract (ported from test_telephony_live_bridge.py) --


@pytest.mark.asyncio
async def test_deliver_outcome_merges_turns_for_persister_without_mutating_payload():
    """Pins the same non-mutation contract as
    TelephonyLiveBridge._deliver_outcome: the persister gets turns merged into
    a NEW dict, while the payload object handed to the monitor (stored by
    reference, later JSON-serialized by GET /dev/call-status/{call_sid}) is
    left untouched."""
    from src.api import call_store
    from src.interfaces.llm import LLMMessage

    b, sess, agent = _bridge(room_name="ROOM-MERGE")
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

    assert delivered["call_sid"] == "ROOM-MERGE"
    turns = delivered["payload"]["turns"]
    assert [(t.role, t.content) for t in turns if t.role != "system"] == [
        ("user", "yeh app safe hai?"), ("assistant", "bilkul safe hai")]

    assert "turns" not in payload
    stored = dev_call_control.monitor.get("ROOM-MERGE")["outcome"]
    assert "turns" not in stored
    assert stored is payload   # same object — confirms no copy-then-mutate either


@pytest.mark.asyncio
async def test_deliver_outcome_failed_stamps_notes_for_persister_not_monitor():
    from src.api import call_store

    b, sess, agent = _bridge(room_name="ROOM-NOTES-FAIL")

    delivered = {}

    async def _persister(call_sid, p):
        delivered["payload"] = p

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._deliver_outcome({"type": "outcome_failed"})
    finally:
        call_store.set_call_outcome_persister(None)

    assert delivered["payload"]["notes"] == "outcome analysis failed"
    stored = dev_call_control.monitor.get("ROOM-NOTES-FAIL")["outcome"]
    assert "notes" not in stored


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_deliver_outcome_adds_tenant_id_for_persister_without_mutating_payload():
    """Fix 1: the persister's dict must carry tenant_id (so record_outcome can
    scope its lookup and avoid a cross-tenant room-name collision), but the
    ORIGINAL payload object stored on dev_call_control.monitor by reference
    must stay untouched — same non-mutation contract as the turns-merge test
    above, just for the new key."""
    from src.api import call_store

    b, sess, agent = _bridge(room_name="ROOM-TENANT", tenant_id="tenant_abc")
    payload = {"type": "outcome", "outcome": "interested", "summary": "s"}

    delivered = {}

    async def _persister(call_sid, p):
        delivered["payload"] = p

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._deliver_outcome(payload)
    finally:
        call_store.set_call_outcome_persister(None)

    assert delivered["payload"]["tenant_id"] == "tenant_abc"
    assert "tenant_id" not in payload
    stored = dev_call_control.monitor.get("ROOM-TENANT")["outcome"]
    assert "tenant_id" not in stored
    assert stored is payload


@pytest.mark.asyncio
async def test_deliver_outcome_tenant_id_defaults_to_none_when_not_constructed_with_one():
    from src.api import call_store

    b, sess, agent = _bridge(room_name="ROOM-NO-TENANT")   # tenant_id not passed

    delivered = {}

    async def _persister(call_sid, p):
        delivered["payload"] = p

    call_store.set_call_outcome_persister(_persister)
    try:
        await b._deliver_outcome({"type": "outcome", "outcome": "interested"})
    finally:
        call_store.set_call_outcome_persister(None)

    assert delivered["payload"]["tenant_id"] is None


@pytest.mark.asyncio
async def test_deliver_outcome_success_notes_untouched():
    from src.api import call_store

    b, sess, agent = _bridge(room_name="ROOM-NOTES-OK")

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


# --- full run() smoke test ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_full_smoke_flows_and_tears_down():
    """End-to-end: run() drives model events to a terminal turn while the
    inbound audio stream is still open, then tears down cleanly once the
    stream closes. Uses a hold-based fake stream so the test — not a race —
    controls exactly when the inbound loop exits."""
    events = [
        RealtimeEvent(type="input_transcript", text="hello"),
        RealtimeEvent(type="output_transcript", text="hi there"),
        RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                      tool_args={"action": "end", "updated_slots": {}}, tool_id="x"),
        RealtimeEvent(type="turn_complete"),
    ]
    stream = _FakeAudioStream(events=[], hold=True)
    b, sess, agent = _bridge(events=events, audio_stream=stream, room_name="ROOM-SMOKE")

    run_task = asyncio.create_task(b.run())
    try:
        async with asyncio.timeout(2.0):
            while not agent.state.is_terminal:
                await asyncio.sleep(0.01)
    finally:
        stream.stop()   # let the inbound loop exit now that the turn is done

    await asyncio.wait_for(run_task, timeout=2.0)

    assert b._stopped is True
    contents = [(m.role, m.content) for m in agent.session.turns]
    assert ("user", "hello") in contents
    assert ("assistant", "hi there") in contents
    assert dev_call_control.monitor.get("ROOM-SMOKE")["status"] == "ended"
