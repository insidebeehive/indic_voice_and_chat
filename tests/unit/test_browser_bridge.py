# tests/unit/test_browser_bridge.py
from __future__ import annotations

import asyncio
import json

import pytest

from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.pipeline.vad import EnergyVAD


class FakeWebSocket:
    """Scripted Starlette-style websocket for bridge tests."""

    def __init__(self, incoming: list[dict]):
        self._incoming = list(incoming)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False

    async def accept(self) -> None:
        pass

    async def receive(self) -> dict:
        if self._incoming:
            return self._incoming.pop(0)
        return {"type": "websocket.disconnect", "code": 1000}

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class _RaisingWebSocket(FakeWebSocket):
    """A websocket whose sends raise, simulating a closed/dead connection —
    the exact RuntimeError Starlette raises when the app tries to send after
    the connection has already closed."""

    async def send_text(self, data: str) -> None:
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
        )

    async def send_bytes(self, data: bytes) -> None:
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
        )


def _bridge(ws, agent=None):
    return BrowserVoiceBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
    )


@pytest.mark.asyncio
async def test_send_json_emits_text_frame():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_json({"type": "status", "status": "listening"})
    assert json.loads(ws.sent_text[0]) == {"type": "status", "status": "listening"}


@pytest.mark.asyncio
async def test_send_pcm_writes_binary_frames():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_pcm(b"\x01\x02\x03\x04")
    # status:speaking, then the audio bytes, then status:listening
    assert b"".join(ws.sent_bytes) == b"\x01\x02\x03\x04"
    statuses = [json.loads(t).get("status") for t in ws.sent_text]
    assert "speaking" in statuses and statuses[-1] == "listening"


@pytest.mark.asyncio
async def test_send_json_swallows_closed_socket_and_stops():
    ws = _RaisingWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_json({"type": "status", "status": "listening"})  # must not raise
    assert bridge._stopped is True


@pytest.mark.asyncio
async def test_send_pcm_swallows_closed_socket_and_stops():
    ws = _RaisingWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_pcm(b"\x01\x02\x03\x04")  # must not raise
    assert bridge._stopped is True


@pytest.mark.asyncio
async def test_send_json_noop_once_stopped():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    bridge._stopped = True
    await bridge._send_json({"type": "status", "status": "listening"})
    assert ws.sent_text == []


@pytest.mark.asyncio
async def test_send_pcm_noop_once_stopped():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    bridge._stopped = True
    await bridge._send_pcm(b"\x01\x02\x03\x04")
    assert ws.sent_bytes == []
    assert ws.sent_text == []


@pytest.mark.asyncio
async def test_send_pcm_sets_cancel_event_on_closed_socket():
    ws = _RaisingWebSocket([])
    bridge = _bridge(ws)
    bridge._cancel_event = asyncio.Event()
    await bridge._send_pcm(b"\x01\x02")
    assert bridge._cancel_event.is_set()


# ---------------------------------------------------------------------------
# Task 3 — run() handshake, opening, turn loop, debug events
# ---------------------------------------------------------------------------
from src.agents.voicebot import TurnOutcome
from src.dialogue.response_parser import VoiceBotResponse
from src.pipeline.engine import TurnResult, TurnMetrics


class _FakeState:
    def __init__(self):
        self.state = type("S", (), {"value": "qualifying"})()
        self.is_terminal = False


class _FakeSlots:
    values = {"interested": True}


class FakeAgent:
    """Minimal agent matching the surface BrowserVoiceBridge touches."""

    def __init__(self):
        self.state = _FakeState()
        self.slots = _FakeSlots()
        self.started = False
        self.opening_played = False
        self.hung_up = False
        self.turns: list[bytes] = []
        self.session = type("Sess", (), {"turns": []})()

    async def start(self):
        self.started = True

    async def play_opening(self, sink):
        self.opening_played = True
        self.session.turns.append(type("Msg", (), {"role": "assistant", "content": "Namaste! Main Priya."})())
        # Empty payload: BrowserVoiceBridge._send_pcm's `if not pcm16: return` makes
        # this a true no-op, so it never touches _play_until (the post-opening echo
        # gate). Verified empirically that ANY non-empty fake clip — including a
        # larger one — sets _play_until to a real future timestamp that this test's
        # zero-wall-clock message flood cannot outrun (all 35 follow-on frames get
        # dropped as false-positive echo before the gate ever expires), which is
        # what was silently swallowing the test's speech frames.
        await sink(b"")  # fake opening audio (empty; no echo-gate window)

    async def handle_turn(self, captured: bytes, sink) -> TurnOutcome:
        self.turns.append(captured)
        await sink(b"\x20\x21")  # fake reply audio
        return TurnOutcome(
            response=VoiceBotResponse(response_text="Theek hai", action="continue"),
            pipeline=TurnResult(
                user_text="Namaste",
                user_language="hi",
                user_confidence=1.0,
                agent_text='{"response":"Theek hai"}',
                audio_bytes_sent=2,
                metrics=TurnMetrics(),
            ),
        )

    async def handle_hangup(self):
        self.hung_up = True


def _loud_frame(vad) -> bytes:
    return b"\xff\x7f" * (vad.frame_bytes // 2)


def _silent_frame(vad) -> bytes:
    return b"\x00\x00" * (vad.frame_bytes // 2)


@pytest.mark.asyncio
async def test_run_handshake_plays_opening_and_processes_a_turn():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # hello, then 10 speech frames (300ms) + 25 silence frames (750ms) => one turn.
    incoming = [{"type": "websocket.receive", "text": json.dumps({"type": "hello", "tenant": "dev"})}]
    for _ in range(10):
        incoming.append({"type": "websocket.receive", "bytes": _loud_frame(vad)})
    for _ in range(25):
        incoming.append({"type": "websocket.receive", "bytes": _silent_frame(vad)})
    ws = FakeWebSocket(incoming)
    agent = FakeAgent()
    bridge = BrowserVoiceBridge(websocket=ws, agent=agent, vad=vad, config=BrowserBridgeConfig())

    await bridge.run()

    assert agent.started and agent.opening_played and agent.hung_up
    assert len(agent.turns) == 1 and len(agent.turns[0]) > 0  # captured audio dispatched
    events = [json.loads(t) for t in ws.sent_text]
    transcripts = [(e["role"], e["text"]) for e in events if e["type"] == "transcript"]
    assert ("user", "Namaste") in transcripts
    assert ("agent", "Theek hai") in transcripts
    assert ("agent", "Namaste! Main Priya.") in transcripts
    states = [e for e in events if e["type"] == "state"]
    assert states and states[-1]["state"] == "qualifying"
    assert states[-1]["slots"] == {"interested": True}


# ---------------------------------------------------------------------------
# Task 3 (cont.) — terminal-agent path: _dispatch_utterance stops the loop
# ---------------------------------------------------------------------------


class TerminalFakeAgent(FakeAgent):
    """FakeAgent variant whose handle_turn marks the agent as terminal."""

    async def handle_turn(self, captured: bytes, sink) -> TurnOutcome:
        outcome = await super().handle_turn(captured, sink)
        self.state.is_terminal = True
        return outcome


@pytest.mark.asyncio
async def test_terminal_agent_stops_run_loop():
    """When agent.state.is_terminal is True after handle_turn, the bridge sets
    _stopped=True, skips the trailing status:listening, and exits cleanly."""
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # hello + 10 loud frames (speech) + 25 silent frames (endpoint) => one turn.
    incoming = [{"type": "websocket.receive", "text": json.dumps({"type": "hello", "tenant": "dev"})}]
    for _ in range(10):
        incoming.append({"type": "websocket.receive", "bytes": _loud_frame(vad)})
    for _ in range(25):
        incoming.append({"type": "websocket.receive", "bytes": _silent_frame(vad)})
    ws = FakeWebSocket(incoming)
    agent = TerminalFakeAgent()
    bridge = BrowserVoiceBridge(websocket=ws, agent=agent, vad=vad, config=BrowserBridgeConfig())

    await bridge.run()

    # Bridge must have exited the loop via the terminal path.
    assert bridge._stopped is True

    # The finally block in run() must still have fired.
    assert agent.hung_up is True

    # One turn was captured and dispatched.
    assert len(agent.turns) == 1 and len(agent.turns[0]) > 0

    # Transcripts for both user and agent were still emitted.
    events = [json.loads(t) for t in ws.sent_text]
    transcripts = [(e["role"], e["text"]) for e in events if e["type"] == "transcript"]
    assert ("user", "Namaste") in transcripts
    assert ("agent", "Theek hai") in transcripts


# ---------------------------------------------------------------------------
# FIX 1 — sub-frame chunk assembly regression tests
# ---------------------------------------------------------------------------


def _loud_bytes(n_frames, vad):
    return (b"\xff\x7f" * (vad.frame_bytes // 2)) * n_frames


def _silent_bytes(n_frames, vad):
    return (b"\x00\x00" * (vad.frame_bytes // 2)) * n_frames


def _chunk(blob, size=85):
    return [blob[i:i + size] for i in range(0, len(blob), size)]


@pytest.mark.asyncio
async def test_subframe_chunks_are_assembled_then_endpointed():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # 10 frames speech (300ms) + 21 frames silence (630ms > 600ms) => exactly one turn.
    speech = _loud_bytes(10, vad)
    silence = _silent_bytes(21, vad)
    incoming = [{"type": "websocket.receive", "text": json.dumps({"type": "hello"})}]
    for ch in _chunk(speech) + _chunk(silence):
        incoming.append({"type": "websocket.receive", "bytes": ch})
    ws = FakeWebSocket(incoming)
    agent = FakeAgent()
    bridge = BrowserVoiceBridge(websocket=ws, agent=agent, vad=vad, config=BrowserBridgeConfig())
    await bridge.run()
    assert len(agent.turns) == 1   # NOT fired 11x early; assembled correctly


@pytest.mark.asyncio
async def test_subframe_chunks_do_not_endpoint_too_early():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    # 10 frames speech (300ms) + only 10 frames silence (300ms < 600ms) => NO turn.
    # Under the old per-message bug this would have fired (each tiny msg counted 30ms).
    speech = _loud_bytes(10, vad)
    silence = _silent_bytes(10, vad)
    incoming = [{"type": "websocket.receive", "text": json.dumps({"type": "hello"})}]
    for ch in _chunk(speech) + _chunk(silence):
        incoming.append({"type": "websocket.receive", "bytes": ch})
    ws = FakeWebSocket(incoming)
    agent = FakeAgent()
    bridge = BrowserVoiceBridge(websocket=ws, agent=agent, vad=vad, config=BrowserBridgeConfig())
    await bridge.run()
    assert len(agent.turns) == 0   # silence below threshold => no premature dispatch


@pytest.mark.asyncio
async def test_bridge_emits_error_event_on_turn_error():
    """A failed turn (provider outage) should surface an 'error' event to the
    console rather than silently dropping the turn."""
    vad = EnergyVAD(sample_rate=16000, frame_ms=30)
    incoming = [{"type": "websocket.receive", "text": json.dumps({"type": "hello"})}]
    for _ in range(10):
        incoming.append({"type": "websocket.receive", "bytes": _loud_frame(vad)})
    for _ in range(25):
        incoming.append({"type": "websocket.receive", "bytes": _silent_frame(vad)})
    ws = FakeWebSocket(incoming)

    class ErroringAgent(FakeAgent):
        async def handle_turn(self, captured, sink):
            self.turns.append(captured)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="", action="continue",
                    parse_error="pipeline error: RuntimeError: boom",
                ),
                pipeline=TurnResult(
                    user_text="", user_language=None, user_confidence=0.0,
                    agent_text="", audio_bytes_sent=0, metrics=TurnMetrics(),
                ),
            )

    agent = ErroringAgent()
    bridge = BrowserVoiceBridge(websocket=ws, agent=agent, vad=vad, config=BrowserBridgeConfig())
    await bridge.run()

    events = [json.loads(t) for t in ws.sent_text]
    assert any(e.get("type") == "error" and "pipeline error" in e.get("message", "") for e in events)
    assert agent.hung_up is True  # call still completed cleanly


@pytest.mark.asyncio
async def test_emit_outcome_sends_ws_message(monkeypatch):
    import src.api.browser_bridge as bb
    from src.campaign.models import CallAnalysis, LeadCallOutcome

    async def fake_analyze(**kwargs):
        return CallAnalysis(
            outcome=LeadCallOutcome.INTERESTED,
            summary="Good call.", notes="Send link.",
        )

    monkeypatch.setattr(bb, "analyze_call", fake_analyze)

    ws = FakeWebSocket([])
    bridge = _bridge(ws, FakeAgent())
    bridge._llm = object()  # non-None; fake_analyze ignores it
    await bridge._emit_outcome()

    sent = [json.loads(m) for m in ws.sent_text]
    outcome_msgs = [m for m in sent if m.get("type") == "outcome"]
    assert outcome_msgs and outcome_msgs[0]["outcome"] == "interested"
    assert outcome_msgs[0]["summary"] == "Good call."


@pytest.mark.asyncio
async def test_emit_outcome_is_idempotent(monkeypatch):
    import src.api.browser_bridge as bb
    from src.campaign.models import CallAnalysis, LeadCallOutcome

    async def fake_analyze(**kwargs):
        return CallAnalysis(outcome=LeadCallOutcome.INTERESTED, summary="x")

    monkeypatch.setattr(bb, "analyze_call", fake_analyze)
    ws = FakeWebSocket([])
    bridge = _bridge(ws, FakeAgent())
    bridge._llm = object()
    await bridge._emit_outcome()
    await bridge._emit_outcome()
    outcome_msgs = [m for m in (json.loads(x) for x in ws.sent_text) if m.get("type") == "outcome"]
    assert len(outcome_msgs) == 1


@pytest.mark.asyncio
async def test_emit_outcome_noop_when_llm_none():
    ws = FakeWebSocket([])
    bridge = _bridge(ws, FakeAgent())  # _llm defaults to None
    await bridge._emit_outcome()
    outcome_msgs = [m for m in (json.loads(x) for x in ws.sent_text) if m.get("type") == "outcome"]
    assert outcome_msgs == []


@pytest.mark.asyncio
async def test_end_control_message_emits_outcome_and_hangs_up(monkeypatch):
    import src.api.browser_bridge as bb
    from src.campaign.models import CallAnalysis, LeadCallOutcome

    async def fake_analyze(**kwargs):
        return CallAnalysis(outcome=LeadCallOutcome.NOT_INTERESTED, summary="ended")

    monkeypatch.setattr(bb, "analyze_call", fake_analyze)

    incoming = [
        {"type": "websocket.receive", "text": json.dumps({"type": "hello", "tenant": "dev"})},
        {"type": "websocket.receive", "text": json.dumps({"type": "end"})},
    ]
    ws = FakeWebSocket(incoming)
    agent = FakeAgent()
    bridge = _bridge(ws, agent)
    bridge._llm = object()
    await bridge.run()

    outcome_msgs = [m for m in (json.loads(x) for x in ws.sent_text) if m.get("type") == "outcome"]
    assert len(outcome_msgs) == 1  # emitted once on `end`, idempotent in finally
    assert outcome_msgs[0]["outcome"] == "not_interested"
    assert agent.hung_up is True


@pytest.mark.asyncio
async def test_run_with_heartbeat_sends_periodic_status(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "_HEARTBEAT_INTERVAL_S", 0.05)
    ws = FakeWebSocket([])
    bridge = _bridge(ws)

    async def _slow_coro():
        await asyncio.sleep(0.17)
        return "done"

    result = await bridge._run_with_heartbeat(_slow_coro())
    assert result == "done"
    statuses = [
        json.loads(t).get("status") for t in ws.sent_text
        if json.loads(t).get("type") == "status"
    ]
    assert statuses.count("thinking") >= 2  # at least 2 heartbeats during the 0.17s sleep at 0.05s interval


@pytest.mark.asyncio
async def test_run_with_heartbeat_stops_after_coro_completes(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "_HEARTBEAT_INTERVAL_S", 0.05)
    ws = FakeWebSocket([])
    bridge = _bridge(ws)

    async def _fast_coro():
        return "done"

    await bridge._run_with_heartbeat(_fast_coro())
    count_after = len(ws.sent_text)
    await asyncio.sleep(0.15)  # would produce more heartbeat sends if the task weren't cancelled
    assert len(ws.sent_text) == count_after


@pytest.mark.asyncio
async def test_run_with_heartbeat_propagates_coro_exception():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)

    async def _failing_coro():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await bridge._run_with_heartbeat(_failing_coro())
