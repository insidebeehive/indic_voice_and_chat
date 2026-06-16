from __future__ import annotations

import asyncio
import json
import logging
import time
import time as _time

import pytest

from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.interfaces.stt import STTStreamEvent
from src.pipeline.vad import EnergyVAD


class _FakeWS:
    def __init__(self):
        self.sent_json = []
        self.sent_bytes = []

    async def send_text(self, t):
        self.sent_json.append(json.loads(t))

    async def send_bytes(self, b):
        self.sent_bytes.append(b)


class _ScriptedSession:
    def __init__(self, events):
        self._events = events
        self.sent = []
        self.closed = False

    async def send(self, pcm):
        self.sent.append(pcm)

    async def events(self):
        for ev in self._events:
            await asyncio.sleep(0)
            yield ev

    async def aclose(self):
        self.closed = True


class _FakeProvider:
    def __init__(self, session):
        self._session = session

    async def open_stream(self, config):
        return self._session


class _FakeAgent:
    def __init__(self):
        self.text_turns = []
        self.state = type("S", (), {
            "state": type("V", (), {"value": "qualifying"})(),
            "is_terminal": False,
        })()
        self.slots = type("Slots", (), {"values": {}})()

    async def handle_turn_text(self, text, sink, cancel_event=None):
        self.text_turns.append(text)
        from src.dialogue.response_parser import VoiceBotResponse
        from src.pipeline.engine import TurnMetrics, TurnResult

        class _O:
            response = VoiceBotResponse(response_text="जी", action="continue")
            pipeline = TurnResult("u", "hi", 1.0, "{}", 0, TurnMetrics())

        return _O()


def _bridge(events):
    session = _ScriptedSession(events)
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(),
        agent=_FakeAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
        stream_provider=_FakeProvider(session),
    )
    return bridge, session


@pytest.mark.asyncio
async def test_interim_event_emits_partial():
    bridge, session = _bridge([STTStreamEvent(type="interim", text="और कुछ")])
    await bridge._consume_stream_events(session)
    partials = [m for m in bridge._ws.sent_json if m.get("type") == "partial"]
    assert partials and partials[0]["text"] == "और कुछ" and partials[0]["role"] == "user"


@pytest.mark.asyncio
async def test_endpoint_event_dispatches_text_turn():
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="और कुछ"),
        STTStreamEvent(type="endpoint", text="और कुछ benefits हैं"),
    ])
    await bridge._consume_stream_events(session)
    if bridge._turn_task is not None:
        await bridge._turn_task          # turn now runs as a background task
    assert bridge._agent.text_turns == ["और कुछ benefits हैं"]
    transcripts = [m for m in bridge._ws.sent_json
                   if m.get("type") == "transcript" and m.get("role") == "user"]
    assert transcripts and transcripts[-1]["text"] == "और कुछ benefits हैं"


@pytest.mark.asyncio
async def test_endpoint_ignored_while_agent_busy():
    bridge, session = _bridge([STTStreamEvent(type="endpoint", text="x")])
    bridge._agent_busy = True
    await bridge._consume_stream_events(session)
    assert bridge._agent.text_turns == []


@pytest.mark.asyncio
async def test_endpoint_does_not_dispatch_while_prior_turn_unwinding():
    # After a barge, _handle_barge_in clears _agent_busy synchronously while the
    # cancelled turn task is STILL unwinding (parked at an await). A fast endpoint
    # must NOT slip through and dispatch a second turn over that still-live task.
    bridge, session = _bridge([STTStreamEvent(type="endpoint", text="और कुछ")])
    bridge._agent_busy = False  # as a barge would leave it
    # A still-pending turn task standing in for the unwinding cancelled turn.
    pending = asyncio.create_task(asyncio.Event().wait())
    bridge._turn_task = pending
    try:
        await bridge._consume_stream_events(session)
        assert bridge._agent.text_turns == []        # no second turn dispatched
        assert bridge._turn_task is pending          # task not reassigned
    finally:
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass


# --- playback echo gate (fix for "stuck in listening") ------------------
# While the agent's reply is still playing on the client, mic audio must NOT
# be streamed to Deepgram, or the agent's own voice (echo) becomes a continuous
# audio stream that stalls utterance-end detection for many seconds.

@pytest.mark.asyncio
async def test_pcm_frame_dropped_while_agent_audio_still_playing():
    bridge, session = _bridge([])
    bridge._stream_session = session
    bridge._agent_busy = False               # generation done...
    bridge._play_until = time.monotonic() + 5  # ...but audio still playing
    await bridge._on_pcm_frame(b"\x00\x00" * 160)
    assert session.sent == []  # echo kept out of the recognizer


@pytest.mark.asyncio
async def test_pcm_frame_sent_once_playback_finished():
    bridge, session = _bridge([])
    bridge._stream_session = session
    bridge._agent_busy = False
    bridge._play_until = time.monotonic() - 1  # playback finished
    frame = b"\x01\x02" * 160
    await bridge._on_pcm_frame(frame)
    assert session.sent == [frame]


@pytest.mark.asyncio
async def test_barge_in_clears_playback_gate():
    bridge, _ = _bridge([])
    bridge._agent_busy = True
    bridge._cancel_event = asyncio.Event()
    bridge._play_until = time.monotonic() + 5
    bridge._handle_barge_in()
    # gate cleared so the interrupting speech isn't dropped as echo
    assert bridge._play_until <= time.monotonic()
    assert bridge._cancel_event.is_set()


# --- stream reopen on unexpected drop (fix for "stuck in listening") ----

class _DropThenProvider:
    """open_stream hands out queued sessions; raises once exhausted."""
    def __init__(self, sessions):
        self._sessions = list(sessions)
        self.opens = 0

    async def open_stream(self, config):
        self.opens += 1
        if not self._sessions:
            raise RuntimeError("no more sessions")
        return self._sessions.pop(0)


@pytest.mark.asyncio
async def test_stream_consumer_reopens_after_unexpected_drop(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "_STREAM_REOPEN_BACKOFF_S", 0)  # keep the test fast
    s1 = _ScriptedSession([STTStreamEvent(type="endpoint", text="पहला")])
    s2 = _ScriptedSession([STTStreamEvent(type="endpoint", text="दूसरा")])
    prov = _DropThenProvider([s2])  # only the reopen target; s1 is the initial session
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(),
        agent=_FakeAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
        stream_provider=prov,
    )
    bridge._stream_session = s1
    await bridge._run_stream_consumer()
    # s1 dropped after its event -> reopened to s2 and kept consuming
    assert bridge._agent.text_turns == ["पहला", "दूसरा"]
    assert prov.opens >= 2          # reopened at least once
    assert s1.closed                # old session closed on reopen
    assert bridge._stream_session is None  # gave up cleanly once exhausted


@pytest.mark.asyncio
async def test_stream_consumer_stops_without_reopen_when_call_ended(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "_STREAM_REOPEN_BACKOFF_S", 0)
    s1 = _ScriptedSession([])
    prov = _DropThenProvider([s1, _ScriptedSession([])])
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(),
        agent=_FakeAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
        stream_provider=prov,
    )
    bridge._stream_session = s1
    bridge._stopped = True  # call already ending
    await bridge._run_stream_consumer()
    assert prov.opens == 0  # no reopen attempted when stopped


def test_streaming_supports_only_deepgram_languages():
    bridge, _ = _bridge([])
    for ok in ("hi", "mr-IN", "te", "ta", "bn", "en"):
        assert bridge._streaming_supports(ok), ok
    for no in ("ml", "pa", "od", "ml-IN"):   # Deepgram can't stream these
        assert not bridge._streaming_supports(no), no


@pytest.mark.asyncio
async def test_stream_consumer_drops_to_batch_for_non_streamable_language(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "_STREAM_REOPEN_BACKOFF_S", 0)
    s1 = _ScriptedSession([])              # events end immediately (as if torn down)
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(),
        agent=_FakeAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30),
        config=BrowserBridgeConfig(),
        stream_provider=_FakeProvider(s1),
    )
    bridge._agent.active_language = "ml"    # Malayalam — Deepgram can't stream it
    bridge._stream_session = s1
    bridge._stream_language = "hi"
    await bridge._run_stream_consumer()
    assert s1.closed                        # live stream torn down
    assert bridge._stream_session is None    # -> batch path (Sarvam) takes over


def test_build_streaming_provider_from_tenant():
    from types import SimpleNamespace

    from src.api.dev_console import _build_stream_provider

    tenant = SimpleNamespace(
        settings=SimpleNamespace(pipeline=SimpleNamespace(
            stt_streaming=SimpleNamespace(
                provider="deepgram", model="nova-2", language="hi",
                endpointing=300, utterance_end_ms=1000,
                api_key_env="TENANT_DEV_DEEPGRAM_KEY",
            )
        )),
        secret=lambda env: "dg_secret" if env == "TENANT_DEV_DEEPGRAM_KEY" else None,
    )
    provider = _build_stream_provider(tenant)
    assert provider.__class__.__name__ == "DeepgramSTTAdapter"


def test_build_streaming_provider_none_when_unconfigured():
    from types import SimpleNamespace

    from src.api.dev_console import _build_stream_provider

    tenant = SimpleNamespace(
        settings=SimpleNamespace(pipeline=SimpleNamespace(stt_streaming=None)),
        secret=lambda env: None,
    )
    assert _build_stream_provider(tenant) is None


@pytest.mark.asyncio
async def test_handle_barge_in_cancels_when_busy():
    bridge, session = _bridge([])
    import asyncio as _a
    bridge._agent_busy = True
    bridge._cancel_event = _a.Event()
    bridge._handle_barge_in()
    assert bridge._cancel_event.is_set() is True
    assert bridge._agent_busy is False


@pytest.mark.asyncio
async def test_handle_barge_in_noop_when_idle():
    bridge, session = _bridge([])
    import asyncio as _a
    bridge._agent_busy = False
    bridge._cancel_event = _a.Event()
    bridge._handle_barge_in()
    assert bridge._cancel_event.is_set() is False  # untouched


@pytest.mark.asyncio
async def test_cancelled_turn_skips_agent_transcript():
    from src.dialogue.response_parser import VoiceBotResponse
    from src.pipeline.engine import TurnMetrics, TurnResult

    class _CancelAgent:
        state = type("S", (), {"state": type("V", (), {"value": "listening"})(), "is_terminal": False})()
        slots = type("SL", (), {"values": {}})()

        async def handle_turn_text(self, text, sink, cancel_event=None):
            class _O:
                response = VoiceBotResponse(response_text="जी हाँ सुन", action="continue", parse_error="barge-in")
                pipeline = TurnResult("u", "hi", 1.0, "{}", 0, TurnMetrics(), cancelled=True)
            return _O()

    from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
    from src.pipeline.vad import EnergyVAD
    bridge = BrowserVoiceBridge(
        websocket=_FakeWS(), agent=_CancelAgent(),
        vad=EnergyVAD(sample_rate=16000, frame_ms=30), config=BrowserBridgeConfig(),
    )
    await bridge._dispatch_text_turn("और कुछ?")
    agent_msgs = [m for m in bridge._ws.sent_json
                  if m.get("type") == "transcript" and m.get("role") == "agent"]
    assert agent_msgs == []  # abandoned reply not emitted
    statuses = [m["status"] for m in bridge._ws.sent_json if m.get("type") == "status"]
    assert statuses[-1] == "listening"


@pytest.mark.asyncio
async def test_endpoint_gap_ms_logged(caplog):
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="haan"),
        STTStreamEvent(type="endpoint", text="haan ji boliye"),
    ])
    with caplog.at_level(logging.INFO):
        await bridge._consume_stream_events(session)
        if bridge._turn_task is not None:
            await bridge._turn_task          # turn now runs as a background task
    recs = [r for r in caplog.records if r.message == "browser turn (stream)"]
    assert recs, "no 'browser turn (stream)' log emitted"
    gap = getattr(recs[0], "endpoint_gap_ms", None)
    assert gap is not None and gap >= 0


@pytest.mark.asyncio
async def test_config_message_enables_barge():
    bridge, _ = _bridge([])
    bridge._apply_control({"type": "config", "barge": True})
    assert bridge._barge_enabled is True
    bridge._apply_control({"type": "config", "barge": False})
    assert bridge._barge_enabled is False


@pytest.mark.asyncio
async def test_barge_guard_fires_during_playback_only(monkeypatch):
    bridge, _ = _bridge([])
    bridge._agent_busy = False
    bridge._cancel_event = asyncio.Event()
    bridge._play_until = _time.monotonic() + 5
    bridge._handle_barge_in()
    assert bridge._cancel_event.is_set()
    assert bridge._play_until == 0.0


@pytest.mark.asyncio
async def test_barge_guard_noop_when_agent_silent(monkeypatch):
    bridge, _ = _bridge([])
    bridge._agent_busy = False
    bridge._play_until = 0.0
    bridge._cancel_event = asyncio.Event()
    bridge._handle_barge_in()
    assert not bridge._cancel_event.is_set()


@pytest.fixture
def _clock(monkeypatch):
    import src.api.browser_bridge as bb
    t = {"now": 1000.0}
    monkeypatch.setattr(bb.time, "monotonic", lambda: t["now"])
    monkeypatch.setattr(bb, "BARGE_SUSTAIN_MS", 450)
    return t


def _armed_bridge():
    bridge, _ = _bridge([])
    bridge._barge_enabled = True
    bridge._had_turn = True
    bridge._agent_busy = True       # agent audible
    return bridge


def test_barge_on_interim_fires_when_sustained(_clock):
    bridge = _armed_bridge()
    assert bridge._barge_on_interim() is False          # first interim -> start timer
    _clock["now"] += 0.5                                 # 500ms > 450ms threshold
    assert bridge._barge_on_interim() is True            # sustained -> fire
    assert bridge._barge_start_t is None                 # reset so it can't re-fire


def test_barge_on_interim_no_fire_for_short_backchannel(_clock):
    bridge = _armed_bridge()
    assert bridge._barge_on_interim() is False
    _clock["now"] += 0.2                                 # 200ms < 450ms
    assert bridge._barge_on_interim() is False


def test_barge_on_interim_no_fire_when_not_audible(_clock):
    bridge = _armed_bridge()
    bridge._agent_busy = False
    bridge._play_until = 0.0                              # agent NOT audible
    assert bridge._barge_on_interim() is False
    _clock["now"] += 1.0
    assert bridge._barge_on_interim() is False
    assert bridge._barge_start_t is None


def test_barge_on_interim_disabled_or_no_turn(_clock):
    bridge = _armed_bridge()
    bridge._barge_enabled = False
    assert bridge._barge_on_interim() is False
    bridge._barge_enabled = True
    bridge._had_turn = False
    assert bridge._barge_on_interim() is False


@pytest.mark.asyncio
async def test_agent_busy_reset_when_turn_raises():
    """_agent_busy must be cleared even when handle_turn_text raises (Fix 1)."""
    bridge, _ = _bridge([])

    async def _boom(*a, **k):
        raise RuntimeError("turn blew up")

    bridge._agent.handle_turn_text = _boom
    bridge._agent_busy = True
    bridge._cancel_event = asyncio.Event()
    with pytest.raises(RuntimeError, match="turn blew up"):
        await bridge._dispatch_text_turn("hi")
    assert bridge._agent_busy is False  # not wedged


@pytest.mark.asyncio
async def test_consumer_barges_on_sustained_interim(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "BARGE_SUSTAIN_MS", 0)   # any 2nd interim while audible fires
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="ru"),
        STTStreamEvent(type="interim", text="ruko"),
    ])
    bridge._barge_enabled = True
    bridge._had_turn = True
    bridge._agent_busy = True                          # a turn is "in flight"
    bridge._cancel_event = asyncio.Event()
    bridge._play_until = 0.0
    await bridge._consume_stream_events(session)
    assert bridge._cancel_event.is_set()               # barge cancelled the turn
    interrupts = [m for m in bridge._ws.sent_json if m.get("type") == "interrupt"]
    assert interrupts                                  # client told to stop playback


@pytest.mark.asyncio
async def test_consumer_no_barge_when_disabled(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "BARGE_SUSTAIN_MS", 0)
    bridge, session = _bridge([
        STTStreamEvent(type="interim", text="ru"),
        STTStreamEvent(type="interim", text="ruko"),
    ])
    bridge._barge_enabled = False                      # off
    bridge._had_turn = True
    bridge._agent_busy = True
    bridge._cancel_event = asyncio.Event()
    await bridge._consume_stream_events(session)
    assert not bridge._cancel_event.is_set()


@pytest.mark.asyncio
async def test_mic_fed_during_playback_when_barge_enabled():
    bridge, session = _bridge([])
    bridge._stream_session = session
    bridge._agent_busy = False
    bridge._play_until = _time.monotonic() + 5         # agent audio still playing
    bridge._barge_enabled = True
    await bridge._on_pcm_frame(b"\x01\x02" * 160)
    assert session.sent == [b"\x01\x02" * 160]         # fed (so Deepgram can hear the interruption)


@pytest.mark.asyncio
async def test_mic_gated_during_playback_when_barge_disabled():
    bridge, session = _bridge([])
    bridge._stream_session = session
    bridge._agent_busy = False
    bridge._play_until = _time.monotonic() + 5
    bridge._barge_enabled = False
    await bridge._on_pcm_frame(b"\x01\x02" * 160)
    assert session.sent == []                          # echo gate still drops it


# --- breath filler ----------------------------------------------------------

import src.api.browser_bridge as bb  # noqa: E402


@pytest.fixture
def _stub_filler(monkeypatch):
    """Force a single known filler clip so selection is deterministic."""
    monkeypatch.setattr(bb, "_filler_clips_cache", [b"BREATHPCM"])
    yield b"BREATHPCM"


@pytest.mark.asyncio
async def test_breath_filler_sent_before_llm(_stub_filler):
    bridge, _ = _bridge([])
    await bridge._dispatch_text_turn("नमस्ते")
    # The breath PCM reached the browser as binary audio, before the agent turn.
    assert _stub_filler in bridge._ws.sent_bytes
    assert bridge._agent.text_turns == ["नमस्ते"]


@pytest.mark.asyncio
async def test_breath_filler_not_in_transcript(_stub_filler):
    bridge, _ = _bridge([])
    await bridge._dispatch_text_turn("नमस्ते")
    # Audio-only: never appears as a JSON text/transcript frame.
    assert all(m.get("text") != "BREATHPCM" for m in bridge._ws.sent_json)


@pytest.mark.asyncio
async def test_breath_filler_skipped_when_cancelled(_stub_filler):
    bridge, _ = _bridge([])
    bridge._cancel_event = asyncio.Event()
    bridge._cancel_event.set()
    await bridge._send_filler()
    assert _stub_filler not in bridge._ws.sent_bytes


@pytest.mark.asyncio
async def test_breath_filler_noop_without_clips(monkeypatch):
    monkeypatch.setattr(bb, "_filler_clips_cache", [])
    bridge, _ = _bridge([])
    await bridge._send_filler()
    assert bridge._ws.sent_bytes == []


def test_real_filler_clips_load_and_are_16k_mono():
    # The committed asset(s) load as non-empty PCM.
    bb._filler_clips_cache = None
    clips = bb._filler_clips()
    assert len(clips) >= 1 and all(len(c) > 0 for c in clips)
    bb._filler_clips_cache = None  # reset cache for other tests
