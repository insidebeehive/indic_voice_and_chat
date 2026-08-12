"""LiveKit WebRTC-room speech-to-speech bridge.

Same dialogue core as ``TelephonyLiveBridge`` (``_BaseLiveBridge``); this speaks
LiveKit's WebRTC room shape instead of a telephony media-stream WebSocket:
caller audio arrives as an ``AudioStream`` of frames off a subscribed track,
and outbound model audio is pushed into an ``AudioSource`` that the room's
outbound track publishes.

**Zero coupling to the LiveKit SDK on purpose.** This module does not import
``livekit.rtc`` or ``livekit.api`` at all — every SDK object (the audio
stream/source, the frame constructor, the room name, an optional hangup
callback) is constructor-injected, so this class is testable with plain fakes.
The real SDK wiring (connecting to a room, building the real
``AudioSource``/``AudioStream``, publishing the outbound track) lives in
``livekit_runner.py`` (a separate module) — this bridge only implements the
transport-hook logic assuming those objects already exist.

Outbound audio is **not independently paced** here (contrast
``TelephonyLiveBridge``'s real-time pacing loop): LiveKit's ``AudioSource.
capture_frame`` is expected to provide real pacing itself via FFI backpressure,
so the sender loop just chunks audio into ~20ms frames and awaits
``capture_frame`` back-to-back.
"""

from __future__ import annotations

import asyncio
import logging

from src.api import dev_call_control
from src.api.live_bridge_base import _BaseLiveBridge
from src.interfaces.realtime import RealtimeConfig
from src.pipeline.audio_utils import resample_pcm16

log = logging.getLogger(__name__)

_FRAME_S = 0.02  # 20ms per outbound chunk, same cadence as the telephony bridge


class LiveKitBridge(_BaseLiveBridge):
    """One bridge per LiveKit room (one call). Caller speaks first (no greeting kickoff)."""

    _greets_first = False

    def __init__(self, *, agent, config: RealtimeConfig, connect_session,
                 llm=None, tenant_timezone: str = "Asia/Kolkata",
                 audio_stream,      # async-iterable yielding .frame.data/.sample_rate/.num_channels/.samples_per_channel
                 audio_source,      # has async capture_frame(frame) and sync clear_queue()
                 frame_factory,     # callable(pcm16_bytes, sample_rate, num_channels) -> frame object
                 room_name: str,    # conversation key — equivalent of Twilio's call_sid
                 on_hangup=None,    # optional async callable() to end the call (e.g. delete the room)
                 output_rate: int = 24000,  # AudioSource's configured output rate (Gemini Live native output)
                 tts=None,
                 tenant_id: str | None = None,  # threaded into the persister payload only —
                                                 # scopes call_store.record_outcome's lookup so a
                                                 # CRM-reused room name can't collide across tenants
                 ) -> None:
        super().__init__(agent=agent, config=config, connect_session=connect_session,
                         llm=llm, tenant_timezone=tenant_timezone)
        self._tts = tts
        self._audio_stream = audio_stream
        self._audio_source = audio_source
        self._frame_factory = frame_factory
        self._room_name = room_name
        self._on_hangup = on_hangup
        self._output_rate = output_rate
        self._tenant_id = tenant_id
        # Reused so dev_call_control.monitor / call_store.deliver_to_persister calls
        # read naturally (they're keyed off "call_sid" on the telephony side; the
        # LiveKit room name plays the same role here).
        self._call_sid: str | None = room_name
        self._up_state = None        # inbound resample state (-> 16kHz, only used if the
                                      # stream's sample_rate ever isn't already 16kHz)
        self._down_state = None      # outbound resample state (model rate -> output_rate)
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None
        self._in_frames = 0          # caller media frames forwarded to the model
        self._out_frames = 0         # model media frames queued for the room

    async def run(self) -> None:
        await self._drive()

    # --- transport hooks ---
    async def _on_start(self) -> None:
        self._sender_task = asyncio.create_task(self._sender_loop())
        # Unlike Twilio (which sets "answered" when the media-stream "start" frame
        # arrives mid-stream), the LiveKit runner only constructs+runs this bridge
        # after the caller's track has been subscribed — the call is already live
        # by the time _on_start runs, so set the status here directly.
        if self._call_sid is not None:
            dev_call_control.monitor.set_status(self._call_sid, "answered")

    async def _on_teardown(self) -> None:
        if self._sender_task is not None:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except BaseException:  # noqa: BLE001
                pass
        if self._call_sid is not None:
            dev_call_control.monitor.set_status(self._call_sid, "ended")
        if self._on_hangup is not None:
            try:
                await self._on_hangup()
            except Exception:  # noqa: BLE001 - teardown must never raise
                log.exception("livekit on_hangup failed on teardown")

    async def _deliver_outcome(self, payload: dict) -> None:
        # Same non-mutation contract as TelephonyLiveBridge._deliver_outcome (pinned
        # by tests on the Twilio side; replicated verbatim here, keyed on the room
        # name): `payload` is stored BY REFERENCE for the monitor (later
        # JSON-serialized straight to the browser), and the persister gets a NEW
        # dict with `turns` merged in so raw LLMMessage objects never reach the
        # monitor's copy.
        if self._call_sid is not None:
            dev_call_control.monitor.set_outcome(self._call_sid, payload)
        from src.api import call_store
        turns = list(getattr(getattr(self._agent, "session", None), "turns", []))
        # tenant_id is added to this NEW dict only — never to `payload` itself,
        # which is stored by reference on dev_call_control.monitor above and
        # must stay byte-for-byte what the caller handed in (pinned by
        # test_deliver_outcome_merges_turns_for_persister_without_mutating_payload).
        persist_payload = {**payload, "turns": turns, "tenant_id": self._tenant_id}
        if payload.get("type") == "outcome_failed" and not persist_payload.get("notes"):
            persist_payload["notes"] = "outcome analysis failed"
        await call_store.deliver_to_persister(self._call_sid, persist_payload)

    async def _inbound_loop(self) -> None:
        try:
            async for event in self._audio_stream:
                if self._stopped:
                    break
                frame = event.frame
                # AudioFrame.data is a memoryview of the raw buffer cast to "h"
                # (native int16 typecode), per livekit/rtc/audio_frame.py. .tobytes()
                # reassembles it into raw bytes in the platform's native byte order,
                # which is little-endian on every deployment target here (x86_64/
                # ARM Northflank) — matching resample_pcm16's expected 16-bit
                # signed little-endian PCM layout.
                pcm = frame.data.tobytes()
                if frame.sample_rate != 16000:
                    pcm, self._up_state = resample_pcm16(pcm, frame.sample_rate, 16000, self._up_state)
                if self._session is not None:
                    await self._session.send_audio(pcm)
                self._in_frames += 1
                if self._in_frames % 500 == 0:   # ~every 5-10s of caller audio at 10-20ms/frame
                    log.info("livekit caller audio -> model", extra={"in_frames": self._in_frames})
        except Exception:  # noqa: BLE001 - a stream/track error ends the call, not crash the bridge
            log.exception("livekit inbound audio stream ended (error)")
        finally:
            self._stopped = True

    async def _send_audio_out(self, pcm16: bytes, rate: int) -> None:
        # Enqueue; the sender task chunks + paces it out (never blocks the events loop).
        if pcm16:
            pcm_out, self._down_state = resample_pcm16(pcm16, rate, self._output_rate, self._down_state)
            self._audio_q.put_nowait(pcm_out)
            self._out_frames += 1
            if self._out_frames % 50 == 1:   # first chunk, then ~periodic
                log.info("livekit model audio -> room", extra={"out_frames": self._out_frames})

    async def _send_interrupt(self) -> None:
        # Barge-in: drop queued+playing agent audio.
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        # AudioSource.clear_queue is a plain (sync) method on the real SDK
        # (livekit/rtc/audio_source.py) — no await.
        self._audio_source.clear_queue()

    async def _sender_loop(self) -> None:
        chunk = int(self._output_rate * _FRAME_S) * 2   # 20ms, 16-bit mono samples -> bytes
        try:
            while True:
                pcm = await self._audio_q.get()
                for i in range(0, len(pcm), chunk):
                    piece = pcm[i:i + chunk]
                    frame = self._frame_factory(piece, self._output_rate, 1)
                    await self._audio_source.capture_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - source closed mid-send (teardown race); stop quietly
            self._stopped = True
