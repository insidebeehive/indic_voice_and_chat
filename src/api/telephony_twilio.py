"""Twilio Media Streams bridge.

Twilio Media Streams protocol (https://www.twilio.com/docs/voice/media-streams):

Inbound JSON frames Twilio sends us:
    {"event": "connected", ...}
    {"event": "start", "start": {"streamSid": "...", "callSid": "...",
                                  "mediaFormat": {"encoding": "audio/x-mulaw",
                                                  "sampleRate": 8000,
                                                  "channels": 1}}}
    {"event": "media", "media": {"payload": "<base64 mulaw>",
                                  "timestamp": "...",
                                  "track": "inbound"}}
    {"event": "mark", ...}
    {"event": "stop", ...}

Outbound JSON frames we send:
    {"event": "media", "streamSid": "<sid>",
     "media": {"payload": "<base64 mulaw>"}}
    {"event": "clear", "streamSid": "<sid>"}    -- drops Twilio's playback queue

The bridge runs the inbound side (consume + endpoint + dispatch to agent)
and the outbound side (TTS audio chunks -> μ-law base64 frames). Both
sides cooperate via the agent's audio_sink callback.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from src.api.outcome_recorder import OutcomeRecorderMixin
from src.interfaces.llm import ILLMProvider
from src.pipeline.audio_utils import (
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    resample_pcm16,
)
from src.pipeline.turn_capture import accumulate_and_detect
from src.pipeline.vad import (
    EndpointConfig,
    EndpointDetector,
    VADDetector,
)

log = logging.getLogger(__name__)


# Twilio always uses μ-law @ 8 kHz mono.
TWILIO_SAMPLE_RATE = 8000


class _AgentLike(Protocol):
    """Minimal interface the bridge needs from VoiceBotAgent (for testing)."""

    async def start(self) -> None: ...

    async def handle_turn(self, captured_audio: bytes, audio_sink) -> object: ...

    async def handle_extended_silence(self) -> None: ...

    async def handle_hangup(self) -> None: ...


class _WebSocketLike(Protocol):
    async def send_text(self, data: str) -> None: ...

    async def receive_text(self) -> str: ...


@dataclass
class TwilioBridgeConfig:
    pcm_sample_rate: int = 16000             # internal pipeline sample rate
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    max_idle_silence_s: float = 12.0


class TwilioMediaBridge(OutcomeRecorderMixin):
    """One bridge instance per call. Drive with ``run()`` from the websocket."""

    def __init__(
        self,
        websocket: _WebSocketLike,
        agent: _AgentLike,
        vad: VADDetector,
        config: Optional[TwilioBridgeConfig] = None,
        llm: Optional[ILLMProvider] = None,
        tenant_timezone: str = "Asia/Kolkata",
    ) -> None:
        self._ws = websocket
        self._agent = agent
        self._vad = vad
        self._config = config or TwilioBridgeConfig()

        self._stream_sid: Optional[str] = None
        self._capture_buffer = bytearray()  # PCM16 at internal sample_rate
        self._endpoint = EndpointDetector(vad.frame_ms, self._config.endpoint)
        self._upsample_state: Optional[tuple] = None
        self._downsample_state: Optional[tuple] = None
        self._stopped = asyncio.Event()
        self._idle_silence_ms = 0
        self._llm = llm
        self._tenant_timezone = tenant_timezone
        self._last_action: Optional[str] = None
        self._outcome_recorded = False

    # --- entrypoint ----------------------------------------------------

    async def run(self) -> None:
        """Drive the websocket until Twilio closes it or the agent ends."""
        await self._agent.start()
        try:
            while not self._stopped.is_set():
                raw = await self._ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "connected":
                    continue
                if event == "start":
                    self._stream_sid = msg.get("start", {}).get("streamSid") or msg.get("streamSid")
                    log.info("twilio stream started", extra={"streamSid": self._stream_sid})
                elif event == "media":
                    await self._on_media_frame(msg["media"])
                elif event == "stop":
                    log.info("twilio stream stopped")
                    break
                # ``mark`` events are ignored; we don't use playback marks here.
        finally:
            try:
                await self._record_outcome()
            except Exception:  # noqa: BLE001 - never let analysis break teardown
                log.exception("record outcome failed")
            await self._agent.handle_hangup()

    # --- inbound -------------------------------------------------------

    async def _on_media_frame(self, media: dict) -> None:
        if media.get("track") not in (None, "inbound"):
            return  # ignore outbound echoes
        payload = media.get("payload")
        if not payload:
            return
        mulaw = base64.b64decode(payload)
        pcm8k = mulaw_to_pcm16(mulaw)

        # Resample 8k -> internal pcm sample rate.
        if self._config.pcm_sample_rate != TWILIO_SAMPLE_RATE:
            pcm, self._upsample_state = resample_pcm16(
                pcm8k, TWILIO_SAMPLE_RATE, self._config.pcm_sample_rate, self._upsample_state
            )
        else:
            pcm = pcm8k

        # VAD on this chunk (for the idle-silence timer) + shared capture core.
        frame = self._vad.detect(pcm)
        if frame.is_speech:
            self._idle_silence_ms = 0
        else:
            self._idle_silence_ms += self._vad.frame_ms

        # Extended silence -> hang up gracefully.
        if self._idle_silence_ms >= self._config.max_idle_silence_s * 1000:
            self._capture_buffer.extend(pcm)
            await self._agent.handle_extended_silence()
            self._stopped.set()
            return

        # Accumulate + endpoint via the shared helper. Reuse the frame we already
        # computed above so detect() runs once per chunk (matters for stateful VADs).
        if accumulate_and_detect(pcm, self._vad, self._endpoint, self._capture_buffer, frame=frame):
            await self._dispatch_utterance()

    async def _dispatch_utterance(self) -> None:
        captured = bytes(self._capture_buffer)
        self._capture_buffer.clear()
        self._endpoint.reset()
        # The agent's pipeline runs STT->LLM->TTS and pushes audio back via our sink.
        outcome = await self._agent.handle_turn(captured, self._send_pcm)
        # Record the final action for the post-call outcome fallback (defensive:
        # the real agent returns a TurnOutcome; test fakes may return None).
        if outcome is not None:
            self._last_action = getattr(getattr(outcome, "response", None), "action", None)
        # If the agent ended the call, stop reading.
        if getattr(self._agent, "state", None) is not None and getattr(
            self._agent.state, "is_terminal", False
        ):
            self._stopped.set()

    # --- outbound ------------------------------------------------------

    async def _send_pcm(self, pcm16: bytes) -> None:
        """Sink callback for agent TTS audio. Converts to μ-law and frames out.

        Frames are paced at real-time (20ms per 160-byte μ-law chunk @ 8kHz)
        using a wall-clock deadline. Without pacing, Twilio buffers the full
        utterance and plays it back at warp speed — the audio sounds garbled.
        """
        if not pcm16 or self._stream_sid is None:
            return
        # Resample internal pcm -> 8k for Twilio.
        if self._config.pcm_sample_rate != TWILIO_SAMPLE_RATE:
            pcm8k, self._downsample_state = resample_pcm16(
                pcm16, self._config.pcm_sample_rate, TWILIO_SAMPLE_RATE, self._downsample_state
            )
        else:
            pcm8k = pcm16
        mulaw = pcm16_to_mulaw(pcm8k)

        # 160 bytes of μ-law @ 8kHz mono = 20ms of audio. Pace each send to
        # match that so playback runs at real-time. Use a wall-clock deadline
        # (not ``sleep(0.02)`` per iteration) so we don't drift on slow sends.
        import time as _time

        chunk = 160
        frame_duration = 0.02  # seconds
        sent = 0
        start = _time.perf_counter()
        for i in range(0, len(mulaw), chunk):
            piece = mulaw[i : i + chunk]
            await self._ws.send_text(json.dumps({
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": base64.b64encode(piece).decode("ascii")},
            }))
            sent += 1
            target = start + sent * frame_duration
            slack = target - _time.perf_counter()
            if slack > 0:
                await asyncio.sleep(slack)


# --- TwiML -----------------------------------------------------------------


def voice_twiml(stream_websocket_url: str) -> str:
    """TwiML response for the inbound voice webhook.

    Tells Twilio to open a media stream to ``stream_websocket_url`` and
    keep the call up while we drive the conversation.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{stream_websocket_url}"/></Connect>'
        "</Response>"
    )


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def softphone_dial_twiml(*, to_number: str, caller_id: str, recording_callback_url: str) -> str:
    """TwiML for the browser-softphone TwiML App Voice URL.

    The human agent's browser (registered via the Voice JS SDK) is the parent
    call; this dials the lead (``to_number``) from the tenant's caller-ID with
    **dual-channel** recording so the agent and lead land on separate tracks.
    When the recording is ready Twilio POSTs ``recording_callback_url`` →
    transcribe → analyze → outcome.
    """
    to = _xml_escape(to_number)
    cid = _xml_escape(caller_id)
    cb = _xml_escape(recording_callback_url)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Dial callerId="{cid}" record="record-from-answer-dual"'
        f' recordingStatusCallback="{cb}"'
        ' recordingStatusCallbackEvent="completed">'
        f"<Number>{to}</Number>"
        "</Dial>"
        "</Response>"
    )
