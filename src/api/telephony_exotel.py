"""Exotel Voicebot Streaming bridge.

Exotel's Voicebot Streaming protocol (analogous to Twilio Media Streams):

The voice webhook responds with a Voicebot Applet XML directive:

    <Response>
      <Connect>
        <Stream url="wss://.../path"/>
      </Connect>
    </Response>

Exotel then opens a WS to that URL and exchanges JSON frames:

    {"event": "connected", ...}
    {"event": "start", "stream_sid": "<sid>", "call_sid": "<sid>",
     "media_format": {"encoding": "pcm", "sample_rate": 8000, "channels": 1}}
    {"event": "media", "stream_sid": "<sid>",
     "media": {"payload": "<base64 PCM16 LE 8kHz mono>",
               "timestamp": "<ms>", "chunk": <int>}}
    {"event": "stop", ...}

Key difference from Twilio:
- Twilio sends μ-law @ 8kHz (1 byte/sample, 160 bytes per 20ms)
- Exotel sends raw PCM 16-bit signed little-endian @ 8kHz (2 bytes/sample, 320 bytes per 20ms)

The bridge below mirrors ``TwilioMediaBridge`` but skips μ-law en/decoding.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from src.api.outcome_recorder import OutcomeRecorderMixin
from src.interfaces.llm import ILLMProvider
from src.pipeline.audio_utils import resample_pcm16
from src.pipeline.vad import (
    EndpointConfig,
    EndpointDetector,
    VADDetector,
)

log = logging.getLogger(__name__)


# Exotel's Voicebot Streaming uses 8 kHz PCM by default.
EXOTEL_SAMPLE_RATE = 8000


class _AgentLike(Protocol):
    async def start(self) -> None: ...
    async def handle_turn(self, captured_audio: bytes, audio_sink) -> object: ...
    async def handle_extended_silence(self) -> None: ...
    async def handle_hangup(self) -> None: ...


class _WebSocketLike(Protocol):
    async def send_text(self, data: str) -> None: ...
    async def receive_text(self) -> str: ...


@dataclass
class ExotelBridgeConfig:
    pcm_sample_rate: int = 16000          # internal pipeline rate
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    max_idle_silence_s: float = 12.0


class ExotelMediaBridge(OutcomeRecorderMixin):
    """One bridge instance per call. Drive with ``run()`` from the WS handler."""

    def __init__(
        self,
        websocket: _WebSocketLike,
        agent: _AgentLike,
        vad: VADDetector,
        config: Optional[ExotelBridgeConfig] = None,
        llm: Optional[ILLMProvider] = None,
        tenant_timezone: str = "Asia/Kolkata",
    ) -> None:
        self._ws = websocket
        self._agent = agent
        self._vad = vad
        self._config = config or ExotelBridgeConfig()

        self._stream_sid: Optional[str] = None
        self._capture_buffer = bytearray()
        self._endpoint = EndpointDetector(vad.frame_ms, self._config.endpoint)
        self._upsample_state: Optional[tuple] = None
        self._downsample_state: Optional[tuple] = None
        self._stopped = asyncio.Event()
        self._idle_silence_ms = 0
        self._llm = llm
        self._tenant_timezone = tenant_timezone
        self._last_action: Optional[str] = None
        self._outcome_recorded = False

    async def run(self) -> None:
        await self._agent.start()
        try:
            while not self._stopped.is_set():
                raw = await self._ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "connected":
                    continue
                if event == "start":
                    self._stream_sid = (
                        msg.get("stream_sid")
                        or msg.get("start", {}).get("stream_sid")
                        or msg.get("start", {}).get("streamSid")
                    )
                    log.info("exotel stream started", extra={"stream_sid": self._stream_sid})
                elif event == "media":
                    await self._on_media_frame(msg["media"])
                elif event == "stop":
                    log.info("exotel stream stopped")
                    break
        finally:
            try:
                await self._record_outcome()
            except Exception:  # noqa: BLE001 - never let analysis break teardown
                log.exception("record outcome failed")
            await self._agent.handle_hangup()

    async def _on_media_frame(self, media: dict) -> None:
        if media.get("track") not in (None, "inbound"):
            return
        payload = media.get("payload")
        if not payload:
            return
        # Exotel: payload is raw PCM16 LE @ 8kHz mono, base64-encoded.
        pcm8k = base64.b64decode(payload)

        if self._config.pcm_sample_rate != EXOTEL_SAMPLE_RATE:
            pcm, self._upsample_state = resample_pcm16(
                pcm8k, EXOTEL_SAMPLE_RATE, self._config.pcm_sample_rate, self._upsample_state
            )
        else:
            pcm = pcm8k

        self._capture_buffer.extend(pcm)

        frame = self._vad.detect(pcm)
        if frame.is_speech:
            self._idle_silence_ms = 0
        else:
            self._idle_silence_ms += self._vad.frame_ms

        if self._idle_silence_ms >= self._config.max_idle_silence_s * 1000:
            await self._agent.handle_extended_silence()
            self._stopped.set()
            return

        if self._endpoint.feed(frame):
            await self._dispatch_utterance()

    async def _dispatch_utterance(self) -> None:
        captured = bytes(self._capture_buffer)
        self._capture_buffer.clear()
        self._endpoint.reset()
        outcome = await self._agent.handle_turn(captured, self._send_pcm)
        if outcome is not None:
            self._last_action = getattr(getattr(outcome, "response", None), "action", None)
        if getattr(self._agent, "state", None) is not None and getattr(
            self._agent.state, "is_terminal", False
        ):
            self._stopped.set()

    async def _send_pcm(self, pcm16: bytes) -> None:
        """Sink for agent TTS audio — PCM16 (no μ-law conversion).

        Paced at real-time: 320 bytes of PCM16 mono @ 8kHz = 20ms of audio.
        """
        if not pcm16 or self._stream_sid is None:
            return
        if self._config.pcm_sample_rate != EXOTEL_SAMPLE_RATE:
            pcm8k, self._downsample_state = resample_pcm16(
                pcm16, self._config.pcm_sample_rate, EXOTEL_SAMPLE_RATE, self._downsample_state
            )
        else:
            pcm8k = pcm16

        # 320 bytes = 160 samples of PCM16 mono @ 8kHz = 20ms
        chunk = 320
        frame_duration = 0.02
        sent = 0
        start = time.perf_counter()
        for i in range(0, len(pcm8k), chunk):
            piece = pcm8k[i : i + chunk]
            await self._ws.send_text(json.dumps({
                "event": "media",
                "stream_sid": self._stream_sid,
                "media": {"payload": base64.b64encode(piece).decode("ascii")},
            }))
            sent += 1
            target = start + sent * frame_duration
            slack = target - time.perf_counter()
            if slack > 0:
                await asyncio.sleep(slack)


# --- ExotelML / Voicebot Applet XML --------------------------------------


def voicebot_xml(stream_websocket_url: str) -> str:
    """Exotel Voicebot Applet response that opens a streaming WS.

    Shape mirrors Twilio's TwiML for portability; Exotel parses the same
    ``<Connect><Stream url=.../></Connect>`` form in their Voicebot Applet
    container.
    """
    from src.api.telephony_twilio import _xml_escape

    url = _xml_escape(stream_websocket_url)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{url}"/></Connect>'
        "</Response>"
    )
