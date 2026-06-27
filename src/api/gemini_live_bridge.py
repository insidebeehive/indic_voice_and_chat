"""Gemini Live (speech-to-speech) dev-console bridge — the browser transport.

The dialogue logic lives in ``_BaseLiveBridge``; this subclass speaks the browser
wire protocol: binary PCM16@16k both ways + JSON status/transcript/interrupt/
outcome, so ``static/dev_console.html`` connects unchanged. Caller audio is
forwarded as-is (already 16k); the model's 24k audio is resampled to 16k.
"""

from __future__ import annotations

import audioop
import json
import logging

from src.api.live_bridge_base import RECORD_TURN_SIGNAL, _BaseLiveBridge
from src.interfaces.realtime import RealtimeConfig

log = logging.getLogger(__name__)

__all__ = ["GeminiLiveBridge", "RECORD_TURN_SIGNAL"]

_SEND_CHUNK = 8192
_OUT_RATE = 16000   # browser plays PCM16 @16k


class GeminiLiveBridge(_BaseLiveBridge):
    """One bridge per browser connection. Drive with ``run()``.

    ``connect_session`` is an async callable ``(RealtimeConfig) -> IRealtimeSession``
    (injectable for tests; defaults to GeminiLiveSession.connect bound to the key).
    """

    # proactive_audio=True in the Live session config makes the model speak first
    # natively — no send_client_content kickoff needed (that disrupted VAD for
    # subsequent turns, causing the model to go silent after the greeting).
    _greets_first = False

    def __init__(self, *, websocket, agent, config: RealtimeConfig, connect_session,
                 llm=None, tenant_timezone: str = "Asia/Kolkata") -> None:
        super().__init__(agent=agent, config=config, connect_session=connect_session,
                         llm=llm, tenant_timezone=tenant_timezone)
        self._ws = websocket
        self._ratecv_state = None

    async def run(self) -> None:
        await self._drive()

    # --- transport hooks ---
    async def _on_start(self) -> None:
        # First text frame is the browser hello; consume it (tenant already bound).
        message = await self._ws.receive()
        if message.get("text"):
            try:
                json.loads(message["text"])
            except (ValueError, TypeError):
                pass

    async def _inbound_loop(self) -> None:
        while not self._stopped:
            message = await self._ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                if self._session is not None:
                    await self._session.send_audio(data)   # caller PCM16@16k -> model
                continue
            text = message.get("text")
            if text is not None:
                try:
                    ctrl = json.loads(text)
                except (ValueError, TypeError):
                    ctrl = {}
                if ctrl.get("type") == "end":
                    self._stopped = True
                    break

    async def _send_audio_out(self, pcm16: bytes, rate: int) -> None:
        if not pcm16:
            return
        out, self._ratecv_state = audioop.ratecv(pcm16, 2, 1, rate, _OUT_RATE, self._ratecv_state)
        if not self._speaking:
            self._speaking = True
            await self._emit_status("speaking")
        for i in range(0, len(out), _SEND_CHUNK):
            await self._ws.send_bytes(out[i:i + _SEND_CHUNK])

    async def _send_interrupt(self) -> None:
        await self._send_json({"type": "interrupt"})

    async def _emit_status(self, status: str) -> None:
        await self._send_json({"type": "status", "status": status})

    async def _emit_transcript(self, role: str, text: str, *, partial: bool) -> None:
        await self._send_json({"type": "partial" if partial else "transcript",
                               "role": role, "text": text})

    async def _deliver_outcome(self, payload: dict) -> None:
        try:
            await self._send_json(payload)
        except Exception:  # noqa: BLE001 - socket gone on raw disconnect; already logged
            log.warning("outcome computed but not delivered (socket closed)")

    async def _send_json(self, obj: dict) -> None:
        await self._ws.send_text(json.dumps(obj))
