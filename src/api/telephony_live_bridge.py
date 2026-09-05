"""Telephony speech-to-speech bridge — Twilio + Exotel media streams.

Same dialogue core as the dev-console S2S path (``_BaseLiveBridge``); this speaks
the phone-call media-stream protocol instead of the browser one. Twilio and
Exotel are the same shape (JSON ``connected/start/media/stop`` over a WS),
differing only in audio encoding (Twilio 8kHz μ-law vs Exotel raw 8kHz PCM), the
stream-id field name, and barge-in (Twilio has a ``clear`` frame). One class,
parameterized by ``encoding`` + ``sid_field``.

Outbound audio is **real-time paced** off the events loop via a sender queue, so
pacing never blocks reading the next Live event (keeps native barge-in snappy).
No opening line — on a real call the callee says "hello" first and the model
replies (matches the dev-console "user speaks first" flow).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time

from src.api import dev_call_control
from src.api.live_bridge_base import _BaseLiveBridge
from src.interfaces.realtime import RealtimeConfig
from src.pipeline.audio_utils import mulaw_to_pcm16, pcm16_to_mulaw, resample_pcm16

log = logging.getLogger(__name__)

_TEL_RATE = 8000        # telephony is 8kHz mono
_FRAME_S = 0.02         # 20ms per media frame
# 20ms @ 8kHz mono: 160 bytes μ-law (1 B/sample) or 320 bytes PCM16 (2 B/sample).
_CHUNK = {"mulaw": 160, "pcm": 320}
_TRANSFER_TIMEOUT_S = 30.0   # how long we wait for CS to respond before giving up


class TelephonyLiveBridge(_BaseLiveBridge):
    """One bridge per phone call. ``encoding``: 'mulaw' (Twilio) | 'pcm' (Exotel)."""

    def __init__(self, *, websocket, agent, config: RealtimeConfig, connect_session,
                 llm=None, tts=None, tenant_timezone: str = "Asia/Kolkata",
                 tenant_id: str | None = None,
                 pronunciation_overrides: dict[str, str] | None = None,
                 encoding: str = "mulaw", sid_field: str = "streamSid",
                 supports_clear: bool = True, call_sid_field: str = "callSid",
                 transfer_webhook_url: str | None = None,
                 transfer_webhook_secret: str | None = None,
                 transfer_failure_text: str | None = None) -> None:
        super().__init__(agent=agent, config=config, connect_session=connect_session,
                         llm=llm, tenant_timezone=tenant_timezone)
        self._ws = websocket
        self._tts = tts
        self._tenant_id = tenant_id
        # CRM-level TTS pronunciation overrides (src.models.crm.Crm.pronunciation_overrides),
        # denormalized onto tenant.settings at resolution time — merged over the generic
        # DEFAULT_PRONUNCIATIONS default for the transfer-failure apology below.
        self._pronunciation_overrides = pronunciation_overrides
        self._encoding = encoding
        self._sid_field = sid_field
        self._supports_clear = supports_clear
        self._call_sid_field = call_sid_field   # Twilio: "callSid" / Exotel: "call_sid"
        self._transfer_webhook_url = transfer_webhook_url
        self._transfer_webhook_secret = transfer_webhook_secret
        self._transfer_failure_text = transfer_failure_text
        self._stream_sid: str | None = None
        self._call_sid: str | None = None        # provider Call SID (dev-console monitor key)
        self._up_state = None        # 8k->16k resample state (inbound)
        self._down_state = None      # 24k->8k resample state (outbound)
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None
        self._play_deadline = 0.0
        self._in_frames = 0          # caller media frames forwarded to the model
        self._out_frames = 0         # model media frames queued for the caller

    async def run(self) -> None:
        await self._drive()

    # --- transport hooks ---
    async def _on_start(self) -> None:
        self._sender_task = asyncio.create_task(self._sender_loop())

    async def _on_teardown(self) -> None:
        if self._sender_task is not None:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except BaseException:  # noqa: BLE001
                pass
        if self._call_sid is not None:
            dev_call_control.monitor.set_status(self._call_sid, "ended")

    async def _deliver_outcome(self, payload: dict) -> None:
        # Publish to the dev-console call monitor so a placed call shows its outcome.
        # `payload` is stored BY REFERENCE here and later JSON-serialized straight
        # to the browser by GET /dev/call-status/{call_sid} — it must never be
        # mutated in place, and must never carry raw LLMMessage turns (opaque
        # `thought_signature` bytes can hard-fail FastAPI's jsonable_encoder).
        if self._call_sid is not None:
            dev_call_control.monitor.set_outcome(self._call_sid, payload)
        # Persist to the conversations row (keyed by provider Call SID), if a
        # persister is wired. No-op for the dev console / tests without a DB.
        # Turns are merged into a NEW dict for the persister only — `payload`
        # itself is left untouched (see the monitor note above).
        from src.api import call_store
        turns = list(getattr(getattr(self._agent, "session", None), "turns", []))
        persist_payload = {**payload, "turns": turns}
        if payload.get("type") == "outcome_failed" and not persist_payload.get("notes"):
            # Distinguish "analysis crashed" from a genuine no-outcome call so a
            # tenant CRM (or anyone reading the conversations row later) can tell
            # the difference — record_outcome/_persist_call_outcome otherwise
            # write outcome/summary/notes as all-None here.
            persist_payload["notes"] = "outcome analysis failed"
        await call_store.deliver_to_persister(self._call_sid, persist_payload)

    async def _inbound_loop(self) -> None:
        from starlette.websockets import WebSocketDisconnect
        try:
            while not self._stopped:
                raw = await self._ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "connected":
                    continue
                if event == "start":
                    start = msg.get("start", {}) or {}
                    self._stream_sid = start.get(self._sid_field) or msg.get(self._sid_field)
                    self._call_sid = (start.get(self._call_sid_field)
                                      or msg.get(self._call_sid_field))
                    log.info("telephony stream started",
                             extra={"sid": self._stream_sid, "call_sid": self._call_sid})
                    if self._call_sid is not None:
                        dev_call_control.monitor.set_status(self._call_sid, "answered")
                elif event == "media":
                    await self._on_media(msg.get("media") or {})
                elif event == "stop":
                    break
        except WebSocketDisconnect:
            pass  # caller hung up — normal end
        finally:
            self._stopped = True

    async def _on_media(self, media: dict) -> None:
        if media.get("track") not in (None, "inbound"):
            return
        payload = media.get("payload")
        if not payload or self._session is None:
            return
        raw = base64.b64decode(payload)
        pcm8k = mulaw_to_pcm16(raw) if self._encoding == "mulaw" else raw
        pcm16k, self._up_state = resample_pcm16(pcm8k, _TEL_RATE, 16000, self._up_state)
        await self._session.send_audio(pcm16k)   # caller audio -> model
        self._in_frames += 1
        if self._in_frames % 250 == 0:           # ~every 5s of caller audio
            log.info("telephony caller audio -> model", extra={"in_frames": self._in_frames})

    async def _send_audio_out(self, pcm16: bytes, rate: int) -> None:
        # Enqueue; the sender task paces it out (never blocks the events loop).
        if pcm16 and self._stream_sid is not None:
            pcm8k, self._down_state = resample_pcm16(pcm16, rate, _TEL_RATE, self._down_state)
            self._audio_q.put_nowait(pcm8k)
            self._out_frames += 1
            if self._out_frames % 50 == 1:       # first chunk, then ~periodic
                log.info("telephony model audio -> caller", extra={"out_frames": self._out_frames})

    async def _send_interrupt(self) -> None:
        # Barge-in: drop queued+playing agent audio and reset pacing.
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._play_deadline = 0.0
        if self._supports_clear and self._stream_sid is not None:
            await self._ws.send_text(json.dumps(
                {"event": "clear", self._sid_field: self._stream_sid}))

    async def _sender_loop(self) -> None:
        chunk = _CHUNK[self._encoding]
        try:
            while True:
                pcm8k = await self._audio_q.get()
                out = pcm16_to_mulaw(pcm8k) if self._encoding == "mulaw" else pcm8k
                now = time.perf_counter()
                if self._play_deadline < now:
                    self._play_deadline = now
                for i in range(0, len(out), chunk):
                    piece = out[i:i + chunk]
                    await self._ws.send_text(json.dumps({
                        "event": "media", self._sid_field: self._stream_sid,
                        "media": {"payload": base64.b64encode(piece).decode("ascii")}}))
                    self._play_deadline += _FRAME_S
                    slack = self._play_deadline - time.perf_counter()
                    if slack > 0:
                        await asyncio.sleep(slack)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - WS closed mid-send (teardown race); stop quietly
            self._stopped = True

    # --- transfer hold ---------------------------------------------------

    async def _on_transfer_hold(self) -> None:
        """AI disconnects (Gemini Live session closes); Twilio WS stays open so
        the caller stays on hold. Waits for CS to post the transfer result.
        On failure, plays a TTS apology before tearing down."""
        from src.api import transfer_store

        call_sid = self._call_sid
        if not call_sid:
            return

        if not self._tenant_id:
            log.error("transfer hold without tenant id; failing transfer",
                      extra={"call_sid": call_sid})
            await self._play_transfer_failure_apology()
            return

        # Close the Gemini Live session — AI stops. Inbound audio will be
        # discarded (_on_media checks self._session is None).
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception:  # noqa: BLE001
                log.exception("live session close during transfer hold")
            self._session = None

        # Notify the coordination server so it can start finding a human.
        await self._fire_transfer_webhook(call_sid)

        # Wait for CS result. Timeout → treat as failure (no human found).
        fut = transfer_store.register(self._tenant_id, call_sid)
        try:
            result = await asyncio.wait_for(fut, timeout=_TRANSFER_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.info("transfer hold timed out; treating as failure",
                     extra={"call_sid": call_sid})
            result = "failure"
        except asyncio.CancelledError:
            transfer_store.cancel_pending(self._tenant_id, call_sid)
            return

        log.info("transfer hold resolved", extra={"call_sid": call_sid, "result": result})
        if result != "success":
            await self._play_transfer_failure_apology()

    async def _fire_transfer_webhook(self, call_sid: str) -> None:
        if not self._transfer_webhook_url:
            return
        from src.integration.tenant_events import deliver
        from src.utils.public_url import public_origin
        # public_origin() is the bare scheme://netloc — platform_webhook_base_url()
        # itself carries a path suffix (e.g. /api/v1/telephony), so naively
        # concatenating it here would double the /api/v1/ segment.
        base = public_origin()
        body = {
            "event": "call.transfer_requested",
            "call_sid": call_sid,
            "transfer_result_url": f"{base}/api/v1/calls/{call_sid}/transfer-result",
        }
        try:
            await deliver(self._transfer_webhook_url, body, self._transfer_webhook_secret)
        except Exception:  # noqa: BLE001
            log.exception("call.transfer_requested webhook failed",
                          extra={"call_sid": call_sid})

    async def _play_transfer_failure_apology(self) -> None:
        if self._tts is None or self._stream_sid is None:
            return
        text = (self._transfer_failure_text
                or "Maaf kijiye, abhi koi agent uplabdh nahi hai. Dhanyavaad.")
        from src.interfaces.tts import TTSConfig
        try:
            result = await self._tts.synthesize(
                text,
                TTSConfig(sample_rate=16000, extra_pronunciations=self._pronunciation_overrides),
            )
            pcm8k, self._down_state = resample_pcm16(
                result.audio, result.sample_rate, _TEL_RATE, self._down_state)
            self._audio_q.put_nowait(pcm8k)
            # Sleep for real-time audio duration so the sender task can drain the queue
            # before _on_teardown cancels it.
            duration_s = len(pcm8k) / (_TEL_RATE * 2)
            await asyncio.sleep(duration_s + 1.0)
        except Exception:  # noqa: BLE001
            log.exception("transfer failure apology TTS failed")
