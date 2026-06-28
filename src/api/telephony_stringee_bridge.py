"""Stringee IVR bridge: per-call turn controller + audio helpers + registry.

Turn-based (no streaming): Stringee records each utterance and POSTs it to our
event webhook; we run the agent's batch handle_turn and return the next SCCO.
See docs/superpowers/specs/2026-06-09-stringee-ivr-design.md.
"""

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import secrets
import time
import wave
from collections.abc import Awaitable, Callable

from src.api.outcome_recorder import OutcomeRecorderMixin
from src.api.telephony_stringee import (
    SILENCE_TIMEOUT_MS,
    answer_scco,
    closing_scco,
    reply_scco,
    reprompt_scco,
)
from src.interfaces.llm import ILLMProvider

log = logging.getLogger(__name__)


# --- audio helpers ------------------------------------------------------

def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16-LE mono in a WAV container Stringee's `play` can fetch."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm16(blob: bytes) -> tuple[bytes, int]:
    """Extract raw PCM16-LE mono + sample rate from a WAV blob.

    Falls back to treating the blob as headerless PCM @ 8 kHz if it isn't a
    recognizable RIFF/WAVE container (defensive — Stringee recordings vary).
    """
    if len(blob) < 44 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return blob, 8000
    with wave.open(io.BytesIO(blob), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    return frames, rate


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono PCM16-LE between sample rates (no-op if equal)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    converted, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
    return converted


class BufferingAudioSink:
    """AudioSink that accumulates PCM instead of streaming it.

    handle_turn / play_opening push the agent's TTS PCM through an
    ``async (bytes) -> None`` sink; for IVR we collect it so it can be
    WAV-encoded and hosted for Stringee's `play`.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    async def __call__(self, pcm16: bytes) -> None:
        if pcm16:
            self._chunks.append(pcm16)

    @property
    def pcm(self) -> bytes:
        return b"".join(self._chunks)


class AudioStore:
    """Short-lived token -> WAV bytes map for serving reply audio to Stringee.

    Entries expire after ``ttl_seconds`` (a call's audio is fetched once,
    seconds after we hand Stringee the URL). Eviction is lazy on get/put.
    """

    def __init__(self, ttl_seconds: float = 120.0) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, tuple[float, bytes]] = {}

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for tok in [k for k, (ts, _) in self._items.items() if ts < cutoff]:
            self._items.pop(tok, None)

    def put(self, wav: bytes) -> str:
        self._sweep()
        token = secrets.token_urlsafe(16)
        self._items[token] = (time.monotonic(), wav)
        return token

    def get(self, token: str) -> bytes | None:
        self._sweep()
        item = self._items.get(token)
        return item[1] if item else None


# --- turn controller --------------------------------------------------------

# Agent reply actions that end the call.
_TERMINAL_ACTIONS = {"end", "close_positive", "close_negative"}
_REPROMPT_TEXT = "Maaf kijiye, dobara boliye?"
# PSTN telephony is 8 kHz; serve play audio at 8 kHz so Stringee can play it.
_PLAY_SAMPLE_RATE = 8000

Fetch = Callable[[str], Awaitable[bytes]]


class StringeeIvrBridge(OutcomeRecorderMixin):
    """One per call. HTTP-driven: start_call once, handle_turn per webhook."""

    def __init__(
        self,
        *,
        call_id: str,
        agent,
        llm: ILLMProvider | None,
        tenant_timezone: str,
        tts_sample_rate: int,
        base_url: str,
        tenant_slug: str,
        fetch: Fetch,
    ) -> None:
        self.call_id = call_id
        self._agent = agent
        self._llm = llm
        self._tenant_timezone = tenant_timezone
        self._rate = tts_sample_rate
        self._base = base_url.rstrip("/")
        self._slug = tenant_slug
        self._fetch = fetch
        self.audio = AudioStore()
        self._last_action: str | None = None
        self._outcome_recorded = False
        self.touched = time.monotonic()
        self._prewarmed_scco: list[dict] | None = None
        self._prewarm_lock = asyncio.Lock()

    # -- url builders --
    def _event_url(self) -> str:
        return f"{self._base}/event/{self._slug}?call_id={self.call_id}"

    def _host(self, pcm16: bytes) -> str:
        # Serve 8 kHz WAV: Stringee (PSTN telephony) plays 8 kHz; a 16 kHz file
        # often won't play. Sarvam renders at self._rate (16 kHz) so downsample.
        pcm8 = resample_pcm16(pcm16, self._rate, _PLAY_SAMPLE_RATE)
        wav = pcm16_to_wav(pcm8, _PLAY_SAMPLE_RATE)
        token = self.audio.put(wav)
        log.info("stringee hosted audio", extra={"call_id": self.call_id,
                 "token": token, "pcm_in": len(pcm16), "wav_out": len(wav)})
        return f"{self._base}/audio/{token}"

    # -- lifecycle --
    async def _synthesize_opening(self) -> list[dict]:
        try:
            await self._agent.start()
            # Diagnostic: use talk (Stringee native TTS) to confirm IVR works
            # before switching back to play + external audio URL.
            script = getattr(self._agent, "_script", None)
            opening_text = (getattr(script, "opening", None) or "").strip()
            talk_text = opening_text or _REPROMPT_TEXT
            log.info("stringee opening (talk)", extra={
                "call_id": self.call_id, "text_len": len(talk_text)})
            return [
                {"action": "talk", "text": talk_text, "bargeIn": True},
                {"action": "recordMessage", "eventUrl": self._event_url(),
                 "format": "wav", "silenceTimeout": SILENCE_TIMEOUT_MS, "beepStart": False},
            ]
        except Exception:  # noqa: BLE001 - never answer with a 500 / dead line
            log.exception("stringee start_call failed")
            return reprompt_scco(text=_REPROMPT_TEXT, event_url=self._event_url())

    async def prewarm(self) -> None:
        """Synthesize the opening audio during the ringing phase so start_call()
        returns instantly when the callee picks up."""
        async with self._prewarm_lock:
            if self._prewarmed_scco is None:
                self._prewarmed_scco = await self._synthesize_opening()
                log.info("stringee bridge prewarmed", extra={"call_id": self.call_id})

    async def start_call(self) -> list[dict]:
        async with self._prewarm_lock:
            if self._prewarmed_scco is not None:
                log.info("stringee start_call (prewarmed)", extra={"call_id": self.call_id})
                return self._prewarmed_scco
            return await self._synthesize_opening()

    async def handle_turn(self, *, recording_url: str) -> list[dict]:
        self.touched = time.monotonic()
        try:
            blob = await self._fetch(recording_url)
            pcm, rate = wav_to_pcm16(blob)
            pcm = resample_pcm16(pcm, rate, self._rate)
        except Exception:  # noqa: BLE001 - never drop the call on a bad recording
            log.exception("stringee recording fetch/decode failed")
            return reprompt_scco(text=_REPROMPT_TEXT, event_url=self._event_url())

        try:
            sink = BufferingAudioSink()
            outcome = await self._agent.handle_turn(pcm, sink)
        except Exception:  # noqa: BLE001 - a provider failure must not drop the call
            log.exception("stringee agent turn failed")
            return reprompt_scco(text=_REPROMPT_TEXT, event_url=self._event_url())
        self._last_action = outcome.response.action
        reply_pcm = sink.pcm

        if outcome.response.action in _TERMINAL_ACTIONS:
            return closing_scco(audio_url=self._host(reply_pcm))
        if not (outcome.response.response_text or "").strip():
            return reprompt_scco(text=_REPROMPT_TEXT, event_url=self._event_url())
        return reply_scco(audio_url=self._host(reply_pcm), event_url=self._event_url())

    async def end(self) -> None:
        await self._record_outcome()  # OutcomeRecorderMixin
        await self._agent.handle_hangup()


# --- registry ---------------------------------------------------------------


class _Registry:
    """In-memory call_id -> bridge map (single-instance/sticky; see spec)."""

    def __init__(self, ttl_seconds: float = 900.0) -> None:
        self._ttl = ttl_seconds
        self._calls: dict[str, StringeeIvrBridge] = {}

    def put(self, bridge: StringeeIvrBridge) -> None:
        self._sweep()
        self._calls[bridge.call_id] = bridge

    def get(self, call_id: str) -> StringeeIvrBridge | None:
        return self._calls.get(call_id)

    async def end(self, call_id: str) -> None:
        bridge = self._calls.pop(call_id, None)
        if bridge is not None:
            await bridge.end()

    def iter_bridges(self) -> list[StringeeIvrBridge]:
        return list(self._calls.values())

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for cid in [c for c, b in self._calls.items() if b.touched < cutoff]:
            log.warning("stringee call %s evicted by TTL without end(); outcome not recorded", cid)
            self._calls.pop(cid, None)


registry = _Registry()
