"""Gemini TTS adapter.

Uses the Gemini generateContent API with responseModalities=["AUDIO"] to
synthesize speech — same GEMINI_API_KEY as the LLM, no separate Cloud TTS
API needed.

Output: raw PCM16 mono at the requested sample rate. Gemini returns 24 kHz;
we downsample to 16 kHz (or whatever sample_rate requests) via linear
interpolation.

Model: gemini-2.5-flash-preview-tts (dedicated TTS, May 2025) or fallback to
gemini-2.0-flash-exp when the preview isn't available.
"""

from __future__ import annotations

import base64
import logging
import os
import struct
from typing import Any, AsyncIterator

import httpx

from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
_FALLBACK_MODEL = "gemini-2.0-flash-exp"
_NATIVE_RATE = 24000
_TIMEOUT = 15.0

# Gemini TTS voices — same catalog as Gemini Live (S2S mode).
_VOICES = [
    {"voice_id": "Aoede",   "gender": "female"},
    {"voice_id": "Charon",  "gender": "male"},
    {"voice_id": "Fenrir",  "gender": "female"},
    {"voice_id": "Kore",    "gender": "female"},
    {"voice_id": "Leda",    "gender": "female"},
    {"voice_id": "Orus",    "gender": "male"},
    {"voice_id": "Puck",    "gender": "male"},
    {"voice_id": "Zephyr",  "gender": "female"},
]
_DEFAULT_VOICE = "Aoede"


class GeminiTTSAdapter(ITTSProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        self._model = config.get("model", _DEFAULT_MODEL)
        self._timeout = float(config.get("timeout", _TIMEOUT))

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        if not self._api_key:
            raise ValueError("GeminiTTSAdapter requires GEMINI_API_KEY env var")
        voice = config.voice_id or _DEFAULT_VOICE
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice}
                    }
                },
            },
        }
        url = f"{_BASE_URL}/{self._model}:generateContent"
        # The API key rides in the `?key=` query param, so only the
        # key-less `url`/`url2` constants are logged, never the request as
        # sent. `body` includes the customer's text, which DEBUG is meant to
        # carry in full.
        debug_event(log, "gemini tts request", url=url, body=body)
        model_used = self._model
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, params={"key": self._api_key}, json=body)
            if not resp.is_success:
                log.error("Gemini TTS error %d: %s", resp.status_code, resp.text[:500])
                # Try fallback model once
                if resp.status_code in (400, 404) and self._model != _FALLBACK_MODEL:
                    url2 = f"{_BASE_URL}/{_FALLBACK_MODEL}:generateContent"
                    model_used = _FALLBACK_MODEL
                    debug_event(log, "gemini tts fallback request", url=url2, body=body,
                                primary_model=self._model, primary_status=resp.status_code)
                    resp = await client.post(url2, params={"key": self._api_key}, json=body)
                    if not resp.is_success:
                        log.error(
                            "Gemini TTS fallback error %d: %s",
                            resp.status_code, resp.text[:500],
                        )
            resp.raise_for_status()

        data = resp.json()
        raw_pcm24 = base64.b64decode(
            _inline_audio_b64(data, model=model_used, text_chars=len(text)))  # PCM16 at 24 kHz

        target_rate = config.sample_rate or 16000
        if target_rate != _NATIVE_RATE:
            pcm = _resample(raw_pcm24, _NATIVE_RATE, target_rate)
        else:
            pcm = raw_pcm24

        duration_ms = (len(pcm) / max(target_rate * 2, 1)) * 1000.0
        debug_event(log, "gemini tts response", model=model_used, audio_bytes=len(pcm),
                    duration_ms=duration_ms, sample_rate=target_rate)
        return TTSResult(
            audio=pcm,
            duration_ms=duration_ms,
            sample_rate=target_rate,
        )

    async def synthesize_stream(
        self, text_stream: AsyncIterator[str], config: TTSConfig
    ) -> AsyncIterator[bytes]:
        async for segment in text_stream:
            if not segment:
                continue
            result = await self.synthesize(segment, config)
            yield result.audio

    def get_available_voices(self, language: str) -> list[dict]:
        return list(_VOICES)


class GeminiTTSNoAudioError(RuntimeError):
    """Gemini answered 200 but returned no audio (blocked prompt, a candidate
    with no content, or a stop before any audio). The message names why."""


def _inline_audio_b64(data: dict, *, model: str, text_chars: int) -> str:
    """The base64 audio from a generateContent response, or raise
    GeminiTTSNoAudioError naming why there is none.

    A 200 can still carry no audio: the prompt blocked (promptFeedback
    .blockReason), no candidates, or a candidate whose finishReason (SAFETY,
    OTHER, MAX_TOKENS, ...) came with no `content`. Indexing straight into
    candidates[0].content.parts[0].inlineData turned all of those into a bare
    KeyError: 'content' with the reason thrown away.
    """
    candidates = data.get("candidates") or []
    cand = candidates[0] if candidates else {}
    parts = ((cand.get("content") or {}).get("parts")) or []
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return inline["data"]
    feedback = data.get("promptFeedback") or {}
    flagged = [
        r.get("category") for r in (cand.get("safetyRatings") or feedback.get("safetyRatings") or [])
        if r.get("blocked") or r.get("probability") in ("HIGH", "MEDIUM")
    ]
    reason = {
        "model": model,
        "finish_reason": cand.get("finishReason"),
        "finish_message": cand.get("finishMessage"),
        "block_reason": feedback.get("blockReason"),
        "flagged_safety": flagged,
        "candidates": len(candidates),
        "part_kinds": sorted({k for part in parts for k in part}),
        "text_chars": text_chars,
    }
    # Customer text stays out of this WARNING (length only); the DEBUG
    # "gemini tts request" event already carries the full body.
    log.warning("gemini tts returned no audio", extra=reason)
    raise GeminiTTSNoAudioError(
        "Gemini TTS returned no audio: "
        + ", ".join(f"{k}={v}" for k, v in reason.items() if v not in (None, [], ""))
    )


def _resample(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Linear-interpolation resampler for PCM16 mono."""
    n_in = len(pcm) // 2
    if n_in == 0:
        return b""
    samples = struct.unpack(f"<{n_in}h", pcm[:n_in * 2])
    ratio = in_rate / out_rate
    n_out = int(n_in / ratio)
    buf = bytearray(n_out * 2)
    view = memoryview(buf).cast("h")
    for i in range(n_out):
        t = i * ratio
        lo = int(t)
        frac = t - lo
        s0 = samples[lo]
        s1 = samples[lo + 1] if lo + 1 < n_in else s0
        view[i] = int(s0 + frac * (s1 - s0))
    return bytes(buf)
