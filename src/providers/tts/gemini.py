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
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, params={"key": self._api_key}, json=body)
            if not resp.is_success:
                log.error("Gemini TTS error %d: %s", resp.status_code, resp.text[:500])
                # Try fallback model once
                if resp.status_code in (400, 404) and self._model != _FALLBACK_MODEL:
                    url2 = f"{_BASE_URL}/{_FALLBACK_MODEL}:generateContent"
                    resp = await client.post(url2, params={"key": self._api_key}, json=body)
                    if not resp.is_success:
                        log.error(
                            "Gemini TTS fallback error %d: %s",
                            resp.status_code, resp.text[:500],
                        )
            resp.raise_for_status()

        data = resp.json()
        inline = data["candidates"][0]["content"]["parts"][0]["inlineData"]
        raw_pcm24 = base64.b64decode(inline["data"])   # raw PCM16 at 24 kHz

        target_rate = config.sample_rate or 16000
        if target_rate != _NATIVE_RATE:
            pcm = _resample(raw_pcm24, _NATIVE_RATE, target_rate)
        else:
            pcm = raw_pcm24

        return TTSResult(
            audio=pcm,
            duration_ms=(len(pcm) / max(target_rate * 2, 1)) * 1000.0,
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
