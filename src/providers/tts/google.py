"""Google Cloud Text-to-Speech adapter.

Uses the REST API with an API key. Returns LINEAR16 PCM at the requested
sample rate — no resampling needed.

Env var: GOOGLE_TTS_API_KEY (or falls back to GEMINI_API_KEY if the same
Google Cloud project has both Generative Language and TTS APIs enabled).
"""

from __future__ import annotations

import base64
import logging
import os
import struct
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger(__name__)

from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult

_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
_TIMEOUT = 10.0

_VOICES: dict[str, list[dict]] = {
    "hi-IN": [
        {"voice_id": "hi-IN-Neural2-A", "gender": "female"},
        {"voice_id": "hi-IN-Neural2-B", "gender": "male"},
        {"voice_id": "hi-IN-Neural2-C", "gender": "female"},
        {"voice_id": "hi-IN-Neural2-D", "gender": "male"},
        {"voice_id": "hi-IN-Wavenet-A",  "gender": "female"},
        {"voice_id": "hi-IN-Wavenet-B",  "gender": "male"},
        {"voice_id": "hi-IN-Wavenet-C",  "gender": "female"},
        {"voice_id": "hi-IN-Wavenet-D",  "gender": "male"},
    ],
    "en-IN": [
        {"voice_id": "en-IN-Neural2-A", "gender": "female"},
        {"voice_id": "en-IN-Neural2-B", "gender": "male"},
        {"voice_id": "en-IN-Neural2-C", "gender": "female"},
        {"voice_id": "en-IN-Neural2-D", "gender": "male"},
    ],
}
_DEFAULT_VOICE = {"hi-IN": "hi-IN-Neural2-A", "en-IN": "en-IN-Neural2-A"}


class GoogleTTSAdapter(ITTSProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._api_key = (
            config.get("api_key")
            or os.environ.get("GOOGLE_TTS_API_KEY")
            or os.environ.get("GEMINI_API_KEY")  # same project, TTS API enabled
        )
        self._timeout = float(config.get("timeout", _TIMEOUT))

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        if not self._api_key:
            raise ValueError(
                "GoogleTTSAdapter requires GOOGLE_TTS_API_KEY (or GEMINI_API_KEY) env var"
            )
        lang = config.language or "hi-IN"
        voice_name = config.voice_id or _DEFAULT_VOICE.get(lang, f"{lang}-Neural2-A")
        body = {
            "input": {"text": text},
            "voice": {"languageCode": lang, "name": voice_name},
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": config.sample_rate,
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(_TTS_URL, params={"key": self._api_key}, json=body)
            if not resp.is_success:
                log.error(
                    "Google TTS API error %d: %s",
                    resp.status_code,
                    resp.text[:500],
                )
            resp.raise_for_status()
        raw = base64.b64decode(resp.json()["audioContent"])
        pcm, rate = _extract_pcm(raw, fallback_rate=config.sample_rate)
        return TTSResult(
            audio=pcm,
            duration_ms=(len(pcm) / max(rate * 2, 1)) * 1000.0,
            sample_rate=rate,
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
        return list(_VOICES.get(language, _VOICES.get("hi-IN", [])))


def _extract_pcm(blob: bytes, fallback_rate: int) -> tuple[bytes, int]:
    """Strip WAV header from Google's LINEAR16 response and return (pcm, rate)."""
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return blob, fallback_rate
    sample_rate = fallback_rate
    pos = 12
    while pos + 8 <= len(blob):
        chunk_id = blob[pos:pos + 4]
        chunk_size = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        body_start = pos + 8
        if chunk_id == b"fmt ":
            sample_rate = struct.unpack("<I", blob[body_start + 4:body_start + 8])[0]
        elif chunk_id == b"data":
            return blob[body_start:body_start + chunk_size], sample_rate
        pos = body_start + chunk_size + (chunk_size & 1)
    return blob, fallback_rate
