"""Sarvam STT adapter.

Sarvam's REST API is request/response — there is no native streaming. We
buffer the input audio iterator into a single payload and call the batch
endpoint. ``transcribe_stream`` therefore yields exactly one ``STTResult``;
true streaming will land in Phase 3 once a streaming-capable provider is
wired in (or Sarvam adds it).

Endpoint reference: https://docs.sarvam.ai/api-reference-docs/speech-to-text
"""

from __future__ import annotations

import io
import os
import wave
from typing import Any, AsyncIterator, Optional

import httpx

from src.interfaces.stt import ISTTProvider, STTConfig, STTResult


def _ensure_wav(audio: bytes, sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono bytes in a WAV container; pass real WAV through unchanged."""
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio)
    return buf.getvalue()


SARVAM_BASE_URL = "https://api.sarvam.ai"
DEFAULT_MODEL = "saaras:v3"

# Per Sarvam docs (subset; full list maintained upstream).
SUPPORTED_LANGUAGES = [
    "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
]


class SarvamSTTAdapter(ISTTProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._model = config.get("model") or DEFAULT_MODEL
        self._api_key = config.get("api_key") or os.environ.get("SARVAM_API_KEY")
        self._base_url = config.get("base_url", SARVAM_BASE_URL)
        self._timeout = config.get("timeout", 30.0)
        if not self._api_key:
            raise ValueError(
                "SarvamSTTAdapter requires an API key (config 'api_key' or "
                "SARVAM_API_KEY env var)"
            )

    def _headers(self) -> dict[str, str]:
        return {"api-subscription-key": self._api_key}

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        wav = _ensure_wav(audio, config.sample_rate or 16000)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data: dict[str, str] = {"model": config.model or self._model}
        if config.language:
            data["language_code"] = config.language
        if config.enable_timestamps:
            data["with_timestamps"] = "true"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/speech-to-text",
                headers=self._headers(),
                data=data,
                files=files,
            )
            resp.raise_for_status()
            payload = resp.json()

        return _parse_response(payload)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: STTConfig,
    ) -> AsyncIterator[STTResult]:
        # Buffer the full stream then call the batch endpoint. Replace with
        # a true streaming implementation when provider support exists.
        buf = bytearray()
        async for chunk in audio_stream:
            buf.extend(chunk)
        result = await self.transcribe(bytes(buf), config)
        yield result

    def get_supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)


def _parse_response(payload: dict[str, Any]) -> STTResult:
    # Sarvam returns: {"transcript": str, "language_code": str, "timestamps": [...] | None}
    text = payload.get("transcript", "") or ""
    language: Optional[str] = payload.get("language_code")
    confidence = float(payload.get("confidence", 1.0)) if payload.get("transcript") else 0.0
    word_timestamps = payload.get("timestamps")
    return STTResult(
        text=text,
        confidence=confidence,
        language=language,
        word_timestamps=word_timestamps,
        raw_response=payload,
    )
