"""Groq STT adapter (Whisper-large-v3).

Groq hosts an OpenAI-compatible Whisper endpoint at
``https://api.groq.com/openai/v1/audio/transcriptions``. The endpoint
accepts multipart audio + a model identifier and returns the OpenAI
transcription response shape:

    {"text": "...", "language": "...", "duration": ...}

Whisper does not stream natively, so ``transcribe_stream`` buffers the
input iterator and dispatches a single batch call — same pattern as the
Sarvam adapter.
"""

from __future__ import annotations

import io
import logging
import os
import wave
from typing import Any, AsyncIterator, Optional

import httpx

from src.interfaces.stt import ISTTProvider, STTConfig, STTResult

log = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "whisper-large-v3"


# Whisper supports a wide set; advertise the indic-relevant subset that
# matches our other adapters.
SUPPORTED_LANGUAGES = [
    "en", "hi", "bn", "gu", "kn", "ml", "mr", "pa", "ta", "te", "ur",
]


class GroqSTTAdapter(ISTTProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._model = config.get("model") or DEFAULT_MODEL
        self._api_key = config.get("api_key") or os.environ.get("GROQ_API_KEY")
        self._base_url = config.get("base_url", GROQ_BASE_URL)
        self._timeout = config.get("timeout", 30.0)
        if not self._api_key:
            raise ValueError(
                "GroqSTTAdapter requires an API key (config 'api_key' or "
                "GROQ_API_KEY env var)"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        # The live telephony bridge hands us raw, headerless PCM16 (mono, at
        # ``config.sample_rate``). Groq/Whisper needs a parseable container, so
        # wrap bare PCM in a WAV header. Audio that already carries a RIFF/WAVE
        # header (or other encoded formats) is passed through untouched.
        wav = _ensure_wav(audio, config.sample_rate)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data: dict[str, str] = {
            "model": config.model or self._model,
            "response_format": "verbose_json",
        }
        if config.language:
            # Whisper wants ISO-639-1 two-letter codes; Sarvam-style ``hi-IN``
            # gets trimmed to ``hi`` for compatibility.
            data["language"] = config.language.split("-")[0]
        if config.enable_timestamps:
            data["timestamp_granularities[]"] = "word"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/audio/transcriptions",
                headers=self._headers(),
                data=data,
                files=files,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                # httpx's default message is just "Client error '400 Bad
                # Request' for url '...'" — the actual rejection reason lives
                # in the response body and is otherwise lost, making a 400
                # here undiagnosable from logs/transcript alone.
                body = e.response.text[:500]
                log.error("groq stt %s: %s", e.response.status_code, body)
                raise httpx.HTTPStatusError(
                    f"{e}: {body}", request=e.request, response=e.response
                ) from e
            payload = resp.json()

        return _parse_response(payload)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: STTConfig,
    ) -> AsyncIterator[STTResult]:
        # Whisper is request/response only — buffer then dispatch once.
        buf = bytearray()
        async for chunk in audio_stream:
            buf.extend(chunk)
        yield await self.transcribe(bytes(buf), config)

    def get_supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)


def _ensure_wav(audio: bytes, sample_rate: int) -> bytes:
    """Return ``audio`` as a WAV-framed byte string.

    Bytes that already begin with a ``RIFF....WAVE`` header are returned
    unchanged. Anything else is treated as raw little-endian PCM16 mono and
    wrapped in a minimal WAV container at ``sample_rate``.
    """
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # PCM16
        wav.setframerate(sample_rate)
        wav.writeframes(audio)
    return buf.getvalue()


def _parse_response(payload: dict[str, Any]) -> STTResult:
    text = payload.get("text", "") or ""
    language = payload.get("language")
    # Whisper response shape doesn't include a confidence scalar; we report
    # 1.0 on non-empty output, 0.0 on empty.
    confidence = 1.0 if text else 0.0
    word_timestamps = payload.get("words")
    return STTResult(
        text=text,
        confidence=confidence,
        language=language,
        word_timestamps=word_timestamps,
        raw_response=payload,
    )
