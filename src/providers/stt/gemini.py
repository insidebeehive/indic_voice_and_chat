"""Gemini STT adapter.

Uses generateContent with inline audio data (base64 WAV) to transcribe
speech — same GEMINI_API_KEY as the LLM/TTS adapters, no separate API needed.

Model: gemini-2.0-flash (fast, strong on Indian languages and code-switching).
``transcribe_stream`` buffers the iterator and dispatches a single call, same
pattern as Sarvam and Groq (generateContent is request/response, not streaming).
"""

from __future__ import annotations

import base64
import io
import os
import wave
from typing import Any, AsyncIterator

import httpx

from src.interfaces.stt import ISTTProvider, STTConfig, STTResult

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_MODEL = "gemini-2.5-flash"
_TIMEOUT = 30.0

SUPPORTED_LANGUAGES = [
    "hi-IN", "en-IN", "en-US", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
    "mr-IN", "pa-IN", "ta-IN", "te-IN", "ur-IN",
    "ar-XA", "de-DE", "es-ES", "fr-FR", "id-ID", "it-IT",
    "ja-JP", "ko-KR", "nl-NL", "pl-PL", "pt-BR", "ru-RU", "tr-TR", "vi-VN",
]


class GeminiSTTAdapter(ISTTProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        self._model = config.get("model", _DEFAULT_MODEL)
        self._timeout = float(config.get("timeout", _TIMEOUT))
        if not self._api_key:
            raise ValueError("GeminiSTTAdapter requires GEMINI_API_KEY env var")

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        wav = _ensure_wav(audio, config.sample_rate or 16000)
        audio_b64 = base64.b64encode(wav).decode()

        lang_hint = ""
        if config.language:
            lang_hint = f" The audio is in {config.language}."

        body = {
            "contents": [{
                "parts": [
                    {"text": f"Transcribe this audio exactly as spoken, preserving the original language and script. Return only the transcript text with no commentary or formatting.{lang_hint}"},
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                ],
            }],
            "generationConfig": {"temperature": 0},
        }
        url = f"{_BASE_URL}/{self._model}:generateContent?key={self._api_key}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            payload = resp.json()

        text = ""
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            pass

        return STTResult(
            text=text,
            confidence=1.0 if text else 0.0,
            language=config.language,
            raw_response=payload,
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: STTConfig,
    ) -> AsyncIterator[STTResult]:
        buf = bytearray()
        async for chunk in audio_stream:
            buf.extend(chunk)
        yield await self.transcribe(bytes(buf), config)

    def get_supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)


def _ensure_wav(audio: bytes, sample_rate: int) -> bytes:
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio)
    return buf.getvalue()
