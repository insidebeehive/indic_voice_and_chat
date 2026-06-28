"""Azure Cognitive Services Text-to-Speech adapter.

Uses the REST API with a subscription key. Returns raw PCM16 mono at the
requested sample rate — no header stripping or resampling needed.

Env vars:
  AZURE_SPEECH_KEY      — subscription key
  AZURE_SPEECH_REGION   — e.g. "eastus", "centralindia"
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import httpx

from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult

_TIMEOUT = 12.0

_RATE_TO_FORMAT: dict[int, str] = {
    8000:  "raw-8khz-16bit-mono-pcm",
    16000: "raw-16khz-16bit-mono-pcm",
    24000: "raw-24khz-16bit-mono-pcm",
    48000: "raw-48khz-16bit-mono-pcm",
}
_DEFAULT_FORMAT = "raw-16khz-16bit-mono-pcm"
_DEFAULT_RATE   = 16000

# Azure Neural voices with Hindi, English-IN, and regional Indian support.
_VOICES: dict[str, list[dict]] = {
    "hi-IN": [
        {"voice_id": "hi-IN-SwaraNeural",   "gender": "female"},
        {"voice_id": "hi-IN-MadhurNeural",  "gender": "male"},
        {"voice_id": "hi-IN-AaravNeural",   "gender": "male"},
        {"voice_id": "hi-IN-AnanyaNeural",  "gender": "female"},
    ],
    "en-IN": [
        {"voice_id": "en-IN-NeerjaNeural",  "gender": "female"},
        {"voice_id": "en-IN-PrabhatNeural", "gender": "male"},
    ],
    "mr-IN": [
        {"voice_id": "mr-IN-AarohiNeural",  "gender": "female"},
        {"voice_id": "mr-IN-ManoharNeural", "gender": "male"},
    ],
    "ta-IN": [
        {"voice_id": "ta-IN-PallaviNeural", "gender": "female"},
        {"voice_id": "ta-IN-ValluvarNeural","gender": "male"},
    ],
    "te-IN": [
        {"voice_id": "te-IN-ShrutiNeural",  "gender": "female"},
        {"voice_id": "te-IN-MohanNeural",   "gender": "male"},
    ],
    "bn-IN": [
        {"voice_id": "bn-IN-TanishaaNeural","gender": "female"},
        {"voice_id": "bn-IN-BashkarNeural", "gender": "male"},
    ],
    "gu-IN": [
        {"voice_id": "gu-IN-DhwaniNeural",  "gender": "female"},
        {"voice_id": "gu-IN-NiranjanNeural","gender": "male"},
    ],
    "kn-IN": [
        {"voice_id": "kn-IN-SapnaNeural",   "gender": "female"},
        {"voice_id": "kn-IN-GaganNeural",   "gender": "male"},
    ],
    "ml-IN": [
        {"voice_id": "ml-IN-SobhanaNeural", "gender": "female"},
        {"voice_id": "ml-IN-MidhunNeural",  "gender": "male"},
    ],
}
_DEFAULT_VOICE = "hi-IN-SwaraNeural"


class AzureTTSAdapter(ITTSProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._key = config.get("api_key") or os.environ.get("AZURE_SPEECH_KEY")
        self._region = config.get("region") or os.environ.get("AZURE_SPEECH_REGION", "eastus")
        self._timeout = float(config.get("timeout", _TIMEOUT))

    @property
    def _tts_url(self) -> str:
        return (
            f"https://{self._region}.tts.speech.microsoft.com"
            "/cognitiveservices/v1"
        )

    def _output_format(self, sample_rate: int) -> str:
        return _RATE_TO_FORMAT.get(sample_rate, _DEFAULT_FORMAT)

    def _build_ssml(self, text: str, lang: str, voice: str) -> str:
        safe = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
            f' xml:lang="{lang}">'
            f'<voice name="{voice}">{safe}</voice>'
            f"</speak>"
        )

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        if not self._key:
            raise ValueError("AzureTTSAdapter requires AZURE_SPEECH_KEY env var")
        lang = config.language or "hi-IN"
        voice = config.voice_id or _VOICES.get(lang, [{}])[0].get("voice_id", _DEFAULT_VOICE)
        rate = config.sample_rate or _DEFAULT_RATE
        output_format = self._output_format(rate)
        ssml = self._build_ssml(text, lang, voice)

        headers = {
            "Ocp-Apim-Subscription-Key": self._key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": output_format,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._tts_url, headers=headers, content=ssml.encode())
            resp.raise_for_status()
        pcm = resp.content
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
