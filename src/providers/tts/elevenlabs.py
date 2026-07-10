"""ElevenLabs TTS adapter.

Supports both preset and custom/cloned voices via voice_id.

Recommended models:
  eleven_flash_v2_5      — lowest latency, good quality (default)
  eleven_turbo_v2_5      — balanced speed/quality
  eleven_multilingual_v2 — best multilingual quality (Hindi, etc.)

Configuration keys:
  api_key   : ElevenLabs API key (falls back to ELEVENLABS_API_KEY env var)
  model     : model ID (default: eleven_flash_v2_5)
  voice_id  : default voice ID — preset or cloned (default: Rachel)

The voice_id in TTSConfig overrides the adapter-level default per call,
so different tenants or campaigns can use different cloned voices.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import httpx

from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult


log = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "eleven_flash_v2_5"
# ElevenLabs preset voice IDs (common English/multilingual voices).
# For custom/cloned voices the voice_id comes from the ElevenLabs dashboard.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel

_PRESET_VOICES: list[dict] = [
    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel",  "gender": "female"},
    {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi",    "gender": "female"},
    {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella",   "gender": "female"},
    {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni",  "gender": "male"},
    {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh",    "gender": "male"},
    {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold",  "gender": "male"},
    {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam",    "gender": "male"},
    {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam",     "gender": "male"},
]

_DEFAULT_TIMEOUT_S = 10.0
_TTS_ATTEMPTS = 2


def _pcm_output_format(sample_rate: int) -> str:
    """Map a sample rate to the ElevenLabs pcm_<rate> output format string."""
    allowed = {8000, 16000, 22050, 24000, 44100}
    rate = sample_rate if sample_rate in allowed else 16000
    return f"pcm_{rate}"


class ElevenLabsTTSAdapter(ITTSProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        # Defer missing-key error to synthesis time so get_available_voices()
        # works without a key (used by /dev/tts-voices for the voice dropdown).
        self._api_key = config.get("api_key") or os.environ.get("ELEVENLABS_API_KEY") or ""
        self._model = config.get("model") or DEFAULT_MODEL
        self._default_voice_id = config.get("voice_id") or DEFAULT_VOICE_ID
        self._timeout = float(config.get("timeout", _DEFAULT_TIMEOUT_S))

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _voice(self, config: TTSConfig) -> str:
        return config.voice_id or self._default_voice_id

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        if not self._api_key:
            raise ValueError(
                "ElevenLabsTTSAdapter requires an API key "
                "(config 'api_key' or ELEVENLABS_API_KEY env var)"
            )
        voice_id = self._voice(config)
        output_format = _pcm_output_format(config.sample_rate)
        url = (
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"
            f"?output_format={output_format}"
        )
        body = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        timeout = httpx.Timeout(self._timeout, connect=min(self._timeout, 5.0))
        audio: bytes | None = None
        last_exc: Exception | None = None

        for attempt in range(_TTS_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, headers=self._headers(), json=body)
                    resp.raise_for_status()
                    audio = resp.content
                break
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                log.warning("elevenlabs tts transient error (attempt %d/%d): %s",
                            attempt + 1, _TTS_ATTEMPTS, e)
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt + 1 < _TTS_ATTEMPTS:
                    last_exc = e
                    log.warning("elevenlabs tts %s (attempt %d/%d); retrying",
                                e.response.status_code, attempt + 1, _TTS_ATTEMPTS)
                    continue
                raise

        if audio is None:
            raise last_exc  # type: ignore[misc]

        sample_rate = config.sample_rate
        duration_ms = (len(audio) / max(sample_rate * 2, 1)) * 1000.0
        return TTSResult(audio=audio, duration_ms=duration_ms, sample_rate=sample_rate)

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        config: TTSConfig,
    ) -> AsyncIterator[bytes]:
        voice_id = self._voice(config)
        output_format = _pcm_output_format(config.sample_rate)
        url = (
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream"
            f"?output_format={output_format}"
        )
        timeout = httpx.Timeout(self._timeout, connect=min(self._timeout, 5.0))

        async for segment in text_stream:
            if not segment:
                continue
            body = {
                "text": segment,
                "model_id": self._model,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", url, headers=self._headers(), json=body) as resp:
                        resp.raise_for_status()
                        async for chunk in resp.aiter_bytes(chunk_size=4096):
                            if chunk:
                                yield chunk
            except Exception:
                log.exception("elevenlabs stream error for segment %r", segment[:40])
                raise

    def get_available_voices(self, language: str) -> list[dict]:
        # If a key is configured, fetch the account's actual voices (includes
        # cloned/custom voices). Fall back to preset list when no key is set.
        if not self._api_key:
            return list(_PRESET_VOICES)
        try:
            import httpx as _httpx
            resp = _httpx.get(
                f"{ELEVENLABS_BASE_URL}/voices",
                headers={"xi-api-key": self._api_key},
                timeout=8.0,
            )
            resp.raise_for_status()
            voices = resp.json().get("voices", [])
            return [
                {
                    "voice_id": v["voice_id"],
                    "name": v.get("name", v["voice_id"]),
                    "gender": (v.get("labels") or {}).get("gender", ""),
                    "category": v.get("category", ""),
                }
                for v in voices
            ]
        except Exception:
            log.warning("elevenlabs: failed to fetch voices, falling back to preset list")
            return list(_PRESET_VOICES)
