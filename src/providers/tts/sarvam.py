"""Sarvam TTS adapter.

Sarvam's ``/text-to-speech`` returns base64-encoded audio per request — there
is no native streaming. ``synthesize_stream`` calls ``synthesize`` per text
segment in the input iterator, yielding the audio bytes as each segment
finishes. Replace with provider streaming when available.

Endpoint reference: https://docs.sarvam.ai/api-reference-docs/text-to-speech
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, AsyncIterator

import httpx

from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.pipeline.text_normalize import normalize_for_tts


log = logging.getLogger(__name__)

# A TTS request must fail well within the per-sentence watchdog
# (TTS_SENTENCE_TIMEOUT_S = 25s, src/pipeline/engine.py): otherwise a hung
# Sarvam request gets treated as a dropped segment instead of completing.
# With a tight per-request timeout the hang fails fast and one retry can
# recover a transient blip — total worst case (2 attempts) stays under the
# per-sentence watchdog.
_DEFAULT_TIMEOUT_S = 8.0
_TTS_ATTEMPTS = 2  # initial try + 1 retry


SARVAM_BASE_URL = "https://api.sarvam.ai"
DEFAULT_MODEL = "bulbul:v2"
DEFAULT_SPEAKER = "anushka"

# Per Sarvam's current ``bulbul:v2`` speaker roster. ``bulbul:v1`` was
# retired in 2025 (the API now only accepts ``bulbul:v2``, ``bulbul:v3``,
# or ``bulbul:v3-beta``). ``meera`` / ``arjun`` no longer exist as speakers.
#
# In bulbul:v2 the speaker set is the SAME across every supported target
# language (the model is multilingual — a speaker renders any of the languages
# below), so we apply one canonical roster to all of them.
_BULBUL_V2_SPEAKERS: list[dict] = [
    {"voice_id": "anushka", "gender": "female"},
    {"voice_id": "manisha", "gender": "female"},
    {"voice_id": "vidya", "gender": "female"},
    {"voice_id": "arya", "gender": "female"},
    {"voice_id": "abhilash", "gender": "male"},
    {"voice_id": "karun", "gender": "male"},
    {"voice_id": "hitesh", "gender": "male"},
]

# Target languages bulbul:v2 supports (BCP-47 codes the API accepts).
_BULBUL_V2_LANGUAGES = [
    "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
]

LANGUAGE_VOICES: dict[str, list[dict]] = {
    lang: [dict(s) for s in _BULBUL_V2_SPEAKERS] for lang in _BULBUL_V2_LANGUAGES
}


class SarvamTTSAdapter(ITTSProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._model = config.get("model") or DEFAULT_MODEL
        self._api_key = config.get("api_key") or os.environ.get("SARVAM_API_KEY")
        self._base_url = config.get("base_url", SARVAM_BASE_URL)
        self._timeout = config.get("timeout", _DEFAULT_TIMEOUT_S)
        if not self._api_key:
            raise ValueError(
                "SarvamTTSAdapter requires an API key (config 'api_key' or "
                "SARVAM_API_KEY env var)"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
        }

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        # Speak currency amounts (₹100 / Rs 100 -> "100 रुपये") and rewrite
        # English/brand words Sarvam mispronounces into Devanagari — but only for
        # Devanagari-script languages, so a switch to Telugu/Malayalam doesn't get
        # the wrong script injected.
        text = normalize_for_tts(text, config.language, extra=config.extra_pronunciations)
        body: dict[str, Any] = {
            "inputs": [text],
            "target_language_code": config.language,
            "speaker": config.voice_id or DEFAULT_SPEAKER,
            "speech_sample_rate": config.sample_rate,
            "model": self._model,
            "pace": config.speed,
            "pitch": config.pitch,
        }
        timeout = httpx.Timeout(self._timeout, connect=min(self._timeout, 5.0))
        payload = None
        last_exc: Exception | None = None
        for attempt in range(_TTS_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{self._base_url}/text-to-speech",
                        headers=self._headers(),
                        json=body,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                break
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                log.warning("sarvam tts transient error (attempt %d/%d): %s",
                            attempt + 1, _TTS_ATTEMPTS, e)
            except httpx.HTTPStatusError as e:
                # Retry only transient 5xx; surface 4xx (bad key/request) at once.
                if e.response.status_code >= 500 and attempt + 1 < _TTS_ATTEMPTS:
                    last_exc = e
                    log.warning("sarvam tts %s (attempt %d/%d); retrying",
                                e.response.status_code, attempt + 1, _TTS_ATTEMPTS)
                    continue
                raise
        if payload is None:
            raise last_exc  # type: ignore[misc]  # set whenever the loop didn't break

        # Sarvam returns: {"audios": ["<base64>", ...]} — each entry is a
        # WAV-wrapped PCM blob (verified with bulbul:v2). Strip the WAV
        # container so downstream consumers (TwilioMediaBridge._send_pcm)
        # see raw 16-bit mono PCM at the requested sample rate; otherwise
        # the 44-byte header gets decoded as audio samples and causes a
        # noise burst at the start.
        audios = payload.get("audios") or []
        if not audios:
            raise RuntimeError(f"Sarvam TTS returned no audio: {payload}")
        raw = base64.b64decode(audios[0])
        audio_bytes, sample_rate = _extract_pcm(raw, fallback_rate=config.sample_rate)
        duration_ms = (len(audio_bytes) / max(sample_rate * 2, 1)) * 1000.0
        return TTSResult(
            audio=audio_bytes,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
        )

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        config: TTSConfig,
    ) -> AsyncIterator[bytes]:
        async for segment in text_stream:
            if not segment:
                continue
            result = await self.synthesize(segment, config)
            yield result.audio

    def get_available_voices(self, language: str) -> list[dict]:
        return list(LANGUAGE_VOICES.get(language, []))


# --- helpers ---------------------------------------------------------------


def _extract_pcm(blob: bytes, fallback_rate: int) -> tuple[bytes, int]:
    """Return ``(raw_pcm16_mono, sample_rate)`` from a Sarvam audio blob.

    If the blob is WAV-wrapped (``RIFF...WAVE``) — Sarvam's bulbul:v2 case —
    parse the ``fmt `` chunk for the real sample rate, locate the ``data``
    chunk, and return its payload. Otherwise return the blob as-is with the
    caller-supplied fallback rate.
    """
    import struct

    if len(blob) < 44 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return blob, fallback_rate

    # Walk the chunks (header is 12 bytes; then any number of ``<id><len><payload>``)
    sample_rate = fallback_rate
    pos = 12
    pcm: bytes = b""
    while pos + 8 <= len(blob):
        chunk_id = blob[pos : pos + 4]
        chunk_size = struct.unpack("<I", blob[pos + 4 : pos + 8])[0]
        body_start = pos + 8
        body_end = body_start + chunk_size
        if chunk_id == b"fmt ":
            sample_rate = struct.unpack("<I", blob[body_start + 4 : body_start + 8])[0]
        elif chunk_id == b"data":
            pcm = blob[body_start:body_end]
            break
        # WAV chunks are padded to even length.
        pos = body_end + (chunk_size & 1)

    if not pcm:
        return blob, fallback_rate
    return pcm, sample_rate
