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
DEFAULT_MODEL = "bulbul:v3"
# Female default: campaigns configured for a female agent (see
# ``build_voicebot_system_prompt``'s gender directive, which is driven by
# ``agent.gender`` and picks gendered Hindi/Marathi grammatical forms) rely on
# the speaker's gender matching. ``priya`` is a verified-working (16 kHz,
# hand-tested against the live API 2026-09) unambiguous female v3 speaker —
# same choice Sarvam's own docs call out as a top pick for Indian-language
# coverage.
DEFAULT_SPEAKER = "priya"

# bulbul:v2 was deprecated by Sarvam in 2026-09 (the API now rejects it with
# "Model 'bulbul:v2' has been deprecated. Please use 'bulbul:v3' instead.");
# bulbul:v1 was retired earlier, in 2025. As of 2026-09 the API accepts
# ``bulbul:v3`` and ``bulbul:v3-beta`` only. ``anushka``/``meera``/``arjun``
# (the old v1/v2 speakers) do not exist on v3 — attempting them 400s with
# "Speaker '<x>' is not compatible with model bulbul:v3."
#
# In bulbul:v3 the speaker set is the SAME across every supported target
# language (the model is multilingual — a speaker renders any of the languages
# below), so we apply one canonical roster to all of them.
#
# Roster + genders below are cross-checked against Sarvam's public docs
# (docs.sarvam.ai "change the speaker voice" how-to, 2026-09) rather than the
# API itself (which doesn't publish gender). Two names carry real everyday
# ambiguity in India regardless of what the docs list them as — "mani" (also
# used as a short form of female names like Manisha/Manjula, though the docs
# list it male) and "sunny" (used by both e.g. actor Sunny Deol (m) and
# performer Sunny Leone (f), though the docs list it male) — flagged here
# rather than silently trusted; verify before relying on either for a
# gender-sensitive campaign.
_BULBUL_V3_SPEAKERS: list[dict] = [
    {"voice_id": "aditya", "gender": "male"},
    {"voice_id": "ritu", "gender": "female"},
    {"voice_id": "ashutosh", "gender": "male"},
    {"voice_id": "priya", "gender": "female"},
    {"voice_id": "neha", "gender": "female"},
    {"voice_id": "rahul", "gender": "male"},
    {"voice_id": "pooja", "gender": "female"},
    {"voice_id": "rohan", "gender": "male"},
    {"voice_id": "simran", "gender": "female"},
    {"voice_id": "kavya", "gender": "female"},
    {"voice_id": "amit", "gender": "male"},
    {"voice_id": "dev", "gender": "male"},
    {"voice_id": "ishita", "gender": "female"},
    {"voice_id": "shreya", "gender": "female"},
    {"voice_id": "ratan", "gender": "male"},
    {"voice_id": "varun", "gender": "male"},
    {"voice_id": "manan", "gender": "male"},
    {"voice_id": "sumit", "gender": "male"},
    {"voice_id": "roopa", "gender": "female"},
    {"voice_id": "kabir", "gender": "male"},
    {"voice_id": "aayan", "gender": "male"},
    {"voice_id": "shubh", "gender": "male"},
    {"voice_id": "advait", "gender": "male"},
    {"voice_id": "anand", "gender": "male"},
    {"voice_id": "tanya", "gender": "female"},
    {"voice_id": "tarun", "gender": "male"},
    {"voice_id": "sunny", "gender": "male"},   # unconfirmed — see note above
    {"voice_id": "mani", "gender": "male"},    # unconfirmed — see note above
    {"voice_id": "gokul", "gender": "male"},
    {"voice_id": "vijay", "gender": "male"},
    {"voice_id": "shruti", "gender": "female"},
    {"voice_id": "suhani", "gender": "female"},
    {"voice_id": "mohit", "gender": "male"},
    {"voice_id": "kavitha", "gender": "female"},
    {"voice_id": "rehan", "gender": "male"},
    {"voice_id": "soham", "gender": "male"},
    {"voice_id": "rupali", "gender": "female"},
]

# Target languages bulbul:v3 supports (BCP-47 codes the API accepts). Same 11
# as bulbul:v2 — cross-checked against Sarvam's public model docs (2026-09),
# not independently confirmed against the live API by this change.
_BULBUL_V3_LANGUAGES = [
    "hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
]

LANGUAGE_VOICES: dict[str, list[dict]] = {
    lang: [dict(s) for s in _BULBUL_V3_SPEAKERS] for lang in _BULBUL_V3_LANGUAGES
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
                if e.response.status_code < 500:
                    # A 4xx means WE sent something wrong (bad model/speaker/
                    # param) and Sarvam's response body says exactly what —
                    # e.g. "Model 'bulbul:v2' has been deprecated." Log it
                    # before raising, or the only thing that reaches the log
                    # is httpx's uninformative "Client error '400 ...'" status
                    # line and diagnosing requires reproducing the call by
                    # hand. Never log the request body/headers here — they
                    # carry the API key and the customer's text.
                    log.error("sarvam tts %s: %s", e.response.status_code,
                              e.response.text[:500])
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
