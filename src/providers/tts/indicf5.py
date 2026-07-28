"""IndicF5 TTS adapter — self-hosted fine-tuned voice server (RunPod).

``synthesize`` streams from ``POST {base_url}{_STREAM_PATH}`` with
``{"text", "lang", "speed"}`` — chunked HTTP, raw PCM16 mono chunks (no
per-chunk WAV framing), end-of-stream on connection close. The exact stream
path is UNCONFIRMED pending live verification against the real pod — see
``_STREAM_PATH`` below. There is no voice parameter (the fine-tune IS the
voice), so ``synthesize_stream`` synthesizes per text segment like the
Sarvam adapter, and ``get_available_voices`` reports the one fine-tuned
voice for every IndicF5 language.

Config: ``base_url`` in the provider config, or the platform-level
``INDICF5_TTS_URL`` env var (e.g. ``https://<pod-id>-8000.proxy.runpod.net``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import httpx

from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.pipeline.audio_utils import resample_pcm16
from src.pipeline.text_normalize import normalize_for_tts

log = logging.getLogger(__name__)

# Same rationale as the Sarvam adapter: a TTS request must fail well within
# the 20s turn budget, so keep a tight per-request timeout plus one retry.
# Self-hosted inference on a pod can be slower than a hosted API on cold
# paths, hence slightly more headroom than Sarvam's 8s.
_DEFAULT_TIMEOUT_S = 10.0
_TTS_ATTEMPTS = 2  # initial try + 1 retry

# UNCONFIRMED — best guess pending live verification against the real pod.
# If IndicF5's actual path differs, this is the only line that needs to change.
_STREAM_PATH = "/tts/stream"

# IndicF5 always renders at this fixed rate regardless of what's requested.
# The old WAV-wrapped /tts response let us read this from the header; the new
# raw-PCM streaming response has no header, so it's hardcoded here instead.
_NATIVE_SAMPLE_RATE = 24000

DEFAULT_VOICE = "indicf5"

# IndicF5 covers these Indic languages; the server takes bare ISO 639-1
# codes ("mr"), while our TTSConfig carries BCP-47 ("mr-IN").
_LANGUAGES = [
    "as-IN", "bn-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
    "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
]

# The fine-tune is a single MALE voice — campaigns selecting this provider
# must set agent.gender: male so the prompt's gendered grammatical forms
# ("kar raha hun", not "kar rahi hun") match what callers hear.
LANGUAGE_VOICES: dict[str, list[dict]] = {
    lang: [{"voice_id": DEFAULT_VOICE, "gender": "male"}] for lang in _LANGUAGES
}


def _server_lang(language: str) -> str:
    """Map BCP-47 ("mr-IN") to the server's bare code ("mr")."""
    return (language or "").split("-")[0].lower() or "hi"


class IndicF5TTSAdapter(ITTSProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        base_url = config.get("base_url") or os.environ.get("INDICF5_TTS_URL")
        if not base_url:
            raise ValueError(
                "IndicF5TTSAdapter requires a server URL (config 'base_url' or "
                "INDICF5_TTS_URL env var)"
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = config.get("timeout", _DEFAULT_TIMEOUT_S)

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        # Same normalization as Sarvam: speak currency amounts and rewrite
        # words TTS mispronounces, scoped by language/script.
        text = normalize_for_tts(text, config.language, extra=config.extra_pronunciations)
        body = {
            "text": text,
            "lang": _server_lang(config.language),
            "speed": config.speed,
        }
        timeout = httpx.Timeout(self._timeout, connect=min(self._timeout, 5.0))
        chunks: list[bytes] | None = None
        last_exc: Exception | None = None
        for attempt in range(_TTS_ATTEMPTS):
            try:
                collected: list[bytes] = []
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", f"{self._base_url}{_STREAM_PATH}", json=body,
                    ) as resp:
                        resp.raise_for_status()
                        async for chunk in resp.aiter_bytes(chunk_size=4096):
                            if chunk:
                                collected.append(chunk)
                chunks = collected
                break
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                log.warning("indicf5 tts stream transient error (attempt %d/%d): %s",
                            attempt + 1, _TTS_ATTEMPTS, e)
            except httpx.HTTPStatusError as e:
                # Retry only transient 5xx; surface 4xx (bad request) at once.
                # A mid-stream failure after some chunks were already read also
                # lands here (or above) and discards `collected` entirely —
                # there is no partial-audio recovery, the whole request retries.
                if e.response.status_code >= 500 and attempt + 1 < _TTS_ATTEMPTS:
                    last_exc = e
                    log.warning("indicf5 tts stream %s (attempt %d/%d); retrying",
                                e.response.status_code, attempt + 1, _TTS_ATTEMPTS)
                    continue
                raise
        if chunks is None:
            raise last_exc  # type: ignore[misc]  # set whenever the loop didn't break
        audio_bytes = b"".join(chunks)
        if not audio_bytes:
            raise RuntimeError("IndicF5 TTS returned empty audio")

        # Raw PCM16 mono, no WAV header on this endpoint — IndicF5 always
        # renders at its fixed native rate (24kHz), same as the old /tts path,
        # just no longer readable from a header since there isn't one anymore.
        actual_rate = _NATIVE_SAMPLE_RATE
        # The pipeline sends TTS audio to the sink as-is and the bridges are
        # wired for the REQUESTED rate (TTSConfig(sample_rate=16000) →
        # 16k→8k telephony conversion happens there) — TTSResult.sample_rate
        # is informational, not honored. 24kHz passed through unresampled
        # would play at ~2/3 speed, so convert here, like Sarvam's API does
        # natively when asked for a speech_sample_rate.
        if actual_rate != config.sample_rate:
            audio_bytes, _ = resample_pcm16(audio_bytes, actual_rate, config.sample_rate)
        duration_ms = (len(audio_bytes) / max(config.sample_rate * 2, 1)) * 1000.0
        return TTSResult(
            audio=audio_bytes,
            duration_ms=duration_ms,
            sample_rate=config.sample_rate,
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
