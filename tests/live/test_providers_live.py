"""Smoke tests against real provider APIs.

One minimal call per provider. Designed to be cheap (cents) and quick
(seconds). These verify keys are valid + the adapter wiring is correct
without staging a full conversation.

Run:
    VOX_LIVE_TESTS=1 pytest tests/live/test_providers_live.py -m live -v -s

The ``-s`` shows latency prints so you can sanity-check turnaround times.
"""

from __future__ import annotations

import asyncio
import base64
import time
import wave
from io import BytesIO

import pytest

from src.auth.context import TenantContext
from src.interfaces.llm import LLMConfig, LLMMessage
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.providers.llm.gemini import GeminiLLMAdapter
from src.providers.stt.groq_whisper import GroqSTTAdapter
from src.providers.telephony.twilio import TwilioAdapter
from src.providers.tts.sarvam import SarvamTTSAdapter


pytestmark = pytest.mark.live


# --- Audio helper ------------------------------------------------------


def _synthetic_speech_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a 1-second sine sweep wrapped in WAV for STT smoke tests.

    Whisper will likely return empty or a non-language utterance — that's
    fine, we're verifying connectivity + auth, not transcription accuracy.
    """
    import math
    import struct

    n = int(sample_rate * duration_s)
    samples = b"".join(
        struct.pack(
            "<h",
            int(15000 * math.sin(2 * math.pi * (400 + i / 200) * i / sample_rate)),
        )
        for i in range(n)
    )
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples)
    return buf.getvalue()


# --- STT ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_stt_smoke(dev_tenant_ctx: TenantContext, capsys) -> None:
    api_key = dev_tenant_ctx.secret(dev_tenant_ctx.settings.pipeline.stt.api_key_env)
    adapter = GroqSTTAdapter({
        "api_key": api_key,
        "model": dev_tenant_ctx.settings.pipeline.stt.model or "whisper-large-v3",
    })

    audio = _synthetic_speech_wav(duration_s=1.0)
    t0 = time.perf_counter()
    result = await adapter.transcribe(audio, STTConfig(language="en"))
    dt = (time.perf_counter() - t0) * 1000.0

    with capsys.disabled():
        print(f"\n[groq STT] latency={dt:.0f}ms text={result.text!r} language={result.language!r}")

    # The adapter must return *something* (possibly empty) and not raise.
    assert isinstance(result.text, str)
    assert dt < 10_000, "STT round-trip exceeded 10s"


# --- LLM ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_llm_smoke(dev_tenant_ctx: TenantContext, capsys) -> None:
    api_key = dev_tenant_ctx.secret(dev_tenant_ctx.settings.pipeline.llm.api_key_env)
    adapter = GeminiLLMAdapter({
        "api_key": api_key,
        "model": dev_tenant_ctx.settings.pipeline.llm.model or "gemini-2.0-flash",
    })

    t0 = time.perf_counter()
    result = await adapter.generate(
        [
            LLMMessage(role="system", content="Reply with a single short JSON object."),
            LLMMessage(
                role="user",
                content='Return exactly: {"ok": true, "lang": "hi"}',
            ),
        ],
        LLMConfig(temperature=0.0, max_tokens=64, response_format="json"),
    )
    dt = (time.perf_counter() - t0) * 1000.0

    with capsys.disabled():
        print(f"\n[gemini LLM] latency={dt:.0f}ms text={result.text!r}")
        print(f"  usage={result.usage} finish={result.finish_reason}")

    assert result.text
    assert dt < 15_000, "LLM round-trip exceeded 15s"


@pytest.mark.asyncio
async def test_gemini_llm_streaming_smoke(dev_tenant_ctx: TenantContext, capsys) -> None:
    api_key = dev_tenant_ctx.secret(dev_tenant_ctx.settings.pipeline.llm.api_key_env)
    adapter = GeminiLLMAdapter({"api_key": api_key, "model": "gemini-2.0-flash"})

    t0 = time.perf_counter()
    chunks: list[str] = []
    async for tok in adapter.generate_stream(
        [LLMMessage(role="user", content="Count from one to five, single line.")],
        LLMConfig(temperature=0.0, max_tokens=64, response_format="text"),
    ):
        if not chunks:
            ttft = (time.perf_counter() - t0) * 1000.0
        chunks.append(tok)
    dt = (time.perf_counter() - t0) * 1000.0
    full = "".join(chunks)

    with capsys.disabled():
        print(f"\n[gemini stream] TTFT={ttft:.0f}ms total={dt:.0f}ms chunks={len(chunks)}")
        print(f"  full={full!r}")

    assert chunks, "no chunks streamed"
    assert full.strip()


# --- TTS ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_sarvam_tts_smoke(dev_tenant_ctx: TenantContext, capsys) -> None:
    api_key = dev_tenant_ctx.secret(dev_tenant_ctx.settings.pipeline.tts.api_key_env)
    adapter = SarvamTTSAdapter({"api_key": api_key})

    t0 = time.perf_counter()
    result = await adapter.synthesize(
        "Namaste, yeh ek test message hai.",
        TTSConfig(
            language="hi-IN",
            voice_id=dev_tenant_ctx.settings.pipeline.tts.voice_id or "priya",
            sample_rate=16000,
        ),
    )
    dt = (time.perf_counter() - t0) * 1000.0

    with capsys.disabled():
        print(f"\n[sarvam TTS] latency={dt:.0f}ms audio_bytes={len(result.audio)} "
              f"duration_ms={result.duration_ms:.0f} sr={result.sample_rate}")

    assert len(result.audio) > 100
    assert result.sample_rate == 16000
    assert dt < 15_000


# --- Telephony (credential probe — no real call placed) ----------------


@pytest.mark.asyncio
async def test_twilio_credential_probe(dev_tenant_ctx: TenantContext, capsys) -> None:
    """List recent calls — verifies credentials work without placing a call."""
    sid = dev_tenant_ctx.secret(dev_tenant_ctx.settings.pipeline.telephony.account_sid_env)
    token = dev_tenant_ctx.secret(dev_tenant_ctx.settings.pipeline.telephony.auth_token_env)
    adapter = TwilioAdapter({"account_sid": sid, "auth_token": token})

    # Hit the Twilio API directly through the wrapped client to verify creds.
    # The list call is read-only and free.
    def _list():
        return list(adapter._client.calls.list(limit=1))  # type: ignore[attr-defined]

    t0 = time.perf_counter()
    calls = await asyncio.to_thread(_list)
    dt = (time.perf_counter() - t0) * 1000.0

    with capsys.disabled():
        print(f"\n[twilio creds] latency={dt:.0f}ms account valid; recent calls={len(calls)}")

    assert dt < 10_000
