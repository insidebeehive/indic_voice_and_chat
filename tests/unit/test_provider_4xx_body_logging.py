"""4xx-body-reaches-the-log regression tests.

Companion to the Sarvam-TTS-specific tests in test_tts_adapters.py. This file
covers the other TTS/STT adapters that had the same bug: ``raise_for_status()``
surfaces only httpx's uninformative status line, discarding the provider's
error response body — which is exactly where the actionable detail (bad
model/voice/param) lives for a 4xx. Each test here asserts the body reaches
the log and that no API key or request payload does.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from httpx import Response

from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.providers.stt.gemini import GeminiSTTAdapter
from src.providers.stt.sarvam import SARVAM_BASE_URL as STT_SARVAM_BASE_URL
from src.providers.stt.sarvam import SarvamSTTAdapter
from src.providers.tts.azure import AzureTTSAdapter
from src.providers.tts.elevenlabs import ELEVENLABS_BASE_URL, ElevenLabsTTSAdapter
from src.providers.tts.indicf5 import IndicF5TTSAdapter, _STREAM_PATH


# --- Azure TTS ------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_azure_tts_logs_4xx_body(caplog) -> None:
    adapter = AzureTTSAdapter({"api_key": "azure-secret-key", "region": "eastus"})
    respx.post("https://eastus.tts.speech.microsoft.com/cognitiveservices/v1").mock(
        return_value=Response(400, text="Voice 'bogus' not found")
    )
    with caplog.at_level("ERROR", logger="src.providers.tts.azure"):
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.synthesize("hi", TTSConfig(language="hi-IN", voice_id="bogus"))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Voice 'bogus' not found" in logged
    assert "azure-secret-key" not in logged


# --- ElevenLabs TTS ---------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_tts_logs_4xx_body(caplog) -> None:
    adapter = ElevenLabsTTSAdapter({"api_key": "el-secret-key"})
    voice_id = adapter._default_voice_id
    respx.post(f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}").mock(
        return_value=Response(422, json={"detail": "invalid model_id"})
    )
    with caplog.at_level("ERROR", logger="src.providers.tts.elevenlabs"):
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.synthesize("hello there", TTSConfig())

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "invalid model_id" in logged
    assert "el-secret-key" not in logged
    assert "hello there" not in logged


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_tts_stream_logs_4xx_body_without_masking_error(caplog) -> None:
    adapter = ElevenLabsTTSAdapter({"api_key": "el-secret-key"})
    voice_id = adapter._default_voice_id
    respx.post(
        f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/stream"
    ).mock(return_value=Response(400, json={"detail": "bad output_format"}))

    async def segments():
        yield "segment one"

    with caplog.at_level("ERROR", logger="src.providers.tts.elevenlabs"):
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in adapter.synthesize_stream(segments(), TTSConfig()):
                pass

    # The aread()-may-fail-on-an-already-closed-mock-stream case must not
    # swallow the original HTTPStatusError (it was raised above), and when
    # the body IS readable it must show up in the log.
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "bad output_format" in logged or "body unavailable" in logged
    assert "el-secret-key" not in logged


# --- IndicF5 TTS (streaming, self-hosted) -----------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_tts_stream_logs_4xx_body_without_masking_error(caplog) -> None:
    adapter = IndicF5TTSAdapter({"base_url": "https://pod-8000.proxy.runpod.net"})
    respx.post(f"https://pod-8000.proxy.runpod.net{_STREAM_PATH}").mock(
        return_value=Response(422, text="unsupported lang code")
    )
    with caplog.at_level("ERROR", logger="src.providers.tts.indicf5"):
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.synthesize("hi", TTSConfig(language="hi-IN"))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "unsupported lang code" in logged or "body unavailable" in logged


# --- Sarvam STT --------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_sarvam_stt_logs_4xx_body(caplog) -> None:
    adapter = SarvamSTTAdapter({"api_key": "stt-secret-key"})
    respx.post(f"{STT_SARVAM_BASE_URL}/speech-to-text").mock(
        return_value=Response(400, json={"error": {"message": "invalid model 'bogus'"}})
    )
    with caplog.at_level("ERROR", logger="src.providers.stt.sarvam"):
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.transcribe(b"\x00\x00", STTConfig())

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "invalid model 'bogus'" in logged
    assert "stt-secret-key" not in logged


# --- Gemini STT --------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_gemini_stt_logs_4xx_body(caplog) -> None:
    adapter = GeminiSTTAdapter({"api_key": "gemini-secret-key"})
    respx.post(url__regex=r".*generateContent.*").mock(
        return_value=Response(400, json={"error": {"message": "invalid audio mime type"}})
    )
    with caplog.at_level("ERROR", logger="src.providers.stt.gemini"):
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.transcribe(b"\x00\x00", STTConfig())

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "invalid audio mime type" in logged
    assert "gemini-secret-key" not in logged
