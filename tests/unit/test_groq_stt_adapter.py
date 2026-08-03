from __future__ import annotations

import pytest
import respx
from httpx import Response

from src.interfaces.stt import STTConfig
from src.providers.stt.groq_whisper import (
    GROQ_BASE_URL,
    GroqSTTAdapter,
)


@pytest.fixture
def adapter() -> GroqSTTAdapter:
    return GroqSTTAdapter({"api_key": "test-key"})


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_returns_parsed_result(adapter: GroqSTTAdapter) -> None:
    respx.post(f"{GROQ_BASE_URL}/audio/transcriptions").mock(
        return_value=Response(
            200,
            json={
                "text": "Namaste, kaise ho?",
                "language": "hi",
                "duration": 1.2,
            },
        )
    )
    result = await adapter.transcribe(b"\x00\x00", STTConfig(language="hi-IN"))
    assert result.text == "Namaste, kaise ho?"
    assert result.language == "hi"
    assert result.confidence == 1.0


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_strips_language_region(adapter: GroqSTTAdapter) -> None:
    """Whisper expects ISO-639-1, not Sarvam's ``hi-IN`` form."""
    route = respx.post(f"{GROQ_BASE_URL}/audio/transcriptions").mock(
        return_value=Response(200, json={"text": "ok", "language": "hi"})
    )
    await adapter.transcribe(b"\x00", STTConfig(language="hi-IN"))
    # respx captured the request — check the form-data has stripped region
    request = route.calls.last.request
    body = request.content.decode("latin-1", errors="replace")
    assert "name=\"language\"" in body
    assert "hi-IN" not in body
    assert "hi" in body


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_empty_response(adapter: GroqSTTAdapter) -> None:
    respx.post(f"{GROQ_BASE_URL}/audio/transcriptions").mock(
        return_value=Response(200, json={"text": "", "language": "hi"})
    )
    result = await adapter.transcribe(b"\x00", STTConfig())
    assert result.text == ""
    assert result.confidence == 0.0


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_stream_yields_single_buffered_result(adapter: GroqSTTAdapter) -> None:
    respx.post(f"{GROQ_BASE_URL}/audio/transcriptions").mock(
        return_value=Response(200, json={"text": "buffered", "language": "en"})
    )

    async def chunks():
        yield b"\x00\x01"
        yield b"\x02\x03"

    results = []
    async for r in adapter.transcribe_stream(chunks(), STTConfig()):
        results.append(r)
    assert len(results) == 1
    assert results[0].text == "buffered"


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_wraps_raw_pcm_in_wav(adapter: GroqSTTAdapter) -> None:
    """Live bridge sends headerless PCM16; Groq needs a real WAV container.

    Regression: raw PCM mislabelled ``audio.wav`` made Whisper return
    ``400 could not process file``.
    """
    route = respx.post(f"{GROQ_BASE_URL}/audio/transcriptions").mock(
        return_value=Response(200, json={"text": "ok", "language": "hi"})
    )
    raw_pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"  # no RIFF/WAVE header
    await adapter.transcribe(raw_pcm, STTConfig(sample_rate=16000))

    body = route.calls.last.request.content
    # The multipart file part must now carry a valid WAV container: a RIFF
    # header, the WAVE/fmt /data chunks, then the original PCM as the payload.
    assert b"RIFF" in body and b"WAVE" in body and b"data" in body
    riff = body.index(b"RIFF")
    assert body.index(raw_pcm) > riff  # PCM sits inside the container, after the header


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_passes_through_existing_wav(adapter: GroqSTTAdapter) -> None:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8)
    existing = buf.getvalue()

    route = respx.post(f"{GROQ_BASE_URL}/audio/transcriptions").mock(
        return_value=Response(200, json={"text": "ok", "language": "hi"})
    )
    await adapter.transcribe(existing, STTConfig())
    # A pre-framed WAV is forwarded unchanged (no double-wrapping).
    assert existing in route.calls.last.request.content


@pytest.mark.asyncio
@respx.mock
async def test_transcribe_400_includes_response_body_in_raised_error(
    adapter: GroqSTTAdapter,
) -> None:
    """httpx's default HTTPStatusError message is just the status line — the
    actual rejection reason lives in the response body and must be surfaced,
    or a 400 is undiagnosable from the transcript/logs alone (this is exactly
    what happened on Stage: only 'Client error 400 Bad Request' was visible,
    with no indication of what Groq actually objected to)."""
    respx.post(f"{GROQ_BASE_URL}/audio/transcriptions").mock(
        return_value=Response(
            400, json={"error": {"message": "Audio file is too short"}}
        )
    )
    with pytest.raises(Exception) as exc_info:
        await adapter.transcribe(b"\x00\x00", STTConfig(language="hi-IN"))
    assert "Audio file is too short" in str(exc_info.value)


@pytest.mark.asyncio
async def test_constructor_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GroqSTTAdapter({})


def test_supported_languages_includes_indic(adapter: GroqSTTAdapter) -> None:
    langs = adapter.get_supported_languages()
    assert "hi" in langs
    assert "en" in langs
    assert "ta" in langs
