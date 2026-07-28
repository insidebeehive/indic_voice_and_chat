from __future__ import annotations

import base64

import httpx
import pytest
import respx
from httpx import Response

from src.interfaces.tts import TTSConfig
from src.providers.tts.indicf5 import IndicF5TTSAdapter, _STREAM_PATH
from src.providers.tts.sarvam import SARVAM_BASE_URL, SarvamTTSAdapter


@pytest.fixture
def adapter() -> SarvamTTSAdapter:
    return SarvamTTSAdapter({"api_key": "test-key"})


_INDICF5_URL = "https://pod-8000.proxy.runpod.net"


@pytest.fixture
def indicf5() -> IndicF5TTSAdapter:
    return IndicF5TTSAdapter({"base_url": _INDICF5_URL})


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_decodes_base64_audio(adapter: SarvamTTSAdapter) -> None:
    pcm = b"\x01\x02\x03\x04" * 1000  # 4000 bytes => 0.125s @ 16kHz mono 16-bit
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(200, json={"audios": [base64.b64encode(pcm).decode()]})
    )
    result = await adapter.synthesize("Namaste", TTSConfig(language="hi-IN"))
    assert result.audio == pcm
    assert result.sample_rate == 16000
    assert result.duration_ms == pytest.approx(125.0, rel=0.1)


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_retries_once_on_transient_hang(adapter: SarvamTTSAdapter) -> None:
    # A transient hang (read timeout) must be retried, not bubble up as a turn
    # stall. The retry recovers and audio is returned.
    pcm = b"\x01\x02" * 8
    route = respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        side_effect=[
            httpx.ReadTimeout("hang"),
            Response(200, json={"audios": [base64.b64encode(pcm).decode()]}),
        ]
    )
    result = await adapter.synthesize("Namaste", TTSConfig(language="hi-IN"))
    assert route.call_count == 2
    assert result.audio == pcm


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_raises_after_exhausting_retries(adapter: SarvamTTSAdapter) -> None:
    route = respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        side_effect=[httpx.ReadTimeout("a"), httpx.ReadTimeout("b")]
    )
    with pytest.raises(httpx.TimeoutException):
        await adapter.synthesize("Namaste", TTSConfig())
    assert route.call_count == 2  # initial + 1 retry, then give up


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_does_not_retry_4xx(adapter: SarvamTTSAdapter) -> None:
    route = respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(401, json={"error": "bad key"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        await adapter.synthesize("Namaste", TTSConfig())
    assert route.call_count == 1  # auth errors surface immediately


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_raises_when_no_audio(adapter: SarvamTTSAdapter) -> None:
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(200, json={"audios": []})
    )
    with pytest.raises(RuntimeError):
        await adapter.synthesize("Namaste", TTSConfig())


@pytest.mark.asyncio
@respx.mock
async def test_synthesize_stream_yields_per_segment(adapter: SarvamTTSAdapter) -> None:
    audio_a = base64.b64encode(b"AAAA" * 100).decode()
    audio_b = base64.b64encode(b"BBBB" * 100).decode()

    responses = iter([
        Response(200, json={"audios": [audio_a]}),
        Response(200, json={"audios": [audio_b]}),
    ])
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        side_effect=lambda req: next(responses)
    )

    async def segments():
        yield "Pehla."
        yield ""  # empty should be skipped
        yield "Doosra."

    out: list[bytes] = []
    async for chunk in adapter.synthesize_stream(segments(), TTSConfig()):
        out.append(chunk)
    assert len(out) == 2
    assert out[0] == b"AAAA" * 100
    assert out[1] == b"BBBB" * 100


def test_get_available_voices(adapter: SarvamTTSAdapter) -> None:
    hi = adapter.get_available_voices("hi-IN")
    assert any(v["voice_id"] == "anushka" for v in hi)
    assert adapter.get_available_voices("xx-XX") == []


@pytest.mark.asyncio
async def test_constructor_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SarvamTTSAdapter({})


@pytest.mark.asyncio
@respx.mock
async def test_sarvam_passes_extra_pronunciations_to_normalize(
    adapter: SarvamTTSAdapter, monkeypatch,
) -> None:
    captured = {}

    def _fake_normalize(text, language=None, extra=None):
        captured["extra"] = extra
        return text

    monkeypatch.setattr("src.providers.tts.sarvam.normalize_for_tts", _fake_normalize)
    pcm = b"\x01\x02\x03\x04" * 1000
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(200, json={"audios": [base64.b64encode(pcm).decode()]})
    )

    config = TTSConfig(language="hi-IN", extra_pronunciations={"XYZ": "एक्स वाय ज़ेड"})
    await adapter.synthesize("hello XYZ", config)

    assert captured["extra"] == {"XYZ": "एक्स वाय ज़ेड"}


# --- IndicF5 (self-hosted fine-tuned voice server) -----------------------


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_synthesize_resamples_24k_to_requested_rate(indicf5: IndicF5TTSAdapter) -> None:
    # IndicF5 always renders at its fixed native rate (24kHz) regardless of
    # the request, but the pipeline/bridges are wired for the REQUESTED rate
    # and don't honor TTSResult.sample_rate — unresampled 24k plays at ~2/3
    # speed. The adapter must resample to the request. Raw PCM, no WAV header.
    pcm = b"\x01\x02" * 2400  # 4800 bytes => 0.1s @ 24kHz mono 16-bit
    route = respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(
        return_value=Response(200, content=pcm))
    result = await indicf5.synthesize("नमस्कार", TTSConfig(language="mr-IN", sample_rate=16000))
    assert result.sample_rate == 16000
    # 0.1s of audio at 16kHz mono 16-bit = 3200 bytes (duration preserved).
    assert len(result.audio) == pytest.approx(3200, abs=8)
    assert result.duration_ms == pytest.approx(100.0, rel=0.05)
    body = route.calls[0].request.read()
    import json as _json
    payload = _json.loads(body)
    assert payload["lang"] == "mr"  # BCP-47 mr-IN mapped to the server's bare code


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_no_resample_when_rates_match(indicf5: IndicF5TTSAdapter) -> None:
    pcm = b"\x01\x02" * 1600  # 3200 bytes @ 24kHz (IndicF5's fixed native rate)
    respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(
        return_value=Response(200, content=pcm))
    result = await indicf5.synthesize("hi", TTSConfig(language="hi-IN", sample_rate=24000))
    assert result.audio == pcm  # byte-identical, no needless conversion
    assert result.sample_rate == 24000


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_retries_5xx_then_succeeds(indicf5: IndicF5TTSAdapter) -> None:
    pcm = b"\x00\x01" * 100
    route = respx.post(f"{_INDICF5_URL}{_STREAM_PATH}")
    # 24k (native) here so no resample obscures the byte-identity check —
    # this test is about retry behavior; resampling has its own test above.
    route.side_effect = [Response(503), Response(200, content=pcm)]
    result = await indicf5.synthesize("Namaste", TTSConfig(language="hi-IN", sample_rate=24000))
    assert result.audio == pcm
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_does_not_retry_4xx(indicf5: IndicF5TTSAdapter) -> None:
    route = respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(return_value=Response(422))
    with pytest.raises(httpx.HTTPStatusError):
        await indicf5.synthesize("hi", TTSConfig(language="hi-IN"))
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_reassembles_multiple_stream_chunks(indicf5: IndicF5TTSAdapter) -> None:
    # Content larger than the adapter's 4096-byte chunk_size forces httpx's
    # aiter_bytes to split delivery into multiple chunks — this proves the
    # adapter accumulates ALL chunks (not just the first) into the final
    # audio, rather than silently truncating a multi-chunk stream.
    pcm = bytes(range(256)) * 50  # 12800 bytes, spans multiple 4096-byte chunks
    respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(
        return_value=Response(200, content=pcm))
    result = await indicf5.synthesize("hi", TTSConfig(language="hi-IN", sample_rate=24000))
    assert result.audio == pcm  # exact reassembly, no truncation/corruption


def test_indicf5_voices_cover_indic_languages(indicf5: IndicF5TTSAdapter) -> None:
    for lang in ("mr-IN", "hi-IN", "ta-IN", "bn-IN"):
        voices = indicf5.get_available_voices(lang)
        assert voices and voices[0]["voice_id"] == "indicf5"
    assert indicf5.get_available_voices("fr-FR") == []


@pytest.mark.asyncio
async def test_indicf5_constructor_requires_url(monkeypatch) -> None:
    monkeypatch.delenv("INDICF5_TTS_URL", raising=False)
    with pytest.raises(ValueError):
        IndicF5TTSAdapter({})


def test_indicf5_registered_in_provider_registry() -> None:
    from src.providers import TTS_PROVIDERS
    assert TTS_PROVIDERS["indicf5"] is IndicF5TTSAdapter


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_passes_extra_pronunciations_to_normalize(
    indicf5: IndicF5TTSAdapter, monkeypatch,
) -> None:
    captured = {}

    def _fake_normalize(text, language=None, extra=None):
        captured["extra"] = extra
        return text

    monkeypatch.setattr("src.providers.tts.indicf5.normalize_for_tts", _fake_normalize)
    pcm = b"\x01\x02" * 100
    respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(return_value=Response(200, content=pcm))

    config = TTSConfig(language="hi-IN", extra_pronunciations={"XYZ": "एक्स वाय ज़ेड"})
    await indicf5.synthesize("hello XYZ", config)

    assert captured["extra"] == {"XYZ": "एक्स वाय ज़ेड"}
