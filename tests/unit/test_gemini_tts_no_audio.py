"""Gemini TTS: a 200 with no audio must say why, not raise KeyError('content').

Production (ticket 10922, Kannada reply) failed with a bare KeyError: 'content'
from candidates[0]["content"], which discards Gemini's finishReason."""
from __future__ import annotations

import base64
import logging

import pytest

from src.providers.tts import gemini as g


def _audio_response(pcm: bytes) -> dict:
    return {"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": "audio/L16;rate=24000", "data": base64.b64encode(pcm).decode()}}
    ]}, "finishReason": "STOP"}]}


def test_audio_is_returned_when_present() -> None:
    data = _audio_response(b"\x01\x00\x02\x00")
    assert base64.b64decode(g._inline_audio_b64(data, model="m", text_chars=5)) == b"\x01\x00\x02\x00"


def test_audio_found_even_if_not_the_first_part() -> None:
    data = _audio_response(b"\x01\x00")
    data["candidates"][0]["content"]["parts"].insert(0, {"text": "note"})
    assert base64.b64decode(g._inline_audio_b64(data, model="m", text_chars=5)) == b"\x01\x00"


def test_candidate_without_content_names_the_finish_reason(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=g.__name__)
    data = {"candidates": [{"finishReason": "OTHER", "finishMessage": "no speech produced"}]}
    with pytest.raises(g.GeminiTTSNoAudioError) as exc:
        g._inline_audio_b64(data, model="gemini-2.5-flash-preview-tts", text_chars=120)
    assert "finish_reason=OTHER" in str(exc.value)
    assert "no speech produced" in str(exc.value)
    rec = next(r for r in caplog.records if r.getMessage() == "gemini tts returned no audio")
    assert rec.finish_reason == "OTHER"
    assert rec.text_chars == 120


def test_blocked_prompt_names_block_reason_and_flagged_categories() -> None:
    data = {"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "probability": "HIGH", "blocked": True},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
    ]}}
    with pytest.raises(g.GeminiTTSNoAudioError) as exc:
        g._inline_audio_b64(data, model="m", text_chars=3)
    msg = str(exc.value)
    assert "block_reason=SAFETY" in msg
    assert "HARM_CATEGORY_HARASSMENT" in msg
    assert "HARM_CATEGORY_HATE_SPEECH" not in msg
    assert "candidates=0" in msg


@pytest.mark.asyncio
async def test_synthesize_raises_no_audio_error_not_keyerror(monkeypatch, caplog) -> None:
    """End to end through synthesize(): the 200-without-audio response that
    crashed in production now raises GeminiTTSNoAudioError with the reason,
    and the WARNING carries the text length, never the text."""
    import httpx

    body = {"candidates": [{"finishReason": "OTHER"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        g.httpx, "AsyncClient",
        lambda *a, **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    caplog.set_level(logging.WARNING, logger=g.__name__)
    adapter = g.GeminiTTSAdapter({"api_key": "k"})
    from src.interfaces.tts import TTSConfig
    secret_text = "ಕ್ಷಮಿಸಿ, ಸದ್ಯಕ್ಕೆ human agent"
    with pytest.raises(g.GeminiTTSNoAudioError, match="finish_reason=OTHER"):
        await adapter.synthesize(secret_text, TTSConfig())
    rec = next(r for r in caplog.records if r.getMessage() == "gemini tts returned no audio")
    assert rec.text_chars == len(secret_text)
    assert secret_text not in repr(vars(rec))
