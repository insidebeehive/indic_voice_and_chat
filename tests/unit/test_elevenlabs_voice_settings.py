"""Per-tenant ElevenLabs voice_settings (stability/similarity_boost/style/
use_speaker_boost) — see TenantTTSConfig (src/config_tenant.py) and
ElevenLabsTTSAdapter (src/providers/tts/elevenlabs.py).

Byte-identical-when-unconfigured is the main guarantee under test: a tenant
that never sets these fields must send EXACTLY the historical
{"stability": 0.5, "similarity_boost": 0.75} body, with no style/
use_speaker_boost keys at all — same pattern as
tests/unit/test_gemini_tts_no_audio.py's httpx.MockTransport use.
"""
from __future__ import annotations

import json

import httpx
import pytest

from src.interfaces.tts import TTSConfig
from src.providers.tts import elevenlabs as el


def _install_mock_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        el.httpx, "AsyncClient",
        lambda *a, **kw: real_client(transport=httpx.MockTransport(handler), **kw))


def _capture_handler(captured: list, *, audio: bytes = b"\x00\x01\x02\x03"):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=audio)
    return handler


@pytest.mark.asyncio
async def test_synthesize_defaults_when_nothing_configured(monkeypatch) -> None:
    captured: list = []
    _install_mock_transport(monkeypatch, _capture_handler(captured))
    adapter = el.ElevenLabsTTSAdapter({"api_key": "k"})
    await adapter.synthesize("hello", TTSConfig())
    assert len(captured) == 1
    assert captured[0]["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.75}


@pytest.mark.asyncio
async def test_synthesize_configured_values_override_defaults(monkeypatch) -> None:
    captured: list = []
    _install_mock_transport(monkeypatch, _capture_handler(captured))
    adapter = el.ElevenLabsTTSAdapter({
        "api_key": "k", "stability": 0.9, "similarity_boost": 0.2,
    })
    await adapter.synthesize("hello", TTSConfig())
    assert captured[0]["voice_settings"] == {"stability": 0.9, "similarity_boost": 0.2}


@pytest.mark.asyncio
async def test_synthesize_style_and_speaker_boost_only_when_set(monkeypatch) -> None:
    captured: list = []
    _install_mock_transport(monkeypatch, _capture_handler(captured))
    adapter = el.ElevenLabsTTSAdapter({
        "api_key": "k", "style": 0.3, "use_speaker_boost": True,
    })
    await adapter.synthesize("hello", TTSConfig())
    assert captured[0]["voice_settings"] == {
        "stability": 0.5, "similarity_boost": 0.75,
        "style": 0.3, "use_speaker_boost": True,
    }


@pytest.mark.asyncio
async def test_synthesize_speaker_boost_false_is_still_included(monkeypatch) -> None:
    """use_speaker_boost=False is a real configured value, not "unset" — the
    adapter must check `is not None`, not truthiness, or a tenant that
    explicitly turned it off would silently get the field omitted."""
    captured: list = []
    _install_mock_transport(monkeypatch, _capture_handler(captured))
    adapter = el.ElevenLabsTTSAdapter({"api_key": "k", "use_speaker_boost": False})
    await adapter.synthesize("hello", TTSConfig())
    assert captured[0]["voice_settings"]["use_speaker_boost"] is False


@pytest.mark.asyncio
async def test_synthesize_stream_uses_same_voice_settings(monkeypatch) -> None:
    captured: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=b"\x00\x01")

    _install_mock_transport(monkeypatch, handler)
    adapter = el.ElevenLabsTTSAdapter({"api_key": "k", "stability": 0.8})

    async def _one_segment():
        yield "hello"

    chunks = [c async for c in adapter.synthesize_stream(_one_segment(), TTSConfig())]
    assert b"".join(chunks) == b"\x00\x01"
    assert captured[0]["voice_settings"] == {"stability": 0.8, "similarity_boost": 0.75}


def test_build_voice_settings_ignores_none_values() -> None:
    """merge_provider_config never sends a literal None through (it drops
    unset tenant fields before building the adapter config dict) — but the
    adapter's own builder is defensive about it too, so a config dict built
    any other way (e.g. a stray {"stability": None}) still degrades to the
    default rather than sending a null to ElevenLabs."""
    settings = el._build_voice_settings({"stability": None, "similarity_boost": None,
                                          "style": None, "use_speaker_boost": None})
    assert settings == {"stability": 0.5, "similarity_boost": 0.75}


def test_falls_back_to_default_model_for_foreign_model_id(caplog) -> None:
    """Runtime safety net for a tenant row written before the PATCH-merge fix
    (src/api/tenants.py's _merge_layer_fields): a stored `model` left over
    from a PREVIOUS provider (Sarvam's "bulbul:v3") must not reach the
    ElevenLabs API verbatim -- it warns (model name only) and falls back to
    DEFAULT_MODEL. This is the exact production incident (ElevenLabs
    rejecting a Sarvam model_id with a 4xx) surfacing as a WARNING instead."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.providers.tts.elevenlabs"):
        adapter = el.ElevenLabsTTSAdapter({"api_key": "k", "model": "bulbul:v3"})
    assert adapter._model == el.DEFAULT_MODEL
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "bulbul:v3" in warnings[0].getMessage()


def test_keeps_a_real_eleven_model() -> None:
    adapter = el.ElevenLabsTTSAdapter({"api_key": "k", "model": "eleven_turbo_v2_5"})
    assert adapter._model == "eleven_turbo_v2_5"
