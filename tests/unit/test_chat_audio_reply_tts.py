"""Voice-note reply synthesis: an inbound `audio` turn gets an audio reply too.

Mirrors the customer's modality (audio-in -> audio-out, text-in -> text-out
only), always keeps the text reply, and never lets TTS failure/timeout/
misconfiguration break the turn. See `_synthesize_reply_audio` in
src/api/chat.py for the full contract.

Uses fakes throughout — no real TTS provider or media store is ever
contacted.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import struct
import wave
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import chat as chat_api
from src.models.chat import ChatMessage, ChatSession
from src.models.database import Base


def _make_pcm16(num_samples: int = 400, *, sample_rate: int = 16000) -> bytes:
    """Realistic headerless PCM16 mono: a low-frequency sine wave, not
    arbitrary bytes — this is what every real TTS provider actually returns
    (see the module docstrings of src/providers/tts/*.py), and it's what
    `_pcm16_to_wav` (src/api/chat.py) must wrap into a playable WAV. A fake
    that returned opaque placeholder bytes (the old `b"FAKE-MP3-BYTES"`)
    could never have caught the real defect: no format survives being
    opaque."""
    freq = 220.0
    samples = [
        int(3000 * math.sin(2 * math.pi * freq * (i / sample_rate)))
        for i in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


class _FakeMediaStore:
    def __init__(self):
        self.uploaded: list[tuple[str, str, bytes]] = []

    async def upload(self, data, key, content_type):
        self.uploaded.append((key, content_type, data))

    async def signed_url(self, key, ttl_seconds):
        return f"https://cdn/{key}"


@dataclass
class _FakeResp:
    response_text: str = "I heard you"
    sources_used: list = field(default_factory=list)
    suggested_followups: list = field(default_factory=list)
    action: str = "none"
    language: str = "hi"
    confidence: str = "high"


@dataclass
class _FakeTurnResult:
    response: _FakeResp = field(default_factory=_FakeResp)
    escalation: Optional[dict] = None
    call_offer: Optional[dict] = None
    input_tokens: int = 0
    output_tokens: int = 0
    llm_provider: str = ""
    llm_model: str = ""
    metrics: object = None


class _FakeTTSResult:
    def __init__(self, audio: bytes | None = None, *, sample_rate: int = 16000):
        # Real providers return raw PCM16 mono, never a compressed format
        # (see src/providers/tts/*.py) — the fake must too, at a STATED
        # sample rate, so tests can assert the WAV wrapper (`_pcm16_to_wav`)
        # actually round-trips the real bytes/rate, not a fake's own idea of
        # "audio".
        self.audio = audio if audio is not None else _make_pcm16(sample_rate=sample_rate)
        self.duration_ms = 500.0
        self.sample_rate = sample_rate


class _FakeTTSProvider:
    """Stands in for an ITTSProvider. `raise_exc`/`hang_s` model the two
    failure modes `_synthesize_reply_audio` must swallow. `audio`/`sample_rate`
    let a test hand back specific (e.g. empty, or odd-length) bytes instead of
    the default realistic PCM16 clip."""

    def __init__(
        self, *, raise_exc: Exception | None = None, hang_s: float = 0.0,
        audio: bytes | None = None, sample_rate: int = 16000,
    ):
        self.raise_exc = raise_exc
        self.hang_s = hang_s
        self.audio = audio
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, object]] = []

    async def synthesize(self, text, config):
        self.calls.append((text, config))
        if self.hang_s:
            await asyncio.sleep(self.hang_s)
        if self.raise_exc:
            raise self.raise_exc
        return _FakeTTSResult(self.audio, sample_rate=self.sample_rate)

    async def synthesize_stream(self, text_stream, config):
        raise NotImplementedError

    def get_available_voices(self, language):
        return []


class _FakeTTSProviders:
    """Stands in for `TenantProviders` — only the `get_tts` method the new
    code calls."""

    def __init__(self, provider: _FakeTTSProvider | None = None, *, raise_on_get: bool = False):
        self.provider = provider
        self.raise_on_get = raise_on_get
        self.get_tts_calls: list = []

    def get_tts(self, tenant):
        self.get_tts_calls.append(tenant)
        if self.raise_on_get:
            raise RuntimeError("no TTS provider configured for this tenant")
        return self.provider


def _make_fake_tenant(*, pronunciation_overrides=None):
    tenant = MagicMock()
    tenant.id = "t1"
    tenant.slug = "demo"
    tenant.settings.chat_support.chat_idle_timeout_seconds = 300
    # MUST be a real dict-or-None: text_normalize.apply_pronunciations does
    # `{**DEFAULT_PRONUNCIATIONS, **(extra or {})}`, which raises on a bare
    # MagicMock — unlike most attributes here this one is real production
    # code, not mocked away.
    tenant.settings.pronunciation_overrides = pronunciation_overrides
    return tenant


@pytest_asyncio.fixture
async def ws_ctx():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async with sm() as db:
        db.add(ChatSession(id="sess1", tenant_id="t1", language="hi",
                           status="active", mode="ai", extra_data={}))
        await db.commit()

    media_store = _FakeMediaStore()
    chat_api.set_media_store(media_store)
    chat_api.set_chat_sessionmaker(sm)

    fake_agent = MagicMock()
    fake_agent.handle_message = AsyncMock(return_value=_FakeTurnResult())
    fake_agent._llm = None  # bare MagicMock() would auto-vivify a truthy _llm and win over .llm below
    fake_agent.llm = MagicMock()
    fake_agent.llm.transcribe_audio = AsyncMock(return_value="hello there")
    fake_agent.session = MagicMock()

    async def fake_factory(tenant, scoped_id, *, customer_id=None, ticket_id=None):
        return fake_agent

    chat_api.set_chatbot_factory(fake_factory)

    yield sm, media_store, fake_agent

    chat_api.set_media_store(None)
    chat_api.set_chat_sessionmaker(None)
    chat_api.set_chatbot_factory(None)
    chat_api.set_tts_providers(None)
    await engine.dispose()


def _send_audio_and_collect(fake_tenant, *, extra_frames_expected: int = 0):
    """Connects, sends one base64 audio frame, and returns the parsed frames
    up to and including the `message` reply (typing, audio_ack, message)."""
    import src.auth.middleware as mw

    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)
        audio_bytes = b"fake_audio_data"
        encoded = base64.b64encode(audio_bytes).decode()
        frames = []
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({
                "type": "audio", "data": encoded, "mime": "audio/webm;codecs=opus",
            }))
            # typing, audio_ack, message (the reply) — always in this order.
            for _ in range(3 + extra_frames_expected):
                frames.append(json.loads(ws.receive_text()))
        return frames


@pytest.mark.asyncio
async def test_audio_turn_with_working_tts_gets_audio_reply(ws_ctx):
    """Inbound audio -> reply `message` frame carries audio_url/audio_mime,
    and the synthesized clip is persisted + retrievable via the media store —
    AND is an actually-valid, actually-playable WAV file wrapping the exact
    PCM16 bytes the provider returned. This is the test that would have
    caught the shipped defect (headerless PCM served as `audio/mpeg`)."""
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="yahan hai", language="hi")))

    raw_pcm = _make_pcm16(sample_rate=16000)
    provider = _FakeTTSProvider(audio=raw_pcm, sample_rate=16000)
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    typing, ack, reply = frames[0], frames[1], frames[2]
    assert typing["type"] == "typing"
    assert ack["type"] == "audio_ack"
    assert reply["type"] == "message"
    assert reply["text"] == "yahan hai"
    assert reply["audio_url"].startswith("/api/v1/chat/media/")
    assert reply["audio_mime"] == "audio/wav"

    # Two uploads: the inbound recording, then the synthesized reply.
    assert len(media_store.uploaded) == 2
    reply_key, reply_mime, reply_bytes = media_store.uploaded[1]
    assert reply_mime == "audio/wav"

    # The uploaded/advertised mime must match the ACTUAL bytes: parse them
    # back with the stdlib `wave` module (not a string/magic-number sniff) and
    # check every parameter a player relies on, plus a full round-trip of the
    # PCM frame data back to what the provider returned.
    with wave.open(io.BytesIO(reply_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2  # PCM16
        assert wav_file.getframerate() == 16000
        frames_out = wav_file.readframes(wav_file.getnframes())
    assert frames_out == raw_pcm
    assert provider.calls, "tts provider was never called"

    # Persisted on the AGENT's row, retrievable the same way inbound audio is.
    async with sm() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == "sess1", ChatMessage.role == "agent")
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].media_url == reply_key
    assert rows[0].media_mime == "audio/wav"
    assert rows[0].type == "audio"
    assert reply["audio_url"] == f"/api/v1/chat/media/{rows[0].id}"


@pytest.mark.asyncio
async def test_text_turn_has_no_audio_fields(ws_ctx):
    """Mirroring is pinned: a text-in turn NEVER gets synthesized audio, even
    with a perfectly good TTS provider configured."""
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="hello back", language="en")))

    provider = _FakeTTSProvider()
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    import src.auth.middleware as mw
    fake_tenant = _make_fake_tenant()
    with patch.object(mw, "tenant_from_id", AsyncMock(return_value=fake_tenant)):
        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        client = TestClient(app)
        with client.websocket_connect("/api/v1/chat/ws/sess1") as ws:
            ws.send_text(json.dumps({"type": "message", "text": "hi"}))
            typing = json.loads(ws.receive_text())
            reply = json.loads(ws.receive_text())
    assert typing["type"] == "typing"
    assert reply["type"] == "message"
    assert reply["text"] == "hello back"
    assert "audio_url" not in reply
    assert "audio_mime" not in reply
    assert not provider.calls, "TTS must never be invoked for a text-in turn"
    assert len(media_store.uploaded) == 0


@pytest.mark.asyncio
async def test_tts_provider_raises_falls_back_to_text_only(ws_ctx):
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="answer", language="hi")))

    provider = _FakeTTSProvider(raise_exc=RuntimeError("provider outage"))
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert reply["text"] == "answer"
    assert "audio_url" not in reply
    assert "audio_mime" not in reply
    # Only the inbound recording was uploaded — synthesis never produced bytes to store.
    assert len(media_store.uploaded) == 1


@pytest.mark.asyncio
async def test_tts_timeout_falls_back_to_text_only(ws_ctx, monkeypatch):
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="answer", language="hi")))

    monkeypatch.setattr(chat_api, "_TTS_SYNTH_TIMEOUT_S", 0.05)
    provider = _FakeTTSProvider(hang_s=5.0)
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert reply["text"] == "answer"
    assert "audio_url" not in reply
    assert "audio_mime" not in reply
    assert len(media_store.uploaded) == 1


@pytest.mark.asyncio
async def test_tenant_with_no_tts_provider_configured_falls_back_to_text_only(ws_ctx):
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="answer", language="hi")))

    chat_api.set_tts_providers(_FakeTTSProviders(raise_on_get=True))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert reply["text"] == "answer"
    assert "audio_url" not in reply
    assert "audio_mime" not in reply


@pytest.mark.asyncio
async def test_no_tts_providers_wired_at_all_falls_back_to_text_only(ws_ctx):
    """The common case for most deployments/tests: `set_tts_providers` was
    never called at all (module default `None`)."""
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="answer", language="hi")))
    # Deliberately do NOT call chat_api.set_tts_providers — module default.

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert "audio_url" not in reply
    assert "audio_mime" not in reply


@pytest.mark.asyncio
async def test_empty_reply_text_skips_synthesis(ws_ctx):
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="", language="hi")))

    provider = _FakeTTSProvider()
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert "audio_url" not in reply
    assert not provider.calls, "synthesis must not be attempted for empty reply text"


@pytest.mark.asyncio
async def test_over_cap_reply_text_skips_synthesis(ws_ctx):
    sm, media_store, fake_agent = ws_ctx
    long_text = "x" * (chat_api._TTS_MAX_REPLY_CHARS + 1)
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text=long_text, language="hi")))

    provider = _FakeTTSProvider()
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert "audio_url" not in reply
    assert not provider.calls, "synthesis must not be attempted over the character cap"


@pytest.mark.asyncio
async def test_at_cap_reply_text_still_synthesizes(ws_ctx):
    """Boundary check: exactly at the cap still synthesizes (only STRICTLY
    over the cap is skipped)."""
    sm, media_store, fake_agent = ws_ctx
    exact_text = "x" * chat_api._TTS_MAX_REPLY_CHARS
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text=exact_text, language="hi")))

    provider = _FakeTTSProvider()
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert reply["audio_url"].startswith("/api/v1/chat/media/")
    assert provider.calls


@pytest.mark.asyncio
async def test_normalize_for_tts_applied_with_tenant_pronunciation_overrides(ws_ctx):
    """`normalize_for_tts` must actually be CALLED (not just "audio appears")
    with the tenant's own `pronunciation_overrides` threaded through."""
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="WhatsApp par bhejo", language="hi")))

    provider = _FakeTTSProvider()
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    overrides = {"Foo": "Phonetic-Foo"}
    fake_tenant = _make_fake_tenant(pronunciation_overrides=overrides)

    with patch.object(chat_api, "normalize_for_tts", wraps=chat_api.normalize_for_tts) as spy:
        frames = _send_audio_and_collect(fake_tenant)

    reply = frames[2]
    assert reply["audio_url"].startswith("/api/v1/chat/media/")
    spy.assert_called_once_with("WhatsApp par bhejo", "hi", extra=overrides)
    # And the TTSConfig handed to the provider carries a bcp47-formatted
    # language, not the bare base code. It deliberately does NOT carry
    # `extra_pronunciations` — the overrides are already burned into the text
    # passed to `synthesize` (asserted above via the spy), and forwarding
    # them through the config as well would make providers that internally
    # re-run `normalize_for_tts` (sarvam, indicf5) apply them a SECOND time
    # against already-normalized text (see the comment in
    # `_synthesize_reply_audio_uncapped`, src/api/chat.py).
    (_text, config), = provider.calls
    assert config.language == "hi-IN"
    assert config.extra_pronunciations is None


@pytest.mark.asyncio
async def test_tts_upload_failure_falls_back_to_text_only(ws_ctx):
    """A working TTS provider but a media-store upload failure for the
    synthesized clip must still leave the turn's text reply intact."""
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="answer", language="hi")))

    provider = _FakeTTSProvider()
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    real_upload = media_store.upload
    calls = {"n": 0}

    async def flaky_upload(data, key, content_type):
        calls["n"] += 1
        if calls["n"] == 2:  # the reply's own upload (1st is the inbound recording)
            raise RuntimeError("s3 outage")
        return await real_upload(data, key, content_type)

    media_store.upload = flaky_upload

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert reply["text"] == "answer"
    assert "audio_url" not in reply
    assert "audio_mime" not in reply


@pytest.mark.asyncio
async def test_empty_audio_from_provider_falls_back_to_text_only(ws_ctx):
    """Finding 3: a provider returning `TTSResult(audio=b"")` (a 200 with an
    empty body — Azure can do this on some failure modes) must be treated as
    a synthesis failure, not a zero-byte clip with a play button. No upload
    of the (non-existent) reply audio should happen."""
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="answer", language="hi")))

    provider = _FakeTTSProvider(audio=b"")
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert reply["text"] == "answer"
    assert "audio_url" not in reply
    assert "audio_mime" not in reply
    assert provider.calls, "the provider was still called — it just returned nothing"
    # Only the inbound recording was uploaded — no zero-byte reply clip.
    assert len(media_store.uploaded) == 1


@pytest.mark.asyncio
async def test_odd_length_pcm_from_provider_still_produces_valid_wav(ws_ctx):
    """Finding 1: an odd PCM16 byte count (a truncated trailing sample) must
    be handled — the trailing byte dropped — rather than silently producing a
    WAV whose header/data chunk sizes describe a fractional frame."""
    sm, media_store, fake_agent = ws_ctx
    fake_agent.handle_message = AsyncMock(
        return_value=_FakeTurnResult(response=_FakeResp(response_text="answer", language="hi")))

    even_pcm = _make_pcm16(num_samples=100, sample_rate=16000)
    odd_pcm = even_pcm + b"\x7f"  # one dangling, incomplete sample byte
    provider = _FakeTTSProvider(audio=odd_pcm, sample_rate=16000)
    chat_api.set_tts_providers(_FakeTTSProviders(provider))

    fake_tenant = _make_fake_tenant()
    frames = _send_audio_and_collect(fake_tenant)
    reply = frames[2]
    assert reply["type"] == "message"
    assert reply["audio_url"].startswith("/api/v1/chat/media/")
    assert reply["audio_mime"] == "audio/wav"

    reply_key, reply_mime, reply_bytes = media_store.uploaded[1]
    with wave.open(io.BytesIO(reply_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        frames_out = wav_file.readframes(wav_file.getnframes())
    # The dangling byte was dropped, not smuggled into the data chunk.
    assert frames_out == even_pcm
    assert len(frames_out) % 2 == 0


def test_pcm16_to_wav_round_trips_and_reports_expected_parameters():
    """Direct unit test of `_pcm16_to_wav` (no WS/agent/media-store machinery)
    — this is the manual `wave`-module verification the fix is built around,
    pinned as an actual test rather than only a one-off script."""
    pcm = _make_pcm16(num_samples=1600, sample_rate=16000)  # 100ms @ 16kHz
    wav_bytes = chat_api._pcm16_to_wav(pcm, 16000)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 1600
        assert wav_file.readframes(1600) == pcm
    # 44-byte canonical PCM WAV header + the raw PCM payload, no more/less.
    assert len(wav_bytes) == 44 + len(pcm)


def test_pcm16_to_wav_drops_trailing_odd_byte():
    pcm = _make_pcm16(num_samples=10, sample_rate=16000)
    wav_bytes = chat_api._pcm16_to_wav(pcm + b"\x01", 16000)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.readframes(wav_file.getnframes()) == pcm
