"""Manual (human softphone) call: stereo split, transcript build, finalize."""

from __future__ import annotations

import audioop
import io
import wave
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.analysis.manual_call import build_transcript, finalize_manual_call
from src.campaign.models import LeadCallOutcome
from src.interfaces.llm import LLMMessage, LLMResult
from src.interfaces.stt import STTConfig, STTResult
from src.models.conversation import Conversation
from src.models.database import Base
from src.models.tenant import Tenant
from src.pipeline.audio_utils import pcm16_to_wav, wav_split_stereo


def _stereo_wav(left: bytes, right: bytes, rate: int = 8000) -> bytes:
    stereo = audioop.tostereo(left, 2, 1.0, 0.0)  # left only
    stereo = audioop.add(stereo, audioop.tostereo(right, 2, 0.0, 1.0), 2)  # + right only
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(stereo)
    return buf.getvalue()


def test_wav_split_stereo_separates_channels() -> None:
    left = b"\x10\x00" * 100
    right = b"\x20\x00" * 100
    wav = _stereo_wav(left, right, rate=8000)
    got_left, got_right, sr = wav_split_stereo(wav)
    assert sr == 8000
    assert got_left == left
    assert got_right == right


def test_wav_split_stereo_mono_returns_same_both() -> None:
    pcm = b"\x05\x00" * 50
    wav = pcm16_to_wav(pcm, 8000)
    a, b, sr = wav_split_stereo(wav)
    assert a == b == pcm


def test_build_transcript_interleaves_by_timestamp() -> None:
    agent = STTResult(text="hello there", confidence=1.0, word_timestamps=[
        {"word": "hello", "start": 0.0}, {"word": "there", "start": 2.0}])
    lead = STTResult(text="hi yes", confidence=1.0, word_timestamps=[
        {"word": "hi", "start": 1.0}, {"word": "yes", "start": 3.0}])
    transcript = build_transcript([("assistant", agent), ("user", lead)])
    # interleaved by start time → assistant(hello) user(hi) assistant(there) user(yes)
    assert [(m.role, m.content) for m in transcript] == [
        ("assistant", "hello"),
        ("user", "hi"),
        ("assistant", "there"),
        ("user", "yes"),
    ]


def test_build_transcript_coarse_fallback_without_timestamps() -> None:
    agent = STTResult(text="namaste", confidence=1.0)
    lead = STTResult(text="haan boliye", confidence=1.0)
    transcript = build_transcript([("assistant", agent), ("user", lead)])
    assert [(m.role, m.content) for m in transcript] == [
        ("assistant", "namaste"),
        ("user", "haan boliye"),
    ]


class _FakeSTT:
    """Returns a canned transcript per channel (keyed by call order)."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        text = self._texts[self.calls] if self.calls < len(self._texts) else ""
        self.calls += 1
        return STTResult(text=text, confidence=1.0)


class _FakeLLM:
    """Returns a fixed analysis JSON so analyze_call yields a known outcome."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(text=self._payload, finish_reason="stop")


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add(Tenant(id="t1", slug="t1", name="T1"))
        s.add(Conversation(
            id="call_manual", tenant_id="t1", agent_type="human", channel="softphone",
            status="in_progress", pipeline_config={}, provider_call_sid="CA-manual",
            stt_provider="groq", llm_provider="gemini", mode="layered"))
        await s.commit()
    yield maker
    await engine.dispose()


async def test_finalize_manual_call_writes_same_outcome_shape(sm) -> None:
    payload = (
        '{"outcome": "callback_requested", "summary": "Lead asked to be called back.",'
        ' "notes": "Prefers evening.", "callback_datetime": null, "callback_phrase": "shaam ko"}'
    )
    stt = _FakeSTT(["Hello, are you interested?", "Yes, call me back this evening"])
    llm = _FakeLLM(payload)
    channels = [("assistant", b"agent-audio"), ("user", b"lead-audio")]

    async with sm() as session:
        row = await finalize_manual_call(
            session, provider_call_sid="CA-manual", channels=channels,
            stt=stt, stt_config=STTConfig(), llm=llm,
            tenant_timezone="Asia/Kolkata", now=datetime.now(timezone.utc),
            duration_ms=42_000, recording_url="https://rec.example/r1",
        )
    assert row is not None
    # Same fields an AI call writes — the enum *value* string, summary, notes, status.
    assert row.outcome == LeadCallOutcome.CALLBACK_REQUESTED.value
    assert row.summary == "Lead asked to be called back."
    assert "Prefers evening." in row.notes
    assert "rec.example" not in (row.notes or "")     # recording URL NOT sent — call_id is enough
    assert row.status == "ended"
    assert row.duration_ms == 42_000
    assert stt.calls == 2                              # one batch pass per channel
