from __future__ import annotations

import asyncio

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine


class _FakeLLM:
    async def generate(self, messages, config):  # pragma: no cover - unused
        raise NotImplementedError

    async def generate_stream(self, messages, config):
        for tok in ['{"response_text": "', "नमस्ते जी।", '", "action": "continue"}']:
            yield tok


class _FakeTTS:
    async def synthesize(self, text, config):
        return TTSResult(audio=b"\x00\x00" * 80, duration_ms=10.0, sample_rate=16000)

    async def synthesize_stream(self, text_stream, config):  # pragma: no cover
        if False:
            yield b""


class _FakeSTT:
    async def transcribe(self, audio, config):  # pragma: no cover - unused here
        raise NotImplementedError

    async def transcribe_stream(self, audio_stream, config):  # pragma: no cover
        if False:
            yield None


def _engine():
    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    return PipelineEngine(_FakeSTT(), _FakeLLM(), _FakeTTS(), cfg)


@pytest.mark.asyncio
async def test_run_turn_text_skips_stt_and_speaks_response():
    engine = _engine()
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?",
        history=[LLMMessage(role="system", content="be Anaaya")],
        audio_sink=sink,
    )
    assert result.user_text == "और कुछ benefits हैं?"
    assert result.metrics.stt_latency_ms == 0
    assert '"response_text"' in result.agent_text
    assert sink_calls
    assert "नमस्ते जी।" in "".join(result.sentences_spoken)


class _SlowThenFastTTS:
    """First call sleeps past TTS_SENTENCE_TIMEOUT_S; later calls return instantly."""

    def __init__(self):
        self.calls = 0

    async def synthesize(self, text, config):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(10)  # never returns within the (monkeypatched-small) timeout
            raise AssertionError("should have been cancelled by the per-sentence watchdog")
        return TTSResult(audio=b"\x00\x00" * 80, duration_ms=10.0, sample_rate=16000)


@pytest.mark.asyncio
async def test_run_turn_text_drops_one_slow_sentence_via_watchdog(monkeypatch):
    import src.pipeline.engine as engine_mod
    monkeypatch.setattr(engine_mod, "TTS_SENTENCE_TIMEOUT_S", 0.05)

    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    tts = _SlowThenFastTTS()
    engine = PipelineEngine(_FakeSTT(), _FakeLLM(), tts, cfg)
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?", history=[], audio_sink=sink,
    )
    assert result.metrics.tts_segments_dropped == 1
    assert result.cancelled is False  # one dropped sentence, not enough to abort the whole turn


class _TwoSentenceLLM:
    async def generate_stream(self, messages, config):
        for tok in [
            '{"response_text": "पहला वाक्य। दूसरा वाक्य।", "action": "continue"}',
        ]:
            yield tok


class _AlwaysFailsTTS:
    async def synthesize(self, text, config):
        raise RuntimeError("provider down")


@pytest.mark.asyncio
async def test_run_turn_text_aborts_after_consecutive_tts_failures():
    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    engine = PipelineEngine(_FakeSTT(), _TwoSentenceLLM(), _AlwaysFailsTTS(), cfg)
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?", history=[], audio_sink=sink,
    )
    assert result.metrics.tts_segments_dropped >= 2
    assert result.cancelled is True
    assert sink_calls == []


@pytest.mark.asyncio
async def test_run_turn_text_cancel_stops_before_audio():
    engine = _engine()
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    cancel = asyncio.Event()
    cancel.set()  # pre-cancelled: the token loop breaks before processing/audio

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?",
        history=[],
        audio_sink=sink,
        cancel_event=cancel,
    )
    assert result.cancelled is True
    assert sink_calls == []            # no audio synthesized/sent
    assert result.audio_bytes_sent == 0
    assert result.sentences_spoken == []
