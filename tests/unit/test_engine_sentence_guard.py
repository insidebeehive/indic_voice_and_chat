"""Tests for PipelineEngine's pre-TTS `sentence_guard` seam (run_turn_text).

Covers only the engine's side of the contract — that it calls the guard per
sentence, respects None/replacement, stops speaking after a trip, fails open
on a guard exception, and is byte-identical when no guard is injected at all.
Policy (what a real guard checks) is tested separately in
tests/unit/test_voicebot_sentence_guard.py.
"""

from __future__ import annotations

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.interfaces.stt import STTConfig, STTResult
from src.interfaces.tts import TTSConfig, TTSResult
from src.pipeline.engine import PipelineConfig, PipelineEngine


class _FakeSTT:
    async def transcribe(self, audio, config):  # pragma: no cover - unused
        raise NotImplementedError

    async def transcribe_stream(self, audio_stream, config):  # pragma: no cover
        if False:
            yield None


class _FakeSTTWithText:
    """Used only by the run_turn (batch-STT) forwarding test below — unlike
    _FakeSTT, transcribe() actually returns text so run_turn proceeds into
    the LLM/TTS overlap instead of exiting early."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def transcribe(self, audio, config):
        return STTResult(text=self._text, confidence=1.0, language="en")

    async def transcribe_stream(self, audio_stream, config):  # pragma: no cover
        if False:
            yield None


class _FakeTTS:
    """Records exactly what it was asked to synthesize, in order."""

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    async def synthesize(self, text, config):
        self.synthesized.append(text)
        return TTSResult(audio=b"\x00\x00" * 10, duration_ms=1.0, sample_rate=16000)

    async def synthesize_stream(self, text_stream, config):  # pragma: no cover
        if False:
            yield b""


class _TwoSentenceLLM:
    """Plain-text response with two clearly separated sentences, so the
    detector yields them as two independent chunks."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def generate_stream(self, messages, config):
        yield self._text


class _KeyObfuscatedJSONLLM:
    """Streams a valid JSON envelope whose `response_text` KEY is spelled with
    a \\u0065 escape instead of a literal 'e'. `json.loads` still resolves it
    to the ordinary key "response_text" (JSON decodes escapes in keys too),
    but `_SpokenTextExtractor._locate_value_start`'s literal substring search
    for `'"response_text"'` never matches the raw streamed text — so
    `spoke_anything` stays False for the whole turn, and the ONLY way this
    text ever reaches TTS is via run_turn_text's post-loop
    `is_json and not spoke_anything` fallback (`_speakable_from_json` on the
    full accumulated text). Used to exercise the sentence_guard hookup on
    that fallback path specifically, not the normal streaming path."""

    async def generate_stream(self, messages, config):
        yield (
            '{"respons\\u0065_text": "All good so far. Your balance is 500.", '
            '"action": "continue"}'
        )


def _engine(llm, *, stt=None, response_format: str = "text") -> tuple[PipelineEngine, _FakeTTS]:
    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format=response_format, max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    tts = _FakeTTS()
    return PipelineEngine(stt or _FakeSTT(), llm, tts, cfg), tts


@pytest.mark.asyncio
async def test_no_guard_injected_is_byte_identical_to_before():
    """Pinning test: with sentence_guard omitted (default None), nothing
    about this turn's outcome changes — protects every existing call that
    doesn't know this feature exists."""
    engine, tts = _engine(_TwoSentenceLLM("First sentence. Second sentence."))
    sink_calls: list[bytes] = []

    async def sink(a: bytes) -> None:
        sink_calls.append(a)

    result = await engine.run_turn_text("hello", history=[], audio_sink=sink)

    assert tts.synthesized == ["First sentence.", "Second sentence."]
    assert result.sentences_spoken == ["First sentence.", "Second sentence."]
    assert result.metrics.sentence_guard_trips == 0
    assert result.cancelled is False
    assert len(sink_calls) == 2


@pytest.mark.asyncio
async def test_grounded_sentence_is_spoken_unchanged():
    """A guard that never objects behaves exactly like no guard at all."""
    engine, tts = _engine(_TwoSentenceLLM("All good here. Still fine."))

    def guard(sentence: str):
        return None  # never trips

    async def sink(a: bytes) -> None:
        pass

    result = await engine.run_turn_text(
        "hello", history=[], audio_sink=sink, sentence_guard=guard,
    )

    assert tts.synthesized == ["All good here.", "Still fine."]
    assert result.sentences_spoken == ["All good here.", "Still fine."]
    assert result.metrics.sentence_guard_trips == 0


@pytest.mark.asyncio
async def test_guard_trip_substitutes_and_stops_later_sentences():
    """Once the guard trips on a sentence, that sentence is replaced and NO
    further sentence this turn reaches TTS — even though the LLM already
    produced them (the offending sentence is the one that matters; earlier
    audio, if any, can't be recalled, but nothing later should compound it)."""
    engine, tts = _engine(
        _TwoSentenceLLM("Your balance is fine. Ignore this trailing sentence.")
    )

    def guard(sentence: str):
        if "balance" in sentence:
            return "SAFE FALLBACK LINE"
        return None

    async def sink(a: bytes) -> None:
        pass

    result = await engine.run_turn_text(
        "hello", history=[], audio_sink=sink, sentence_guard=guard,
    )

    assert tts.synthesized == ["SAFE FALLBACK LINE"]
    assert result.sentences_spoken == ["SAFE FALLBACK LINE"]
    assert result.metrics.sentence_guard_trips == 1
    # The turn itself is not "cancelled" — this is a content substitution,
    # not a failure/barge-in, and downstream code branches on `cancelled`.
    assert result.cancelled is False


@pytest.mark.asyncio
async def test_guard_exception_fails_open_sentence_spoken_unchanged():
    """A buggy guard must never be able to take down a live call: an
    exception from the callback is swallowed (logged) and treated exactly
    like a None return."""
    engine, tts = _engine(_TwoSentenceLLM("First sentence. Second sentence."))

    def guard(sentence: str):
        raise RuntimeError("guard bug")

    async def sink(a: bytes) -> None:
        pass

    result = await engine.run_turn_text(
        "hello", history=[], audio_sink=sink, sentence_guard=guard,
    )

    assert tts.synthesized == ["First sentence.", "Second sentence."]
    assert result.sentences_spoken == ["First sentence.", "Second sentence."]
    assert result.metrics.sentence_guard_trips == 0


@pytest.mark.asyncio
async def test_guard_applies_to_sentence_emitted_via_flush():
    """A trailing fragment with no terminator (only force-emitted by
    detector.flush() at the end of the turn, not the main streaming loop)
    must still go through the guard — flush() is one of the three enqueue
    sites _emit_sentence wraps."""
    # No trailing period on the second clause, so it never completes via a
    # terminator in the main loop; only detector.flush() emits it.
    engine, tts = _engine(_TwoSentenceLLM("All fine here. Your balance is 500"))

    def guard(sentence: str):
        if "balance" in sentence:
            return "SAFE FALLBACK LINE"
        return None

    async def sink(a: bytes) -> None:
        pass

    result = await engine.run_turn_text(
        "hello", history=[], audio_sink=sink, sentence_guard=guard,
    )

    assert tts.synthesized == ["All fine here.", "SAFE FALLBACK LINE"]
    assert result.sentences_spoken == ["All fine here.", "SAFE FALLBACK LINE"]
    assert result.metrics.sentence_guard_trips == 1


@pytest.mark.asyncio
async def test_guard_applies_to_is_json_fallback_extraction():
    """When the streaming extractor never locates `response_text` live (see
    _KeyObfuscatedJSONLLM) but the full envelope is still parseable at the
    end of the turn, the post-loop `_speakable_from_json` fallback is the
    only path that ever produces speakable text — the guard must cover it
    too, not just the normal per-token streaming path."""
    engine, tts = _engine(_KeyObfuscatedJSONLLM(), response_format="json")

    def guard(sentence: str):
        if "balance" in sentence:
            return "SAFE FALLBACK LINE"
        return None

    async def sink(a: bytes) -> None:
        pass

    result = await engine.run_turn_text(
        "hello", history=[], audio_sink=sink, sentence_guard=guard,
    )

    assert tts.synthesized == ["All good so far.", "SAFE FALLBACK LINE"]
    assert result.sentences_spoken == ["All good so far.", "SAFE FALLBACK LINE"]
    assert result.metrics.sentence_guard_trips == 1


@pytest.mark.asyncio
async def test_run_turn_forwards_sentence_guard_to_run_turn_text():
    """run_turn (the batch-STT entry point) must forward `sentence_guard`
    through to the shared run_turn_text implementation — not just
    run_turn_text itself, which every other test in this file exercises."""
    engine, tts = _engine(
        _TwoSentenceLLM("Your balance is fine. Ignore this trailing sentence."),
        stt=_FakeSTTWithText("hello"),
    )

    def guard(sentence: str):
        if "balance" in sentence:
            return "SAFE FALLBACK LINE"
        return None

    async def sink(a: bytes) -> None:
        pass

    result = await engine.run_turn(
        captured_audio=b"\x00\x00", history=[], audio_sink=sink, sentence_guard=guard,
    )

    assert tts.synthesized == ["SAFE FALLBACK LINE"]
    assert result.sentences_spoken == ["SAFE FALLBACK LINE"]
    assert result.metrics.sentence_guard_trips == 1
