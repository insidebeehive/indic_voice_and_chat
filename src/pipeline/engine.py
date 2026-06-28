"""Streaming voice pipeline engine.

Coordinates STT -> LLM -> TTS for a single conversational turn:

    captured_audio (bytes)
        |
        v
    STT.transcribe   -> user_text (with confidence + language)
        |
        v
    LLM.generate_stream -> token stream
        |
        v       (split on sentence boundaries via SentenceDetector)
        v
    TTS.synthesize_stream -> audio chunks
        |
        v
    audio_sink (caller-provided callable)

Design choices:
- Stages overlap: as soon as the LLM emits one complete sentence, we kick
  off TTS on it while the LLM keeps generating the next sentence.
- The full LLM text is also returned at the end so the caller can parse the
  structured JSON response (the streamed audio is just the speakable part).
- Per-stage latency is recorded in ``TurnMetrics`` for benchmarking.
- Cancellable via the supplied ``asyncio.Event`` (set by interruption
  handler to drop in-flight audio).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)

from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage
from src.interfaces.stt import ISTTProvider, STTConfig
from src.interfaces.tts import ITTSProvider, TTSConfig
from src.pipeline.sentence_detector import SentenceDetector


AudioSink = Callable[[bytes], Awaitable[None]]


def _speakable_from_json(raw: str) -> str:
    """Extract the spoken text (the ``response_text`` field) from a structured
    JSON LLM response.

    When the LLM runs in ``response_format=json`` mode it emits an envelope
    like ``{"response_text": "...", "action": "...", "updated_slots": {...}}``.
    Only ``response_text`` should be spoken — feeding the raw envelope to TTS
    makes it read field names ("response_text" -> "response underscore text"),
    braces, and slot keys aloud. Tolerant of markdown code fences and
    surrounding prose; returns '' when no ``response_text`` can be recovered.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s[:4].lower() == "json":
            s = s[4:].strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001 - tolerant: fall back to a {...} search
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:  # noqa: BLE001
                obj = None
    if isinstance(obj, dict):
        return str(obj.get("response_text") or "")
    return ""


class _SpokenTextExtractor:
    """Incrementally pull the ``response_text`` value out of a streaming JSON
    envelope, so TTS can start on the first sentence instead of waiting for the
    whole envelope (which includes trailing metadata) to finish generating.

    ``feed(token)`` returns any newly-decoded characters of ``response_text``
    (JSON string escapes handled), or '' if none are available yet. Once the
    value's closing quote is seen, further tokens return ''.
    """

    _KEY = '"response_text"'

    def __init__(self) -> None:
        self._joined = ""
        self._value_start = -1  # index where the value's content begins
        self._consumed = 0      # decoded chars already returned
        self._closed = False

    def feed(self, token: str) -> str:
        self._joined += token
        if self._closed:
            return ""
        if self._value_start < 0:
            self._locate_value_start()
            if self._value_start < 0:
                return ""
        return self._emit_new()

    def _locate_value_start(self) -> None:
        s = self._joined
        k = s.find(self._KEY)
        if k < 0:
            return
        i = k + len(self._KEY)
        while i < len(s) and s[i] in " \t\r\n":
            i += 1
        if i >= len(s) or s[i] != ":":
            return
        i += 1
        while i < len(s) and s[i] in " \t\r\n":
            i += 1
        if i >= len(s) or s[i] != '"':
            return
        self._value_start = i + 1

    def _emit_new(self) -> str:
        s = self._joined
        i = self._value_start
        decoded: list[str] = []
        closed = False
        _simple = {'"': '"', "\\": "\\", "/": "/", "n": "\n",
                   "t": "\t", "r": "\r", "b": "\b", "f": "\f"}
        while i < len(s):
            c = s[i]
            if c == "\\":
                if i + 1 >= len(s):
                    break  # incomplete escape — wait for more tokens
                e = s[i + 1]
                if e in _simple:
                    decoded.append(_simple[e]); i += 2; continue
                if e == "u":
                    if i + 6 > len(s):
                        break  # incomplete \uXXXX
                    try:
                        decoded.append(chr(int(s[i + 2:i + 6], 16)))
                    except ValueError:
                        decoded.append(s[i + 2:i + 6])
                    i += 6
                    continue
                decoded.append(e); i += 2; continue
            if c == '"':
                closed = True
                break
            decoded.append(c)
            i += 1
        full = "".join(decoded)
        new = full[self._consumed:]
        self._consumed = len(full)
        if closed:
            self._closed = True
        return new


@dataclass
class TurnMetrics:
    stt_latency_ms: int = 0
    llm_ttft_ms: int = 0
    llm_total_ms: int = 0
    tts_first_chunk_ms: int = 0
    tts_total_ms: int = 0
    total_latency_ms: int = 0


@dataclass
class TurnResult:
    user_text: str
    user_language: Optional[str]
    user_confidence: float
    agent_text: str  # full raw LLM output (for parsing)
    audio_bytes_sent: int
    metrics: TurnMetrics
    cancelled: bool = False
    sentences_spoken: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig


class PipelineEngine:
    """One-call-per-instance is fine; reuse across calls is also OK since
    state is held only in local variables of ``run_turn``.
    """

    def __init__(
        self,
        stt: ISTTProvider,
        llm: ILLMProvider,
        tts: ITTSProvider,
        config: PipelineConfig,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._config = config

    async def run_turn(
        self,
        captured_audio: bytes,
        history: list[LLMMessage],
        audio_sink: AudioSink,
        cancel_event: Optional[asyncio.Event] = None,
        *,
        language: Optional[str] = None,
    ) -> TurnResult:
        """Run one perception-reasoning-action cycle.

        ``history`` is the full list of LLMMessages including the system
        prompt and prior turns. The caller is responsible for appending
        the new user turn before calling ``run_turn``... or not, it's fine
        either way: ``run_turn`` does NOT mutate ``history``.

        ``language`` (when set) overrides the configured STT + TTS language for
        this turn — the dialogue layer passes the conversation's active language
        so STT transcribes and TTS speaks in the caller's current language.
        """
        cancel_event = cancel_event or asyncio.Event()
        metrics = TurnMetrics()
        t_overall = time.perf_counter()

        # --- STT ---------------------------------------------------------
        stt_cfg = replace(self._config.stt, language=language) if language else self._config.stt
        t0 = time.perf_counter()
        stt_result = await self._stt.transcribe(captured_audio, stt_cfg)
        metrics.stt_latency_ms = int((time.perf_counter() - t0) * 1000)

        # If STT returned nothing useful, exit early — caller decides what
        # to do (re-prompt, end the call, etc.).
        if not stt_result.text.strip():
            metrics.total_latency_ms = int((time.perf_counter() - t_overall) * 1000)
            return TurnResult(
                user_text="",
                user_language=stt_result.language,
                user_confidence=stt_result.confidence,
                agent_text="",
                audio_bytes_sent=0,
                metrics=metrics,
            )

        # STT done — hand the transcript to the shared LLM->TTS path.
        return await self.run_turn_text(
            stt_result.text,
            history,
            audio_sink,
            cancel_event,
            user_language=stt_result.language,
            user_confidence=stt_result.confidence,
            stt_latency_ms=metrics.stt_latency_ms,
            t_overall=t_overall,
            language=language,
        )

    async def run_turn_text(
        self,
        user_text: str,
        history: list[LLMMessage],
        audio_sink: AudioSink,
        cancel_event: Optional[asyncio.Event] = None,
        *,
        user_language: Optional[str] = None,
        user_confidence: float = 1.0,
        stt_latency_ms: int = 0,
        t_overall: Optional[float] = None,
        language: Optional[str] = None,
    ) -> TurnResult:
        """LLM->TTS for an already-transcribed user turn (no STT).

        Used by the streaming-STT path: Deepgram has already produced the
        transcript, so we skip STT entirely and run the LLM/TTS overlap.

        ``language`` (when set) overrides the configured TTS language for this
        turn — the conversation's active language, so the reply is spoken in the
        caller's current language.
        """
        cancel_event = cancel_event or asyncio.Event()
        tts_cfg = replace(self._config.tts, language=language) if language else self._config.tts
        if t_overall is None:
            t_overall = time.perf_counter()
        metrics = TurnMetrics()
        metrics.stt_latency_ms = stt_latency_ms

        messages = list(history) + [LLMMessage(role="user", content=user_text)]

        # first_chunk_soft: let the FIRST sentence break on a clause boundary so
        # TTS (and thus first audio) starts sooner; later sentences stay normal.
        detector = SentenceDetector(first_chunk_soft=True)
        full_text_parts: list[str] = []
        sentences_spoken: list[str] = []
        bytes_sent = 0
        first_token_at: Optional[float] = None
        first_audio_at: Optional[float] = None

        t_llm_start = time.perf_counter()
        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def tts_worker() -> None:
            nonlocal first_audio_at, bytes_sent
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    return
                if cancel_event.is_set():
                    continue
                try:
                    result = await self._tts.synthesize(sentence, tts_cfg)
                except Exception as _tts_err:  # noqa: BLE001
                    log.error("TTS synthesize failed: %s", _tts_err)
                    continue
                if cancel_event.is_set():
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                bytes_sent += len(result.audio)
                sentences_spoken.append(sentence)
                await audio_sink(result.audio)

        tts_task = asyncio.create_task(tts_worker())

        is_json = getattr(self._config.llm, "response_format", None) == "json"
        extractor = _SpokenTextExtractor() if is_json else None
        spoke_anything = False

        try:
            async for token in self._llm.generate_stream(messages, self._config.llm):
                if cancel_event.is_set():
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                full_text_parts.append(token)
                speakable = extractor.feed(token) if extractor is not None else token
                if speakable:
                    spoke_anything = True
                    for sentence in detector.feed(speakable):
                        await sentence_queue.put(sentence)

            if not cancel_event.is_set():
                if is_json and not spoke_anything:
                    for sentence in detector.feed(
                        _speakable_from_json("".join(full_text_parts))
                    ):
                        await sentence_queue.put(sentence)
                for sentence in detector.flush():
                    await sentence_queue.put(sentence)
        finally:
            await sentence_queue.put(None)
            await tts_task

        metrics.llm_total_ms = int((time.perf_counter() - t_llm_start) * 1000)
        if first_token_at is not None:
            metrics.llm_ttft_ms = int((first_token_at - t_llm_start) * 1000)
        if first_audio_at is not None:
            metrics.tts_first_chunk_ms = int((first_audio_at - t_llm_start) * 1000)
            metrics.tts_total_ms = int((time.perf_counter() - first_audio_at) * 1000)
        metrics.total_latency_ms = int((time.perf_counter() - t_overall) * 1000)

        return TurnResult(
            user_text=user_text,
            user_language=user_language,
            user_confidence=user_confidence,
            agent_text="".join(full_text_parts),
            audio_bytes_sent=bytes_sent,
            metrics=metrics,
            cancelled=cancel_event.is_set(),
            sentences_spoken=sentences_spoken,
        )
