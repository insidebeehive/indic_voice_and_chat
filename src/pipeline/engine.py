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
from src.utils.logging import debug_event


AudioSink = Callable[[bytes], Awaitable[None]]

# Pre-TTS sentence guard, injected by the caller (VoiceBotAgent) — see
# run_turn_text below. Deliberately synchronous and content-agnostic: the
# engine has no idea what the callable checks (KB grounding, profanity,
# anything else) or why. It just runs it on each candidate sentence in the
# one spot that can still stop that sentence from being spoken, and treats
# `None` as "unchanged" / a string as "speak this instead". Default `None`
# everywhere means a caller that never passes one gets byte-identical
# behaviour to before this existed.
SentenceGuard = Callable[[str], Optional[str]]

LLM_TURN_TIMEOUT_S = 10.0  # legitimate end-to-end LLM-generation budget for one turn
LLM_FIRST_TOKEN_TIMEOUT_S = 7.0  # bounds the wait for the FIRST token specifically —
                                  # the LLM_TURN_TIMEOUT_S check above only runs once a
                                  # token has already arrived, so a stall before that
                                  # (e.g. the provider's connection hangs before it starts
                                  # streaming) would otherwise be an unbounded wait.
TTS_SENTENCE_TIMEOUT_S = 10.0  # per-sentence rolling watchdog. Tightened from 25s
                                # (which existed to cover IndicF5's/ElevenLabs' own
                                # 2-attempt retry at up to 10s/attempt = 20s) on the
                                # premise that past ~10s of dead air the caller has
                                # already given up regardless of whether the sentence
                                # eventually synthesizes — so this now typically cuts
                                # a slow provider off mid-retry rather than letting its
                                # 2nd attempt finish. If that turns out to drop too
                                # many otherwise-recoverable sentences, tighten the
                                # providers' own per-attempt timeout (e.g. indicf5.py's
                                # _DEFAULT_TIMEOUT_S) to fit both attempts under this
                                # budget instead of raising this back up.
MAX_CONSECUTIVE_TTS_FAILURES = 2


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
    used_fallback = False
    try:
        obj = json.loads(s)
    except Exception:  # noqa: BLE001 - tolerant: fall back to a {...} search
        used_fallback = True
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:  # noqa: BLE001
                obj = None
    recovered = str(obj.get("response_text") or "") if isinstance(obj, dict) else ""
    if used_fallback:
        # Only on the non-happy path: a clean direct `json.loads` logs
        # nothing here (see the `elif` below for the one thing it can still
        # be worth logging). Otherwise this is the one place that answers
        # "what did the customer actually hear" when the envelope came back
        # malformed -- the raw text plus whatever (if anything) the regex
        # fallback salvaged from it. Gated on `used_fallback` alone, not on
        # `recovered` being empty: `used_fallback and recovered` is a
        # malformed envelope the regex still rescued a value from, and
        # `used_fallback and not recovered` is a genuine parse failure --
        # both are real non-happy-path outcomes. What must NOT land here is
        # a clean parse with an empty/absent response_text: nothing failed
        # to parse there, the model just chose to say nothing (e.g. a
        # hangup action), which is a legitimate outcome, not a parse bug --
        # see docs/debug-logging.md "An event that misleads is worse than no
        # event". That case is reported separately below.
        debug_event(
            log,
            "pipeline envelope_parse recovered" if recovered else "pipeline envelope_parse failed",
            raw=raw, recovered=recovered, used_regex_fallback=used_fallback,
        )
    elif not recovered:
        # Direct json.loads succeeded but response_text was empty or absent.
        # Not a parse failure -- the envelope was well-formed and the model
        # deliberately returned no spoken text. Worth its own event: an
        # operator investigating "the call went silent" should be able to
        # tell this apart from a genuine envelope_parse failure without
        # reading the raw text by eye.
        debug_event(log, "pipeline envelope_parse empty", raw=raw)
    return recovered


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

    @property
    def closed(self) -> bool:
        """True once ``response_text``'s closing quote has actually been seen.

        Read once at end-of-turn (see run_turn_text) to tell a normally
        finished extraction from one truncated mid-value -- ``feed`` itself
        is never logged (per-token), so this is the only place that
        distinction is visible without re-reading the raw envelope.
        """
        return self._closed

    @property
    def consumed(self) -> int:
        """Chars of ``response_text`` decoded and handed out via ``feed`` so far."""
        return self._consumed

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
    tts_segments_dropped: int = 0
    # Incremented once per turn when the (optional) sentence_guard replaces a
    # sentence — see run_turn_text. Generic on purpose (mirrors
    # tts_segments_dropped): the engine only counts that a substitution
    # happened, never why, so this flows through the existing metrics_dict /
    # "voice turn metrics" log path (VoiceBotAgent.apply_signal) without the
    # engine needing to know anything about guard policy.
    sentence_guard_trips: int = 0


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
        sentence_guard: Optional[SentenceGuard] = None,
    ) -> TurnResult:
        """Run one perception-reasoning-action cycle.

        ``history`` is the full list of LLMMessages including the system
        prompt and prior turns. The caller is responsible for appending
        the new user turn before calling ``run_turn``... or not, it's fine
        either way: ``run_turn`` does NOT mutate ``history``.

        ``language`` (when set) overrides the configured STT + TTS language for
        this turn — the dialogue layer passes the conversation's active language
        so STT transcribes and TTS speaks in the caller's current language.

        ``sentence_guard`` — see run_turn_text — is forwarded unchanged; STT
        happens here first, then the LLM->TTS overlap runs exactly as in
        run_turn_text.
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
            sentence_guard=sentence_guard,
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
        sentence_guard: Optional[SentenceGuard] = None,
    ) -> TurnResult:
        """LLM->TTS for an already-transcribed user turn (no STT).

        Used by the streaming-STT path: Deepgram has already produced the
        transcript, so we skip STT entirely and run the LLM/TTS overlap.

        ``language`` (when set) overrides the configured TTS language for this
        turn — the conversation's active language, so the reply is spoken in the
        caller's current language.

        ``sentence_guard``, when given, is called with each sentence right
        before it would be handed to TTS (see ``_emit_sentence`` below).
        Returning ``None`` speaks the sentence unchanged; returning a string
        substitutes it AND stops any further sentences this turn from being
        spoken. The engine has no opinion on what the guard checks — see the
        module-level ``SentenceGuard`` docstring. Default ``None`` reproduces
        today's behaviour exactly (no guard call, no substitution, nothing
        dropped).
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
        guard_tripped = False

        async def _emit_sentence(sentence: str) -> None:
            """Give ``sentence_guard`` (if any) a chance to veto this
            sentence, then enqueue whatever should actually be spoken.

            This runs on the PRODUCER side — right where a completed
            sentence comes out of ``SentenceDetector`` — rather than after
            the full response is assembled, because that's the only point
            left where a check can still stop THIS sentence's own audio
            from going out: by the time the full response is parseable,
            earlier (and this) sentence's TTS may already be in flight or
            done (see the module docstring's overlap design, and
            voicebot.py's handling of `sentences_spoken`).

            Once tripped, later sentences in the same turn are dropped here
            silently (not even passed to the guard): the sentence that
            stated the ungrounded figure has already been swapped for the
            safe line, earlier audio that went out before it can't be
            unsaid, and speaking on past the substitute would just bury it.

            A guard exception is treated exactly like a `None` return
            (unchanged) — logged, never raised further. No `await` happens
            around the guard call itself, so a slow or hung guard can't
            stall this loop or the TTS queue; only a plain regex-speed
            callable belongs here (see PipelineEngine docstring / callers).
            """
            nonlocal guard_tripped
            if guard_tripped:
                return
            if sentence_guard is None:
                await sentence_queue.put(sentence)
                return
            try:
                replacement = sentence_guard(sentence)
            except Exception as exc:  # noqa: BLE001 - fail OPEN: a guard bug must never break a live call
                log.exception("sentence_guard raised; speaking sentence unchanged")
                # log.exception above carries no structured fields and no
                # session context. Without this, a guard bug and a guard
                # saying "fine" are indistinguishable from outside -- the
                # sentence goes out unchecked either way, which is exactly
                # the case the guard exists to prevent.
                debug_event(
                    log, "pipeline sentence_guard raised",
                    sentence=sentence, error=f"{type(exc).__name__}: {exc}",
                )
                replacement = None
            if replacement is None:
                await sentence_queue.put(sentence)
                return
            guard_tripped = True
            metrics.sentence_guard_trips += 1
            await sentence_queue.put(replacement)

        async def tts_worker() -> None:
            nonlocal first_audio_at, bytes_sent
            consecutive_failures = 0
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    return
                if cancel_event.is_set():
                    continue
                try:
                    result = await asyncio.wait_for(
                        self._tts.synthesize(sentence, tts_cfg),
                        timeout=TTS_SENTENCE_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    log.error(
                        "TTS synthesize timed out after %.0fs: %r",
                        TTS_SENTENCE_TIMEOUT_S, sentence[:60],
                    )
                    # ERROR above truncates to 60 chars and carries no
                    # extra= fields -- this is the "what did the customer NOT
                    # hear" companion: the full sentence that got dropped.
                    debug_event(
                        log, "pipeline tts_synthesize timed_out",
                        sentence=sentence, timeout_s=TTS_SENTENCE_TIMEOUT_S,
                        consecutive_failures=consecutive_failures + 1,
                    )
                    metrics.tts_segments_dropped += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_TTS_FAILURES:
                        log.error(
                            "aborting turn: %d consecutive TTS failures",
                            consecutive_failures,
                        )
                        debug_event(
                            log, "pipeline turn aborted",
                            reason="consecutive_tts_failures",
                            consecutive_failures=consecutive_failures,
                            sentences_spoken=len(sentences_spoken),
                        )
                        cancel_event.set()
                    continue
                except Exception as _tts_err:  # noqa: BLE001
                    log.error("TTS synthesize failed: %s", _tts_err)
                    debug_event(
                        log, "pipeline tts_synthesize failed",
                        sentence=sentence, error=f"{type(_tts_err).__name__}: {_tts_err}",
                        consecutive_failures=consecutive_failures + 1,
                    )
                    metrics.tts_segments_dropped += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_TTS_FAILURES:
                        log.error(
                            "aborting turn: %d consecutive TTS failures",
                            consecutive_failures,
                        )
                        debug_event(
                            log, "pipeline turn aborted",
                            reason="consecutive_tts_failures",
                            consecutive_failures=consecutive_failures,
                            sentences_spoken=len(sentences_spoken),
                        )
                        cancel_event.set()
                    continue
                consecutive_failures = 0
                if cancel_event.is_set():
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                await audio_sink(result.audio)
                bytes_sent += len(result.audio)
                sentences_spoken.append(sentence)

        tts_task = asyncio.create_task(tts_worker())

        is_json = getattr(self._config.llm, "response_format", None) == "json"
        extractor = _SpokenTextExtractor() if is_json else None
        spoke_anything = False

        try:
            stream = self._llm.generate_stream(messages, self._config.llm).__aiter__()
            token_index = 0
            while True:
                try:
                    if token_index == 0:
                        # See LLM_FIRST_TOKEN_TIMEOUT_S: the total-generation-budget
                        # check below only runs once a token has already arrived, so
                        # this bounds the one wait that check can't cover.
                        token = await asyncio.wait_for(
                            stream.__anext__(), timeout=LLM_FIRST_TOKEN_TIMEOUT_S
                        )
                    else:
                        token = await stream.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    log.error(
                        "LLM produced no first token within %.0fs; ending turn early",
                        LLM_FIRST_TOKEN_TIMEOUT_S,
                    )
                    debug_event(
                        log, "pipeline llm_generate no_first_token",
                        timeout_s=LLM_FIRST_TOKEN_TIMEOUT_S,
                    )
                    cancel_event.set()
                    break
                token_index += 1
                if cancel_event.is_set():
                    break
                if time.perf_counter() - t_llm_start > LLM_TURN_TIMEOUT_S:
                    log.error(
                        "LLM generation exceeded %.0fs budget; ending turn early",
                        LLM_TURN_TIMEOUT_S,
                    )
                    # gemini.py's own "generate_stream response" DEBUG line
                    # only fires once the async generator is exhausted
                    # naturally -- breaking out of consumption here means it
                    # never fires for this turn, so the partial text collected
                    # before the budget hit would otherwise be lost entirely.
                    #
                    # `partial_raw_output` is the RAW LLM output collected so
                    # far, not the spoken text: in JSON mode (the voice
                    # default) that's the raw envelope, the same value
                    # TurnResult.agent_text carries. `tokens_received` is
                    # `token_index - 1`, not `token_index`: token_index was
                    # already incremented above by the time this check runs,
                    # but the token that triggered the budget check is
                    # dropped by the `break` below (it never reaches
                    # full_text_parts / partial_raw_output), so counting it
                    # here would make tokens_received one higher than the
                    # number of tokens actually present in partial_raw_output.
                    # That one extra token was received from the stream but
                    # is not reflected in either field.
                    debug_event(
                        log, "pipeline llm_generate budget_exceeded",
                        timeout_s=LLM_TURN_TIMEOUT_S, tokens_received=token_index - 1,
                        partial_raw_output="".join(full_text_parts),
                    )
                    cancel_event.set()
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                full_text_parts.append(token)
                speakable = extractor.feed(token) if extractor is not None else token
                if speakable:
                    spoke_anything = True
                    for sentence in detector.feed(speakable):
                        await _emit_sentence(sentence)

            if not cancel_event.is_set():
                if is_json and not spoke_anything:
                    for sentence in detector.feed(
                        _speakable_from_json("".join(full_text_parts))
                    ):
                        await _emit_sentence(sentence)
                for sentence in detector.flush():
                    await _emit_sentence(sentence)
        finally:
            # Captured here, BEFORE draining the TTS queue: this must reflect
            # only the LLM's own work (token generation + flush), not however
            # long the trailing TTS synthesis for the last sentence(s) takes
            # afterward — that time is already correctly counted in
            # tts_total_ms below. Capturing it after `await tts_task` (the
            # bug this fixes) silently inflated llm_total_ms by that TTS
            # tail, worst with a slow TTS provider.
            metrics.llm_total_ms = int((time.perf_counter() - t_llm_start) * 1000)
            await sentence_queue.put(None)
            await tts_task

        if extractor is not None:
            # The DECISION, not a second copy of the text: whether the
            # streamed response_text value ever closed. When the stream
            # finishes normally gemini.py's own "generate_stream response"
            # DEBUG line already has the full raw envelope; when it doesn't
            # (aborted above, or cancelled below), that line never fires at
            # all, and `closed=False` here is often the only trace that the
            # envelope was left mid-value.
            debug_event(
                log, "pipeline spoken_text_extraction summary",
                extracted_chars=extractor.consumed, closed=extractor.closed,
                spoke_anything=spoke_anything, cancelled=cancel_event.is_set(),
            )

        if cancel_event.is_set() and detector.pending:
            # Barge-in, or an internal abort (LLM budget / consecutive TTS
            # failures above), discards whatever SentenceDetector is still
            # holding once cancellation is observed -- deliberately not
            # calling detector.flush() first, so a cut-off fragment isn't
            # spoken after the fact. Gated on detector.pending (not just
            # cancel_event) because cancel_event can be set with nothing
            # buffered: a barge-in landing exactly on a sentence boundary, or
            # the MAX_CONSECUTIVE_TTS_FAILURES abort path above, where every
            # sentence had already been emitted before the abort. Firing this
            # event with pending_text='' in those cases reads as "buffered
            # speech was dropped" when nothing was -- see docs/debug-logging.md
            # "An event that misleads is worse than no event". Note this path
            # can still race detector.flush(): cancel_event is set from
            # inside tts_worker and may arrive after the
            # `if not cancel_event.is_set():` above has already run flush(),
            # in which case detector.pending is '' here too and this simply
            # doesn't fire -- it does not mean flush() is never reached on a
            # cancelled turn, only that this event and flush() are mutually
            # exclusive on any given turn.
            debug_event(
                log, "pipeline turn pending_discarded",
                pending_text=detector.pending, sentences_spoken=len(sentences_spoken),
            )

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
