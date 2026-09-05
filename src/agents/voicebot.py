"""VoiceBot agent.

Wraps the pipeline engine with conversation control: prompt building, turn
sequencing, slot updates, structured response parsing, state-machine event
firing, and session persistence.

Telephony I/O lives outside this class — the agent receives a captured
audio buffer per turn from the telephony layer (Twilio Media Streams
websocket in Phase 3) and emits audio chunks to a sink the telephony layer
provides.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from src.agents.base import AgentSession, BaseAgent
from src.agents.state_machine import AgentStateMachine, Event, State
from src.dialogue.language import normalize_lang, resolve_active_language, to_bcp47
from src.dialogue.prompts import VoiceBotScript, build_voicebot_system_prompt
from src.dialogue.response_parser import (
    VoiceBotResponse,
    _SENTIMENTS,
    _VOICEBOT_ACTIONS,
    _VOICEBOT_PHASES,
    parse_voicebot_response,
)
from src.dialogue.slots import SlotFiller, SlotSchema
from src.interfaces.llm import LLMMessage
from src.pipeline.engine import AudioSink, PipelineEngine, TurnMetrics, TurnResult


log = logging.getLogger(__name__)


def _join_spoken_sentences(sentences: list[str]) -> str:
    """Reconstruct the text TTS actually spoke from ``TurnResult.sentences_spoken``.

    The sentence detector's soft first-chunk boundary (splitting on a comma so
    TTS starts sooner) doesn't preserve the space around that split, so a plain
    join can glue two fragments together ("accha,dhanyavaad"). Join with a
    space and collapse runs of whitespace instead of assuming exact byte
    fidelity to the original streamed text.
    """
    return re.sub(r"\s+", " ", " ".join(sentences)).strip()


# Targeted field recovery for a JSON envelope whose overall parse failed. The
# observed corruption (a stray quote/brace near the END of the envelope — see
# response_parser._extract_json's tolerant full-object parser) typically
# leaves early fields like `action` intact even though the object as a whole
# won't parse. This is NOT a general JSON-repair library — just enough to
# avoid hard-overriding `action` to "continue" when the model's real decision
# (e.g. a close_positive/transfer) is still recoverable from the raw text.
_ACTION_RE = re.compile(r'"action"\s*:\s*"(\w+)"')
_SENTIMENT_FIELD_RE = re.compile(r'"sentiment"\s*:\s*"(\w+)"')
_PHASE_FIELD_RE = re.compile(r'"conversation_phase"\s*:\s*"(\w+)"')
_SLOTS_KEY_RE = re.compile(r'"updated_slots"\s*:\s*\{')


def _find_updated_slots_span(raw_text: str) -> Optional[tuple[int, int]]:
    """Locate the ``updated_slots`` object's ``[start, end)`` span (the
    opening '{' through its matching closing '}', inclusive) via
    balanced-brace matching, tolerating corruption elsewhere in the envelope.
    String-aware (honors ``\\"`` escapes) so a literal ``{``/``}`` inside a
    slot's string value doesn't miscount and truncate the span early.
    Returns ``None`` if the key isn't present or the object never balances."""
    m = _SLOTS_KEY_RE.search(raw_text)
    if not m:
        return None
    start = m.end() - 1  # index of the opening '{'
    depth = 0
    in_string = False
    i = start
    while i < len(raw_text):
        ch = raw_text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    return None


def _recover_updated_slots(raw_text: str) -> Optional[dict[str, Any]]:
    """Extract ``updated_slots`` via balanced-brace matching starting right
    after the key, tolerating corruption elsewhere in the envelope."""
    span = _find_updated_slots_span(raw_text)
    if span is None:
        return None
    start, end = span
    try:
        obj = json.loads(raw_text[start:end], strict=False)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _recover_fields_from_raw(raw_text: str) -> dict[str, Any]:
    """Best-effort recovery of action/sentiment/conversation_phase/
    updated_slots from raw LLM text whose full JSON parse failed. Returns
    only the keys it could recover.

    The model is instructed to emit (see build_voicebot_system_prompt's field
    spec in prompts.py — the actual live instruction; VOICEBOT_RESPONSE_SCHEMA
    in this same prompts.py module documents the same shape but is never sent
    to any provider, so it isn't the ground truth for field order): response_text,
    language, action, conversation_phase, sentiment, updated_slots,
    action_reason, internal_notes. `action`/`sentiment`/`conversation_phase`
    all come BEFORE `updated_slots` there — but field order isn't a hard
    guarantee, so a slot dict that happens to contain a key literally named
    "action"/"sentiment"/"conversation_phase" (e.g. a badly-named campaign
    slot) whose value happens to be a valid enum member could still shadow the
    real field. Rather than depend on order, the `updated_slots` object's own
    span is excised from the text before searching for the other three
    fields, so a decoy inside it can never match regardless of which side of
    `updated_slots` the real field actually lands on.
    """
    out: dict[str, Any] = {}
    if not raw_text:
        return out
    slots_span = _find_updated_slots_span(raw_text)
    if slots_span is not None:
        start, end = slots_span
        search_text = raw_text[:start] + raw_text[end:]
    else:
        search_text = raw_text
    m = _ACTION_RE.search(search_text)
    if m and m.group(1) in _VOICEBOT_ACTIONS:
        out["action"] = m.group(1)
    m = _SENTIMENT_FIELD_RE.search(search_text)
    if m and m.group(1) in _SENTIMENTS:
        out["sentiment"] = m.group(1)
    m = _PHASE_FIELD_RE.search(search_text)
    if m and m.group(1) in _VOICEBOT_PHASES:
        out["conversation_phase"] = m.group(1)
    if slots_span is not None:
        slots = _recover_updated_slots(raw_text)
        if slots is not None:
            out["updated_slots"] = slots
    return out


def _apply_recovered_fields(response: VoiceBotResponse, raw_text: str) -> None:
    """Mutate ``response`` in place with whatever _recover_fields_from_raw can
    salvage from ``raw_text``, falling back to action="continue" (the prior
    hardcoded behaviour) only when nothing could be recovered."""
    recovered = _recover_fields_from_raw(raw_text)
    response.action = recovered.get("action", "continue")
    if "updated_slots" in recovered:
        response.updated_slots = recovered["updated_slots"]
    if "sentiment" in recovered:
        response.sentiment = recovered["sentiment"]
    if "conversation_phase" in recovered:
        response.conversation_phase = recovered["conversation_phase"]


@dataclass
class TurnOutcome:
    """The full result of one user-utterance / agent-response cycle."""

    response: VoiceBotResponse
    pipeline: TurnResult


# Map LLM-emitted ``action`` to the state-machine event that follows.
_ESCALATION_ACTIONS = {"transfer", "schedule_callback"}
_END_ACTIONS = {"close_positive", "close_negative", "end"}

# LLM generation and each TTS sentence now enforce their own internal budgets
# (LLM_TURN_TIMEOUT_S / LLM_FIRST_TOKEN_TIMEOUT_S / TTS_SENTENCE_TIMEOUT_S in
# src/pipeline/engine.py) — including a dedicated timeout on the wait for the
# LLM's first token, the one open-ended wait those budgets couldn't otherwise
# bound. With that gap closed at the source, this outer cap is a genuine rare
# last-resort (every step it could catch is already independently bounded), so
# it's deliberately generous rather than tuned to the ~10s-of-dead-air ceiling
# that governs the actual per-step budgets a caller routinely experiences.
# (Briefly tightened to 7s/3s during 2026-08 before the first-token timeout
# existed, which would have false-positived on legitimately slow multi-sentence
# turns — reverted once the real gap was closed instead.)
HARD_TURN_TIMEOUT_S = 90.0
_BACKSTOP_GRACE_S = 30.0

# The one-shot retry (see _finish_turn) fires only after a turn has ALREADY
# failed to parse — if it also stalls before a token arrives, that's a strong
# signal the provider is genuinely down right now, not a transient blip.
# Doubling down on the full primary backstop would be exactly backwards, so
# the retry gets its own, even tighter budget: fail fast into the canned
# fallback rather than making the caller wait through a second near-full wait.
_RETRY_HARD_TIMEOUT_S = 5.0
_RETRY_BACKSTOP_GRACE_S = 2.0

# Sliding-window context. The full transcript lives in ``session.turns`` (used for
# the UI and post-call outcome analysis), but only the system prompt + the last
# MAX_HISTORY_TURNS exchanges are sent to the LLM each turn. Without this the prompt
# grows ~2 messages per turn, so per-turn TTFT/latency climbs as a call goes on
# ("agent takes longer to respond after a while"). Keeping the system prompt
# preserves persona + lead data; ~6 recent exchanges preserve the active thread.
MAX_HISTORY_TURNS = 6


class VoiceBotAgent(BaseAgent):
    def __init__(
        self,
        session: AgentSession,
        state_machine: AgentStateMachine,
        slot_schema: SlotSchema,
        script: VoiceBotScript,
        engine: PipelineEngine,
        store=None,
        extra_directives: Optional[list[str]] = None,
        kb_context: Optional[str] = None,
        record_metric: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        # Always recognise lead_gender so the LLM can infer and report it even
        # when the campaign YAML doesn't define it as a slot.
        if "lead_gender" not in slot_schema.specs:
            from src.dialogue.slots import SlotSpec, SlotType
            slot_schema.specs["lead_gender"] = SlotSpec(
                name="lead_gender", type=SlotType.ENUM, values=["male", "female"])
        slots = SlotFiller(slot_schema)
        super().__init__(session=session, state_machine=state_machine, slots=slots, store=store)
        self._script = script
        self._engine = engine
        if script.pronunciations:
            from dataclasses import replace as _replace_cfg
            # Merge over whatever's already on the engine's TTS config (e.g. a
            # CRM's pronunciation_overrides, threaded in at engine-construction
            # time — src/bootstrap.py) rather than replacing it outright, so a
            # campaign-level pronunciation fix doesn't silently wipe the CRM's
            # own overrides. Campaign wins on key collision — the more specific
            # scope, matching apply_pronunciations' "extra wins" precedence.
            merged_pronunciations = {
                **(engine._config.tts.extra_pronunciations or {}),
                **script.pronunciations,
            }
            engine._config = _replace_cfg(
                engine._config,
                tts=_replace_cfg(engine._config.tts, extra_pronunciations=merged_pronunciations),
            )
        self._extra_directives = extra_directives
        self._kb_context = kb_context
        self._record_metric = record_metric
        # The conversation's active language. Starts at the campaign default and
        # switches when the caller speaks/asks for another language (resolved each
        # turn from the LLM-reported + STT-detected signals). Drives per-turn
        # STT + TTS via the engine and the opening line.
        self._active_language = normalize_lang(script.language_default) or "hi"
        self._system_prompt = build_voicebot_system_prompt(
            script=script,
            schema=slot_schema,
            lead_data=session.lead_data,
            extra_directives=extra_directives,
            kb_context=kb_context,
        )
        self.session.turns.append(LLMMessage(role="system", content=self._system_prompt))

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def active_language(self) -> str:
        """The conversation's current language (base code, e.g. 'hi'/'mr'). Starts
        at the campaign default and switches when the caller changes language."""
        return self._active_language

    def _history_window(self) -> list[LLMMessage]:
        """Bounded LLM context: the system prompt + the last MAX_HISTORY_TURNS
        exchanges. ``session.turns`` keeps the full transcript untouched; this
        only narrows what the engine sends to the LLM, so per-turn latency stays
        flat as the call grows (see MAX_HISTORY_TURNS). Always returns the system
        prompt (turns[0]) followed by the most recent messages."""
        turns = self.session.turns
        if len(turns) <= 1:
            return list(turns)
        return turns[:1] + turns[1:][-(2 * MAX_HISTORY_TURNS):]

    async def start(self) -> None:
        """Move from IDLE to LISTENING. Call once when the call connects."""
        await self.state.fire_if_possible(Event.CALL_CONNECTED)
        await self.persist_state()

    async def play_opening(self, audio_sink: AudioSink) -> None:
        """Speak the campaign opening line as the agent's first turn.

        For outbound campaigns the agent must speak first — the user just
        answered the phone and is silent. We synthesize the script's
        opening line, push the audio through ``audio_sink``, append it to
        the conversation history, and stay in LISTENING for the user's
        reply. Skips silently if there's no opening configured.
        """
        # opening_male/opening_female are keyed on AGENT gender (they carry the
        # agent's own verb inflection — raha/rahi). Lead address is handled via
        # the {lead_salutation} template token computed in _template_vars().
        agent_gender = (self._script.gender or "").strip().lower()
        if agent_gender == "male" and self._script.opening_male:
            opening = self._script.opening_male.strip()
        elif agent_gender == "female" and self._script.opening_female:
            opening = self._script.opening_female.strip()
        else:
            opening = (self._script.opening or "").strip()
        if not opening:
            return
        # Substitute simple template tokens with known lead data.
        import re as _re
        raw = opening.format(**self._template_vars())
        # Collapse extra spaces that arise when a name token is empty.
        rendered = _re.sub(r"  +", " ", raw).strip()

        # The TTS goes through the pipeline engine's TTS provider so the
        # adapter-level streaming + sample-rate handling stays consistent
        # with the per-turn synthesis path. Reuse the engine's CONFIGURED TTS
        # (voice_id, sample_rate, …) — only overriding the language — so the
        # opening uses the SELECTED voice, not the provider default.
        from dataclasses import replace as _replace

        from src.interfaces.tts import TTSConfig as _TTSConfig
        base_tts = getattr(getattr(self._engine, "_config", None), "tts", None)
        opening_lang = to_bcp47(self._active_language)
        # Merge the script's own pronunciations over whatever's already on
        # base_tts (e.g. a CRM's pronunciation_overrides) rather than replacing
        # it outright -- otherwise a campaign with no script.pronunciations
        # (script.pronunciations or None -> None) would silently wipe the CRM's
        # overrides for the opening line specifically. Script wins on key
        # collision (mirrors __init__'s merge above and apply_pronunciations'
        # "extra wins" precedence).
        merged_pronunciations = {
            **(getattr(base_tts, "extra_pronunciations", None) or {}),
            **(self._script.pronunciations or {}),
        } or None
        opening_tts = (
            _replace(
                base_tts, language=opening_lang,
                extra_pronunciations=merged_pronunciations,
            )
            if base_tts is not None else
            _TTSConfig(language=opening_lang, extra_pronunciations=merged_pronunciations)
        )
        try:
            tts_result = await self._engine._tts.synthesize(  # type: ignore[attr-defined]
                rendered, opening_tts,
            )
        except Exception:
            log.exception("opening synthesis failed; skipping")
            return
        if tts_result.audio:
            await audio_sink(tts_result.audio)
        self.session.turns.append(LLMMessage(role="assistant", content=rendered))
        await self.persist_turn("agent", rendered, metadata={"phase": "opening"})

    def _template_vars(self) -> dict[str, str]:
        data = dict(self.session.lead_data or {})
        data.setdefault("lead_name", data.get("name", ""))
        data.setdefault("agent_name", self._script.agent_name)
        data.setdefault("company_name", self._script.company_name)
        # Agent grammatical gender token — use in opening templates instead of
        # hardcoding "rahi"/"raha": e.g. "baat kar {agent_raha_rahi} hoon".
        ag = (self._script.gender or "").strip().lower()
        data.setdefault("agent_raha_rahi", "रहा" if ag == "male" else "रही" if ag == "female" else "")
        # Lead salutation token — inserts " Sir"/" Ma'am"/"" based on lead gender.
        lg = (data.get("lead_gender") or "").strip().lower()
        data.setdefault("lead_salutation", " Sir" if lg == "male" else " Ma'am" if lg == "female" else "")
        # lead_address = combined "Name Sir" / "Name Ma'am" / "Name जी" / "Sir" / ""
        # Use {lead_address} in greeting templates instead of separate tokens.
        _name = (data.get("lead_name") or "").strip()
        _sal = data.get("lead_salutation", "")
        if _name and _sal:
            _addr = f" {_name}{_sal}"
        elif _name:
            _addr = f" {_name} जी"
        else:
            _addr = _sal  # " Sir" / " Ma'am" / ""
        data.setdefault("lead_address", _addr)
        return {k: str(v) for k, v in data.items()}

    async def _run_turn_with_backstop(
        self,
        coro,
        cancel_event: asyncio.Event,
        *,
        hard_timeout_s: Optional[float] = None,
        grace_s: Optional[float] = None,
    ) -> TurnResult:
        """Run a pipeline-turn coroutine with a hard backstop timeout.

        LLM generation and each TTS sentence now enforce their own internal
        budgets (PipelineEngine.run_turn_text), so this outer cap is a
        last-resort guard against a truly wedged call, not the primary timeout
        mechanism. On expiry we SET cancel_event (the same signal barge-in
        uses) rather than cancelling the task directly — a bare task-cancel
        can land inside `await tts_task` inside run_turn_text, which does NOT
        propagate to the independently-running TTS worker Task, letting it
        keep calling audio_sink after this turn has already returned control
        to LISTENING (the original mid-sentence cutoff bug). Only if the
        coroutine still hasn't wound down after a grace period do we fall
        back to a hard cancel.

        ``hard_timeout_s``/``grace_s`` default (via ``None``, resolved here
        rather than as literal default values) to the current module-level
        HARD_TURN_TIMEOUT_S/_BACKSTOP_GRACE_S — a plain default value would
        bind at class-definition time and stop tracking test monkeypatches of
        those module attributes. The retry in _finish_turn passes its own,
        tighter values explicitly.
        """
        if hard_timeout_s is None:
            hard_timeout_s = HARD_TURN_TIMEOUT_S
        if grace_s is None:
            grace_s = _BACKSTOP_GRACE_S
        task = asyncio.create_task(coro)
        done, _ = await asyncio.wait({task}, timeout=hard_timeout_s)
        if task in done:
            return task.result()

        log.error("turn exceeded hard cap of %.0fs; signalling cancellation", hard_timeout_s)
        cancel_event.set()
        done, _ = await asyncio.wait({task}, timeout=grace_s)
        if task in done:
            return task.result()

        log.error("turn still wedged after %.0fs grace period; force-cancelling", grace_s)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return TurnResult(
            user_text="", user_language=None, user_confidence=0.0,
            agent_text="", audio_bytes_sent=0, metrics=TurnMetrics(), cancelled=True,
        )

    async def handle_turn(self, captured_audio: bytes, audio_sink: AudioSink) -> TurnOutcome:
        """Drive one user-utterance -> agent-response cycle.

        State transitions happen at the natural boundaries:
        LISTENING -> PROCESSING (utterance complete) -> RESPONDING ->
        LISTENING (response delivered) | ESCALATING | ENDED.
        """
        if self.state.state is not State.LISTENING:
            raise RuntimeError(
                f"handle_turn called from {self.state.state.value}, expected listening"
            )

        # Utterance complete (the telephony layer determined this via VAD).
        await self.state.fire(Event.UTTERANCE_COMPLETE)

        cancel_event = asyncio.Event()
        try:
            pipeline_result = await self._run_turn_with_backstop(
                self._engine.run_turn(
                    captured_audio=captured_audio,
                    history=self._history_window(),
                    audio_sink=audio_sink,
                    cancel_event=cancel_event,
                    language=to_bcp47(self._active_language),
                ),
                cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure (incl. timeout) must not drop the call
            # STT/LLM/TTS outage (e.g. a retired model 404). Walk the state
            # machine back to LISTENING (PROCESSING -> RESPONDING -> LISTENING)
            # so the conversation survives instead of crashing the call.
            log.exception("pipeline turn failed; recovering to LISTENING")
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="",
                    action="continue",
                    parse_error=f"pipeline error: {type(exc).__name__}: {exc}",
                ),
                pipeline=TurnResult(
                    user_text="",
                    user_language=None,
                    user_confidence=0.0,
                    agent_text="",
                    audio_bytes_sent=0,
                    metrics=TurnMetrics(),
                ),
            )

        if pipeline_result.cancelled:
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="", action="continue", parse_error="barge-in"
                ),
                pipeline=pipeline_result,
            )

        return await self._finish_turn(pipeline_result, audio_sink=audio_sink, cancel_event=cancel_event)

    async def _finish_turn(
        self,
        pipeline_result: TurnResult,
        *,
        audio_sink: AudioSink,
        cancel_event: asyncio.Event,
    ) -> TurnOutcome:
        """Record turns, parse the structured response, apply slots, and advance
        the state machine. Shared by handle_turn (batch STT) and
        handle_turn_text (streaming STT)."""
        # Backdated by the original attempt's own measured duration, so that
        # when the retry call below computes its total_latency_ms as
        # time.perf_counter() - t_overall (see PipelineEngine.run_turn_text),
        # the result genuinely covers the full episode — the original failed
        # attempt plus the retry — rather than only the retry's own (much
        # shorter) duration.
        t_recovery_start = time.perf_counter() - (pipeline_result.metrics.total_latency_ms / 1000)

        # Empty STT — no real user turn happened. Walk the state machine back to
        # LISTENING and let the silence handler decide what to do next.
        if not pipeline_result.user_text:
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="", action="continue", parse_error="empty STT"
                ),
                pipeline=pipeline_result,
            )

        response = parse_voicebot_response(pipeline_result.agent_text)
        if response.parse_error:
            spoken = _join_spoken_sentences(pipeline_result.sentences_spoken)
            if spoken:
                # The malformed JSON's response_text was already extracted and
                # spoken (sentence-by-sentence, ahead of the full response) before
                # this parse failure surfaced downstream — audio has already gone
                # out. Reuse it instead of the canned fallback line so the
                # transcript/history match what the caller actually heard, and
                # let the conversation continue normally rather than forcing a
                # redundant "please repeat that" the caller never heard asked.
                response.response_text = spoken
                _apply_recovered_fields(response, pipeline_result.agent_text)
            else:
                # Nothing was spoken this turn at all — no audio has gone out yet,
                # so it's safe to regenerate once before falling back. Reuses the
                # already-transcribed user_text; does not re-run STT. Goes through
                # the same backstop wrapper as every other pipeline call in this
                # class, but with its OWN tighter budget (_RETRY_HARD_TIMEOUT_S /
                # _RETRY_BACKSTOP_GRACE_S) rather than the primary attempt's — a
                # retry that also stalls before a token arrives means the provider
                # is genuinely down right now, so doubling down on the full
                # last-resort wait would only make an already-lost call wait
                # longer. Reuses the turn's REAL cancel_event (not a fresh one) so
                # a barge-in during the retry can still interrupt it and so the
                # backstop's own expiry mechanism (cancel_event.set()) works.
                log.warning(
                    "retrying turn: %s produced no spoken output", response.parse_error
                )
                try:
                    retried = await self._run_turn_with_backstop(
                        self._engine.run_turn_text(
                            pipeline_result.user_text,
                            self._history_window(),
                            audio_sink,
                            cancel_event,
                            user_language=pipeline_result.user_language,
                            user_confidence=pipeline_result.user_confidence,
                            stt_latency_ms=pipeline_result.metrics.stt_latency_ms,
                            t_overall=t_recovery_start,
                            language=to_bcp47(self._active_language),
                        ),
                        cancel_event,
                        hard_timeout_s=_RETRY_HARD_TIMEOUT_S,
                        grace_s=_RETRY_BACKSTOP_GRACE_S,
                    )
                except Exception:  # noqa: BLE001 - the retry itself must not crash the turn
                    log.exception("retry after empty/unparseable LLM response failed")
                    retried = None
                if retried is not None:
                    retried_spoken = _join_spoken_sentences(retried.sentences_spoken)
                    if not retried.cancelled:
                        pipeline_result = retried
                        response = parse_voicebot_response(pipeline_result.agent_text)
                        if response.parse_error and retried_spoken:
                            response.response_text = retried_spoken
                            _apply_recovered_fields(response, pipeline_result.agent_text)
                    elif retried_spoken:
                        # The retry itself got cancelled (LLM budget exceeded,
                        # barge-in, or MAX_CONSECUTIVE_TTS_FAILURES — all set
                        # cancel_event / return cancelled=True in engine.py), but
                        # real content was already spoken to the caller before the
                        # cancellation fired. Reuse the spoken text rather than
                        # discarding it — discarding here would recreate the exact
                        # transcript/audio divergence bug this whole recovery path
                        # exists to fix, just relocated to the retry.
                        #
                        # But do NOT recover action/sentiment/conversation_phase
                        # here (no _apply_recovered_fields call): cancellation of
                        # a retry can be a barge-in, and this file has a
                        # deliberate policy elsewhere (see the two
                        # parse_error="barge-in" sites in handle_turn /
                        # handle_turn_text) of forcing action="continue" whenever
                        # the caller interrupted before hearing the full reply.
                        # Acting on a terminal decision (close_positive/transfer)
                        # the caller never actually heard announced is wrong,
                        # so we force "continue" here too instead of recovering
                        # a possibly-terminal action from a raw JSON fragment the
                        # caller didn't finish hearing.
                        pipeline_result = retried
                        response = VoiceBotResponse(
                            response_text=retried_spoken,
                            action="continue",
                            parse_error=response.parse_error or "retry cancelled",
                        )
        # Update the conversation's active language from this turn's signals (the
        # LLM's explicitly-reported language wins; STT detection is the fallback).
        # Takes effect from the next turn's STT/TTS.
        self._active_language = resolve_active_language(
            self._active_language,
            stt_lang=pipeline_result.user_language,
            llm_lang=(response.raw or {}).get("language"),
        )
        await self.apply_signal(
            user_text=pipeline_result.user_text,
            agent_text=response.response_text,
            action=response.action,
            updated_slots=response.updated_slots,
            sentiment=response.sentiment,
            phase=response.conversation_phase,
            metrics_dict=pipeline_result.metrics.__dict__,
        )
        return TurnOutcome(response=response, pipeline=pipeline_result)

    async def apply_signal(
        self,
        *,
        user_text: str,
        agent_text: str,
        action: str,
        updated_slots: Optional[dict[str, Any]] = None,
        sentiment: Optional[str] = None,
        phase: Optional[str] = None,
        metrics_dict: Optional[dict[str, Any]] = None,
        metrics_mode: str = "layered",
        metrics_provider_override: Optional[dict[str, Optional[str]]] = None,
    ) -> dict[str, Any]:
        """Record one completed turn (transcript + slots + sentiment) and advance
        the state machine from the turn's ``action``. Shared by the cascade
        (_finish_turn, which parses the JSON envelope) and the S2S Live bridge
        (which gets the same fields from a ``record_turn_signal`` tool-call).

        Assumes the machine is mid-turn (an utterance completed): fires
        LLM_RESPONSE_READY, then the action-appropriate transition back to
        LISTENING / ESCALATING->ENDED / ENDED. Returns the applied slot dict."""
        if user_text:
            self.session.turns.append(LLMMessage(role="user", content=user_text))
            await self.persist_turn("user", user_text)

        await self.state.fire(Event.LLM_RESPONSE_READY)

        applied = self.slots.apply_updates(updated_slots or {})

        # Merge lead_gender back into session.lead_data so _template_vars()
        # picks it up immediately for the next turn's greeting tokens.
        if "lead_gender" in applied and self.session.lead_data is not None:
            self.session.lead_data["lead_gender"] = applied["lead_gender"]
            # Rebuild the system prompt so the "gender unknown → skip gendered
            # titles" directive is replaced with the now-known gender.  In
            # layered mode the prompt is sent per-turn (as turns[0]), so we must
            # update it in-place; otherwise the LLM keeps ignoring the salutation.
            updated_prompt = build_voicebot_system_prompt(
                script=self._script,
                schema=self.slots.schema,
                lead_data=self.session.lead_data,
                extra_directives=self._extra_directives,
                kb_context=self._kb_context,
            )
            self._system_prompt = updated_prompt
            if self.session.turns:
                self.session.turns[0] = LLMMessage(role="system", content=updated_prompt)

        if agent_text:
            self.session.turns.append(LLMMessage(role="assistant", content=agent_text))
            await self.persist_turn(
                "agent",
                agent_text,
                metadata={
                    "action": action,
                    "sentiment": sentiment,
                    "phase": phase,
                    "applied_slots": applied,
                    "metrics": metrics_dict or {},
                },
            )
            if metrics_dict:
                # Redis-persisted turn metadata (above) is TTL-bound and expires;
                # this log line is the durable record used for cross-provider
                # latency benchmarking (Northflank's log retention outlives the
                # session). Only fired for real pipeline turns — callers that
                # skip STT/LLM/TTS timing (e.g. the S2S Live bridge's
                # apply_signal call) don't pass metrics_dict at all, so this
                # naturally stays silent there.
                if metrics_provider_override is not None:
                    stt_provider = metrics_provider_override.get("stt_provider")
                    llm_provider = metrics_provider_override.get("llm_provider")
                    tts_provider = metrics_provider_override.get("tts_provider")
                else:
                    stt_provider = type(getattr(self._engine, "_stt", None)).__name__
                    llm_provider = type(getattr(self._engine, "_llm", None)).__name__
                    tts_provider = type(getattr(self._engine, "_tts", None)).__name__
                log.info("voice turn metrics", extra={
                    "session_id": self.session.session_id,
                    "campaign_id": self.session.campaign_id,
                    "stt_provider": stt_provider,
                    "llm_provider": llm_provider,
                    "tts_provider": tts_provider,
                    "metrics": metrics_dict,
                })
                if self._record_metric is not None:
                    try:
                        await self._record_metric({
                            "session_id": self.session.session_id,
                            "campaign_id": self.session.campaign_id,
                            "mode": metrics_mode,
                            "stt_provider": stt_provider,
                            "llm_provider": llm_provider,
                            "tts_provider": tts_provider,
                            "action": action,
                            "metrics": metrics_dict,
                        })
                    except Exception:  # noqa: BLE001 - never break a live call on a metrics-write failure
                        log.warning(
                            "record_metric failed; continuing without persistence",
                            exc_info=True,
                        )
        if sentiment:
            self.session.sentiment_history.append(sentiment)

        if action in _ESCALATION_ACTIONS:
            # RESPONDING -> ESCALATING -> ENDED. We complete the escalation
            # immediately: the agent's conversational role is done (the actual
            # transfer/callback is handled downstream from the disposition), and
            # leaving the agent in ESCALATING would crash the next turn dispatch
            # ("handle_turn_text called from escalating, expected listening").
            await self.state.fire(Event.ESCALATION_REQUESTED)
            await self.state.fire(Event.ESCALATION_COMPLETE)
        elif action in _END_ACTIONS:
            await self.state.fire(Event.RESPONSE_DELIVERED)
            await self.state.fire(Event.HANGUP)
        else:
            await self.state.fire(Event.RESPONSE_DELIVERED)

        await self.persist_state(extra={"last_action": action})
        return applied

    async def handle_turn_text(
        self, user_text: str, audio_sink: AudioSink, cancel_event=None
    ) -> TurnOutcome:
        """Drive one turn from an already-transcribed utterance (streaming STT).

        Mirrors handle_turn but skips STT: the transcript is supplied directly.
        An optional ``cancel_event`` (asyncio.Event or similar) is forwarded to
        the engine so a barge-in from the telephony layer can abort the LLM/TTS
        pipeline mid-stream.
        """
        if self.state.state is not State.LISTENING:
            raise RuntimeError(
                f"handle_turn_text called from {self.state.state.value}, expected listening"
            )

        await self.state.fire(Event.UTTERANCE_COMPLETE)

        cancel_event = cancel_event or asyncio.Event()
        try:
            pipeline_result = await self._run_turn_with_backstop(
                self._engine.run_turn_text(
                    user_text,
                    self._history_window(),
                    audio_sink,
                    cancel_event,
                    language=to_bcp47(self._active_language),
                ),
                cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure (incl. timeout) must not drop the call
            log.exception("pipeline turn (text) failed; recovering to LISTENING")
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="",
                    action="continue",
                    parse_error=f"pipeline error: {type(exc).__name__}: {exc}",
                ),
                pipeline=TurnResult(
                    user_text="",
                    user_language=None,
                    user_confidence=0.0,
                    agent_text="",
                    audio_bytes_sent=0,
                    metrics=TurnMetrics(),
                ),
            )

        if pipeline_result.cancelled:
            # Barge-in: user interrupted before hearing the reply. Keep the user
            # turn (it was said and processed); drop the abandoned agent reply;
            # return to LISTENING. The interruption follows as the next turn.
            if pipeline_result.user_text:
                self.session.turns.append(
                    LLMMessage(role="user", content=pipeline_result.user_text)
                )
                await self.persist_turn("user", pipeline_result.user_text)
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="", action="continue", parse_error="barge-in"
                ),
                pipeline=pipeline_result,
            )

        return await self._finish_turn(pipeline_result, audio_sink=audio_sink, cancel_event=cancel_event)

    async def handle_silence_timeout(self, audio_sink: AudioSink) -> Optional[TurnOutcome]:
        """User went silent in LISTENING — re-prompt or end the call.

        Currently emits no real audio (the LLM call is skipped). The
        telephony layer is expected to play a pre-rolled "are you there?"
        prompt and then call ``handle_turn`` again. We just advance the
        state machine.
        """
        if self.state.state is not State.LISTENING:
            return None
        await self.state.fire(Event.SILENCE_TIMEOUT)
        # Auto-return to LISTENING so the call doesn't get stuck.
        await self.state.fire(Event.RESPONSE_DELIVERED)
        return None

    async def handle_extended_silence(self) -> None:
        if self.state.state is State.LISTENING:
            await self.state.fire(Event.EXTENDED_SILENCE)
        await self.persist_state()

    async def handle_hangup(self) -> None:
        # If we're already terminal (e.g. close_positive set last_action),
        # don't overwrite the persisted final state.
        if self.state.is_terminal:
            return
        await self.state.fire_if_possible(Event.HANGUP)
        await self.persist_state()
