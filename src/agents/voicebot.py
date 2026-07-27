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
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from src.agents.base import AgentSession, BaseAgent
from src.agents.state_machine import AgentStateMachine, Event, State
from src.dialogue.language import normalize_lang, resolve_active_language, to_bcp47
from src.dialogue.prompts import VoiceBotScript, build_voicebot_system_prompt
from src.dialogue.response_parser import VoiceBotResponse, parse_voicebot_response
from src.dialogue.slots import SlotFiller, SlotSchema
from src.interfaces.llm import LLMMessage
from src.pipeline.engine import AudioSink, PipelineEngine, TurnMetrics, TurnResult


log = logging.getLogger(__name__)


@dataclass
class TurnOutcome:
    """The full result of one user-utterance / agent-response cycle."""

    response: VoiceBotResponse
    pipeline: TurnResult


# Map LLM-emitted ``action`` to the state-machine event that follows.
_ESCALATION_ACTIONS = {"transfer", "schedule_callback"}
_END_ACTIONS = {"close_positive", "close_negative", "end"}

# Hard ceiling on one turn's STT+LLM+TTS. Providers have a fat tail (a Gemini
# stream occasionally stalls; observed an 11s turn live), and an unbounded
# ``await`` on a hung provider wedges the agent forever (``_agent_busy`` never
# clears). On timeout we walk the state machine back to LISTENING so the call
# survives. Set above normal turn latency (~3-7s, outliers ~11s) to avoid
# false-firing on merely-slow turns.
TURN_TIMEOUT_S = 20.0

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
            engine._config = _replace_cfg(
                engine._config,
                tts=_replace_cfg(engine._config.tts, extra_pronunciations=script.pronunciations),
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
        opening_tts = (
            _replace(
                base_tts, language=opening_lang,
                extra_pronunciations=self._script.pronunciations or None,
            )
            if base_tts is not None else
            _TTSConfig(language=opening_lang, extra_pronunciations=self._script.pronunciations or None)
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

        try:
            pipeline_result = await asyncio.wait_for(
                self._engine.run_turn(
                    captured_audio=captured_audio,
                    history=self._history_window(),
                    audio_sink=audio_sink,
                    language=to_bcp47(self._active_language),
                ),
                timeout=TURN_TIMEOUT_S,
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

        return await self._finish_turn(pipeline_result)

    async def _finish_turn(self, pipeline_result: TurnResult) -> TurnOutcome:
        """Record turns, parse the structured response, apply slots, and advance
        the state machine. Shared by handle_turn (batch STT) and
        handle_turn_text (streaming STT)."""
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
                log.info("voice turn metrics", extra={
                    "session_id": self.session.session_id,
                    "campaign_id": self.session.campaign_id,
                    "stt_provider": type(getattr(self._engine, "_stt", None)).__name__,
                    "llm_provider": type(getattr(self._engine, "_llm", None)).__name__,
                    "tts_provider": type(getattr(self._engine, "_tts", None)).__name__,
                    "metrics": metrics_dict,
                })
                if self._record_metric is not None:
                    try:
                        await self._record_metric({
                            "session_id": self.session.session_id,
                            "campaign_id": self.session.campaign_id,
                            "mode": "layered",
                            "stt_provider": type(getattr(self._engine, "_stt", None)).__name__,
                            "llm_provider": type(getattr(self._engine, "_llm", None)).__name__,
                            "tts_provider": type(getattr(self._engine, "_tts", None)).__name__,
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

        try:
            pipeline_result = await asyncio.wait_for(
                self._engine.run_turn_text(
                    user_text,
                    self._history_window(),
                    audio_sink,
                    cancel_event,
                    language=to_bcp47(self._active_language),
                ),
                timeout=TURN_TIMEOUT_S,
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

        return await self._finish_turn(pipeline_result)

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
