"""Transport-agnostic core for speech-to-speech (Gemini Live) bridges.

The dialogue logic — drive a Live session, map its events to audio out +
transcripts + a ``record_turn_signal`` tool-call (→ ``apply_signal``) + native
interruption + per-turn commit + post-call outcome — is identical whether the
transport is the browser dev console or a telephony media stream. Subclasses
supply only the transport: ``_inbound_loop`` (read caller audio → the model),
``_send_audio_out`` (model audio → the wire), ``_send_interrupt`` (flush
playback), and optionally ``_on_start`` / ``_emit_status`` / ``_emit_transcript``
/ ``_deliver_outcome``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

from src.agents.state_machine import Event, State
from src.analysis.call_outcome import analyze_call
from src.interfaces.llm import LLMMessage
from src.interfaces.realtime import IRealtimeSession, RealtimeConfig, RealtimeTool

log = logging.getLogger(__name__)

# The dialogue-control tool the S2S model calls alongside its audio (consumed by
# VoiceBotAgent.apply_signal — the same action/slots the cascade parses from JSON).
RECORD_TURN_SIGNAL = RealtimeTool(
    name="record_turn_signal",
    description="Record the dialogue action and any slot values learned this turn.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING", "enum": [
                "continue", "clarify", "transfer", "schedule_callback",
                "send_info", "close_positive", "close_negative", "end"]},
            "updated_slots": {"type": "OBJECT"},
        },
        "required": ["action"],
    },
)


# Kickoff that makes the agent greet first (dev console). Sent as a turn-based
# message (send_client_content under the hood), which keeps the realtime audio /
# VAD path clean — a realtime-input text kickoff disrupts VAD so caller audio
# never endpoints (see GeminiLiveSession.send_text). Greeting first warms the
# Live session so the caller's first reply gets an instant response instead of
# hitting a cold turn-1 (the "first hello lag").
#
# The opening is kept SHORT (one line) on purpose — script-independent — so a
# caller who prefers another language can interject right away and the call
# continues in their language from turn 1, without a "which language?" menu.
_GREETING_KICKOFF = (
    "[The call has just connected and the caller is on the line. Begin now: open "
    "with your opening line, but keep it to ONE short, warm sentence — then STOP "
    "and wait for the caller to reply. Keep it brief so they respond quickly, and "
    "continue the whole call in whatever language they use in their reply.]"
)


class _BaseLiveBridge:
    """Drive a Live session for one call. Subclass and implement the transport."""

    # Whether the agent opens the call (greets first). The dev console sets this
    # so the model speaks immediately; telephony leaves it False (on a real call
    # the callee says "hello" first).
    _greets_first = False

    # Idle watchdog: if no model event arrives for this long, the Live session is
    # treated as hung — it stopped endpointing / went silent and would otherwise
    # leave the caller stuck in "listening" forever with NO outcome. Force
    # teardown so an outcome is still produced. Generous, because between turns
    # (while the caller is silent) no events flow either — caller speech produces
    # incremental input_transcript events that keep resetting the clock.
    _idle_timeout_s = 30.0
    _idle_check_s = 5.0

    def __init__(self, *, agent, config: RealtimeConfig, connect_session,
                 llm=None, tenant_timezone: str = "Asia/Kolkata") -> None:
        self._agent = agent
        self._config = config
        self._connect_session = connect_session
        self._llm = llm
        self._tenant_timezone = tenant_timezone

        self._session: IRealtimeSession | None = None
        self._stopped = False
        self._outcome_emitted = False
        self._outcome_payload = None  # dict set by _emit_outcome; read for billing
        self._last_action: str | None = None
        # per-turn accumulators
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action: str | None = None
        self._pending_slots: dict = {}
        self._speaking = False
        self._turn_start_at = 0.0        # monotonic ts: start of the current turn
        self._first_audio_at: float | None = None  # monotonic ts: first "audio" event this turn
        self._last_input_transcript_at: float | None = None  # monotonic ts: last "input_transcript" event this turn (proxy for "caller stopped talking")
        self._last_event_at = 0.0   # monotonic ts of the last model event (idle watchdog)
        # one-shot diagnostics: did the model ever HEAR the caller / RESPOND?
        self._dbg_heard_caller = False
        self._dbg_model_audio = False

    # --- run skeleton (shared) ------------------------------------------
    async def _drive(self) -> None:
        events_task = None
        watchdog_task = None
        try:
            # Inside the guard so a failure here (e.g. a store/Redis hiccup in
            # agent.start) tears down cleanly instead of crashing the WS handler.
            await self._agent.start()
            await self._on_start()
            self._session = await self._connect_session(self._config)
            self._mark_activity()
            self._turn_start_at = time.monotonic()
            events_task = asyncio.create_task(self._consume_events())
            watchdog_task = asyncio.create_task(self._idle_watchdog())
            await self._emit_status("listening")
            await self._maybe_greet()
            await self._inbound_loop()
        except Exception:  # noqa: BLE001 - never crash the socket handler
            log.exception("live bridge crashed")
        finally:
            for task in (watchdog_task, events_task):
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except BaseException:  # noqa: BLE001
                        pass
            # Closing the realtime session / transport can raise when the call
            # dropped abnormally — must NOT skip outcome analysis below. Each
            # teardown step is isolated so the outcome is always computed.
            if self._session is not None:
                try:
                    await self._session.aclose()
                except Exception:  # noqa: BLE001
                    log.exception("realtime session close failed on teardown")
            try:
                await self._on_teardown()
            except Exception:  # noqa: BLE001
                log.exception("transport teardown failed")
            # Salvage an in-progress turn (call ended mid-reply) so the transcript
            # + outcome aren't lost.
            try:
                if ((self._user_buf.strip() or self._agent_buf.strip())
                        and not getattr(self._agent.state, "is_terminal", False)):
                    await self._commit_turn()
            except Exception:  # noqa: BLE001 - never let salvage break teardown
                log.exception("turn salvage on teardown failed")
            try:
                await self._emit_outcome()
            except Exception:  # noqa: BLE001 - outcome analysis must never break teardown
                log.exception("outcome emission failed on teardown")
            try:
                await self._agent.handle_hangup()
            except Exception:  # noqa: BLE001
                log.exception("agent hangup failed on teardown")

    async def _maybe_greet(self) -> None:
        """If this transport has the agent greet first, send the kickoff so the
        model opens the call immediately (and the session is warm by the time the
        caller replies). No-op for transports where the caller speaks first."""
        if not self._greets_first or self._session is None:
            return
        try:
            await self._session.send_text(_GREETING_KICKOFF)
        except Exception:  # noqa: BLE001 - a failed kickoff must not break the call
            log.exception("greeting kickoff failed; continuing without it")

    def _mark_activity(self) -> None:
        """Record that a model event just arrived (resets the idle watchdog)."""
        self._last_event_at = time.monotonic()

    async def _idle_watchdog(self) -> None:
        """Force teardown of a hung Live session. If no model event has arrived
        for ``_idle_timeout_s`` the session has gone silent (it stopped
        endpointing the caller / dropped the turn) and would otherwise hang
        forever with no response and no outcome. Setting ``_stopped`` breaks the
        inbound loop on its next frame (both transports stream audio
        continuously), so ``_drive`` tears down and emits the outcome."""
        while not self._stopped:
            await asyncio.sleep(self._idle_check_s)
            if self._stopped or self._session is None:
                return
            idle = time.monotonic() - self._last_event_at
            if idle >= self._idle_timeout_s:
                log.warning("live: no model events for %.0fs — session hung; "
                            "forcing teardown so an outcome is produced", idle)
                self._stopped = True
                return

    # --- model event handling (shared) ----------------------------------
    async def _consume_events(self) -> None:
        assert self._session is not None
        try:
            async for ev in self._session.events():
                self._mark_activity()
                if ev.type == "audio":
                    if not self._dbg_model_audio:
                        self._dbg_model_audio = True
                        log.info("live: model is producing audio (responding)")
                    if self._first_audio_at is None:
                        self._first_audio_at = time.monotonic()
                    await self._send_audio_out(ev.audio, ev.audio_rate)
                elif ev.type == "input_transcript":
                    if not self._dbg_heard_caller:
                        self._dbg_heard_caller = True
                        log.info("live: model heard the caller (input_transcript)",
                                 extra={"first_text": ev.text[:60]})
                    self._last_input_transcript_at = time.monotonic()
                    self._user_buf += ev.text
                    await self._emit_transcript("user", self._user_buf, partial=True)
                elif ev.type == "output_transcript":
                    self._agent_buf += ev.text
                    await self._emit_transcript("agent", self._agent_buf, partial=True)
                elif ev.type == "tool_call":
                    if ev.tool_name == "record_turn_signal":
                        self._pending_action = ev.tool_args.get("action") or self._pending_action
                        self._pending_slots.update(ev.tool_args.get("updated_slots") or {})
                    await self._session.send_tool_response(
                        tool_id=ev.tool_id, name=ev.tool_name, response={"ok": True})
                elif ev.type == "interrupted":
                    self._speaking = False
                    await self._send_interrupt()
                    await self._emit_status("listening")
                elif ev.type == "turn_complete":
                    await self._commit_turn()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a model/stream error ends the call, not crash
            log.exception("live event stream ended (error)")
        finally:
            # The event stream has ended (clean close, error, or teardown cancel).
            # If we got here while the call is still live, the upstream Live socket
            # dropped on us (the preview native-audio model intermittently closes
            # mid-call — connect() already retries the *open* for this reason).
            # Nothing else would notice: the inbound loop keeps feeding audio to a
            # dead session and the caller sits in "listening" with no reply forever.
            # Signal stop so the bridge tears down cleanly and still emits an
            # outcome. (Already-stopped → a normal end; no warning.)
            if not self._stopped:
                if not getattr(self._agent.state, "is_terminal", False):
                    log.warning("live event stream ended while call active "
                                "(upstream session closed?) — stopping bridge")
                self._stopped = True

    async def _commit_turn(self) -> None:
        """Record the completed turn (transcript + slots) and advance state."""
        if getattr(self._agent.state, "is_terminal", False):
            return
        user = self._user_buf.strip()
        agent = self._agent_buf.strip()
        action = self._pending_action or "continue"
        if user or agent:
            if user:
                await self._emit_transcript("user", user, partial=False)
            if agent:
                await self._emit_transcript("agent", agent, partial=False)
            # Drive LISTENING->PROCESSING so apply_signal's transitions are valid.
            if self._agent.state.state is State.LISTENING:
                await self._agent.state.fire(Event.UTTERANCE_COMPLETE)
            now = time.monotonic()
            # Anchor from the last input_transcript event this turn — the best
            # available proxy for "the caller just finished speaking" (the
            # underlying Live session does its own VAD internally but never
            # surfaces a discrete end-of-speech event; see RealtimeEvent). This
            # can lag the true end-of-speech moment slightly, since incoming
            # audio isn't transcribed instantly, but it is far closer than
            # turn_start_at, which would include however long the caller spoke.
            # Falls back to turn_start_at only when no caller speech was heard
            # this turn at all (e.g. the agent-greets-first kickoff turn).
            anchor = (
                self._last_input_transcript_at
                if self._last_input_transcript_at is not None
                else self._turn_start_at
            )
            # S2S has no discrete STT/LLM/TTS phases (one end-to-end audio
            # model) — tts_first_chunk_ms is reinterpreted here as "time to
            # first spoken audio from the realtime model" (0 if the model
            # never produced audio this turn), and every other cascade-only
            # field stays 0. Reuses the existing turn_metrics schema; no new
            # column, disambiguated by mode="s2s" below.
            metrics_dict = {
                "stt_latency_ms": 0,
                "llm_ttft_ms": 0,
                "llm_total_ms": 0,
                "tts_first_chunk_ms": (
                    max(0, int((self._first_audio_at - anchor) * 1000))
                    if self._first_audio_at is not None else 0
                ),
                "tts_total_ms": 0,
                "total_latency_ms": max(0, int((now - anchor) * 1000)),
                "tts_segments_dropped": 0,
            }
            await self._agent.apply_signal(
                user_text=user, agent_text=agent, action=action,
                updated_slots=self._pending_slots,
                metrics_dict=metrics_dict,
                metrics_mode="s2s",
                metrics_provider_override={
                    "stt_provider": None,
                    "llm_provider": type(self._session).__name__ if self._session is not None else None,
                    "tts_provider": None,
                },
            )
            log.info("live turn committed", extra={
                "user_chars": len(user), "agent_chars": len(agent), "action": action,
                "user": user[:120], "agent": agent[:120]})
            self._last_action = action
        else:
            # turn_complete with nothing transcribed/spoken: the model produced no
            # response this turn (e.g. it went silent after a language switch). We
            # stay in listening; log it so a recurring pattern is visible rather
            # than a silent "stuck in listening".
            log.info("live turn completed with no response (model silent)")
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action = None
        self._pending_slots = {}
        self._speaking = False
        self._turn_start_at = time.monotonic()
        self._first_audio_at = None
        self._last_input_transcript_at = None
        if getattr(self._agent.state, "is_terminal", False):
            if action == "transfer":
                await self._on_transfer_hold()
            self._stopped = True
            await self._emit_outcome()
        else:
            await self._emit_status("listening")

    async def _emit_outcome(self) -> None:
        if self._outcome_emitted or self._llm is None:
            return
        self._outcome_emitted = True
        try:
            transcript = [m for m in getattr(self._agent.session, "turns", [])
                          if isinstance(m, LLMMessage)]
            analysis = await analyze_call(
                transcript=transcript, slots=self._agent.slots.values,
                telephony_status=None, final_action=self._last_action,
                tenant_timezone=self._tenant_timezone, now=datetime.now(UTC), llm=self._llm)
        except Exception:  # noqa: BLE001
            log.exception("call outcome analysis failed")
            # Analysis failed, but the transcript is still worth persisting —
            # without this, a failed analysis would silently drop the turns too.
            # `turns` is deliberately left OUT of this shared/base payload:
            # TelephonyLiveBridge._deliver_outcome re-derives turns itself from
            # self._agent.session.turns and merges them into a fresh dict before
            # handing off to the persister, so it doesn't need them here; and
            # GeminiLiveBridge._deliver_outcome json.dumps's this payload straight
            # to the browser, where raw LLMMessage objects aren't serializable.
            await self._deliver_outcome({"type": "outcome_failed"})
            return
        cb = analysis.callback_datetime
        log.info("call outcome", extra={"outcome": analysis.outcome.value,
                 "source": analysis.analysis_source, "summary": analysis.summary[:200]})
        # Stash for the WS handler to persist to the conversation row (billing).
        self._outcome_payload = {
            "outcome": analysis.outcome.value, "summary": analysis.summary,
            "notes": analysis.notes, "callback_datetime": cb.isoformat() if cb else None,
            "source": analysis.analysis_source}
        await self._deliver_outcome({"type": "outcome", **self._outcome_payload,
                                     "callback_phrase": analysis.callback_phrase})

    # --- transport hooks (subclass implements) --------------------------
    async def _inbound_loop(self) -> None:
        """Read caller audio off the transport and forward via session.send_audio."""
        raise NotImplementedError

    async def _send_audio_out(self, pcm16: bytes, rate: int) -> None:
        """Send the model's PCM16 (at ``rate``) to the transport."""
        raise NotImplementedError

    async def _send_interrupt(self) -> None:
        """Flush any buffered/playing agent audio on the transport (barge-in)."""
        raise NotImplementedError

    async def _on_start(self) -> None:
        """Optional: pre-session transport handshake (e.g. read the browser hello)."""

    async def _on_teardown(self) -> None:
        """Optional: stop any transport-side tasks (e.g. the telephony sender)."""

    async def _emit_status(self, status: str) -> None:
        """Optional: surface a status to the transport (browser UI)."""

    async def _emit_transcript(self, role: str, text: str, *, partial: bool) -> None:
        """Optional: surface a transcript to the transport (browser UI)."""

    async def _on_transfer_hold(self) -> None:
        """Called when the 'transfer' action fires, before final bridge teardown.
        Override to pause here while the coordination server looks for a human agent.
        The Twilio WebSocket stays open (caller stays on hold); only the AI session
        should be closed here."""

    async def _deliver_outcome(self, payload: dict) -> None:
        """Optional: deliver the post-call outcome to the transport (browser UI)."""
