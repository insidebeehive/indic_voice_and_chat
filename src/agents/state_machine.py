"""Agent state machine (PRD §8).

States:
    IDLE        - waiting to be assigned a session
    LISTENING   - capturing user audio / text
    PROCESSING  - STT done, LLM reasoning
    RESPONDING  - playing TTS / sending chat reply
    ESCALATING  - transferring to human, scheduling callback
    ENDED       - terminal; no further transitions

Events drive transitions. Invalid transitions raise ``InvalidTransition`` so
bugs surface loudly in tests rather than corrupting conversation state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Optional

from src.utils.logging import debug_event

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ESCALATING = "escalating"
    ENDED = "ended"


class Event(str, Enum):
    CALL_CONNECTED = "call_connected"
    UTTERANCE_COMPLETE = "utterance_complete"
    LLM_RESPONSE_READY = "llm_response_ready"
    RESPONSE_DELIVERED = "response_delivered"
    INTERRUPTED = "interrupted"
    ESCALATION_REQUESTED = "escalation_requested"
    ESCALATION_COMPLETE = "escalation_complete"
    SILENCE_TIMEOUT = "silence_timeout"
    EXTENDED_SILENCE = "extended_silence"
    MAX_DURATION_REACHED = "max_duration_reached"
    HANGUP = "hangup"


# Transition table: (from_state, event) -> next_state
_TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.IDLE, Event.CALL_CONNECTED): State.LISTENING,
    (State.LISTENING, Event.UTTERANCE_COMPLETE): State.PROCESSING,
    (State.PROCESSING, Event.LLM_RESPONSE_READY): State.RESPONDING,
    (State.RESPONDING, Event.RESPONSE_DELIVERED): State.LISTENING,
    (State.RESPONDING, Event.INTERRUPTED): State.LISTENING,
    (State.RESPONDING, Event.ESCALATION_REQUESTED): State.ESCALATING,
    (State.LISTENING, Event.SILENCE_TIMEOUT): State.RESPONDING,
    (State.LISTENING, Event.EXTENDED_SILENCE): State.ENDED,
    (State.ESCALATING, Event.ESCALATION_COMPLETE): State.ENDED,
}

# Events that always end the conversation regardless of current state
_TERMINAL_EVENTS = {Event.MAX_DURATION_REACHED, Event.HANGUP}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class TransitionRecord:
    from_state: State
    event: Event
    to_state: State
    at: datetime = field(default_factory=datetime.utcnow)


TransitionListener = Callable[[TransitionRecord], Awaitable[None]]


class AgentStateMachine:
    def __init__(self, initial: State = State.IDLE) -> None:
        self._state = initial
        self._history: list[TransitionRecord] = []
        self._listeners: list[TransitionListener] = []
        self._lock = asyncio.Lock()

    @property
    def state(self) -> State:
        return self._state

    @property
    def history(self) -> list[TransitionRecord]:
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._state is State.ENDED

    def add_listener(self, listener: TransitionListener) -> None:
        self._listeners.append(listener)

    def can_handle(self, event: Event) -> bool:
        if event in _TERMINAL_EVENTS:
            return self._state is not State.ENDED
        return (self._state, event) in _TRANSITIONS

    async def fire(self, event: Event) -> State:
        async with self._lock:
            if self._state is State.ENDED:
                # Rejected, then raised -- callers of fire() are expected to
                # let this propagate (see the module docstring), but the
                # raise alone reaches only whichever try/except is up the
                # stack, if any. This line is the query key: it lands
                # regardless of whether anything catches the exception.
                # `sm_event`, not `event` -- debug_event's own second
                # positional parameter is named `event` (the event NAME
                # string, e.g. "state_machine transition rejected"), so
                # passing event=event.value here collides with it exactly
                # like a LogRecord key would.
                debug_event(
                    log, "state_machine transition rejected",
                    from_state=self._state.value, sm_event=event.value,
                    reason="already_ended",
                )
                raise InvalidTransition(f"cannot fire {event} from ENDED")
            if event in _TERMINAL_EVENTS:
                next_state = State.ENDED
            else:
                key = (self._state, event)
                if key not in _TRANSITIONS:
                    debug_event(
                        log, "state_machine transition rejected",
                        from_state=self._state.value, sm_event=event.value,
                        reason="no_transition_defined",
                    )
                    raise InvalidTransition(
                        f"no transition from {self._state.value} on {event.value}"
                    )
                next_state = _TRANSITIONS[key]
            record = TransitionRecord(
                from_state=self._state,
                event=event,
                to_state=next_state,
            )
            self._state = next_state
            self._history.append(record)
            debug_event(
                log, "state_machine transition applied",
                from_state=record.from_state.value, sm_event=record.event.value,
                to_state=record.to_state.value,
            )

        # Listeners run outside the lock so they can themselves call fire().
        for cb in self._listeners:
            await cb(record)
        return next_state

    async def fire_if_possible(self, event: Event) -> Optional[State]:
        """Fire only if the transition is legal — otherwise no-op."""
        if not self.can_handle(event):
            # The silent half of "rejected": no exception, so a caller that
            # never checks the return value (several in voicebot.py don't)
            # has no other way to learn the event was dropped on the floor.
            debug_event(
                log, "state_machine transition skipped",
                from_state=self._state.value, sm_event=event.value,
                reason="illegal_transition",
            )
            return None
        return await self.fire(event)
