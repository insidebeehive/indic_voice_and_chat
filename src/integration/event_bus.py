"""Internal event bus + event types (PRD §9).

In-process publish/subscribe with async handlers. Subscribers can match by
exact event type or use ``"*"`` to receive everything (useful for the
webhook fan-out + audit logger).

Handler errors are caught and logged so a misbehaving subscriber doesn't
take down the publisher. Order of delivery to multiple subscribers is the
order they registered, but they run concurrently via ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from src.utils.logging import debug_event

log = logging.getLogger(__name__)


# Canonical event types from PRD §9. Keep in sync.
class EventType:
    CALL_INITIATED = "call.initiated"
    CALL_ANSWERED = "call.answered"
    CALL_COMPLETED = "call.completed"
    CALL_FAILED = "call.failed"
    TURN_COMPLETED = "turn.completed"
    INTENT_DETECTED = "intent.detected"
    SLOT_FILLED = "slot.filled"
    LEAD_QUALIFIED = "lead.qualified"
    LEAD_SCORED = "lead.scored"
    AGENT_ESCALATED = "agent.escalated"
    PROVIDER_ERROR = "provider.error"
    PROVIDER_TIMEOUT = "provider.timeout"


@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    source: str = ""


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register ``handler`` for ``event_type``. Use ``"*"`` for all."""
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    @property
    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subscribers.values())

    async def publish(self, event: Event) -> None:
        """Fan out to exact-match + ``"*"`` subscribers concurrently."""
        handlers = []
        handlers.extend(self._subscribers.get(event.type, []))
        handlers.extend(self._subscribers.get("*", []))
        if not handlers:
            # A published event that reaches nobody is otherwise invisible at
            # any level -- e.g. a lead.qualified event with no WhatsApp
            # handoff subscribed looks, from outside, identical to the
            # handoff having silently declined to act on it.
            debug_event(
                log, "integration event_bus publish_skipped",
                event_type=event.type, reason="no_subscribers",
            )
            return
        debug_event(
            log, "integration event_bus publish_dispatched",
            event_type=event.type, handler_count=len(handlers),
        )
        await asyncio.gather(*(self._safe_invoke(h, event) for h in handlers))

    @staticmethod
    async def _safe_invoke(handler: EventHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:  # noqa: BLE001
            log.exception("event handler raised", extra={"event_type": event.type})


# --- Convenience emitters -----------------------------------------------


async def emit_call_initiated(
    bus: EventBus,
    *,
    tenant_id: str,
    campaign_id: str,
    lead_id: str,
    phone_number: str,
) -> None:
    await bus.publish(Event(
        type=EventType.CALL_INITIATED,
        payload={
            "tenant_id": tenant_id,
            "campaign_id": campaign_id,
            "lead_id": lead_id,
            "phone_number": phone_number,
        },
    ))


async def emit_call_completed(
    bus: EventBus,
    *,
    tenant_id: str,
    session_id: str,
    campaign_id: str,
    lead_id: str,
    disposition: str,
    duration_ms: int,
) -> None:
    await bus.publish(Event(
        type=EventType.CALL_COMPLETED,
        payload={
            "tenant_id": tenant_id,
            "session_id": session_id,
            "campaign_id": campaign_id,
            "lead_id": lead_id,
            "disposition": disposition,
            "duration_ms": duration_ms,
        },
    ))


async def emit_lead_qualified(
    bus: EventBus,
    *,
    tenant_id: str,
    session_id: str,
    lead_id: str,
    interest_level: str,
    slots: dict[str, Any],
) -> None:
    await bus.publish(Event(
        type=EventType.LEAD_QUALIFIED,
        payload={
            "tenant_id": tenant_id,
            "session_id": session_id,
            "lead_id": lead_id,
            "interest_level": interest_level,
            "slots": slots,
        },
    ))
