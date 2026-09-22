"""VoiceBot → ChatBot (WhatsApp) handoff.

Listens on the EventBus for ``lead.qualified`` (PRD §9). When a qualifying
event carries a ``whatsapp_number`` slot, pushes a templated follow-up
message via the configured ``IChatChannel``. Idempotency: the same
session_id is only handed off once.

The follow-up message is built from a small template registry keyed by
``interest_level``. Production deployments will swap the template store for
a CMS-backed one; for Phase 5 a constant dict is enough.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.integration.crm_client import IChatChannel
from src.integration.event_bus import Event, EventBus, EventType
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


_DEFAULT_TEMPLATES: dict[str, dict[str, str]] = {
    "hot": {
        "en": (
            "Hi! Thanks for the call earlier. As promised, here are the details. "
            "Reply with any questions and I'll get back to you."
        ),
        "hi": (
            "Namaste! Abhi ke call ke liye dhanyavaad. Yahan details hain. "
            "Koi sawal ho toh reply kar dijiye, main madad karungi."
        ),
    },
    "warm": {
        "en": "Hi! Following up on our call — when you're ready, just reply here.",
        "hi": "Namaste! Aapse baat karke accha laga. Jab time ho, yahan reply kar dijiye.",
    },
}


@dataclass
class HandoffConfig:
    templates: dict[str, dict[str, str]] = field(default_factory=lambda: dict(_DEFAULT_TEMPLATES))
    qualifying_levels: tuple[str, ...] = ("hot", "warm", "interested_callback")
    default_language: str = "hi"


class WhatsAppHandoff:
    def __init__(
        self,
        bus: EventBus,
        channel: IChatChannel,
        config: Optional[HandoffConfig] = None,
    ) -> None:
        self._bus = bus
        self._channel = channel
        self._cfg = config or HandoffConfig()
        self._dispatched: set[str] = set()  # session_id dedup
        bus.subscribe(EventType.LEAD_QUALIFIED, self._on_qualified)

    @property
    def dispatched_count(self) -> int:
        return len(self._dispatched)

    async def _on_qualified(self, event: Event) -> None:
        payload = event.payload or {}
        session_id = payload.get("session_id")
        if not session_id or session_id in self._dispatched:
            debug_event(
                log, "integration lead_qualified_handoff skipped",
                session_id=session_id,
                reason="no_session_id" if not session_id else "already_dispatched",
            )
            return
        interest = (payload.get("interest_level") or "").lower()
        if interest not in self._cfg.qualifying_levels:
            debug_event(
                log, "integration lead_qualified_handoff skipped",
                session_id=session_id, interest_level=interest,
                reason="interest_not_qualifying",
            )
            return
        slots = payload.get("slots") or {}
        whatsapp = slots.get("whatsapp_number")
        if not whatsapp:
            # The gap this pass exists to close: a hot/warm lead qualifies but
            # never gave a WhatsApp number, so no follow-up goes out -- from
            # outside, indistinguishable from the handoff being broken.
            debug_event(
                log, "integration lead_qualified_handoff skipped",
                session_id=session_id, interest_level=interest,
                reason="no_whatsapp_number",
            )
            return

        language = slots.get("language") or self._cfg.default_language
        template = self._pick_template(interest, language)
        if not template:
            log.warning("no handoff template for interest %r / language %r", interest, language)
            return

        debug_event(
            log, "integration lead_qualified_handoff request",
            session_id=session_id, whatsapp=whatsapp, language=language,
            interest_level=interest,
        )
        try:
            msg_id = await self._channel.send_message(whatsapp, template, language=language)
            self._dispatched.add(session_id)
            debug_event(
                log, "integration lead_qualified_handoff response",
                session_id=session_id, whatsapp=whatsapp, msg_id=msg_id,
            )
        except Exception:  # noqa: BLE001
            log.exception("whatsapp handoff failed", extra={"session_id": session_id})

    def _pick_template(self, interest: str, language: str) -> Optional[str]:
        # Treat dispositional aliases as ``warm`` for templating.
        key = "warm" if interest == "interested_callback" else interest
        bucket = self._cfg.templates.get(key)
        if bucket is None:
            return None
        return bucket.get(language) or bucket.get(self._cfg.default_language) or next(iter(bucket.values()), None)
