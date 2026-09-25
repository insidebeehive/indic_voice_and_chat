"""Durable retry queue for ``session_closed`` tenant-lifecycle webhooks.

Scoped to ``session_closed`` ONLY -- deliberately NOT ``escalation_requested``
(see ``src/api/chat.py``'s ``_escalate_session``: when that webhook fails, the
session is reverted to bot mode so the customer isn't stranded, and a later
durable delivery would open a CRM handoff for a session that's no longer
awaiting one). ``session_closed`` has no such revert path -- the chat session
is already ``ended`` by the time it's sent -- so a delivery that fails inline
is safe, and useful, to keep retrying after the fact.

Why this exists: the tenant's CRM endpoint can intermittently take >5s to
respond, and ``src.integration.tenant_events.deliver``'s in-line budget (3
attempts, 5s httpx timeout each, ~16s total) can exhaust before the CRM
recovers. Without this, that ~16s window is the ONLY chance the CRM ever gets
to receive ``session_closed`` -- past it, the CRM ticket stays open forever
even though our side genuinely ended the session.

A row is enqueued by ``enqueue_webhook_outbox`` (called from
``src.api.chat_webhooks.send_bo_webhook`` when its in-line ``deliver()``
attempt fails for a durable-eligible call) and later claimed and retried by
``src/main.py``'s ``_webhook_outbox_loop`` / ``run_webhook_outbox_once``.

``body`` stores the EXACT dict ``send_bo_webhook`` builds (``{"event":
event_type, **payload}``) so a later retry replays byte-for-byte the same
payload the in-line attempt tried to send, rather than trying to reconstruct
it from session state that may no longer exist by retry time.

``url`` is captured at enqueue time but is a fallback of last resort, not the
primary source at send time -- see ``_webhook_outbox_loop``'s own docstring
for why re-resolving the tenant's current URL/secret at send time (and
skipping, not falling back to this stored URL, when that resolution comes up
empty) is the safer choice.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base, get_sessionmaker
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

# Bounded status enum -- never free text (same convention as
# ChatToolMetricRow.outcome in src/models/chat_turn_metrics.py).
STATUS_PENDING = "pending"
STATUS_DELIVERED = "delivered"
STATUS_DEAD = "dead"


class WebhookOutbox(Base):
    __tablename__ = "webhook_outbox"
    __table_args__ = (
        # Serves the claim query in src/main.py's run_webhook_outbox_once:
        # "pending rows due now", oldest-due first.
        Index("idx_webhook_outbox_status_next_attempt", "status", "next_attempt_at"),
    )

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # NOT a FK to chat_sessions -- same convention as
    # ChatTurnMetric.session_id (src/models/chat_turn_metrics.py): a row here
    # must keep outliving the chat_sessions row it references (the whole
    # point is retrying long after the session -- and its in-memory
    # bookkeeping -- is gone), so cascading its delete would defeat the
    # durable-retry contract this table exists for.
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    body: Mapped[dict] = mapped_column(JSON, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, index=True,
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    # "pending" | "delivered" | "dead"
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=STATUS_PENDING, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(),
    )
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))


async def enqueue_webhook_outbox(
    *, tenant_id: str, session_id: str, event_type: str, url: str, body: dict[str, Any],
) -> None:
    """Enqueue a durable-retry row after an in-line delivery attempt (the
    caller's own ``deliver()`` budget) has been exhausted.

    Best-effort: never raises. A failure to enqueue must not turn an
    already-failed webhook attempt into a hard error for whatever triggered
    it (a customer's WS "end" frame, the idle-timeout auto-close sweep) --
    same never-raises contract as
    ``src.models.chat_turn_metrics.record_chat_turn_metric``. The row is
    simply not queued; the event is lost the same way it was before this
    table existed, no worse.
    """
    try:
        sessionmaker = get_sessionmaker()
        # L1: same naive-UTC "now" for both created_at and next_attempt_at,
        # computed once in Python here rather than left to the column's
        # server_default=func.now(). The 24h dead-cap (run_webhook_outbox_once
        # in src/main.py) measures elapsed time FROM created_at -- if that
        # came from the DB server's own clock instead, a server not pinned to
        # UTC would silently shift when every row goes dead, and the
        # created_at-vs-next_attempt_at delta on the very first read would be
        # measuring two different clocks against each other.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with sessionmaker() as db:
            db.add(WebhookOutbox(
                id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                session_id=session_id,
                event_type=event_type,
                url=url,
                body=body,
                attempts=0,
                next_attempt_at=now,
                status=STATUS_PENDING,
                created_at=now,
            ))
            await db.commit()
        debug_event(
            log, "webhook_outbox enqueue response",
            tenant_id=tenant_id, session_id=session_id, event_type=event_type,
        )
    except Exception:  # noqa: BLE001 - must never break the caller's own failure path
        log.warning(
            "webhook_outbox enqueue failed; session_closed event will not be retried",
            exc_info=True,
            extra={"tenant_id": tenant_id, "session_id": session_id, "event_type": event_type},
        )
