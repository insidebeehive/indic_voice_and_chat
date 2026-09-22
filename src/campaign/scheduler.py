"""Call scheduler + retry logic.

Owns three policies:
1. Calling-hours window (delegated to ``CallingHoursPolicy``).
2. Outbound rate limit (calls/minute) implemented as a sliding window.
3. Retry timing (max attempts + interval, schedules ``next_retry_at``).

The scheduler is purely advisory — it picks which lead to call next from
an in-memory queue and tells the orchestrator when it's allowed to dial.
The orchestrator owns the actual dispatch and concurrency cap.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Optional

from src.campaign.dnd_filter import IST, CallingHoursPolicy, DNDFilter
from src.campaign.models import Lead, LeadStatus
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    max_retry_attempts: int = 3
    retry_interval_hours: int = 2


@dataclass
class RateLimitConfig:
    calls_per_minute: int = 20
    max_concurrent_calls: int = 10


@dataclass
class SchedulerDecision:
    """What ``poll`` returns."""

    leads: list[Lead] = field(default_factory=list)
    blocked_by_hours: bool = False
    blocked_by_rate: bool = False
    blocked_by_concurrency: bool = False
    next_eligible_at: Optional[datetime] = None


class CallScheduler:
    def __init__(
        self,
        hours: CallingHoursPolicy,
        dnd_filter: DNDFilter,
        retry: Optional[RetryConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ) -> None:
        self._hours = hours
        self._dnd = dnd_filter
        self._retry = retry or RetryConfig()
        self._rate = rate_limit or RateLimitConfig()
        self._dispatched_at: collections.deque[datetime] = collections.deque()
        # Edge-detection state for DEBUG logging: `poll()` re-evaluates every
        # lead/every blocking condition on every call, and the orchestrator's
        # run loop calls `poll()` in a near-busy spin while blocked or while a
        # lead sits waiting (DND, retry-not-due) -- logging on every check
        # would flood exactly like the audio-frame case docs/debug-logging.md
        # warns about. These track the last-seen state per (scheduler /
        # lead), so a debug_event fires only when it actually changes.
        self._last_block_state: Optional[str] = None
        self._last_ineligible_reason: dict[str, str] = {}

    @property
    def retry_config(self) -> RetryConfig:
        return self._retry

    # --- Lead-state transitions -----------------------------------------

    def mark_attempted(self, when: Optional[datetime] = None) -> None:
        """Record a dispatched call for rate-limit accounting."""
        ts = self._now(when)
        self._dispatched_at.append(ts)
        self._evict_stale(ts)

    def schedule_retry(self, lead: Lead, now: Optional[datetime] = None) -> Lead:
        """Bump retry counter, set ``next_retry_at``, mark RETRY or FAILED."""
        when = self._now(now)
        lead.retry_count += 1
        if lead.retry_count >= self._retry.max_retry_attempts:
            lead.status = LeadStatus.FAILED
            lead.next_retry_at = None
            if log.isEnabledFor(logging.DEBUG):
                debug_event(
                    log, "campaign schedule_retry failed",
                    lead_id=lead.id, campaign_id=lead.campaign_id,
                    retry_count=lead.retry_count, max_retry_attempts=self._retry.max_retry_attempts,
                )
            return lead
        lead.status = LeadStatus.RETRY
        lead.next_retry_at = when + timedelta(hours=self._retry.retry_interval_hours)
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "campaign schedule_retry scheduled",
                lead_id=lead.id, campaign_id=lead.campaign_id,
                retry_count=lead.retry_count, next_retry_at=lead.next_retry_at.isoformat(),
            )
        return lead

    # --- Queries ----------------------------------------------------------

    def dnd_blocked(self, leads: Iterable[Lead]) -> list[Lead]:
        """Leads whose number is in the DND store. Query only -- status
        transitions belong to the orchestrator, which owns lead state.

        Uses the same ``self._dnd.is_blocked`` check as ``_lead_eligible``
        (which also respects the filter's ``enabled`` flag) so the two never
        disagree about what counts as blocked.
        """
        return [lead for lead in leads if self._dnd.is_blocked(lead.phone_number)]

    # --- Polling --------------------------------------------------------

    def poll(
        self,
        leads: Iterable[Lead],
        active_count: int,
        now: Optional[datetime] = None,
        max_pick: Optional[int] = None,
    ) -> SchedulerDecision:
        """Return up to ``max_pick`` leads ready to dial right now."""
        when = self._now(now)
        decision = SchedulerDecision()

        if not self._hours.can_call_now(when):
            decision.blocked_by_hours = True
            decision.next_eligible_at = self._hours.next_call_window(when)
            self._log_block_transition("hours", next_eligible_at=decision.next_eligible_at.isoformat())
            return decision

        if active_count >= self._rate.max_concurrent_calls:
            decision.blocked_by_concurrency = True
            self._log_block_transition(
                "concurrency", active_count=active_count,
                max_concurrent_calls=self._rate.max_concurrent_calls,
            )
            return decision

        self._evict_stale(when)
        rate_room = max(0, self._rate.calls_per_minute - len(self._dispatched_at))
        if rate_room == 0:
            decision.blocked_by_rate = True
            # Earliest the rate window will free a slot
            if self._dispatched_at:
                decision.next_eligible_at = self._dispatched_at[0] + timedelta(seconds=60)
            self._log_block_transition(
                "rate", dispatched_count=len(self._dispatched_at),
                calls_per_minute=self._rate.calls_per_minute,
            )
            return decision

        self._log_block_transition(None)

        concurrency_room = self._rate.max_concurrent_calls - active_count
        budget = min(rate_room, concurrency_room)
        if max_pick is not None:
            budget = min(budget, max_pick)

        for lead in leads:
            if budget <= 0:
                break
            if not self._lead_eligible(lead, when):
                continue
            decision.leads.append(lead)
            budget -= 1
        return decision

    # --- Internals ------------------------------------------------------

    def _log_block_transition(self, state: Optional[str], **detail: object) -> None:
        """Log only when the poll-blocked reason CHANGES.

        ``poll()`` is called every tick of the orchestrator's run loop,
        including a near-busy spin while blocked, so logging every call
        would flood the exact way a per-audio-frame log would (see
        docs/debug-logging.md's "log transitions, not frames"). This fires
        once per transition -- e.g. once when calling-hours close, not once
        per millisecond until they reopen.
        """
        if state == self._last_block_state:
            return
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "campaign poll blocked_transition",
                previous_state=self._last_block_state, state=state, **detail,
            )
        self._last_block_state = state

    def _lead_eligible(self, lead: Lead, when: datetime) -> bool:
        reason: Optional[str] = None
        detail: dict[str, object] = {}
        if lead.status not in (LeadStatus.PENDING, LeadStatus.RETRY):
            reason, detail = "status", {"lead_status": lead.status.value}
        elif self._dnd.is_blocked(lead.phone_number):
            reason, detail = "dnd_blocked", {}
        elif lead.next_retry_at is not None:
            target = lead.next_retry_at
            if target.tzinfo is None:
                target = target.replace(tzinfo=IST)
            if when < target:
                reason, detail = "retry_not_due", {"next_retry_at": target.isoformat()}

        # Same flooding concern as `_log_block_transition` above, but keyed
        # per lead: a DND-blocked or not-yet-due lead is re-checked on every
        # `poll()` call for as long as it sits in the queue, so only log when
        # THIS lead's verdict changes. Historically a DND-blocked lead never
        # left the queue this way (nothing transitioned `lead.status`), so a
        # single "dnd_blocked" line followed by silence was itself the signal
        # something was stuck. That's no longer the failure mode: the
        # orchestrator's run loop now calls `dnd_blocked()` (below) at the top
        # of every iteration and transitions each blocked lead to
        # `LeadStatus.DND` *before* calling `poll()`, so under that loop a
        # lead this branch flags is removed from `remaining` -- and stops
        # being passed to `poll()` at all -- on the very next iteration. This
        # DND branch remains as a defensive, redundant check for any caller
        # that invokes `poll()` directly without going through the
        # orchestrator's sweep (e.g. the scheduler's own unit tests), not as
        # the primary mechanism that clears a DND-blocked lead.
        previous = self._last_ineligible_reason.get(lead.id)
        if reason != previous:
            if log.isEnabledFor(logging.DEBUG):
                debug_event(
                    log, "campaign lead_eligibility transition",
                    lead_id=lead.id, campaign_id=lead.campaign_id,
                    previous_reason=previous, reason=reason, **detail,
                )
            if reason is None:
                self._last_ineligible_reason.pop(lead.id, None)
            else:
                self._last_ineligible_reason[lead.id] = reason

        return reason is None

    def _evict_stale(self, when: Optional[datetime] = None) -> None:
        threshold = self._now(when) - timedelta(seconds=60)
        while self._dispatched_at and self._dispatched_at[0] < threshold:
            self._dispatched_at.popleft()

    def _now(self, when: Optional[datetime]) -> datetime:
        if when is None:
            return datetime.now(IST)
        if when.tzinfo is None:
            return when.replace(tzinfo=IST)
        return when.astimezone(IST)
