"""Campaign orchestrator / executor.

Owns the per-campaign run loop:

    while not campaign.completed:
        decision = scheduler.poll(leads, active_count)
        for lead in decision.leads:
            spawn agent_runner(lead) -> CallResult
        on each completion:
            scheduler.mark_attempted()
            crm.update_lead(call_result)
            event_bus.emit(call.completed | lead.qualified | ...)
        if decision.next_eligible_at: sleep until then or until interrupt

The actual call dispatch is abstracted behind ``DispatchAgent`` — the
orchestrator passes a ``Lead`` and gets back a ``CallResult``. Tests inject
a fake. Production wiring (Phase 6+) hooks this into the Twilio bridge
factory introduced in Phase 3.

Concurrency cap is enforced via ``asyncio.Semaphore`` matching
``RateLimitConfig.max_concurrent_calls``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Optional

from src.campaign.dnd_filter import IST
from src.campaign.models import (
    CallDisposition,
    CallResult,
    Campaign,
    CampaignStatus,
    Lead,
    LeadStatus,
)
from src.campaign.scheduler import CallScheduler
from src.integration.crm_client import ICRMClient
from src.integration.event_bus import (
    EventBus,
    EventType,
    emit_call_completed,
    emit_call_initiated,
    emit_lead_qualified,
)
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


# Function the orchestrator calls per dispatched lead. Returns the call's
# outcome. Implementations (Phase 6+) start a Twilio call and run the
# VoiceBotAgent loop; tests pass a small fake.
DispatchAgent = Callable[[Campaign, Lead], Awaitable[CallResult]]


_QUALIFYING_DISPOSITIONS = {
    CallDisposition.INTERESTED_CALLBACK,
    CallDisposition.INTERESTED_TRANSFER,
}
_RETRY_DISPOSITIONS = {
    CallDisposition.BUSY_RETRY,
    CallDisposition.VOICEMAIL,
}
_DND_DISPOSITIONS = {CallDisposition.DND_REQUESTED}


@dataclass
class CampaignRun:
    """Mutable state for one campaign run."""

    campaign: Campaign
    leads: dict[str, Lead] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)  # in-flight lead ids
    completed_leads: set[str] = field(default_factory=set)

    @property
    def remaining(self) -> list[Lead]:
        return [
            lead
            for lead in self.leads.values()
            if lead.status not in (LeadStatus.COMPLETED, LeadStatus.FAILED, LeadStatus.DND)
            and lead.id not in self.active
        ]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_leads": len(self.leads),
            "calls_attempted": self.campaign.calls_attempted,
            "calls_answered": self.campaign.calls_answered,
            "leads_qualified": self.campaign.leads_qualified,
            "active": len(self.active),
            "completed": len(self.completed_leads),
        }


class CampaignOrchestrator:
    def __init__(
        self,
        scheduler: CallScheduler,
        dispatch: DispatchAgent,
        bus: EventBus,
        crm: ICRMClient,
        max_concurrent: int = 10,
    ) -> None:
        self._sched = scheduler
        self._dispatch = dispatch
        self._bus = bus
        self._crm = crm
        self._sem = asyncio.Semaphore(max_concurrent)

    # --- public entrypoints ---------------------------------------------

    async def run(
        self,
        campaign: Campaign,
        leads: list[Lead],
        *,
        max_iterations: Optional[int] = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now_fn: Callable[[], datetime] = lambda: datetime.now(IST),
    ) -> CampaignRun:
        """Run the campaign to completion (or to ``max_iterations`` for tests)."""
        run = CampaignRun(campaign=campaign, leads={lead.id: lead for lead in leads})
        run.campaign.total_leads = len(run.leads)
        run.campaign.status = CampaignStatus.ACTIVE

        tasks: list[asyncio.Task] = []
        i = 0
        while True:
            if max_iterations is not None and i >= max_iterations:
                break
            i += 1

            now = now_fn()
            decision = self._sched.poll(
                leads=run.remaining,
                active_count=len(run.active),
                now=now,
            )

            # Spawn dispatches for picked leads.
            for lead in decision.leads:
                lead.status = LeadStatus.IN_FLIGHT
                run.active.add(lead.id)
                self._sched.mark_attempted(now)
                run.campaign.calls_attempted += 1
                if log.isEnabledFor(logging.DEBUG):
                    debug_event(
                        log, "campaign dispatch spawned",
                        lead_id=lead.id, campaign_id=run.campaign.id,
                        phone_number=lead.phone_number,
                        calls_attempted=run.campaign.calls_attempted,
                    )
                tasks.append(asyncio.create_task(self._handle_call(run, lead)))

            # Exit only when there's nothing left to dispatch AND every
            # dispatched task has fully completed (not just popped from
            # ``run.active`` mid-flight).
            if (
                not run.remaining
                and not run.active
                and all(t.done() for t in tasks)
            ):
                run.campaign.status = CampaignStatus.COMPLETED
                break

            await sleep_fn(0)

            if decision.blocked_by_concurrency or decision.blocked_by_rate or decision.blocked_by_hours:
                if max_iterations is not None and decision.blocked_by_hours:
                    break
                await sleep_fn(0.001)
        # Wait for every dispatched task to complete before returning.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if not run.remaining:
            run.campaign.status = CampaignStatus.COMPLETED
        return run

    # --- per-call worker ------------------------------------------------

    async def _handle_call(self, run: CampaignRun, lead: Lead) -> None:
        async with self._sem:
            await emit_call_initiated(
                self._bus,
                tenant_id=run.campaign.tenant_id,
                campaign_id=run.campaign.id,
                lead_id=lead.id,
                phone_number=lead.phone_number,
            )
            try:
                result = await self._dispatch(run.campaign, lead)
            except Exception:  # noqa: BLE001 — dispatch errors mustn't kill the campaign
                log.exception("dispatch crashed", extra={"lead_id": lead.id})
                self._sched.schedule_retry(lead)
                run.active.discard(lead.id)
                return
            await self._on_call_result(run, lead, result)

    async def _on_call_result(self, run: CampaignRun, lead: Lead, result: CallResult) -> None:
        # Push to CRM regardless of disposition.
        try:
            await self._crm.update_lead(result)
        except Exception:  # noqa: BLE001
            log.exception("crm update failed", extra={"lead_id": lead.id})

        # Bookkeeping.
        run.campaign.calls_answered += int(
            result.disposition not in (CallDisposition.VOICEMAIL,)
        )

        # Update lead state from disposition.
        if result.disposition in _DND_DISPOSITIONS:
            lead.status = LeadStatus.DND
            await self._crm.mark_dnd(lead.phone_number)
        elif result.disposition in _RETRY_DISPOSITIONS:
            self._sched.schedule_retry(lead)
        else:
            lead.status = LeadStatus.COMPLETED

        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "campaign lead_result decided",
                lead_id=lead.id, campaign_id=run.campaign.id,
                disposition=result.disposition.value, resulting_status=lead.status.value,
                outcome=(result.outcome.value if result.outcome else None),
            )

        run.active.discard(lead.id)
        if lead.status in (LeadStatus.COMPLETED, LeadStatus.FAILED, LeadStatus.DND):
            run.completed_leads.add(lead.id)

        # Emit completion event.
        await emit_call_completed(
            self._bus,
            tenant_id=run.campaign.tenant_id,
            session_id=result.session_id,
            campaign_id=run.campaign.id,
            lead_id=lead.id,
            disposition=result.disposition.value,
            duration_ms=result.duration_ms,
        )

        # Emit qualification event when applicable.
        if result.disposition in _QUALIFYING_DISPOSITIONS:
            run.campaign.leads_qualified += 1
            await emit_lead_qualified(
                self._bus,
                tenant_id=run.campaign.tenant_id,
                session_id=result.session_id,
                lead_id=lead.id,
                interest_level=(result.interest_level or "warm"),
                slots=result.slots,
            )
