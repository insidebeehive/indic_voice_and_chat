from __future__ import annotations

from datetime import datetime
from typing import Optional

import pytest

from src.campaign.dnd_filter import IST, CallingHoursPolicy, DNDFilter, InMemoryDNDStore
from src.campaign.models import (
    CallDisposition,
    CallResult,
    Campaign,
    CampaignStatus,
    Lead,
    LeadStatus,
)
from src.campaign.orchestrator import CampaignOrchestrator
from src.campaign.scheduler import CallScheduler, RateLimitConfig, RetryConfig
from src.integration.crm_client import FakeCRMClient
from src.integration.event_bus import Event, EventBus, EventType


# --- Helpers ------------------------------------------------------------


def _ist(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def _scheduler(
    rate: Optional[RateLimitConfig] = None,
    retry: Optional[RetryConfig] = None,
    dnd: Optional[DNDFilter] = None,
) -> CallScheduler:
    return CallScheduler(
        hours=CallingHoursPolicy(start="00:00", end="23:59", skip_weekday=None),
        dnd_filter=dnd or DNDFilter(InMemoryDNDStore()),
        rate_limit=rate or RateLimitConfig(calls_per_minute=100, max_concurrent_calls=10),
        retry=retry or RetryConfig(max_retry_attempts=3, retry_interval_hours=2),
    )


def _campaign(id: str = "c1") -> Campaign:
    return Campaign(id=id, tenant_id="t_test", name="Test campaign")


def _lead(id: str, phone: str = "+91999") -> Lead:
    return Lead(id=id, tenant_id="t_test", campaign_id="c1", phone_number=phone)


def _result(lead_id: str, disposition: CallDisposition, **kw) -> CallResult:
    now = datetime.utcnow()
    return CallResult(
        session_id=f"sess-{lead_id}",
        tenant_id="t_test",
        campaign_id="c1",
        lead_id=lead_id,
        disposition=disposition,
        duration_ms=kw.get("duration_ms", 5000),
        slots=kw.get("slots", {}),
        interest_level=kw.get("interest_level"),
        started_at=now,
        ended_at=now,
    )


def _record_events(bus: EventBus) -> list[Event]:
    captured: list[Event] = []

    async def cap(e: Event) -> None:
        captured.append(e)

    bus.subscribe("*", cap)
    return captured


# --- Tests --------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_dispatches_all_leads_and_emits_events() -> None:
    bus = EventBus()
    events = _record_events(bus)
    crm = FakeCRMClient()

    async def dispatch(c, lead: Lead) -> CallResult:
        return _result(lead.id, CallDisposition.INTERESTED_CALLBACK,
                       interest_level="hot", slots={"interest_level": "hot"})

    orch = CampaignOrchestrator(scheduler=_scheduler(), dispatch=dispatch, bus=bus, crm=crm)
    leads = [_lead(f"l{i}") for i in range(3)]
    run = await orch.run(_campaign(), leads, max_iterations=20, sleep_fn=_zero_sleep)

    assert run.campaign.calls_attempted == 3
    assert run.campaign.calls_answered == 3
    assert run.campaign.leads_qualified == 3
    assert run.campaign.status is CampaignStatus.COMPLETED

    types = [e.type for e in events]
    assert types.count(EventType.CALL_INITIATED) == 3
    assert types.count(EventType.CALL_COMPLETED) == 3
    assert types.count(EventType.LEAD_QUALIFIED) == 3
    # CRM was updated for every lead.
    assert len(crm.updates) == 3


@pytest.mark.asyncio
async def test_run_handles_busy_with_retry() -> None:
    bus = EventBus()
    crm = FakeCRMClient()
    sched = _scheduler(retry=RetryConfig(max_retry_attempts=3, retry_interval_hours=2))

    call_count = {"l1": 0}

    async def dispatch(c, lead: Lead) -> CallResult:
        call_count[lead.id] += 1
        return _result(lead.id, CallDisposition.BUSY_RETRY)

    orch = CampaignOrchestrator(scheduler=sched, dispatch=dispatch, bus=bus, crm=crm)
    run = await orch.run(_campaign(), [_lead("l1")], max_iterations=5, sleep_fn=_zero_sleep)

    # The lead should be in RETRY state with a future next_retry_at, so on
    # subsequent loop iterations it isn't picked up again.
    lead = run.leads["l1"]
    assert lead.status is LeadStatus.RETRY
    assert lead.retry_count == 1
    assert lead.next_retry_at is not None
    assert call_count["l1"] == 1


@pytest.mark.asyncio
async def test_run_dnd_disposition_updates_crm_and_sets_lead_status() -> None:
    bus = EventBus()
    crm = FakeCRMClient()

    async def dispatch(c, lead: Lead) -> CallResult:
        return _result(lead.id, CallDisposition.DND_REQUESTED)

    orch = CampaignOrchestrator(scheduler=_scheduler(), dispatch=dispatch, bus=bus, crm=crm)
    run = await orch.run(_campaign(), [_lead("l1", "+919999999999")], max_iterations=5, sleep_fn=_zero_sleep)

    assert run.leads["l1"].status is LeadStatus.DND
    assert crm.dnd_requests == ["+919999999999"]


@pytest.mark.asyncio
async def test_run_dnd_preexisting_lead_completes_without_call() -> None:
    """A lead whose number was already in the DND store before it was ever
    dialled is a different situation from one discovered mid-call:
    `_on_call_result` is the only place that normally sets `LeadStatus.DND`,
    and that path requires a call to have actually happened. This lead must
    never be dispatched at all -- yet it still has to leave `remaining`, or
    the campaign sits PENDING forever (skipped every poll tick, never
    reaching COMPLETED). Before the fix this assertion on `campaign.status`
    fails because `remaining` never empties."""
    bus = EventBus()
    crm = FakeCRMClient()
    dnd = DNDFilter(InMemoryDNDStore(["+919999999999"]))
    sched = _scheduler(dnd=dnd)
    dispatched: list[str] = []

    async def dispatch(c, lead: Lead) -> CallResult:
        dispatched.append(lead.id)
        return _result(lead.id, CallDisposition.NOT_INTERESTED)

    orch = CampaignOrchestrator(scheduler=sched, dispatch=dispatch, bus=bus, crm=crm)
    run = await orch.run(
        _campaign(), [_lead("l1", "+919999999999")],
        max_iterations=5, sleep_fn=_zero_sleep,
    )

    assert run.campaign.status is CampaignStatus.COMPLETED
    assert run.leads["l1"].status is LeadStatus.DND
    # The dispatch agent was never invoked -- no call was placed for a
    # number that was already known to be DND.
    assert dispatched == []
    assert run.campaign.calls_attempted == 0
    # Terminal, so it counts as completed -- `_on_call_result` adds any lead
    # reaching COMPLETED/FAILED/DND, and a pre-blocked lead is no less
    # finished than one DND'd by a call. Without this, stats would report
    # "0 of 1 completed" for a campaign that has nothing left to do, which is
    # the same wrong conclusion the stall itself produced.
    assert run.completed_leads == {"l1"}
    assert run.stats["completed"] == 1


@pytest.mark.asyncio
async def test_run_dnd_preexisting_mixed_with_clean_lead() -> None:
    """One lead is pre-blocked, one is clean. The blocked lead must not hold
    up the clean one: the campaign should still complete, the clean lead
    should still get called, and the counters should reflect exactly the
    one real call -- not the blocked lead."""
    bus = EventBus()
    crm = FakeCRMClient()
    dnd = DNDFilter(InMemoryDNDStore(["+919999999999"]))
    sched = _scheduler(dnd=dnd)
    dispatched: list[str] = []

    async def dispatch(c, lead: Lead) -> CallResult:
        dispatched.append(lead.id)
        return _result(lead.id, CallDisposition.NOT_INTERESTED)

    orch = CampaignOrchestrator(scheduler=sched, dispatch=dispatch, bus=bus, crm=crm)
    leads = [_lead("blocked", "+919999999999"), _lead("clean", "+918888888888")]
    run = await orch.run(_campaign(), leads, max_iterations=10, sleep_fn=_zero_sleep)

    assert run.campaign.status is CampaignStatus.COMPLETED
    assert run.leads["blocked"].status is LeadStatus.DND
    assert run.leads["clean"].status is LeadStatus.COMPLETED
    assert dispatched == ["clean"]
    assert run.campaign.calls_attempted == 1


@pytest.mark.asyncio
async def test_run_dnd_disabled_calls_prelisted_lead_normally() -> None:
    """With the filter disabled, a number sitting in the DND store is not
    consulted at all: `DNDFilter.is_blocked` short-circuits to False, and
    `dnd_blocked()` must agree, so the lead is dispatched exactly like any
    other -- the orchestrator's sweep must not reimplement its own opinion
    of "blocked" independent of the filter's `enabled` flag."""
    bus = EventBus()
    crm = FakeCRMClient()
    dnd = DNDFilter(InMemoryDNDStore(["+919999999999"]), enabled=False)
    sched = _scheduler(dnd=dnd)
    dispatched: list[str] = []

    async def dispatch(c, lead: Lead) -> CallResult:
        dispatched.append(lead.id)
        return _result(lead.id, CallDisposition.NOT_INTERESTED)

    orch = CampaignOrchestrator(scheduler=sched, dispatch=dispatch, bus=bus, crm=crm)
    run = await orch.run(
        _campaign(), [_lead("l1", "+919999999999")],
        max_iterations=5, sleep_fn=_zero_sleep,
    )

    assert dispatched == ["l1"]
    assert run.leads["l1"].status is LeadStatus.COMPLETED
    assert run.campaign.calls_attempted == 1


@pytest.mark.asyncio
async def test_run_voicemail_marks_for_retry() -> None:
    bus = EventBus()
    crm = FakeCRMClient()

    async def dispatch(c, lead: Lead) -> CallResult:
        return _result(lead.id, CallDisposition.VOICEMAIL)

    orch = CampaignOrchestrator(scheduler=_scheduler(), dispatch=dispatch, bus=bus, crm=crm)
    run = await orch.run(_campaign(), [_lead("l1")], max_iterations=5, sleep_fn=_zero_sleep)

    # VOICEMAIL doesn't count as 'answered'
    assert run.campaign.calls_answered == 0
    # And it's a retry-class disposition
    assert run.leads["l1"].status is LeadStatus.RETRY


@pytest.mark.asyncio
async def test_run_dispatch_exception_schedules_retry() -> None:
    bus = EventBus()
    crm = FakeCRMClient()

    async def bad_dispatch(c, lead: Lead) -> CallResult:
        raise RuntimeError("provider down")

    orch = CampaignOrchestrator(scheduler=_scheduler(), dispatch=bad_dispatch, bus=bus, crm=crm)
    run = await orch.run(_campaign(), [_lead("l1")], max_iterations=5, sleep_fn=_zero_sleep)
    assert run.leads["l1"].status is LeadStatus.RETRY
    assert run.leads["l1"].retry_count == 1


@pytest.mark.asyncio
async def test_run_concurrency_cap_respected() -> None:
    """At most ``max_concurrent_calls`` dispatches should run in parallel."""
    bus = EventBus()
    crm = FakeCRMClient()
    sched = _scheduler(rate=RateLimitConfig(calls_per_minute=100, max_concurrent_calls=2))

    in_flight = 0
    max_seen = 0

    async def dispatch(c, lead: Lead) -> CallResult:
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        # Yield to event loop so concurrency is observable
        for _ in range(5):
            import asyncio as _a
            await _a.sleep(0)
        in_flight -= 1
        return _result(lead.id, CallDisposition.NOT_INTERESTED)

    orch = CampaignOrchestrator(
        scheduler=sched,
        dispatch=dispatch,
        bus=bus,
        crm=crm,
        max_concurrent=2,
    )
    leads = [_lead(f"l{i}") for i in range(6)]
    await orch.run(_campaign(), leads, max_iterations=20, sleep_fn=_zero_sleep)
    assert max_seen <= 2


@pytest.mark.asyncio
async def test_run_completes_when_all_leads_terminal() -> None:
    bus = EventBus()
    crm = FakeCRMClient()

    async def dispatch(c, lead: Lead) -> CallResult:
        return _result(lead.id, CallDisposition.NOT_INTERESTED)

    orch = CampaignOrchestrator(scheduler=_scheduler(), dispatch=dispatch, bus=bus, crm=crm)
    run = await orch.run(_campaign(), [_lead("l1"), _lead("l2")], max_iterations=20, sleep_fn=_zero_sleep)
    assert run.campaign.status is CampaignStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_stats_reflect_progress() -> None:
    bus = EventBus()
    crm = FakeCRMClient()

    async def dispatch(c, lead: Lead) -> CallResult:
        if lead.id == "l1":
            return _result(lead.id, CallDisposition.INTERESTED_CALLBACK, interest_level="hot")
        return _result(lead.id, CallDisposition.NOT_INTERESTED)

    orch = CampaignOrchestrator(scheduler=_scheduler(), dispatch=dispatch, bus=bus, crm=crm)
    run = await orch.run(_campaign(), [_lead("l1"), _lead("l2")], max_iterations=20, sleep_fn=_zero_sleep)
    s = run.stats
    assert s["total_leads"] == 2
    assert s["calls_attempted"] == 2
    assert s["calls_answered"] == 2
    assert s["leads_qualified"] == 1
    assert s["completed"] == 2


async def _zero_sleep(seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_dispatch_built_from_analyze_call_flows_to_crm() -> None:
    """End-to-end: a dispatch agent that builds its CallResult via
    build_call_result (real outcome analysis) flows correctly through the
    orchestrator -> CRM, with the outcome mapped to the legacy disposition."""
    from src.analysis.call_outcome import build_call_result
    from src.campaign.models import LeadCallOutcome
    from src.interfaces.llm import LLMMessage, LLMResult

    class _FakeLLM:
        async def generate(self, messages, config) -> LLMResult:
            return LLMResult(
                text='{"outcome":"interested","summary":"Lead is in.","notes":"Send link."}',
                finish_reason="stop",
            )

        async def generate_stream(self, messages, config):  # pragma: no cover
            raise NotImplementedError

    bus = EventBus()
    crm = FakeCRMClient()
    now = datetime.utcnow()

    async def dispatch(c, lead: Lead) -> CallResult:
        return await build_call_result(
            session_id=f"sess-{lead.id}", tenant_id="t_test", campaign_id="c1",
            lead_id=lead.id,
            transcript=[LLMMessage(role="user", content="haan link bhejo")],
            slots={"interest_level": "hot"}, telephony_status=None,
            final_action="close_positive", tenant_timezone="Asia/Kolkata",
            now=now, llm=_FakeLLM(), started_at=now, ended_at=now,
            interest_level="hot",
        )

    orch = CampaignOrchestrator(scheduler=_scheduler(), dispatch=dispatch, bus=bus, crm=crm)
    await orch.run(_campaign(), [_lead("l1")], max_iterations=10, sleep_fn=_zero_sleep)

    assert len(crm.updates) == 1
    pushed = crm.updates[0]
    assert pushed.outcome == LeadCallOutcome.INTERESTED
    assert pushed.disposition == CallDisposition.INTERESTED_TRANSFER  # mapped from outcome
    assert pushed.summary == "Lead is in."
