"""Tests for OutcomeRecorderMixin._record_outcome (src/api/outcome_recorder.py).

Covers the fix for a permanent-failure bug: `_outcome_recorded` used to be
set True BEFORE `analyze_agent_call` was attempted, so a single transient
failure (LLM hiccup, provider timeout) made that call's outcome unrecordable
forever -- a retried or duplicate teardown could never try again. The fix
moves the flag to after analysis+persistence both succeed, with a separate
re-entrancy guard (`_outcome_recording`) so two teardown calls racing at the
same time still can't both run the analysis and persist twice.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from src.api import call_store, outcome_recorder
from src.api.outcome_recorder import OutcomeRecorderMixin
from src.campaign.models import CallAnalysis, LeadCallOutcome


@pytest.fixture(autouse=True)
def _reset_persister():
    """`call_store`'s persister is process-global state; make sure a fake
    registered by one test can't leak into the next."""
    yield
    call_store.set_call_outcome_persister(None)


class _FakeBridge(OutcomeRecorderMixin):
    """Minimal host: only the attributes OutcomeRecorderMixin documents as
    required (see its class docstring)."""

    def __init__(self, call_sid: str = "call-sid-1") -> None:
        self._agent = object()  # non-None is all analyze_agent_call needs to not short-circuit
        self._llm = object()    # non-None: a real ILLMProvider is never called directly here,
                                 # analyze_agent_call itself is monkeypatched per test
        self._tenant_timezone = "Asia/Kolkata"
        self._last_action = "some_action"
        self._outcome_recorded = False
        self._call_sid = call_sid


def _analysis(summary: str = "ok") -> CallAnalysis:
    return CallAnalysis(
        outcome=LeadCallOutcome.INTERESTED, summary=summary, notes="",
        callback_datetime=None, callback_phrase=None, analysis_source="llm",
    )


@pytest.mark.asyncio
async def test_transient_failure_then_success_persists_on_the_retry(monkeypatch):
    """A teardown whose analyze_agent_call raises must not be the last word
    for that call: a SECOND teardown call (retry/duplicate) after the
    transient condition clears has to actually run the analysis and persist
    it -- proving the flag is no longer poisoned by the first failure."""
    bridge = _FakeBridge(call_sid="call-retry")
    attempts = {"n": 0}

    async def _flaky_analyze(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient LLM failure")
        return _analysis("recovered")

    monkeypatch.setattr(outcome_recorder, "analyze_agent_call", _flaky_analyze)

    persisted = []

    async def _fake_persister(call_sid, payload):
        persisted.append((call_sid, payload))

    call_store.set_call_outcome_persister(_fake_persister)

    # First teardown: analysis raises. Before the fix this alone would have
    # set _outcome_recorded=True, permanently blocking any future attempt.
    await bridge._record_outcome()
    assert attempts["n"] == 1
    assert bridge._outcome_recorded is False, (
        "a transient analysis failure must NOT set the idempotency flag -- "
        "otherwise no retry can ever happen for this call"
    )
    assert persisted == []

    # Second teardown call for the same bridge (retry/duplicate): analysis
    # now succeeds and must be persisted.
    await bridge._record_outcome()
    assert attempts["n"] == 2
    assert bridge._outcome_recorded is True
    assert len(persisted) == 1
    call_sid, payload = persisted[0]
    assert call_sid == "call-retry"
    assert payload["outcome"] == LeadCallOutcome.INTERESTED.value
    assert payload["summary"] == "recovered"

    # A third call must now be a true no-op: no further analysis, no double
    # persist -- the idempotency guard still works once genuinely recorded.
    await bridge._record_outcome()
    assert attempts["n"] == 2
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_concurrent_double_teardown_persists_exactly_once(monkeypatch):
    """Two teardown calls firing at (almost) the same time for the same
    bridge -- e.g. a duplicate provider webhook racing the normal hangup path
    -- must still only run analyze_agent_call and persist the outcome once.
    Moving `_outcome_recorded` to after success reopens exactly this race if
    nothing else guards it, so this pins the `_outcome_recording` re-entrancy
    guard added alongside the fix."""
    import asyncio

    bridge = _FakeBridge(call_sid="call-concurrent")
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = {"n": 0}

    async def _slow_analyze(*args, **kwargs):
        call_count["n"] += 1
        started.set()
        # Yield control back to the event loop so the second _record_outcome()
        # call gets a chance to run while this one is still "in analysis" --
        # this is what a real await on an LLM/provider call would do.
        await release.wait()
        return _analysis("concurrent")

    monkeypatch.setattr(outcome_recorder, "analyze_agent_call", _slow_analyze)

    persisted = []

    async def _fake_persister(call_sid, payload):
        persisted.append((call_sid, payload))

    call_store.set_call_outcome_persister(_fake_persister)

    async def _second_call_after_first_started():
        await started.wait()
        # The first call is now blocked mid-analysis; a second teardown
        # arriving now must see the in-flight guard and no-op immediately
        # rather than starting its own analyze_agent_call.
        await bridge._record_outcome()

    first = asyncio.ensure_future(bridge._record_outcome())
    second = asyncio.ensure_future(_second_call_after_first_started())

    await started.wait()
    await asyncio.sleep(0)  # let `second` reach its own await and block
    release.set()
    await asyncio.gather(first, second)

    assert call_count["n"] == 1, "analyze_agent_call must run exactly once across both teardowns"
    assert len(persisted) == 1, "the outcome must be persisted exactly once, not twice"
    assert bridge._outcome_recorded is True


@pytest.mark.asyncio
async def test_no_llm_never_marks_recorded_and_never_persists():
    """Non-regression: a bridge with no LLM configured stays a permanent,
    clearly-distinguishable no-op (see the debug_event two lines into
    _record_outcome) -- it must not be conflated with 'recorded'."""
    bridge = _FakeBridge()
    bridge._llm = None
    await bridge._record_outcome()
    assert bridge._outcome_recorded is False
