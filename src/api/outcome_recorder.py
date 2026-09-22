"""Shared call-outcome recording for telephony bridges.

Telephony has no live UI (unlike the browser dev console), so on call-end we
analyze the finished call and *log* the outcome — the hook for DB persistence
later. Mixed into the Twilio/Exotel bridges so the logic lives in one place.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.analysis.call_outcome import analyze_agent_call
from src.interfaces.llm import ILLMProvider
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


class OutcomeRecorderMixin:
    """Adds ``_record_outcome()`` to a bridge.

    The host bridge must set (in its ``__init__``): ``self._agent``,
    ``self._llm``, ``self._tenant_timezone``, ``self._last_action``, and
    ``self._outcome_recorded = False``.
    """

    # Declared for type-checkers; real values are set by the host's __init__.
    _agent: object
    _llm: Optional[ILLMProvider]
    _tenant_timezone: str
    _last_action: Optional[str]
    _outcome_recorded: bool
    # Re-entrancy guard, separate from `_outcome_recorded`. `_outcome_recorded`
    # is only set once analysis+persistence has actually succeeded (or the
    # analysis was a deterministic no-op — see below), so a transient failure
    # leaves it False and a later teardown call can retry. That alone would
    # let two teardown calls racing at the same time (e.g. a duplicate
    # provider webhook) both pass the `_outcome_recorded` check before either
    # finishes and both run analyze_agent_call + persist. This flag closes
    # that gap: it flips True synchronously (no `await` in between) before the
    # first `await` in the try block below, so a second call arriving before
    # the first completes sees it and no-ops instead of racing it. It always
    # resets to False in the `finally` so a genuine retry after failure is not
    # blocked by it.
    _outcome_recording: bool = False

    async def _record_outcome(self) -> None:
        """Analyze the finished call and log its outcome. Idempotent; no-op
        without an LLM. Never raises — analysis must not break teardown."""
        if self._outcome_recorded or self._llm is None:
            # Two very different reasons collapse into the same silent no-op:
            # already recorded (fine, idempotent teardown) vs. no LLM
            # configured at all, in which case this call's outcome is NEVER
            # analyzed or persisted, for its entire lifetime, with nothing at
            # any level saying so.
            debug_event(
                log, "outcome_recorder record_outcome skipped",
                already_recorded=self._outcome_recorded,
                llm_configured=(self._llm is not None),
            )
            return
        if self._outcome_recording:
            # Another teardown for this same bridge is already mid-analysis
            # (see the `_outcome_recording` class docstring above) — do not
            # run a second analyze_agent_call + persist concurrently with it.
            debug_event(
                log, "outcome_recorder record_outcome skipped",
                already_recorded=False, llm_configured=True, already_in_flight=True,
            )
            return
        self._outcome_recording = True
        try:
            try:
                analysis = await analyze_agent_call(
                    self._agent,
                    llm=self._llm,
                    tenant_timezone=self._tenant_timezone,
                    final_action=self._last_action,
                    now=datetime.now(timezone.utc),
                )
            except Exception:  # noqa: BLE001 - never let analysis break teardown
                # `_outcome_recorded` is deliberately NOT set here: this is a
                # transient-failure path, and leaving it False lets a later
                # teardown call (retry/duplicate) attempt the analysis again
                # instead of the first failure being permanent for this call.
                log.exception("call outcome analysis failed")
                return
            if analysis is None:
                # A deterministic no-op (no agent to analyze), not a failure —
                # retrying would just get None again, so mark it recorded.
                debug_event(log, "outcome_recorder record_outcome analysis_empty")
                self._outcome_recorded = True
                return
            cb = analysis.callback_datetime
            log.info(
                "call outcome",
                extra={
                    "outcome": analysis.outcome.value,
                    "source": analysis.analysis_source,
                    "summary": analysis.summary[:200],
                    "callback": cb.isoformat() if cb else None,
                },
            )
            # Persist to the conversations row (keyed by the provider Call SID), if
            # the host knows its SID and a persister is wired. No-op otherwise.
            from src.api import call_store
            call_sid = getattr(self, "_provider_call_sid", None) or getattr(self, "_call_sid", None)
            debug_event(
                log, "outcome_recorder record_outcome persist_dispatch",
                call_sid=call_sid, has_call_sid=(call_sid is not None),
                outcome=analysis.outcome.value,
            )
            await call_store.deliver_to_persister(call_sid, {
                "type": "outcome", "outcome": analysis.outcome.value,
                "summary": analysis.summary, "notes": analysis.notes,
                "callback_datetime": cb.isoformat() if cb else None,
                "source": analysis.analysis_source,
            })
            # Only set once analysis AND persistence have both succeeded.
            self._outcome_recorded = True
        finally:
            self._outcome_recording = False
