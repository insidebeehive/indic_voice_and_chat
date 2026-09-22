"""Persistence helpers for the call record (``conversations`` table).

Call Lead inserts an ``in_progress`` row keyed by the provider Call SID; at
teardown the bridge looks the row up by that SID and writes the outcome +
duration + cost. Cost is Σ(provider cost/min from the ``provider_costs`` catalog
for the providers actually used) × duration. Kept here so the endpoint and the
bridge teardown share one implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.context import TenantContext
from src.integration.tenant_events import build_envelope, channel_label
from src.models.conversation import Conversation, Turn
from src.models.tenant import ProviderCost
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

# A call counts against the concurrency cap while it is being placed or is live.
ACTIVE_STATUSES = ("in_progress", "answered")


async def insert_call(
    session: AsyncSession,
    *,
    call_id: str,
    tenant: TenantContext,
    provider_call_sid: str,
    channel: str = "voice",
    campaign_id: Optional[str] = None,
    lead_id: Optional[str] = None,
    voice: Optional[str] = None,
    mode: Optional[str] = None,
    agent_type: str = "voicebot",
    extra_event_data: Optional[dict] = None,
) -> Conversation:
    """Insert an ``in_progress`` conversation row snapshotting the config used.

    Shared by Call Lead (telephony) and the browser/webconsole path so both
    record the same per-call config for statistics + billing. ``mode`` overrides
    the tenant default — the browser console can run S2S on a layered-default
    tenant (or vice-versa), and the recorded mode drives the cost calculation.

    ``agent_type`` marks who talked: ``"voicebot"`` (our AI, the default) or
    ``"human"`` (a CRM agent on the browser softphone). A human softphone call is
    forced to ``layered`` mode and snapshots only STT+LLM — the cost actually
    incurred is post-call transcription (STT) + outcome analysis (LLM); there is
    no live TTS or S2S, so those are left unset and excluded from the platform
    total. The outcome itself is produced by the same analyzer AI calls use.
    """
    p = tenant.settings.pipeline
    is_human = agent_type == "human"
    eff_mode = "layered" if is_human else (mode or p.mode)
    realtime_provider = p.realtime.provider if (eff_mode == "s2s" and p.realtime) else None
    tts_provider = None if is_human else p.tts.provider
    v = None if is_human else (voice or p.tts.voice_id or (p.realtime.voice if p.realtime else None))
    row = Conversation(
        id=call_id, tenant_id=tenant.id, campaign_id=campaign_id, lead_id=lead_id,
        agent_type=agent_type, channel=channel, status="in_progress",
        pipeline_config=p.model_dump(), provider_call_sid=provider_call_sid,
        mode=eff_mode, stt_provider=p.stt.provider, llm_provider=p.llm.provider,
        tts_provider=tts_provider, realtime_provider=realtime_provider, voice=v,
        telephony_provider=(p.telephony.provider or None),
    )
    session.add(row)
    await session.commit()
    debug_event(
        log, "call_store insert_call response",
        call_id=call_id, tenant_id=tenant.id, provider_call_sid=provider_call_sid,
        channel=channel, campaign_id=campaign_id, lead_id=lead_id,
        agent_type=agent_type, mode=eff_mode, stt_provider=p.stt.provider,
        llm_provider=p.llm.provider, tts_provider=tts_provider,
        realtime_provider=realtime_provider, voice=v,
        telephony_provider=(p.telephony.provider or None),
    )
    event_data: dict = {"provider_call_sid": provider_call_sid, "mode": eff_mode,
                        "campaign_id": campaign_id, "lead_id": lead_id}
    if extra_event_data:
        event_data.update(extra_event_data)
    await emit_tenant_event(build_envelope(
        event_type="call.initiated", call_id=call_id, tenant_id=tenant.id,
        channel=channel_label(agent_type), data=event_data))
    return row


# --- Outcome persister hook ---------------------------------------------
# The bridges have no DB session of their own; at teardown they hand the
# (call_sid, outcome payload) to this hook, which the app wires in its lifespan
# to open a session and write the outcome + cost. Unset (tests / dev console
# without DB) → teardown is a clean no-op.
_persister: Optional[Callable[[str, dict], Awaitable[None]]] = None


def set_call_outcome_persister(fn: Optional[Callable[[str, dict], Awaitable[None]]]) -> None:
    global _persister
    _persister = fn


async def deliver_to_persister(call_sid: Optional[str], payload: dict) -> None:
    """Hand a finished call's outcome to the persister, if one is wired.

    Never raises — outcome persistence must not break call teardown.
    """
    if _persister is None or not call_sid:
        # Exactly the shape this sweep hunts for: a call outcome computed
        # upstream and handed here to be persisted, dropped with no trace
        # because either nothing was wired (tests / dev console without DB)
        # or the caller never resolved a call_sid at all.
        debug_event(
            log, "call_store deliver_to_persister skipped",
            call_sid=call_sid, persister_wired=(_persister is not None),
            payload_type=payload.get("type") if isinstance(payload, dict) else None,
        )
        return
    try:
        await _persister(call_sid, payload)
        debug_event(
            log, "call_store deliver_to_persister response",
            call_sid=call_sid,
            payload_type=payload.get("type") if isinstance(payload, dict) else None,
        )
    except Exception:  # noqa: BLE001 — teardown must survive a DB hiccup
        log.exception("call outcome persistence failed", extra={"sid": call_sid})


# --- Outbound tenant event notifier hook --------------------------------
# Like the persister, but for the tenant's OUTBOUND events webhook. When a call
# starts (insert_call) or ends with an outcome (record_outcome), call_store hands
# a ready-built envelope to this hook; the app wires it (lifespan) to resolve the
# tenant's events_webhook_url + secret and POST it (signed, with retry). Both the
# softphone (human) and voice-bot paths pass through these functions, so one hook
# covers both. Unset (tests / no webhook configured) → a clean no-op.
_event_notifier: Optional[Callable[[dict], Awaitable[None]]] = None


def set_tenant_event_notifier(fn: Optional[Callable[[dict], Awaitable[None]]]) -> None:
    global _event_notifier
    _event_notifier = fn


async def emit_tenant_event(envelope: dict) -> None:
    """Hand a call-event envelope to the tenant notifier, if wired. Never raises —
    event emission must not break call insert/finalize."""
    if _event_notifier is None:
        debug_event(
            log, "call_store emit_tenant_event skipped",
            event_type=envelope.get("event_type"), notifier_wired=False,
        )
        return
    try:
        await _event_notifier(envelope)
        debug_event(
            log, "call_store emit_tenant_event response",
            event_type=envelope.get("event_type"),
        )
    except Exception:  # noqa: BLE001 - delivery must not break call handling
        log.exception("tenant event emit failed",
                      extra={"event": envelope.get("event_type")})


async def mark_answered(
    session: AsyncSession, provider_call_sid: str, *, tenant_id: Optional[str] = None
) -> Optional[Conversation]:
    """Mark a call answered: flip an in_progress row to ``answered`` and emit a
    ``call.answered`` tenant event. Returns the row, or None if no match. (There
    is no single DB choke point for "answered" across paths, so callers invoke
    this where the call connects.)

    ``tenant_id``: optional tenant scope. Twilio/Exotel/Stringee Call SIDs are
    provider-generated and globally unique, so their callers pass nothing here
    and get the exact same global-by-SID lookup as before. LiveKit room names
    are CRM-chosen and NOT guaranteed unique across tenants, so its caller
    (``livekit_runner.py``) always passes ``tenant_id`` to prevent one tenant's
    call from matching another tenant's row that happens to reuse the same
    room name.
    """
    stmt = select(Conversation).where(Conversation.provider_call_sid == provider_call_sid)
    if tenant_id is not None:
        stmt = stmt.where(Conversation.tenant_id == tenant_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        # Exactly the "turn persisted against a session row that no longer
        # exists" shape this sweep hunts for, just for "answered" instead of a
        # turn: a provider's answer webhook fires and there is no trace, at
        # any level, of why the answered-mark never landed (wrong SID, wrong
        # tenant scope for a LiveKit room-name reuse, or the insert_call that
        # should have preceded it never happened).
        debug_event(
            log, "call_store mark_answered miss",
            provider_call_sid=provider_call_sid, tenant_id=tenant_id,
        )
        return None
    was_in_progress = row.status == "in_progress"
    if was_in_progress:
        row.status = "answered"
        await session.commit()
    debug_event(
        log, "call_store mark_answered response",
        call_id=row.id, provider_call_sid=provider_call_sid, tenant_id=row.tenant_id,
        previous_status=("in_progress" if was_in_progress else row.status),
        status_changed=was_in_progress,
    )
    await emit_tenant_event(build_envelope(
        event_type="call.answered", call_id=row.id, tenant_id=row.tenant_id,
        channel=channel_label(row.agent_type),
        data={"provider_call_sid": row.provider_call_sid}))
    return row


async def count_active_calls(session: AsyncSession, tenant_id: str) -> int:
    """How many of this tenant's calls are currently placing/live."""
    return (await session.execute(
        select(func.count()).select_from(Conversation).where(
            Conversation.tenant_id == tenant_id,
            Conversation.status.in_(ACTIVE_STATUSES),
        )
    )).scalar_one()


def _components_used(
    *, mode: Optional[str], stt_provider: Optional[str], llm_provider: Optional[str],
    tts_provider: Optional[str], realtime_provider: Optional[str],
    stt_model: str = "", llm_model: str = "", tts_model: str = "", realtime_model: str = "",
) -> list[tuple[str, str, str]]:
    """The (kind, provider, model) triples the PLATFORM bills for one call.

    Telephony is intentionally excluded: the tenant brings its own telephony
    provider key, so that cost is theirs (shown separately as tentative, never
    in the platform total).
    """
    triples: list[tuple[str, str, str]] = []
    if mode == "s2s":
        if realtime_provider:
            triples.append(("s2s", realtime_provider, realtime_model or ""))
    else:
        if stt_provider:
            triples.append(("stt", stt_provider, stt_model or ""))
        if llm_provider:
            triples.append(("llm", llm_provider, llm_model or ""))
        if tts_provider:
            triples.append(("tts", tts_provider, tts_model or ""))
    return triples


# A missing ProviderCost row bills the leg at $0.0 -- a reporting gap, not a
# runtime fault, so this only ever warns (never raises). Rate lookups happen
# per call leg, so an unpriced provider/model would otherwise warn on every
# single call; this set makes each distinct (kind, provider, model) combo
# warn once per process, same idiom as _PushFailureWarner in
# src/observability/turn_metrics_push.py (there: one outage, here: one
# missing catalog row, both re-armed only by a process restart -- a catalog
# gap doesn't "recover" the way a push outage does, so there is no
# mark_recovered() counterpart).
_warned_rate_misses: set[tuple[str, str, str]] = set()


async def _rate(session: AsyncSession, kind: str, provider: str, model: str) -> float:
    """Rate for (kind, provider, model); fall back to the provider-level ("") row."""
    row = await session.get(ProviderCost, (kind, provider, model or ""))
    if row is None and model:
        row = await session.get(ProviderCost, (kind, provider, ""))
    if row is None:
        # A missing catalog row bills this leg at $0.0 silently -- the
        # decision that changes the reported cost, with nothing recording
        # that it was made by omission rather than by an actual $0 rate.
        key = (kind, provider, model)
        if key not in _warned_rate_misses:
            _warned_rate_misses.add(key)
            log.warning(
                "no ProviderCost row for kind=%s provider=%s model=%s (nor its "
                "provider-level fallback) -- billing this leg at $0.0 until a "
                "row is added (further warnings for this combination "
                "suppressed for this process)",
                kind, provider, model,
            )
        return 0.0
    return row.cost_per_min


async def telephony_tentative_cost(
    session: AsyncSession, provider: Optional[str], duration_ms: Optional[int]
) -> float:
    """Telephony cost for a call — TENTATIVE only (the tenant pays its own
    telephony provider). Never part of the platform-billed total."""
    if not provider or not duration_ms or duration_ms <= 0:
        return 0.0
    rate = await _rate(session, "telephony", provider, "")
    return round(rate * (duration_ms / 60_000.0), 6)


async def compute_call_cost(
    session: AsyncSession,
    *,
    mode: Optional[str],
    stt_provider: Optional[str] = None,
    llm_provider: Optional[str] = None,
    tts_provider: Optional[str] = None,
    realtime_provider: Optional[str] = None,
    telephony_provider: Optional[str] = None,
    stt_model: str = "", llm_model: str = "", tts_model: str = "", realtime_model: str = "",
    duration_ms: Optional[int],
) -> float:
    """Platform-billed cost = Σ(cost/min for STT/LLM/TTS or S2S) × duration.

    Excludes telephony (the tenant's own key). ``telephony_provider`` is accepted
    for signature compatibility but not billed.
    """
    if not duration_ms or duration_ms <= 0:
        return 0.0
    triples = _components_used(
        mode=mode, stt_provider=stt_provider, llm_provider=llm_provider,
        tts_provider=tts_provider, realtime_provider=realtime_provider,
        stt_model=stt_model, llm_model=llm_model, tts_model=tts_model,
        realtime_model=realtime_model,
    )
    minutes = duration_ms / 60_000.0
    per_min = 0.0
    for kind, provider, model in triples:
        per_min += await _rate(session, kind, provider, model)
    return round(per_min * minutes, 6)


async def record_outcome(
    session: AsyncSession,
    provider_call_sid: str,
    *,
    status: str = "ended",
    outcome: Optional[str] = None,
    summary: Optional[str] = None,
    notes: Optional[str] = None,
    callback_at: Optional[datetime] = None,
    duration_ms: Optional[int] = None,
    ended_at: Optional[datetime] = None,
    turns: Optional[list] = None,
    tenant_id: Optional[str] = None,
) -> Optional[Conversation]:
    """Find the call row by provider Call SID and write its outcome + cost.

    Cost is computed from the providers recorded on the row + ``duration_ms``.
    If ``turns`` is given (non-empty), the transcript is written via
    ``save_turns`` right after the row is resolved — before any of the outcome/
    cost fields below are touched, so a later failure in this function can't
    cost us the transcript — and ``row.total_turns`` is set to the count saved.
    Returns the updated row, or None if no row matches the SID.

    ``tenant_id``: optional tenant scope — see ``mark_answered``'s docstring for
    why this matters (LiveKit room-name collisions across tenants). ``None``
    (every existing Twilio/Exotel/Stringee caller) preserves the exact global
    lookup behavior from before this parameter existed.
    """
    stmt = select(Conversation).where(Conversation.provider_call_sid == provider_call_sid)
    if tenant_id is not None:
        stmt = stmt.where(Conversation.tenant_id == tenant_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        log.warning("no conversation for call sid", extra={"sid": provider_call_sid})
        return None

    debug_event(
        log, "call_store record_outcome request",
        call_id=row.id, provider_call_sid=provider_call_sid, tenant_id=tenant_id,
        status=status, outcome=outcome, summary=summary, notes=notes,
        callback_at=callback_at.isoformat() if callback_at else None,
        duration_ms=duration_ms, turns_supplied=(len(turns) if turns else 0),
    )
    if turns:
        row.total_turns = await save_turns(session, conversation_id=row.id, turns=turns)

    row.status = status
    if outcome is not None:
        row.outcome = outcome
    if summary is not None:
        row.summary = summary
    if notes is not None:
        row.notes = notes
    if callback_at is not None:
        row.callback_at = callback_at
    row.ended_at = ended_at or datetime.utcnow()
    if duration_ms is not None:
        row.duration_ms = duration_ms
    elif row.duration_ms is None and row.started_at is not None:
        # Derive call duration from when the row was created (call placed).
        row.duration_ms = max(0, int((row.ended_at - row.started_at).total_seconds() * 1000))
    # Models live in the per-call pipeline_config snapshot (provider columns are
    # on the row; models are not, so read them from the config).
    pc = row.pipeline_config or {}
    row.cost = await compute_call_cost(
        session,
        mode=row.mode,
        stt_provider=row.stt_provider,
        llm_provider=row.llm_provider,
        tts_provider=row.tts_provider,
        realtime_provider=row.realtime_provider,
        telephony_provider=row.telephony_provider,
        stt_model=(pc.get("stt") or {}).get("model") or "",
        llm_model=(pc.get("llm") or {}).get("model") or "",
        tts_model=(pc.get("tts") or {}).get("model") or "",
        realtime_model=(pc.get("realtime") or {}).get("model") or "",
        duration_ms=row.duration_ms,
    )
    await session.commit()
    debug_event(
        log, "call_store record_outcome response",
        call_id=row.id, status=row.status, outcome=row.outcome, cost=row.cost,
        duration_ms=row.duration_ms, total_turns=row.total_turns,
    )
    # Terminal event: the LLM outcome IS the tenant's end-call signal. Fires for
    # both voice-bot (via the persister) and softphone (manual_call → record_outcome).
    await emit_tenant_event(build_envelope(
        event_type="call.completed", call_id=row.id, tenant_id=row.tenant_id,
        channel=channel_label(row.agent_type),
        data={"outcome": outcome, "summary": summary, "notes": notes,
              "callback_datetime": callback_at.isoformat() if callback_at else None,
              "status": row.status, "duration_ms": row.duration_ms,
              "provider_call_sid": row.provider_call_sid}))
    return row


# A call that never reaches finalization (record_outcome) — e.g. a telephony call
# whose recording webhook never fires, an abandoned/diagnostic hit, or a
# connection that dies before the finalize step — would otherwise linger in an
# active status forever. A periodic sweep closes them.
STALE_CALL_MINUTES = 30


async def reap_stale_calls(
    session: AsyncSession, *, older_than_minutes: int = STALE_CALL_MINUTES
) -> int:
    """Auto-close conversation rows stuck in an active status (in_progress /
    answered) longer than ``older_than_minutes`` — their finalization never
    fired. Sets ``status='ended'`` + ``ended_at`` + a note. Returns the count
    closed. Safe to run repeatedly (already-ended rows are untouched)."""
    cutoff = datetime.utcnow() - timedelta(minutes=older_than_minutes)
    rows = (await session.execute(
        select(Conversation).where(
            Conversation.status.in_(ACTIVE_STATUSES),
            Conversation.started_at < cutoff,
        )
    )).scalars().all()
    note = f"auto-closed: no finalization within {older_than_minutes}m"
    for row in rows:
        row.status = "ended"
        row.ended_at = row.ended_at or datetime.utcnow()
        row.notes = f"{row.notes}\n{note}" if row.notes else note
    if rows:
        # A periodic sweep silently rewriting call status with zero logging
        # at any level -- this is exactly a "state transition" the docs call
        # out, and the only place these calls' true end state (never
        # finalized) is recorded.
        if log.isEnabledFor(logging.DEBUG):
            debug_event(
                log, "call_store reap_stale_calls response",
                older_than_minutes=older_than_minutes, cutoff=cutoff.isoformat(),
                closed_call_ids=[r.id for r in rows], closed_count=len(rows),
            )
        await session.commit()
    return len(rows)


async def save_turns(
    session: AsyncSession,
    *,
    conversation_id: str,
    turns: list,
) -> int:
    """Persist the in-memory transcript (list of LLMMessage) to the turns table.

    Skips the system-prompt turn (role='system') — it's reconstructable from
    the tenant config and would bloat the table. Returns the number of rows
    written. Callers must only invoke this once the conversation row is known
    to exist — ``Turn.conversation_id`` is a NOT NULL FK, so on Postgres this
    raises rather than orphaning rows if it doesn't. In practice this is only
    ever called after ``record_outcome`` has already resolved ``row`` by
    ``provider_call_sid``, so that precondition always holds.
    """
    num = 0
    skipped_system = 0
    skipped_empty = 0
    for msg in turns:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role == "system":
            skipped_system += 1
            continue
        if not content:
            # The exact shape this sweep hunts for: a non-system turn (e.g. a
            # human agent's reply, or an assistant turn) silently dropped from
            # the transcript because its content came through empty, with the
            # in-memory transcript looking otherwise complete to every caller.
            skipped_empty += 1
            continue
        num += 1
        session.add(Turn(
            conversation_id=conversation_id,
            turn_number=num,
            role=role,
            content=content if isinstance(content, str) else str(content),
        ))
    if skipped_empty:
        debug_event(
            log, "call_store save_turns empty_content_dropped",
            conversation_id=conversation_id, skipped_count=skipped_empty,
            turns_supplied=len(turns),
        )
    debug_event(
        log, "call_store save_turns response",
        conversation_id=conversation_id, turns_saved=num,
        turns_supplied=len(turns), skipped_system=skipped_system,
        skipped_empty_content=skipped_empty,
    )
    if num:
        await session.commit()
        log.info("saved %d turns for conversation %s", num, conversation_id)
    return num
