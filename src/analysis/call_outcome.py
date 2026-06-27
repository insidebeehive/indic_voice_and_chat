"""Post-call lead outcome analysis.

Transport-agnostic: given a transcript (+ slots, optional telephony status),
returns a CallAnalysis. Conversational outcomes come from a single LLM JSON
pass; unreachable outcomes short-circuit from the telephony status (Task 4).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.campaign.models import (
    CONVERSATIONAL_OUTCOMES,
    CallAnalysis,
    CallResult,
    LeadCallOutcome,
    disposition_from_outcome,
    outcome_from_telephony,
)
from src.dialogue.response_parser import _extract_json
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage

log = logging.getLogger(__name__)

ANALYSIS_TIMEOUT_S = 15.0

_ACTION_FALLBACK = {
    "close_positive": LeadCallOutcome.INTERESTED,
    "schedule_callback": LeadCallOutcome.CALLBACK_REQUESTED,
    "transfer": LeadCallOutcome.ESCALATED,
    "close_negative": LeadCallOutcome.NOT_INTERESTED,
}

_TELEPHONY_SUMMARY = {
    LeadCallOutcome.NO_ANSWER: "No answer.",
    LeadCallOutcome.BUSY: "Line was busy.",
    LeadCallOutcome.CALL_FAILED: "Call failed to connect.",
    LeadCallOutcome.VOICEMAIL: "Reached voicemail.",
}

_SYSTEM_PROMPT = """You analyze a finished sales/outreach phone call and classify its outcome.

Return ONLY a JSON object with these keys:
- "outcome": one of ["interested", "callback_requested", "not_interested", "refused", "escalated", "angry_hostile"]
- "summary": 2-3 sentence recap of the call, in ENGLISH
- "notes": actionable observations (objections, preferences, next steps), in ENGLISH — at most 2 short sentences
- "callback_datetime": ISO-8601 datetime string in the given timezone if the lead asked to be called back at a resolvable time, else null
- "callback_phrase": the lead's raw words about timing (e.g. "kal shaam 5 baje"), else null

Outcome guidance:
- interested: lead is positively engaged / wants to proceed.
- callback_requested: lead asked to be contacted again later.
- not_interested: politely declines.
- refused: refuses to engage / wants to be left alone / do-not-call.
- escalated: needs/requests a human agent or transfer.
- angry_hostile: abusive, threatening, or very hostile.

Resolve relative times ("kal", "Monday", "do din baad") against NOW in the given TIMEZONE.
Write summary and notes in English even though the call is in Hindi/Hinglish.
Keep the whole reply brief and well under 120 words so the JSON always completes."""


def _render_transcript(transcript: list[LLMMessage]) -> str:
    lines = []
    for m in transcript:
        if m.role == "system":
            continue
        who = "Agent" if m.role == "assistant" else "Lead"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def _resolve_callback(value: Any, tz: ZoneInfo) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def _zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("unknown tenant timezone %r; defaulting to Asia/Kolkata", tz_name)
        return ZoneInfo("Asia/Kolkata")


def _fallback(final_action: Optional[str], note: str) -> CallAnalysis:
    outcome = _ACTION_FALLBACK.get(final_action or "", LeadCallOutcome.NOT_INTERESTED)
    # summary is intentionally empty on fallback; the reason lives in notes
    return CallAnalysis(
        outcome=outcome,
        summary="",
        notes=f"(auto-derived; {note})",
        analysis_source="fallback",
    )


async def analyze_call(
    *,
    transcript: list[LLMMessage],
    slots: dict[str, Any],
    telephony_status: Optional[str],
    final_action: Optional[str],
    tenant_timezone: str,
    now: datetime,
    llm: ILLMProvider,
) -> CallAnalysis:
    """Classify a finished call. Never raises — failures fall back."""
    tz = _zone(tenant_timezone)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    now_local = now.astimezone(tz)

    unreachable = outcome_from_telephony(telephony_status)
    if unreachable is not None:
        return CallAnalysis(
            outcome=unreachable,
            summary=_TELEPHONY_SUMMARY.get(unreachable, ""),
            analysis_source="telephony",
        )

    # If the lead never spoke (empty transcript, or agent greeted but got silence),
    # skip the LLM — it would pick "refused" on a one-sided transcript, which is
    # semantically wrong. Return NO_ANSWER instead.
    has_lead_content = any(
        (m.content or "").strip()
        for m in transcript
        if m.role not in ("system", "assistant")
    )
    if not has_lead_content:
        return CallAnalysis(
            outcome=LeadCallOutcome.NO_ANSWER,
            summary="Call ended with no response from the lead.",
            analysis_source="fallback",
        )

    user_msg = (
        f"NOW: {now_local.isoformat()}\nTIMEZONE: {tenant_timezone}\n"
        f"COLLECTED DATA: {json.dumps(slots, ensure_ascii=False)}\n\n"
        f"TRANSCRIPT:\n{_render_transcript(transcript)}"
    )
    messages = [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_msg),
    ]
    # Headroom so the summary+notes envelope always completes as valid JSON.
    # 1024 was observed truncating mid-"notes" (an unterminated string -> parse
    # failure -> wrong fallback outcome); the prompt now also caps output length.
    cfg = LLMConfig(temperature=0.2, max_tokens=1536, response_format="json")

    try:
        result = await asyncio.wait_for(
            llm.generate(messages, cfg), timeout=ANALYSIS_TIMEOUT_S
        )
        # Tolerant parse: Gemini intermittently wraps JSON in ```json fences or
        # adds prose even in JSON mode. Reuse the main turn parser's extractor.
        obj, perr = _extract_json(result.text)
        if obj is None:
            raise ValueError(f"no JSON in analysis response: {perr}")
        outcome = LeadCallOutcome(obj["outcome"])
        if outcome not in CONVERSATIONAL_OUTCOMES:
            raise ValueError(f"non-conversational outcome from LLM: {outcome}")
    except Exception as exc:  # noqa: BLE001 - analysis must never crash the caller
        log.warning("call analysis LLM pass failed: %s", exc)
        return _fallback(final_action, f"analysis failed: {type(exc).__name__}")

    return CallAnalysis(
        outcome=outcome,
        summary=str(obj.get("summary") or ""),
        notes=str(obj.get("notes") or ""),
        callback_datetime=_resolve_callback(obj.get("callback_datetime"), tz),
        callback_phrase=(obj.get("callback_phrase") or None),
        analysis_source="llm",
    )


async def analyze_agent_call(
    agent: Any,
    *,
    llm: Optional[ILLMProvider],
    tenant_timezone: str,
    final_action: Optional[str],
    now: datetime,
    telephony_status: Optional[str] = None,
) -> Optional[CallAnalysis]:
    """Run outcome analysis for a finished call given the live agent.

    Pulls the transcript and collected slots off the agent's session, then
    delegates to ``analyze_call``. Returns ``None`` when there's no LLM or
    agent to analyze with (so callers can no-op). Never raises — ``analyze_call``
    handles its own failures. Defensive about the agent shape so it works with
    any bridge's agent (browser/telephony).
    """
    if llm is None or agent is None:
        return None
    session = getattr(agent, "session", None)
    turns = getattr(session, "turns", []) if session is not None else []
    transcript = [m for m in turns if isinstance(m, LLMMessage)]
    slots_obj = getattr(agent, "slots", None)
    slots = getattr(slots_obj, "values", {}) if slots_obj is not None else {}
    return await analyze_call(
        transcript=transcript,
        slots=slots,
        telephony_status=telephony_status,
        final_action=final_action,
        tenant_timezone=tenant_timezone,
        now=now,
        llm=llm,
    )


async def build_call_result(
    *,
    session_id: str,
    tenant_id: str,
    campaign_id: str,
    lead_id: str,
    transcript: list[LLMMessage],
    slots: dict[str, Any],
    telephony_status: Optional[str],
    final_action: Optional[str],
    tenant_timezone: str,
    now: datetime,
    llm: ILLMProvider,
    started_at: datetime,
    ended_at: datetime,
    duration_ms: int = 0,
    total_turns: int = 0,
    sentiment_history: Optional[list[str]] = None,
    interest_level: Optional[str] = None,
) -> CallResult:
    """Turn a finished call into a fully-populated ``CallResult``.

    This is the 'build CallResult' step a campaign dispatch agent runs after a
    call completes: it analyzes the call (outcome + summary + notes + callback)
    and maps the outcome to the legacy ``disposition`` the orchestrator/CRM
    consume.

    ``analyze_call`` never raises (it falls back internally), so in practice this
    builder doesn't either; the orchestrator also wraps each dispatch in
    try/except (a failed dispatch → retry), so a stray error won't lose the lead.
    """
    analysis = await analyze_call(
        transcript=transcript,
        slots=slots,
        telephony_status=telephony_status,
        final_action=final_action,
        tenant_timezone=tenant_timezone,
        now=now,
        llm=llm,
    )
    return CallResult(
        session_id=session_id,
        tenant_id=tenant_id,
        campaign_id=campaign_id,
        lead_id=lead_id,
        disposition=disposition_from_outcome(analysis.outcome),
        outcome=analysis.outcome,
        summary=analysis.summary,
        notes=analysis.notes,
        callback_datetime=analysis.callback_datetime,
        interest_level=interest_level,
        slots=slots,
        duration_ms=duration_ms,
        total_turns=total_turns,
        sentiment_history=sentiment_history or [],
        started_at=started_at,
        ended_at=ended_at,
    )
