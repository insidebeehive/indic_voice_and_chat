"""Conversations API — list calls and re-run outcome analysis on demand.

Two endpoints (both require tenant Bearer auth):

    GET  /api/v1/conversations
        List recent calls for the tenant with outcome, summary, and notes.
        CRM can poll this to reconcile calls where the webhook was missed.

    POST /api/v1/conversations/{call_id}/reanalyze
        Re-run outcome analysis from the stored transcript and update the row.
        CRM calls this (using the call_id from the call.completed webhook) when
        the original outcome is missing or needs to be refreshed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.models.conversation import Conversation, Turn

log = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("")
async def list_conversations(
    limit: int = 20,
    offset: int = 0,
    tenant: TenantContext = Depends(current_tenant),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return recent calls for the tenant, newest first.

    Each item includes ``call_id``, ``outcome``, ``summary``, ``notes``,
    ``status``, ``duration_ms``, ``started_at``, and ``ended_at``.
    Use ``?limit=`` / ``?offset=`` for pagination (default 20 per page).
    """
    limit = max(1, min(limit, 100))
    rows = (await db.execute(
        select(Conversation)
        .where(Conversation.tenant_id == tenant.id)
        .order_by(Conversation.started_at.desc())
        .limit(limit).offset(offset)
    )).scalars().all()

    return {
        "conversations": [_serialize(r) for r in rows],
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# Re-analyze
# ---------------------------------------------------------------------------

@router.post("/{call_id}/reanalyze")
async def reanalyze_conversation(
    call_id: str,
    request: Request,
    tenant: TenantContext = Depends(current_tenant),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Re-run outcome analysis for a finished call using its stored transcript.

    Use this when the original outcome is NULL (e.g. the analysis LLM timed out
    during the call) or when you want a fresh read of the conversation. Updates
    ``outcome``, ``summary``, and ``notes`` in-place and returns the new values.

    ``call_id`` comes from the ``call_id`` field of the ``call.completed``
    webhook event that was delivered when the call ended.
    """
    # --- Load conversation (must belong to this tenant) ---
    row = (await db.execute(
        select(Conversation).where(
            Conversation.id == call_id,
            Conversation.tenant_id == tenant.id,
        )
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    # --- Load stored turns ---
    turn_rows = (await db.execute(
        select(Turn)
        .where(Turn.conversation_id == call_id)
        .order_by(Turn.turn_number)
    )).scalars().all()

    if not turn_rows:
        raise HTTPException(
            status_code=422,
            detail="no transcript stored for this call — cannot re-analyze",
        )

    # Build LLMMessage list for analyze_call.
    from src.interfaces.llm import LLMMessage
    transcript = [LLMMessage(role=t.role, content=t.content) for t in turn_rows]

    # --- Get LLM provider ---
    providers = getattr(request.app.state, "providers", None)
    if providers is None:
        raise HTTPException(status_code=503, detail="provider registry not available")
    try:
        llm = providers.get_llm(tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"LLM provider unavailable: {e}")

    # --- Re-run analysis ---
    from src.analysis.call_outcome import analyze_call
    try:
        analysis = await analyze_call(
            transcript=transcript,
            slots=row.slots_data or {},
            telephony_status=None,
            final_action=None,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
            now=datetime.now(UTC),
            llm=llm,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("reanalyze failed for %s", call_id)
        raise HTTPException(status_code=502, detail=f"analysis failed: {e}")

    # --- Update row ---
    row.outcome = analysis.outcome.value
    row.summary = analysis.summary
    row.notes = analysis.notes
    if analysis.callback_datetime:
        row.callback_at = analysis.callback_datetime.replace(tzinfo=None)
    await db.commit()

    log.info("reanalyzed conversation", extra={
        "call_id": call_id, "tenant": tenant.slug,
        "outcome": row.outcome, "source": analysis.analysis_source,
    })
    return {
        "call_id": call_id,
        "outcome": row.outcome,
        "summary": row.summary,
        "notes": row.notes,
        "analysis_source": analysis.analysis_source,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(row: Conversation) -> dict:
    return {
        "call_id": row.id,
        "channel": row.channel,
        "mode": row.mode,
        "status": row.status,
        "outcome": row.outcome,
        "summary": row.summary,
        "notes": row.notes,
        "callback_at": row.callback_at.isoformat() if row.callback_at else None,
        "duration_ms": row.duration_ms,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
        "provider_call_sid": row.provider_call_sid,
    }
