"""Call Lead (async) + call status.

- ``POST /api/v1/campaigns/{id}/calls`` — place one outbound call for a lead.
  Async: it returns a ``call_id`` immediately; the outcome lands later (the
  bridge writes it at teardown). Guards: the campaign must be ``active`` and the
  tenant must be under its ``max_concurrent_calls`` cap (else 429). On success a
  ``conversations`` row is inserted (``in_progress``, keyed by the provider Call
  SID) recording the config used — mode, stt/llm/tts/realtime providers, voice,
  telephony provider — for statistics + cost.
- ``GET  /api/v1/calls/{call_id}`` — poll the call's status/outcome/cost.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.answer_paths import ANSWER_PATHS
from src.api.call_store import count_active_calls, insert_call
from src.api.deps import get_db_session
from src.config_tenant import platform_webhook_base_url
from src.auth import TenantContext, current_tenant
from src.interfaces.telephony import CallConfig
from src.models.campaign import Campaign as DbCampaign
from src.models.conversation import Conversation
from src.providers import get_telephony_provider

log = logging.getLogger(__name__)
router = APIRouter(tags=["calls"])


# --- Schemas ------------------------------------------------------------


class CallLeadRequest(BaseModel):
    to_number: str = Field(min_length=1)
    from_number: str | None = None
    voice: str | None = None
    lead_id: str | None = None


class CallLeadResponse(BaseModel):
    call_id: str
    status: str
    provider_call_sid: str


class CallStatusResponse(BaseModel):
    call_id: str
    status: str
    outcome: str | None = None
    summary: str | None = None
    notes: str | None = None
    callback_at: str | None = None
    cost: float | None = None
    duration_ms: int | None = None


class SummarizeOutcomeResponse(BaseModel):
    call_id: str
    outcome: str
    summary: str
    notes: str | None = None
    callback_at: str | None = None


# --- Routes -------------------------------------------------------------


@router.post("/campaigns/{campaign_id}/calls", response_model=CallLeadResponse, status_code=202)
async def call_lead(
    campaign_id: str,
    req: CallLeadRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CallLeadResponse:
    campaign = await session.get(DbCampaign, campaign_id)
    if campaign is None or campaign.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="campaign not found")
    if campaign.status != "active":
        raise HTTPException(
            status_code=409, detail=f"campaign is {campaign.status!r}, not active")

    pipeline = tenant.settings.pipeline
    tel = pipeline.telephony
    provider = (tel.provider or "").lower()

    # webconsole tenants are tested in the browser, not dialed out.
    if provider == "webconsole":
        raise HTTPException(
            status_code=409,
            detail=("this tenant's telephony is 'webconsole' — it has no outbound "
                    "dialing. Test it from the browser console (/console or "
                    "/dev/voice); those sessions are still recorded + billed."))

    # Enforce the per-tenant concurrency cap.
    cap = tenant.settings.max_concurrent_calls
    if await count_active_calls(session, tenant.id) >= cap:
        raise HTTPException(
            status_code=429,
            detail=f"max concurrent calls reached ({cap}); retry when a call ends")

    # Per-tenant compliance gate: calling hours + DND, from the runtime registry
    # (skipped only in tests/contexts where the registry isn't wired).
    registry = getattr(request.app.state, "registry", None)
    if registry is not None:
        dnd = registry.dnd.get(tenant)
        if not dnd.hours.can_call_now():
            raise HTTPException(
                status_code=403, detail="outside this tenant's configured calling hours")
        if dnd.filter.is_blocked(req.to_number):
            raise HTTPException(
                status_code=403, detail="destination is on the tenant's DND list")

    from_number = req.from_number or (tel.outbound_from or {}).get(provider) or tel.from_number
    if not from_number:
        raise HTTPException(status_code=400, detail="no caller-ID configured for this tenant")
    webhook_base = platform_webhook_base_url()   # platform-level; not per-tenant
    if not webhook_base:
        raise HTTPException(status_code=400, detail="platform WEBHOOK_BASE_URL is not set")

    # Dial with the TENANT's telephony creds (resolved for its configured
    # provider), not the platform env — otherwise every tenant's call bills/
    # identifies as the platform's account. A cred that isn't configured yet
    # resolves to None so the adapter can still fall back to env (until the
    # tenant's keys are migrated into the DB).
    def _cred(name):
        try:
            return tenant.secret(name) if name else None
        except Exception:  # noqa: BLE001 - missing env → let the adapter decide
            return None

    creds = tel.active_creds()
    acct, auth = _cred(creds.account_sid_env), _cred(creds.auth_token_env)
    # Stringee: the callout needs a non-null userId, or it goes out as a
    # phone->phone external call and the Answer URL/SCCO never runs (silent bot).
    uid = _cred(creds.user_id_env)
    try:
        adapter = get_telephony_provider({
            "provider": provider,
            "account_sid": acct, "auth_token": auth,
            # Stringee's server adapter reads api_key_sid/api_key_secret (its
            # account keys are stored as account_sid/auth_token) — pass them so it
            # dials with the tenant's creds, not STRINGEE_API_KEY_SID from env.
            "api_key_sid": acct, "api_key_secret": auth,
            "user_id": uid,
            # Regional Stringee REST host (e.g. asia-2.api.stringee.com); None →
            # adapter falls back to api.stringee.com.
            "base_url": tel.stringee_base_url,
        })
    except Exception as e:  # noqa: BLE001 — e.g. missing credentials
        raise HTTPException(status_code=400, detail=f"telephony adapter unavailable: {e}")

    # Slug-scope the answer URL so the answering bridge resolves THIS tenant by
    # slug (and runs with its config), instead of reverse-resolving our caller-ID
    # — which would require the number to be in tenant_phone_numbers. Same pattern
    # as the dev console. Providers without a slug-scoped route keep the bare base.
    answer_url = webhook_base.rstrip("/")
    answer_path = ANSWER_PATHS.get(provider)
    if answer_path:
        answer_url = f"{answer_url}/{answer_path}/{tenant.slug}"
    cfg = CallConfig(
        to_number=req.to_number.strip(),
        from_number=from_number,
        webhook_url=answer_url,
    )
    try:
        call_session = await adapter.initiate_call(cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("call lead failed", extra={"tenant": tenant.slug, "provider": provider})
        raise HTTPException(status_code=502, detail=f"call failed: {e}")

    call_id = f"call_{uuid.uuid4().hex[:16]}"
    await insert_call(
        session, call_id=call_id, tenant=tenant, provider_call_sid=call_session.session_id,
        channel="voice", campaign_id=campaign_id, lead_id=req.lead_id, voice=req.voice,
    )
    log.info("call lead placed", extra={
        "tenant": tenant.slug, "campaign": campaign_id, "call_id": call_id,
        "sid": call_session.session_id})
    return CallLeadResponse(
        call_id=call_id, status="in_progress", provider_call_sid=call_session.session_id)


@router.get("/calls/{call_id}", response_model=CallStatusResponse)
async def get_call(
    call_id: str,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CallStatusResponse:
    row = await session.get(Conversation, call_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="call not found")
    return CallStatusResponse(
        call_id=row.id, status=row.status, outcome=row.outcome,
        summary=row.summary, notes=row.notes,
        callback_at=row.callback_at.isoformat() if row.callback_at else None,
        cost=row.cost, duration_ms=row.duration_ms,
    )


@router.post("/calls/{call_id}/summarize-outcome", response_model=SummarizeOutcomeResponse)
async def summarize_outcome(
    call_id: str,
    request: Request,
    audio: UploadFile = File(...),
    audio_mime: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> SummarizeOutcomeResponse:
    """Transcribe a call recording and return (+ persist) the outcome + summary.

    CRM sends the voice recording when it has the file but the platform could not
    analyze it automatically (e.g. recording-unavailable outcome). Returns the
    same structured outcome an AI call produces.
    """
    from datetime import datetime, timezone

    from src.analysis.call_outcome import analyze_call
    from src.api.call_store import record_outcome
    from src.interfaces.llm import LLMMessage
    from src.providers import get_llm_provider

    row = await session.get(Conversation, call_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="call not found")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio file is empty")
    mime = audio_mime or audio.content_type or "audio/mpeg"

    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="provider registry not available")

    llm = registry.get_llm(tenant)
    # Prefer the tenant's LLM if it supports audio; fall back to platform Gemini
    # (strong on Indian languages + handles long mono mp3 recordings).
    transcriber = llm if getattr(llm, "transcribe_audio", None) else None
    if transcriber is None:
        try:
            transcriber = get_llm_provider({"provider": "gemini"})
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"no audio transcriber available; ensure GEMINI_API_KEY is set: {e}",
            )

    try:
        text = await transcriber.transcribe_audio(audio_bytes, mime)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"transcription failed: {e}")

    transcript = [LLMMessage(role="user", content=text)] if (text or "").strip() else []
    analysis = await analyze_call(
        transcript=transcript,
        slots={},
        telephony_status=None,
        final_action=None,
        tenant_timezone=tenant.settings.timezone,
        now=datetime.now(timezone.utc),
        llm=llm,
    )

    sid = row.provider_call_sid or call_id
    await record_outcome(
        session, sid,
        status="ended",
        outcome=analysis.outcome.value,
        summary=analysis.summary,
        notes=analysis.notes or "",
        callback_at=analysis.callback_datetime,
    )
    log.info("summarize-outcome complete", extra={
        "tenant": tenant.slug, "call_id": call_id,
        "outcome": analysis.outcome.value, "source": analysis.analysis_source,
    })
    return SummarizeOutcomeResponse(
        call_id=call_id,
        outcome=analysis.outcome.value,
        summary=analysis.summary,
        notes=analysis.notes or "",
        callback_at=analysis.callback_datetime.isoformat() if analysis.callback_datetime else None,
    )
