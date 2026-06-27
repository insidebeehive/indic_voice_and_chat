"""CRM telephony integration — call registration and patch-in handoff.

Two endpoints that let the CRM manage the telephony side while delegating
the AI conversation to us:

    POST /api/v1/telephony/register-call
        CRM pre-registers a call it is about to place (Pattern A). Creates the
        conversation row before Twilio/Exotel fires our answer webhook. The
        CRM places the call using our slug-scoped answer URL, and the existing
        answer handler + AI bridge take over when the customer answers.

    POST /api/v1/telephony/handoff
        CRM holds a live call and patches us in (Pattern B). We call the
        provider's live-call update API to redirect the active call's media
        stream to our AI bridge WebSocket, then emit call.initiated +
        call.answered events as if we had placed the call ourselves.

Both endpoints are standard tenant Bearer-auth. The existing
``POST /api/v1/campaigns/{id}/calls`` route is unchanged.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.answer_paths import ANSWER_PATHS
from src.api.call_store import insert_call, mark_answered
from src.api.deps import get_db_session
from src.api.telephony_hooks import (
    _bridge_factory,
    _exotel_bridge_factory,
)
from src.auth import TenantContext, current_tenant
from src.models.conversation import Conversation
from src.providers import get_telephony_provider

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony-crm"])

# Providers that support live media-stream redirect.
_REDIRECT_PROVIDERS = {"twilio", "exotel"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterCallRequest(BaseModel):
    provider: str = Field(description="'twilio' or 'exotel'")
    provider_call_sid: str = Field(min_length=1)
    lead_id: str | None = None
    campaign_id: str | None = None


class HandoffRequest(BaseModel):
    provider: str = Field(description="'twilio' or 'exotel'")
    call_sid: str = Field(min_length=1, description="Live call SID held by the CRM")
    lead_id: str | None = None
    campaign_id: str | None = None
    # Optional: if the CRM uses a separate Twilio/Exotel account from the tenant,
    # pass its credentials here. Omit to use the tenant's stored credentials.
    crm_account_sid: str | None = None
    crm_auth_token: str | None = None


# ---------------------------------------------------------------------------
# Pattern A: pre-register a call the CRM is placing
# ---------------------------------------------------------------------------


@router.post("/register-call", status_code=201)
async def register_call(
    req: RegisterCallRequest,
    tenant: TenantContext = Depends(current_tenant),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Pre-register a call that the CRM is placing via the telephony provider.

    Call this *before* the CRM dials so a ``conversations`` row exists when
    Twilio/Exotel fires our slug-scoped answer webhook. The CRM must place
    the call using our answer URL:

        ``https://{host}/api/v1/telephony/{provider}/voice/{tenant_slug}``

    The AI bridge takes over when the customer answers — no further API call
    is needed. Returns 200 (not 201) if the SID is already registered, so
    it is safe to call idempotently.
    """
    provider = req.provider.lower()
    if provider not in _REDIRECT_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider {req.provider!r} is not supported for CRM handoff "
                   f"(supported: {', '.join(sorted(_REDIRECT_PROVIDERS))})",
        )

    # Idempotency: return the existing row if the SID is already known.
    existing = (await db.execute(
        select(Conversation).where(
            Conversation.provider_call_sid == req.provider_call_sid,
            Conversation.tenant_id == tenant.id,
        )
    )).scalar_one_or_none()
    if existing is not None:
        return {"call_id": existing.id, "status": existing.status}

    call_id = f"call_{uuid.uuid4().hex[:16]}"
    await insert_call(
        db,
        call_id=call_id,
        tenant=tenant,
        provider_call_sid=req.provider_call_sid,
        campaign_id=req.campaign_id,
        lead_id=req.lead_id,
        extra_event_data={"source": "crm_register"},
    )
    log.info(
        "crm registered call",
        extra={"call_id": call_id, "sid": req.provider_call_sid,
               "tenant": tenant.slug, "provider": provider},
    )
    return {"call_id": call_id, "status": "in_progress"}


# ---------------------------------------------------------------------------
# Pattern B: CRM holds a live call and patches us in
# ---------------------------------------------------------------------------


@router.post("/handoff")
async def handoff_call(
    req: HandoffRequest,
    tenant: TenantContext = Depends(current_tenant),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Redirect a live call held by the CRM to our AI bridge (patch-in).

    The CRM has a call in progress (customer answered, CRM is holding them).
    Calling this endpoint causes the telephony provider to redirect the call's
    media stream to our AI bridge WebSocket. The AI then handles the rest of
    the conversation; call.completed arrives via the tenant's events_webhook_url
    when the bridge tears down.

    Flow:
      1. CRM places outbound call to customer.
      2. Customer answers; CRM holds them (IVR / hold music).
      3. CRM calls this endpoint with the live call SID.
      4. We redirect the call's audio stream to our AI bridge.
      5. AI greets and handles the conversation.
      6. call.completed webhook fires at teardown with outcome + summary.
    """
    provider = req.provider.lower()
    if provider not in _REDIRECT_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider {req.provider!r} is not supported for live redirect "
                   f"(Stringee is turn-based; supported: {', '.join(sorted(_REDIRECT_PROVIDERS))})",
        )

    # Check that the AI bridge is ready before doing anything irreversible.
    factory = _bridge_factory if provider == "twilio" else _exotel_bridge_factory
    if factory is None:
        raise HTTPException(status_code=503, detail="AI bridge not ready")

    # Duplicate guard: if we already have this SID, return it.
    existing = (await db.execute(
        select(Conversation).where(
            Conversation.provider_call_sid == req.call_sid,
            Conversation.tenant_id == tenant.id,
        )
    )).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"call SID already registered as {existing.id}",
        )

    # Build the streaming WSS URL that points to our existing bridge WS handler.
    answer_path = ANSWER_PATHS.get(provider, f"{provider}/voice")
    # Derive the HTTPS base from the tenant's webhook_base_url config; fall back
    # to the platform env var. We need a hostname to build the wss:// URL.
    from src.config_tenant import platform_webhook_base_url
    webhook_base = tenant.settings.webhook_base_url or platform_webhook_base_url()
    if not webhook_base:
        raise HTTPException(status_code=500, detail="WEBHOOK_BASE_URL not configured")
    host = webhook_base.rstrip("/").removeprefix("https://").removeprefix("http://")
    stream_wss_url = f"wss://{host}/api/v1/telephony/{provider}/stream/{tenant.slug}"

    # Build the telephony adapter. Use CRM-supplied credentials if provided
    # (separate account), otherwise fall back to the tenant's stored creds.
    def _cred(env_name: str | None) -> str | None:
        if not env_name:
            return None
        try:
            return tenant.secret(env_name)
        except Exception:  # noqa: BLE001
            return None

    if req.crm_account_sid and req.crm_auth_token:
        acct, auth = req.crm_account_sid, req.crm_auth_token
    else:
        creds = tenant.settings.pipeline.telephony.active_creds()
        acct = _cred(creds.account_sid_env)
        auth = _cred(creds.auth_token_env)

    try:
        adapter = get_telephony_provider(
            {"provider": provider, "account_sid": acct, "auth_token": auth}
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"telephony adapter unavailable: {e}")

    # Redirect the live call. If the provider rejects it (call already ended,
    # wrong SID, auth failure), raise 502 and do NOT create a conversation row.
    try:
        await adapter.redirect_to_stream(req.call_sid, stream_wss_url)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "crm handoff redirect failed",
            extra={"sid": req.call_sid, "tenant": tenant.slug, "error": str(e)},
        )
        raise HTTPException(status_code=502, detail=f"provider redirect failed: {e}")

    # Redirect succeeded — create the conversation row and emit events.
    call_id = f"call_{uuid.uuid4().hex[:16]}"
    await insert_call(
        db,
        call_id=call_id,
        tenant=tenant,
        provider_call_sid=req.call_sid,
        campaign_id=req.campaign_id,
        lead_id=req.lead_id,
        extra_event_data={"source": "crm_handoff"},
    )
    # The call is already answered (customer is live); emit call.answered now.
    await mark_answered(db, req.call_sid)

    log.info(
        "crm handoff initiated",
        extra={"call_id": call_id, "sid": req.call_sid,
               "tenant": tenant.slug, "provider": provider,
               "stream": stream_wss_url},
    )
    return {"call_id": call_id, "status": "answered", "provider_call_sid": req.call_sid}
