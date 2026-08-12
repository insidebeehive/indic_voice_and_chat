"""CRM telephony integration — call registration.

One endpoint that lets the CRM manage the telephony side while delegating
the AI conversation to us:

    POST /api/v1/telephony/register-call
        CRM pre-registers a call it is about to place. Creates the
        conversation row before Twilio/Exotel fires our answer webhook. The
        CRM places the call using our slug-scoped answer URL, and the existing
        answer handler + AI bridge take over when the customer answers.

This endpoint is standard tenant Bearer-auth. The existing
``POST /api/v1/campaigns/{id}/calls`` route is unchanged.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.call_store import insert_call
from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.models.conversation import Conversation

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony-crm"])

# Providers that support call registration: twilio/exotel via our
# answer-webhook / streaming-media integration; livekit via room-join
# (see docs/integrations/livekit-room-handoff.md).
_SUPPORTED_PROVIDERS = {"twilio", "exotel", "livekit"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterCallRequest(BaseModel):
    provider: str = Field(description="'twilio', 'exotel', or 'livekit'")
    provider_call_sid: str = Field(min_length=1)
    campaign_id: str | None = None


# ---------------------------------------------------------------------------
# Register a call the CRM is placing
# ---------------------------------------------------------------------------


@router.post("/register-call", status_code=201)
async def register_call(
    req: RegisterCallRequest,
    response: Response,
    tenant: TenantContext = Depends(current_tenant),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Pre-register a call the CRM is placing. Behavior branches by provider:

    **twilio / exotel:** call this *before* the CRM dials so a
    ``conversations`` row exists when Twilio/Exotel fires our slug-scoped
    answer webhook. The CRM must place the call using our answer URL:

        ``https://{host}/api/v1/telephony/{provider}/voice/{tenant_slug}``

    The AI bridge takes over when the customer answers — no further API call
    is needed.

    **livekit:** registration here is *optional* — see
    ``docs/integrations/livekit-room-handoff.md``. LiveKit calls are not
    dialed through an answer URL at all: our webhook route reacts to a SIP
    participant joining a LiveKit room, and the CRM creates that room + SIP
    participant directly (per the doc). Pre-registering via this endpoint
    only gets you two things the webhook-triggered auto-create path can't:
    ``call.initiated`` fires at dial time instead of at room-join time, and
    an unknown ``campaign_id`` is validated up front (400) instead of
    silently falling back to the tenant's default campaign. If you use it,
    ``provider_call_sid`` must equal the exact LiveKit room name you are
    about to create.

    Returns 200 (not 201) if the SID is already registered, so it is safe to
    call idempotently.
    """
    provider = req.provider.lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider {req.provider!r} is not supported for call registration "
                   f"(supported: {', '.join(sorted(_SUPPORTED_PROVIDERS))})",
        )

    # Idempotency: return the existing row if the SID is already known.
    existing = (await db.execute(
        select(Conversation).where(
            Conversation.provider_call_sid == req.provider_call_sid,
            Conversation.tenant_id == tenant.id,
        )
    )).scalar_one_or_none()
    if existing is not None:
        response.status_code = 200
        return {"call_id": existing.id, "status": existing.status}

    call_id = f"call_{uuid.uuid4().hex[:16]}"
    try:
        await insert_call(
            db,
            call_id=call_id,
            tenant=tenant,
            provider_call_sid=req.provider_call_sid,
            campaign_id=req.campaign_id,
            extra_event_data={"source": "crm_register"},
        )
    except IntegrityError as e:
        await db.rollback()
        raise HTTPException(
            status_code=400,
            detail="campaign_id does not reference an existing campaign — "
                   "register it via POST /campaigns first",
        ) from e
    log.info(
        "crm registered call",
        extra={"call_id": call_id, "sid": req.provider_call_sid,
               "tenant": tenant.slug, "provider": provider},
    )
    return {"call_id": call_id, "status": "in_progress"}
