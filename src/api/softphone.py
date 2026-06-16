"""Browser softphone token endpoint (``POST /api/v1/softphone/token``).

A CRM's **backend** calls this (with its tenant bearer token) to mint a
short-lived browser credential for one of its human agents; it then hands that
credential to the agent's browser, which loads the telephony provider's JS SDK
(Twilio Voice / Stringee Web) and dials the lead. Our long-lived tenant token
never reaches the browser — only the short-lived provider token does.

Routes by the tenant's configured telephony provider. Providers without a
browser SDK (Exotel, DiDLogic/SIP) return 400 until a WebRTC↔SIP gateway exists.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.auth import TenantContext, current_tenant
from src.providers.telephony.softphone import (
    DEFAULT_TTL_SECONDS,
    SoftphoneConfigError,
    SoftphoneUnsupported,
    mint_browser_credentials,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/softphone", tags=["softphone"])


class SoftphoneTokenRequest(BaseModel):
    # The CRM's human-agent id — becomes the SDK identity / Stringee userId.
    agent_identity: str = Field(min_length=1, max_length=128)
    ttl_seconds: int = Field(default=DEFAULT_TTL_SECONDS, ge=60, le=86_400)


class SoftphoneTokenResponse(BaseModel):
    provider: str
    token: str
    identity: str
    ttl_seconds: int
    params: dict[str, str] = Field(default_factory=dict)


@router.post("/token", response_model=SoftphoneTokenResponse)
async def mint_softphone_token(
    req: SoftphoneTokenRequest,
    tenant: TenantContext = Depends(current_tenant),
) -> SoftphoneTokenResponse:
    try:
        creds = mint_browser_credentials(
            tenant, req.agent_identity, ttl_seconds=req.ttl_seconds
        )
    except SoftphoneUnsupported as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SoftphoneConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    log.info("softphone token minted", extra={
        "tenant": tenant.slug, "provider": creds.provider, "identity": creds.identity})
    return SoftphoneTokenResponse(
        provider=creds.provider, token=creds.token, identity=creds.identity,
        ttl_seconds=creds.ttl_seconds, params=creds.params,
    )
