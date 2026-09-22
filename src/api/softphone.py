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
from src.utils.logging import debug_event

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
        # Previously unlogged server-side at any level -- the CRM backend sees
        # a 400, but an operator investigating "softphone doesn't work for
        # tenant X" from Loki had no trace of why.
        debug_event(log, "softphone token_mint rejected", reason="unsupported_provider",
                    tenant_slug=tenant.slug, agent_identity=req.agent_identity, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except SoftphoneConfigError as e:
        debug_event(log, "softphone token_mint rejected", reason="config_error",
                    tenant_slug=tenant.slug, agent_identity=req.agent_identity, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    log.info("softphone token minted", extra={
        "tenant": tenant.slug, "provider": creds.provider, "identity": creds.identity})
    params = dict(creds.params)
    if creds.provider == "stringee":
        # The browser must place the call FROM the caller-ID number (a Stringee
        # number), not the agent identity — Stringee rejects an app-to-phone call
        # whose `from` is a non-number user. Pass it through for softphone.js.
        tel = tenant.settings.pipeline.telephony
        caller_id = ((tel.outbound_from or {}).get("stringee") or tel.from_number or "").lstrip("+")
        if caller_id:
            params["from_number"] = caller_id
        else:
            # A minted token the browser can never actually place a call with --
            # Stringee will reject the app-to-phone call at dial time, far from
            # here and with nothing in our logs pointing back to this mint.
            debug_event(log, "softphone token_mint stringee_caller_id_missing",
                        tenant_slug=tenant.slug)
    return SoftphoneTokenResponse(
        provider=creds.provider, token=creds.token, identity=creds.identity,
        ttl_seconds=creds.ttl_seconds, params=params,
    )
