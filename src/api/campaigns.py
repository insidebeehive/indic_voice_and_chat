"""Campaign endpoints — DB-backed (campaigns + leads tables).

Part of the purely-API, DB-backed restructure. A tenant creates a campaign
(persisted, ``status=active``), uploads leads, then drives calls one lead at a
time via ``POST /campaigns/{id}/calls`` (see ``src/api/calls.py``) and finally
ends the campaign (``status=ended``). The old orchestrator auto-dial endpoints
(start/pause/resume/stats) and their in-memory store were retired in favour of
this per-lead model; ``CampaignOrchestrator`` still exists and is exercised
directly in the integration tests.
"""

from __future__ import annotations

import io
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.auth.audit import log_denied
from src.campaign.models import LeadImportError, parse_leads_csv
from src.models.campaign import Campaign as DbCampaign
from src.models.campaign import Lead as DbLead

log = logging.getLogger(__name__)
router = APIRouter(prefix="/campaigns", tags=["campaigns"])


# --- Schemas ------------------------------------------------------------


class CreateCampaignRequest(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1)
    # The campaign script (campaign YAML). Stored as ``config_yaml``.
    script: str = ""


class CampaignResponse(BaseModel):
    id: str
    name: str
    status: str
    total_leads: int
    calls_attempted: int
    calls_answered: int
    leads_qualified: int

    @classmethod
    def from_row(cls, c: DbCampaign) -> "CampaignResponse":
        return cls(
            id=c.id, name=c.name, status=c.status, total_leads=c.total_leads,
            calls_attempted=c.calls_attempted, calls_answered=c.calls_answered,
            leads_qualified=c.leads_qualified,
        )


class CampaignListResponse(BaseModel):
    campaigns: list[CampaignResponse]
    total: int


class LeadResponse(BaseModel):
    id: str
    phone_number: str
    name: str | None = None
    status: str
    retry_count: int


class LeadsResponse(BaseModel):
    leads: list[LeadResponse]
    total: int


class LeadUploadResponse(BaseModel):
    campaign_id: str
    leads_added: int
    errors: list[dict]


# --- Helpers ------------------------------------------------------------


async def _scoped(session: AsyncSession, campaign_id: str, tenant: TenantContext) -> DbCampaign:
    """Fetch a campaign and 404 if it doesn't belong to ``tenant``."""
    campaign = await session.get(DbCampaign, campaign_id)
    if campaign is None or campaign.tenant_id != tenant.id:
        log_denied(
            logging.WARNING, "cross-tenant campaign access denied",
            event="cross_tenant_access_denied",
            reason=("campaign_not_owned" if campaign is not None else "campaign_not_found"),
            tenant=tenant.slug, tenant_id=tenant.id,
            resource="campaign", resource_id=campaign_id,
            found=campaign is not None,
            owner_tenant_id=(campaign.tenant_id if campaign is not None else None),
        )
        raise HTTPException(status_code=404, detail="campaign not found")
    return campaign


# --- Routes -------------------------------------------------------------


@router.post("", response_model=CampaignResponse, status_code=201)
async def create_campaign(
    req: CreateCampaignRequest,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CampaignResponse:
    campaign_id = req.id or f"camp_{uuid.uuid4().hex[:12]}"
    if await session.get(DbCampaign, campaign_id) is not None:
        raise HTTPException(status_code=409, detail=f"campaign {campaign_id!r} already exists")
    campaign = DbCampaign(
        id=campaign_id, tenant_id=tenant.id, name=req.name,
        status="active", config_yaml=req.script,
    )
    session.add(campaign)
    await session.commit()
    await session.refresh(campaign)
    return CampaignResponse.from_row(campaign)


@router.get("", response_model=CampaignListResponse)
async def list_campaigns(
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CampaignListResponse:
    rows = (await session.execute(
        select(DbCampaign).where(DbCampaign.tenant_id == tenant.id)
        .order_by(DbCampaign.created_at)
    )).scalars().all()
    items = [CampaignResponse.from_row(c) for c in rows]
    return CampaignListResponse(campaigns=items, total=len(items))


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(
    campaign_id: str,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CampaignResponse:
    return CampaignResponse.from_row(await _scoped(session, campaign_id, tenant))


@router.post("/{campaign_id}/end", response_model=CampaignResponse)
async def end_campaign(
    campaign_id: str,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CampaignResponse:
    """End Campaign — flip status to ``ended`` (terminal)."""
    campaign = await _scoped(session, campaign_id, tenant)
    campaign.status = "ended"
    await session.commit()
    await session.refresh(campaign)
    return CampaignResponse.from_row(campaign)


@router.post("/{campaign_id}/leads", response_model=LeadUploadResponse)
async def upload_leads(
    campaign_id: str,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> LeadUploadResponse:
    await _scoped(session, campaign_id, tenant)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        leads, errors = parse_leads_csv(data, campaign_id=campaign_id, tenant_id=tenant.id)
    except LeadImportError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Skip leads whose id already exists (idempotent re-upload).
    existing = set((await session.execute(
        select(DbLead.id).where(DbLead.campaign_id == campaign_id)
    )).scalars().all())
    added = 0
    for lead in leads:
        if lead.id in existing:
            continue
        session.add(DbLead(
            id=lead.id, tenant_id=lead.tenant_id, campaign_id=campaign_id,
            phone_number=lead.phone_number, name=lead.name,
            language_pref=lead.language_pref, crm_lead_id=lead.crm_lead_id,
            extra_data=lead.metadata, status=lead.status.value,
        ))
        added += 1

    campaign = await session.get(DbCampaign, campaign_id)
    campaign.total_leads = len(existing) + added
    await session.commit()
    return LeadUploadResponse(
        campaign_id=campaign_id,
        leads_added=added,
        errors=[{"row": r, "reason": msg} for r, msg in errors],
    )


@router.get("/{campaign_id}/leads", response_model=LeadsResponse)
async def list_leads(
    campaign_id: str,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> LeadsResponse:
    await _scoped(session, campaign_id, tenant)
    rows = (await session.execute(
        select(DbLead).where(DbLead.campaign_id == campaign_id).order_by(DbLead.created_at)
    )).scalars().all()
    items = [
        LeadResponse(id=lead.id, phone_number=lead.phone_number, name=lead.name,
                     status=lead.status, retry_count=lead.retry_count)
        for lead in rows
    ]
    return LeadsResponse(leads=items, total=len(items))
