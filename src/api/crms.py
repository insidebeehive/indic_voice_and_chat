"""Admin CRUD for Crm entities + their tool catalogs.

- ``GET    /crms``       list every CRM (id, name, base_url)
- ``POST   /crms``       create a CRM + its initial tool list
- ``GET    /crms/{id}``  detail including the full tool list
- ``PATCH  /crms/{id}``  update CRM-level fields and/or replace the tool list

Admin-only (``require_admin``) — a CRM is shared platform config, not
tenant-scoped. A tenant is linked to a CRM via ``PATCH /tenants/{id}``
(``crm_id`` field), not through this router.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth.middleware import require_admin
from src.models.crm import Crm, CrmTool

router = APIRouter(prefix="/crms", tags=["crms"])


class CrmToolIn(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    method: str = "GET"
    parameters: dict = Field(default_factory=dict)


class CrmToolOut(CrmToolIn):
    id: int


class CrmCreateRequest(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    events_webhook_url_template: Optional[str] = None
    auth_type: str = "api_key"
    tools: list[CrmToolIn] = Field(default_factory=list)


class CrmUpdateRequest(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    events_webhook_url_template: Optional[str] = None
    auth_type: Optional[str] = None
    tools: Optional[list[CrmToolIn]] = None  # if provided, REPLACES the entire tool list


class CrmSummary(BaseModel):
    id: str
    name: str
    base_url: str


class CrmDetail(CrmSummary):
    events_webhook_url_template: Optional[str]
    auth_type: str
    tools: list[CrmToolOut]


@router.get("", response_model=list[CrmSummary])
async def list_crms(
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> list[CrmSummary]:
    rows = (await session.execute(select(Crm))).scalars().all()
    return [CrmSummary(id=r.id, name=r.name, base_url=r.base_url) for r in rows]


@router.post("", response_model=CrmDetail)
async def create_crm(
    req: CrmCreateRequest,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> CrmDetail:
    existing = await session.get(Crm, req.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"CRM {req.id!r} already exists")
    crm = Crm(id=req.id, name=req.name, base_url=req.base_url,
               events_webhook_url_template=req.events_webhook_url_template,
               auth_type=req.auth_type)
    session.add(crm)
    for t in req.tools:
        session.add(CrmTool(crm_id=req.id, name=t.name, description=t.description,
                             endpoint=t.endpoint, method=t.method, parameters=t.parameters))
    await session.commit()
    return await _detail(session, req.id)


@router.get("/{crm_id}", response_model=CrmDetail)
async def get_crm(
    crm_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> CrmDetail:
    return await _detail(session, crm_id)


@router.patch("/{crm_id}", response_model=CrmDetail)
async def update_crm(
    crm_id: str,
    req: CrmUpdateRequest,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> CrmDetail:
    crm = await session.get(Crm, crm_id)
    if crm is None:
        raise HTTPException(status_code=404, detail="CRM not found")
    if req.name is not None:
        crm.name = req.name
    if req.base_url is not None:
        crm.base_url = req.base_url
    if req.events_webhook_url_template is not None:
        crm.events_webhook_url_template = req.events_webhook_url_template
    if req.auth_type is not None:
        crm.auth_type = req.auth_type
    if req.tools is not None:
        existing_tools = (await session.execute(
            select(CrmTool).where(CrmTool.crm_id == crm_id)
        )).scalars().all()
        for t in existing_tools:
            await session.delete(t)
        await session.flush()
        for t in req.tools:
            session.add(CrmTool(crm_id=crm_id, name=t.name, description=t.description,
                                 endpoint=t.endpoint, method=t.method, parameters=t.parameters))
    await session.commit()
    return await _detail(session, crm_id)


async def _detail(session: AsyncSession, crm_id: str) -> CrmDetail:
    crm = await session.get(Crm, crm_id)
    if crm is None:
        raise HTTPException(status_code=404, detail="CRM not found")
    tools = (await session.execute(
        select(CrmTool).where(CrmTool.crm_id == crm_id)
    )).scalars().all()
    return CrmDetail(
        id=crm.id, name=crm.name, base_url=crm.base_url,
        events_webhook_url_template=crm.events_webhook_url_template,
        auth_type=crm.auth_type,
        tools=[CrmToolOut(id=t.id, name=t.name, description=t.description,
                           endpoint=t.endpoint, method=t.method, parameters=t.parameters)
               for t in tools],
    )
