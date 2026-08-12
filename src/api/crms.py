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
from src.auth import secrets as crypto
from src.auth.middleware import require_admin
from src.models.crm import Crm, CrmSecret, CrmTool

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
    # LiveKit project shared by every tenant registered under this CRM
    # (src.config_tenant.resolve_livekit_creds' CRM-level fallback). Non-secret
    # URL is stored directly on the Crm row; the key/secret pair — if both are
    # provided — is encrypted into CrmSecret. Write-only: never returned.
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None


class CrmUpdateRequest(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    events_webhook_url_template: Optional[str] = None
    auth_type: Optional[str] = None
    tools: Optional[list[CrmToolIn]] = None  # if provided, REPLACES the entire tool list
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None


class CrmSummary(BaseModel):
    id: str
    name: str
    base_url: str
    livekit_url: Optional[str] = None


class CrmDetail(CrmSummary):
    events_webhook_url_template: Optional[str]
    auth_type: str
    tools: list[CrmToolOut]
    # Whether both LiveKit secret halves are configured — names only, never
    # values, same convention as tenants.py's *_configured/*_set fields.
    livekit_configured: bool = False


@router.get("", response_model=list[CrmSummary])
async def list_crms(
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> list[CrmSummary]:
    rows = (await session.execute(select(Crm))).scalars().all()
    return [CrmSummary(id=r.id, name=r.name, base_url=r.base_url, livekit_url=r.livekit_url)
            for r in rows]


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
               auth_type=req.auth_type, livekit_url=req.livekit_url)
    session.add(crm)
    for t in req.tools:
        session.add(CrmTool(crm_id=req.id, name=t.name, description=t.description,
                             endpoint=t.endpoint, method=t.method, parameters=t.parameters))
    await _upsert_livekit_secrets(session, req.id, req.livekit_api_key, req.livekit_api_secret)
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
    if req.livekit_url is not None:
        crm.livekit_url = req.livekit_url
    await _upsert_livekit_secrets(session, crm_id, req.livekit_api_key, req.livekit_api_secret)
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


_LIVEKIT_SECRET_NAMES = ("livekit_api_key", "livekit_api_secret")


async def _upsert_livekit_secrets(
    session: AsyncSession, crm_id: str,
    api_key: Optional[str], api_secret: Optional[str],
) -> None:
    """Encrypt + upsert whichever of the two LiveKit secret halves were
    provided (mirrors the lookup-then-update-or-add pattern in
    ``src/api/tenants.py``'s ``update_tenant``). Either field may be omitted
    independently — only the provided ones are written."""
    values = {"livekit_api_key": api_key, "livekit_api_secret": api_secret}
    provided = {name: v for name, v in values.items() if v is not None}
    if not provided:
        return
    if not crypto.has_key():
        raise HTTPException(
            status_code=503,
            detail="VOX_SECRET_KEY is not set — cannot encrypt LiveKit credentials")
    for name, value in provided.items():
        existing = (await session.execute(
            select(CrmSecret).where(CrmSecret.crm_id == crm_id, CrmSecret.name == name)
        )).scalar_one_or_none()
        if existing is not None:
            existing.value_encrypted = crypto.encrypt(value)
        else:
            session.add(CrmSecret(crm_id=crm_id, name=name, value_encrypted=crypto.encrypt(value)))


async def _detail(session: AsyncSession, crm_id: str) -> CrmDetail:
    crm = await session.get(Crm, crm_id)
    if crm is None:
        raise HTTPException(status_code=404, detail="CRM not found")
    tools = (await session.execute(
        select(CrmTool).where(CrmTool.crm_id == crm_id)
    )).scalars().all()
    configured_secrets = (await session.execute(
        select(CrmSecret.name).where(
            CrmSecret.crm_id == crm_id, CrmSecret.name.in_(_LIVEKIT_SECRET_NAMES))
    )).scalars().all()
    return CrmDetail(
        id=crm.id, name=crm.name, base_url=crm.base_url,
        events_webhook_url_template=crm.events_webhook_url_template,
        auth_type=crm.auth_type, livekit_url=crm.livekit_url,
        # True only when the whole triple (url + both secrets) is usable —
        # matches exactly what resolve_livekit_creds requires to resolve this
        # CRM's LiveKit project, not just "the secrets happen to be set".
        livekit_configured=(
            bool(crm.livekit_url) and set(configured_secrets) == set(_LIVEKIT_SECRET_NAMES)
        ),
        tools=[CrmToolOut(id=t.id, name=t.name, description=t.description,
                           endpoint=t.endpoint, method=t.method, parameters=t.parameters)
               for t in tools],
    )
