"""CRM tool registration for the ChatBot (PRD §4.6).

Tenants register external CRM API endpoints the ChatBot may call as tools:

- ``POST   /chat/tools``          register (upsert) a batch of tools
- ``GET    /chat/tools``          list this tenant's tools (no secrets)
- ``DELETE /chat/tools/{name}``   unregister a tool

A tool's auth token is stored encrypted in ``tenant_secrets`` (never in the
chat_tools row, never returned by the API).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.auth import secrets as crypto
from src.chatbot.catalog import ALL_TOOLS
from src.models.chat import ChatTool
from src.models.database import get_sessionmaker
from src.models.tenant import TenantSecret

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat-tools"])


def _token_secret_name(tool_name: str) -> str:
    return f"chat_tool:{tool_name}:token"


# --- Schemas ------------------------------------------------------------


class ToolRegistration(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    method: str = "GET"
    auth_type: Optional[str] = None  # bearer|api_key|none
    auth_token: Optional[str] = None  # stored encrypted; never returned
    parameters: dict = Field(default_factory=dict)


class RegisterToolsRequest(BaseModel):
    tools: list[ToolRegistration]


class RegisterToolsResponse(BaseModel):
    registered: int
    tool_names: list[str]


class ToolInfo(BaseModel):
    name: str
    description: str
    endpoint: str
    method: str
    auth_type: Optional[str] = None
    parameters: dict
    auth_configured: bool
    operator_id: Optional[str] = None


class ToolsListResponse(BaseModel):
    tools: list[ToolInfo]


class ResolvedToolInfo(BaseModel):
    name: str
    description: str
    endpoint: str
    auth_type: Optional[str]
    token_configured: bool  # never expose the actual token value


class ResolvedToolsResponse(BaseModel):
    source: str  # "tenant" | "platform_fallback" | "none"
    tools: list[ResolvedToolInfo]


# --- Routes -------------------------------------------------------------


@router.post("/tools", response_model=RegisterToolsResponse)
async def register_tools(
    req: RegisterToolsRequest,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> RegisterToolsResponse:
    names: list[str] = []
    for t in req.tools:
        auth_config: dict = {}
        if t.auth_token:
            if not crypto.has_key():
                raise HTTPException(
                    status_code=400,
                    detail="VOX_SECRET_KEY is not set — cannot encrypt the tool auth token")
            secret_name = _token_secret_name(t.name)
            existing_secret = (await session.execute(
                select(TenantSecret).where(
                    TenantSecret.tenant_id == tenant.id, TenantSecret.name == secret_name)
            )).scalar_one_or_none()
            enc = crypto.encrypt(t.auth_token)
            if existing_secret is not None:
                existing_secret.value_encrypted = enc
            else:
                session.add(TenantSecret(
                    tenant_id=tenant.id, name=secret_name, value_encrypted=enc))
            auth_config["token_secret_name"] = secret_name

        row = (await session.execute(
            select(ChatTool).where(ChatTool.tenant_id == tenant.id, ChatTool.name == t.name)
        )).scalar_one_or_none()
        if row is not None:
            row.description = t.description
            row.endpoint = t.endpoint
            row.method = t.method
            row.auth_type = t.auth_type
            # Keep an existing token secret if this update didn't supply a new one.
            if auth_config or t.auth_token:
                row.auth_config = auth_config
            row.parameters = t.parameters
        else:
            session.add(ChatTool(
                tenant_id=tenant.id, name=t.name, description=t.description,
                endpoint=t.endpoint, method=t.method, auth_type=t.auth_type,
                auth_config=auth_config, parameters=t.parameters))
        names.append(t.name)
    await session.commit()
    log.info("registered chat tools", extra={"tenant": tenant.slug, "tools": names})
    return RegisterToolsResponse(registered=len(names), tool_names=names)


@router.get("/tools", response_model=ToolsListResponse)
async def list_tools(
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> ToolsListResponse:
    rows = (await session.execute(
        select(ChatTool).where(ChatTool.tenant_id == tenant.id).order_by(ChatTool.name)
    )).scalars().all()
    return ToolsListResponse(tools=[
        ToolInfo(
            name=r.name, description=r.description, endpoint=r.endpoint, method=r.method,
            auth_type=r.auth_type, parameters=r.parameters or {},
            auth_configured=bool((r.auth_config or {}).get("token_secret_name")),
            operator_id=((r.auth_config or {}).get("extra_headers") or {}).get("operatorid"),
        ) for r in rows
    ])


@router.get("/tools/resolved", response_model=ResolvedToolsResponse)
async def list_resolved_tools(
    tenant: TenantContext = Depends(current_tenant),
) -> ResolvedToolsResponse:
    """The tools this tenant will ACTUALLY get on its next chat turn, computed
    fresh from the DB/env (bypasses the in-process CRM-tools cache).

    Unlike ``GET /tools`` — which only reads the tenant's own registered
    ``chat_tools`` rows — this reflects the platform-fallback path too: a
    tenant with zero DB rows can still be served the full platform catalog
    via ``PLATFORM_CRM_*`` env vars or its own ``crm:*`` secrets, and that
    tenant would otherwise see an empty list from ``GET /tools`` with no way
    to tell the fallback is active.
    """
    from src.bootstrap import resolve_crm_tools

    sessionmaker = get_sessionmaker()
    specs, execs, source = await resolve_crm_tools(tenant, sessionmaker)
    return ResolvedToolsResponse(
        source=source,
        tools=[
            ResolvedToolInfo(
                name=s.name,
                description=s.description,
                endpoint=execs[s.name]["endpoint"],
                auth_type=execs[s.name]["auth_type"],
                token_configured=bool(execs[s.name].get("token")),
            )
            for s in specs
        ],
    )


@router.delete("/tools/{tool_name}")
async def delete_tool(
    tool_name: str = Path(min_length=1),
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    row = (await session.execute(
        select(ChatTool).where(ChatTool.tenant_id == tenant.id, ChatTool.name == tool_name)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="tool not found")
    secret_name = (row.auth_config or {}).get("token_secret_name")
    await session.delete(row)
    if secret_name:
        sec = (await session.execute(
            select(TenantSecret).where(
                TenantSecret.tenant_id == tenant.id, TenantSecret.name == secret_name)
        )).scalar_one_or_none()
        if sec is not None:
            await session.delete(sec)
    await session.commit()
    return {"tool_name": tool_name, "deleted": True}


# --- Catalog seeding --------------------------------------------------------


class SeedFromCatalogRequest(BaseModel):
    crm_base_url: str = Field(min_length=1, description="Base URL of the operator's CRM API")
    auth_type: Optional[str] = "bearer"  # bearer|api_key|none
    auth_token: Optional[str] = None
    operator_id: Optional[str] = Field(None, description="CRM operator UUID — sent as operatorid header on every request")
    tools: Optional[list[str]] = None  # None → seed all catalog tools


class SeedFromCatalogResponse(BaseModel):
    registered: int
    tool_names: list[str]
    unknown: list[str]


@router.post("/tools/from-catalog", response_model=SeedFromCatalogResponse)
async def seed_tools_from_catalog(
    req: SeedFromCatalogRequest,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> SeedFromCatalogResponse:
    """Register the standard betting-platform CRM tools for this tenant.

    The caller supplies only their CRM base URL and auth token; the catalog
    provides descriptions, parameter specs, and default path templates.
    Existing tools with the same name are updated (upsert).
    """
    names_requested = req.tools or list(ALL_TOOLS.keys())
    unknown = [n for n in names_requested if n not in ALL_TOOLS]

    base = req.crm_base_url.rstrip("/")
    registered: list[str] = []

    for name in names_requested:
        if name not in ALL_TOOLS:
            continue
        spec = ALL_TOOLS[name]
        endpoint = base + spec["default_path"]

        auth_config: dict = {}
        if req.auth_token:
            if not crypto.has_key():
                raise HTTPException(
                    status_code=400,
                    detail="VOX_SECRET_KEY is not set — cannot encrypt the tool auth token")
            secret_name = _token_secret_name(name)
            existing_secret = (await session.execute(
                select(TenantSecret).where(
                    TenantSecret.tenant_id == tenant.id,
                    TenantSecret.name == secret_name)
            )).scalar_one_or_none()
            enc = crypto.encrypt(req.auth_token)
            if existing_secret is not None:
                existing_secret.value_encrypted = enc
            else:
                session.add(TenantSecret(
                    tenant_id=tenant.id, name=secret_name, value_encrypted=enc))
            auth_config["token_secret_name"] = secret_name
        if req.operator_id:
            auth_config["extra_headers"] = {"operatorid": req.operator_id}

        row = (await session.execute(
            select(ChatTool).where(ChatTool.tenant_id == tenant.id, ChatTool.name == name)
        )).scalar_one_or_none()
        if row is not None:
            row.description = spec["description"]
            row.endpoint = endpoint
            row.method = spec.get("method", "GET")
            row.auth_type = req.auth_type
            if auth_config or req.auth_token or req.operator_id:
                row.auth_config = auth_config
            row.parameters = spec["parameters"]
        else:
            session.add(ChatTool(
                tenant_id=tenant.id, name=name,
                description=spec["description"],
                endpoint=endpoint,
                method=spec.get("method", "GET"),
                auth_type=req.auth_type,
                auth_config=auth_config,
                parameters=spec["parameters"],
            ))
        registered.append(name)

    await session.commit()
    log.info("seeded catalog tools", extra={"tenant": tenant.slug, "tools": registered})
    return SeedFromCatalogResponse(
        registered=len(registered), tool_names=registered, unknown=unknown)
