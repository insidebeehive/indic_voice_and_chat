"""Provider cost catalog + voice list endpoints.

- ``GET  /api/v1/providers``                  — list every provider + cost/min
- ``PUT  /api/v1/providers/{kind}/{provider}`` — admin: maintain a rate
- ``GET  /api/v1/voices?provider=&language=``  — static voice roster (tenant-authed)

The cost catalog is the single source of truth read by ``GET /providers`` and the
per-call cost calculation; ``PUT`` upserts so rates can be kept current as vendor
pricing changes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext
from src.auth.middleware import optional_tenant, require_admin
from src.models.tenant import ProviderCost
from src.providers.model_catalog import list_models
from src.providers.voice_catalog import list_voices

router = APIRouter(tags=["catalog"])


async def tenant_or_admin(
    request: Request, tenant: TenantContext | None = Depends(optional_tenant)
) -> None:
    """Allow a valid tenant **or** admin bearer (e.g. the cost catalog, which
    both the tenant page and the admin page list)."""
    if tenant is not None:
        return
    await require_admin(request)  # raises 401/403 if not a valid admin token


# --- Schemas ------------------------------------------------------------


class ProviderCostItem(BaseModel):
    kind: str
    provider: str
    model: str = ""        # "" = provider-level (telephony / fallback)
    cost_per_min: float
    # Chat (text) token rates — meaningful for kind="llm" only; 0 elsewhere.
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0


class ProvidersResponse(BaseModel):
    providers: list[ProviderCostItem]


class UpdateProviderCostRequest(BaseModel):
    cost_per_min: float = Field(ge=0)
    model: str = ""        # specific model variant; "" for provider-level
    # Optional: omitted (None) means "leave this rate unchanged" — the live UI
    # only ever sends cost_per_min/model, so these must not default-zero the
    # stored token rates on every plain per-minute rate edit.
    cost_per_1k_input_tokens: float | None = Field(default=None, ge=0)
    cost_per_1k_output_tokens: float | None = Field(default=None, ge=0)


class VoiceItem(BaseModel):
    voice_id: str
    gender: str | None = None


class VoicesResponse(BaseModel):
    provider: str
    language: str
    voices: list[VoiceItem]


# --- Routes -------------------------------------------------------------


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(tenant_or_admin),
) -> ProvidersResponse:
    """List every (provider, model) rate, ordered by kind, provider, model."""
    rows = (await session.execute(
        select(ProviderCost).order_by(
            ProviderCost.kind, ProviderCost.provider, ProviderCost.model)
    )).scalars().all()
    return ProvidersResponse(providers=[
        ProviderCostItem(kind=r.kind, provider=r.provider, model=r.model,
                         cost_per_min=r.cost_per_min,
                         cost_per_1k_input_tokens=r.cost_per_1k_input_tokens,
                         cost_per_1k_output_tokens=r.cost_per_1k_output_tokens)
        for r in rows
    ])


@router.put("/providers/{kind}/{provider}", response_model=ProviderCostItem)
async def update_provider_cost(
    kind: str,
    provider: str,
    req: UpdateProviderCostRequest,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> ProviderCostItem:
    """Upsert a (provider, model) cost/min (+ per-1k-token LLM rates). Admin-only.
    New rate is read live.

    ``model`` defaults to "" (provider-level / telephony).
    """
    row = await session.get(ProviderCost, (kind, provider, req.model))
    if row is None:
        row = ProviderCost(kind=kind, provider=provider, model=req.model,
                           cost_per_min=req.cost_per_min,
                           cost_per_1k_input_tokens=req.cost_per_1k_input_tokens or 0.0,
                           cost_per_1k_output_tokens=req.cost_per_1k_output_tokens or 0.0)
        session.add(row)
    else:
        row.cost_per_min = req.cost_per_min
        # Partial update: only overwrite a token rate when the caller actually
        # sent one. Omitting it (the only live UI caller only sends
        # cost_per_min/model) must not silently zero out the stored rate.
        if req.cost_per_1k_input_tokens is not None:
            row.cost_per_1k_input_tokens = req.cost_per_1k_input_tokens
        if req.cost_per_1k_output_tokens is not None:
            row.cost_per_1k_output_tokens = req.cost_per_1k_output_tokens
    await session.commit()
    return ProviderCostItem(kind=kind, provider=provider, model=req.model,
                            cost_per_min=row.cost_per_min,
                            cost_per_1k_input_tokens=row.cost_per_1k_input_tokens,
                            cost_per_1k_output_tokens=row.cost_per_1k_output_tokens)


class ModelsResponse(BaseModel):
    # kind -> provider -> [model ids]; first per list is the recommended default.
    models: dict[str, dict[str, list[str]]]


@router.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """Selectable provider + model variants per kind (stt/llm/tts/s2s).

    Public reference data (model ids only) — drives the Register UI's
    provider/model dropdowns, which must populate before any token is entered.
    """
    return ModelsResponse(models=list_models())


@router.get("/voices", response_model=VoicesResponse)
async def get_voices(
    provider: str = Query(..., description="sarvam | gemini_live"),
    language: str = Query("hi-IN", description="BCP-47 language tag (TTS only)"),
) -> VoicesResponse:
    """Return the available voices for a provider (+ language for TTS)."""
    voices = list_voices(provider, language)
    return VoicesResponse(
        provider=provider,
        language=language,
        voices=[VoiceItem(**v) for v in voices],
    )
