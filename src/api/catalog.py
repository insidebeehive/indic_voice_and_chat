"""Provider cost catalog + voice list endpoints.

- ``GET  /api/v1/providers``                  — list every provider + cost/min
- ``PUT  /api/v1/providers/{kind}/{provider}`` — admin: maintain a rate
- ``GET  /api/v1/voices?provider=&language=``  — static voice roster (public
  reference data, same class as ``/models`` — no auth dependency); for
  provider=elevenlabs a caller presenting a valid platform-admin bearer token
  instead gets the live account roster (names + cloned/custom voices)

The cost catalog is the single source of truth read by ``GET /providers`` and the
per-call cost calculation; ``PUT`` upserts so rates can be kept current as vendor
pricing changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext
from src.auth.middleware import is_admin_token, optional_tenant, require_admin
from src.models.tenant import ProviderCost
from src.providers.model_catalog import list_models
from src.providers.voice_catalog import list_voices, supported_providers
from src.utils.logging import debug_event

log = logging.getLogger(__name__)
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
    # None = no cached-input rate configured for this row (chat cost falls
    # back to cost_per_1k_input_tokens — see src/api/chat_cost.py); this is
    # deliberately not defaulted to 0.0 like the two rates above, since 0.0
    # is itself a legitimate configured ("this provider doesn't charge for
    # cache hits") value and must stay distinguishable from "unset".
    cost_per_1k_cached_tokens: float | None = None


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
    # Same partial-update rule as the two rates above: omitted means "leave
    # unchanged" (the column itself defaults to unset/NULL for a brand-new
    # row rather than 0.0 — see ProviderCost.cost_per_1k_cached_tokens). An
    # explicit 0.0 IS a real, distinct update ("this provider doesn't charge
    # for cache hits"), so this cannot reuse the `or 0.0` idiom the two rates
    # above use on insert — that would turn an explicit 0.0 and "omitted"
    # into the same thing again, exactly the ambiguity this column exists to
    # avoid.
    cost_per_1k_cached_tokens: float | None = Field(default=None, ge=0)


class VoiceItem(BaseModel):
    voice_id: str
    gender: str | None = None
    # Only populated where the roster carries it: ElevenLabs presets (static
    # catalog) and ElevenLabs' live account roster (admin-only, below). Every
    # other provider's roster has no name, just a voice_id.
    name: str | None = None
    # Live-roster-only ("premade" / "cloned" / "generated" / ...) — never set
    # by the static catalog (see list_voices/_normalize, voice_catalog.py).
    category: str | None = None


class VoicesResponse(BaseModel):
    provider: str
    language: str
    voices: list[VoiceItem]


def _caller_is_platform_admin(request: Request) -> bool:
    """Non-raising admin check for a route that must stay usable by anonymous
    and tenant callers alike (``/voices`` has no ``Depends`` auth at all —
    see the module docstring). Deliberately NOT ``require_admin``, which
    raises 401/403 for anyone else; this only ever gates which *roster* the
    same 200 response carries.
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return False
    return is_admin_token(auth.split(" ", 1)[1].strip())


# In-process cache for ElevenLabs' live account roster (admin callers only —
# see get_voices below). One platform-wide ElevenLabs account backs every
# tenant's cloned voices, so this is a single global entry, not per-tenant.
# ~5 minutes: long enough that an admin paging through the voice picker
# doesn't trigger a live `GET /v1/voices` on every request, short enough that
# a voice cloned/renamed in the ElevenLabs dashboard shows up soon after.
_ELEVENLABS_LIVE_CACHE_TTL_S = 300.0
_elevenlabs_live_cache: dict[str, Any] = {"voices": None, "fetched_at": 0.0}


async def _live_elevenlabs_voices() -> list[dict]:
    """The platform ElevenLabs account's actual voice roster (names, gender,
    category; cloned/custom voices included), cached for
    ``_ELEVENLABS_LIVE_CACHE_TTL_S``.

    ``ElevenLabsTTSAdapter.get_available_voices`` is synchronous httpx, so it
    runs off the event loop via ``asyncio.to_thread``. It already falls back
    to the static preset list on any failure (no key configured, network
    error, non-2xx) -- see its own docstring/implementation -- so a raise out
    of the thread here is not expected, but is still caught defensively
    rather than turning into a 500 for what is reference data. The API key
    itself is never logged, here or in the adapter.
    """
    now = time.monotonic()
    cached = _elevenlabs_live_cache
    if cached["voices"] is not None and (now - cached["fetched_at"]) < _ELEVENLABS_LIVE_CACHE_TTL_S:
        return cached["voices"]
    from src.providers import TTS_PROVIDERS  # local import: same lazy pattern as dev_console's /dev/tts-voices

    adapter = TTS_PROVIDERS["elevenlabs"]({})  # {} -> picks up ELEVENLABS_API_KEY env var; ctor never raises on a missing key
    try:
        voices = await asyncio.to_thread(adapter.get_available_voices, "")
    except Exception:
        log.warning("catalog: elevenlabs live voice fetch raised unexpectedly, falling back to presets")
        from src.providers.tts.elevenlabs import _PRESET_VOICES
        voices = list(_PRESET_VOICES)
    # Cache only a real account roster. The adapter hides a failed fetch by
    # returning its preset list, whose entries carry no "category" (every
    # voice from the live API does), so a failure isn't cached: otherwise one
    # transient ElevenLabs/network error would hide cloned voices from admins
    # for the whole TTL.
    if any("category" in v for v in voices):
        cached["voices"] = voices
        cached["fetched_at"] = now
    return voices


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
                         cost_per_1k_output_tokens=r.cost_per_1k_output_tokens,
                         cost_per_1k_cached_tokens=r.cost_per_1k_cached_tokens)
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
    # CRUD boundary: this rate feeds every per-call/per-token cost calculation
    # (tenant billing, chat token cost) from the next read onward with no
    # further gate -- a fat-fingered PUT here is invisible anywhere else in
    # the system until a bill looks wrong. Before/after values, and whether
    # this created a new row or updated one, so an operator can answer "when
    # did this rate change and from what" without a DB audit table.
    before = None if row is None else {
        "cost_per_min": row.cost_per_min,
        "cost_per_1k_input_tokens": row.cost_per_1k_input_tokens,
        "cost_per_1k_output_tokens": row.cost_per_1k_output_tokens,
        "cost_per_1k_cached_tokens": row.cost_per_1k_cached_tokens,
    }
    if row is None:
        row = ProviderCost(kind=kind, provider=provider, model=req.model,
                           cost_per_min=req.cost_per_min,
                           cost_per_1k_input_tokens=req.cost_per_1k_input_tokens or 0.0,
                           cost_per_1k_output_tokens=req.cost_per_1k_output_tokens or 0.0,
                           # No `or 0.0` here, unlike the two rates above:
                           # those columns are NOT NULL/default-0.0, so a
                           # missing request value and a real 0.0 rate are
                           # already the same thing for them. This column is
                           # nullable specifically so a brand-new row created
                           # from a cost_per_min-only PUT (the live UI's only
                           # call shape) starts "unconfigured", not "free" —
                           # see ProviderCost.cost_per_1k_cached_tokens.
                           cost_per_1k_cached_tokens=req.cost_per_1k_cached_tokens)
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
        # Same partial-update rule for the cached rate. This can only ever
        # move an existing row from "unconfigured" to "configured (incl.
        # 0.0)" or from one configured value to another — there is no
        # request shape that puts it back to NULL, the same limitation the
        # two rates above already have (no "unset" wire value distinct from
        # "omitted").
        if req.cost_per_1k_cached_tokens is not None:
            row.cost_per_1k_cached_tokens = req.cost_per_1k_cached_tokens
    debug_event(
        log, "catalog provider_cost_update result",
        kind=kind, provider=provider, model=req.model, row_created=before is None,
        before=before,
        after={
            "cost_per_min": row.cost_per_min,
            "cost_per_1k_input_tokens": row.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": row.cost_per_1k_output_tokens,
            "cost_per_1k_cached_tokens": row.cost_per_1k_cached_tokens,
        },
    )
    await session.commit()
    return ProviderCostItem(kind=kind, provider=provider, model=req.model,
                            cost_per_min=row.cost_per_min,
                            cost_per_1k_input_tokens=row.cost_per_1k_input_tokens,
                            cost_per_1k_output_tokens=row.cost_per_1k_output_tokens,
                            cost_per_1k_cached_tokens=row.cost_per_1k_cached_tokens)


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
    request: Request,
    response: Response,
    provider: str = Query(
        ...,
        description=(
            "TTS provider (sarvam, gemini, google, azure, elevenlabs, indicf5) "
            "or gemini_live (S2S realtime voices)"
        ),
    ),
    language: str = Query("hi-IN", description="BCP-47 language tag (TTS only)"),
) -> VoicesResponse:
    """Return the available voices for a provider (+ language for TTS).

    ElevenLabs is a special case: the static catalog only has the 8 preset
    voices (below), never a tenant's cloned/custom ones -- fetching those
    needs a live call against the single platform-wide ElevenLabs account.
    This route has no auth dependency at all (public reference data, e.g. for
    the pre-login Register page), so that live roster -- which would leak
    every tenant's custom voice names to anyone -- is only ever returned to a
    caller presenting a valid platform-admin bearer token; everyone else
    (anonymous or a tenant token) keeps getting the static preset list,
    unchanged.
    """
    voices = list_voices(provider, language)
    if (provider or "").strip().lower() == "elevenlabs" and _caller_is_platform_admin(request):
        voices = await _live_elevenlabs_voices()
        # The body now depends on who asked; keep any cache from reusing it.
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Authorization"
    if not voices:
        # Catalog lookup that resolved to nothing -- list_voices returns []
        # both for a provider it doesn't know at all and for a real
        # per-language provider given a language it has no roster for (see
        # its own docstring); neither is an error (200, empty list), so this
        # is the only trace of "was this provider/language just unsupported,
        # or is the request malformed" anywhere in the stack. Cheap and rare
        # (a dropdown-populating admin/tenant action, not per-turn) — no
        # guard needed.
        debug_event(
            log, "catalog voices_lookup empty", provider=provider, language=language,
            known_providers=supported_providers(),
        )
    return VoicesResponse(
        provider=provider,
        language=language,
        voices=[VoiceItem(**v) for v in voices],
    )
