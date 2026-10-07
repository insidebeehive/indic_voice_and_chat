"""Platform-level STT/LLM/TTS pipeline defaults, editable live from the admin
console (`/admin`'s "Platform pipeline defaults" card) instead of only via a
`config/default.yaml` edit + deploy.

- `GET    /api/v1/platform/pipeline`         — effective/source per layer
- `PUT    /api/v1/platform/pipeline/{layer}` — set an override, applied live
- `DELETE /api/v1/platform/pipeline/{layer}` — revert to the yaml value, live

A layer with no stored row means config/default.yaml's own value applies, as
it always has -- `source` in every response names which is in effect.

"Applied live" means: `TenantProviders.global_defaults[layer]` (the shared
dict every bridge factory / softphone / chat TTS holds by reference) is
updated in place, the platform LLM singleton is rebuilt on the next call if
`layer == "llm"`, and every cached per-tenant client is evicted so it
rebuilds against the new default. What does NOT switch immediately: a chat
session with an already-open WebSocket, or a call already in progress, keeps
its already-built client until the next reconnect/call. With more than one
uvicorn worker, only the worker that served this request applies it
in-process; the others pick it up at their next restart -- this project runs
one worker in prod, so that's not built here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth.audit import current_admin_label
from src.auth.middleware import require_admin
from src.config import get_settings
from src.config_tenant import apply_platform_pipeline_override
from src.models.platform_pipeline import PlatformPipelineDefault
from src.providers.model_catalog import MODELS, models_for
from src.utils.logging import debug_event

log = logging.getLogger(__name__)
router = APIRouter(prefix="/platform", tags=["platform"])

_LAYERS = ("stt", "llm", "tts")
PipelineLayer = Literal["stt", "llm", "tts"]

# Attribute names the TTS adapters store their own resolved API key under
# (resolved in each adapter's `__init__`, already folding in that provider's
# env-var fallback chain -- e.g. google.py's GOOGLE_TTS_API_KEY -> GEMINI_API_KEY
# -- so this never re-derives env var names itself). Most adapters use
# `_api_key` (elevenlabs.py, gemini.py, google.py); azure.py uses `_key`.
# Checked generically by attribute name rather than per-provider, so a new
# adapter following either convention is covered automatically. sarvam.py
# already raises in `__init__` when its key is missing (caught by the
# `factory(cfg)` call below, before this check runs), and indicf5.py has no
# API key at all (it takes a server `base_url` instead) -- neither attribute
# exists on those adapters, so they're correctly left unblocked here.
_TTS_KEY_ATTRS = ("_api_key", "_key")


def _tts_key_missing(adapter: object) -> bool:
    for attr in _TTS_KEY_ATTRS:
        if hasattr(adapter, attr):
            return not getattr(adapter, attr)
    return False


class PipelineProviderModel(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None


class PlatformPipelineLayerOut(BaseModel):
    layer: str
    effective: PipelineProviderModel
    source: Literal["yaml", "override"]
    yaml: PipelineProviderModel
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class PlatformPipelineResponse(BaseModel):
    layers: list[PlatformPipelineLayerOut]


class UpdatePlatformPipelineRequest(BaseModel):
    provider: str
    model: Optional[str] = None


def _yaml_layer_dict(layer: str) -> dict:
    return getattr(get_settings().pipeline, layer).model_dump()


def _yaml_pm(layer: str) -> PipelineProviderModel:
    yaml_cfg = _yaml_layer_dict(layer)
    return PipelineProviderModel(provider=yaml_cfg.get("provider"), model=yaml_cfg.get("model"))


def _live_pm(request: Request, layer: str, fallback: PipelineProviderModel) -> PipelineProviderModel:
    """The layer's currently-running provider/model, read straight from
    `providers.global_defaults[layer]` -- the live, in-process dict every
    bridge factory / softphone / chat TTS actually builds its client from
    (see `TenantProviders.reset_platform_defaults`) -- rather than the DB
    row, so GET reports what's ACTUALLY running rather than merely what was
    last written (the two can differ right after a live-apply that failed
    the buildability check below, or before this process has loaded a DB
    override on its own boot). Falls back to `fallback` when `providers`
    isn't wired (e.g. a route test with a bare app) or hasn't got this layer.
    """
    providers = getattr(request.app.state, "providers", None)
    live = getattr(providers, "global_defaults", None) if providers is not None else None
    cfg = (live or {}).get(layer)
    if not cfg:
        return fallback
    return PipelineProviderModel(provider=cfg.get("provider"), model=cfg.get("model"))


def _new_cfg_for(layer: str, provider: Optional[str], model: Optional[str]) -> dict:
    """The layer config exactly as it will be applied -- `provider=None`
    means "revert to yaml" (DELETE's case), everything else is the same
    overlay `_apply_live` applies live. Always re-derives the yaml dict
    fresh (never a previously-applied override), same reason
    `apply_platform_pipeline_override`'s own docstring gives.
    """
    yaml_cfg = _yaml_layer_dict(layer)
    if provider is None:
        return dict(yaml_cfg)
    return apply_platform_pipeline_override(layer, yaml_cfg, provider, model)


def _check_buildable(request: Request, layer: str, cfg: dict) -> None:
    """Build (and discard) a client for `cfg` using the registry's own
    factory for `layer`, so an unregistered provider (e.g. `stt=deepgram`,
    which is only in `STREAMING_STT_PROVIDERS`, not the `STT_PROVIDERS`
    `stt_factory` looks up) or one whose platform API key is missing (e.g.
    `anthropic` without `ANTHROPIC_API_KEY`) is caught HERE, before anything
    is committed or applied -- not at the next voice call / chat session
    that tries to use it. Raises whatever the factory raises; callers decide
    what that means for them (PUT turns it into a 422, DELETE just warns and
    still reverts -- yaml is the deploy-time truth).

    The built client is discarded either way: every STT/LLM/TTS adapter
    constructs an SDK client object (or does nothing beyond storing config)
    in `__init__` -- none opens a network connection there, so this never
    makes a real request.

    For `layer == "tts"` specifically, a successful `factory(cfg)` isn't
    enough on its own: some TTS adapters store a missing API key as a falsy
    attribute in `__init__` and only raise once `synthesize()` is actually
    called (elevenlabs.py, gemini.py, google.py, azure.py) -- so a provider
    with no key on this host would otherwise build "successfully" here and
    only fail at the next real voice call / chat message. `_tts_key_missing`
    checks the adapter's own resolved key attribute for that case.

    No-ops when `providers` isn't wired, or this layer's `<layer>_factory`
    attribute isn't set (e.g. a route test with a bare app/stub) -- there is
    nothing to validate against in that case, same as `_apply_live`.
    """
    providers = getattr(request.app.state, "providers", None)
    factory = getattr(providers, f"{layer}_factory", None) if providers is not None else None
    if factory is None:
        return
    adapter = factory(cfg)
    if layer == "tts" and _tts_key_missing(adapter):
        raise ValueError(f"tts provider {cfg.get('provider')!r} is missing its API key")


def _apply_live(request: Request, layer: str, provider: Optional[str], model: Optional[str]) -> None:
    """Apply a pipeline-default change to the live process immediately.
    `provider=None` reverts `layer` to its yaml value (DELETE's case).
    """
    new_cfg = _new_cfg_for(layer, provider, model)
    providers = getattr(request.app.state, "providers", None)
    if providers is not None and hasattr(providers, "reset_platform_defaults"):
        providers.reset_platform_defaults(layer, new_cfg)
    registry = getattr(request.app.state, "registry", None)
    if registry is not None and hasattr(registry, "evict_all"):
        registry.evict_all()


@router.get("/pipeline", response_model=PlatformPipelineResponse)
async def get_platform_pipeline(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> PlatformPipelineResponse:
    """Per-layer effective value, its source (yaml vs. override), and the
    yaml value underneath it (so the console can show what "reset" reverts
    to). `effective` is read from the LIVE registry (`providers.global_defaults`),
    not the DB row -- it reports what's actually running, which is the same
    thing right after a successful PUT/DELETE or a boot that loaded this row,
    but can diverge from the row on a test harness with no registry wired
    (handled by `_live_pm`'s fallback below). `source` still comes from
    whether a DB row exists -- the DB is the source of truth for "is there an
    override at all", the registry is the source of truth for "what does it
    say right now"."""
    layers = []
    for layer in _LAYERS:
        yaml_pm = _yaml_pm(layer)
        row = await session.get(PlatformPipelineDefault, layer)
        row_pm = PipelineProviderModel(provider=row.provider, model=row.model) if row is not None else yaml_pm
        effective_pm = _live_pm(request, layer, fallback=row_pm)
        if row is not None:
            layers.append(PlatformPipelineLayerOut(
                layer=layer, effective=effective_pm,
                source="override", yaml=yaml_pm, updated_at=row.updated_at, updated_by=row.updated_by,
            ))
        else:
            layers.append(PlatformPipelineLayerOut(
                layer=layer, effective=effective_pm, source="yaml", yaml=yaml_pm,
            ))
    return PlatformPipelineResponse(layers=layers)


@router.put("/pipeline/{layer}", response_model=PlatformPipelineLayerOut)
async def update_platform_pipeline_default(
    layer: PipelineLayer,
    req: UpdatePlatformPipelineRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> PlatformPipelineLayerOut:
    """Upsert this layer's override and apply it live. 422 when `provider`
    isn't registered for this layer in the model catalog, or `model` isn't
    one of that provider's catalog entries (including: the provider has NO
    model dimension -- e.g. Azure/Google TTS -- and a model was sent anyway,
    or it DOES have one and none was sent) -- or when the catalog says this
    provider is valid but it can't actually be built (unregistered in the
    code-side registry, e.g. `stt=deepgram`; or missing its platform API key,
    e.g. `anthropic` without `ANTHROPIC_API_KEY`). The catalog and the
    buildability check are independent gates -- the catalog guards against
    typos and models the UI shouldn't even offer, the buildability check
    guards against the two drift/ops failure modes above that a
    catalog-only check can't see."""
    provider = req.provider.strip()
    catalog_for_layer = MODELS.get(layer, {})
    if provider not in catalog_for_layer:
        raise HTTPException(status_code=422, detail=f"unknown {layer} provider '{provider}'")
    available_models = models_for(layer, provider)
    model = (req.model or "").strip() or None
    if available_models:
        if model is None or model not in available_models:
            raise HTTPException(
                status_code=422,
                detail=f"unknown {layer} model for provider '{provider}': {req.model!r}",
            )
    else:
        model = None  # this provider has no model dimension -- any sent value is ignored/cleared

    new_cfg = _new_cfg_for(layer, provider, model)
    try:
        _check_buildable(request, layer, new_cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"{layer} provider {provider!r} could not be built: {exc}",
        ) from exc

    row = await session.get(PlatformPipelineDefault, layer)
    before = None if row is None else {"provider": row.provider, "model": row.model}
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    actor = current_admin_label() or "admin"
    if row is None:
        row = PlatformPipelineDefault(layer=layer, provider=provider, model=model,
                                       updated_at=now, updated_by=actor)
        session.add(row)
    else:
        row.provider = provider
        row.model = model
        row.updated_at = now
        row.updated_by = actor
    await session.commit()

    debug_event(
        log, "platform pipeline_override update",
        layer=layer, before=before, after={"provider": provider, "model": model}, actor=actor,
    )

    _apply_live(request, layer, provider, model)

    return PlatformPipelineLayerOut(
        layer=layer, effective=PipelineProviderModel(provider=provider, model=model),
        source="override", yaml=_yaml_pm(layer), updated_at=row.updated_at, updated_by=row.updated_by,
    )


@router.delete("/pipeline/{layer}", response_model=PlatformPipelineLayerOut)
async def delete_platform_pipeline_default(
    layer: PipelineLayer,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> PlatformPipelineLayerOut:
    """Remove this layer's override and re-apply the yaml value live.

    Unlike PUT, a failed buildability check here does NOT block the revert:
    yaml is the deploy-time truth, so if even the yaml config can't be built
    (e.g. its platform API key was pulled from the environment after the
    admin set this override), reverting to it is still strictly no worse
    than keeping the override around -- a warning is logged instead."""
    row = await session.get(PlatformPipelineDefault, layer)
    actor = current_admin_label() or "admin"

    yaml_cfg = _new_cfg_for(layer, None, None)
    try:
        _check_buildable(request, layer, yaml_cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "platform pipeline revert: yaml config for layer=%r could not be "
            "built (%s) -- reverting anyway, yaml is the deploy-time truth",
            layer, exc,
        )

    if row is not None:
        before = {"provider": row.provider, "model": row.model}
        await session.delete(row)
        await session.commit()
        debug_event(log, "platform pipeline_override revert", layer=layer, before=before, actor=actor)

    _apply_live(request, layer, None, None)

    yaml_pm = _yaml_pm(layer)
    return PlatformPipelineLayerOut(layer=layer, effective=yaml_pm, source="yaml", yaml=yaml_pm)
