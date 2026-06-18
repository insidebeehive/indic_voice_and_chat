"""Register Tenant endpoint (``POST /api/v1/tenants``) — admin-authed.

A CRM/partner self-registers a tenant through this API instead of dropping a
YAML file. The body carries **provider choices** for STT/LLM/TTS/realtime (which
use the shared master keys — no keys accepted here) and the **telephony**
credentials (the only per-tenant secrets — encrypted at rest into
``tenant_secrets``). We build the same ``TenantPipelineConfig`` the YAML path
produced, persist the tenant + phone numbers + telephony secrets, issue one API
token (returned once, stored only as a hash), and refresh the live resolver.
"""

from __future__ import annotations

import logging
import re
import secrets as pysecrets
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.middleware import require_admin
from src.config_tenant import (
    TelephonyCreds,
    TenantPipelineConfig,
    TenantRealtimeConfig,
    TenantSTTConfig,
    TenantTelephonyConfig,
    TenantTTSConfig,
)
from src.config_tenant import TenantLLMConfig as _LLM
from src.models.conversation import Conversation
from src.models.tenant import ProviderCost, Tenant, TenantApiKey, TenantPhoneNumber, TenantSecret

log = logging.getLogger(__name__)
router = APIRouter(prefix="/tenants", tags=["tenants"])

# Shared master-key env var per provider. STT/LLM/TTS/realtime resolve their key
# from these platform env vars (via TenantContext.secret's fallback to os.environ);
# they are NEVER stored per tenant. Telephony keys are the per-tenant exception.
_MASTER_KEY_ENV = {
    "sarvam": "SARVAM_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "gemini_live": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}


# --- Schemas ------------------------------------------------------------


class LayerChoice(BaseModel):
    """Provider choice for one cascade layer — no keys (uses master keys)."""
    provider: str
    model: Optional[str] = None
    language: Optional[str] = None
    voice_id: Optional[str] = None  # tts only
    speed: Optional[float] = None   # tts only


class RealtimeChoice(BaseModel):
    provider: str
    model: Optional[str] = None
    voice: Optional[str] = None
    language_code: Optional[str] = None


class TelephonyConfigIn(BaseModel):
    provider: str
    from_number: Optional[str] = None
    # No per-tenant inbound webhook base URL — it's the platform WEBHOOK_BASE_URL
    # (always our app, common to every tenant).
    events_webhook_url: Optional[str] = None        # outbound call-event callback
    events_webhook_secret_env: Optional[str] = None  # env var naming the signing secret
    # Telephony credentials — the ONLY per-tenant secrets. Encrypted at rest.
    # e.g. {"account_sid": "AC...", "auth_token": "..."}. Optional for providers
    # (Stringee) whose adapter reads its keys from the platform env directly.
    keys: dict[str, str] = Field(default_factory=dict)
    phone_numbers: list[str] = Field(default_factory=list)


class RegisterTenantRequest(BaseModel):
    name: str = Field(min_length=1)
    slug: Optional[str] = None
    timezone: str = "Asia/Kolkata"
    default_language: str = "hi"
    mode: str = Field(default="layered", pattern="^(layered|s2s)$")
    max_concurrent_calls: int = Field(default=1, ge=1)
    stt: Optional[LayerChoice] = None
    llm: Optional[LayerChoice] = None
    tts: Optional[LayerChoice] = None
    realtime: Optional[RealtimeChoice] = None
    telephony: TelephonyConfigIn


class RegisterTenantResponse(BaseModel):
    tenant_id: str
    slug: str
    api_token: str


# --- Helpers ------------------------------------------------------------


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or f"tenant-{uuid.uuid4().hex[:8]}"


def _layer_key_env(provider: Optional[str]) -> Optional[str]:
    return _MASTER_KEY_ENV.get(provider) if provider else None


# Telephony credential logical-name → the TenantTelephonyConfig *_env field it sets.
# account_sid/auth_token drive server dialing; the api_key_* / twiml_app_sid drive
# the browser softphone (Twilio). Stringee reuses account_sid/auth_token.
_TEL_KEY_ENV_FIELD = {
    "account_sid": "account_sid_env", "sid": "account_sid_env",
    "auth_token": "auth_token_env", "token": "auth_token_env",
    "api_key_sid": "api_key_sid_env", "twilio_api_key_sid": "api_key_sid_env",
    "api_key_secret": "api_key_secret_env", "twilio_api_key_secret": "api_key_secret_env",
    "twiml_app_sid": "twiml_app_sid_env",
    # Stringee outbound app-user id — a non-null userId keeps the callout an
    # app-user->phone call so the Answer URL/SCCO runs (else silent phone->phone).
    "user_id": "user_id_env", "stringee_user_id": "user_id_env",
}


def _map_telephony_keys(slug: str, keys: dict[str, str]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Map provider key names to encrypted secret rows + the *_env fields they set.

    Returns ``(secret_rows, env_fields)`` where ``secret_rows`` is ``[(env_name,
    value)]`` to encrypt into ``tenant_secrets`` and ``env_fields`` is
    ``{TenantTelephonyConfig field: env_name}`` (e.g. ``{"account_sid_env": "..."}``).
    """
    secret_rows: list[tuple[str, str]] = []
    env_fields: dict[str, str] = {}
    for logical, value in keys.items():
        name = f"TENANT_{slug.upper().replace('-', '_')}_{logical.upper()}"
        secret_rows.append((name, value))
        field = _TEL_KEY_ENV_FIELD.get(logical)
        if field:
            env_fields[field] = name
    return secret_rows, env_fields


# --- Route --------------------------------------------------------------


@router.post("", response_model=RegisterTenantResponse, status_code=201)
async def register_tenant(
    req: RegisterTenantRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> RegisterTenantResponse:
    slug = req.slug or _slugify(req.name)
    existing = (await session.execute(
        select(Tenant.id).where(Tenant.slug == slug)
    )).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"tenant slug {slug!r} already exists")

    tenant_id = f"t_{uuid.uuid4().hex[:16]}"

    # Telephony secrets keyed by synthetic names that pipeline_config references;
    # TenantContext.secret(<name>) finds them in the decrypted secrets dict.
    tel = req.telephony
    if tel.keys and not crypto.has_key():
        raise HTTPException(
            status_code=503,
            detail="VOX_SECRET_KEY is not set — cannot encrypt telephony keys",
        )
    secret_rows, env_fields = _map_telephony_keys(slug, tel.keys)

    pipeline = TenantPipelineConfig(
        mode=req.mode,
        stt=TenantSTTConfig(
            provider=req.stt.provider if req.stt else None,
            model=req.stt.model if req.stt else None,
            language=req.stt.language if req.stt else None,
            api_key_env=_layer_key_env(req.stt.provider) if req.stt else None,
        ),
        llm=_LLM(
            provider=req.llm.provider if req.llm else None,
            model=req.llm.model if req.llm else None,
            api_key_env=_layer_key_env(req.llm.provider) if req.llm else None,
        ),
        tts=TenantTTSConfig(
            provider=req.tts.provider if req.tts else None,
            model=req.tts.model if req.tts else None,
            language=req.tts.language if req.tts else None,
            voice_id=req.tts.voice_id if req.tts else None,
            speed=req.tts.speed if req.tts else None,
            api_key_env=_layer_key_env(req.tts.provider) if req.tts else None,
        ),
        realtime=TenantRealtimeConfig(
            provider=req.realtime.provider,
            model=req.realtime.model,
            voice=req.realtime.voice,
            language_code=req.realtime.language_code,
            api_key_env=_layer_key_env(req.realtime.provider),
        ) if req.realtime else None,
        telephony=TenantTelephonyConfig(
            provider=tel.provider,
            from_number=tel.from_number,
            events_webhook_url=tel.events_webhook_url,
            events_webhook_secret_env=tel.events_webhook_secret_env,
            **env_fields,
            # also record the creds under the provider-specific slot so a tenant
            # that later adds a second provider's keys resolves each correctly.
            creds_by_provider=(
                {tel.provider.lower(): TelephonyCreds(**env_fields)}
                if env_fields and tel.provider else {}
            ),
        ),
    )

    session.add(Tenant(
        id=tenant_id, slug=slug, name=req.name, status="active",
        timezone=req.timezone, default_language=req.default_language,
        mode=req.mode, max_concurrent_calls=req.max_concurrent_calls,
        pipeline_config=pipeline.model_dump(),
    ))
    for ph in tel.phone_numbers:
        session.add(TenantPhoneNumber(
            phone_number=ph, tenant_id=tenant_id, provider=tel.provider))
    for name, value in secret_rows:
        session.add(TenantSecret(
            tenant_id=tenant_id, name=name, value_encrypted=crypto.encrypt(value)))

    api_token = f"vox_{pysecrets.token_urlsafe(32)}"
    session.add(TenantApiKey(
        token_hash=hash_api_token(api_token), tenant_id=tenant_id, label="register"))

    await session.commit()

    # Refresh the live resolver so the new tenant resolves immediately.
    resolver = getattr(request.app.state, "tenant_resolver", None)
    if resolver is not None and hasattr(resolver, "refresh"):
        await resolver.refresh(tenant_id)
        if hasattr(request.app.state, "tenants"):
            request.app.state.tenants = resolver.loaded_settings()

    log.info("registered tenant", extra={"tenant_id": tenant_id, "slug": slug})
    return RegisterTenantResponse(tenant_id=tenant_id, slug=slug, api_token=api_token)


# --- Update tenant credentials/config (admin) ---------------------------


class TelephonyUpdateIn(BaseModel):
    """Partial telephony update — only the provided fields change. ``keys`` are
    merged (upserted) into the tenant's existing encrypted secrets."""
    provider: Optional[str] = None
    from_number: Optional[str] = None
    events_webhook_url: Optional[str] = None
    events_webhook_secret_env: Optional[str] = None
    keys: dict[str, str] = Field(default_factory=dict)
    phone_numbers: Optional[list[str]] = None


class UpdateTenantRequest(BaseModel):
    status: Optional[str] = Field(default=None, pattern="^(active|suspended)$")
    telephony: Optional[TelephonyUpdateIn] = None


class UpdateTenantResponse(BaseModel):
    tenant_id: str
    slug: str
    status: str
    telephony_provider: Optional[str] = None
    # Which telephony credential slots are now configured (names only, never
    # values) — lets an admin confirm e.g. the Stringee account_sid is set.
    telephony_creds_configured: list[str]


async def _refresh_resolver(request: Request, tenant_id: str) -> None:
    """Reload one tenant into the live resolver + drop its cached provider clients."""
    resolver = getattr(request.app.state, "tenant_resolver", None)
    if resolver is not None and hasattr(resolver, "refresh"):
        await resolver.refresh(tenant_id)
        if hasattr(request.app.state, "tenants"):
            request.app.state.tenants = resolver.loaded_settings()
    providers = getattr(request.app.state, "providers", None)
    if providers is not None and hasattr(providers, "evict"):
        providers.evict(tenant_id)


@router.patch("/{tenant_id}", response_model=UpdateTenantResponse)
async def update_tenant(
    tenant_id: str,
    req: UpdateTenantRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> UpdateTenantResponse:
    """Update an existing tenant's telephony credentials/config or status (admin).

    Telephony ``keys`` are encrypted and **merged** into the tenant's secrets
    (existing names are overwritten, new ones added), and the matching ``*_env``
    references are written into ``pipeline_config.telephony``. The live resolver
    is refreshed so the change takes effect immediately — e.g. fixing a wrong
    Stringee API Key SID without re-registering the tenant.
    """
    t = await _require_tenant(session, tenant_id)

    if req.status is not None:
        t.status = req.status

    if req.telephony is not None:
        tu = req.telephony
        pc = dict(t.pipeline_config or {})
        tel_cfg = dict(pc.get("telephony") or {})
        if tu.provider is not None:
            tel_cfg["provider"] = tu.provider
        if tu.from_number is not None:
            tel_cfg["from_number"] = tu.from_number
        if tu.events_webhook_url is not None:
            tel_cfg["events_webhook_url"] = tu.events_webhook_url
        if tu.events_webhook_secret_env is not None:
            tel_cfg["events_webhook_secret_env"] = tu.events_webhook_secret_env

        if tu.keys:
            if not crypto.has_key():
                raise HTTPException(
                    status_code=503,
                    detail="VOX_SECRET_KEY is not set — cannot encrypt telephony keys")
            secret_rows, env_fields = _map_telephony_keys(t.slug, tu.keys)
            for name, value in secret_rows:
                existing = (await session.execute(
                    select(TenantSecret).where(
                        TenantSecret.tenant_id == tenant_id, TenantSecret.name == name)
                )).scalar_one_or_none()
                if existing is not None:
                    existing.value_encrypted = crypto.encrypt(value)
                else:
                    session.add(TenantSecret(
                        tenant_id=tenant_id, name=name,
                        value_encrypted=crypto.encrypt(value)))
            # write the creds into the configured provider's slot (so they're
            # selected by provider) and mirror to top-level for back-compat.
            prov = (tel_cfg.get("provider") or "").lower()
            cbp = dict(tel_cfg.get("creds_by_provider") or {})
            slot = dict(cbp.get(prov) or {})
            slot.update(env_fields)
            if prov:
                cbp[prov] = slot
                tel_cfg["creds_by_provider"] = cbp
            tel_cfg.update(env_fields)

        pc["telephony"] = tel_cfg
        t.pipeline_config = pc  # reassign (new object) so the JSON column is marked dirty

        if tu.phone_numbers is not None:
            for pn in (await session.execute(
                select(TenantPhoneNumber).where(TenantPhoneNumber.tenant_id == tenant_id)
            )).scalars().all():
                await session.delete(pn)
            for ph in tu.phone_numbers:
                session.add(TenantPhoneNumber(
                    phone_number=ph, tenant_id=tenant_id, provider=tel_cfg.get("provider")))

    await session.commit()
    await _refresh_resolver(request, tenant_id)
    log.info("updated tenant", extra={"tenant_id": tenant_id})

    pc = t.pipeline_config or {}
    tel_cfg = pc.get("telephony") or {}
    return UpdateTenantResponse(
        tenant_id=t.id, slug=t.slug, status=t.status,
        telephony_provider=tel_cfg.get("provider"),
        # names only, never values — for the active provider's slot.
        telephony_creds_configured=_configured_creds(tel_cfg),
    )


# --- Backoffice: list tenants + per-tenant analytics & billing -----------


class LayerInfo(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None


class TenantSummary(BaseModel):
    tenant_id: str
    slug: str
    name: str
    status: str
    mode: str
    max_concurrent_calls: int
    stt: LayerInfo
    llm: LayerInfo
    tts: LayerInfo
    realtime: LayerInfo
    telephony_provider: Optional[str] = None
    # Non-secret telephony config (so the backoffice can prefill it) + the NAMES
    # (never values) of the creds configured for the active provider.
    telephony_from_number: Optional[str] = None
    telephony_events_webhook_url: Optional[str] = None
    telephony_events_webhook_secret_env: Optional[str] = None
    telephony_creds_configured: list[str] = Field(default_factory=list)


class TenantListResponse(BaseModel):
    tenants: list[TenantSummary]
    total: int


def _layer(pc: dict, key: str) -> LayerInfo:
    d = pc.get(key) or {}
    return LayerInfo(provider=d.get("provider"), model=d.get("model"))


_CRED_ENV_FIELDS = (
    "account_sid_env", "auth_token_env", "api_key_sid_env",
    "api_key_secret_env", "twiml_app_sid_env", "user_id_env",
)


def _configured_creds(tel_cfg: dict) -> list[str]:
    """Names (never values) of the cred slots set for the active provider."""
    active = TenantTelephonyConfig(**tel_cfg).active_creds()
    return [f for f in _CRED_ENV_FIELDS if getattr(active, f)]


@router.get("", response_model=TenantListResponse)
async def list_tenants(
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> TenantListResponse:
    """List every tenant with its mode + selected providers/models (admin)."""
    rows = (await session.execute(select(Tenant).order_by(Tenant.created_at))).scalars().all()
    items = []
    for t in rows:
        pc = t.pipeline_config or {}
        tel = pc.get("telephony") or {}
        items.append(TenantSummary(
            tenant_id=t.id, slug=t.slug, name=t.name, status=t.status,
            mode=t.mode, max_concurrent_calls=t.max_concurrent_calls,
            stt=_layer(pc, "stt"), llm=_layer(pc, "llm"), tts=_layer(pc, "tts"),
            realtime=_layer(pc, "realtime"),
            telephony_provider=tel.get("provider"),
            telephony_from_number=tel.get("from_number"),
            telephony_events_webhook_url=tel.get("events_webhook_url"),
            telephony_events_webhook_secret_env=tel.get("events_webhook_secret_env"),
            telephony_creds_configured=_configured_creds(tel),
        ))
    return TenantListResponse(tenants=items, total=len(items))


async def _require_tenant(session: AsyncSession, tenant_id: str) -> Tenant:
    t = await session.get(Tenant, tenant_id)
    if t is None:
        raise HTTPException(status_code=404, detail="tenant not found")
    return t


class TenantAnalytics(BaseModel):
    tenant_id: str
    total_calls: int
    by_status: dict[str, int]
    by_outcome: dict[str, int]
    # Manual (human softphone) vs AI (voicebot) call counts, keyed by agent_type.
    by_agent_type: dict[str, int]
    total_duration_ms: int
    avg_duration_ms: int


@router.get("/{tenant_id}/analytics", response_model=TenantAnalytics)
async def tenant_analytics(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> TenantAnalytics:
    """Call analytics for one tenant, aggregated from the conversations table."""
    await _require_tenant(session, tenant_id)
    rows = (await session.execute(
        select(Conversation.status, Conversation.outcome, Conversation.duration_ms,
               Conversation.agent_type)
        .where(Conversation.tenant_id == tenant_id)
    )).all()
    by_status: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    by_agent_type: dict[str, int] = {}
    total_dur = 0
    for status, outcome, dur, agent_type in rows:
        by_status[status or "unknown"] = by_status.get(status or "unknown", 0) + 1
        # Count rows with no outcome under "no_outcome" so by_outcome totals to
        # total_calls (matching by_status) — calls in progress or that ended
        # before analysis have no outcome yet.
        okey = outcome or "no_outcome"
        by_outcome[okey] = by_outcome.get(okey, 0) + 1
        akey = agent_type or "unknown"
        by_agent_type[akey] = by_agent_type.get(akey, 0) + 1
        total_dur += int(dur or 0)
    n = len(rows)
    return TenantAnalytics(
        tenant_id=tenant_id, total_calls=n, by_status=by_status, by_outcome=by_outcome,
        by_agent_type=by_agent_type,
        total_duration_ms=total_dur, avg_duration_ms=(total_dur // n if n else 0),
    )


class TenantBilling(BaseModel):
    tenant_id: str
    total_calls: int
    billable_minutes: float
    platform_cost: float                 # what we charge (STT/LLM/TTS or S2S)
    avg_cost_per_call: float
    tentative_telephony_cost: float      # tenant's own telephony — informational only
    currency: str = "USD"


@router.get("/{tenant_id}/billing", response_model=TenantBilling)
async def tenant_billing(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> TenantBilling:
    """Billing summary: platform cost (telephony excluded) + a tentative
    telephony figure computed from the tenant's telephony provider rate."""
    await _require_tenant(session, tenant_id)
    rows = (await session.execute(
        select(Conversation.cost, Conversation.duration_ms, Conversation.telephony_provider)
        .where(Conversation.tenant_id == tenant_id)
    )).all()
    # telephony rates (model="") for the tentative figure
    tel_rates = dict((p, c) for p, c in (await session.execute(
        select(ProviderCost.provider, ProviderCost.cost_per_min)
        .where(ProviderCost.kind == "telephony", ProviderCost.model == "")
    )).all())

    platform = 0.0
    tentative_tel = 0.0
    total_ms = 0
    for cost, dur, tel in rows:
        platform += float(cost or 0.0)
        total_ms += int(dur or 0)
        if tel and dur:
            tentative_tel += tel_rates.get(tel, 0.0) * (int(dur) / 60_000.0)
    n = len(rows)
    return TenantBilling(
        tenant_id=tenant_id, total_calls=n,
        billable_minutes=round(total_ms / 60_000.0, 4),
        platform_cost=round(platform, 6),
        avg_cost_per_call=round(platform / n, 6) if n else 0.0,
        tentative_telephony_cost=round(tentative_tel, 6),
    )
