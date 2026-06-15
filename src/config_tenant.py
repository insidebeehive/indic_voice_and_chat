"""Per-tenant settings loader.

Each tenant has a YAML file at ``config/tenants/<slug>.yaml`` that overlays
the platform defaults. API keys are referenced **by env var name** (never
raw values) so secrets never land in version control or the DB.

Schema (all sections optional — global defaults fill the gaps):

    id: t_acme
    slug: acme
    name: Acme Telecom
    status: active                  # active | suspended
    default_language: hi
    webhook_secret_env: TENANT_ACME_WEBHOOK_SECRET

    pipeline:
      stt:
        provider: sarvam
        model: saaras:v2
        api_key_env: TENANT_ACME_SARVAM_KEY
      llm:
        provider: groq
        api_key_env: TENANT_ACME_GROQ_KEY
      tts:
        provider: sarvam
        voice_id: meera
        api_key_env: TENANT_ACME_SARVAM_KEY
      telephony:
        provider: twilio
        from_number: "+918888888888"
        account_sid_env: TENANT_ACME_TWILIO_SID
        auth_token_env: TENANT_ACME_TWILIO_TOKEN

    compliance:
      calling_hours: {start: "10:00", end: "19:00"}
      dnd_check_enabled: true

    crm:
      kind: fake | salesforce | hubspot
      endpoint_env: TENANT_ACME_CRM_URL
      token_env: TENANT_ACME_CRM_TOKEN

    whatsapp:
      provider: fake | meta_cloud
      phone_id_env: TENANT_ACME_WA_PHONE_ID
      token_env: TENANT_ACME_WA_TOKEN

    phone_numbers:           # Twilio numbers tied to this tenant
      - "+918888888888"
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class MissingEnvError(RuntimeError):
    """Raised when a tenant YAML references an env var that isn't set."""


class TenantConfigError(ValueError):
    """Raised when a tenant config is structurally incomplete.

    The most common case: a provider is declared but its credential env-var
    name is missing, which would silently fall back to the global platform
    key at runtime. We raise this at load time so misconfigured tenants are
    caught on bootstrap rather than billing the platform.
    """


# --- Sub-schemas --------------------------------------------------------


class TenantSTTConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    confidence_threshold: Optional[float] = None
    api_key_env: Optional[str] = None


class TenantStreamingSTTConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    endpointing: Optional[int] = None
    utterance_end_ms: Optional[int] = None
    api_key_env: Optional[str] = None


class TenantRealtimeConfig(BaseModel):
    """Speech-to-speech (audio-in/audio-out) provider, e.g. Gemini Live. Active
    only when pipeline.mode == 's2s' (or the dev console's S2S path is used)."""
    provider: Optional[str] = None          # e.g. "gemini_live"
    model: Optional[str] = None             # e.g. "gemini-3.1-flash-live-preview"
    voice: Optional[str] = None             # default prebuilt voice
    allowed_voices: list[str] = Field(default_factory=list)
    language_code: Optional[str] = None     # e.g. "hi-IN"
    api_key_env: Optional[str] = None


class TenantLLMConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None
    api_key_env: Optional[str] = None


class TenantTTSConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    voice_id: Optional[str] = None
    speed: Optional[float] = None
    api_key_env: Optional[str] = None


class TenantTelephonyConfig(BaseModel):
    provider: Optional[str] = None
    from_number: Optional[str] = None
    webhook_base_url: Optional[str] = None
    account_sid_env: Optional[str] = None
    auth_token_env: Optional[str] = None
    # Raw SIP trunk (e.g. DiDLogic) host, e.g. "sip.didlogic.com". The trunk auth
    # user/password reuse account_sid_env / auth_token_env (the per-tenant secrets).
    sip_server: Optional[str] = None
    # Per-provider caller-IDs for the dev-console "place call" panel. The Telephony
    # dropdown picks the provider; this maps provider -> the number to dial *from*
    # (each provider needs its own owned number). e.g. {"twilio": "+1...", "exotel": "+91..."}.
    outbound_from: dict[str, str] = Field(default_factory=dict)


class TenantVectorStoreConfig(BaseModel):
    provider: Optional[str] = None
    index_path: Optional[str] = None       # auto-namespaced if unset
    embedding_dim: Optional[int] = None


class TenantPipelineConfig(BaseModel):
    # "layered" = the STT->LLM->TTS cascade (default); "s2s" = end-to-end
    # speech-to-speech via pipeline.realtime (e.g. Gemini Live).
    mode: Literal["layered", "s2s"] = "layered"
    stt: TenantSTTConfig = Field(default_factory=TenantSTTConfig)
    stt_streaming: Optional[TenantStreamingSTTConfig] = None
    realtime: Optional[TenantRealtimeConfig] = None
    llm: TenantLLMConfig = Field(default_factory=TenantLLMConfig)
    tts: TenantTTSConfig = Field(default_factory=TenantTTSConfig)
    telephony: TenantTelephonyConfig = Field(default_factory=TenantTelephonyConfig)
    vector_store: TenantVectorStoreConfig = Field(default_factory=TenantVectorStoreConfig)


class TenantCompliance(BaseModel):
    calling_hours_start: Optional[str] = None
    calling_hours_end: Optional[str] = None
    dnd_check_enabled: Optional[bool] = None
    ai_disclosure: Optional[bool] = None
    max_retry_attempts: Optional[int] = None
    retry_interval_hours: Optional[int] = None


class TenantCRMConfig(BaseModel):
    kind: str = "fake"
    endpoint_env: Optional[str] = None
    token_env: Optional[str] = None


class TenantWhatsAppConfig(BaseModel):
    provider: str = "fake"
    phone_id_env: Optional[str] = None
    token_env: Optional[str] = None


class TenantSettings(BaseModel):
    """Validated tenant configuration loaded from YAML."""

    id: str = Field(min_length=1)
    slug: str = Field(min_length=1, max_length=63)
    name: str = Field(min_length=1)
    status: str = "active"
    default_language: str = "hi"
    timezone: str = "Asia/Kolkata"  # IANA tz; resolves relative callback times
    max_concurrent_calls: int = 1   # per-tenant cap on simultaneous live calls
    webhook_secret_env: Optional[str] = None

    pipeline: TenantPipelineConfig = Field(default_factory=TenantPipelineConfig)
    compliance: TenantCompliance = Field(default_factory=TenantCompliance)
    crm: TenantCRMConfig = Field(default_factory=TenantCRMConfig)
    whatsapp: TenantWhatsAppConfig = Field(default_factory=TenantWhatsAppConfig)
    phone_numbers: list[str] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"invalid IANA timezone: {v!r}") from e
        return v

    # --- secret resolution ---------------------------------------------

    def secret(self, env_var: Optional[str]) -> Optional[str]:
        """Resolve a referenced env var name. Returns None if name is None."""
        if env_var is None:
            return None
        value = os.environ.get(env_var)
        if value is None:
            raise MissingEnvError(
                f"tenant {self.slug!r} references env var {env_var!r} which is not set"
            )
        return value


# --- Loader -------------------------------------------------------------


_TENANT_DIR_DEFAULT = Path("config/tenants")


def _resolve_dir(tenant_dir: Optional[Path]) -> Path:
    return Path(tenant_dir or os.environ.get("VOX_TENANT_DIR") or _TENANT_DIR_DEFAULT)


def load_tenant(slug: str, tenant_dir: Optional[Path] = None) -> TenantSettings:
    """Load + validate one tenant by slug.

    Validation runs in two stages:
    1. Pydantic schema check (types, required fields).
    2. ``validate_credentials`` — every declared provider must also declare
       its credential env-var names so we never silently fall back to a
       platform-wide key at runtime.
    """
    base = _resolve_dir(tenant_dir)
    path = base / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"tenant config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    try:
        settings = TenantSettings(**data)
    except ValidationError as e:
        raise ValueError(f"{path}: invalid tenant config: {e}") from e
    validate_credentials(settings, source=str(path))
    return settings


# --- Credential validation ----------------------------------------------


def validate_credentials(settings: TenantSettings, *, source: str = "") -> None:
    """Ensure every declared provider also declares its credential env vars.

    A tenant *may* omit a layer entirely (e.g. chat-only tenants don't need
    telephony). But if ``provider`` is set, the env-var fields that supply
    that provider's credentials must also be set. This closes the
    silent-fallback footgun where an under-configured tenant would bill the
    platform's API keys.

    Raises ``TenantConfigError`` listing all gaps so admins fix them in one
    round-trip rather than chasing missing fields one at a time.
    """
    gaps: list[str] = []
    p = settings.pipeline

    if p.stt.provider and not p.stt.api_key_env:
        gaps.append(f"pipeline.stt.api_key_env (provider={p.stt.provider!r})")
    if p.llm.provider and not p.llm.api_key_env:
        gaps.append(f"pipeline.llm.api_key_env (provider={p.llm.provider!r})")
    if p.tts.provider and not p.tts.api_key_env:
        gaps.append(f"pipeline.tts.api_key_env (provider={p.tts.provider!r})")
    if p.telephony.provider:
        if not p.telephony.account_sid_env:
            gaps.append(
                f"pipeline.telephony.account_sid_env (provider={p.telephony.provider!r})"
            )
        if not p.telephony.auth_token_env:
            gaps.append(
                f"pipeline.telephony.auth_token_env (provider={p.telephony.provider!r})"
            )
    if p.realtime and p.realtime.provider and not p.realtime.api_key_env:
        gaps.append(f"pipeline.realtime.api_key_env (provider={p.realtime.provider!r})")
    if p.mode == "s2s" and not (p.realtime and p.realtime.provider):
        gaps.append("pipeline.realtime (provider) — required when pipeline.mode == 's2s'")

    if gaps:
        prefix = f"{source}: " if source else ""
        raise TenantConfigError(
            f"{prefix}tenant {settings.slug!r} is missing credential env-var "
            f"references for declared providers: {gaps}. Without these, the "
            f"adapter falls back to the platform-wide env vars at runtime, "
            f"which silently bills the platform for the tenant's calls."
        )


def discover_tenant_slugs(tenant_dir: Optional[Path] = None) -> list[str]:
    """Return slugs for every YAML in the tenant config directory."""
    base = _resolve_dir(tenant_dir)
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))


def load_all_tenants(tenant_dir: Optional[Path] = None) -> dict[str, TenantSettings]:
    """Load every tenant in ``config/tenants/``. Returns ``{slug: settings}``."""
    return {slug: load_tenant(slug, tenant_dir) for slug in discover_tenant_slugs(tenant_dir)}


# --- Merge with global defaults ----------------------------------------


def merge_provider_config(
    tenant_layer: BaseModel,
    global_layer: dict[str, Any],
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Overlay tenant-set fields onto the global default dict.

    Only non-None tenant fields override globals — that's the "partial
    override" semantics promised in the plan.
    """
    out = dict(global_layer)
    for k, v in tenant_layer.model_dump().items():
        if v is not None and not k.endswith("_env"):
            out[k] = v
    if api_key is not None:
        out["api_key"] = api_key
    return out
