"""Per-tenant settings loader.

Each tenant has a YAML file at ``config/tenants/<slug>.yaml`` that overlays
the platform defaults. Telephony credentials are referenced **by env var
name** (never raw values) so secrets never land in version control or the DB.

Schema (all sections optional — global defaults fill the gaps):

    id: t_acme
    slug: acme
    name: Acme Telecom
    status: active                  # active | suspended
    default_language: hi

    pipeline:
      stt:
        provider: sarvam
        model: saaras:v3
        # api_key_env is optional/informational here: STT always resolves
        # its key from the platform master env var (e.g. SARVAM_API_KEY),
        # never from a per-tenant value — never consulted at runtime.
      llm:
        provider: groq
        # same as above: LLM always uses the platform master key.
      tts:
        provider: sarvam
        voice_id: meera
        # same as above: TTS always uses the platform master key.
      chat_voice:                     # CHAT voice-note replies only (not calls)
        enabled: false                # opt-in; TTS is billed per reply
        tts:                          # optional — falls back to pipeline.tts
          provider: sarvam
          voice_id: priya
      telephony:
        provider: twilio
        from_number: "+918888888888"
        # Telephony IS genuinely per-tenant — these are the real, required
        # credential references (validate_credentials() enforces them).
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

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger(__name__)

# debug_event (src/utils/logging.py) and token_fingerprint (src/auth/audit.py)
# are imported LAZILY inside each function that needs them, not at module
# level: src.utils.logging imports src.auth.audit, which (via the src.auth
# package's own __init__) drags in src.auth.context, which imports
# TenantSettings from THIS module -- a real circular import, not a
# hypothetical one (verified: a module-level import here raises "cannot
# import name 'TenantSettings' from partially initialized module
# 'src.config_tenant'"). Same reason platform_webhook_base_url() below
# already lazy-imports src.config and resolve_livekit_creds() lazy-imports
# sqlalchemy/src.auth.secrets/src.models.crm.

if TYPE_CHECKING:  # pragma: no cover - type-checking only, avoids import cycles at runtime
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.auth.context import TenantContext


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


class ChatVoiceConfig(BaseModel):
    """Voice-note replies in CHAT — deliberately separate from the voice-call
    cascade's ``pipeline.tts``.

    ``enabled`` is opt-in and defaults to False on purpose: TTS is billed per
    character/second by every provider wired here, on every synthesized reply.
    A tenant that never asked for spoken chat replies must not start incurring
    that cost merely because one of its customers sent a voice note. Off means
    the customer still gets the full TEXT answer — nothing is lost but the
    audio half (see ``_synthesize_reply_audio`` in src/api/chat.py).

    ``tts`` is the chat-specific TTS block. It exists because an s2s tenant
    (pipeline.mode == "s2s", calls handled end-to-end by pipeline.realtime)
    legitimately has NO ``pipeline.tts`` at all — validate_credentials has
    never required one. Resolving chat replies through ``pipeline.tts`` for
    such a tenant fell all the way through to the PLATFORM default in
    config/default.yaml (sarvam) and the platform's own SARVAM_API_KEY,
    silently billing the platform for that tenant's chat audio. Leave this
    empty and the tenant's own ``pipeline.tts`` is reused instead (the sane
    default for a layered tenant that already pays for a cascade voice); leave
    both empty and there is no chat TTS at all — see
    ``resolve_chat_tts_config``.
    """
    enabled: bool = False
    tts: TenantTTSConfig = Field(default_factory=TenantTTSConfig)


class TelephonyCreds(BaseModel):
    """The set of credential *_env references for one telephony provider.

    Each field names a (synthetic) env var whose value lives encrypted in
    tenant_secrets and resolves via TenantContext.secret. ``account_sid_env`` /
    ``auth_token_env`` are the account credentials (Twilio account SID/token,
    Stringee API Key SID/Secret); the ``api_key_*`` / ``twiml_app_sid_env`` are
    Twilio browser-softphone-only extras.
    """
    account_sid_env: Optional[str] = None
    auth_token_env: Optional[str] = None
    api_key_sid_env: Optional[str] = None
    api_key_secret_env: Optional[str] = None
    twiml_app_sid_env: Optional[str] = None
    # Stringee-only: the originating app user id for outbound callouts. A non-null
    # ``userId`` makes the callout an app-user->phone call (so Stringee fetches the
    # Answer URL and runs the SCCO); without it the call degrades to phone->phone
    # external and the bot never speaks. Names a secret resolved via .secret().
    user_id_env: Optional[str] = None


# Providers with no telephony adapter and no per-tenant account: no adapter is
# ever built for them as the CONFIGURED provider, so a missing
# account_sid_env/auth_token_env has nothing to silently fall back from and the
# platform-billing risk validate_credentials guards against does not arise on
# that path. Narrowly scoped on purpose: the top-level cred fields are not read
# only for the configured provider — creds_for() takes an arbitrary provider and
# falls through to them (src/api/dev_console.py passes a request-supplied one),
# so an exempt tenant that also sets pipeline.telephony.outbound_from.<other>
# can still reach a real adapter with unset creds. Case-insensitive match
# against provider, consistent with how src/api/calls.py and creds_for()
# normalize the provider string.
#
# "none" is what POST /tenants writes for a tenant registered without telephony
# (RegisterTenantRequest's default in src/api/tenants.py). It is not in
# TELEPHONY_PROVIDERS either, so building an adapter for it raises
# UnknownProviderError rather than falling back to platform credentials.
CREDENTIAL_FREE_TELEPHONY_PROVIDERS = frozenset({"webconsole", "none"})


class TenantTelephonyConfig(BaseModel):
    provider: Optional[str] = None
    from_number: Optional[str] = None
    # Stringee REST API base for this tenant's project. Stringee projects live on
    # regional hosts (e.g. https://asia-2.api.stringee.com); the global
    # api.stringee.com rejects a regional keySid with r:5 "keySid invalid". Used
    # for the recording download (and passed to the adapter for the callout).
    # Default unset → the adapter/download use api.stringee.com.
    stringee_base_url: Optional[str] = None
    # LiveKit server/project URL for this tenant's room-join bridge (wss://...).
    # Non-secret — the API key/secret pair lives alongside it in
    # creds_by_provider["livekit"] via the existing account_sid_env/auth_token_env
    # fields (TelephonyCreds is provider-agnostic; no new secret fields needed).
    livekit_url: Optional[str] = None
    # NB: there is no per-tenant inbound webhook base URL — it's always *our* app,
    # common to every tenant, so it lives at the platform level (WEBHOOK_BASE_URL /
    # settings.pipeline.telephony.webhook_base_url, see ``platform_webhook_base_url``).
    # Top-level credential refs (legacy / single-provider tenants). Kept for
    # backward compatibility; when ``creds_by_provider`` has an entry for the
    # active provider it takes precedence (see ``active_creds``).
    account_sid_env: Optional[str] = None
    auth_token_env: Optional[str] = None
    api_key_sid_env: Optional[str] = None
    api_key_secret_env: Optional[str] = None
    twiml_app_sid_env: Optional[str] = None
    user_id_env: Optional[str] = None
    # Per-provider credential slots, so a tenant that has keys for more than one
    # provider (e.g. Twilio + Stringee) resolves the set matching its configured
    # ``provider`` — not whichever was registered first. {provider: TelephonyCreds}.
    creds_by_provider: dict[str, TelephonyCreds] = Field(default_factory=dict)
    # Per-provider caller-IDs for the dev-console "place call" panel. The Telephony
    # dropdown picks the provider; this maps provider -> the number to dial *from*
    # (each provider needs its own owned number). e.g. {"twilio": "+1...", "exotel": "+91..."}.
    outbound_from: dict[str, str] = Field(default_factory=dict)

    def creds_for(self, provider: Optional[str]) -> "TelephonyCreds":
        """Credential refs for ``provider`` — its per-provider slot if present,
        else the top-level fields (back-compat for single-provider tenants)."""
        key = (provider or self.provider or "").lower()
        slot = self.creds_by_provider.get(key)
        if slot is not None:
            return slot
        return TelephonyCreds(
            account_sid_env=self.account_sid_env,
            auth_token_env=self.auth_token_env,
            api_key_sid_env=self.api_key_sid_env,
            api_key_secret_env=self.api_key_secret_env,
            twiml_app_sid_env=self.twiml_app_sid_env,
            user_id_env=self.user_id_env,
        )

    def active_creds(self) -> "TelephonyCreds":
        """Credential refs for the tenant's *configured* telephony provider."""
        return self.creds_for(self.provider)


def _tenant_livekit_creds(tel: TenantTelephonyConfig) -> TelephonyCreds:
    """The tenant-level LiveKit credential slot — explicitly, NOT via
    ``TelephonyCreds.creds_for()``'s back-compat fallback to the top-level
    ``account_sid_env``/``auth_token_env`` fields.

    Those top-level fields are always populated with the tenant's ACTIVE
    telephony provider's creds (Twilio/Stringee/etc. — see
    ``src/api/tenants.py``'s ``register_tenant``/``update_tenant``), for
    single-provider back-compat. Falling back to them here would silently
    hand a telephony provider's account secret to the LiveKit SDK for any
    tenant that has an active non-LiveKit provider AND a bare
    ``livekit_url`` set without ever registering real LiveKit creds —
    exactly the "tenant partially configured" case that must instead fall
    through cleanly to the CRM-level project, not half-resolve with the
    wrong provider's secret.
    """
    from src.utils.logging import debug_event

    if "livekit" in tel.creds_by_provider:
        debug_event(log, "tenant_config livekit_creds_slot decision", source="creds_by_provider")
        return tel.creds_by_provider["livekit"]
    if (tel.provider or "").lower() == "livekit":
        debug_event(log, "tenant_config livekit_creds_slot decision", source="active_provider")
        return tel.active_creds()
    # Deliberately empty, NOT tel.active_creds() -- see the docstring above:
    # falling back to the top-level account_sid_env/auth_token_env here would
    # hand a different telephony provider's account secret to the LiveKit SDK.
    debug_event(
        log, "tenant_config livekit_creds_slot decision", source="none",
        active_telephony_provider=tel.provider,
    )
    return TelephonyCreds()


async def resolve_livekit_creds(
    session: Optional[AsyncSession], tenant: TenantContext
) -> Optional[tuple[str, str, str]]:
    """Returns ``(url, api_key, api_secret)``, or ``None`` if nothing usable
    is configured.

    Precedence: a tenant-specific LiveKit project (rare — set directly on the
    tenant's own telephony config) wins if fully configured; otherwise falls
    back to the CRM-level project shared by every tenant registered under that
    CRM (the common case — one LiveKit project per CRM partner, not per
    tenant). A live async DB lookup at call time, not something pre-loaded
    into ``TenantContext`` at auth-resolve time — deliberately kept out of
    ``src/auth/db_resolver.py``'s hot path. ``session=None`` is accepted for
    DB-less callers (no ``sessionmaker`` available): the tenant-level branch
    below never touches it, so that path still resolves; the CRM-level
    fallback is simply unavailable in that case.
    """
    from src.utils.logging import debug_event

    tel = tenant.settings.pipeline.telephony
    creds = _tenant_livekit_creds(tel)
    url = tel.livekit_url
    api_key = tenant.secret_optional(creds.account_sid_env)
    api_secret = tenant.secret_optional(creds.auth_token_env)
    # Presence/length only for api_key/api_secret -- both are the LiveKit
    # project's real credentials (reused via the telephony account_sid_env/
    # auth_token_env slots, see _tenant_livekit_creds) and never appear at
    # any level. This is the full precedence chain the docstring describes:
    # an operator debugging "why did this tenant's LiveKit room join fail"
    # can see here whether it was the tenant-level project (and which part
    # was missing) before it falls through to the CRM-level one below.
    if url and api_key and api_secret:
        debug_event(
            log, "tenant_config livekit_creds_resolve result", tier="tenant",
            tenant_id=tenant.id, url=url,
            api_key_len=len(api_key), api_secret_len=len(api_secret),
        )
        return url, api_key, api_secret
    debug_event(
        log, "tenant_config livekit_creds_resolve tenant_tier_incomplete",
        tenant_id=tenant.id, url_set=bool(url), api_key_set=bool(api_key),
        api_secret_set=bool(api_secret),
    )

    crm_id = tenant.settings.crm_id
    if not crm_id or session is None:
        debug_event(
            log, "tenant_config livekit_creds_resolve no_crm_fallback",
            tenant_id=tenant.id, crm_id=crm_id, session_available=session is not None,
        )
        return None

    from sqlalchemy import select

    from src.auth import secrets as crypto
    from src.models.crm import Crm, CrmSecret

    crm = await session.get(Crm, crm_id)
    if crm is None or not crm.livekit_url:
        debug_event(
            log, "tenant_config livekit_creds_resolve crm_tier_unusable",
            tenant_id=tenant.id, crm_id=crm_id, crm_found=crm is not None,
            crm_livekit_url_set=bool(crm.livekit_url) if crm is not None else None,
        )
        return None

    rows = (await session.execute(
        select(CrmSecret).where(
            CrmSecret.crm_id == crm_id,
            CrmSecret.name.in_(("livekit_api_key", "livekit_api_secret")),
        )
    )).scalars().all()
    by_name = {r.name: r.value_encrypted for r in rows}
    if "livekit_api_key" not in by_name or "livekit_api_secret" not in by_name:
        debug_event(
            log, "tenant_config livekit_creds_resolve crm_tier_secrets_missing",
            tenant_id=tenant.id, crm_id=crm_id,
            secret_names_found=sorted(by_name.keys()),
        )
        return None

    debug_event(
        log, "tenant_config livekit_creds_resolve result", tier="crm",
        tenant_id=tenant.id, crm_id=crm_id, url=crm.livekit_url,
    )
    return (
        crm.livekit_url,
        crypto.decrypt(by_name["livekit_api_key"]),
        crypto.decrypt(by_name["livekit_api_secret"]),
    )


def platform_webhook_base_url() -> Optional[str]:
    """The platform-level telephony webhook base URL (WEBHOOK_BASE_URL env →
    settings.pipeline.telephony.webhook_base_url). It only seeds the OUTBOUND
    callout's Answer URL and is always *our* app — common to every tenant (inbound
    is host-derived + number-resolved, so it never reads this). Not per-tenant."""
    from src.config import get_settings  # lazy: avoid import cycle at module load
    return get_settings().pipeline.telephony.webhook_base_url


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
    chat_voice: ChatVoiceConfig = Field(default_factory=ChatVoiceConfig)


def resolve_chat_tts_config(pipeline: TenantPipelineConfig) -> Optional[TenantTTSConfig]:
    """The EFFECTIVE TTS config for chat voice-note replies, or None.

    Precedence:
      1. ``pipeline.chat_voice.tts`` when it declares a provider — the chat
         override.
      2. else ``pipeline.tts`` when IT declares a provider — a layered tenant
         reusing the cascade voice it already configured and pays for.
      3. else None — nothing resolvable. Callers MUST NOT fall through to the
         platform default here: that is the silent-platform-billing bug this
         function exists to prevent (an s2s tenant has no pipeline.tts at all).

    ``provider`` is the discriminator, not "any field set": a block with only
    e.g. ``voice_id`` and no provider cannot build a client, so it falls
    through rather than half-resolving.

    Deliberately ignores ``chat_voice.enabled`` — this answers "what WOULD be
    used", which is what both callers need: the registry checks ``enabled``
    separately, and ``validate_credentials`` must flag the enabled-but-
    unresolvable combination.
    """
    from src.utils.logging import debug_event

    if pipeline.chat_voice.tts.provider:
        debug_event(
            log, "tenant_config chat_tts_resolve decision", source="chat_voice_override",
            provider=pipeline.chat_voice.tts.provider, model=pipeline.chat_voice.tts.model,
            voice_id=pipeline.chat_voice.tts.voice_id,
            pipeline_tts_provider=pipeline.tts.provider,
        )
        return pipeline.chat_voice.tts
    if pipeline.tts.provider:
        # The incident this function exists to prevent a repeat of: a tenant
        # relying on this fallback looks IDENTICAL to one that configured its
        # own chat voice, from every angle except this log line -- see
        # ChatVoiceConfig's docstring and TenantSummary.chat_voice's "source"
        # field (src/api/tenants.py), which surfaces the same distinction to
        # the backoffice.
        debug_event(
            log, "tenant_config chat_tts_resolve decision", source="pipeline_cascade_fallback",
            provider=pipeline.tts.provider, model=pipeline.tts.model,
            voice_id=pipeline.tts.voice_id,
            chat_voice_tts_provider=pipeline.chat_voice.tts.provider,
        )
        return pipeline.tts
    debug_event(
        log, "tenant_config chat_tts_resolve decision", source="none",
        chat_voice_tts_provider=pipeline.chat_voice.tts.provider,
        pipeline_tts_provider=pipeline.tts.provider,
    )
    return None


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
    # The CRM system's operator identifier for this tenant. Injected into every
    # CRM tool call as the "operator_id" session context param so the CRM can
    # scope API responses to the right operator. Defaults to the tenant's own id.
    operator_id: Optional[str] = None


class TenantWhatsAppConfig(BaseModel):
    provider: str = "fake"
    phone_id_env: Optional[str] = None
    token_env: Optional[str] = None


class ChatSupportConfig(BaseModel):
    """BO (back-office) handover settings for the chat module."""
    support_timezone: str = "Asia/Kolkata"
    # Keyed by day-range string ("mon-fri", "sat", "sun"); value = "HH:MM-HH:MM".
    # Absent key = closed that day. Empty dict = check disabled (always available).
    support_hours: dict = Field(default_factory=dict)
    # Seconds of customer silence before the AI chat auto-closes and fires
    # session_closed to the CRM. 0 disables the timeout entirely.
    chat_idle_timeout_seconds: int = 300


class DepositVerificationConfig(BaseModel):
    """Deposit dispute screenshot verification webhook settings."""
    enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_secret_env: Optional[str] = None
    # gt=0: 0 or negative would either fire the timeout immediately (0) or
    # never mark a stalled request due (negative), and PATCH /tenants/{id}
    # has no other guard against either.
    timeout_minutes: int = Field(default=5, gt=0)
    # Vendor API contract this tenant's webhook speaks. "multipart_verdict" is
    # the original/default contract (src/chatbot/deposit_verification.py +
    # src/api/deposit_verification.py); "json_ticket_relay" is a newer
    # vendor's JSON-ticket contract, added additively — default preserves all
    # existing behavior unchanged.
    contract: Literal["multipart_verdict", "json_ticket_relay"] = "multipart_verdict"
    # gt=0: a signed screenshot URL with a zero/negative TTL would be
    # rejected or already-expired the moment the vendor tries to fetch it.
    screenshot_url_ttl_seconds: int = Field(default=3600, gt=0)
    mobile_metadata_keys: list[str] = Field(default_factory=lambda: ["mobile", "phone"])


class TenantSettings(BaseModel):
    """Validated tenant configuration loaded from YAML."""

    id: str = Field(min_length=1)
    slug: str = Field(min_length=1, max_length=63)
    name: str = Field(min_length=1)
    status: str = "active"
    default_language: str = "hi"
    timezone: str = "Asia/Kolkata"  # IANA tz; resolves relative callback times
    max_concurrent_calls: int = 1   # per-tenant cap on simultaneous live calls

    # Outbound event webhook — receives all lifecycle events (calls, chat handovers).
    # Signed with HMAC-SHA256 when events_webhook_secret_env is set.
    events_webhook_url: Optional[str] = None
    events_webhook_secret_env: Optional[str] = None
    pipeline: TenantPipelineConfig = Field(default_factory=TenantPipelineConfig)
    compliance: TenantCompliance = Field(default_factory=TenantCompliance)
    crm: TenantCRMConfig = Field(default_factory=TenantCRMConfig)
    # The Crm entity (src.models.crm.Crm) this tenant is linked to — its id, not
    # the per-tenant crm sub-config above. Drives resolve_crm_tools()'s tier-2
    # DB-backed tool catalog fallback (src/bootstrap.py).
    crm_id: Optional[str] = None
    # ChatBot system-prompt pack (src/dialogue/packs) denormalized from the
    # linked Crm row's prompt_pack at tenant-resolution time (src/auth/db_resolver.py) —
    # "generic" when unset/NULL or when the tenant has no linked CRM at all.
    prompt_pack: str = "generic"
    # TTS pronunciation overrides denormalized from the linked Crm row's
    # pronunciation_overrides at tenant-resolution time (src/auth/db_resolver.py) —
    # merged over src.pipeline.text_normalize.DEFAULT_PRONUNCIATIONS at
    # synthesis time (src/bootstrap.py). None when unset/NULL or when the
    # tenant has no linked CRM at all -- that CRM simply gets the generic
    # default with no extra terms.
    pronunciation_overrides: Optional[dict[str, str]] = None
    whatsapp: TenantWhatsAppConfig = Field(default_factory=TenantWhatsAppConfig)
    chat_support: ChatSupportConfig = Field(default_factory=ChatSupportConfig)
    deposit_verification: DepositVerificationConfig = Field(default_factory=DepositVerificationConfig)
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
        # Boundary: every telephony/webhook/LiveKit credential a tenant uses
        # resolves through here. The VALUE is a credential and never appears
        # at any level (docs/debug-logging.md) -- only the env var NAME (a
        # reference, not a secret) plus a fingerprint + length, so an
        # operator can confirm "yes, this tenant's account_sid_env resolved,
        # and it's the same 40-char value as last week" without the value
        # ever leaving os.environ.
        from src.auth.audit import token_fingerprint
        from src.utils.logging import debug_event
        debug_event(
            log, "tenant_config secret resolved",
            tenant_slug=self.slug, env_var=env_var,
            value_fp=token_fingerprint(value, domain="vox-logfp-tenant-secret-v1"),
            value_len=len(value),
        )
        return value


# --- Loader -------------------------------------------------------------


_TENANT_DIR_DEFAULT = Path("config/tenants")


def _resolve_dir(tenant_dir: Optional[Path]) -> Path:
    return Path(tenant_dir or os.environ.get("VOX_TENANT_DIR") or _TENANT_DIR_DEFAULT)


# {dotted path into the parsed YAML: model class} for the sub-blocks worth
# checking for unknown keys -- deliberately not a full recursive walk of
# every nested model (this only runs at tenant load time, but a typo three
# levels deep is rare and the maintenance cost of mirroring the whole schema
# here isn't worth it). These are the blocks past incidents actually hit: a
# misspelled field under pipeline.<layer> resolves to that layer's default
# with NO error, exactly like config/default.yaml's inert `tts.model` (see
# docs/debug-logging.md's coverage note on this file).
_UNKNOWN_KEY_CHECK_PATHS: dict[str, type[BaseModel]] = {
    "pipeline.stt": TenantSTTConfig, "pipeline.llm": TenantLLMConfig,
    "pipeline.tts": TenantTTSConfig, "pipeline.telephony": TenantTelephonyConfig,
    "pipeline.chat_voice": ChatVoiceConfig, "pipeline.realtime": TenantRealtimeConfig,
    "pipeline.vector_store": TenantVectorStoreConfig,
    "crm": TenantCRMConfig, "compliance": TenantCompliance,
    "whatsapp": TenantWhatsAppConfig, "chat_support": ChatSupportConfig,
    "deposit_verification": DepositVerificationConfig,
}


def _unknown_keys(data: dict) -> dict[str, list[str]]:
    """{dotted path: [unknown key, ...]} for every checked sub-block, plus the
    top-level itself. Empty dict when nothing is unrecognized. Pydantic's
    default ``extra="ignore"`` (verified: TenantSTTConfig(provider=...,
    bogus_key=...) constructs cleanly with no trace of ``bogus_key``
    afterwards) means a typo here is INDISTINGUISHABLE from a value that was
    never set, once the model exists -- this has to run on the raw dict,
    before TenantSettings(**data) discards the evidence.
    """
    out: dict[str, list[str]] = {}
    top_unknown = [k for k in data if k not in TenantSettings.model_fields]
    if top_unknown:
        out["<top-level>"] = sorted(top_unknown)
    pipeline = data.get("pipeline") or {}
    for dotted, cls in _UNKNOWN_KEY_CHECK_PATHS.items():
        prefix, _, key = dotted.partition(".")
        block = (pipeline if prefix == "pipeline" else data).get(key or prefix)
        if not isinstance(block, dict):
            continue
        unknown = [k for k in block if k not in cls.model_fields]
        if unknown:
            out[dotted] = sorted(unknown)
    return out


def load_tenant(slug: str, tenant_dir: Optional[Path] = None) -> TenantSettings:
    """Load + validate one tenant by slug.

    Validation runs in two stages:
    1. Pydantic schema check (types, required fields).
    2. ``validate_credentials`` — every declared provider must also declare
       its credential env-var names so we never silently fall back to a
       platform-wide key at runtime.
    """
    from src.utils.logging import debug_event

    base = _resolve_dir(tenant_dir)
    path = base / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"tenant config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    if log.isEnabledFor(logging.DEBUG):
        # Guarded: walks the whole parsed dict against every checked
        # sub-model's field set, once per tenant per load -- load_tenant runs
        # at bootstrap and on every backoffice-triggered reload, not per
        # request, but the walk itself is real work, not a scalar read.
        unknown = _unknown_keys(data)
        if unknown:
            debug_event(
                log, "tenant_config load unknown_keys_discarded",
                tenant_slug=slug, source=str(path), unknown_by_block=unknown,
            )
    try:
        settings = TenantSettings(**data)
    except ValidationError as e:
        raise ValueError(f"{path}: invalid tenant config: {e}") from e
    validate_credentials(settings, source=str(path))
    debug_event(
        log, "tenant_config load result", tenant_slug=slug, tenant_id=settings.id,
        source=str(path), pipeline_mode=settings.pipeline.mode,
        stt_provider=settings.pipeline.stt.provider,
        llm_provider=settings.pipeline.llm.provider,
        tts_provider=settings.pipeline.tts.provider,
        telephony_provider=settings.pipeline.telephony.provider,
        crm_id=settings.crm_id, prompt_pack=settings.prompt_pack,
    )
    return settings


# --- Credential validation ----------------------------------------------


def validate_credentials(settings: TenantSettings, *, source: str = "") -> None:
    """Ensure telephony providers declare their credential env vars.

    Telephony genuinely is per-tenant: if ``pipeline.telephony.provider`` is
    set, ``account_sid_env``/``auth_token_env`` must also be set, or calls
    would silently fall back to the platform's own telephony credentials
    (billing/placing calls on the wrong account). We raise this at load time
    so misconfigured tenants are caught on bootstrap rather than at
    first-call. Providers in ``CREDENTIAL_FREE_TELEPHONY_PROVIDERS`` (e.g.
    ``webconsole``) are exempt from this one check: they have no telephony
    adapter and no per-tenant account, so there is nothing to fall back from
    and the platform-billing risk cannot arise. Every other rule below still
    applies to them.

    STT/LLM/TTS/realtime/stt_streaming are different: those adapters
    *always* resolve their API key from the platform-level master env var
    (e.g. ``GEMINI_API_KEY``), never from a per-tenant ``api_key_env``. So
    ``api_key_env`` on those layers is optional and purely informational —
    it is not consulted at runtime and is not required here. ``pipeline.chat_voice``
    is validated for a resolvable **provider** (not a key) only when ``enabled``
    is true, for the same platform-billing reason telephony is validated.

    Raises ``TenantConfigError`` listing all gaps so admins fix them in one
    round-trip rather than chasing missing fields one at a time.
    """
    gaps: list[str] = []
    p = settings.pipeline

    if p.telephony.provider and (p.telephony.provider or "").lower() not in CREDENTIAL_FREE_TELEPHONY_PROVIDERS:
        if not p.telephony.account_sid_env:
            gaps.append(
                f"pipeline.telephony.account_sid_env (provider={p.telephony.provider!r})"
            )
        if not p.telephony.auth_token_env:
            gaps.append(
                f"pipeline.telephony.auth_token_env (provider={p.telephony.provider!r})"
            )
    if p.mode == "s2s" and not (p.realtime and p.realtime.provider):
        gaps.append("pipeline.realtime (provider) — required when pipeline.mode == 's2s'")

    if p.chat_voice.enabled and resolve_chat_tts_config(p) is None:
        gaps.append(
            "pipeline.chat_voice.tts (provider) — required when "
            "pipeline.chat_voice.enabled is true and pipeline.tts declares no "
            "provider to fall back to"
        )

    if gaps:
        prefix = f"{source}: " if source else ""
        raise TenantConfigError(
            f"{prefix}tenant {settings.slug!r} is missing credential env-var "
            f"references for declared providers: {gaps}. Without these, the "
            f"adapter falls back to the platform-wide env vars at runtime, "
            f"which silently bills the platform for the tenant's calls."
        )
    # No debug_event on the failure branch above: TenantConfigError's own
    # message already carries every gap, and every caller either re-raises it
    # (load_tenant) or turns it into a 422 naming the gaps (update_tenant,
    # src/api/tenants.py) -- a second copy here would just be the same
    # string twice. The success case has NO trace anywhere otherwise, so an
    # operator can't tell "validation never ran for this tenant" apart from
    # "it ran and passed" -- this line is that distinction.
    from src.utils.logging import debug_event
    debug_event(
        log, "tenant_config validate_credentials passed",
        tenant_slug=settings.slug, source=source,
        telephony_provider=p.telephony.provider, mode=p.mode,
        chat_voice_enabled=p.chat_voice.enabled,
    )


def discover_tenant_slugs(tenant_dir: Optional[Path] = None) -> list[str]:
    """Return slugs for every YAML in the tenant config directory."""
    base = _resolve_dir(tenant_dir)
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))


def load_all_tenants(tenant_dir: Optional[Path] = None) -> dict[str, TenantSettings]:
    """Load every tenant in ``config/tenants/``. Returns ``{slug: settings}``."""
    from src.utils.logging import debug_event

    slugs = discover_tenant_slugs(tenant_dir)
    debug_event(log, "tenant_config load_all_tenants request", slugs=slugs, count=len(slugs))
    return {slug: load_tenant(slug, tenant_dir) for slug in slugs}


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
    overridden: list[str] = []
    discarded_env_refs: list[str] = []
    for k, v in tenant_layer.model_dump().items():
        if k.endswith("_env"):
            # *_env fields are credential REFERENCES (an env var name), not
            # values -- they never belong in a provider config dict (that
            # would ship a var NAME to the adapter in place of a resolved
            # secret). Recorded here as a name only, matching the rest of
            # this package's credential rule.
            if v is not None:
                discarded_env_refs.append(k)
            continue
        if v is not None:
            out[k] = v
            overridden.append(k)
    if api_key is not None:
        out["api_key"] = api_key
    # This is the merge every "why is this tenant using provider X" question
    # in the task brief resolves through -- overridden names the fields THIS
    # tenant set; anything in `out` not in `overridden` fell through from
    # global_layer (config/default.yaml), which is the exact "silently
    # inherited" shape the chat-voice-TTS incident turned out to be.
    # Guarded: model_dump() + two list builds run on every cache-miss
    # (registry.py, once per tenant/layer) and on every backoffice tenant-list
    # render (src/api/tenants.py's _layer(), N tenants x 4 layers) -- neither
    # is per-turn, but neither is a bare scalar read either.
    if log.isEnabledFor(logging.DEBUG):
        from src.utils.logging import debug_event
        # `out` itself is never logged: if a caller ever does pass `api_key`
        # (no current call site does -- see the finding recorded for this
        # function), it lands in `out["api_key"]` and `out` would then BE the
        # credential. Logging the individual non-credential fields instead
        # of the dict keeps that impossible by construction rather than by
        # remembering to strip a key every time this function changes.
        debug_event(
            log, "tenant_config merge_provider_config result",
            layer_model=type(tenant_layer).__name__,
            tenant_overridden_fields=overridden,
            tenant_overridden_values={k: out[k] for k in overridden},
            platform_default_fields=[k for k in out if k not in overridden and k != "api_key"],
            discarded_env_ref_fields=discarded_env_refs,
            api_key_set=api_key is not None,
        )
    return out
