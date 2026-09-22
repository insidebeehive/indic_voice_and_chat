"""TenantContext — the per-request handle through the rest of the system.

A ``TenantContext`` carries the validated settings + a small helper for
resolving secrets. Every state-holding component that wants to behave
tenant-aware accepts one of these.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from src.config_tenant import TenantSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TenantContext:
    """Immutable per-request tenant handle."""

    settings: TenantSettings
    # Decrypted per-tenant secrets (TELEPHONY keys only), keyed by their config
    # name (e.g. ``twilio_sid``). Loaded by the DB resolver. Everything else
    # (STT/LLM/TTS/S2S) resolves from the shared master env, so ``secret()`` checks
    # this map first and falls back to ``os.environ`` for non-telephony keys.
    secrets_resolved: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.settings.id

    @property
    def slug(self) -> str:
        return self.settings.slug

    @property
    def name(self) -> str:
        return self.settings.name

    def secret(self, env_var: Optional[str]) -> Optional[str]:
        if env_var is None:
            return None
        if env_var in self.secrets_resolved:          # per-tenant telephony key
            # Mirrors TenantSettings.secret()'s own debug_event (src/config_tenant.py)
            # for the master-env fallback below -- without this, the per-tenant
            # decrypted path (the common case for telephony creds) was the one
            # branch of this function that left no trace of what resolved.
            from src.auth.audit import token_fingerprint
            from src.utils.logging import debug_event
            value = self.secrets_resolved[env_var]
            debug_event(
                log, "auth secret resolved", tenant_id=self.id, secret_name=env_var,
                source="per_tenant_secret",
                value_fp=token_fingerprint(value, domain="vox-logfp-tenant-secret-v1"),
                value_len=len(value),
            )
            return value
        return self.settings.secret(env_var)          # master env (stt/llm/tts/s2s)

    def secret_optional(self, env_var: Optional[str]) -> Optional[str]:
        """Like ``secret()`` but returns None instead of raising when the name is
        unset — for OPTIONAL secrets (e.g. the webhook signing key) where absence
        just means "send unsigned". Per-tenant decrypted secret first, then env."""
        if env_var is None:
            return None
        from src.auth.audit import token_fingerprint
        from src.utils.logging import debug_event
        if env_var in self.secrets_resolved:          # per-tenant decrypted secret
            value = self.secrets_resolved[env_var]
            debug_event(
                log, "auth secret resolved", tenant_id=self.id, secret_name=env_var,
                source="per_tenant_secret_optional",
                value_fp=token_fingerprint(value, domain="vox-logfp-tenant-secret-v1"),
                value_len=len(value),
            )
            return value
        value = os.environ.get(env_var)                # env fallback, never raises
        debug_event(
            log, "auth secret resolved", tenant_id=self.id, secret_name=env_var,
            source="env_optional", found=value is not None,
            value_fp=token_fingerprint(value, domain="vox-logfp-tenant-secret-v1") if value else None,
            value_len=len(value) if value else None,
        )
        return value


def hash_api_token(plaintext: str) -> str:
    """SHA-256 of the bearer token — the only form stored in the DB."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
