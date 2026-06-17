"""Browser softphone credential minting — human agent ↔ lead.

A CRM's **human agents** dial leads from their browser (a softphone); our AI is
not in the conversation. The browser speaks WebRTC, so it needs a **short-lived
provider credential** to register with the telephony provider's browser SDK.
This module mints those credentials, provider-agnostically.

Only providers that ship a browser WebRTC SDK are supported:
  - **Twilio**  → Voice JavaScript SDK; we mint a Twilio ``AccessToken`` with a
    ``VoiceGrant`` pointing at a TwiML App (whose Voice URL is our
    ``/telephony/twilio/softphone-twiml`` dial endpoint).
  - **Stringee** → Web/Client SDK; we mint a *client* access JWT (same HS256 +
    ``cty: stringee-api;v=1`` header as the server token, but carrying a
    ``userId`` and ``rest_api: false``).

Other providers (Exotel, DiDLogic/SIP) have no browser SDK and raise
``SoftphoneUnsupported`` until a WebRTC↔SIP gateway is stood up.

The minted tokens are short-lived (≈1 h) and therefore safe to hand to a
browser — unlike our non-expiring tenant API token. The CRM's **backend** calls
our token endpoint server-side and passes the result to its browser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.auth.context import TenantContext

# Telephony providers with a usable browser softphone SDK.
SUPPORTED_PROVIDERS = ("twilio", "stringee")

# Stringee client tokens default to 1h; Twilio AccessToken TTL is in seconds too.
DEFAULT_TTL_SECONDS = 3600


class SoftphoneUnsupported(Exception):
    """The tenant's telephony provider has no browser softphone path."""


class SoftphoneConfigError(Exception):
    """The provider supports softphone but the tenant is missing credentials."""


@dataclass
class SoftphoneCredentials:
    """A short-lived browser credential + whatever the SDK needs to register."""

    provider: str
    token: str
    identity: str
    ttl_seconds: int
    params: dict[str, str] = field(default_factory=dict)


def supports_softphone(provider: Optional[str]) -> bool:
    return (provider or "").lower() in SUPPORTED_PROVIDERS


def mint_browser_credentials(
    tenant: TenantContext,
    agent_identity: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    now: Optional[int] = None,
) -> SoftphoneCredentials:
    """Mint a browser credential for ``tenant``'s configured telephony provider.

    ``agent_identity`` is the CRM's human-agent id (becomes the SDK identity /
    Stringee ``userId``). Raises ``SoftphoneUnsupported`` for providers without a
    browser SDK, or ``SoftphoneConfigError`` when required credentials are unset.
    """
    if not agent_identity:
        raise SoftphoneConfigError("agent_identity is required")
    provider = (tenant.settings.pipeline.telephony.provider or "").lower()
    if provider == "twilio":
        return _mint_twilio(tenant, agent_identity, ttl_seconds)
    if provider == "stringee":
        return _mint_stringee(tenant, agent_identity, ttl_seconds, now)
    raise SoftphoneUnsupported(
        f"telephony provider {provider!r} has no browser softphone SDK; "
        f"supported: {list(SUPPORTED_PROVIDERS)} (others need a WebRTC↔SIP gateway)"
    )


def _mint_twilio(tenant: TenantContext, identity: str, ttl: int) -> SoftphoneCredentials:
    from twilio.jwt.access_token import AccessToken
    from twilio.jwt.access_token.grants import VoiceGrant

    c = tenant.settings.pipeline.telephony.active_creds()
    account_sid = tenant.secret(c.account_sid_env)
    api_key_sid = tenant.secret(c.api_key_sid_env)
    api_key_secret = tenant.secret(c.api_key_secret_env)
    twiml_app_sid = tenant.secret(c.twiml_app_sid_env)

    missing = [
        label for label, val in (
            ("account_sid", account_sid),
            ("api_key_sid", api_key_sid),
            ("api_key_secret", api_key_secret),
            ("twiml_app_sid", twiml_app_sid),
        ) if not val
    ]
    if missing:
        raise SoftphoneConfigError(
            f"Twilio softphone needs {missing} configured for tenant "
            f"{tenant.slug!r} (API Key SID/Secret + TwiML App SID)"
        )

    token = AccessToken(account_sid, api_key_sid, api_key_secret, identity=identity, ttl=ttl)
    token.add_grant(VoiceGrant(outgoing_application_sid=twiml_app_sid, incoming_allow=True))
    jwt_str = token.to_jwt()
    if isinstance(jwt_str, bytes):  # twilio<9 returned bytes
        jwt_str = jwt_str.decode("ascii")
    return SoftphoneCredentials(
        provider="twilio", token=jwt_str, identity=identity, ttl_seconds=ttl,
    )


def _mint_stringee(
    tenant: TenantContext, identity: str, ttl: int, now: Optional[int]
) -> SoftphoneCredentials:
    import jwt  # PyJWT

    c = tenant.settings.pipeline.telephony.active_creds()
    # Stringee reuses its account api_key_sid/secret (stored as account_sid/
    # auth_token). Per-tenant only — NO platform-env fallback: a tenant without
    # Stringee keys configured can't mint a token (fail loudly, don't borrow the
    # platform's account).
    api_key_sid = tenant.secret(c.account_sid_env) if c.account_sid_env else None
    api_key_secret = tenant.secret(c.auth_token_env) if c.auth_token_env else None
    if not (api_key_sid and api_key_secret):
        raise SoftphoneConfigError(
            f"Stringee softphone needs the tenant's api_key_sid + api_key_secret for "
            f"{tenant.slug!r} (set the tenant's Stringee account_sid/auth_token)"
        )

    issued = now if now is not None else int(time.time())
    payload = {
        "jti": f"{api_key_sid}-{issued}",
        "iss": api_key_sid,
        "exp": issued + ttl,
        "userId": identity,          # client token: the human agent identity
        "rest_api": False,           # a CLIENT token, not a server REST token
    }
    token = jwt.encode(
        payload, api_key_secret, algorithm="HS256",
        headers={"cty": "stringee-api;v=1"},
    )
    if isinstance(token, bytes):  # PyJWT<2 returned bytes
        token = token.decode("ascii")
    return SoftphoneCredentials(
        provider="stringee", token=token, identity=identity, ttl_seconds=ttl,
    )
