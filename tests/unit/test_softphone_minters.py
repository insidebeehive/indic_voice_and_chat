"""Browser softphone credential minting (Twilio + Stringee)."""

from __future__ import annotations

import jwt
import pytest

from src.auth.context import TenantContext
from src.config_tenant import (
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
)
from src.providers.telephony.softphone import (
    SoftphoneConfigError,
    SoftphoneUnsupported,
    mint_browser_credentials,
    supports_softphone,
)


def _tenant(provider: str, *, secrets: dict[str, str] | None = None, **tel) -> TenantContext:
    settings = TenantSettings(
        id="t1", slug="acme", name="Acme",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider=provider, **tel)),
    )
    return TenantContext(settings=settings, secrets_resolved=secrets or {})


def test_supports_softphone() -> None:
    assert supports_softphone("twilio")
    assert supports_softphone("Stringee")
    assert not supports_softphone("exotel")
    assert not supports_softphone("didlogic")
    assert not supports_softphone(None)


def test_twilio_minter_produces_access_token_with_voice_grant() -> None:
    tenant = _tenant(
        "twilio",
        account_sid_env="A_SID", api_key_sid_env="K_SID",
        api_key_secret_env="K_SECRET", twiml_app_sid_env="APP",
        secrets={
            "A_SID": "AC_account", "K_SID": "SK_key",
            "K_SECRET": "the-secret", "APP": "AP_twiml",
        },
    )
    creds = mint_browser_credentials(tenant, "agent-42", ttl_seconds=600)
    assert creds.provider == "twilio"
    assert creds.identity == "agent-42"
    assert creds.ttl_seconds == 600
    # Decode without verifying the signature to inspect the grant.
    claims = jwt.decode(creds.token, options={"verify_signature": False})
    assert claims["iss"] == "SK_key"           # the API Key SID signs the token
    assert claims["sub"] == "AC_account"       # the account SID
    voice = claims["grants"]["voice"]
    assert voice["outgoing"]["application_sid"] == "AP_twiml"
    assert claims["grants"]["identity"] == "agent-42"


def test_twilio_minter_missing_creds_raises_config_error() -> None:
    tenant = _tenant("twilio", account_sid_env="A_SID", secrets={"A_SID": "AC"})
    with pytest.raises(SoftphoneConfigError) as e:
        mint_browser_credentials(tenant, "agent-1")
    # names the missing API key + TwiML App fields
    assert "api_key_sid" in str(e.value)
    assert "twiml_app_sid" in str(e.value)


def test_stringee_minter_client_jwt() -> None:
    tenant = _tenant(
        "stringee",
        account_sid_env="S_SID", auth_token_env="S_SECRET",
        secrets={"S_SID": "stringee-key", "S_SECRET": "stringee-secret"},
    )
    creds = mint_browser_credentials(tenant, "agent-9", ttl_seconds=1200, now=1_000_000)
    assert creds.provider == "stringee"
    header = jwt.get_unverified_header(creds.token)
    assert header["cty"] == "stringee-api;v=1"
    claims = jwt.decode(
        creds.token, "stringee-secret", algorithms=["HS256"],
        options={"verify_exp": False})
    assert claims["iss"] == "stringee-key"
    assert claims["userId"] == "agent-9"     # a CLIENT token carries the user id
    assert claims["rest_api"] is False
    assert claims["exp"] == 1_000_000 + 1200


def test_stringee_minter_missing_creds_raises(monkeypatch) -> None:
    # No tenant secrets AND no platform STRINGEE_* env → config error.
    monkeypatch.delenv("STRINGEE_API_KEY_SID", raising=False)
    monkeypatch.delenv("STRINGEE_API_KEY_SECRET", raising=False)
    tenant = _tenant("stringee")
    with pytest.raises(SoftphoneConfigError):
        mint_browser_credentials(tenant, "agent-1")


def test_unsupported_provider_raises() -> None:
    tenant = _tenant("exotel")
    with pytest.raises(SoftphoneUnsupported) as e:
        mint_browser_credentials(tenant, "agent-1")
    assert "exotel" in str(e.value)


def test_blank_identity_rejected() -> None:
    tenant = _tenant("twilio")
    with pytest.raises(SoftphoneConfigError):
        mint_browser_credentials(tenant, "")
