"""Route tests for POST /softphone/token."""

from __future__ import annotations

import jwt
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api import softphone
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import (
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
)

HEADERS = {"Authorization": "Bearer test-token"}


def _register(provider: str, *, token: str = "test-token", **tel) -> None:
    register_tenant_for_test(
        TenantSettings(
            id=f"t_{provider}", slug=provider, name=provider.title(),
            pipeline=TenantPipelineConfig(
                telephony=TenantTelephonyConfig(provider=provider, **tel)),
        ),
        plaintext_tokens=[token],
    )


@pytest_asyncio.fixture
async def client(monkeypatch):
    # Twilio softphone creds resolve from env (the in-memory resolver has no
    # encrypted-secrets map; settings.secret() falls back to os.environ).
    monkeypatch.setenv("ENV_ACCT", "AC_account")
    monkeypatch.setenv("ENV_KEYSID", "SK_key")
    monkeypatch.setenv("ENV_KEYSECRET", "the-secret")
    monkeypatch.setenv("ENV_APP", "AP_twiml")

    set_tenant_resolver(None)
    app = FastAPI()
    app.include_router(softphone.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=HEADERS) as c:
        yield c
    set_tenant_resolver(None)


async def test_twilio_token_minted(client) -> None:
    _register(
        "twilio",
        account_sid_env="ENV_ACCT", api_key_sid_env="ENV_KEYSID",
        api_key_secret_env="ENV_KEYSECRET", twiml_app_sid_env="ENV_APP",
    )
    resp = await client.post("/softphone/token", json={"agent_identity": "agent-7"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "twilio"
    assert body["identity"] == "agent-7"
    claims = jwt.decode(body["token"], options={"verify_signature": False})
    assert claims["grants"]["voice"]["outgoing"]["application_sid"] == "AP_twiml"


async def test_unsupported_provider_returns_400(client) -> None:
    _register("exotel")
    resp = await client.post("/softphone/token", json={"agent_identity": "agent-1"})
    assert resp.status_code == 400
    assert "exotel" in resp.json()["detail"]


async def test_missing_twilio_creds_returns_400(client) -> None:
    # provider supports softphone but no API key / TwiML App configured
    _register("twilio")
    resp = await client.post("/softphone/token", json={"agent_identity": "agent-1"})
    assert resp.status_code == 400


async def test_requires_auth(client) -> None:
    _register("twilio")
    resp = await client.post(
        "/softphone/token", json={"agent_identity": "a"}, headers={"Authorization": ""})
    assert resp.status_code == 401


async def test_blank_identity_rejected_422(client) -> None:
    _register("twilio")
    resp = await client.post("/softphone/token", json={"agent_identity": ""})
    assert resp.status_code == 422


async def test_stringee_token_carries_caller_id_from_number(client, monkeypatch) -> None:
    # The browser must place the call FROM the caller-ID number, so the token
    # response carries it (bare, no "+") in params.from_number.
    monkeypatch.setenv("STRINGEE_API_KEY_SID", "SK.test")
    monkeypatch.setenv("STRINGEE_API_KEY_SECRET", "stringee-secret-1234567890")
    _register("stringee", from_number="+918204268005")
    resp = await client.post("/softphone/token", json={"agent_identity": "dev"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "stringee"
    assert body["params"]["from_number"] == "918204268005"
