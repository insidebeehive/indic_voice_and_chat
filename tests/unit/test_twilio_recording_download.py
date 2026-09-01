"""Regression tests for C3: `_download_twilio_recording` must never fetch an
attacker-controlled URL with the tenant's real Twilio Account SID + Auth
Token attached (SSRF + credential leak). Covers the real validation path —
unlike test_softphone_webhook.py's `ctx` fixture, which monkeypatches the
download function away entirely."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
import respx
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import telephony_hooks
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.campaign.models import LeadCallOutcome
from src.config_tenant import TenantPipelineConfig, TenantSettings, TenantTelephonyConfig
from src.models.conversation import Conversation
from src.models.database import Base
from src.models.tenant import Tenant
from src.utils import http_fetch


class _FakeProviders:
    """Never actually reached — recording download is rejected before STT/LLM."""

    def get_stt(self, tenant):  # noqa: ANN001
        raise AssertionError("STT should not be reached when the download is rejected")

    def get_llm(self, tenant):  # noqa: ANN001
        raise AssertionError("LLM should not be reached when the download is rejected")


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add(Tenant(id="t1", slug="acme", name="Acme"))
        await s.commit()

    async def _session_override():
        async with maker() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(
        TenantSettings(
            id="t1", slug="acme", name="Acme",
            pipeline=TenantPipelineConfig(
                telephony=TenantTelephonyConfig(
                    provider="twilio", from_number="+15550001111",
                    account_sid_env="TWILIO_SID_TEST", auth_token_env="TWILIO_TOK_TEST"),
            ),
        ),
        plaintext_tokens=["tok"],
    )
    monkeypatch.setenv("TWILIO_SID_TEST", "AC_REAL")
    monkeypatch.setenv("TWILIO_TOK_TEST", "secret-token")
    telephony_hooks.set_softphone_providers(_FakeProviders())
    telephony_hooks.set_softphone_sessionmaker(maker)

    app = FastAPI()
    app.include_router(telephony_hooks.router, prefix="/api/v1")
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, maker
    telephony_hooks.set_softphone_providers(None)
    telephony_hooks.set_softphone_sessionmaker(None)
    set_tenant_resolver(None)
    await engine.dispose()


@respx.mock
async def test_webhook_rejects_attacker_url_and_marks_unavailable(ctx) -> None:
    """The recording webhook must never fetch an unmocked/unallowlisted host —
    respx raises if an unmocked request goes out under respx.mock, which is
    itself the proof no request reached it — and must still 200 with the
    call marked RECORDING_UNAVAILABLE instead of 500ing."""
    client, maker = ctx
    # Log the manual call first (as the dial TwiML would) so record_outcome has
    # a row to update.
    await client.post(
        "/api/v1/telephony/twilio/softphone-twiml/acme",
        data={"To": "+918618795697", "From": "client:a", "CallSid": "CA-SSRF"})

    resp = await client.post(
        "/api/v1/telephony/twilio/softphone-recording/acme",
        data={"CallSid": "CA-SSRF", "RecordingUrl": "https://attacker.example.com/steal",
              "RecordingDuration": "10"})
    assert resp.status_code == 200

    async with maker() as s:
        row = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "CA-SSRF")
        )).scalar_one()
    assert row.outcome == LeadCallOutcome.RECORDING_UNAVAILABLE.value
    assert row.status == "ended"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,account_sid",
    [
        # not https
        ("http://api.twilio.com/2010-04-01/Accounts/AC_REAL/Recordings/RE123", "AC_REAL"),
        # wrong host
        ("https://attacker.com/x.wav", "AC_REAL"),
        # subdomain-suffix bypass attempt — anchored pattern must reject
        ("https://api.twilio.com.evil.com/2010-04-01/Accounts/AC_REAL/Recordings/RE123", "AC_REAL"),
        # right host, wrong account
        ("https://api.twilio.com/2010-04-01/Accounts/AC_DIFFERENT_ACCOUNT/Recordings/RE123", "AC_REAL"),
        # dot-segment: raw path scopes to AC_REAL but httpx normalizes ".."
        # before sending, so the request actually reaches AC_DIFFERENT_ACCOUNT
        ("https://api.twilio.com/2010-04-01/Accounts/AC_REAL/../AC_DIFFERENT_ACCOUNT/Recordings/RE123", "AC_REAL"),
    ],
)
async def test_download_rejects_unsafe_urls(url, account_sid) -> None:
    # is_public_host patched True so a rejection can only come from the
    # https/allowlist/account-scoping checks in assert_safe_url and
    # _download_twilio_recording — not from a real DNS lookup failing (which
    # would otherwise make the "api.twilio.com.evil.com" anchoring-bypass
    # case pass for the wrong reason, and would flake offline for all cases).
    with respx.mock, patch.object(http_fetch, "is_public_host", return_value=True):
        # no routes registered — any network call fails the test
        with pytest.raises(ValueError):
            await telephony_hooks._download_twilio_recording(url, account_sid, "tok")


@respx.mock
@pytest.mark.asyncio
async def test_download_accepts_valid_scoped_url_with_basic_auth() -> None:
    url = "https://api.twilio.com/2010-04-01/Accounts/AC_REAL/Recordings/RE123.wav"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=b"WAVDATA"))

    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await telephony_hooks._download_twilio_recording(url, "AC_REAL", "tok")
    assert out == b"WAVDATA"
    req = route.calls.last.request
    assert (req.headers.get("authorization") or "").startswith("Basic ")


@respx.mock
@pytest.mark.asyncio
async def test_download_does_not_follow_redirect() -> None:
    url = "https://api.twilio.com/2010-04-01/Accounts/AC_REAL/Recordings/RE123.wav"
    redirect_target = "https://attacker.example.com/steal.wav"
    respx.get(url).mock(
        return_value=httpx.Response(302, headers={"Location": redirect_target}))
    target_route = respx.get(redirect_target).mock(
        return_value=httpx.Response(200, content=b"SHOULD-NOT-BE-FETCHED"))

    # fetch_capped raises ValueError on a redirect response (see http_fetch.py) —
    # narrowed from a blind Exception so this actually proves the redirect
    # refusal fired, not some unrelated failure.
    with patch.object(http_fetch, "is_public_host", return_value=True):
        with pytest.raises(ValueError, match="redirect"):
            await telephony_hooks._download_twilio_recording(url, "AC_REAL", "tok")
    assert target_route.call_count == 0
