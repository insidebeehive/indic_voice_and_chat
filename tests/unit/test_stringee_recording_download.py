from __future__ import annotations

import httpx
import pytest
import respx

from src.api import telephony_hooks
from src.auth.context import TenantContext
from src.config_tenant import TenantSettings


def _tenant() -> TenantContext:
    # No creds env set → no auth header; exercises the plain (signed-URL) path.
    return TenantContext(settings=TenantSettings(id="t", slug="t", name="T"))


def _tenant_with_creds() -> TenantContext:
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    return TenantContext(
        settings=TenantSettings(
            id="t", slug="t", name="T",
            pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
                provider="stringee", account_sid_env="SID", auth_token_env="SEC"))),
        secrets_resolved={"SID": "keysid", "SEC": "keysecret"})


@respx.mock
@pytest.mark.asyncio
async def test_download_goes_to_https_with_auth_header() -> None:
    """Fetch over https (Stringee 301s http→https) authenticated with the
    X-STRINGEE-AUTH server token — the same auth the callout uses."""
    route = respx.get("https://api.stringee.com/v1/call/recording/abc").mock(
        return_value=httpx.Response(200, content=b"DATA"))
    out = await telephony_hooks._download_stringee_recording(
        "http://api.stringee.com/v1/call/recording/abc", _tenant_with_creds())
    assert out == b"DATA"
    req = route.calls.last.request
    assert req.url.scheme == "https"                       # upgraded up front
    assert req.headers.get("X-STRINGEE-AUTH")              # authenticated via header


@respx.mock
@pytest.mark.asyncio
async def test_download_follows_http_to_https_301() -> None:
    """Stringee 301-redirects http→https for recording URLs; the download must
    follow it instead of raising on the 3xx."""
    http_url = "http://api.stringee.com/v1/call/recording/abc"
    https_url = "https://api.stringee.com/v1/call/recording/abc"
    respx.get(http_url).mock(
        return_value=httpx.Response(301, headers={"Location": https_url}))
    respx.get(https_url).mock(
        return_value=httpx.Response(200, content=b"RIFFwavdata"))

    out = await telephony_hooks._download_stringee_recording(http_url, _tenant())
    assert out == b"RIFFwavdata"
