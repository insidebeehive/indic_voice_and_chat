from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from src.api import telephony_hooks
from src.auth.context import TenantContext
from src.config_tenant import TenantSettings
from src.utils import http_fetch


def test_recording_mime_sniffs_wav_vs_mp3() -> None:
    assert telephony_hooks._recording_mime(b"RIFF....WAVE") == "audio/wav"
    assert telephony_hooks._recording_mime(b"\xff\xe3\x28\xc4") == "audio/mpeg"


def test_audio_transcriber_uses_tenant_llm_if_capable() -> None:
    class _LLM:
        async def transcribe_audio(self, audio, mime_type="audio/mpeg"):
            return "x"
    llm = _LLM()
    assert telephony_hooks._audio_transcriber(llm) is llm


def test_audio_transcriber_falls_back_to_gemini(monkeypatch) -> None:
    """A tenant whose analysis LLM can't transcribe audio (e.g. Groq) still gets a
    Gemini transcriber from the platform key."""
    import src.providers as providers
    sentinel = object()
    monkeypatch.setattr(
        providers, "get_llm_provider",
        lambda cfg: sentinel if cfg.get("provider") == "gemini" else None)

    class _GroqLike:  # no transcribe_audio
        pass
    assert telephony_hooks._audio_transcriber(_GroqLike()) is sentinel


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
async def test_download_retries_on_404_until_recording_ready() -> None:
    """The recording lags the event — Stringee 404s briefly after the call. The
    download retries on 404 (with backoff) instead of giving up."""
    route = respx.get("https://api.stringee.com/v1/call/recording/abc").mock(
        side_effect=[httpx.Response(404), httpx.Response(404),
                     httpx.Response(200, content=b"WAV")])
    sleeps: list[float] = []

    async def _fake_sleep(s):
        sleeps.append(s)

    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await telephony_hooks._download_stringee_recording(
            "http://api.stringee.com/v1/call/recording/abc", _tenant(), _sleep=_fake_sleep)
    assert out == b"WAV"
    assert route.call_count == 3        # 404, 404, 200
    assert len(sleeps) == 2             # waited before each retry


@respx.mock
@pytest.mark.asyncio
async def test_download_rewrites_host_to_tenant_regional_base() -> None:
    """When the tenant's Stringee project is on a regional REST host, the recording
    URL (which arrives pointing at api.stringee.com) is rewritten to that host —
    otherwise the global host returns r:5 'keySid invalid'."""
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    tenant = TenantContext(
        settings=TenantSettings(
            id="t", slug="t", name="T",
            pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
                provider="stringee", account_sid_env="SID", auth_token_env="SEC",
                stringee_base_url="https://asia-2.api.stringee.com"))),
        secrets_resolved={"SID": "keysid", "SEC": "keysecret"})
    route = respx.get("https://asia-2.api.stringee.com/v1/call/recording/abc").mock(
        return_value=httpx.Response(200, content=b"WAV"))
    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await telephony_hooks._download_stringee_recording(
            "http://api.stringee.com/v1/call/recording/abc", tenant)
    assert out == b"WAV"
    assert route.calls.last.request.url.host == "asia-2.api.stringee.com"
    assert route.calls.last.request.url.path == "/v1/call/recording/abc"


@respx.mock
@pytest.mark.asyncio
async def test_download_goes_to_https_with_auth_header() -> None:
    """Fetch over https (Stringee 301s http→https) authenticated with the
    X-STRINGEE-AUTH server token — the same auth the callout uses."""
    route = respx.get("https://api.stringee.com/v1/call/recording/abc").mock(
        return_value=httpx.Response(200, content=b"DATA"))
    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await telephony_hooks._download_stringee_recording(
            "http://api.stringee.com/v1/call/recording/abc", _tenant_with_creds())
    assert out == b"DATA"
    req = route.calls.last.request
    assert req.url.scheme == "https"                       # upgraded up front
    assert req.headers.get("X-STRINGEE-AUTH")              # authenticated via header


@respx.mock
@pytest.mark.asyncio
async def test_download_forces_https_when_regional_base_has_http_scheme() -> None:
    """Regression test: a tenant's configured ``stringee_base_url`` may already
    carry a scheme (e.g. copy-pasted as ``http://asia-2.api.stringee.com``).
    The old code only forced https when the base had NO scheme at all
    (``"://" not in base``), so an http-scheme base silently stayed on http.
    The fetch must always go out over https regardless."""
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    tenant = TenantContext(
        settings=TenantSettings(
            id="t", slug="t", name="T",
            pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
                provider="stringee", account_sid_env="SID", auth_token_env="SEC",
                stringee_base_url="http://asia-2.api.stringee.com"))),
        secrets_resolved={"SID": "keysid", "SEC": "keysecret"})

    https_route = respx.get("https://asia-2.api.stringee.com/v1/call/recording/abc").mock(
        return_value=httpx.Response(200, content=b"WAV"))
    http_route = respx.get("http://asia-2.api.stringee.com/v1/call/recording/abc").mock(
        return_value=httpx.Response(200, content=b"SHOULD-NOT-BE-FETCHED"))

    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await telephony_hooks._download_stringee_recording(
            "http://api.stringee.com/v1/call/recording/abc", tenant)
    assert out == b"WAV"
    assert https_route.call_count == 1
    assert http_route.call_count == 0     # never fetched over http, no redirect chase either


@respx.mock
@pytest.mark.asyncio
async def test_download_uses_https_directly_without_relying_on_redirect() -> None:
    """The URL scheme is forced to https up front, before any request is made —
    the download never relies on Stringee's http→https 301 (a mocked 301 route
    below is deliberately left uncalled to prove that)."""
    http_url = "http://api.stringee.com/v1/call/recording/abc"
    https_url = "https://api.stringee.com/v1/call/recording/abc"
    http_route = respx.get(http_url).mock(
        return_value=httpx.Response(301, headers={"Location": https_url}))
    respx.get(https_url).mock(
        return_value=httpx.Response(200, content=b"RIFFwavdata"))

    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await telephony_hooks._download_stringee_recording(http_url, _tenant())
    assert out == b"RIFFwavdata"
    assert http_route.call_count == 0     # never fetched over http, no redirect chase either


@respx.mock
@pytest.mark.asyncio
async def test_download_forces_https_for_ivr_turn_audio() -> None:
    """``_download`` (the live IVR turn fetch, distinct from
    ``_download_stringee_recording``) must also force the scheme to https
    before validating/fetching — this codebase's own test fixtures model
    Stringee recording URLs as http://, and fetch_capped requires https up
    front, so an unmodified http:// input would be rejected outright and,
    since this runs inline on a live call turn, fail silently into an
    infinite reprompt loop."""
    http_url = "http://api.stringee.com/v1/call/recording/xyz"
    https_url = "https://api.stringee.com/v1/call/recording/xyz"
    http_route = respx.get(http_url).mock(
        return_value=httpx.Response(200, content=b"SHOULD-NOT-BE-FETCHED"))
    https_route = respx.get(https_url).mock(
        return_value=httpx.Response(200, content=b"WAV"))

    with patch.object(http_fetch, "is_public_host", return_value=True):
        out = await telephony_hooks._download(http_url, _tenant())
    assert out == b"WAV"
    assert https_route.call_count == 1
    assert http_route.call_count == 0
