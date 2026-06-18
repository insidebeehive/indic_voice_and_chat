from __future__ import annotations

import httpx
import pytest
import respx

from src.api import telephony_hooks
from src.auth.context import TenantContext
from src.config_tenant import TenantSettings


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
