"""Unit tests for src/utils/public_url.py — the shared config-first,
header-fallback derivation of the app's public scheme://netloc.

The regression this whole module exists to fix: platform_webhook_base_url()
is documented and used ending in a path suffix (e.g.
"https://host/api/v1/telephony"), NOT a bare origin. Naively concatenating
that value with a request path doubles the path
(".../api/v1/telephony/api/v1/telephony/...") and breaks every URL built
from it. public_origin() must discard that path entirely.
"""

from __future__ import annotations

from src.utils import public_url as pu


class _FakeURL:
    def __init__(self, netloc: str, scheme: str) -> None:
        self.netloc = netloc
        self.scheme = scheme


class _FakeRequest:
    """Minimal stand-in for Starlette's Request/WebSocket — both expose
    ``.headers`` (dict-like, ``.get``) and ``.url`` (``.netloc``/``.scheme``),
    which is all public_origin() touches."""

    def __init__(self, headers: dict | None = None, netloc: str = "testserver",
                 scheme: str = "http") -> None:
        self.headers = headers or {}
        self.url = _FakeURL(netloc, scheme)


# --- the regression test ----------------------------------------------


def test_configured_base_url_path_suffix_is_discarded(monkeypatch) -> None:
    """THE regression test: platform_webhook_base_url() carrying its own
    /api/v1/telephony path must not leak into the returned origin."""
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: "https://host/api/v1/telephony")
    assert pu.public_origin() == "https://host"


def test_configured_base_url_without_scheme_still_produces_https_origin(monkeypatch) -> None:
    monkeypatch.setattr(
        pu, "platform_webhook_base_url", lambda: "host.example.com/api/v1/telephony")
    assert pu.public_origin() == "https://host.example.com"


def test_configured_base_url_with_trailing_slash_and_no_path(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: "https://host.example.com/")
    assert pu.public_origin() == "https://host.example.com"


# --- header-fallback (platform_webhook_base_url unset) -----------------


def test_unset_falls_back_to_forwarded_headers(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: None)
    req = _FakeRequest(
        headers={"x-forwarded-host": "public.example.com", "x-forwarded-proto": "https"},
        netloc="internal-container:8000", scheme="http")
    assert pu.public_origin(req) == "https://public.example.com"


def test_unset_falls_back_to_request_url_without_forwarded_headers(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: None)
    req = _FakeRequest(headers={}, netloc="testserver", scheme="http")
    assert pu.public_origin(req) == "http://testserver"


def test_unset_https_request_without_forwarded_headers(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: None)
    req = _FakeRequest(headers={}, netloc="example.com", scheme="https")
    assert pu.public_origin(req) == "https://example.com"


def test_unset_websocket_wss_scheme_without_forwarded_headers(monkeypatch) -> None:
    """A WebSocket's .url.scheme is 'ws'/'wss', not 'http'/'https' — the
    secure-scheme check must also recognize 'wss'."""
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: None)
    ws = _FakeRequest(headers={}, netloc="example.com", scheme="wss")
    assert pu.public_origin(ws) == "https://example.com"


def test_unset_and_no_request_returns_empty_string(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: None)
    assert pu.public_origin(None) == ""


def test_empty_string_config_treated_as_unset(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: "")
    req = _FakeRequest(headers={}, netloc="testserver", scheme="http")
    assert pu.public_origin(req) == "http://testserver"


# --- public_http_url / public_ws_url ------------------------------------


def test_public_http_url_joins_configured_origin_with_path(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: "https://host/api/v1/telephony")
    assert (pu.public_http_url(None, "/api/v1/calls/CA1/transfer-result")
            == "https://host/api/v1/calls/CA1/transfer-result")


def test_public_http_url_strips_extra_leading_slashes_on_path(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: "https://host/api/v1/telephony")
    assert pu.public_http_url(None, "api/v1/telephony") == "https://host/api/v1/telephony"


def test_public_ws_url_forces_wss_from_https_config(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: "https://host/api/v1/telephony")
    assert (pu.public_ws_url(None, "api/v1/telephony/twilio/stream/dev")
            == "wss://host/api/v1/telephony/twilio/stream/dev")


def test_public_ws_url_forces_ws_from_header_derived_http(monkeypatch) -> None:
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: None)
    req = _FakeRequest(headers={}, netloc="testserver", scheme="http")
    assert (pu.public_ws_url(req, "api/v1/telephony/twilio/stream/dev")
            == "ws://testserver/api/v1/telephony/twilio/stream/dev")


# --- origin_from_headers: pure header derivation, NEVER config-first ----
#
# This is the round-1 regression test: telephony_hooks.py's _ws_stream_url,
# _forwarded_base, and _stringee_base must stay purely header-derived and
# must never read platform_webhook_base_url(), even when it IS configured —
# unlike public_origin(), which is intentionally config-first. A prior
# implementation attempt wired those three functions through public_origin()
# by mistake, which would have silently retargeted every inbound Twilio/
# Exotel/Stringee callback URL whenever WEBHOOK_BASE_URL pointed at a
# different host than the one actually serving the request.


def test_origin_from_headers_ignores_configured_base_url(monkeypatch) -> None:
    """Even with platform_webhook_base_url() set, origin_from_headers() must
    derive purely from the request/websocket — it must not read config at
    all (unlike public_origin(), which is config-first)."""
    monkeypatch.setattr(
        pu, "platform_webhook_base_url", lambda: "https://configured.example/api/v1/telephony")
    req = _FakeRequest(
        headers={"x-forwarded-host": "public.example.com", "x-forwarded-proto": "https"},
        netloc="internal-container:8000", scheme="http")
    assert pu.origin_from_headers(req) == "https://public.example.com"


def test_origin_from_headers_matches_public_origin_header_fallback(monkeypatch) -> None:
    """origin_from_headers() must behave identically to public_origin()'s own
    header-fallback path when config is unset."""
    monkeypatch.setattr(pu, "platform_webhook_base_url", lambda: None)
    req = _FakeRequest(headers={}, netloc="example.com", scheme="https")
    assert pu.origin_from_headers(req) == pu.public_origin(req) == "https://example.com"
