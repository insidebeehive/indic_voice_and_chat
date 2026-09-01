"""Derive the public scheme://netloc this app is reachable at, for building
absolute callback/websocket URLs (TwiML `<Stream url=...>`, Stringee event
URLs, transfer-result webhook URLs, and — later — provider signature
verification, which needs the exact URL a webhook request arrived at).

Two derivations live here:

- ``origin_from_headers()`` — purely header-derived (``x-forwarded-host``/
  ``x-forwarded-proto``, honoring a reverse proxy, else falling back to the
  request's/websocket's own URL). Never reads any config. This is what
  ``src/api/telephony_hooks.py``'s ``_ws_stream_url``/``_forwarded_base``/
  ``_stringee_base`` use directly — inbound telephony webhook-response URLs
  must always resolve to the host that actually received the request, never
  to a possibly-different configured value (see
  ``platform_webhook_base_url()``'s docstring in ``src/config_tenant.py``).
- ``public_origin()`` — config-first, header-fallback:

  1. If ``platform_webhook_base_url()`` (``src/config_tenant.py``) is set, use
     its ``scheme://netloc`` — and ONLY that. That value is documented and used
     ending in a path suffix (e.g. ``https://host/api/v1/telephony``), so its
     ``path`` must be discarded here; callers append their own path on top of
     the bare origin this returns. Naively string-concatenating the configured
     value with a request path doubles that suffix (see
     ``src/chatbot/deposit_verification.py`` for the same fix, applied first).
  2. Otherwise, delegate to ``origin_from_headers()``.

  Used by ``_fire_transfer_webhook`` (``src/api/telephony_live_bridge.py``),
  where the platform base URL IS the intended target (an outbound callout's
  Answer URL, not an inbound webhook response).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from src.config_tenant import platform_webhook_base_url


def origin_from_headers(request_or_websocket: Any) -> str:
    """Bare ``scheme://netloc`` derived ONLY from the request's/websocket's own
    headers/URL — no config check. Honors a reverse proxy's
    ``x-forwarded-host``/``x-forwarded-proto``, else falls back to the
    request's/websocket's own ``.url``. Returns ``""`` when
    ``request_or_websocket`` is ``None``.
    """
    if request_or_websocket is None:
        return ""

    headers = request_or_websocket.headers
    url = request_or_websocket.url
    host = headers.get("x-forwarded-host") or url.netloc
    forwarded_proto = headers.get("x-forwarded-proto")
    secure = forwarded_proto == "https" or url.scheme in ("https", "wss")
    scheme = "https" if secure else "http"
    return f"{scheme}://{host}"


def public_origin(request_or_websocket: Any = None) -> str:
    """Bare ``scheme://netloc`` this app is publicly reachable at.

    Config-first: when ``platform_webhook_base_url()`` is set, this NEVER
    includes that value's own path suffix (e.g. ``/api/v1/telephony``) —
    callers append their own path. Falls back to ``origin_from_headers()``
    when no platform base URL is configured.
    """
    cfg = platform_webhook_base_url()
    if cfg:
        parsed = urlsplit(cfg if "://" in cfg else f"https://{cfg}")
        if parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"

    return origin_from_headers(request_or_websocket)


def public_http_url(request_or_websocket: Any, path: str) -> str:
    """``public_origin(...)`` + ``/`` + ``path`` (leading slashes normalized)."""
    return f"{public_origin(request_or_websocket)}/{path.lstrip('/')}"


def public_ws_url(request_or_websocket: Any, path: str) -> str:
    """Same as ``public_http_url`` but forces a ``ws``/``wss`` scheme
    (``http`` -> ``ws``, ``https`` -> ``wss``)."""
    origin = public_origin(request_or_websocket)
    parsed = urlsplit(origin)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{ws_scheme}://{parsed.netloc}/{path.lstrip('/')}"
