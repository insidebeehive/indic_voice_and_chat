"""Best-effort client IP for logging and correlation.

The app runs behind a reverse proxy (Northflank). ``X-Forwarded-For`` is a
comma-separated, left-to-right chain in which each proxy hop APPENDS the peer
address it saw. A client can forge entries at the START of that header before
it ever reaches the first real proxy, so the leading entries are
attacker-controlled in the general case. Given exactly N trusted proxy hops in
front of this app, the Nth entry from the RIGHT is the one written by the
outermost trusted hop -- i.e. ``parts[-hops]``, 1-indexed from the right.
``hops`` comes from ``VOX_TRUSTED_PROXY_HOPS`` (default 1, clamped to >= 1).

    hops  X-Forwarded-For                     result
    1     "9.9.9.9, 203.0.113.7"              203.0.113.7   (leading entry is client-forged)
    1     "203.0.113.7"                       203.0.113.7   (single entry, parts[-1])
    2     "9.9.9.9, 203.0.113.7, 10.0.0.5"    203.0.113.7   (parts[-2])
    2     "203.0.113.7"                       (None, "xff_insufficient_hops")

Resolution order: ``X-Forwarded-For`` (per the formula above) -> ``X-Real-IP``
-> the socket peer (``.client.host``) -> ``(None, "unknown")``.

SECURITY -- the value returned here is best-effort and proxy-reported. NEVER
use it in an authorization decision: no IP allowlisting, no granting access, no
"trusted network" shortcut. It is for logging, correlation and telemetry only.
Anything that gates access on it inherits the forgeability of a header this
process does not control end to end.

Why ``xff_insufficient_hops`` deliberately does NOT fall back to ``X-Real-IP``
or the socket peer: a chain shorter than ``hops`` means the request did not
traverse our infrastructure the way we are configured to believe it did. In
that state every entry in the header -- and any ``X-Real-IP`` next to it --
could have been written by the client, so picking one anyway would launder a
forged value into the logs under a source label implying it was
proxy-verified. Admitting "unknown" is strictly better than guessing. A
sustained nonzero rate of ``xff_insufficient_hops`` is itself the signal that
``VOX_TRUSTED_PROXY_HOPS`` is misconfigured for the deployment -- that is the
intended detection mechanism, not a bug to paper over with a fallback.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any, Optional

from starlette.datastructures import Address, Headers

_TRUSTED_PROXY_HOPS_DEFAULT = 1


def _trusted_proxy_hops() -> int:
    """Number of trusted proxy hops in front of this app.

    Read from ``VOX_TRUSTED_PROXY_HOPS`` at call time (not import time, so
    tests and re-deploys can change it), clamped to >= 1, defaulting to
    ``_TRUSTED_PROXY_HOPS_DEFAULT`` when unset or unparseable.
    """
    raw = os.getenv("VOX_TRUSTED_PROXY_HOPS")
    if raw is None:
        return _TRUSTED_PROXY_HOPS_DEFAULT
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return _TRUSTED_PROXY_HOPS_DEFAULT
    return max(1, value)


class _ScopeConnection:
    """Minimal Request/WebSocket look-alike built from a raw ASGI ``scope``.

    ``ClientIPMiddleware`` runs before any Request/WebSocket object exists, but
    ``client_ip()`` is also called with real Request/WebSocket objects. Rather
    than branching inside ``client_ip()`` on "scope dict or connection object",
    the middleware wraps the scope in this adapter so ``client_ip()`` has
    exactly ONE code path: plain attribute access on ``.headers`` / ``.client``,
    matching ``origin_from_headers()``'s duck-typed convention in
    ``src/utils/public_url.py``.

    ``.headers`` uses ``Headers(scope=scope)``, which references
    ``scope["headers"]`` directly (no copy). ``.client`` reproduces Starlette's
    own ``Request.client`` / ``WebSocket.client`` shape exactly -- an ``Address``
    named tuple with a ``.host`` attribute, or ``None`` when the server did not
    report a peer.
    """

    __slots__ = ("headers", "client")

    def __init__(self, scope: dict[str, Any]) -> None:
        self.headers = Headers(scope=scope)
        host_port = scope.get("client")
        self.client = Address(*host_port) if host_port is not None else None


def client_ip(request_or_websocket: Any) -> tuple[Optional[str], str]:
    """Return ``(ip, source)`` for a Starlette ``Request`` or ``WebSocket``.

    Duck-typed on ``.headers`` and ``.client`` (same convention as
    ``origin_from_headers()``); ``_ScopeConnection`` satisfies it too, so
    middleware and route code share one path.

    ``source`` is one of ``"x_forwarded_for"``, ``"xff_insufficient_hops"``,
    ``"x_real_ip"``, ``"socket"``, ``"unknown"``. See the module docstring for
    the trust model -- in particular, never use this value to authorize.
    """
    headers = request_or_websocket.headers

    # Repeated header lines are semantically one comma-joined value (RFC 7230),
    # so join them in order before splitting. A PRESENT X-Forwarded-For is
    # authoritative: it is never allowed to fall through to the weaker sources
    # below, even when it yields nothing usable.
    forwarded_values = headers.getlist("x-forwarded-for")
    if forwarded_values:
        parts = [part.strip() for part in ", ".join(forwarded_values).split(",")]
        parts = [part for part in parts if part]
        hops = _trusted_proxy_hops()
        if len(parts) < hops:
            return None, "xff_insufficient_hops"
        return parts[-hops], "x_forwarded_for"

    real_ip = headers.get("x-real-ip")
    if real_ip and real_ip.strip():
        return real_ip.strip(), "x_real_ip"

    client = request_or_websocket.client
    if client is not None and client.host:
        return client.host, "socket"

    return None, "unknown"


# Ambient (ip, source) for the connection being served in this context. The
# stored ip may itself be None (e.g. "unknown"/"xff_insufficient_hops"), which
# is why the value type is tuple[Optional[str], str] rather than tuple[str, str].
_client_ip_ctx: ContextVar[Optional[tuple[Optional[str], str]]] = ContextVar(
    "client_ip_ctx", default=None
)


def current_client_ip() -> tuple[Optional[str], str]:
    """Ambient ``(ip, source)`` set by ``ClientIPMiddleware`` for the current
    context, or ``(None, "no_context")`` outside any request/websocket."""
    value = _client_ip_ctx.get()
    if value is None:
        return None, "no_context"
    return value


class ClientIPMiddleware:
    """Pure-ASGI middleware that resolves the client IP once per connection and
    publishes it on a ContextVar for the duration of that connection.

    Deliberately NOT a ``BaseHTTPMiddleware`` subclass: BaseHTTPMiddleware only
    runs for ``http`` scopes, and this app's largest credential-guessing surface
    is its WebSocket routes -- those must be covered too.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        token = _client_ip_ctx.set(client_ip(_ScopeConnection(scope)))
        try:
            await self.app(scope, receive, send)
        finally:
            _client_ip_ctx.reset(token)
