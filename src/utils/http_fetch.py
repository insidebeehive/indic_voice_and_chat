"""Shared SSRF-safe server-side URL fetch helper.

Any code path that fetches an attacker-influenceable URL server-side — a
caller-supplied ``media_url``, a vendor "go fetch this recording" webhook
field — must route through here instead of calling ``httpx`` directly. It
centralizes three protections:

- **Public-host check**: the hostname must not resolve to a private/loopback/
  link-local/reserved/multicast address (blocks SSRF against internal
  services).
- **Host allowlisting** (opt-in via ``allowed_host_pattern`` /
  ``extra_allowed_hosts``): for vendor URLs (Twilio/Stringee) where we know
  the expected host(s) up front, restrict the fetch to just those — this is
  what stops an attacker-supplied webhook URL from redirecting our
  credentialed request anywhere else.
- **Bounded, streamed fetch**: a byte-size cap and no redirect-following, so
  a malicious/huge/redirecting response can't be used to exfiltrate data or
  pivot the request elsewhere.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import SplitResult, urlsplit

import httpx

MAX_FETCH_BYTES = 1 * 1024 * 1024  # match chat.py's historical cap value
FETCH_TIMEOUT = httpx.Timeout(20.0, connect=10.0)  # match chat.py's historical value


def is_public_host(hostname: str) -> bool:
    """Reject hostnames resolving to private/loopback/link-local addresses —
    guards against using a fetch to probe internal network services."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def assert_safe_url(
    url: str,
    *,
    allowed_host_pattern: str | None = None,
    extra_allowed_hosts: set[str] | None = None,
    require_https: bool = True,
) -> SplitResult:
    """Validate ``url`` is safe to fetch server-side. Raises ``ValueError`` on
    rejection; returns the parsed ``SplitResult`` on success.

    ``allowed_host_pattern``, if given, MUST be fully anchored (start with
    ``^`` and end with ``$``) — this is a hard invariant so a loose/unanchored
    pattern can never be passed by a caller (it would allow a bypass like
    ``api.evil.twilio.com.attacker.com`` matching an unanchored
    ``api.twilio.com`` pattern).

    Check order matters: when an allowlist (``allowed_host_pattern`` /
    ``extra_allowed_hosts``) is configured, it is enforced BEFORE
    ``is_public_host`` ever runs a DNS lookup. Resolving a hostname is itself
    a probe/exfil primitive, so a non-allowlisted host has no reason to be
    resolved at all. ``is_public_host`` only runs for a host that already
    passed the allowlist, or for a caller (like chat.py's untrusted
    ``media_url``) that configured no allowlist in the first place.
    """
    # Caller-programming-error check: unconditional, before anything per-request.
    if allowed_host_pattern is not None:
        if not (allowed_host_pattern.startswith("^") and allowed_host_pattern.endswith("$")):
            raise ValueError("allowed_host_pattern must be fully anchored (^...$)")

    parts = urlsplit(url)
    if require_https and parts.scheme != "https":
        raise ValueError("url must be an https URL")
    if not parts.hostname:
        raise ValueError("url must be an https URL")

    if allowed_host_pattern is not None or extra_allowed_hosts:
        hostname = parts.hostname.lower()
        host_ok = False
        if allowed_host_pattern is not None and re.fullmatch(allowed_host_pattern, hostname):
            host_ok = True
        if not host_ok and extra_allowed_hosts and hostname in {
            h.lower() for h in extra_allowed_hosts
        }:
            host_ok = True
        if not host_ok:
            raise ValueError(f"url host {hostname!r} is not allowlisted")

    if not is_public_host(parts.hostname):
        raise ValueError("url resolves to a non-public address")

    return parts


async def fetch_capped(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
    allowed_host_pattern: str | None = None,
    extra_allowed_hosts: set[str] | None = None,
    max_bytes: int = MAX_FETCH_BYTES,
    accept_content_types: tuple[str, ...] | None = None,
    timeout: httpx.Timeout | None = None,
) -> tuple[bytes, str]:
    """Fetch ``url`` server-side with SSRF guards + a byte-size cap.

    Validates via ``assert_safe_url`` first (https required), then streams
    the response body, rejecting it if it exceeds ``max_bytes`` or (when
    ``accept_content_types`` is given) if the response Content-Type doesn't
    start with one of them. Redirects are NOT followed — the caller gets a
    ``ValueError`` instead, since blindly following a redirect on a
    credentialed request is itself an SSRF vector.

    ``timeout``, if given, overrides the module default ``FETCH_TIMEOUT`` for
    this call — some callers (e.g. a live IVR turn fetch) need a tighter
    worst-case stall than the default.
    """
    assert_safe_url(
        url,
        allowed_host_pattern=allowed_host_pattern,
        extra_allowed_hosts=extra_allowed_hosts,
        require_https=True,
    )

    async with httpx.AsyncClient(timeout=timeout or FETCH_TIMEOUT) as client:
        async with client.stream("GET", url, headers=headers, auth=auth) as resp:
            if resp.is_redirect:
                raise ValueError("url redirects are not followed")
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if accept_content_types and not any(
                content_type.startswith(prefix) for prefix in accept_content_types
            ):
                raise ValueError(f"unsupported content-type: {content_type or 'unknown'}")
            chunks = bytearray()
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                chunks.extend(chunk)
                if len(chunks) > max_bytes:
                    raise ValueError("url content exceeds size limit")
            return bytes(chunks), content_type
