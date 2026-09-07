"""Shared credential-safe URL redaction for log messages.

Extracted from ``src.utils.logging.LokiPushHandler._redacted_url`` (Phase 1)
so other modules that log a URL on failure (e.g.
``src.observability.turn_metrics_push``, Phase 2) can reuse the exact same
redaction instead of re-implementing it slightly differently.
"""

from __future__ import annotations


def redact_url(url: str) -> str:
    """Best-effort credential-safe rendering of ``url`` for a log/warning
    message: strips userinfo from the netloc, and drops the query string and
    fragment entirely (some push URLs carry an API key as a query param, not
    basic auth). Anything that doesn't parse as a normal http(s) URL is NOT
    echoed at all -- e.g. a malformed "user:pass@host/path" with no
    "https://" prefix parses its userinfo as the scheme, which would
    otherwise leak past a netloc-only check."""
    try:
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return "<redacted>"
        netloc = parts.netloc.rsplit("@", 1)[-1] if "@" in parts.netloc else parts.netloc
        return urlunsplit(parts._replace(netloc=netloc, query="", fragment=""))
    except Exception:
        return "<redacted>"
