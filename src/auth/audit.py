"""Shared audit-logging helpers.

Provides ``token_fingerprint`` (a one-way, non-reversible fingerprint safe to
place in log lines wherever a raw token/id would otherwise leak PII), the
``log_denied`` auth-rejection primitive with its flood-suppression window, and
the ambient ``admin_label`` ContextVar that attributes admin-authenticated
requests to a named operator.

Import direction: this module may import ``src.utils.client_ip``; it must
NEVER import ``src.utils.logging`` (which imports this module for its
admin-label log filter).
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Optional

from src.utils.client_ip import current_client_ip

log = logging.getLogger(__name__)


def token_fingerprint(value: str, *, domain: str = "vox-logfp-v1") -> str:
    """Returns a 12-hex-char fingerprint of `value`, safe to log.

    NOT the same hash used for credential verification (see
    src/auth/context.py's hash_api_token) — deliberately domain-separated
    via the `domain` prefix so this fingerprint can never be confused with,
    or substituted for, the actual stored credential-verification hash.
    Publishing a prefix of the real verification hash would be a partial
    disclosure of the stored secret; this is a distinct derivation.

    Use a distinct `domain` string per value-type (e.g. "vox-logfp-sid-v1"
    for session ids, "vox-logfp-tel-v1" for phone numbers) so a fingerprint
    computed in one context can never be cross-referenced against another.
    """
    return hashlib.sha256(f"{domain}:{value}".encode("utf-8")).hexdigest()[:12]


# R1: the dedup key is (reason, client_ip) — deliberately NOT including any
# token identifier. In real credential stuffing the attacker presents a
# DIFFERENT token on every request, so keying on token_fp would mint a fresh
# dedup key each request and suppression would never fire — defeating the
# exact threat this exists to bound. token_fp still appears as a plain field
# inside each sampled record; it is just not part of the decision to suppress.
_SUPPRESS_N = 10              # full records emitted per key per window
_SUPPRESS_WINDOW_S = 60.0
_MAX_TRACKED_KEYS = 10_000    # bounded so an IP-rotating attacker can't grow this unbounded

# key -> {"count": int, "window_start": float}
_suppression_state: dict[tuple[str, str], dict] = {}


def log_denied(
    level: int,
    message: str,
    *,
    event: str,
    reason: str,
    route: str | None = None,
    tenant: str | None = None,
    tenant_id: str | None = None,
    token_fp: str | None = None,
    suppress: bool = True,
    exc_info: Any = False,
    **extra: Any,
) -> None:
    """Log an auth-rejection-shaped record with a stable, structured extra dict.

    When `suppress` is True (the default — pass False for anything that must
    never be sampled away, e.g. compliance denials), applies a
    per-(reason, client_ip) suppression window: the first ``_SUPPRESS_N``
    occurrences of a given (reason, client_ip) pair within
    ``_SUPPRESS_WINDOW_S`` seconds are logged in full; further occurrences in
    the same window are silently counted; at window close (detected lazily, on
    the next call for that key after the window has elapsed) ONE summary record
    is emitted carrying ``suppressed_count``. Note: this flush is lazy/
    pull-based — if no further call arrives for a given key after its window
    elapses (e.g. an attacker stops entirely, or permanently rotates to a new
    client_ip), that window's suppressed count is never flushed and is lost.
    This is an accepted tradeoff to avoid a background timer; it means the
    guarantee is "no full record is silently dropped while requests for that
    key continue" rather than an unconditional guarantee.

    ``client_ip`` / ``client_ip_source`` are read ambiently and stamped onto
    every record by ``_ClientIPLogFilter`` in src/utils/logging.py — do NOT
    pass them here. This function reads ``current_client_ip()`` only to build
    the suppression KEY.

    ``exc_info`` is an explicit parameter rather than part of ``**extra``:
    ``logging`` refuses to let ``extra`` overwrite a reserved LogRecord
    attribute, so an ``exc_info`` key inside ``extra`` would raise KeyError.

    Keys in ``**extra`` must likewise avoid reserved LogRecord attribute names
    (``name``, ``msg``, ``args``, ``message``, ``module``, ``filename``,
    ``lineno``, ``levelname``, ``created``, ``exc_text``, ``stack_info``,
    ``taskName``, …).
    """
    payload: dict[str, Any] = {"event": event, "reason": reason}
    if route is not None:
        payload["route"] = route
    if tenant is not None:
        payload["tenant"] = tenant
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    if token_fp is not None:
        payload["token_fp"] = token_fp
    payload.update(extra)

    if not suppress:
        log.log(level, message, extra=payload, exc_info=exc_info)
        return

    ip, source = current_client_ip()
    key = (reason, ip or source)     # ip may be None (unknown / xff_insufficient_hops)
    now = time.monotonic()
    state = _suppression_state.get(key)

    if state is None:
        if len(_suppression_state) >= _MAX_TRACKED_KEYS:
            _evict_oldest(level)     # flushes the evicted key's pending summary first
        _suppression_state[key] = {"count": 1, "window_start": now}
        log.log(level, message, extra=payload, exc_info=exc_info)
        return

    if now - state["window_start"] >= _SUPPRESS_WINDOW_S:
        _flush(key, state, level)    # emits the OLD window's summary if count > _SUPPRESS_N
        state["count"] = 1
        state["window_start"] = now
        log.log(level, message, extra=payload, exc_info=exc_info)
        return

    state["count"] += 1
    if state["count"] <= _SUPPRESS_N:
        log.log(level, message, extra=payload, exc_info=exc_info)
    # else: counted silently; the summary is emitted at window close.


def _flush(key: tuple[str, str], state: dict, level: int) -> None:
    """Emit the closing summary for `key`'s finished window, if it suppressed
    anything. Level is taken from the current call: every `reason` in this
    codebase maps to exactly one level at every one of its call sites, so the
    closing level always equals the level of the window it summarizes."""
    suppressed = state["count"] - _SUPPRESS_N
    if suppressed > 0:
        # key[1] (the IP/source this window actually belongs to) is stamped
        # explicitly as `suppressed_client_ip` rather than relying on the
        # ambient _ClientIPLogFilter's `client_ip`. On the eviction path
        # (_evict_oldest) the flushed key can belong to a totally different
        # client than whoever's request triggered the eviction, so the
        # ambient client_ip would attribute the summary to the wrong IP.
        # Using a distinct field name (instead of pre-populating `client_ip`)
        # also avoids any ambiguity/ordering dependency with the filter.
        log.log(level, "auth rejections suppressed", extra={
            "event": "auth_rejected",
            "reason": key[0],
            "suppressed_client_ip": key[1],
            "suppressed_count": suppressed,
            "suppressed_window_s": _SUPPRESS_WINDOW_S,
            "suppression_limit": _SUPPRESS_N,
        })


def _evict_oldest(level: int) -> None:
    """Drop the single oldest tracked key (linear scan; the map is capped at
    _MAX_TRACKED_KEYS). Its pending summary is flushed first, so eviction under
    an IP-rotating flood still reports what it suppressed rather than losing it."""
    key = min(_suppression_state, key=lambda k: _suppression_state[k]["window_start"])
    _flush(key, _suppression_state[key], level)
    del _suppression_state[key]


def reset_suppression_state() -> None:
    """Clear all suppression windows. Test-only: without it, one test's
    rejections bleed into the next test's caplog assertions."""
    _suppression_state.clear()


# Ambient admin label for the request being served, set by
# src.auth.middleware.require_admin / require_admin_ws on SUCCESSFUL admin
# auth. Mirrors src/utils/client_ip.py's client-ip ContextVar pattern.
#
# require_admin is a plain `async def` FastAPI dependency awaited inline —
# it has no "after the response" hook to reset in, unlike middleware. This is
# safe without a reset because each ASGI request/connection runs in its own
# asyncio Task, and Task creation copies the context: a set() inside one
# request's task cannot leak into the server's base context or into a
# sibling request's task. admin_label_scope() below is provided for callers
# that DO own a bounded scope (tests, background tasks) and should reset
# when done.
_admin_label_ctx: ContextVar[Optional[str]] = ContextVar("admin_label_ctx", default=None)


def current_admin_label() -> Optional[str]:
    """Ambient admin label for this context, or None when the current request
    was not authenticated as a platform admin."""
    return _admin_label_ctx.get()


def set_admin_label(label: str) -> Token:
    """Publish `label` for the remainder of this context. Returns the reset
    token; callers that own a scope should reset it (see admin_label_scope)."""
    return _admin_label_ctx.set(label)


def reset_admin_label(token: Token) -> None:
    _admin_label_ctx.reset(token)


@contextmanager
def admin_label_scope(label: str):
    """set/reset-in-finally scope — for callers that DO own a bounded scope
    (tests, background tasks)."""
    token = _admin_label_ctx.set(label)
    try:
        yield
    finally:
        _admin_label_ctx.reset(token)
