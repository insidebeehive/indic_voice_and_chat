"""Ambient per-turn trace id — a cheap, deliberately-temporary stopgap.

Threads a short opaque id through the existing structured JSON log lines so a
duplicate/repeated turn (two turn-start log lines with different trace ids
close together in time) can be spotted by grepping logs, without needing a
customer's screenshot. This is throwaway-by-design: it gets deleted once real
OpenTelemetry tracing lands later, not maintained alongside it.

Mirrors src/auth/audit.py's `_admin_label_ctx` ContextVar pattern (getter,
setter, reset, and a `*_scope` context manager for bounded scopes).
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Optional

_trace_id_ctx: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


def current_trace_id() -> Optional[str]:
    """Ambient trace id for this context, or None outside any turn scope."""
    return _trace_id_ctx.get()


def new_trace_id() -> str:
    """Mint a new opaque trace id, one per turn (not per session).

    Matches src/bootstrap.py's short-opaque-id convention
    (``uuid.uuid4().hex[:12]``).
    """
    return uuid.uuid4().hex[:12]


def set_trace_id(value: str) -> Token:
    """Publish `value` for the remainder of this context. Returns the reset
    token; callers that own a scope should reset it (see trace_id_scope)."""
    return _trace_id_ctx.set(value)


def reset_trace_id(token: Token) -> None:
    _trace_id_ctx.reset(token)


@contextmanager
def trace_id_scope(value: str):
    """set/reset-in-finally scope for the duration of one turn.

    Automatically per-asyncio-Task scoped (ContextVar semantics), so
    concurrent turns on different Tasks never see each other's trace id, and
    nesting correctly restores the outer value on exit.
    """
    token = _trace_id_ctx.set(value)
    try:
        yield
    finally:
        _trace_id_ctx.reset(token)
