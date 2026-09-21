"""Ambient chat session/ticket ids for log correlation.

Every log line emitted while handling a chat connection should carry the
``session_id`` and ``ticket_id`` it belongs to, so a line can be traced back to
a conversation. Passing them through ``extra={}`` at each call site only works
where the caller happens to hold them: ``_synthesize_reply_audio``
(``src/api/chat.py``) takes a tenant and some text and has neither, and every
adapter under ``src/providers/`` is handed a config dict with no session
context at all. Those are exactly the layers a debug session needs to follow.

So they are published here instead and stamped onto records by
``_ChatContextLogFilter`` (``src/utils/logging.py``), the same way
``src/utils/trace_id.py`` handles the per-turn trace id. A call site that holds
better values can still pass them explicitly — the filter never overwrites
what ``extra={}`` already set.

``trace_id`` is per TURN and these are per CONNECTION: a session has many
turns, so the two answer different questions and both are worth having.

Mirrors the getter/setter/scope shape of ``src/utils/trace_id.py`` and
``src/auth/audit.py``'s ``_admin_label_ctx``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Optional

_session_id_ctx: ContextVar[Optional[str]] = ContextVar("chat_session_id", default=None)
_ticket_id_ctx: ContextVar[Optional[str]] = ContextVar("chat_ticket_id", default=None)


def current_chat_session_id() -> Optional[str]:
    """Ambient chat session id, or None outside any chat scope."""
    return _session_id_ctx.get()


def current_chat_ticket_id() -> Optional[str]:
    """Ambient chat ticket id, or None outside any chat scope (or when the
    session has no ticket — it is optional on ChatSession)."""
    return _ticket_id_ctx.get()


def set_chat_context(
    session_id: Optional[str], ticket_id: Optional[str] = None,
) -> tuple[Token, Token]:
    """Publish the ids for the remainder of this context.

    Returns both reset tokens; callers owning a scope should reset them (see
    ``chat_context_scope``, which is the ergonomic form).
    """
    return _session_id_ctx.set(session_id), _ticket_id_ctx.set(ticket_id)


@contextmanager
def chat_context_scope(
    session_id: Optional[str], ticket_id: Optional[str] = None,
) -> Iterator[None]:
    """Bind the ids for the duration of the block, then restore.

    Automatically per-asyncio-Task scoped by ContextVar semantics, so two
    concurrent connections cannot see each other's ids — which matters here
    more than for trace_id, since a wrong session id on a log line is worse
    than none: it points an investigation at another customer's conversation.
    """
    session_token, ticket_token = set_chat_context(session_id, ticket_id)
    try:
        yield
    finally:
        _session_id_ctx.reset(session_token)
        _ticket_id_ctx.reset(ticket_token)
