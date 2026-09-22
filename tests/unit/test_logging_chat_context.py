"""Chat session/ticket ids reach log records from layers that never see them.

The point of the ContextVar+filter pair (src/utils/chat_context.py,
_ChatContextLogFilter) is correlation from code that holds no session object:
_synthesize_reply_audio takes a tenant and some text, and every adapter under
src/providers/ is handed a config dict. Threading the ids through extra={}
only reaches call sites that already have them, which is how the voice-reply
skip came to log a tenant_id and nothing that tied it to a conversation.
"""

from __future__ import annotations

import io
import logging

from pythonjsonlogger import jsonlogger

from src.utils.chat_context import chat_context_scope
from src.utils.logging import _ChatContextLogFilter, debug_event


def _capture() -> tuple[logging.Logger, io.StringIO]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(jsonlogger.JsonFormatter("%(message)s"))
    handler.addFilter(_ChatContextLogFilter())
    log = logging.getLogger(f"chatctx-{id(buf)}")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.addHandler(handler)
    return log, buf


def test_ids_reach_a_caller_that_never_received_them() -> None:
    """The whole point: a function with no session argument still emits them.
    Catches the filter not being applied, or the ContextVar not being read."""
    log, buf = _capture()

    def provider_layer() -> None:        # no session_id/ticket_id in scope
        debug_event(log, "provider_request", model="gemini-3.5-flash")

    with chat_context_scope("cs_abc123", "7525"):
        provider_layer()

    out = buf.getvalue()
    assert '"session_id": "cs_abc123"' in out
    assert '"ticket_id": "7525"' in out


def test_nothing_stamped_outside_a_chat_scope() -> None:
    """Most records in this system are emitted outside any chat connection.
    Stamping them with nulls would change the shape of every log line for no
    signal -- the same rule _TraceIdLogFilter follows."""
    log, buf = _capture()
    debug_event(log, "startup_event", x=1)

    out = buf.getvalue()
    assert "session_id" not in out
    assert "ticket_id" not in out


def test_an_explicit_value_is_not_overwritten() -> None:
    """A call site holding a better value keeps it -- e.g. a background task
    logging about a DIFFERENT session than the connection it runs under.
    Overwriting would point an investigation at the wrong conversation."""
    log, buf = _capture()
    with chat_context_scope("cs_ambient", "111"):
        debug_event(log, "explicit_wins", session_id="cs_explicit")

    out = buf.getvalue()
    assert '"session_id": "cs_explicit"' in out
    assert "cs_ambient" not in out


def test_scope_restores_on_exit_and_isolates_nesting() -> None:
    """A leaked id outlives its connection and mislabels later lines, which is
    worse than no id: it points at a conversation that was not involved."""
    log, buf = _capture()
    with chat_context_scope("cs_outer", "1"):
        with chat_context_scope("cs_inner", "2"):
            debug_event(log, "inner")
        debug_event(log, "outer_again")
    debug_event(log, "after")

    lines = [ln for ln in buf.getvalue().strip().splitlines()]
    assert '"session_id": "cs_inner"' in lines[0]
    assert '"session_id": "cs_outer"' in lines[1]
    assert "session_id" not in lines[2]
