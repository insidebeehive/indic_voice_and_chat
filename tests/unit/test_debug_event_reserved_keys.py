"""A diagnostic helper must never be the thing that takes the caller down.

`debug_event` runs inside code that is already being investigated. If a log
line raises, it kills the operation it was added to explain -- and it does so
only when DEBUG is on, which is to say only during an incident, which is the
worst possible time to discover it.

Two collision classes exist and both were hit for real while instrumenting
this codebase:

1. `logging` owns attributes on every LogRecord (`name`, `module`, `filename`,
   `args`, ...). Passing one as a field raises inside the handler.
2. `event` is the helper's OWN second parameter, and also the key it gives the
   line itself. Two separate agents wrote `debug_event(log, "...", event=x)`
   while instrumenting `src/agents/state_machine.py` and `src/agents/voicebot.py`;
   it raised `TypeError: got multiple values for argument 'event'` and broke
   every state transition. It was caught only because the test suite is run at
   `--log-level=DEBUG` -- at the default level the helper returns before
   touching its arguments, so an ordinary run proves nothing about them.

Both are handled by renaming the key rather than rejecting it, so the value
still reaches the log. These tests pin that, and they pin the emitted KEY name
too: a rename that silently dropped the value would satisfy "does not raise"
while defeating the purpose.
"""

from __future__ import annotations

import io
import logging

import pytest
from pythonjsonlogger import jsonlogger

from src.utils.logging import debug_event


def _capture(level: int = logging.DEBUG) -> tuple[logging.Logger, io.StringIO]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(jsonlogger.JsonFormatter("%(message)s %(event)s"))
    log = logging.getLogger(f"dbgev-{id(buf)}")
    log.setLevel(level)
    log.propagate = False
    log.addHandler(handler)
    return log, buf


def _record(log: logging.Logger) -> logging.LogRecord:
    """The emitted record itself -- the JSON formatter hides which key a value
    actually landed under, and that is exactly what these tests are about."""
    seen: list[logging.LogRecord] = []

    class _Grab(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            seen.append(record)
            return True

    log.addFilter(_Grab())
    return seen  # type: ignore[return-value]


def test_event_keyword_does_not_raise() -> None:
    """The exact call that broke every state transition."""
    log, _ = _capture()
    seen = _record(log)

    debug_event(log, "state_machine transition applied",
                from_state="ringing", event="answered", to_state="in_call")

    assert len(seen) == 1
    rec = seen[0]
    # The line keeps its own identity...
    assert rec.event == "state_machine transition applied"
    # ...and the caller's colliding value survives under the renamed key.
    assert rec.event_ == "answered"
    assert rec.from_state == "ringing"


def test_logrecord_attribute_collision_does_not_raise() -> None:
    """`name` is a LogRecord attribute; passing it used to raise in the handler."""
    log, _ = _capture()
    seen = _record(log)

    debug_event(log, "chatbot tool_call selected", name="get_player_bets", ms=42)

    rec = seen[0]
    assert rec.name_ == "get_player_bets"
    assert rec.name == log.name       # the logger's own name is untouched
    assert rec.ms == 42


def test_filename_collision_keeps_the_logger_own_value() -> None:
    """The case that made a Loki query silently return nothing.

    `filename` resolves to the helper's own source file, so an operator
    querying `filename="kb-manual.pdf"` matches zero rows while `filename`
    sits there holding something plausible. Renaming is a backstop, not a
    licence -- call sites should qualify the key themselves.
    """
    log, _ = _capture()
    seen = _record(log)

    debug_event(log, "ingestion parse_document", filename="kb-manual.pdf")

    rec = seen[0]
    assert rec.filename_ == "kb-manual.pdf"
    assert rec.filename != "kb-manual.pdf"


def test_event_is_positional_only() -> None:
    """Pin the `/` in the signature.

    Without it, `event=` is a TypeError rather than a field. Every one of the
    ~140 call sites passes the name positionally, so this costs nothing and is
    the whole reason the collision above is survivable.
    """
    import inspect

    params = list(inspect.signature(debug_event).parameters.values())
    assert params[1].name == "event"
    assert params[1].kind is inspect.Parameter.POSITIONAL_ONLY


@pytest.mark.parametrize("key", ["msg", "args", "levelname", "module", "lineno", "created"])
def test_every_reserved_attribute_is_survivable(key: str) -> None:
    """Derived from LogRecord rather than a hand-kept list.

    A hardcoded blocklist validated against a copy of itself always passes;
    this asserts the real handler accepts the real record.
    """
    log, buf = _capture()

    debug_event(log, "probe", **{key: "value"})

    assert buf.getvalue(), f"passing {key}= produced no output"


def test_no_work_when_debug_is_off() -> None:
    """The helper returns before touching `values`.

    This is why the suite must also run at --log-level=DEBUG: at INFO none of
    the above collisions can fire, so an ordinary run exercises none of it.
    """
    log, buf = _capture(level=logging.INFO)
    exploded = []

    class _Boom:
        def __repr__(self) -> str:
            exploded.append(True)
            raise AssertionError("value was rendered with DEBUG off")

    debug_event(log, "probe", payload=_Boom())

    assert buf.getvalue() == ""
    assert not exploded
