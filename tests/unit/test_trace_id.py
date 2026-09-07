"""Per-turn trace id: the ContextVar/scope in src/utils/trace_id.py and the
``_TraceIdLogFilter`` that stamps it onto structured JSON log lines.

Throwaway-by-design stopgap (see src/utils/trace_id.py's module docstring):
this exists so a duplicate/repeated turn shows up as two turn-start log lines
with different trace ids close together in time.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re

import pytest

from src.utils.logging import configure_logging
from src.utils.trace_id import current_trace_id, new_trace_id, trace_id_scope

E2E_LOGGER = "tests.unit.test_trace_id.e2e"

_HEX12_RE = re.compile(r"^[0-9a-f]{12}$")


# --------------------------------------------------------------------------
# new_trace_id / current_trace_id / trace_id_scope basics.
# --------------------------------------------------------------------------

def test_new_trace_id_is_12_hex_chars() -> None:
    value = new_trace_id()
    assert _HEX12_RE.match(value), value


def test_new_trace_id_successive_calls_differ() -> None:
    assert new_trace_id() != new_trace_id()


def test_current_trace_id_outside_any_scope() -> None:
    assert current_trace_id() is None


def test_trace_id_scope_sets_and_restores() -> None:
    assert current_trace_id() is None
    with trace_id_scope("abc123"):
        assert current_trace_id() == "abc123"
    assert current_trace_id() is None


def test_trace_id_scope_nesting_restores_outer_value() -> None:
    """Standard ContextVar.reset(token) correctness: exiting the inner scope
    must restore the outer value, not clear it to None."""
    with trace_id_scope("outer"):
        assert current_trace_id() == "outer"
        with trace_id_scope("inner"):
            assert current_trace_id() == "inner"
        assert current_trace_id() == "outer"
    assert current_trace_id() is None


# --------------------------------------------------------------------------
# Concurrency isolation across asyncio Tasks.
# --------------------------------------------------------------------------

def test_concurrent_tasks_do_not_leak_trace_id() -> None:
    """Two concurrent turns on different asyncio Tasks must never observe
    each other's trace id -- the property that makes this safe for
    concurrent WS connections/turns."""

    async def _run() -> tuple[list[str], list[str]]:
        seen_a: list[str] = []
        seen_b: list[str] = []

        async def turn(label: str, value: str, seen: list[str]) -> None:
            with trace_id_scope(value):
                for _ in range(5):
                    await asyncio.sleep(0)  # yield, let the other task interleave
                    seen.append(current_trace_id())

        await asyncio.gather(
            turn("a", "trace-a", seen_a),
            turn("b", "trace-b", seen_b),
        )
        return seen_a, seen_b

    seen_a, seen_b = asyncio.run(_run())
    assert seen_a == ["trace-a"] * 5
    assert seen_b == ["trace-b"] * 5


# --------------------------------------------------------------------------
# _TraceIdLogFilter, end to end through configure_logging().
# --------------------------------------------------------------------------

@pytest.fixture
def json_log_stream():
    """Install the REAL production logging config, redirected to a buffer.

    Mirrors tests/unit/test_client_ip.py's fixture of the same name/shape.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level

    configure_logging("INFO")
    handler = root.handlers[-1]
    buffer = io.StringIO()
    handler.setStream(buffer)
    try:
        yield buffer
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _records(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def _record_with_message(buffer: io.StringIO, message: str) -> dict:
    matches = [r for r in _records(buffer) if r.get("message") == message]
    assert len(matches) == 1, f"expected exactly one {message!r} record, got {matches!r}"
    return matches[0]


def test_log_line_inside_scope_carries_trace_id(json_log_stream: io.StringIO) -> None:
    log = logging.getLogger(E2E_LOGGER)
    with trace_id_scope("scoped-value"):
        log.info("inside scope")

    record = _record_with_message(json_log_stream, "inside scope")
    assert record["trace_id"] == "scoped-value"


def test_log_line_outside_scope_has_no_trace_id(json_log_stream: io.StringIO) -> None:
    logging.getLogger(E2E_LOGGER).info("outside scope")

    record = _record_with_message(json_log_stream, "outside scope")
    assert "trace_id" not in record


def test_filter_does_not_overwrite_explicit_trace_id(json_log_stream: io.StringIO) -> None:
    """A caller that passes trace_id via extra={} wins over the ambient
    value, including while a scope is active (matches _AdminLabelLogFilter's
    and _ClientIPLogFilter's "don't clobber an explicit value" behavior)."""
    log = logging.getLogger(E2E_LOGGER)
    with trace_id_scope("ambient-value"):
        log.info("explicit trace id", extra={"trace_id": "explicit"})

    record = _record_with_message(json_log_stream, "explicit trace id")
    assert record["trace_id"] == "explicit"
