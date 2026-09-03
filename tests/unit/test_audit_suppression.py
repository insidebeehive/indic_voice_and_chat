"""Tests for src.auth.audit's log_denied flood-suppression and the ambient
admin_label ContextVar / log filter.

Mirrors the fixture shapes used in tests/unit/test_client_ip.py: an autouse
fixture resets module-global suppression state between tests, and
`json_log_stream` installs the REAL production logging pipeline (handler +
filters + JsonFormatter) redirected to an in-memory buffer for the
admin-label filter tests.
"""

from __future__ import annotations

import io
import json
import logging
import time as time_module

import pytest

from src.auth import audit
from src.auth.audit import admin_label_scope, log_denied, reset_suppression_state
from src.utils.logging import configure_logging

E2E_LOGGER = "tests.unit.test_audit_suppression.e2e"


@pytest.fixture(autouse=True)
def _reset_suppression_state():
    """Without this, one test's rejections bleed into the next test's
    caplog assertions (suppression state is module-global)."""
    reset_suppression_state()
    yield
    reset_suppression_state()


# --------------------------------------------------------------------------
# Suppression window basics.
# --------------------------------------------------------------------------

def test_suppression_window_with_different_tokens(monkeypatch, caplog):
    """The exact scenario that breaks a token-keyed dedup design: 200 calls,
    same reason + same client_ip, but a DIFFERENT token_fp every time. Only
    the first _SUPPRESS_N are logged in full; the rest are counted silently
    until the window closes."""
    monkeypatch.setattr(audit, "current_client_ip", lambda: ("9.9.9.9", "socket"))
    caplog.set_level(logging.WARNING, logger="src.auth.audit")

    for i in range(200):
        log_denied(
            logging.WARNING,
            "denied",
            event="auth_rejected",
            reason="invalid_token",
            token_fp=f"fp{i}",
        )

    full_records = [r for r in caplog.records if r.message == "denied"]
    summary_records = [
        r for r in caplog.records if r.message == "auth rejections suppressed"
    ]
    assert len(full_records) == audit._SUPPRESS_N == 10
    assert len(summary_records) == 0  # summary only fires at window close

    caplog.clear()
    real_now = time_module.monotonic()
    monkeypatch.setattr(
        time_module, "monotonic", lambda: real_now + audit._SUPPRESS_WINDOW_S + 1
    )

    log_denied(
        logging.WARNING,
        "denied",
        event="auth_rejected",
        reason="invalid_token",
        token_fp="fp-final",
    )

    new_full = [r for r in caplog.records if r.message == "denied"]
    new_summary = [r for r in caplog.records if r.message == "auth rejections suppressed"]
    assert len(new_full) == 1
    assert len(new_summary) == 1
    assert new_summary[0].suppressed_count == 190
    assert new_summary[0].reason == "invalid_token"
    # window-close path: the ambient IP at flush time is the same key's IP
    # ("9.9.9.9" the whole way through), so this alone wouldn't catch a
    # misattribution bug — but the field must still be correct here too.
    assert new_summary[0].suppressed_client_ip == "9.9.9.9"


def test_different_reasons_have_independent_budgets(monkeypatch, caplog):
    monkeypatch.setattr(audit, "current_client_ip", lambda: ("1.1.1.1", "socket"))
    caplog.set_level(logging.WARNING, logger="src.auth.audit")

    for _ in range(15):
        log_denied(logging.WARNING, "denied-a", event="auth_rejected", reason="reason_a")
    for _ in range(15):
        log_denied(logging.WARNING, "denied-b", event="auth_rejected", reason="reason_b")

    full_a = [r for r in caplog.records if r.message == "denied-a"]
    full_b = [r for r in caplog.records if r.message == "denied-b"]
    assert len(full_a) == 10
    assert len(full_b) == 10


def test_suppress_false_bypasses_suppression_entirely(monkeypatch, caplog):
    monkeypatch.setattr(audit, "current_client_ip", lambda: ("2.2.2.2", "socket"))
    caplog.set_level(logging.WARNING, logger="src.auth.audit")

    for _ in range(20):
        log_denied(
            logging.WARNING,
            "denied-nosupp",
            event="auth_rejected",
            reason="reason_nosupp",
            suppress=False,
        )

    full = [r for r in caplog.records if r.message == "denied-nosupp"]
    assert len(full) == 20
    assert ("reason_nosupp", "2.2.2.2") not in audit._suppression_state


def test_max_tracked_keys_evicts_oldest(monkeypatch, caplog):
    monkeypatch.setattr(audit, "_MAX_TRACKED_KEYS", 3)
    caplog.set_level(logging.WARNING, logger="src.auth.audit")

    times = iter([100.0, 200.0, 300.0, 400.0])
    monkeypatch.setattr(time_module, "monotonic", lambda: next(times))

    for i in range(4):
        monkeypatch.setattr(
            audit, "current_client_ip", lambda ip=f"3.3.3.{i}": (ip, "socket")
        )
        log_denied(logging.WARNING, "denied", event="auth_rejected", reason=f"reason_{i}")

    assert len(audit._suppression_state) <= 3
    # oldest window_start (100.0 -> reason_0) must have been evicted
    assert ("reason_0", "3.3.3.0") not in audit._suppression_state


def test_eviction_summary_attributes_correct_client_ip(monkeypatch, caplog):
    """Regression for the eviction-path misattribution bug: the evicted key's
    summary must carry ITS OWN client_ip (via `suppressed_client_ip`), not the
    ambient IP of whichever later request happened to trigger the eviction.
    Repro from the review: 190 suppressed rejections from 6.6.6.6 were logged
    with the flushing request's IP (1.2.3.4) instead."""
    monkeypatch.setattr(audit, "_MAX_TRACKED_KEYS", 1)
    caplog.set_level(logging.WARNING, logger="src.auth.audit")

    # Key A: client 6.6.6.6, flooded so 5 occurrences are suppressed beyond
    # the first _SUPPRESS_N=10.
    monkeypatch.setattr(audit, "current_client_ip", lambda: ("6.6.6.6", "socket"))
    for _ in range(15):
        log_denied(logging.WARNING, "denied-a", event="auth_rejected", reason="flood")

    caplog.clear()

    # A completely different client (1.2.3.4) triggers eviction of key A,
    # since _MAX_TRACKED_KEYS is capped at 1 and this is a brand-new key.
    monkeypatch.setattr(audit, "current_client_ip", lambda: ("1.2.3.4", "socket"))
    log_denied(logging.WARNING, "denied-b", event="auth_rejected", reason="other_reason")

    summaries = [r for r in caplog.records if r.message == "auth rejections suppressed"]
    assert len(summaries) == 1
    assert summaries[0].reason == "flood"
    assert summaries[0].suppressed_count == 5
    assert summaries[0].suppressed_client_ip == "6.6.6.6"
    assert summaries[0].suppressed_client_ip != "1.2.3.4"


def test_exc_info_true_attaches_exception_without_raising(monkeypatch, caplog):
    monkeypatch.setattr(audit, "current_client_ip", lambda: ("4.4.4.4", "socket"))
    caplog.set_level(logging.WARNING, logger="src.auth.audit")

    try:
        raise ValueError("boom")
    except ValueError:
        log_denied(
            logging.WARNING,
            "denied-exc",
            event="auth_rejected",
            reason="reason_exc",
            exc_info=True,
        )

    record = next(r for r in caplog.records if r.message == "denied-exc")
    assert record.exc_info is not None


def test_none_client_ip_uses_source_string_as_key(monkeypatch, caplog):
    """Outside any request context, current_client_ip() returns
    (None, "no_context"); log_denied must not crash and must still suppress,
    keying on the source string since ip is None."""
    monkeypatch.setattr(audit, "current_client_ip", lambda: (None, "no_context"))
    caplog.set_level(logging.WARNING, logger="src.auth.audit")

    for _ in range(12):
        log_denied(
            logging.WARNING, "denied-none", event="auth_rejected", reason="reason_none_ip"
        )

    full = [r for r in caplog.records if r.message == "denied-none"]
    assert len(full) == 10
    assert ("reason_none_ip", "no_context") in audit._suppression_state


# --------------------------------------------------------------------------
# Admin-label ContextVar + log filter (mirrors test_client_ip.py's
# json_log_stream pattern).
# --------------------------------------------------------------------------

@pytest.fixture
def json_log_stream():
    """Install the REAL production logging config, redirected to a buffer.

    Restores the root logger's handlers/level afterwards so pytest's own
    caplog/report handlers survive.
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


def test_admin_label_stamped_inside_scope(json_log_stream: io.StringIO) -> None:
    with admin_label_scope("ops-alice"):
        logging.getLogger(E2E_LOGGER).info("inside scope")

    record = _record_with_message(json_log_stream, "inside scope")
    assert record["admin_label"] == "ops-alice"


def test_admin_label_absent_outside_scope(json_log_stream: io.StringIO) -> None:
    logging.getLogger(E2E_LOGGER).info("outside scope")

    record = _record_with_message(json_log_stream, "outside scope")
    assert "admin_label" not in record


def test_explicit_admin_label_wins_over_ambient_scope(json_log_stream: io.StringIO) -> None:
    with admin_label_scope("ops-alice"):
        logging.getLogger(E2E_LOGGER).info(
            "explicit wins", extra={"admin_label": "explicit"}
        )

    record = _record_with_message(json_log_stream, "explicit wins")
    assert record["admin_label"] == "explicit"
