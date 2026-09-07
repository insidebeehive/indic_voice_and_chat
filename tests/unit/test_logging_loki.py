"""Grafana Cloud Loki log push (Phase 1 observability).

Mirrors ``tests/unit/test_logging.py``'s style (plain functions, no fixtures)
and mocks the outbound HTTP with ``respx``, following the convention used in
``tests/unit/test_deposit_ticket_outbound.py``.
"""

from __future__ import annotations

import json
import logging
import time

import httpx
import respx
from pythonjsonlogger import jsonlogger

from src.utils.logging import LokiPushHandler, configure_logging

LOKI_URL = "https://logs-prod-example.grafana.net/loki/api/v1/push"


def _reset_root() -> None:
    # Mirrors configure_logging()'s own removal loop: only close() handlers
    # we might have created (LokiPushHandler, to stop its thread) and leave
    # anything else -- e.g. pytest's own LogCaptureHandler -- alone.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        if isinstance(h, LokiPushHandler):
            try:
                h.close()
            except Exception:
                pass


def _make_formatter() -> jsonlogger.JsonFormatter:
    # Same construction as src.utils.logging.configure_logging's stdout formatter.
    return jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )


def _make_record(msg: str = "hello world", level: int = logging.INFO, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def _assert_is_ns_timestamp(ts: str) -> None:
    """Loki wants a unix-nanosecond timestamp as a string. Assert it actually
    looks like nanoseconds (19 digits, close to time.time_ns()) rather than
    just "isdigit()", which a second/millisecond timestamp would pass too."""
    assert ts.isdigit()
    assert len(ts) == 19
    assert abs(int(ts) - time.time_ns()) < 5_000_000_000  # within 5s


def test_no_loki_handler_when_url_unset() -> None:
    _reset_root()
    try:
        configure_logging("INFO")  # loki_url defaults to None
        handlers = logging.getLogger().handlers
        assert not any(isinstance(h, LokiPushHandler) for h in handlers)
    finally:
        _reset_root()


def test_loki_handler_attached_and_shares_formatter_when_url_set() -> None:
    _reset_root()
    try:
        configure_logging("INFO", loki_url=LOKI_URL, loki_auth="user:key", service_name="test-svc")
        root = logging.getLogger()
        loki_handlers = [h for h in root.handlers if isinstance(h, LokiPushHandler)]
        other_handlers = [h for h in root.handlers if not isinstance(h, LokiPushHandler)]
        assert len(loki_handlers) == 1
        assert loki_handlers[0].service_name == "test-svc"
        assert loki_handlers[0].url == LOKI_URL
        # Same formatter instance as the stdout handler -> byte-identical JSON
        # on both sinks, per the phase-1 requirement.
        assert len(other_handlers) == 1
        assert other_handlers[0].formatter is loki_handlers[0].formatter
    finally:
        # configure_logging()'s own handler-removal loop close()s the
        # LokiPushHandler; that's the thing under test in
        # test_close_returns_promptly_even_with_a_live_worker_thread below.
        _reset_root()


def test_emit_formats_line_identically_to_json_formatter() -> None:
    formatter = _make_formatter()
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", start_thread=False)
    handler.setFormatter(formatter)
    try:
        record = _make_record(foo="bar")
        expected_line = formatter.format(record)

        handler.emit(record)
        batch = handler._drain_queue()

        assert len(batch) == 1
        ts_ns, line, level = batch[0]
        assert line == expected_line
        assert level == "info"
        _assert_is_ns_timestamp(ts_ns)
    finally:
        handler.close()


@respx.mock
def test_flush_builds_loki_payload_shape() -> None:
    route = respx.post(LOKI_URL).mock(return_value=httpx.Response(204))
    formatter = _make_formatter()
    handler = LokiPushHandler(LOKI_URL, "user:key", service_name="test-svc", start_thread=False)
    handler.setFormatter(formatter)
    try:
        handler.emit(_make_record("first info line"))
        handler.emit(_make_record("first warning line", level=logging.WARNING))
        handler.flush()

        assert route.called
        request = route.calls.last.request
        assert request.headers["authorization"].startswith("Basic ")

        payload = json.loads(request.content)
        assert set(payload.keys()) == {"streams"}
        streams_by_level = {s["stream"]["level"]: s for s in payload["streams"]}
        assert set(streams_by_level) == {"info", "warning"}

        info_stream = streams_by_level["info"]
        assert info_stream["stream"] == {"service": "test-svc", "level": "info"}
        assert len(info_stream["values"]) == 1
        ts, line = info_stream["values"][0]
        _assert_is_ns_timestamp(ts)
        parsed = json.loads(line)
        assert parsed["message"] == "first info line"
        assert parsed["level"] == "INFO"
    finally:
        handler.close()


@respx.mock
def test_flush_chunks_large_backlog_instead_of_one_oversized_post() -> None:
    """A manual/shutdown drain must not build a single unbounded payload from
    the whole queue -- chunk it by batch_size like the background worker does,
    or a large backlog becomes one POST a real Loki endpoint would reject."""
    route = respx.post(LOKI_URL).mock(return_value=httpx.Response(204))
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", batch_size=2, start_thread=False)
    handler.setFormatter(_make_formatter())
    try:
        for i in range(5):
            handler.emit(_make_record(f"line {i}"))
        handler.flush()

        assert route.call_count == 3  # ceil(5 / 2)
        total_values = 0
        for call in route.calls:
            payload = json.loads(call.request.content)
            for stream in payload["streams"]:
                assert len(stream["values"]) <= 2
                total_values += len(stream["values"])
        assert total_values == 5
    finally:
        handler.close()


@respx.mock
def test_failed_push_http_error_does_not_raise() -> None:
    respx.post(LOKI_URL).mock(return_value=httpx.Response(500))
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", start_thread=False)
    handler.setFormatter(_make_formatter())
    try:
        handler.emit(_make_record("will fail to push"))
        handler.flush()  # must not raise despite the 500
        assert handler._warned is True  # the failure was actually noticed, not just ignored
    finally:
        handler.close()


@respx.mock
def test_failed_push_network_error_does_not_raise_and_later_logging_still_works() -> None:
    route = respx.post(LOKI_URL).mock(side_effect=httpx.ConnectError("boom"))
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", start_thread=False)
    handler.setFormatter(_make_formatter())
    try:
        handler.emit(_make_record("first, will error"))
        handler.flush()  # swallowed, not raised
        assert handler._warned is True

        # Logging infrastructure failures must be invisible to the app: a
        # subsequent log call must still work fine (and not raise) afterward.
        route.mock(return_value=httpx.Response(204))
        handler.emit(_make_record("second, should succeed"))
        handler.flush()
        assert route.call_count == 2  # the earlier failed attempt + this recovery
        # A successful push resets the "warned once" latch, so a later, new
        # outage isn't silenced forever by this earlier, already-resolved one.
        assert handler._warned is False
    finally:
        handler.close()


@respx.mock
def test_warn_once_reactivates_after_recovery() -> None:
    route = respx.post(LOKI_URL)
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", start_thread=False)
    handler.setFormatter(_make_formatter())
    try:
        route.mock(return_value=httpx.Response(500))
        handler.emit(_make_record("fails"))
        handler.flush()
        assert handler._warned is True

        route.mock(return_value=httpx.Response(204))
        handler.emit(_make_record("succeeds"))
        handler.flush()
        assert handler._warned is False

        route.mock(return_value=httpx.Response(500))
        handler.emit(_make_record("fails again"))
        handler.flush()
        assert handler._warned is True  # would stay False forever without the reset
    finally:
        handler.close()


def test_flush_swallows_unexpected_payload_build_error() -> None:
    """A bug in _build_payload (not just a network failure) must not raise out
    of flush() or kill the worker thread -- this is the class's own explicit
    contract (see its docstring)."""
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", start_thread=False)
    handler.setFormatter(_make_formatter())

    def _boom(*_a, **_kw):
        raise ValueError("boom")

    handler._build_payload = _boom  # instance override shadows the staticmethod
    try:
        handler.emit(_make_record("x"))
        handler.flush()  # must not raise
        assert handler._warned is True
    finally:
        handler.close()


def test_colon_less_auth_warns_and_sends_unauthenticated(capsys) -> None:
    handler = LokiPushHandler(LOKI_URL, "not-a-user-colon-key-pair", service_name="svc", start_thread=False)
    try:
        err = capsys.readouterr().err
        assert "GRAFANA_LOKI_PUSH_AUTH" in err
        assert handler._client.auth is None
    finally:
        handler.close()


def test_close_returns_promptly_even_with_a_live_worker_thread() -> None:
    """close() must wake a thread parked in queue.get(timeout=flush_interval)
    immediately via the stop sentinel, not wait out the whole interval."""
    handler = LokiPushHandler(
        LOKI_URL, None, service_name="svc", flush_interval=30.0, start_thread=True
    )
    started = time.monotonic()
    handler.close()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"close() took {elapsed:.2f}s, expected well under the 30s flush_interval"
    assert not handler._thread.is_alive()


def test_idle_worker_does_not_busy_spin() -> None:
    """With nothing ever logged, the worker must block for ~flush_interval at
    a time, not poll in a tight loop (previously fell back to a 0.05s timeout
    forever once idle, because last_flush never advanced without a batch)."""
    poll_calls = []
    handler = LokiPushHandler(
        LOKI_URL, None, service_name="svc", flush_interval=0.3, start_thread=False
    )
    try:
        real_get = handler._queue.get

        def _counting_get(*args, **kwargs):
            poll_calls.append(time.monotonic())
            return real_get(*args, **kwargs)

        handler._queue.get = _counting_get
        # start_thread=False above means handler._thread is still None; run
        # the loop on our own bounded thread instead of the handler's.
        import threading as _threading
        t = _threading.Thread(target=handler._run, daemon=True)
        t.start()
        time.sleep(1.0)
        handler._stop_event.set()
        handler._queue.put_nowait(handler._STOP_SENTINEL)
        t.join(timeout=2.0)
        assert not t.is_alive()
    finally:
        handler.close()

    # ~1s idle at a 0.3s flush_interval should poll a handful of times
    # (~3-4), not the ~20/s a 0.05s busy-spin fallback would produce.
    assert len(poll_calls) <= 8, f"expected infrequent polling while idle, got {len(poll_calls)} calls"


@respx.mock
def test_close_caps_shutdown_drain_time_for_a_huge_backlog() -> None:
    """close() must not take a duration proportional to backlog size: a large
    queue behind a slow/stuck endpoint is bounded by shutdown_budget, with
    the remainder dropped (and warned about) rather than blocking shutdown."""
    def _slow_response(request):
        time.sleep(0.05)
        return httpx.Response(204)

    respx.post(LOKI_URL).mock(side_effect=_slow_response)
    handler = LokiPushHandler(
        LOKI_URL, None, service_name="svc",
        batch_size=1, shutdown_budget=0.2, start_thread=True,
    )
    handler.setFormatter(_make_formatter())
    try:
        for i in range(50):
            handler.emit(_make_record(f"line {i}"))

        started = time.monotonic()
        handler.close()
        elapsed = time.monotonic() - started

        # Bounded by shutdown_budget + one in-flight push + overhead, NOT by
        # 50 * 0.05s (2.5s) which is what an unbounded chunked drain would cost.
        assert elapsed < 2.0, f"close() took {elapsed:.2f}s, expected well under 2s"
        assert not handler._thread.is_alive()
    finally:
        pass  # already closed


@respx.mock
def test_flush_caps_drain_time_for_a_huge_backlog() -> None:
    """flush() must not take a duration proportional to backlog size, same as
    close(): logging.shutdown()'s atexit hook calls flush() THEN close() on
    every handler, so an unbounded flush() would defeat close()'s own bound.
    A large backlog behind a slow/stuck endpoint is bounded by
    shutdown_budget, with the remainder dropped (and warned about) rather
    than blocking (e.g.) process shutdown."""
    def _slow_response(request):
        time.sleep(0.05)
        return httpx.Response(204)

    respx.post(LOKI_URL).mock(side_effect=_slow_response)
    handler = LokiPushHandler(
        LOKI_URL, None, service_name="svc",
        batch_size=1, shutdown_budget=0.2, start_thread=False,
    )
    handler.setFormatter(_make_formatter())
    try:
        for i in range(50):
            handler.emit(_make_record(f"line {i}"))

        started = time.monotonic()
        handler.flush()
        elapsed = time.monotonic() - started

        # Bounded by shutdown_budget + one in-flight push + overhead, NOT by
        # 50 * 0.05s (2.5s) which is what an unbounded chunked flush would cost.
        assert elapsed < 2.0, f"flush() took {elapsed:.2f}s, expected well under 2s"
    finally:
        handler.close()


def test_redacted_url_hides_userinfo_query_and_non_http_schemes() -> None:
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", start_thread=False)
    try:
        handler.url = "https://user:s3cr3t@logs.example.net/loki/api/v1/push?apikey=s3cr3t"
        redacted = handler._redacted_url()
        assert "s3cr3t" not in redacted
        assert redacted == "https://logs.example.net/loki/api/v1/push"

        # No "https://" prefix -> "user" parses as the scheme, not netloc
        # userinfo; a netloc-only redaction would miss this entirely.
        handler.url = "user:s3cr3t@logs.example.net/loki/api/v1/push"
        assert handler._redacted_url() == "<redacted>"
        assert "s3cr3t" not in handler._redacted_url()
    finally:
        handler.close()


@respx.mock
def test_batch_size_zero_does_not_crash_flush() -> None:
    """batch_size is clamped to >= 1 in __init__ -- range(0, n, 0) would
    otherwise raise ValueError out of _flush_chunked on the first flush."""
    respx.post(LOKI_URL).mock(return_value=httpx.Response(204))
    handler = LokiPushHandler(LOKI_URL, None, service_name="svc", batch_size=0, start_thread=False)
    handler.setFormatter(_make_formatter())
    try:
        assert handler.batch_size == 1
        handler.emit(_make_record("x"))
        handler.flush()  # must not raise
    finally:
        handler.close()
