"""Structured JSON logging setup.

Single entry point: ``configure_logging(level)`` — call once at app startup.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import Optional

import httpx
from pythonjsonlogger import jsonlogger

from src.auth.audit import current_admin_label
from src.utils.client_ip import current_client_ip
from src.utils.redact import redact_url
from src.utils.trace_id import current_trace_id


class _ClientIPLogFilter(logging.Filter):
    """Stamp every record with the ambient client IP from ``ClientIPMiddleware``.

    Attached to the handler rather than to a logger, so it applies to everything
    that reaches stdout regardless of which logger emitted it. A record that
    already carries ``client_ip`` (a caller that passed it explicitly via
    ``extra={...}``) is left untouched -- the explicit value always wins.

    The dependency direction is one-way: this module imports
    ``src.utils.client_ip``; ``client_ip.py`` must never import this module.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "client_ip"):
            ip, source = current_client_ip()
            record.client_ip = ip
            record.client_ip_source = source
        return True


class _AdminLabelLogFilter(logging.Filter):
    """Stamp records emitted during an admin-authenticated request with the
    operator label of the admin token that authenticated it.

    Unlike _ClientIPLogFilter this stamps ONLY when a label is actually set —
    the overwhelming majority of traffic is not admin traffic, and adding
    `admin_label: null` to every record would change the shape of every log
    line in the system for no signal. An explicit `admin_label` passed via
    extra={} still wins.

    Dependency direction is one-way: this module imports src.auth.audit;
    src/auth/audit.py must never import this module.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "admin_label"):
            label = current_admin_label()
            if label is not None:
                record.admin_label = label
        return True


class _TraceIdLogFilter(logging.Filter):
    """Stamp records emitted during a scoped turn (see src/utils/trace_id.py)
    with that turn's ambient trace id.

    Throwaway-by-design stopgap for spotting duplicate/repeated turns by
    grepping logs -- gets deleted once real OpenTelemetry tracing lands.

    Like _AdminLabelLogFilter this stamps ONLY when a trace id is actually
    set -- most log lines are emitted outside any turn scope, and adding
    `trace_id: null` to every record would change the shape of every log
    line in the system for no signal. An explicit `trace_id` passed via
    extra={} still wins.

    Dependency direction is one-way: this module imports src.utils.trace_id;
    src/utils/trace_id.py must never import this module.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            trace_id = current_trace_id()
            if trace_id is not None:
                record.trace_id = trace_id
        return True


class LokiPushHandler(logging.Handler):
    """Forwards formatted JSON log lines to a Grafana Cloud Loki push endpoint.

    Non-blocking: ``emit()`` only enqueues (a bounded, non-blocking
    ``queue.Put``) — the actual HTTP POST happens on a single background
    daemon thread that batches by size and by a time interval. A plain
    ``logging.handlers.QueueHandler``/``QueueListener`` pair doesn't quite fit
    here: ``QueueListener`` only dispatches when a record is dequeued, with no
    periodic/time-based flush independent of new records arriving, and we need
    to flush a partial batch on a timer even when logging is quiet. So this
    handler manages its own worker thread over a ``queue.Queue`` instead.

    A failed push (network error, non-2xx response) is swallowed and warned
    to stderr (not through the logging system itself, to avoid feeding a
    failure back into the thing that's failing) — logging infrastructure must
    never raise into application code or crash the process. The warning is
    rate-limited to one per outage (a success re-arms it), not literally once
    ever -- a `flush()` that pushes several chunks can warn once per chunk
    that fails.
    """

    # Sentinel put on the queue by close() to wake a thread blocked in
    # queue.Queue.get(timeout=...) immediately, instead of waiting out
    # whatever timeout it's currently blocked on.
    _STOP_SENTINEL = None

    def __init__(
        self,
        url: str,
        auth: Optional[str] = None,
        *,
        service_name: str = "app",
        batch_size: int = 100,
        flush_interval: float = 2.0,
        timeout: float = 5.0,
        max_queue_size: int = 10_000,
        shutdown_budget: float = 5.0,
        start_thread: bool = True,
    ) -> None:
        super().__init__()
        self.url = url
        self.service_name = service_name
        self.batch_size = max(1, batch_size)  # range(0, n, 0) would raise
        self.flush_interval = flush_interval
        self.timeout = timeout
        # Shutdown drain is bounded to this many seconds of *new* chunk
        # attempts regardless of backlog size -- see close()/_run().
        self.shutdown_budget = shutdown_budget
        self._queue: "queue.Queue[Optional[tuple[str, str, str]]]" = queue.Queue(maxsize=max_queue_size)

        auth_tuple = None
        if auth:
            if ":" in auth:
                user, _, key = auth.partition(":")
                auth_tuple = (user, key)
            else:
                self._warn_stderr(
                    "GRAFANA_LOKI_PUSH_AUTH set without a ':' separator; "
                    "sending unauthenticated (expected 'user:api_key')"
                )
        self._client = httpx.Client(timeout=timeout, auth=auth_tuple)

        self._stop_event = threading.Event()
        self._warned = False
        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._run, name="loki-push-handler", daemon=True
            )
            self._thread.start()

    # -- logging.Handler interface -----------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            ts_ns = str(int(record.created * 1_000_000_000))
            level = record.levelname.lower()
        except Exception:
            self.handleError(record)
            return
        try:
            self._queue.put_nowait((ts_ns, line, level))
        except queue.Full:
            pass  # drop rather than block the caller

    def flush(self) -> None:
        """Synchronously drain whatever is currently queued and push it.

        Chunked by ``batch_size`` like the background worker does, so a large
        backlog (e.g. drained at shutdown) doesn't become one oversized POST
        that a real Loki endpoint would reject outright. Never raises -- see
        ``_flush``.

        Bounded by ``shutdown_budget``, the same cap ``close()``'s underlying
        drain (in ``_run()``) already uses -- ``logging.shutdown()``'s atexit
        hook calls ``flush()`` THEN ``close()`` on every handler, so an
        unbounded ``flush()`` here would defeat that cap: a large backlog
        (e.g. 10k queued records) behind a stuck/slow endpoint could
        otherwise make process shutdown take a duration proportional to
        backlog size instead of being capped, same as the unbounded-drain
        issue ``close()`` already fixes.
        """
        self._flush_chunked(
            self._drain_queue(), deadline=time.monotonic() + self.shutdown_budget
        )

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            try:
                self._queue.put_nowait(self._STOP_SENTINEL)
            except queue.Full:
                pass  # thread will still notice _stop_event within one poll
            # Bounded regardless of backlog size: the shutdown drain in
            # _run() caps itself to shutdown_budget's worth of *new* chunk
            # attempts, so this only needs to additionally cover one chunk
            # that was already in flight when that budget ran out, plus a
            # small buffer.
            self._thread.join(timeout=self.shutdown_budget + self.timeout + 2.0)
        try:
            self._client.close()
        except Exception:
            pass
        super().close()

    # -- internals ------------------------------------------------------

    @staticmethod
    def _build_payload(batch: list[tuple[str, str, str]], service_name: str) -> dict:
        """Group a batch of (ts_ns, line, level) tuples into Loki's push shape:
        ``{"streams": [{"stream": {...labels}, "values": [[ts, line], ...]}]}``.
        """
        streams: dict[str, dict] = {}
        for ts_ns, line, level in batch:
            entry = streams.setdefault(
                level,
                {"stream": {"service": service_name, "level": level}, "values": []},
            )
            entry["values"].append([ts_ns, line])
        return {"streams": list(streams.values())}

    def _push(self, payload: dict) -> None:
        try:
            resp = self._client.post(self.url, json=payload)
            if resp.status_code >= 300:
                self._warn_once(f"Loki push returned HTTP {resp.status_code}")
            else:
                # Recovered (or first push ever succeeded) -- allow the next
                # failure to warn again, so a real outage isn't silenced
                # forever by one earlier, already-resolved glitch.
                self._warned = False
        except Exception as exc:  # network error, timeout, etc.
            # Don't interpolate the raw exception: httpx exception reprs can
            # carry the full request URL, and GRAFANA_LOKI_PUSH_URL could in
            # principle embed basic-auth userinfo -- redact before logging.
            self._warn_once(f"Loki push failed: {type(exc).__name__} ({self._redacted_url()})")

    def _flush(self, batch: list[tuple[str, str, str]]) -> None:
        """Build + push one batch. Never raises -- a bug in payload building
        (as much as a network failure) must not kill the worker thread or
        propagate out of flush()/close()."""
        if not batch:
            return
        try:
            payload = self._build_payload(batch, self.service_name)
        except Exception as exc:
            self._warn_once(f"Loki payload build failed: {type(exc).__name__}")
            return
        self._push(payload)

    def _flush_chunked(
        self, batch: list[tuple[str, str, str]], deadline: Optional[float] = None
    ) -> None:
        """Push ``batch`` in ``batch_size`` pieces. If ``deadline`` (a
        ``time.monotonic()`` value) is given and passes before a chunk would
        start, remaining chunks are dropped with one warning instead of being
        attempted -- used only for the shutdown drain, so an enormous backlog
        (e.g. Loki down for a long stretch) can't make process shutdown take
        a duration proportional to how much piled up."""
        dropped = 0
        for i in range(0, len(batch), self.batch_size):
            if deadline is not None and time.monotonic() > deadline:
                dropped = len(batch) - i
                break
            self._flush(batch[i:i + self.batch_size])
        if dropped:
            self._warn_stderr(
                f"Loki shutdown drain exceeded its {self.shutdown_budget}s budget; "
                f"dropped {dropped} queued log line(s)"
            )

    def _drain_queue(self) -> list[tuple[str, str, str]]:
        batch: list[tuple[str, str, str]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not self._STOP_SENTINEL:
                batch.append(item)
        return batch

    def _run(self) -> None:
        batch: list[tuple[str, str, str]] = []
        last_flush = time.monotonic()
        while not self._stop_event.is_set():
            try:
                remaining = self.flush_interval - (time.monotonic() - last_flush)
                try:
                    item = self._queue.get(timeout=remaining if remaining > 0 else 0.05)
                    if item is not self._STOP_SENTINEL:
                        batch.append(item)
                except queue.Empty:
                    pass
                now = time.monotonic()
                interval_elapsed = now - last_flush >= self.flush_interval
                if batch and (len(batch) >= self.batch_size or interval_elapsed):
                    self._flush_chunked(batch)
                    batch = []
                if interval_elapsed:
                    # Advance the window even when idle (empty batch) -- otherwise
                    # last_flush never moves while the queue is quiet, `remaining`
                    # stays <= 0 forever, and get() falls to its 0.05s fallback
                    # timeout in a tight poll loop indefinitely.
                    last_flush = now
            except Exception as exc:
                # Defense in depth: _flush already swallows its own errors, so
                # reaching here means something unexpected happened in the loop
                # scaffolding itself. The worker must never die silently and
                # stop shipping logs -- warn, drop the in-progress batch, and
                # keep the loop alive rather than letting the thread exit.
                self._warn_once(f"Loki worker loop error: {type(exc).__name__}")
                batch = []
                last_flush = time.monotonic()
        # Final drain on shutdown so nothing queued right before close() is
        # lost, bounded by shutdown_budget so an enormous backlog can't make
        # process shutdown hang for a duration proportional to its size.
        batch.extend(self._drain_queue())
        self._flush_chunked(batch, deadline=time.monotonic() + self.shutdown_budget)

    def _redacted_url(self) -> str:
        """Credential-safe rendering of ``self.url`` for a warning message.

        Delegates to the shared ``src.utils.redact.redact_url`` helper (also
        used by ``src.observability.turn_metrics_push``) -- kept as a thin
        instance method here so existing call sites/tests referencing
        ``handler._redacted_url()`` are unaffected."""
        return redact_url(self.url)

    def _warn_once(self, msg: str) -> None:
        if self._warned:
            return
        self._warned = True
        self._warn_stderr(f"{msg} (further push failures suppressed until the next success)")

    @staticmethod
    def _warn_stderr(msg: str) -> None:
        try:
            sys.stderr.write(f"LokiPushHandler: {msg}\n")
        except Exception:
            pass


_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def configure_logging(
    level: str = "INFO",
    *,
    loki_url: Optional[str] = None,
    loki_auth: Optional[str] = None,
    service_name: str = "app",
) -> None:
    """Configure root logger with JSON output to stdout.

    Idempotent — safe to call multiple times (e.g. in tests).

    When ``loki_url`` is set, also attaches a ``LokiPushHandler`` that
    forwards the same formatted JSON lines to Grafana Cloud Loki in the
    background. Left unset (the default — matches an unconfigured
    ``GRAFANA_LOKI_PUSH_URL``), this is a clean no-op: stdout logging is
    unaffected and no network activity happens.
    """
    root = logging.getLogger()
    normalized = (level or "INFO").strip().upper()
    if normalized not in _VALID_LOG_LEVELS:
        normalized = "INFO"
    root.setLevel(normalized)

    # Remove any existing handlers so we don't duplicate output. A
    # LokiPushHandler from a prior configure_logging() call is close()'d
    # first to stop its background thread rather than leaking it on every
    # re-configure; other handlers (e.g. a test harness's LogCaptureHandler)
    # are just detached, unchanged from before -- we didn't create them and
    # closing them isn't ours to do.
    for h in list(root.handlers):
        root.removeHandler(h)
        if isinstance(h, LokiPushHandler):
            try:
                h.close()
            except Exception:
                pass

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(_ClientIPLogFilter())
    handler.addFilter(_AdminLabelLogFilter())
    handler.addFilter(_TraceIdLogFilter())
    root.addHandler(handler)

    if loki_url:
        loki_handler = LokiPushHandler(loki_url, loki_auth, service_name=service_name)
        # Same formatter instance as stdout -> byte-identical JSON on both sinks.
        loki_handler.setFormatter(formatter)
        loki_handler.addFilter(_ClientIPLogFilter())
        loki_handler.addFilter(_AdminLabelLogFilter())
        loki_handler.addFilter(_TraceIdLogFilter())
        root.addHandler(loki_handler)

    # Quiet down noisy libraries.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
