"""Client-IP resolution, the ASGI middleware that publishes it, and the log
filter that stamps it onto every record.

The trust model under test (see ``src/utils/client_ip.py``): with N trusted
proxy hops in front of the app, the Nth ``X-Forwarded-For`` entry from the
right is the one the outermost trusted hop wrote; everything to its left is
client-forgeable. A chain shorter than N resolves to
``(None, "xff_insufficient_hops")`` and explicitly does NOT fall back -- the
tests below pin that non-fallback, because "guess something" is the tempting
regression here.

Also covers the full production pipeline: middleware -> ContextVar -> log
filter -> JsonFormatter, for both HTTP and WebSocket scopes.
"""

from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.datastructures import Address, Headers

from src.utils.client_ip import (
    ClientIPMiddleware,
    _trusted_proxy_hops,
    client_ip,
    current_client_ip,
)
from src.utils.logging import configure_logging

E2E_LOGGER = "tests.unit.test_client_ip.e2e"


@pytest.fixture(autouse=True)
def _clean_hops_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test states its own hop count; never inherit the dev's shell."""
    monkeypatch.delenv("VOX_TRUSTED_PROXY_HOPS", raising=False)


class _FakeConnection:
    """Duck-typed stand-in for Starlette's Request/WebSocket: ``client_ip()``
    only ever touches ``.headers`` and ``.client``."""

    __slots__ = ("headers", "client")

    def __init__(self, headers: dict[str, str] | None = None, client=None) -> None:
        self.headers = Headers(headers or {})
        self.client = Address(*client) if client is not None else None


# --------------------------------------------------------------------------
# The approved worked-example table, verbatim.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("hops", "xff", "expected"),
    [
        # client-forged leading entry; the trusted hop appended 203.0.113.7
        (1, "9.9.9.9, 203.0.113.7", ("203.0.113.7", "x_forwarded_for")),
        # single entry, parts[-1]
        (1, "203.0.113.7", ("203.0.113.7", "x_forwarded_for")),
        # parts[-2]
        (2, "9.9.9.9, 203.0.113.7, 10.0.0.5", ("203.0.113.7", "x_forwarded_for")),
        # chain shorter than hops -- do NOT guess and do NOT fall back
        (2, "203.0.113.7", (None, "xff_insufficient_hops")),
    ],
)
def test_worked_examples(
    monkeypatch: pytest.MonkeyPatch, hops: int, xff: str, expected: tuple[str | None, str]
) -> None:
    monkeypatch.setenv("VOX_TRUSTED_PROXY_HOPS", str(hops))
    assert client_ip(_FakeConnection({"x-forwarded-for": xff})) == expected


def test_insufficient_hops_does_not_fall_back_to_real_ip_or_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the non-fallback: a short chain means the request did
    not traverse our infra as configured, so every other hint is suspect too."""
    monkeypatch.setenv("VOX_TRUSTED_PROXY_HOPS", "3")
    conn = _FakeConnection(
        {"x-forwarded-for": "203.0.113.7, 10.0.0.5", "x-real-ip": "198.51.100.9"},
        client=("192.0.2.11", 51234),
    )
    assert client_ip(conn) == (None, "xff_insufficient_hops")


# --------------------------------------------------------------------------
# Header parsing tolerance.
# --------------------------------------------------------------------------

def test_xff_tolerates_whitespace_and_empty_elements() -> None:
    """Strip each part, drop empties, THEN apply hops-from-right (default 1)."""
    conn = _FakeConnection({"x-forwarded-for": "1.2.3.4,  , 5.6.7.8"})
    assert client_ip(conn) == ("5.6.7.8", "x_forwarded_for")


def test_xff_whitespace_only_is_insufficient_not_a_fallback() -> None:
    """A present-but-empty chain still counts as a chain claim we cannot
    verify: 0 usable parts < 1 hop, so it must not silently become a socket
    or X-Real-IP read."""
    conn = _FakeConnection(
        {"x-forwarded-for": "  ,  ", "x-real-ip": "198.51.100.9"},
        client=("192.0.2.11", 51234),
    )
    assert client_ip(conn) == (None, "xff_insufficient_hops")


# --------------------------------------------------------------------------
# Fallback chain.
# --------------------------------------------------------------------------

def test_x_real_ip_used_when_xff_absent() -> None:
    conn = _FakeConnection({"x-real-ip": "198.51.100.9"}, client=("192.0.2.11", 51234))
    assert client_ip(conn) == ("198.51.100.9", "x_real_ip")


def test_socket_used_when_both_forwarding_headers_absent() -> None:
    conn = _FakeConnection({}, client=("192.0.2.11", 51234))
    assert client_ip(conn) == ("192.0.2.11", "socket")


def test_no_headers_and_no_socket_is_unknown() -> None:
    assert client_ip(_FakeConnection({}, client=None)) == (None, "unknown")


# --------------------------------------------------------------------------
# Hop-count config.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["0", "-4", "not-a-number", "", "  "])
def test_trusted_proxy_hops_clamps_to_one(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("VOX_TRUSTED_PROXY_HOPS", raw)
    assert _trusted_proxy_hops() == 1


def test_trusted_proxy_hops_defaults_to_one_when_unset() -> None:
    assert _trusted_proxy_hops() == 1


def test_current_client_ip_outside_any_request() -> None:
    assert current_client_ip() == (None, "no_context")


# --------------------------------------------------------------------------
# Real middleware, real TestClient, HTTP + WebSocket.
# --------------------------------------------------------------------------

def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ClientIPMiddleware)
    log = logging.getLogger(E2E_LOGGER)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        ip, source = current_client_ip()
        log.info("handled ping")
        return {"ip": ip or "", "source": source}

    @app.websocket("/ws")
    async def ws_route(websocket: WebSocket) -> None:
        await websocket.accept()
        ip, source = current_client_ip()
        await websocket.send_json({"ip": ip, "source": source})
        await websocket.close()

    return app


def test_websocket_through_middleware_sees_forwarded_ip() -> None:
    """BaseHTTPMiddleware would never run here -- this is why the middleware is
    pure ASGI. The WS routes are the app's biggest credential-guessing surface."""
    with TestClient(_build_app()) as client:
        with client.websocket_connect(
            "/ws", headers={"x-forwarded-for": "9.9.9.9, 203.0.113.7"}
        ) as websocket:
            assert websocket.receive_json() == {
                "ip": "203.0.113.7",
                "source": "x_forwarded_for",
            }


def test_websocket_without_forwarding_headers_falls_back_to_socket() -> None:
    """Exercises the middleware's scope["client"] -> Address adapter."""
    with TestClient(_build_app()) as client:
        with client.websocket_connect("/ws") as websocket:
            payload = websocket.receive_json()
    assert payload == {"ip": "testclient", "source": "socket"}


def test_http_request_through_middleware_sees_forwarded_ip() -> None:
    with TestClient(_build_app()) as client:
        response = client.get("/ping", headers={"x-forwarded-for": "9.9.9.9, 203.0.113.7"})
    assert response.json() == {"ip": "203.0.113.7", "source": "x_forwarded_for"}


# --------------------------------------------------------------------------
# Logging pipeline: middleware -> ContextVar -> filter -> JsonFormatter.
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


def test_log_line_during_request_carries_client_ip(json_log_stream: io.StringIO) -> None:
    """Full pipeline end to end: synthetic X-Forwarded-For on a TestClient
    request must surface as client_ip in the structured JSON output of a log
    call that never mentions client_ip itself."""
    with TestClient(_build_app()) as client:
        response = client.get("/ping", headers={"x-forwarded-for": "9.9.9.9, 203.0.113.7"})
    assert response.status_code == 200

    record = _record_with_message(json_log_stream, "handled ping")
    assert record["client_ip"] == "203.0.113.7"
    assert record["client_ip_source"] == "x_forwarded_for"


def test_log_outside_request_context_does_not_crash(json_log_stream: io.StringIO) -> None:
    logging.getLogger(E2E_LOGGER).info("ambient log")

    record = _record_with_message(json_log_stream, "ambient log")
    assert record["client_ip"] is None
    assert record["client_ip_source"] == "no_context"


def test_filter_does_not_overwrite_explicit_client_ip(json_log_stream: io.StringIO) -> None:
    """A caller that passes client_ip via extra={} wins over the ambient value,
    including while a request context is active."""
    with TestClient(_build_app()) as client:
        client.get("/ping", headers={"x-forwarded-for": "203.0.113.7"})

    logging.getLogger(E2E_LOGGER).info(
        "explicit ip", extra={"client_ip": "explicit-value"}
    )

    record = _record_with_message(json_log_stream, "explicit ip")
    assert record["client_ip"] == "explicit-value"
    assert "client_ip_source" not in record
