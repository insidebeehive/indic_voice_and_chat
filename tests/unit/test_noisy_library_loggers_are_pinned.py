"""httpx must not log CRM urls, and turning DEBUG on must not un-silence it.

`tool_executor.py` goes to real trouble to keep the player id out of a logged
CRM url -- production lines read `.../players/[redacted]/bets`. One frame away,
httpx logs `HTTP Request: GET <full url>` at INFO for the same call, player id
and all. `configure_logging` pins httpx/httpcore to WARNING, and that single
line is the only thing standing between those ids and stdout/Loki.

It reads as noise reduction, which is how it would get deleted. It is not:
without it the redaction upstream is decorative.

The second case is the one that motivated this file. The whole point of the
DEBUG standard (docs/debug-logging.md) is that an operator flips the level
during an incident. If the pin were left to inherit the root level, that flip
would also turn httpx's INFO line back on -- leaking exactly when the system is
under the most scrutiny and shipping the most data to Loki. `setLevel` on the
named logger survives the root change; this pins that it keeps doing so.
"""

from __future__ import annotations

import logging

import pytest

from src.utils.logging import configure_logging

NOISY = ("uvicorn.access", "httpx", "httpcore")


@pytest.fixture(autouse=True)
def _restore_logging():
    """configure_logging mutates global state; put it back for other tests."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved = {n: logging.getLogger(n).level for n in NOISY}
    yield
    root.setLevel(saved_level)
    root.handlers[:] = saved_handlers
    for name, lvl in saved.items():
        logging.getLogger(name).setLevel(lvl)


@pytest.mark.parametrize("name", NOISY)
def test_pinned_at_warning_by_default(name: str) -> None:
    configure_logging(level="INFO")
    assert logging.getLogger(name).level == logging.WARNING


@pytest.mark.parametrize("name", NOISY)
def test_debug_level_does_not_unsilence_them(name: str) -> None:
    """The case that matters: an operator turning DEBUG on for an incident."""
    configure_logging(level="DEBUG")
    log = logging.getLogger(name)
    assert log.level == logging.WARNING
    assert not log.isEnabledFor(logging.INFO), (
        f"{name} would emit at INFO with root=DEBUG; for httpx that is a full "
        "CRM url including the player id, on every tool call"
    )


def test_httpx_request_line_is_suppressed_at_debug() -> None:
    """Assert on behaviour, not just the configured number.

    A level check passes if someone adds a handler that bypasses it; this
    drives the real logger and asserts nothing came out.
    """
    configure_logging(level="DEBUG")

    records: list[logging.LogRecord] = []

    class _Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    root = logging.getLogger()
    root.addHandler(_Grab())

    logging.getLogger("httpx").info(
        'HTTP Request: GET %s "HTTP/1.1 200 OK"',
        "https://crm.example.com/players/6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef/profile",
    )

    leaked = [r for r in records if "6c1a77a6" in r.getMessage()]
    assert not leaked, "httpx INFO line reached the root handlers with a player id in it"


def test_our_own_loggers_still_follow_the_root_level() -> None:
    """The pin must be surgical -- DEBUG has to still work for the app.

    Pinning too broadly would silence the instrumentation the sweep exists to
    add, which fails in the opposite direction and just as silently.
    """
    configure_logging(level="DEBUG")
    assert logging.getLogger("src.agents.chatbot").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("src.rag.retriever").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("src.providers.llm.gemini").isEnabledFor(logging.DEBUG)
