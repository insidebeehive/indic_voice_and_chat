"""Structured JSON logging setup.

Single entry point: ``configure_logging(level)`` — call once at app startup.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

from src.utils.client_ip import current_client_ip


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


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON output to stdout.

    Idempotent — safe to call multiple times (e.g. in tests).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any existing handlers so we don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    handler.addFilter(_ClientIPLogFilter())
    root.addHandler(handler)

    # Quiet down noisy libraries.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
