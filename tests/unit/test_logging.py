from __future__ import annotations

import logging

from src.utils.logging import configure_logging


def test_configure_logging_invalid_level_falls_back_to_info() -> None:
    configure_logging("garbage")
    assert logging.getLogger().level == logging.INFO
