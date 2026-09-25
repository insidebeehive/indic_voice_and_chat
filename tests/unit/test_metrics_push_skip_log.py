from __future__ import annotations

import logging

import pytest

from src.observability import chat_metrics_push, turn_metrics_push


@pytest.mark.asyncio
@pytest.mark.parametrize("mod, fn", [
    (turn_metrics_push, "aggregate_and_push_turn_metrics"),
    (chat_metrics_push, "aggregate_and_push_chat_metrics"),
])
async def test_push_skip_is_logged_once_per_process(mod, fn, monkeypatch, caplog) -> None:
    """With no push_url the loop ticks every interval; the skip must be
    logged on the first tick only, not once a minute forever."""
    monkeypatch.setattr(mod, "_push_skip_logged", False)
    caplog.set_level(logging.DEBUG, logger=mod.__name__)
    for _ in range(3):
        assert await getattr(mod, fn)(None, None, None) == 0
    skips = [r for r in caplog.records if r.getMessage() == "metrics push skipped"]
    assert len(skips) == 1
