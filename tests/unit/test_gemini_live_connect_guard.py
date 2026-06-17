"""GeminiLiveSession.connect fails clearly when no API key resolves (Phase 5)."""

from __future__ import annotations

import pytest

from src.interfaces.realtime import RealtimeConfig
from src.providers.realtime.gemini_live import GeminiLiveSession


async def test_connect_raises_clearly_without_any_key(monkeypatch):
    # No tenant key and no platform env → a clear error BEFORE any network attempt.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Gemini Live needs an API key"):
        await GeminiLiveSession.connect(RealtimeConfig(model="m"), api_key=None)
