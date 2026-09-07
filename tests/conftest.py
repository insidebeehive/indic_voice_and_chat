"""Shared pytest fixtures.

The unit-test suite runs without any live infrastructure:
- Redis: fakeredis.aioredis.FakeRedis
- Postgres: aiosqlite in-memory (schema applied via Base.metadata.create_all)
- HTTP providers: respx mounts on httpx
- Twilio SDK: pytest-mock patches twilio.rest.Client
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis import aioredis as fakeredis_aio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

# Default env vars so provider constructors don't fail under tests.
os.environ.setdefault("SARVAM_API_KEY", "test-sarvam-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "ACtestsid")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "test-twilio-token")
os.environ.setdefault("VOX_CONFIG_PATH", "config/default.yaml")

from src.config import load_settings, reset_settings_cache  # noqa: E402
from src.models.database import Base  # noqa: E402


@pytest.fixture
def settings():
    reset_settings_cache()
    return load_settings()


@pytest_asyncio.fixture
async def fake_redis():
    client = fakeredis_aio.FakeRedis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def test_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    sm = async_sessionmaker(test_engine, expire_on_commit=False)
    async with sm() as s:
        yield s


@pytest.fixture
def tmp_faiss_index(tmp_path: Path) -> str:
    return str(tmp_path / "test_index")


@pytest.fixture(autouse=True)
def _reset_chat_turn_guards():
    """``_turn_guards`` (src/api/chat.py's duplicate-turn guard) is a
    process-global dict keyed only by ``session_id`` -- and a great many chat
    WS tests across this suite reuse the literal session_id ``"sess1"`` (and
    identical message text like ``"hi"``) from one test to the next. Without
    resetting it here, the guard's post-completion echo window
    (``_DUPLICATE_ECHO_WINDOW_S``, 3s by default) would treat an identical
    ``"sess1"``/``"hi"`` message in a LATER, unrelated test as a false-positive
    duplicate of an EARLIER test's already-completed turn -- real wall-clock
    time between two tests in the same run is often under 3s. The
    message would be silently dropped, and the test's websocket would then
    hang forever waiting for a reply/typing frame that was never sent."""
    from src.api import chat as chat_api
    chat_api._turn_guards.clear()
    yield
    chat_api._turn_guards.clear()
