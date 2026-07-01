"""In-process store for pending human-transfer futures.

When the AI voicebot fires the 'transfer' action, the telephony bridge
registers a Future here keyed by the provider CallSid. The coordination
server resolves it by POSTing to /api/v1/calls/{call_sid}/transfer-result.
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

_pending: dict[str, "asyncio.Future[str]"] = {}


def register(call_sid: str) -> "asyncio.Future[str]":
    """Register a pending transfer. Returns a Future that resolves to 'success'|'failure'."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()
    _pending[call_sid] = fut
    return fut


def resolve(call_sid: str, status: str) -> bool:
    """Resolve a pending transfer Future. Returns True if a live future was found."""
    fut = _pending.pop(call_sid, None)
    if fut is not None and not fut.done():
        fut.set_result(status)
        log.info("transfer resolved", extra={"call_sid": call_sid, "status": status})
        return True
    log.warning("transfer resolve: no pending future", extra={"call_sid": call_sid})
    return False


def cancel_pending(call_sid: str) -> None:
    """Cancel a pending Future (call dropped before CS responded)."""
    fut = _pending.pop(call_sid, None)
    if fut is not None and not fut.done():
        fut.cancel()
