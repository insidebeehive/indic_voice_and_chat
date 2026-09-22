"""In-process store for pending human-transfer futures.

When the AI voicebot fires the 'transfer' action, the telephony bridge
registers a Future here keyed by ``(tenant_id, provider_call_sid)``. The
coordination server resolves it by POSTing to
/api/v1/calls/{call_sid}/transfer-result.

The key is the composite ``(tenant_id, call_sid)`` — not ``call_sid`` alone —
so that cross-tenant access is structurally impossible: a tenant's bearer
token can only ever index its own entries in this dict, because the route
handler resolves using the tenant_id from its own auth context. This makes
the API-layer authorization something the data structure enforces, rather
than a check that has to be remembered (and could be forgotten) at every call
site that touches the store.
"""

from __future__ import annotations

import asyncio
import logging

from src.utils.logging import debug_event

log = logging.getLogger(__name__)

_pending: dict[tuple[str, str], "asyncio.Future[str]"] = {}


def register(tenant_id: str, call_sid: str) -> "asyncio.Future[str]":
    """Register a pending transfer. Returns a Future that resolves to 'success'|'failure'."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()
    _pending[(tenant_id, call_sid)] = fut
    debug_event(log, "transfer_store register request", tenant_id=tenant_id, call_sid=call_sid)
    return fut


def resolve(tenant_id: str, call_sid: str, status: str) -> bool:
    """Resolve a pending transfer Future. Returns True if a live future was found."""
    fut = _pending.pop((tenant_id, call_sid), None)
    if fut is not None and not fut.done():
        fut.set_result(status)
        log.info("transfer resolved", extra={"call_sid": call_sid, "status": status})
        # Supplements the INFO line above with tenant_id, which that line
        # doesn't carry -- useful when the same call_sid space is being
        # reasoned about across tenants during an investigation.
        debug_event(
            log, "transfer_store resolve response",
            tenant_id=tenant_id, call_sid=call_sid, status=status, resolved=True,
        )
        return True
    log.warning("transfer resolve: no pending future", extra={"call_sid": call_sid})
    debug_event(
        log, "transfer_store resolve response",
        tenant_id=tenant_id, call_sid=call_sid, status=status, resolved=False,
        future_existed=(fut is not None),
    )
    return False


def cancel_pending(tenant_id: str, call_sid: str) -> None:
    """Cancel a pending Future (call dropped before CS responded)."""
    fut = _pending.pop((tenant_id, call_sid), None)
    cancelled = False
    if fut is not None and not fut.done():
        fut.cancel()
        cancelled = True
    debug_event(
        log, "transfer_store cancel_pending response",
        tenant_id=tenant_id, call_sid=call_sid,
        future_existed=(fut is not None), cancelled=cancelled,
    )
