"""Redis-backed session store (PRD §6.2).

Keys:
    session:{id}:state    JSON object  (set/replace)
    session:{id}:history  JSON list    (append)
    session:{id}:slots    Hash         (per-field set)

All keys share a single TTL refreshed on every write so an active session
doesn't expire mid-conversation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from redis.asyncio import Redis

from src.utils.logging import debug_event

log = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 1800, tenant_id: Optional[str] = None) -> None:
        self.redis = redis
        self.ttl = ttl_seconds
        # Tenant id is folded into every Redis key so two tenants can share
        # the same physical Redis without colliding.
        self.tenant_id = tenant_id

    # --- Keys ------------------------------------------------------------

    def _prefix(self) -> str:
        return f"tenant:{self.tenant_id}:" if self.tenant_id else ""

    def _state_key(self, session_id: str) -> str:
        return f"{self._prefix()}session:{session_id}:state"

    def _history_key(self, session_id: str) -> str:
        return f"{self._prefix()}session:{session_id}:history"

    def _slots_key(self, session_id: str) -> str:
        return f"{self._prefix()}session:{session_id}:slots"

    # --- State -----------------------------------------------------------

    async def set_state(self, session_id: str, state: dict[str, Any]) -> None:
        key = self._state_key(session_id)
        # Write boundary with no logging of its own until now — the only callers
        # (src/agents/base.py's persist_state) already log the full payload
        # before calling this, but only that ONE caller; anything reaching
        # SessionStore directly (bootstrap wiring, tests, a future caller) had
        # no trace of the write actually happening or what key/TTL it landed
        # under. See docs/debug-logging.md's "why the skip category exists".
        debug_event(
            log, "session_store set_state request",
            session_id=session_id, tenant_id=self.tenant_id, redis_key=key,
            state=state, ttl=self.ttl,
        )
        await self.redis.set(key, json.dumps(state), ex=self.ttl)

    async def get_state(self, session_id: str) -> Optional[dict[str, Any]]:
        key = self._state_key(session_id)
        raw = await self.redis.get(key)
        if raw is None:
            # A miss here means the caller's session state is gone (expired,
            # never written, or a key/tenant-prefix mismatch) — indistinguishable
            # from a fresh session to everything downstream unless this fires.
            debug_event(
                log, "session_store get_state miss",
                session_id=session_id, tenant_id=self.tenant_id, redis_key=key,
            )
            return None
        state = json.loads(raw)
        debug_event(
            log, "session_store get_state hit",
            session_id=session_id, tenant_id=self.tenant_id, redis_key=key, state=state,
        )
        return state

    # --- History ---------------------------------------------------------

    async def append_history(self, session_id: str, turn: dict[str, Any]) -> None:
        key = self._history_key(session_id)
        debug_event(
            log, "session_store append_history request",
            session_id=session_id, tenant_id=self.tenant_id, redis_key=key,
            turn=turn, ttl=self.ttl,
        )
        await self.redis.rpush(key, json.dumps(turn))
        await self.redis.expire(key, self.ttl)

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        key = self._history_key(session_id)
        items = await self.redis.lrange(key, 0, -1)
        history = [json.loads(i) for i in items]
        debug_event(
            log, "session_store get_history response",
            session_id=session_id, tenant_id=self.tenant_id, redis_key=key,
            turn_count=len(history), history=history,
        )
        return history

    # --- Slots -----------------------------------------------------------

    async def set_slot(self, session_id: str, name: str, value: Any) -> None:
        key = self._slots_key(session_id)
        debug_event(
            log, "session_store set_slot request",
            session_id=session_id, tenant_id=self.tenant_id, redis_key=key,
            slot_name=name, value=value, ttl=self.ttl,
        )
        await self.redis.hset(key, name, json.dumps(value))
        await self.redis.expire(key, self.ttl)

    async def get_slots(self, session_id: str) -> dict[str, Any]:
        key = self._slots_key(session_id)
        raw = await self.redis.hgetall(key)
        slots = {
            (k.decode() if isinstance(k, bytes) else k): json.loads(v)
            for k, v in raw.items()
        }
        debug_event(
            log, "session_store get_slots response",
            session_id=session_id, tenant_id=self.tenant_id, redis_key=key,
            slot_count=len(slots), slots=slots,
        )
        return slots

    # --- Lifecycle -------------------------------------------------------

    async def delete(self, session_id: str) -> None:
        # Session teardown: a session that stops working right after this,
        # with no trace of it having been deleted, used to be indistinguishable
        # from a store outage.
        debug_event(
            log, "session_store delete request",
            session_id=session_id, tenant_id=self.tenant_id,
        )
        await self.redis.delete(
            self._state_key(session_id),
            self._history_key(session_id),
            self._slots_key(session_id),
        )
