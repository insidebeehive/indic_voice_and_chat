"""Dev-console telephony control: in-memory call monitor + per-call overrides.

Backs the dev console's "place a Twilio/Exotel call and watch it" panel. Two
tiny module-level stores; both are single-process and effectively single-user
(the dev console), so there is no locking:

- ``monitor`` — ``{call_sid -> {status, outcome}}``: a placed call's lifecycle
  (``calling`` -> ``answered`` -> ``ended``) plus the final outcome payload. The
  telephony bridge writes ``answered``/``ended``/outcome (keyed by the Call SID
  it reads off the media-stream ``start`` frame); the console polls it.
- ``overrides`` — ``{tenant_slug -> {mode, voice, lead_name}}``: the Mode/Voice
  the console picked for the next placed call, consumed once by the bridge
  factory (so a phone call honours the console's selectors, not just dev.yaml).

Entries are swept by TTL on access so a long-running process doesn't accumulate
stale call records.
"""

from __future__ import annotations

import time

_MONITOR_TTL = 1800.0  # 30 min: long enough to finish a call + read the outcome


class _Monitor:
    """``call_sid -> {status, outcome}`` with lazy TTL eviction (mirrors the
    ``AudioStore`` dict-with-TTL pattern in telephony_stringee_bridge)."""

    def __init__(self, ttl_seconds: float = _MONITOR_TTL) -> None:
        self._ttl = ttl_seconds
        self._items: dict[str, dict] = {}

    def _sweep(self) -> None:
        cutoff = time.monotonic() - self._ttl
        for sid in [s for s, v in self._items.items() if v["ts"] < cutoff]:
            self._items.pop(sid, None)

    def _entry(self, call_sid: str) -> dict:
        item = self._items.get(call_sid)
        if item is None:
            item = {"status": "unknown", "outcome": None, "ts": time.monotonic()}
            self._items[call_sid] = item
        return item

    def set_status(self, call_sid: str, status: str) -> None:
        self._sweep()
        item = self._entry(call_sid)
        item["status"] = status
        item["ts"] = time.monotonic()

    def set_outcome(self, call_sid: str, outcome: dict) -> None:
        self._sweep()
        item = self._entry(call_sid)
        item["outcome"] = outcome
        item["status"] = "ended"
        item["ts"] = time.monotonic()

    def get(self, call_sid: str) -> dict | None:
        self._sweep()
        item = self._items.get(call_sid)
        if item is None:
            return None
        return {"status": item["status"], "outcome": item["outcome"]}


monitor = _Monitor()


# --- per-call Mode/Voice override (set by the console, popped by the factory) ---

_overrides: dict[str, dict] = {}


def set_override(
    tenant_slug: str,
    *,
    mode: str,
    voice: str = "",
    gender: str = "",
    caller_name: str = "",
    lead_name: str = "",
) -> None:
    _overrides[tenant_slug] = {
        "mode": mode,
        "voice": voice,
        "gender": gender,
        "caller_name": caller_name,
        "lead_name": lead_name,
    }


def pop_override(tenant_slug: str) -> dict | None:
    """Return and clear the pending override for a tenant (one-shot)."""
    return _overrides.pop(tenant_slug, None)
