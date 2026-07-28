from __future__ import annotations

from src.api import dev_call_control as dcc


def test_monitor_status_then_outcome():
    m = dcc._Monitor()
    assert m.get("c1") is None
    m.set_status("c1", "calling")
    assert m.get("c1") == {"status": "calling", "outcome": None}
    m.set_status("c1", "answered")
    assert m.get("c1")["status"] == "answered"
    m.set_outcome("c1", {"outcome": "interested", "summary": "ok"})
    got = m.get("c1")
    assert got["status"] == "ended"
    assert got["outcome"] == {"outcome": "interested", "summary": "ok"}


def test_monitor_outcome_before_status_creates_entry():
    m = dcc._Monitor()
    m.set_outcome("c2", {"outcome": "x"})
    assert m.get("c2") == {"status": "ended", "outcome": {"outcome": "x"}}


def test_monitor_ttl_evicts_stale_entries():
    m = dcc._Monitor(ttl_seconds=100.0)
    m.set_status("c3", "calling")
    m._items["c3"]["ts"] -= 1000  # make it look old
    assert m.get("c3") is None


def test_override_set_and_pop_is_one_shot():
    dcc.set_override("dev", mode="s2s", voice="Kore", lead_name="Raju")
    assert dcc.pop_override("dev") == {
        "mode": "s2s", "voice": "Kore", "caller_name": "", "lead_name": "Raju",
        "lead_gender": "", "transfer_webhook_url": "",
    }
    assert dcc.pop_override("dev") is None
    assert dcc.pop_override("never-set") is None
