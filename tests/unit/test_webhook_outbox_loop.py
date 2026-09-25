"""Tests for the webhook-outbox durable-retry loop (src/main.py's
run_webhook_outbox_once / _webhook_outbox_loop / prune_webhook_outbox,
src/models/webhook_outbox.py).

Mirrors tests/unit/test_chat_metrics_prune.py's style: test the pure
single-pass function directly, not the infinite `_*_loop` wrapper (see
run_webhook_outbox_once's own docstring on why it's split out that way, same
as reap_stale_calls vs. _reap_stale_calls_loop).

``deliver_fn`` fakes return ``DeliveryResult`` (src.integration.tenant_events),
matching the real contract of the outbox's default deliver_fn
(_default_webhook_outbox_deliver_fn, which wraps deliver_detailed with
stop_on_permanent=True -- N6).

Most tests pass ``now_fn=lambda: <fixed datetime>`` explicitly (N4:
_finalize_webhook_outbox_row reads its own fresh "now" via now_fn, decoupled
from the claim-time `now` argument) so timestamp assertions stay
deterministic instead of racing the real wall clock.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import src.main as main
from src.integration.tenant_events import DeliveryResult
from src.models.database import Base
from src.models.webhook_outbox import STATUS_DEAD, STATUS_DELIVERED, STATUS_PENDING, WebhookOutbox

_TENANT = SimpleNamespace(
    id="t1",
    events_webhook_url="https://crm.example/hook",
    events_webhook_secret_env=None,
    secret_optional=lambda env_var: None,
)


async def _resolve_tenant_ok(tenant_id):
    return _TENANT


async def _resolve_tenant_none(tenant_id):
    return None


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()


async def _seed_row(
    sessionmaker, *, row_id="wh_1", session_id="cs_1", next_attempt_at, attempts=0,
    status=STATUS_PENDING, created_at=None,
) -> None:
    async with sessionmaker() as db:
        db.add(WebhookOutbox(
            id=row_id, tenant_id="t1", session_id=session_id, event_type="session_closed",
            url="https://stale.example/hook", body={"event": "session_closed", "session_id": session_id},
            attempts=attempts, next_attempt_at=next_attempt_at, status=status,
            created_at=created_at or (next_attempt_at - timedelta(seconds=1)),
        ))
        await db.commit()


async def _get_row(sessionmaker, row_id="wh_1") -> WebhookOutbox:
    async with sessionmaker() as db:
        return await db.get(WebhookOutbox, row_id)


# --- delivery / reschedule (baseline behaviour, still correct under the
#     H2 claim-then-finalize split) -------------------------------------------


async def test_delivers_due_row_and_marks_delivered(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))

    async def deliver_ok(url, body, secret):
        return DeliveryResult(ok=True, final_status=200)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_ok,
        now=now, now_fn=lambda: now,
    )

    assert n == 1
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_DELIVERED
    assert row.delivered_at == now
    assert row.last_error is None


async def test_skips_rows_not_yet_due(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now + timedelta(minutes=5))

    async def deliver_ok(url, body, secret):
        return DeliveryResult(ok=True, final_status=200)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_ok,
        now=now, now_fn=lambda: now,
    )

    assert n == 0
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_PENDING


async def test_reschedules_on_failure_with_increasing_next_attempt(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))

    async def deliver_fail(url, body, secret):
        return DeliveryResult(ok=False, final_status=503)  # retryable 5xx

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_fail,
        now=now, now_fn=lambda: now,
    )
    assert n == 1
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_PENDING
    assert row.attempts == 1
    assert row.last_error == "http_503"
    first_next = row.next_attempt_at
    assert first_next == now + timedelta(seconds=60)  # first backoff step: 1m

    # Second failed pass, run at (or after) the first backoff's due time —
    # the next backoff step must be strictly longer than the first.
    now2 = first_next
    n2 = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_fail,
        now=now2, now_fn=lambda: now2,
    )
    assert n2 == 1
    row2 = await _get_row(sessionmaker)
    assert row2.attempts == 2
    assert row2.next_attempt_at == now2 + timedelta(seconds=120)  # second step: 2m
    assert row2.next_attempt_at > first_next


async def test_skips_delivery_when_tenant_unresolved_and_reschedules(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))

    calls = []

    async def deliver_should_not_be_called(url, body, secret):
        calls.append(1)
        return DeliveryResult(ok=True, final_status=200)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_none,
        deliver_fn=deliver_should_not_be_called, now=now, now_fn=lambda: now,
    )

    assert n == 1
    assert calls == []  # never attempted delivery — tenant didn't resolve
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_PENDING
    assert row.attempts == 1
    assert row.last_error == "tenant_unresolved"


async def test_does_not_fall_back_to_stored_url_when_current_resolution_is_empty(
    sessionmaker, monkeypatch,
):
    """The row's own stored `url` (captured at enqueue time) must never be
    used as a silent fallback when the tenant's CURRENT webhook resolution
    comes back empty — see _deliver_webhook_outbox_claim's docstring on why
    that would be a data-exposure risk."""
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))

    async def resolve_url_none(tenant, sm):
        return None

    monkeypatch.setattr(
        "src.integration.tenant_events.resolve_events_webhook_url", resolve_url_none,
    )

    calls = []

    async def deliver_should_not_be_called(url, body, secret):
        calls.append(url)
        return DeliveryResult(ok=True, final_status=200)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok,
        deliver_fn=deliver_should_not_be_called, now=now, now_fn=lambda: now,
    )

    assert n == 1
    assert calls == []
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_PENDING
    assert row.last_error == "no_webhook_url_configured"


# --- H1: one bad row must never abort the batch or roll back rows already
#     finalized earlier in the same pass -------------------------------------


async def test_exception_mid_batch_does_not_roll_back_already_delivered_rows(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    # Distinct next_attempt_at so claim order is deterministic: wh_1, wh_2, wh_3.
    await _seed_row(sessionmaker, row_id="wh_1", session_id="cs_1",
                     next_attempt_at=now - timedelta(seconds=30))
    await _seed_row(sessionmaker, row_id="wh_2", session_id="cs_2",
                     next_attempt_at=now - timedelta(seconds=20))
    await _seed_row(sessionmaker, row_id="wh_3", session_id="cs_3",
                     next_attempt_at=now - timedelta(seconds=10))

    async def deliver_middle_raises(url, body, secret):
        if body.get("session_id") == "cs_2":
            raise RuntimeError("boom")
        return DeliveryResult(ok=True, final_status=200)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok,
        deliver_fn=deliver_middle_raises, now=now, now_fn=lambda: now,
    )

    assert n == 3
    row1 = await _get_row(sessionmaker, "wh_1")
    row2 = await _get_row(sessionmaker, "wh_2")
    row3 = await _get_row(sessionmaker, "wh_3")

    # wh_1 and wh_3 delivered — NOT rolled back by wh_2's exception, since
    # each row is its own committed transaction (H2's per-row finalize), not
    # one shared batch transaction.
    assert row1.status == STATUS_DELIVERED
    assert row3.status == STATUS_DELIVERED
    # wh_2 rescheduled, not stuck / not crashing the pass.
    assert row2.status == STATUS_PENDING
    assert row2.attempts == 1
    assert row2.last_error == "RuntimeError"


# --- H2: short-lease claim, never held open across delivery -----------------


async def test_lease_is_committed_before_delivery_is_attempted(sessionmaker):
    """A separate session must already see the claim's lease (bumped attempts
    + pushed-out next_attempt_at) the moment delivery starts — proving the
    claim transaction was committed BEFORE delivery, not held open across it
    (an earlier design's H2 bug)."""
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))

    seen = []

    async def deliver_checks_committed_lease(url, body, secret):
        async with sessionmaker() as probe:
            row = await probe.get(WebhookOutbox, "wh_1")
            seen.append((row.attempts, row.next_attempt_at))
        return DeliveryResult(ok=True, final_status=200)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok,
        deliver_fn=deliver_checks_committed_lease, now=now, now_fn=lambda: now,
    )

    assert n == 1
    assert len(seen) == 1
    attempts_seen, next_attempt_at_seen = seen[0]
    assert attempts_seen == 1  # claim already bumped it
    assert next_attempt_at_seen == now + timedelta(seconds=main._WEBHOOK_OUTBOX_LEASE_S)


async def test_row_not_reclaimed_within_lease_after_crash_mid_batch(sessionmaker):
    """Simulates a crash: claim a row (committing its lease) but never
    finalize it — a second pass, still inside the lease window, must not
    re-claim (and thus never re-deliver) it."""
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))

    claimed = await main._claim_webhook_outbox_rows(sessionmaker, now=now)
    assert len(claimed) == 1  # claimed + leased, but never finalized (the "crash")

    row = await _get_row(sessionmaker)
    assert row.status == STATUS_PENDING  # still pending -- the lease lives in next_attempt_at, not a new status

    async def deliver_should_not_be_called(url, body, secret):
        raise AssertionError("must not re-deliver a row still inside its lease")

    now2 = now + timedelta(seconds=60)  # well inside the 600s lease
    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok,
        deliver_fn=deliver_should_not_be_called, now=now2, now_fn=lambda: now2,
    )

    assert n == 0


# --- N1: fencing -- a stale finalize (from a claim whose lease already
#     expired and got re-claimed) must not clobber the newer claim's state --


async def test_fencing_stale_finalize_after_reclaim_is_a_no_op(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))

    # First claim: attempts -> 1, leased far into the future.
    claim1 = (await main._claim_webhook_outbox_rows(sessionmaker, now=now))[0]
    assert claim1["attempts"] == 1

    # Simulate this claim's lease having already expired (e.g. the pod that
    # claimed it died) and a SECOND pass re-claiming the row.
    async with sessionmaker() as db:
        row = await db.get(WebhookOutbox, "wh_1")
        row.next_attempt_at = now - timedelta(seconds=1)
        await db.commit()

    now2 = now + timedelta(seconds=700)  # past the first lease
    claim2 = (await main._claim_webhook_outbox_rows(sessionmaker, now=now2))[0]
    assert claim2["attempts"] == 2

    async def deliver_ok(url, body, secret):
        return DeliveryResult(ok=True, final_status=200)

    # The STALE first finalize (still holding claim1, attempts=1) runs now --
    # must be a no-op: the row's real attempts is already 2.
    label1 = await main._finalize_webhook_outbox_row(
        sessionmaker, claim1, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_ok,
        now_fn=lambda: now2,
    )
    assert label1 == "skipped"

    row_after_stale = await _get_row(sessionmaker)
    assert row_after_stale.status == STATUS_PENDING  # untouched by the stale finalize
    assert row_after_stale.attempts == 2  # still claim2's value, not reverted

    # The real (second) finalize applies normally.
    label2 = await main._finalize_webhook_outbox_row(
        sessionmaker, claim2, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_ok,
        now_fn=lambda: now2,
    )
    assert label2 == "delivered"
    row_final = await _get_row(sessionmaker)
    assert row_final.status == STATUS_DELIVERED


# --- N1: per-row delivery is bounded; a timeout is retryable, not fatal -----


async def test_per_row_delivery_timeout_is_treated_as_retryable(sessionmaker, monkeypatch):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5))
    monkeypatch.setattr(main, "_WEBHOOK_OUTBOX_PER_ROW_TIMEOUT_S", 0.05)

    async def deliver_hangs(url, body, secret):
        await asyncio.sleep(10)  # far longer than the (monkeypatched) per-row bound
        return DeliveryResult(ok=True, final_status=200)  # pragma: no cover

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_hangs,
        now=now, now_fn=lambda: now,
    )

    assert n == 1
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_PENDING  # rescheduled, not dead
    assert row.attempts == 1
    assert row.last_error == "TimeoutError"
    assert row.next_attempt_at == now + timedelta(seconds=60)


# --- item 3: a permanent (non-retryable) 4xx only kills a row on its 3rd+
#     delivery attempt -- a transient 403/404 during a brief CRM deploy must
#     not kill a row on attempt 1 or 2 -----------------------------------


async def test_permanent_4xx_reschedules_on_first_two_attempts_then_dies_on_third(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5), created_at=now)

    async def deliver_permanent_401(url, body, secret):
        return DeliveryResult(ok=False, final_status=401, permanent=True)

    # Attempt 1: rescheduled, NOT dead.
    n1 = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_permanent_401,
        now=now, now_fn=lambda: now,
    )
    assert n1 == 1
    row1 = await _get_row(sessionmaker)
    assert row1.status == STATUS_PENDING
    assert row1.attempts == 1
    assert row1.last_error == "http_401"
    assert row1.next_attempt_at == now + timedelta(seconds=60)  # normal backoff, same as any reschedule

    # Attempt 2: still rescheduled.
    now2 = row1.next_attempt_at
    n2 = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_permanent_401,
        now=now2, now_fn=lambda: now2,
    )
    assert n2 == 1
    row2 = await _get_row(sessionmaker)
    assert row2.status == STATUS_PENDING
    assert row2.attempts == 2

    # Attempt 3: NOW it dies.
    now3 = row2.next_attempt_at
    n3 = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_permanent_401,
        now=now3, now_fn=lambda: now3,
    )
    assert n3 == 1
    row3 = await _get_row(sessionmaker)
    assert row3.status == STATUS_DEAD
    assert row3.attempts == 3
    assert row3.last_error == "http_401"


async def test_permanent_4xx_recovering_before_third_attempt_never_dies(sessionmaker):
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5), created_at=now)

    calls = {"n": 0}

    async def deliver_then_recovers(url, body, secret):
        calls["n"] += 1
        if calls["n"] == 1:
            return DeliveryResult(ok=False, final_status=403, permanent=True)
        return DeliveryResult(ok=True, final_status=200)

    await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_then_recovers,
        now=now, now_fn=lambda: now,
    )
    row1 = await _get_row(sessionmaker)
    assert row1.status == STATUS_PENDING  # attempt 1 was permanent, but not dead

    now2 = row1.next_attempt_at
    await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok, deliver_fn=deliver_then_recovers,
        now=now2, now_fn=lambda: now2,
    )
    row2 = await _get_row(sessionmaker)
    assert row2.status == STATUS_DELIVERED  # recovered on attempt 2 -- never reached the 3-strike cap


async def test_retryable_failure_reschedules_same_as_permanent_pre_threshold(sessionmaker):
    """A non-permanent failure (5xx, timeout, or a retryable 4xx like
    408/429) reschedules exactly the same way a permanent failure does
    before hitting the 3-strike threshold -- same backoff schedule either
    way."""
    now = datetime(2026, 1, 1, 12, 0, 0)
    await _seed_row(sessionmaker, next_attempt_at=now - timedelta(seconds=5), created_at=now)

    async def deliver_retryable_429(url, body, secret):
        return DeliveryResult(ok=False, final_status=429, permanent=False)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok,
        deliver_fn=deliver_retryable_429, now=now, now_fn=lambda: now,
    )

    assert n == 1
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_PENDING
    assert row.last_error == "http_429"
    assert row.next_attempt_at == now + timedelta(seconds=60)


# --- N3: a pending row past the max age is retired AT CLAIM TIME, never
#     claimed for a delivery attempt --------------------------------------


async def test_max_age_row_is_marked_dead_at_claim_not_delivered(sessionmaker):
    now = datetime(2026, 1, 2, 12, 0, 0)
    # Enqueued (created_at) more than 24h before `now`.
    await _seed_row(
        sessionmaker, next_attempt_at=now - timedelta(seconds=5),
        created_at=now - timedelta(hours=25), attempts=6,
    )

    calls = []

    async def deliver_should_not_be_called(url, body, secret):
        calls.append(1)
        return DeliveryResult(ok=True, final_status=200)

    n = await main.run_webhook_outbox_once(
        sessionmaker, resolve_tenant=_resolve_tenant_ok,
        deliver_fn=deliver_should_not_be_called, now=now, now_fn=lambda: now,
    )

    assert n == 0  # never claimed for delivery
    assert calls == []
    row = await _get_row(sessionmaker)
    assert row.status == STATUS_DEAD
    assert row.last_error == "max_age"
    assert row.attempts == 6  # unchanged -- claim retired it without bumping


async def test_claim_returns_only_rows_actually_claimed_not_retired(sessionmaker):
    now = datetime(2026, 1, 2, 12, 0, 0)
    await _seed_row(sessionmaker, row_id="wh_old", session_id="cs_old",
                     next_attempt_at=now - timedelta(seconds=5),
                     created_at=now - timedelta(hours=25))
    await _seed_row(sessionmaker, row_id="wh_fresh", session_id="cs_fresh",
                     next_attempt_at=now - timedelta(seconds=5), created_at=now)

    claimed = await main._claim_webhook_outbox_rows(sessionmaker, now=now)

    assert [c["id"] for c in claimed] == ["wh_fresh"]
    old_row = await _get_row(sessionmaker, "wh_old")
    assert old_row.status == STATUS_DEAD
    assert old_row.last_error == "max_age"


# --- L3: prune old terminal rows ---------------------------------------------


async def test_prune_deletes_old_terminal_rows_keeps_pending_and_recent(sessionmaker):
    # prune_webhook_outbox computes its own SQLite cutoff from the REAL
    # datetime.utcnow() (mirrors prune_chat_turn_metrics's identical
    # SQLite-only convention — no injectable `now` there either), so this
    # test's seeded timestamps must be anchored to real utcnow() too, not an
    # arbitrary fixed date.
    now = datetime.utcnow()
    old = now - timedelta(days=8)
    recent = now - timedelta(days=1)

    await _seed_row(sessionmaker, row_id="wh_old_delivered", session_id="cs_a",
                     next_attempt_at=old, created_at=old, status=STATUS_DELIVERED)
    await _seed_row(sessionmaker, row_id="wh_old_dead", session_id="cs_b",
                     next_attempt_at=old, created_at=old, status=STATUS_DEAD)
    await _seed_row(sessionmaker, row_id="wh_old_pending", session_id="cs_c",
                     next_attempt_at=old, created_at=old, status=STATUS_PENDING)
    await _seed_row(sessionmaker, row_id="wh_recent_delivered", session_id="cs_d",
                     next_attempt_at=recent, created_at=recent, status=STATUS_DELIVERED)

    n = await main.prune_webhook_outbox(sessionmaker, retention_days=7)

    assert n == 2  # only the two OLD terminal rows
    assert await _get_row(sessionmaker, "wh_old_delivered") is None
    assert await _get_row(sessionmaker, "wh_old_dead") is None
    # Old but still pending -- never pruned here, regardless of age (N3's
    # claim-time retirement is what retires those, not this prune job).
    assert (await _get_row(sessionmaker, "wh_old_pending")) is not None
    # Recent terminal row -- not old enough yet.
    assert (await _get_row(sessionmaker, "wh_recent_delivered")) is not None


@pytest.mark.parametrize("raw, expected", [
    (None, True), ("", True), ("1", True), ("true", True), ("on", True),
    ("0", False), ("false", False), ("No", False), (" OFF ", False),
])
def test_webhook_outbox_kill_switch(monkeypatch, raw, expected) -> None:
    from src import main as main_mod
    if raw is None:
        monkeypatch.delenv("WEBHOOK_OUTBOX_ENABLED", raising=False)
    else:
        monkeypatch.setenv("WEBHOOK_OUTBOX_ENABLED", raw)
    assert main_mod.webhook_outbox_enabled() is expected
