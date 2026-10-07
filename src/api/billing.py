"""Tenant billing: date-ranged cost computation + immutable monthly snapshots.

``compute_tenant_billing`` is the single source of truth for a tenant's
platform cost over an arbitrary ``[start, end)`` naive-UTC window — it is
used both live, by ``GET /api/v1/tenants/{tenant_id}/billing``
(``src/api/tenants.py``), and to freeze a month into a
``TenantBillingSnapshot`` row (``snapshot_tenant_month`` /
``snapshot_previous_month_all_tenants`` below). Keeping one implementation
means a live figure and the frozen one it was taken from are computed
exactly the same way.

Chat sessions are bucketed by ``ChatSession.started_at`` (session start), not
per-message: voice-note STT cost lives only on ``ChatSession.cost`` and has
no per-message row of its own, so filtering at the message level would lose
it. A session that starts in one range and ends in another is counted
entirely in the range containing its start.

**Timezones.** A tenant's billing "month" (and the default date range) is a
calendar month in the TENANT's own timezone (``Tenant.timezone``, an IANA
name, e.g. ``"Asia/Kolkata"``) — not UTC. ``Conversation.started_at`` /
``ChatSession.started_at`` are stored tz-naive UTC, so every local boundary
is converted to naive UTC (via ``zoneinfo``) before it ever reaches a query;
``compute_tenant_billing`` itself still only deals in UTC ``datetime``s and
knows nothing about timezones. ``resolve_timezone_name`` falls back to UTC
(logging a warning) for an empty or invalid ``Tenant.timezone`` so a bad
value can never 500 a billing request.

**Grace period.** A month is only ever frozen (auto job or manual POST)
once it has cleared ``SNAPSHOT_GRACE_DAYS`` days in the tenant's own
timezone -- see that constant and ``month_freezable``/``freezable_from``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.billing_snapshot import TenantBillingSnapshot
from src.models.chat import ChatSession
from src.models.conversation import Conversation
from src.models.tenant import ProviderCost, Tenant

log = logging.getLogger(__name__)

_UTC = "UTC"

# A local calendar month may only be frozen once it's been over for this many
# LOCAL days. Costs keep landing on a session/call that started within the
# month after the month itself has already turned over: ChatSession.cost
# grows with every turn for as long as that session stays open, and
# Conversation.cost is written once, at call end -- not when the call
# started. Both are bucketed by started_at (see module docstring), so a
# session/call that started in the last hours of the month but is still
# open (or hasn't ended yet) when the month flips would be missed, or frozen
# with a stale/partial cost, if the snapshot were taken the instant the
# month ended. The grace period gives those in-flight costs time to settle
# before the month is locked for good.
SNAPSHOT_GRACE_DAYS = 2


def resolve_timezone_name(tz_name: Optional[str]) -> str:
    """Validate ``tz_name`` as an IANA zone name; fall back to ``"UTC"`` (and
    log a warning) when it's empty or not a real zone. Every other helper in
    this module takes the RESOLVED name (never the raw ``Tenant.timezone``
    value) so the zone actually used for a computation always matches the
    zone recorded alongside it (e.g. ``TenantBillingSnapshot.timezone``)."""
    name = (tz_name or "").strip() or _UTC
    try:
        ZoneInfo(name)
    except Exception:  # noqa: BLE001 - any bad/unknown zone name falls back the same way
        log.warning("billing: invalid tenant timezone %r, falling back to UTC", tz_name)
        return _UTC
    return name


def local_period_to_utc(tz_name: str, from_date: date, to_date: date) -> tuple[datetime, datetime]:
    """Convert an inclusive local calendar-day range ``[from_date, to_date]``
    (both inclusive, in the ``tz_name`` zone) to a half-open ``[start, end)``
    pair of naive UTC datetimes suitable for filtering
    ``Conversation.started_at`` / ``ChatSession.started_at``.

    ``tz_name`` must already be a resolved, valid zone name (see
    ``resolve_timezone_name``) -- this does not fall back itself."""
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(from_date, time.min, tzinfo=tz)
    end_local = datetime.combine(to_date + timedelta(days=1), time.min, tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_local.astimezone(timezone.utc).replace(tzinfo=None)
    return start_utc, end_utc


def current_local_month(tz_name: str, now_utc: datetime) -> tuple[date, date]:
    """Return ``(first_day_of_current_local_month, today_local)`` as dates in
    the ``tz_name`` zone, given ``now_utc`` (naive UTC, e.g.
    ``datetime.now(timezone.utc).replace(tzinfo=None)``)."""
    tz = ZoneInfo(tz_name)
    now_local = now_utc.replace(tzinfo=timezone.utc).astimezone(tz)
    today_local = now_local.date()
    return today_local.replace(day=1), today_local


def previous_local_month(tz_name: str, now_utc: datetime) -> date:
    """First day of the local calendar month BEFORE the one ``now_utc`` falls
    in, in the ``tz_name`` zone -- the month the auto-snapshot job freezes."""
    first_this_month, _ = current_local_month(tz_name, now_utc)
    return (first_this_month - timedelta(days=1)).replace(day=1)


def _next_month(month_start: date) -> date:
    """First day of the calendar month after ``month_start`` (itself assumed
    to already be the 1st of its month)."""
    return (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)


def freezable_from(month_start: date) -> date:
    """First LOCAL calendar date on which ``month_start``'s month may be
    frozen: ``SNAPSHOT_GRACE_DAYS`` after it ends (see that constant for
    why). Pure date arithmetic -- the month's own calendar shape is the same
    in every timezone, so this doesn't need ``tz_name`` itself; it's
    ``month_freezable`` (below) that resolves this against a *local* "today"."""
    return _next_month(month_start) + timedelta(days=SNAPSHOT_GRACE_DAYS)


def month_freezable(tz_name: str, month_start: date, now_utc: datetime) -> bool:
    """Whether ``month_start``'s local calendar month (resolved in
    ``tz_name``) is already past its grace period as of ``now_utc`` (naive
    UTC) -- see ``freezable_from``/``SNAPSHOT_GRACE_DAYS``. The single
    implementation of this rule, used by both the auto job
    (``snapshot_previous_month_all_tenants`` below) and the manual freeze
    endpoint (``src/api/tenants.py``'s ``create_tenant_billing_snapshot``)
    so it can only ever be expressed once."""
    _, today_local = current_local_month(tz_name, now_utc)
    return today_local >= freezable_from(month_start)


def tenant_existed_in_month(
    tenant_created_at: Optional[datetime], tz_name: str, month_start: date,
) -> bool:
    """Whether a tenant (``Tenant.created_at``, naive UTC) already existed
    during the LOCAL calendar month starting ``month_start`` (resolved in
    ``tz_name``) -- i.e. it was created before that month's UTC end. ``None``
    (a legacy row with no recorded creation time) always counts as "existed".

    Single implementation of this rule, shared by the auto job
    (``snapshot_previous_month_all_tenants`` below), the platform-wide
    billing-snapshots view's ``not_frozen`` list, and the manual-freeze POST
    (both in ``src/api/tenants.py``) -- a tenant that didn't exist yet during
    a given month should never be offered a snapshot for it, whether via the
    auto job, the "needs freezing" list, or a direct API call."""
    if tenant_created_at is None:
        return True
    _, month_end_utc = local_period_to_utc(
        tz_name, month_start, _next_month(month_start) - timedelta(days=1),
    )
    return tenant_created_at < month_end_utc


async def compute_tenant_billing(
    session: AsyncSession, tenant_id: str, start: datetime, end: datetime,
) -> dict:
    """Platform cost (voice + chat combined) for ``tenant_id`` over the
    half-open naive-UTC window ``[start, end)``, filtered on
    ``Conversation.started_at`` / ``ChatSession.started_at`` respectively.
    Same fields ``GET .../billing`` has always returned, plus
    ``telephony_rates`` (the ``{provider: cost_per_min}`` map used to compute
    ``tentative_telephony_cost``) so a snapshot can freeze the rates it was
    taken with alongside the figure they produced."""
    rows = (await session.execute(
        select(Conversation.cost, Conversation.duration_ms, Conversation.telephony_provider)
        .where(
            Conversation.tenant_id == tenant_id,
            Conversation.started_at >= start, Conversation.started_at < end,
        )
    )).all()
    # telephony rates (model="") for the tentative figure
    tel_rates = dict((p, c) for p, c in (await session.execute(
        select(ProviderCost.provider, ProviderCost.cost_per_min)
        .where(ProviderCost.kind == "telephony", ProviderCost.model == "")
    )).all())

    voice_cost = 0.0
    tentative_tel = 0.0
    total_ms = 0
    for cost, dur, tel in rows:
        voice_cost += float(cost or 0.0)
        total_ms += int(dur or 0)
        if tel and dur:
            tentative_tel += tel_rates.get(tel, 0.0) * (int(dur) / 60_000.0)
    n = len(rows)

    chat_row = (await session.execute(
        select(
            func.count(ChatSession.id), func.coalesce(func.sum(ChatSession.cost), 0.0),
            func.coalesce(func.sum(ChatSession.input_tokens), 0),
            func.coalesce(func.sum(ChatSession.output_tokens), 0),
        ).where(
            ChatSession.tenant_id == tenant_id,
            ChatSession.started_at >= start, ChatSession.started_at < end,
        )
    )).one()
    chat_sessions, chat_cost, chat_in_tok, chat_out_tok = chat_row

    platform = voice_cost + float(chat_cost or 0.0)
    return {
        "total_calls": n,
        "billable_minutes": round(total_ms / 60_000.0, 4),
        "platform_cost": round(platform, 6),
        "avg_cost_per_call": round(voice_cost / n, 6) if n else 0.0,
        "tentative_telephony_cost": round(tentative_tel, 6),
        "chat_sessions": int(chat_sessions or 0),
        "chat_input_tokens": int(chat_in_tok or 0),
        "chat_output_tokens": int(chat_out_tok or 0),
        "chat_cost": round(float(chat_cost or 0.0), 6),
        "telephony_rates": tel_rates,
    }


async def snapshot_tenant_month(
    session: AsyncSession, tenant: Tenant, month_start: date, created_by: str,
) -> Optional[TenantBillingSnapshot]:
    """Freeze ``tenant``'s billing for the local calendar month starting
    ``month_start`` (first day of that month, in the tenant's own timezone)
    into a new ``TenantBillingSnapshot`` row.

    Returns ``None`` (does nothing) if a snapshot already exists for
    ``(tenant.id, month_start)`` -- checked up front, and again via
    ``IntegrityError`` on the unique constraint (``uq_billing_snapshot_tenant_month``)
    in case another replica races this one between the check and the
    insert."""
    # Captured once, up front: a rollback (in the except block below)
    # expires every attribute on every object in the session, including
    # `tenant`'s -- re-accessing `tenant.id`/`tenant.name` after that would
    # trigger an implicit lazy-reload, which isn't awaitable in an async
    # session and raises MissingGreenlet instead of just working.
    tenant_id = tenant.id
    tenant_name = tenant.name

    existing = (await session.execute(
        select(TenantBillingSnapshot.id).where(
            TenantBillingSnapshot.tenant_id == tenant_id,
            TenantBillingSnapshot.period_month == month_start,
        )
    )).scalar_one_or_none()
    if existing is not None:
        return None

    tz_name = resolve_timezone_name(tenant.timezone)
    start, end = local_period_to_utc(tz_name, month_start, _next_month(month_start) - timedelta(days=1))
    data = await compute_tenant_billing(session, tenant_id, start, end)

    snap = TenantBillingSnapshot(
        tenant_id=tenant_id,
        tenant_name=tenant_name,
        period_month=month_start,
        total_calls=data["total_calls"],
        billable_minutes=data["billable_minutes"],
        platform_cost=data["platform_cost"],
        avg_cost_per_call=data["avg_cost_per_call"],
        tentative_telephony_cost=data["tentative_telephony_cost"],
        telephony_rates=data["telephony_rates"],
        chat_sessions=data["chat_sessions"],
        chat_input_tokens=data["chat_input_tokens"],
        chat_output_tokens=data["chat_output_tokens"],
        chat_cost=data["chat_cost"],
        currency="USD",
        timezone=tz_name,
        created_by=created_by,
    )
    session.add(snap)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        # The race this guards against is specifically on
        # (tenant_id, period_month) -- re-check that before assuming this
        # IntegrityError WAS that race. A different constraint violation
        # (corrupt data, a future column, ...) must not be silently
        # swallowed as "someone else already froze this month"; the caller
        # (the auto job's per-tenant try/except, or the manual POST route)
        # needs to see it as a real failure instead.
        existing_after = (await session.execute(
            select(TenantBillingSnapshot.id).where(
                TenantBillingSnapshot.tenant_id == tenant_id,
                TenantBillingSnapshot.period_month == month_start,
            )
        )).scalar_one_or_none()
        if existing_after is not None:
            return None
        raise
    await session.refresh(snap)
    return snap


async def snapshot_previous_month_all_tenants(
    sessionmaker: async_sessionmaker, now_utc: Optional[datetime] = None,
) -> int:
    """For every tenant, freeze its previous LOCAL calendar month (computed
    per-tenant, in that tenant's own timezone) with ``created_by="auto"``.
    Skips a tenant that already has a snapshot for that month, one whose
    previous local month hasn't cleared ``SNAPSHOT_GRACE_DAYS`` yet (e.g. on
    the local 1st/2nd, this does nothing for that tenant -- see
    ``month_freezable``), and one that didn't exist yet during that month
    (``Tenant.created_at`` on/after the month's end). Each tenant runs in its
    own session/transaction so one tenant's failure (bad data, a transient
    DB error, a non-duplicate ``IntegrityError``, ...) doesn't block the
    rest -- logged and skipped. Returns the number of snapshots actually
    created.

    ``now_utc`` (naive UTC) defaults to the real wall clock; tests pass an
    explicit value so the local-1st/2nd/3rd grace-period behavior above is
    deterministic regardless of which real day the suite happens to run on.
    """
    now = now_utc if now_utc is not None else datetime.now(timezone.utc).replace(tzinfo=None)

    async with sessionmaker() as session:
        tenant_ids = (await session.execute(select(Tenant.id))).scalars().all()

    created = 0
    for tenant_id in tenant_ids:
        try:
            async with sessionmaker() as session:
                tenant = await session.get(Tenant, tenant_id)
                if tenant is None:
                    continue  # deleted between the id listing and this read
                tz_name = resolve_timezone_name(tenant.timezone)
                month_start = previous_local_month(tz_name, now)
                if not month_freezable(tz_name, month_start, now):
                    continue  # late costs may still be settling -- too soon
                if not tenant_existed_in_month(tenant.created_at, tz_name, month_start):
                    continue  # tenant didn't exist yet during this month
                snap = await snapshot_tenant_month(session, tenant, month_start, created_by="auto")
                if snap is not None:
                    created += 1
        except Exception:  # noqa: BLE001 - one tenant's failure must not block the rest
            log.exception("billing snapshot failed for tenant_id=%s", tenant_id)
            continue
    return created
