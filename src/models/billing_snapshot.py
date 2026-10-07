"""Immutable monthly billing snapshot per tenant.

``TenantBillingSnapshot`` persists ``compute_tenant_billing``'s output
(``src/api/billing.py``) for one tenant-month, frozen at the moment it is
taken (``snapshot_tenant_month`` / ``snapshot_previous_month_all_tenants``,
also in ``src/api/billing.py``) so a bill already shown or sent to a tenant
never silently changes afterward, even though:

- ``Tenant`` rows cascade-delete their ``chat_sessions``/``conversations``
  (``ondelete="CASCADE"``), so the live numbers for that month could
  otherwise disappear entirely;
- ``ProviderCost`` telephony rates (``kind="telephony"``) are maintained
  live and applied retroactively to any live query over old calls.

``tenant_id`` is deliberately **NOT** a ``ForeignKey`` to ``tenants.id``: a
snapshot is a billing/audit record that must outlive the tenant row it was
computed from. A real FK (even with ``ondelete="SET NULL"``) would either
cascade-delete the snapshot with the tenant or null out which tenant it was
for — both defeat the point of freezing a bill. ``tenant_name`` is copied in
at snapshot time for the same reason: the tenant's current name is not
joinable once the tenant is gone.

Scoped per calendar month in the tenant's OWN timezone at the time of the
snapshot (``timezone``, an IANA name) — ``period_month`` is the first day of
that local month, not a UTC month boundary. See ``src/api/billing.py``'s
``local_period_to_utc``/``current_local_month``/``previous_local_month`` for
how a local month is turned into the ``[start, end)`` naive-UTC range used to
query ``Conversation.started_at``/``ChatSession.started_at``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import JSON, Date, DateTime, Float, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base


class TenantBillingSnapshot(Base):
    __tablename__ = "tenant_billing_snapshots"
    __table_args__ = (
        UniqueConstraint("tenant_id", "period_month", name="uq_billing_snapshot_tenant_month"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Deliberately not a ForeignKey -- see module docstring: a snapshot must
    # survive the tenant row (and its cascade-deleted chat_sessions/
    # conversations) being deleted. No standalone index here: the unique
    # (tenant_id, period_month) constraint below already indexes tenant_id
    # as its leftmost column, which covers a tenant_id-only lookup
    # (list_tenant_billing_snapshots' WHERE tenant_id = ... ORDER BY
    # period_month DESC) just as well as a second, redundant index would.
    tenant_id: Mapped[str] = mapped_column(String(50), nullable=False)
    tenant_name: Mapped[str] = mapped_column(String(255), nullable=False)
    period_month: Mapped[date] = mapped_column(Date, nullable=False)
    total_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    billable_minutes: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    platform_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_cost_per_call: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tentative_telephony_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Telephony cost_per_min rates actually used to compute
    # tentative_telephony_cost above, by provider -- frozen alongside the
    # figure they produced so a later ProviderCost edit can never be
    # mistaken for having applied retroactively to this snapshot.
    telephony_rates: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    chat_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chat_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chat_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chat_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    # IANA zone name the local calendar month (period_month) was resolved in
    # at snapshot time -- see Tenant.timezone / resolve_timezone_name.
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    # "auto" for the background job (snapshot_previous_month_all_tenants),
    # else the admin label (current_admin_label()) who froze it by hand.
    created_by: Mapped[str] = mapped_column(String(100), nullable=False, default="auto")
