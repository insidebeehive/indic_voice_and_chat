"""Deposit dispute screenshot verification requests.

A ``DepositVerificationRequest`` row tracks one customer's claim that a
deposit went through but wasn't credited: the customer supplies an
``order_id`` and (usually) a screenshot message, and the row holds the
resulting verdict once the CRM-side check resolves. ``status`` starts
``pending`` and moves to ``verified``/``rejected``/``timed_out``/``error``;
``timeout_at`` lets a periodic sweep find requests that never resolved.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base


class DepositVerificationRequest(Base):
    __tablename__ = "deposit_verification_requests"
    __table_args__ = (
        Index(
            "idx_deposit_verification_requests_tenant_status_timeout",
            "tenant_id", "status", "timeout_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    session_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    order_id: Mapped[str] = mapped_column(String(200), nullable=False)
    screenshot_message_id: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending|verified|rejected|timed_out|error
    verdict_payload: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    timeout_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
