"""CRM entity: the downstream operator/player-data backend (e.g. BetStudio).

A ``Crm`` row is CRM-level config shared by every tenant registered against
it (``Tenant.crm_id``): the base URL its tool endpoints are joined onto, the
webhook URL template events are POSTed to (with ``{operator_id}`` substituted
from the tenant's own operator_id at send time), and the default auth header
style. Genuinely per-tenant/operator values — the auth token/x-api-key,
operator_id itself — stay on the tenant (``tenant_secrets`` / pipeline_config),
completely untouched by this entity.

``CrmTool`` rows are this CRM's tool catalog (replaces the old hardcoded
``src.chatbot.catalog.ALL_TOOLS`` dict) — one row per tool, DB-backed so a
new CRM's tools can be registered without a code deploy.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base


class Crm(Base):
    __tablename__ = "crms"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    events_webhook_url_template: Mapped[str | None] = mapped_column(String(500))
    auth_type: Mapped[str] = mapped_column(String(20), default="api_key")  # api_key|bearer
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())


class CrmTool(Base):
    __tablename__ = "crm_tools"
    __table_args__ = (UniqueConstraint("crm_id", "name", name="uq_crm_tools_crm_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    crm_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(500), nullable=False)
    method: Mapped[str] = mapped_column(String(10), default="GET")
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())


class CrmKBDocument(Base):
    """CRM-level KB documents shared across every tenant registered against
    that CRM (replaces the old platform-wide, unscoped ``PlatformKBDocument``).
    """

    __tablename__ = "crm_kb_documents"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    crm_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[Optional[str]] = mapped_column(String(50))
    language: Mapped[Optional[str]] = mapped_column(String(10))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    extra_data: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
