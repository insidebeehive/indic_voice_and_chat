"""Tenant ORM tables.

A tenant is the unit of multi-tenancy: each merchant gets one ``Tenant``
row, plus N ``TenantPhoneNumber`` rows (Twilio numbers they own) and M
``TenantApiKey`` rows (bearer tokens issued for their API access).

API keys are stored hashed — only the hash hits the DB. The plaintext
token is shown once at creation time and never persisted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.database import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    slug: Mapped[str] = mapped_column(String(63), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active")
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata")
    default_language: Mapped[str] = mapped_column(String(10), default="hi")
    mode: Mapped[str] = mapped_column(String(20), default="layered")  # layered | s2s
    max_concurrent_calls: Mapped[int] = mapped_column(Integer, default=1)
    # Full TenantPipelineConfig minus secrets (provider/model/voice/language/
    # from_number/webhook_base_url/outbound_from). Reconstructs a TenantSettings.
    pipeline_config: Mapped[dict] = mapped_column(JSON, default=dict)
    crm_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    phone_numbers: Mapped[list["TenantPhoneNumber"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list["TenantApiKey"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )
    secrets: Mapped[list["TenantSecret"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )


class TenantPhoneNumber(Base):
    """Maps an inbound phone number (Twilio number) to its owning tenant.

    Used by the Twilio voice webhook to look up *which* tenant the call
    belongs to before any further routing. Phone numbers are stored in the
    normalized form (``+E.164``); see ``src.campaign.dnd_filter.normalize_phone``.
    """

    __tablename__ = "tenant_phone_numbers"

    phone_number: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(20), default="twilio")
    label: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )

    tenant: Mapped[Tenant] = relationship(back_populates="phone_numbers")


class TenantApiKey(Base):
    """SHA-256 hash of a bearer token. The plaintext is never stored."""

    __tablename__ = "tenant_api_keys"
    __table_args__ = (
        UniqueConstraint("tenant_id", "label", name="uq_tenant_api_key_label"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")


class TenantSecret(Base):
    """A per-tenant **telephony** provider key, encrypted at rest (Fernet).

    Only telephony keys live here (Twilio SID/token, Exotel key, Stringee
    SID/secret, …) — STT/LLM/TTS/S2S use shared master keys from the platform
    env, never per tenant. ``name`` is the logical key name (e.g. ``twilio_sid``).
    """

    __tablename__ = "tenant_secrets"

    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True
    )
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    tenant: Mapped[Tenant] = relationship(back_populates="secrets")


class ProviderCost(Base):
    """Current cost/min per provider+model, by kind. The cost-maintenance catalog.

    Keyed ``(kind, provider, model)``. ``model`` is the specific model id (e.g.
    ``gemini-2.5-flash``) so STT/LLM/TTS/S2S can be priced per variant; for
    telephony (no model) and as a provider-level fallback it is ``""``.

    Seeded from ``config/provider_costs.yaml`` and maintained live via
    ``PUT /api/v1/providers/{kind}/{provider}`` (model in the body). Single source
    of truth for ``GET /providers`` + per-call cost. Not tenant-scoped.
    """

    __tablename__ = "provider_costs"

    kind: Mapped[str] = mapped_column(String(20), primary_key=True)  # stt|llm|tts|s2s|telephony
    provider: Mapped[str] = mapped_column(String(40), primary_key=True)
    model: Mapped[str] = mapped_column(String(60), primary_key=True, default="")
    cost_per_min: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Chat (text) is billed per-token, not per-minute — used only for kind="llm".
    cost_per_1k_input_tokens: Mapped[float] = mapped_column(Float, default=0.0)
    cost_per_1k_output_tokens: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )
