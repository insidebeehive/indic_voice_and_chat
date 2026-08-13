"""ChatBot persistence: chat sessions + messages (PRD §7).

A ``ChatSession`` is one customer conversation with a tenant's ChatBot; each
turn (customer or agent) is a ``ChatMessage``. The ``session_id`` doubles as the
WebSocket capability — it's minted by an authenticated ``POST /chat/sessions``
and the WS resolves the owning tenant from the session row, so the browser
client never needs to send tenant credentials over the socket.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    customer_id: Mapped[Optional[str]] = mapped_column(String(100))
    customer_name: Mapped[Optional[str]] = mapped_column(String(255))
    language: Mapped[str] = mapped_column(String(10), default="hi")
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|ended|escalated
    # ``metadata`` is reserved on the Declarative class, so map the column under
    # that name via a differently-named attribute.
    extra_data: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    satisfaction: Mapped[Optional[str]] = mapped_column(String(20))  # positive|neutral|negative
    mode: Mapped[str] = mapped_column(String(20), default="ai")  # ai|awaiting_human|human|closed
    claimed_by: Mapped[Optional[str]] = mapped_column(String(100))
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    # Running totals across the session's agent turns (chat cost tracking).
    # NOT NULL / default 0 so SUM() across sessions never needs coalescing.
    cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # customer|agent|human_agent|system
    type: Mapped[str] = mapped_column(String(10), default="text")  # text|image|video|tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    media_url: Mapped[Optional[str]] = mapped_column(String(500))
    media_mime: Mapped[Optional[str]] = mapped_column(String(50))
    sources: Mapped[Optional[dict]] = mapped_column(JSON)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSON)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
    # Cost tracking — set only on the agent's own turn (role="agent"); a
    # customer/human-agent message has no LLM cost of its own.
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    cost: Mapped[Optional[float]] = mapped_column(Float)
    llm_provider: Mapped[Optional[str]] = mapped_column(String(30))
    llm_model: Mapped[Optional[str]] = mapped_column(String(60))


class ChatTool(Base):
    """A tenant-registered CRM API endpoint the ChatBot may call as a tool
    (PRD §4.6). The auth token is NOT stored here — it lives encrypted in
    ``tenant_secrets`` under ``auth_config['token_secret_name']``."""

    __tablename__ = "chat_tools"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_chat_tools_tenant_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(500), nullable=False)
    method: Mapped[str] = mapped_column(String(10), default="GET")
    auth_type: Mapped[Optional[str]] = mapped_column(String(20))  # bearer|api_key|none
    auth_config: Mapped[dict] = mapped_column(JSON, default=dict)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
