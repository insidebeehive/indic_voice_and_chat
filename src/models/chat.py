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

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
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


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)  # customer|agent|system
    type: Mapped[str] = mapped_column(String(10), default="text")  # text|image|video|tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    media_url: Mapped[Optional[str]] = mapped_column(String(500))
    media_mime: Mapped[Optional[str]] = mapped_column(String(50))
    sources: Mapped[Optional[dict]] = mapped_column(JSON)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSON)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
