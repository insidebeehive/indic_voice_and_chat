"""Per-turn STT/LLM/TTS latency metrics, persisted for cross-provider
benchmarking (PRD §7.7). Written best-effort from VoiceBotAgent.apply_signal
so a DB hiccup never affects a live call — see record_turn_metric.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base, get_sessionmaker

log = logging.getLogger(__name__)


class TurnMetric(Base):
    __tablename__ = "turn_metrics"
    __table_args__ = (
        Index("idx_turn_metrics_combo", "stt_provider", "llm_provider", "tts_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    campaign_id: Mapped[Optional[str]] = mapped_column(String(100))
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    stt_provider: Mapped[Optional[str]] = mapped_column(String(100))
    llm_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    tts_provider: Mapped[Optional[str]] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    stt_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_ttft_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tts_first_chunk_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tts_total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


async def record_turn_metric(
    *,
    tenant_id: str,
    session_id: str,
    campaign_id: Optional[str],
    mode: str,
    stt_provider: Optional[str],
    llm_provider: str,
    tts_provider: Optional[str],
    action: str,
    metrics: dict[str, int],
) -> None:
    """Insert one turn-metrics row. Best-effort: never raises — a DB outage
    must degrade to no-persistence, not break a live call (see
    VoiceBotAgent.apply_signal, the only caller)."""
    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as db:
            db.add(TurnMetric(
                tenant_id=tenant_id,
                session_id=session_id,
                campaign_id=campaign_id,
                mode=mode,
                stt_provider=stt_provider,
                llm_provider=llm_provider,
                tts_provider=tts_provider,
                action=action,
                stt_latency_ms=metrics.get("stt_latency_ms", 0),
                llm_ttft_ms=metrics.get("llm_ttft_ms", 0),
                llm_total_ms=metrics.get("llm_total_ms", 0),
                tts_first_chunk_ms=metrics.get("tts_first_chunk_ms", 0),
                tts_total_ms=metrics.get("tts_total_ms", 0),
                total_latency_ms=metrics.get("total_latency_ms", 0),
            ))
            await db.commit()
    except Exception:  # noqa: BLE001 - must never break a live call
        log.warning("record_turn_metric failed; continuing without persistence", exc_info=True)
