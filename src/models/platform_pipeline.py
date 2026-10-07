"""Admin-editable platform-wide STT/LLM/TTS pipeline-default overrides.

``config/default.yaml`` ships the platform's STT/LLM/TTS provider+model, but
changing it means an edit + deploy. This table lets an admin override any of
the three layers live, from the admin console's "Platform pipeline defaults"
card (``src/api/platform.py``), without a restart: a row here is read at
every boot (``src/main.py``'s ``lifespan``) and applied on top of the yaml
value, and writing/deleting a row also applies the change to the already-
running process (``TenantProviders.reset_platform_defaults``,
``src/auth/registry.py``).

One row per layer — ``layer`` is the primary key, so there is at most one
platform-wide default per layer (``"stt"``, ``"llm"``, or ``"tts"``; no other
values are ever written). No row for a layer means ``config/default.yaml``'s
own value is in effect, same as before this table existed.

``model`` is nullable: some providers have no model dimension at all (e.g.
Azure/Google TTS, where voice is the only selectable knob) — ``None`` there
means "this provider has nothing to pick", not "unset".
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base


class PlatformPipelineDefault(Base):
    __tablename__ = "platform_pipeline_defaults"

    layer: Mapped[str] = mapped_column(String(10), primary_key=True)
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    updated_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
