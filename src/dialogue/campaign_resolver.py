"""Per-tenant, DB-backed campaign resolution.

The live call resolves which campaign (agent script + slots) to run from the DB
``campaigns`` table — per-tenant, and per-call by ``campaign_id``. There is
intentionally **NO global fallback**: every tenant must have a configured
campaign (the startup seed migrates one in), so a misconfigured tenant fails
loudly rather than silently running some other tenant's / a global script.

Resolution order (per call):
  1. an explicit ``campaign_id`` (guarded so it belongs to the tenant), else
  2. the tenant's single active campaign, else
  3. raise ``CampaignNotConfigured``.

Resolution happens once per call setup (not per turn), so it queries the DB each
time rather than maintaining a cache to invalidate — correctness over a micro-opt.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.dialogue.campaign_loader import LoadedCampaign, parse_campaign_yaml
from src.models.campaign import Campaign

log = logging.getLogger(__name__)


class CampaignNotConfigured(RuntimeError):
    """A tenant has no usable DB campaign. There is no global fallback, so the
    call cannot proceed — the caller should reject the connection clearly."""


class DbCampaignResolver:
    """Resolve a tenant's campaign (script + slots) from the DB."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sm = sessionmaker

    async def _row(
        self, session: AsyncSession, tenant_id: str, campaign_id: Optional[str]
    ) -> Optional[Campaign]:
        if campaign_id:
            row = await session.get(Campaign, campaign_id)
            # Cross-tenant guard: never serve another tenant's campaign.
            if row is not None and row.tenant_id == tenant_id:
                return row
            if row is not None:
                log.warning("campaign %s not owned by tenant %s; ignoring",
                            campaign_id, tenant_id)
        # The tenant's active campaign (most recently created active one).
        return (await session.execute(
            select(Campaign)
            .where(Campaign.tenant_id == tenant_id, Campaign.status == "active")
            .order_by(Campaign.created_at.desc())
            .limit(1)
        )).scalars().first()

    async def resolve(
        self, tenant_id: str, campaign_id: Optional[str] = None
    ) -> LoadedCampaign:
        """Return the script + slots for this call, or raise
        ``CampaignNotConfigured`` when the tenant has no usable campaign."""
        async with self._sm() as session:
            row = await self._row(session, tenant_id, campaign_id)
        if row is None:
            raise CampaignNotConfigured(
                f"tenant {tenant_id} has no campaign (campaign_id={campaign_id})")
        try:
            return parse_campaign_yaml(row.config_yaml)
        except Exception as e:  # noqa: BLE001 - a broken config_yaml must fail clearly
            raise CampaignNotConfigured(
                f"campaign {row.id} for tenant {tenant_id} has invalid config_yaml: {e}"
            ) from e
