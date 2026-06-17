"""Per-tenant, DB-backed campaign resolution.

The live call resolves which campaign (agent script + slots) to run from the DB
``campaigns`` table — per-tenant, and per-call by ``campaign_id`` — instead of the
single global ``VOX_CAMPAIGN`` file baked into the bridge factories at startup.

Resolution order (per call):
  1. an explicit ``campaign_id`` (guarded so it belongs to the tenant), else
  2. the tenant's single active campaign, else
  3. a YAML fallback (the global ``load_campaign`` default) so the app always has
     a script — e.g. a tenant with no campaign rows yet, or an unparseable config.

Resolution happens once per call setup (not per turn), so it queries the DB each
time rather than maintaining a cache to invalidate — correctness over a micro-opt.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.dialogue.campaign_loader import (
    LoadedCampaign,
    active_campaign_slug,
    load_campaign,
    parse_campaign_yaml,
)
from src.models.campaign import Campaign

log = logging.getLogger(__name__)


class DbCampaignResolver:
    """Resolve a tenant's campaign from the DB, with a YAML fallback.

    ``fallback`` is the script used when a tenant has no usable DB campaign; if
    omitted it lazily loads the global ``VOX_CAMPAIGN`` file (so behaviour matches
    today for un-migrated tenants).
    """

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        fallback: Optional[LoadedCampaign] = None,
    ) -> None:
        self._sm = sessionmaker
        self._fallback = fallback

    def _fallback_campaign(self) -> LoadedCampaign:
        if self._fallback is None:
            self._fallback = load_campaign(active_campaign_slug())
        return self._fallback

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
        """Return the script + slots for this call. Never raises — any DB or parse
        failure falls back to the YAML default so a call always has a script."""
        try:
            async with self._sm() as session:
                row = await self._row(session, tenant_id, campaign_id)
            if row is not None:
                return parse_campaign_yaml(row.config_yaml)
        except Exception:  # noqa: BLE001 - resolution must never break a call
            log.exception("campaign resolution failed; using fallback",
                          extra={"tenant_id": tenant_id, "campaign_id": campaign_id})
        return self._fallback_campaign()
