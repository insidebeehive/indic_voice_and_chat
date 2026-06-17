"""Campaign-upfront script loading.

Reads one campaign YAML (selected by VOX_CAMPAIGN) into a script + slot schema
at startup. The active campaign drives every call this process handles. The
loader is campaign-agnostic — it only parses whatever the YAML declares.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema

log = logging.getLogger(__name__)

DEFAULT_CAMPAIGN_SLUG = "bharat_matka"
DEFAULT_CAMPAIGNS_DIR = Path("config/campaigns")


@dataclass
class LoadedCampaign:
    script: VoiceBotScript
    slots: SlotSchema


def active_campaign_slug() -> str:
    """The campaign slug to load at startup (env VOX_CAMPAIGN, default bharat_matka)."""
    return os.environ.get("VOX_CAMPAIGN", DEFAULT_CAMPAIGN_SLUG)


def parse_campaign_data(data: dict) -> LoadedCampaign:
    """Parse a campaign dict (with or without the top-level ``campaign:`` wrapper)
    into a script + slot schema. Shared by the YAML file loader and the DB-backed
    per-tenant resolver, so both interpret a campaign identically."""
    camp = data.get("campaign", data)  # tolerate with/without the wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    return LoadedCampaign(
        VoiceBotScript.from_campaign_yaml(merged),
        SlotSchema.from_campaign_yaml(camp.get("slots") or {}),
    )


def parse_campaign_yaml(text: str) -> LoadedCampaign:
    """Parse a campaign YAML string (e.g. a DB ``campaigns.config_yaml``)."""
    return parse_campaign_data(yaml.safe_load(text) or {})


def load_campaign(
    slug: str, campaigns_dir: Path = DEFAULT_CAMPAIGNS_DIR
) -> LoadedCampaign:
    """Load ``config/campaigns/<slug>.yaml`` into a script + slot schema.

    Missing/unreadable file -> warn and fall back to the demo script with an
    empty slot schema, so the app still boots.
    """
    path = campaigns_dir / f"{slug}.yaml"
    if not path.exists():
        from src.bootstrap import DEFAULT_DEMO_SCRIPT  # lazy: avoid import cost/cycle

        log.warning("campaign file not found: %s; using demo script", path)
        return LoadedCampaign(DEFAULT_DEMO_SCRIPT, SlotSchema())

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return parse_campaign_data(data)
