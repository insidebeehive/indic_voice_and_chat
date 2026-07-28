"""Campaign parsing.

A campaign's config is stored as a YAML string (a tenant's
``campaigns.config_yaml`` DB column); these functions turn that string into a
script + slot schema. Shared by every campaign consumer so they all interpret
a campaign identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema


@dataclass
class LoadedCampaign:
    script: VoiceBotScript
    slots: SlotSchema


def parse_campaign_data(data: dict) -> LoadedCampaign:
    """Parse a campaign dict (with or without the top-level ``campaign:`` wrapper)
    into a script + slot schema."""
    camp = data.get("campaign", data)  # tolerate with/without the wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    return LoadedCampaign(
        VoiceBotScript.from_campaign_yaml(merged),
        SlotSchema.from_campaign_yaml(camp.get("slots") or {}),
    )


def parse_campaign_yaml(text: str) -> LoadedCampaign:
    """Parse a campaign YAML string (e.g. a DB ``campaigns.config_yaml``)."""
    return parse_campaign_data(yaml.safe_load(text) or {})
