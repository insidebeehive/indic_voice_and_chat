"""Campaign parsing.

A campaign's config is stored as a YAML string (a tenant's
``campaigns.config_yaml`` DB column); these functions turn that string into a
script + slot schema. Shared by every campaign consumer so they all interpret
a campaign identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


@dataclass
class LoadedCampaign:
    script: VoiceBotScript
    slots: SlotSchema


def parse_campaign_data(data: dict) -> LoadedCampaign:
    """Parse a campaign dict (with or without the top-level ``campaign:`` wrapper)
    into a script + slot schema."""
    camp = data.get("campaign", data)  # tolerate with/without the wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    script = VoiceBotScript.from_campaign_yaml(merged)
    slots = SlotSchema.from_campaign_yaml(camp.get("slots") or {})
    # Runs once per campaign resolution (campaign_resolver.py), not per turn --
    # full sizes rather than a guard. Answers "what did this campaign's YAML
    # actually parse into" without a DB read + YAML read by eye.
    debug_event(
        log, "campaign parse_data completed",
        wrapped=("campaign" in data), agent_name=script.agent_name,
        company_name=script.company_name, language_default=script.language_default,
        slot_count=len(slots.specs), required_slot_count=len(slots.required_names()),
        talking_points_count=len(script.talking_points),
        knowledge_count=len(script.knowledge),
        objection_response_count=len(script.objection_responses),
    )
    return LoadedCampaign(script, slots)


def parse_campaign_yaml(text: str) -> LoadedCampaign:
    """Parse a campaign YAML string (e.g. a DB ``campaigns.config_yaml``)."""
    return parse_campaign_data(yaml.safe_load(text) or {})
