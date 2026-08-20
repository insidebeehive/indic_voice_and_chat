from __future__ import annotations

import json
import re

import yaml

from src.dialogue.prompts import (
    CHATBOT_RESPONSE_SCHEMA,
    VOICEBOT_RESPONSE_SCHEMA,
    VoiceBotScript,
    build_chatbot_system_prompt,
    build_voicebot_system_prompt,
)
from src.dialogue.slots import SlotSchema


SCRIPT = {
    "agent_name": "Priya",
    "agent_role": "Customer Engagement Specialist",
    "company_name": "Acme Telecom",
    "language_default": "hi",
    "opening": "Namaste {lead_name} ji, main Priya bol rahi hoon.",
    "talking_points": ["Plan B has 500GB data", "Limited offer"],
    "qualifying_questions": ["Aap kaunsa plan use kar rahe hain?"],
    "objection_responses": {
        "busy": "Bilkul, kal call kar sakti hoon?",
        "is_ai": "Main ek AI hoon.",
    },
    "closing": {"positive": "Dhanyavaad!", "negative": "Koi baat nahi."},
}

SLOT_YAML = """
lead_name: { type: string, required: true }
interest_level: { type: enum, required: true, values: [hot, warm, cold] }
"""


def test_voicebot_prompt_includes_all_sections() -> None:
    script = VoiceBotScript.from_campaign_yaml(SCRIPT)
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    prompt = build_voicebot_system_prompt(script, schema, lead_data={"lead_name": "Manoj"})

    assert "Priya" in prompt
    assert "Acme Telecom" in prompt
    assert "Plan B has 500GB data" in prompt
    assert "Aap kaunsa plan use kar rahe hain?" in prompt
    assert "is_ai" in prompt
    assert "Manoj" in prompt
    assert "* lead_name" in prompt
    assert "* interest_level" in prompt
    # Terse field spec (not the full JSON Schema dump) — field names + enums present.
    assert "response_text" in prompt
    assert "updated_slots" in prompt
    # All enum values must survive the trim.
    for a in ("continue", "clarify", "transfer", "schedule_callback", "send_info",
              "close_positive", "close_negative", "end"):
        assert a in prompt
    for p in ("opening", "pitch", "qualification", "objection", "closing"):
        assert p in prompt
    for s in ("positive", "neutral", "negative", "frustrated"):
        assert s in prompt


def test_chatbot_language_section_is_word_language_aware() -> None:
    # Regression guard for the persistent English→Hinglish bug: the base
    # LANGUAGE section used to order "Roman Hinglish for ANY Roman-script
    # message" ("not the meaning of their words"), directly contradicting the
    # per-turn "MUST be in English" directive appended under Additional
    # directives. The section must distinguish English words from romanized
    # Hindi, and must declare the per-turn directive authoritative.
    prompt = build_chatbot_system_prompt(company_name="Acme", language_default="hi")
    assert "answer English in English" in prompt
    assert "not the meaning of their words" not in prompt  # the old blanket rule
    assert "authoritative" in prompt  # per-turn directive precedence declared
    # The per-turn directive block lands AFTER the LANGUAGE section (recency).
    prompt_with_directive = build_chatbot_system_prompt(
        company_name="Acme", language_default="hi",
        extra_directives=["The user's current message is in English."])
    assert prompt_with_directive.rindex("Additional directives") > \
        prompt_with_directive.rindex("LANGUAGE —")


def test_voicebot_prompt_defers_link_offer_to_call_end() -> None:
    # The link/bonus must never be pitched proactively — only acknowledged
    # once genuine interest is shown, and deferred to the closing turn rather
    # than pushed mid-call.
    script = VoiceBotScript.from_campaign_yaml(SCRIPT)
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    prompt = build_voicebot_system_prompt(script, schema)
    assert "Don't proactively push the link/bonus/next step" in prompt
    assert "share it at the end of the call" in prompt


def test_voicebot_prompt_mentions_required_slots_with_marker() -> None:
    script = VoiceBotScript.from_campaign_yaml(SCRIPT)
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    prompt = build_voicebot_system_prompt(script, schema)
    # required slots get *
    assert "* lead_name" in prompt
    assert "* interest_level" in prompt


def test_voicebot_prompt_extra_directives_appended() -> None:
    script = VoiceBotScript.from_campaign_yaml(SCRIPT)
    schema = SlotSchema()
    prompt = build_voicebot_system_prompt(
        script, schema, extra_directives=["Be polite", "Use formal Hindi"]
    )
    assert "Be polite" in prompt
    assert "Use formal Hindi" in prompt


def test_chatbot_prompt_with_rag_context() -> None:
    prompt = build_chatbot_system_prompt(
        company_name="Acme",
        language_default="en",
        rag_context="Doc 1: Plan A has 100GB.\nDoc 2: Plan B has 500GB.",
    )
    assert "Acme" in prompt
    assert "Plan B has 500GB" in prompt
    assert '"sources_used"' in prompt


def test_chatbot_prompt_has_scope_guardrails() -> None:
    prompt = build_chatbot_system_prompt(company_name="Acme")
    assert "SCOPE" in prompt
    # player- and operator-specific are named separately.
    assert "player-specific" in prompt or "PLAYER-SPECIFIC" in prompt
    assert "OPERATOR/PLATFORM" in prompt
    # agent IS the support — should never defer to an external team.
    assert "YOU are the support" in prompt
    # off-topic handling is still SCOPE item 4 (responds warmly, then redirects);
    # DEPTH-MATCHING is a separate, added section reinforcing not overengaging on it.
    assert "DEPTH-MATCHING" in prompt
    assert "standalone deliverable outside your job" in prompt


def test_chatbot_prompt_operator_scope_branches_on_tool_registration() -> None:
    with_tools = build_chatbot_system_prompt(company_name="Acme", has_operator_tools=True)
    without_tools = build_chatbot_system_prompt(company_name="Acme", has_operator_tools=False)
    assert "Call the operator tool" in with_tools
    assert "get_operator_games_config" in with_tools
    assert "no real-time operator lookup tools" in without_tools
    assert "only concept-level answers are licensed" in without_tools
    # The no-tool branch must still forbid guessing specifics.
    assert "never name or describe a specific game" in without_tools


def test_chatbot_prompt_includes_local_time() -> None:
    prompt = build_chatbot_system_prompt(company_name="Acme", tenant_timezone="Asia/Kolkata")
    assert "Asia/Kolkata" in prompt
    assert re.search(r"\b\d{2}:\d{2}\b", prompt) is not None


def test_chatbot_prompt_falls_back_to_utc_on_invalid_timezone() -> None:
    prompt = build_chatbot_system_prompt(company_name="Acme", tenant_timezone="Not/ARealZone")
    assert "Current local time (UTC)" in prompt


def test_chatbot_prompt_has_identity_confirmation_rule() -> None:
    # Follow-up to the UUID-redaction fix: the bot must not act as a yes/no
    # oracle confirming/denying customer-guessed profile fields (mobile,
    # email, name, DOB) against real on-file data.
    prompt = build_chatbot_system_prompt(company_name="Acme")
    assert "IDENTITY CONFIRMATION" in prompt
    assert "never confirm or deny" in prompt


def test_chatbot_prompt_hinglish_script_matching() -> None:
    prompt = build_chatbot_system_prompt(company_name="Acme", language_default="hi")
    # Must instruct the model to match Roman script for Hinglish input.
    assert "ROMAN" in prompt or "roman" in prompt.lower()
    # Must give a concrete Hinglish example so the model knows what to detect.
    assert "mera balance" in prompt or "Hinglish" in prompt
    # Must NOT collapse Devanagari and Roman into a single "write in Hindi" rule.
    assert "Devanagari" in prompt


def test_response_schemas_are_valid_json() -> None:
    # Smoke test — assert they're JSON-serializable
    json.dumps(VOICEBOT_RESPONSE_SCHEMA)
    json.dumps(CHATBOT_RESPONSE_SCHEMA)


def test_from_campaign_yaml_parses_new_fields_and_aliases() -> None:
    s = VoiceBotScript.from_campaign_yaml({
        "name": "Anaaya", "company": "Bharat Matka", "role": "Sales",
        "language": "hi", "greeting": "Namaste",
        "objective": "Push link", "knowledge": {"safety": "It is safe"},
        "dos": ["Be warm"], "donts": ["No jargon"],
        "personality": "warm", "gender": "female",
        "conversation_style": "Hinglish", "max_turns": 12,
        "closing": "Dhanyavaad!",   # a string, not a dict
    })
    assert s.agent_name == "Anaaya"
    assert s.company_name == "Bharat Matka"
    assert s.agent_role == "Sales"
    assert s.language_default == "hi"
    assert s.opening == "Namaste"
    assert s.objective == "Push link"
    assert s.knowledge == {"safety": "It is safe"}
    assert s.dos == ["Be warm"] and s.donts == ["No jargon"]
    assert s.personality == "warm" and s.gender == "female"
    assert s.conversation_style == "Hinglish" and s.max_turns == 12
    assert s.closing == {"default": "Dhanyavaad!"}   # string normalized to dict


def test_from_campaign_yaml_backcompat_existing_keys() -> None:
    s = VoiceBotScript.from_campaign_yaml({
        "agent_name": "P", "agent_role": "R", "company_name": "C",
        "closing": {"positive": "ok"},
    })
    assert s.agent_name == "P" and s.closing == {"positive": "ok"}
    assert s.knowledge == {} and s.max_turns == 0 and s.dos == []


def test_voicebot_prompt_is_generic_over_script() -> None:
    """The builder must embed whatever the script declares — no hardcoded
    campaign content. Uses sentinel strings (not Bharat Matka)."""
    script = VoiceBotScript.from_campaign_yaml({
        "agent_name": "Zeta", "agent_role": "Helper", "company_name": "Foo Inc",
        "objective": "SENTINEL_OBJECTIVE_X",
        "knowledge": {"q1": "SENTINEL_KNOWLEDGE_Y"},
        "dos": ["SENTINEL_DO_Z"],
        "donts": ["SENTINEL_DONT_W"],
        "personality": "SENTINEL_PERSONA",
        "max_turns": 7,
    })
    prompt = build_voicebot_system_prompt(script, SlotSchema())
    for sentinel in ("SENTINEL_OBJECTIVE_X", "SENTINEL_KNOWLEDGE_Y",
                     "SENTINEL_DO_Z", "SENTINEL_DONT_W", "SENTINEL_PERSONA"):
        assert sentinel in prompt
    # Fixed customer-led policy text is present regardless of campaign.
    assert "LISTEN FIRST" in prompt
    assert "REDIRECT ONLY WHEN" in prompt
    # Soft turn budget surfaced from the script's max_turns.
    assert "7 turns" in prompt


def test_voicebot_prompt_renders_opening_tokens() -> None:
    script = VoiceBotScript.from_campaign_yaml({
        "agent_name": "Anaaya", "agent_role": "Sales", "company_name": "BM",
        "greeting": "Hi {lead_name}, main {agent_name} bol rahi hoon. {unknown_token}",
    })
    prompt = build_voicebot_system_prompt(script, SlotSchema(), lead_data={"lead_name": "Raju"})
    assert "Hi Raju, main Anaaya bol rahi hoon." in prompt
    assert "{agent_name}" not in prompt
    assert "{lead_name}" not in prompt
    # Unknown tokens are left intact rather than crashing.
    assert "{unknown_token}" in prompt


def test_s2s_system_instruction_has_persona_tool_no_envelope() -> None:
    from src.dialogue.prompts import build_s2s_system_instruction
    script = VoiceBotScript.from_campaign_yaml(SCRIPT)
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    instr = build_s2s_system_instruction(script, schema)
    assert "Priya" in instr and "Acme Telecom" in instr
    assert "Plan B has 500GB data" in instr          # talking points / knowledge present
    assert "record_turn_signal" in instr             # tool-based control
    assert "BE BRIEF" in instr                       # verbosity guard
    assert "code-switch" in instr                    # Hinglish encouraged
    # cascade-only artifacts must NOT leak into the S2S instruction
    assert "JSON object" not in instr
    assert "Devanagari" not in instr


def test_s2s_instruction_defers_link_offer_to_call_end() -> None:
    # The link/bonus must never be pitched proactively on the S2S path either —
    # only acknowledged once genuine interest is shown, deferred to the closing
    # turn, matching the cascade path's rule.
    from src.dialogue.prompts import build_s2s_system_instruction
    script = VoiceBotScript.from_campaign_yaml(SCRIPT)
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    instr = build_s2s_system_instruction(script, schema)
    assert "Don't proactively push the link/bonus/next step" in instr
    assert "share it at the end of the call" in instr


def test_gender_directive_enforces_feminine_in_both_prompts() -> None:
    # Language-agnostic now: holds in whatever language the agent is speaking,
    # not just Hindi (so it survives a switch to Marathi etc.).
    from src.dialogue.prompts import build_s2s_system_instruction
    script = VoiceBotScript.from_campaign_yaml({**SCRIPT, "gender": "female"})
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    for instr in (build_s2s_system_instruction(script, schema),
                  build_voicebot_system_prompt(script, schema)):
        assert "female" in instr
        assert "feminine grammatical forms" in instr


def test_no_gender_directive_when_unset() -> None:
    from src.dialogue.prompts import build_s2s_system_instruction
    script = VoiceBotScript.from_campaign_yaml(SCRIPT)   # no gender
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    assert "FEMALE" not in build_s2s_system_instruction(script, schema)


def test_from_campaign_yaml_parses_pronunciations():
    script = VoiceBotScript.from_campaign_yaml({
        "agent_name": "Priya",
        "company_name": "XYZ",
        "pronunciations": {"Anaaya": "अनाया", "XYZ": "एक्स वाय ज़ेड"},
    })
    assert script.pronunciations == {"Anaaya": "अनाया", "XYZ": "एक्स वाय ज़ेड"}


def test_from_campaign_yaml_pronunciations_defaults_empty():
    script = VoiceBotScript.from_campaign_yaml({"agent_name": "Priya", "company_name": "XYZ"})
    assert script.pronunciations == {}
