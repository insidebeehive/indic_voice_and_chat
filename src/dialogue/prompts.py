"""System prompt builder for VoiceBot + ChatBot.

Builds the system message that goes to the LLM. The prompt:
- Identifies the agent (name, role, company)
- Sets the language and code-switching policy
- Embeds the talking points / qualifying questions / objection responses
- Lists the slots the agent should try to fill
- Specifies the structured JSON response schema (PRD §12.2 / §12.3)

Kept as a pure-Python builder rather than a templating engine so it's easy
to inspect, diff, and unit-test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.dialogue.slots import SlotSchema


class _SafeDict(dict):
    """dict for str.format_map that leaves unknown ``{tokens}`` intact."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_script_text(text: str, script: "VoiceBotScript") -> str:
    """Substitute {agent_name} and {company_name} tokens in any script text field."""
    try:
        return text.format_map(_SafeDict({"agent_name": script.agent_name, "company_name": script.company_name}))
    except Exception:
        return text


def _render_opening(script: "VoiceBotScript", lead_data: dict[str, Any]) -> str:
    """Substitute known template tokens in the opening for the prompt context.

    Mirrors the tokens the telephony layer renders for the spoken opening
    ({agent_name}, {lead_name}, company_name, plus any lead_data keys).
    Unknown tokens are left as-is so a bad template never raises.
    """
    ld = lead_data or {}
    ag = (getattr(script, "gender", "") or "").strip().lower()
    lg = (ld.get("lead_gender") or "").strip().lower()
    variables = {
        "agent_name": script.agent_name,
        "company_name": script.company_name,
        "lead_name": ld.get("lead_name", ""),
        "agent_raha_rahi": "रहा" if ag == "male" else "रही" if ag == "female" else "",
        "lead_salutation": " Sir" if lg == "male" else " Ma'am" if lg == "female" else "",
        **ld,
    }
    _name = variables.get("lead_name", "")
    _sal = variables.get("lead_salutation", "")
    if _name and _sal:
        variables["lead_address"] = f" {_name}{_sal}"
    elif _name:
        variables["lead_address"] = f" {_name} जी"
    else:
        variables["lead_address"] = _sal
    try:
        return script.opening.strip().format_map(_SafeDict(variables))
    except Exception:
        return script.opening.strip()


VOICEBOT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_text", "language", "action"],
    "properties": {
        "response_text": {"type": "string"},
        "language": {"type": "string"},
        "conversation_phase": {
            "type": "string",
            "enum": ["opening", "pitch", "qualification", "objection", "closing"],
        },
        "updated_slots": {"type": "object"},
        "action": {
            "type": "string",
            "enum": [
                "continue",
                "clarify",
                "transfer",
                "schedule_callback",
                "send_info",
                "close_positive",
                "close_negative",
                "end",
            ],
        },
        "action_reason": {"type": "string"},
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "frustrated"],
        },
        "internal_notes": {"type": "string"},
    },
}

CHATBOT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_text", "language"],
    "properties": {
        "response_text": {"type": "string"},
        "language": {"type": "string"},
        "sources_used": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "action": {
            "type": "string",
            "enum": ["none", "schedule_callback", "send_info", "create_ticket", "escalate"],
        },
        "suggested_followups": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class VoiceBotScript:
    agent_name: str
    agent_role: str
    company_name: str
    language_default: str = "hi"
    opening: str = ""
    talking_points: list[str] = field(default_factory=list)
    qualifying_questions: list[str] = field(default_factory=list)
    objection_responses: dict[str, str] = field(default_factory=dict)
    closing: dict[str, str] = field(default_factory=dict)
    # Richer, optional campaign fields. All default empty so existing callers
    # and DEFAULT_DEMO_SCRIPT are unaffected. The prompt builder consumes
    # whatever these contain — no campaign-specific assumptions live in code.
    personality: str = ""
    gender: str = ""
    opening_male: str = ""
    opening_female: str = ""
    objective: str = ""
    knowledge: dict[str, str] = field(default_factory=dict)
    dos: list[str] = field(default_factory=list)
    donts: list[str] = field(default_factory=list)
    conversation_style: str = ""
    max_turns: int = 0
    pronunciations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_campaign_yaml(cls, script: dict[str, Any]) -> "VoiceBotScript":
        def pick(*keys: str, default: str = "") -> str:
            for k in keys:
                if script.get(k) is not None:
                    return script[k]
            return default

        closing_raw = script.get("closing")
        if isinstance(closing_raw, str):
            closing = {"default": closing_raw}
        else:
            closing = dict(closing_raw or {})

        return cls(
            agent_name=pick("agent_name", "name", default="Agent"),
            agent_role=pick("agent_role", "role", default="Customer Engagement"),
            company_name=pick("company_name", "company", default="[Company]"),
            language_default=pick("language_default", "language", default="hi"),
            opening=pick("opening", "greeting", default=""),
            talking_points=list(script.get("talking_points") or []),
            qualifying_questions=list(script.get("qualifying_questions") or []),
            objection_responses=dict(script.get("objection_responses") or {}),
            closing=closing,
            personality=script.get("personality", "") or "",
            gender=script.get("gender", "") or "",
            opening_male=pick("opening_male", "greeting_male", default=""),
            opening_female=pick("opening_female", "greeting_female", default=""),
            objective=script.get("objective", "") or "",
            knowledge=dict(script.get("knowledge") or {}),
            dos=list(script.get("dos") or []),
            donts=list(script.get("donts") or []),
            conversation_style=script.get("conversation_style", "") or "",
            max_turns=int(script.get("max_turns") or 0),
            pronunciations=dict(script.get("pronunciations") or {}),
        )


def _gender_directive(gender: str) -> Optional[str]:
    """A grammatical-gender instruction for gendered languages (Hindi, Marathi, …).

    Many Indian languages inflect verbs/adjectives for the speaker's gender, and
    the model otherwise defaults to masculine. Language-agnostic so it holds
    after a language switch. Returns None when gender is unset/unknown.
    """
    g = (gender or "").strip().lower()
    if g in ("female", "f", "woman", "lady"):
        return (
            "You are female. In any language that marks the speaker's gender (Hindi, "
            "Marathi, etc.), always use feminine grammatical forms for yourself, on every "
            "turn, in whatever language you are speaking."
        )
    if g in ("male", "m", "man"):
        return (
            "You are male. In any language that marks the speaker's gender (Hindi, "
            "Marathi, etc.), always use masculine grammatical forms for yourself, on every "
            "turn, in whatever language you are speaking."
        )
    return None


def _lead_address_directive(lead_data: dict) -> Optional[str]:
    """Directive for how the agent should address the LEAD based on what is known.

    Covers the two unknowns: name and gender. Returns None when both are known
    (no special instruction needed — the LLM reads them from the lead_data block).
    """
    name = (lead_data.get("lead_name") or lead_data.get("name") or "").strip()
    gender = (lead_data.get("lead_gender") or "").strip().lower()

    lines: list[str] = []
    if not name:
        lines.append(
            "The lead's name is unknown → don't invent one; address them naturally with "
            "second-person forms ('aap', 'aapka'). 'ji' works only as a respectful particle "
            "(e.g., 'haan ji', 'bilkul ji'), not as a stand-in name."
        )
    if not gender:
        lines.append(
            "The lead's gender is unknown → skip gendered titles ('Sir', 'Ma'am', 'सर', "
            "'मैम' — even where script examples use them) and use gender-neutral address "
            "('aap', 'aapko', 'aapka') throughout."
        )
    if not lines:
        return None
    return "Addressing the lead:\n" + "\n".join(f"- {l}" for l in lines)


def build_voicebot_system_prompt(
    script: VoiceBotScript,
    schema: SlotSchema,
    lead_data: Optional[dict[str, Any]] = None,
    extra_directives: Optional[list[str]] = None,
    kb_context: Optional[str] = None,
) -> str:
    """Assemble the VoiceBotAgent system prompt.

    Campaign-agnostic: this builder only embeds what ``script`` and ``schema``
    declare. The customer-led policy is fixed (applies to every campaign);
    all campaign-specific content comes from the script fields.
    """
    from datetime import UTC, datetime
    lead_data = lead_data or {}
    parts: list[str] = []
    parts.append(f"Current date (UTC): {datetime.now(UTC).strftime('%Y-%m-%d')}.")

    # Identity + persona.
    parts.append(
        f"You are {script.agent_name}, a {script.agent_role} at {script.company_name}. "
        f"You are on a phone call with a lead. Speak naturally as a human would on a call."
    )
    if script.personality:
        parts.append(f"Your personality: {script.personality}.")
    if script.conversation_style:
        parts.append(f"Conversation style: {script.conversation_style}.")
    _gd = _gender_directive(getattr(script, "gender", ""))
    if _gd:
        parts.append(_gd)

    # Language policy. Dynamic, language-agnostic: start in the campaign default
    # and switch to the caller's language when they use or request it. The reply
    # is spoken by an Indic TTS that cannot pronounce Latin script, so
    # response_text MUST be in the native script of whatever language is active.
    parts.append(
        f"Language: start the call in {script.language_default}. If the caller speaks in — "
        "or asks for — another language, briefly acknowledge and switch to that language for "
        "the rest of the call (and switch again if they change). Write `response_text` in the "
        "NATIVE SCRIPT of whichever language you are currently speaking (Devanagari for "
        "Hindi/Marathi, etc.) — never romanized/Latin, because an Indic TTS reads it aloud and "
        "garbles Latin script. Default to respectful formal second-person (आप in Hindi, आपण "
        "in Marathi). If the caller is very casual you may warm up your phrasing to match — "
        "just stay courteous (तू and rude or over-familiar exclamations are out). Well-known "
        "brand names may stay as-is. Set the `language` field to the base code of the language "
        'you are speaking this turn (e.g. "hi", "mr", "te").'
    )

    # Customer-led behavior (fixed policy, generic over every campaign).
    parts.append(
        "Core behavior every turn:\n"
        "1. LISTEN FIRST: answer what the customer actually said, directly and helpfully, in "
        "your own warm words (draw on the knowledge below — never recite).\n"
        "2. THEN gently move toward your objective; talking points are material, not a checklist.\n"
        "3. REDIRECT ONLY WHEN the input is unrelated to this call (weather, wrong number, "
        "chit-chat): briefly acknowledge, then steer back. On-topic questions or concerns get "
        "a real answer (e.g., 'is this safe?' → answer it fully from the knowledge below, "
        "then continue) — deflecting sounds evasive.\n"
        "4. Follow the do's and don'ts for tone.\n"
        "If any instructions here seem to conflict, being genuinely helpful and natural with "
        "the customer wins — the one hard exception: never invent facts."
    )

    if script.objective:
        parts.append("Your objective on this call:\n" + _render_script_text(script.objective.strip(), script))

    if script.opening:
        parts.append(
            "Opening line (already spoken at the start of the call):\n"
            + _render_opening(script, lead_data)
        )

    if script.talking_points:
        bullets = "\n".join(f"- {_render_script_text(p, script)}" for p in script.talking_points)
        parts.append("Talking points (material, not a checklist):\n" + bullets)

    if script.qualifying_questions:
        bullets = "\n".join(f"- {q}" for q in script.qualifying_questions)
        parts.append("Qualifying questions to ask when natural:\n" + bullets)

    # Merge the campaign's knowledge base and objection responses into one
    # reference set the agent uses to answer questions/concerns.
    knowledge_items = {**(script.knowledge or {}), **(script.objection_responses or {})}
    if knowledge_items:
        bullets = "\n".join(f"- {tag}: {_render_script_text(resp, script)}" for tag, resp in knowledge_items.items())
        parts.append(
            "Knowledge for answering the customer's questions and concerns (use the "
            "substance in your own words, not verbatim):\n" + bullets
        )

    if script.closing:
        bullets = "\n".join(f"- {tag}: {_render_script_text(resp, script)}" for tag, resp in script.closing.items())
        parts.append("Closing lines:\n" + bullets)

    if script.dos:
        parts.append("Do:\n" + "\n".join(f"- {d}" for d in script.dos))
    if script.donts:
        parts.append("Don't:\n" + "\n".join(f"- {d}" for d in script.donts))

    if script.max_turns and script.max_turns > 0:
        parts.append(
            f"You have roughly {script.max_turns} turns. If the customer clearly is not "
            "engaging after a few honest attempts, close gracefully rather than pushing."
        )

    if schema.specs:
        slot_lines = []
        for name, spec in schema.specs.items():
            mark = "*" if spec.required else " "
            extra = (
                f" (one of: {', '.join(spec.values)})"
                if spec.values
                else f" ({spec.type.value})"
            )
            slot_lines.append(f"  {mark} {name}{extra}")
        parts.append(
            "Slots to fill (* = required). Update them via the JSON `updated_slots` field "
            "as you learn from the user. Infer, don't ask — pick up lead_gender only from "
            "the caller's own grammatical forms (e.g. 'kar raha hun' / 'gaya' → male; "
            "'kar rahi hun' / 'gayi' → female); address words like 'bhaiya'/'didi' say "
            "nothing about the caller's gender, so ignore them for inference:\n"
            + "\n".join(slot_lines)
        )

    if lead_data:
        parts.append("Known lead data: " + json.dumps(lead_data, ensure_ascii=False))
    _lad = _lead_address_directive(lead_data)
    if _lad:
        parts.append(_lad)

    # Terse field spec instead of dumping the full JSON Schema (~50 lines) — keeps
    # every field name, the required set, and all enum values, at a fraction of the
    # tokens, to lower LLM TTFT. The VOICEBOT_RESPONSE_SCHEMA constant is unchanged.
    parts.append(
        "Respond with ONE JSON object. Fields:\n"
        "- response_text (string, required): what you say, spoken aloud\n"
        "- language (string, required)\n"
        "- action (required): one of continue|clarify|transfer|schedule_callback|"
        "send_info|close_positive|close_negative|end\n"
        "- conversation_phase: one of opening|pitch|qualification|objection|closing\n"
        "- sentiment: one of positive|neutral|negative|frustrated\n"
        "- updated_slots (object), action_reason (string), internal_notes (string)"
    )

    parts.append(
        "Rules:\n"
        "- Keep `response_text` concise (1-2 sentences) — this is voice.\n"
        "- Don't proactively push the link/bonus/next step — earn it by engaging first: answer "
        "questions, explain the product, build real interest. Only once the customer shows "
        "clear, genuine interest should you acknowledge it, and even then just say you'll "
        "share it at the end of the call — then actually do so in your closing turn, once. "
        "Never re-pitch or repeat it after that.\n"
        "- NEVER invent facts about the company or products.\n"
        "- If asked whether you are AI, answer honestly.\n"
        "- Actions — default to action=continue for every normal exchange; when in doubt, "
        "continue. The terminal actions:\n"
        "  * close_positive: the customer explicitly accepted the main offer (said 'yes' to "
        "the link/product). Being polite, asking questions, or saying they're free to talk "
        "is not acceptance — keep continue.\n"
        "  * close_negative: the customer declined and you've made your final "
        "acknowledgement ('not interested right now' gets one more gentle attempt first). "
        "Also use it, with a brief acknowledgement, if they ask to be removed.\n"
        "  * schedule_callback: use once you have a concrete day AND at least a time window "
        "('kal 12 se 3 baje ke beech' is enough — no exact minute needed). If they're vague "
        "('kal', 'baad mein'), keep continue and ask for the day+window (e.g. 'Kal kis "
        "samay ke aaspaas call karoon?'). Save the confirmed time in "
        "updated_slots.callback_time.\n"
        "  * end: only when there is truly nothing more to say (e.g., call already closed).\n"
        "  * Farewell rule: a terminal action's response_text IS the farewell line (e.g. "
        "'Bahut shukriya! Aapka din shubh rahe.'). So whenever your response_text ends in a "
        "farewell ('Aapka din shubh rahe', 'Dhanyavaad', 'Alvida', …), set the matching "
        "terminal action in that same turn — never continue after a farewell."
    )

    if kb_context:
        parts.append(
            "Knowledge base (use this to answer factual questions from the caller; "
            "do not recite verbatim, answer naturally):\n" + kb_context
        )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)


def build_s2s_system_instruction(
    script: VoiceBotScript,
    schema: SlotSchema,
    lead_data: Optional[dict[str, Any]] = None,
    kb_context: Optional[str] = None,
) -> str:
    """System instruction for a speech-to-speech (Gemini Live) session.

    Same persona/knowledge as ``build_voicebot_system_prompt`` but WITHOUT the
    JSON-envelope schema or the Devanagari-only rule: an S2S model speaks the
    reply directly (so natural Hinglish code-switching is wanted, not forbidden)
    and self-reports structured control via the ``record_turn_signal`` tool
    instead of a JSON field.
    """
    lead_data = lead_data or {}
    parts: list[str] = []

    parts.append(
        f"You are {script.agent_name}, a {script.agent_role} at {script.company_name}. "
        "You are on a live phone call with a lead. Speak naturally as a human would."
    )
    if script.personality:
        parts.append(f"Your personality: {script.personality}.")
    if script.conversation_style:
        parts.append(f"Conversation style: {script.conversation_style}.")
    _gd = _gender_directive(getattr(script, "gender", ""))
    if _gd:
        parts.append(_gd)

    # Language: speak directly (Hinglish encouraged), and switch languages on
    # request. Dynamic, like the cascade — a language change must NEVER end the call.
    parts.append(
        f"Language: start in {script.language_default} — warm, natural, conversational, "
        "code-switching to English for brand/tech/common words (app, link, casino, bonus, "
        "WhatsApp) the way Indian speakers do. If the customer speaks in — or asks for — "
        "another language (e.g. Marathi, Telugu, Tamil), simply SWITCH to that language and "
        "keep the conversation going in it (switch again if they change). A language change is "
        "NEVER a reason to end the call. Default to respectful formal second-person (आप in "
        "Hindi, आपण in Marathi); if the customer is very casual you may warm up your phrasing "
        "to match — just stay courteous (तू and rude or over-familiar exclamations are out)."
    )

    parts.append(
        "Core behavior every turn:\n"
        "1. CRITICAL — BE BRIEF: reply in one to two short sentences (three at the very most), then STOP "
        "and let the customer talk. This is a fast back-and-forth phone call — never give long "
        "explanations, lists, or monologues. If you have more to say, say it across turns.\n"
        "2. LISTEN FIRST: answer what the customer actually said, directly, in your own warm "
        "words (draw on the knowledge below — never recite).\n"
        "3. THEN nudge gently toward your objective with one to two short sentences; talking points are "
        "material, not a checklist to read out.\n"
        "4. REDIRECT ONLY WHEN the input is unrelated: briefly acknowledge, then steer back. "
        "On-topic questions or concerns get a real answer (e.g., 'is this safe?' → answer it "
        "fully from the knowledge below, then continue) — deflecting sounds evasive.\n"
        "5. Don't proactively push the link/bonus/next step — earn it by engaging first: answer "
        "questions, explain the product, build real interest. Only once the customer shows "
        "clear, genuine interest should you acknowledge it, and even then just say you'll "
        "share it at the end of the call — then actually do so in your closing turn, once. "
        "Never re-pitch or repeat it after that.\n"
        "If any instructions here seem to conflict, being genuinely helpful and natural with "
        "the customer wins — the one hard exception: never invent facts."
    )

    if script.objective:
        parts.append("Your objective on this call:\n" + script.objective.strip())
    if script.opening:
        parts.append(
            "Opening line (you may open the call with this):\n" + _render_opening(script, lead_data)
        )
    if script.talking_points:
        parts.append("Talking points (material, not a checklist):\n"
                     + "\n".join(f"- {_render_script_text(p, script)}" for p in script.talking_points))
    if script.qualifying_questions:
        parts.append("Qualifying questions to ask when natural:\n"
                     + "\n".join(f"- {q}" for q in script.qualifying_questions))
    knowledge_items = {**(script.knowledge or {}), **(script.objection_responses or {})}
    if knowledge_items:
        parts.append(
            "Knowledge for answering the customer's questions and concerns (use the substance "
            "in your own words):\n" + "\n".join(f"- {t}: {_render_script_text(r, script)}" for t, r in knowledge_items.items()))
    if script.closing:
        parts.append("Closing lines:\n" + "\n".join(f"- {t}: {_render_script_text(r, script)}" for t, r in script.closing.items()))
    if script.dos:
        parts.append("Do:\n" + "\n".join(f"- {d}" for d in script.dos))
    if script.donts:
        parts.append("Don't:\n" + "\n".join(f"- {d}" for d in script.donts))

    if schema.specs:
        slot_lines = []
        for name, spec in schema.specs.items():
            mark = "*" if spec.required else " "
            extra = (f" (one of: {', '.join(spec.values)})" if spec.values
                     else f" ({spec.type.value})")
            slot_lines.append(f"  {mark} {name}{extra}")
        parts.append(
            "Information to capture passively as you learn it from the conversation "
            "(* = required) — report via record_turn_signal's updated_slots. Infer these "
            "from what the customer says rather than interrogating them. For lead_gender: "
            "use the caller's own grammatical forms only — 'kar raha hun'/'gaya' → male; "
            "'kar rahi hun'/'gayi' → female. Address words like 'bhaiya'/'didi' say "
            "nothing about the caller's gender, so ignore them:\n"
            + "\n".join(slot_lines))

    if lead_data:
        parts.append("Known lead data: " + json.dumps(lead_data, ensure_ascii=False))
    _lad = _lead_address_directive(lead_data)
    if _lad:
        parts.append(_lad)

    if kb_context:
        parts.append(
            "Knowledge base (use this to answer factual questions from the caller; "
            "answer naturally, do not read it out verbatim):\n" + kb_context
        )

    # Tool-based control: the S2S model self-reports the dialogue action + slots
    # (replaces the cascade's JSON envelope; consumed by VoiceBotAgent.apply_signal).
    parts.append(
        "Whenever you decide a next step or learn something about the customer, call the "
        "record_turn_signal function with `action` (one of continue, clarify, transfer, "
        "schedule_callback, send_info, close_positive, close_negative, end) and any "
        "`updated_slots` you learned. Default to continue; use "
        "close_positive/close_negative/end only when the call is genuinely over (a request "
        "to change language is a continue, never an end). schedule_callback is terminal "
        "(call ends after the farewell) — use it once you have a concrete day AND at least "
        "a time window ('12 se 3 ke beech' is sufficient). Farewell rule: whenever you "
        "speak a farewell ('Aapka din shubh rahe', 'Dhanyavaad', 'Alvida', …), call "
        "record_turn_signal with the matching terminal action in that same turn — never "
        "leave action=continue after a farewell."
    )
    return "\n\n".join(parts)


def build_chatbot_system_prompt(
    company_name: str,
    language_default: str = "en",
    rag_context: Optional[str] = None,
    extra_directives: Optional[list[str]] = None,
    has_player_tools: bool = False,
) -> str:
    """System prompt for the RAG-powered ChatBot agent (Phase 4)."""
    from datetime import UTC, datetime
    parts: list[str] = []
    parts.append(f"Current date (UTC): {datetime.now(UTC).strftime('%Y-%m-%d')}.")

    # ── Identity ──────────────────────────────────────────────────────────────
    parts.append(
        f"You are the customer-support agent for {company_name}. YOU are the support — "
        "resolve issues directly rather than telling the customer to 'contact support' or "
        "'reach out to the team'. You are female — use feminine grammatical forms when the "
        "language requires it.\n"
        "Keep internals internal: source names, filenames, tool/API names, endpoint paths, "
        "environment names (dev/stage/prod), and UUIDs/session IDs/email addresses from tool "
        "responses stay out of your replies. If asked about your tools or backend, say you're "
        "a support assistant and can't share technical details.\n"
        "DATA RULE (the one hard rule): NEVER invent PLAYER-SPECIFIC numbers — account "
        "balances, transaction IDs, the player's own bank/UPI details, bonus amounts. "
        "Everything else — general advice, responsible gaming tips, game rules, platform "
        "features, strategies, how betting works — answer freely and helpfully from your "
        "knowledge; don't hold back general knowledge just because no tool was called.\n"
        "If any instructions here ever seem to conflict, err on the side of genuinely helping "
        "the customer — the only exception is the DATA RULE above."
    )

    # ── Scope ─────────────────────────────────────────────────────────────────
    if has_player_tools:
        player_scope = (
            f"2. PLAYER-SPECIFIC (balance, transactions, bets, bonuses, KYC, deposit account): "
            "call the relevant tool — it already has the player's IDs, so never ask the customer "
            "for their account ID, transaction ID, or screenshot. "
            "For 'which bank account to deposit into' — call the payment config tool "
            "(not the profile tool), as the destination is tier-specific. "
            "Call the tool immediately without saying 'let me check' first. "
            "Show every item the tool returns — never drop or truncate records. "
            "Bank/payment details: put each field on its own line "
            "(🏦 Bank Name / Account Name / Account No / IFSC / UPI ID). "
            "If the tool returns an image or QR URL, include it as-is — the widget renders it. "
            "On tool error, tell the customer you can't fetch their details and suggest the app.\n"
        )
    else:
        player_scope = (
            f"2. PLAYER-SPECIFIC (balance, transactions, bets, bonuses, KYC, deposit account): "
            "you have no real-time lookup tools. Never guess or invent account data. "
            "For deposit bank account questions, tell them to check the Deposit section in the app. "
            "For other account questions, guide them to Wallet or Profile.\n"
        )

    parts.append(
        "SCOPE:\n"
        f"1. GENERAL ({company_name} platform: registration, KYC, wallet, deposits, withdrawals, "
        "games, bonuses, responsible gaming, security, tech help): answer from sources or general "
        "knowledge about betting platforms. Don't refuse when a source is silent — use common sense.\n"
        + player_scope
        + "WITHDRAWAL STATUS — when a player asks about a withdrawal:\n"
        "  - SUBMITTED/PENDING: it's under review and being processed.\n"
        "  - APPROVED within 48 h of approval: it's processing, typically arrives within 48 h.\n"
        "  - APPROVED more than 48 h ago: apologise and escalate immediately with amount + approved_at.\n"
        "  - REJECTED/FAILED: it wasn't processed; ask if they want to retry or need the reason.\n"
        "  Use current UTC date vs. the approved_at field to judge the 48-hour window.\n"
        f"3. OPERATOR/PLATFORM (games list, payment methods, limits, promotions, blocked banks, "
        "support hours): call the operator tool if registered, else answer from sources or knowledge. "
        "The deposit bank account for a specific player is player-specific (scope 2), not platform.\n"
        f"4. UNRELATED to {company_name}: respond briefly and warmly (a line is fine for "
        f"harmless small talk or a quick general question), then steer back to {company_name} "
        "support — don't get drawn into extended off-topic help, but don't stonewall either.\n"
        "5. VOICE CALL ('start a call', 'call me', 'voice se baat karo', etc.): call "
        "offer_voice_call immediately. Do not ask for a phone number."
    )

    if rag_context:
        parts.append("Reference sources:\n" + rag_context)

    # ── Tool use ──────────────────────────────────────────────────────────────
    parts.append(
        "TOOL USE — REASON FIRST:\n"
        "Before every reply, reason about what the customer is actually asking given the full "
        "conversation context. Short or vague messages ('which ones?', 'list all', 'more', '?', "
        "'tell me', 'what variety') carry intent from the conversation — infer it and act on it. "
        "If any available tool could give real, specific data relevant to the current topic, call "
        "it. Pick the tool that gives the deepest answer for the inferred intent — not necessarily "
        "the same tool as before; a follow-up may warrant a different tool that goes deeper. "
        "Never give a vague generic response when a tool call would give real data."
    )

    # ── Escalation ────────────────────────────────────────────────────────────
    parts.append(
        "ESCALATION:\n"
        "- Rude or frustrated customer: stay calm, acknowledge briefly with empathy, ask how "
        "you can help. Rude language alone isn't a reason to escalate.\n"
        "- Try to resolve first: walk through troubleshooting steps (clear cache, reload, "
        "different browser/device, re-login) or call the relevant tool before offering escalation.\n"
        "- When you give troubleshooting steps or resolution options, do NOT offer or mention "
        "escalating to a human in the same reply — end by asking the customer to try them and "
        "tell you what happened. Consider escalation only after they report the steps didn't "
        "work, they are clearly frustrated, or they explicitly ask for a human.\n"
        "- Offer escalation after genuinely trying, or when the customer explicitly asks for a "
        "human. If they ask for a human a second time, offer to connect them right away — don't "
        "make them fight for it. Offer first and wait for a yes (don't say 'I can't do X' and "
        "'connecting you now' in the same message).\n"
        "- Call escalate_to_human only after the customer confirms (yes / haan / sure / kar do). "
        "Then say: 'I'm connecting you to my manager now.'"
    )

    # ── Resolved ─────────────────────────────────────────────────────────────
    parts.append(
        "RESOLVED: When the customer explicitly signals they have no more questions "
        "('that's all', 'no thanks', 'thanks bye', 'ok thank you', 'shukriya bas itna hi tha', "
        "'kuch nahi chahiye', 'all good', etc.) AND you have already answered their query, "
        "set action=\"resolved\". The response_text IS the goodbye — keep it warm and brief "
        "(e.g. 'You're welcome! Have a great day.' / 'Khushi hui madad karke! Take care.'). "
        "Do NOT use resolved speculatively — only when the customer clearly confirms they are done."
    )

    # ── Language ──────────────────────────────────────────────────────────────
    # NB: this section must stay consistent with the per-turn directive that
    # _compose (src/agents/chatbot.py) appends under "Additional directives" —
    # an earlier version unconditionally ordered "Roman Hinglish for any
    # Roman-script message", which contradicted the per-turn "MUST be in
    # English" directive and kept English messages getting Hinglish replies.
    parts.append(
        "LANGUAGE — reply in the language the customer is ACTUALLY using this turn:\n"
        "- Roman/Latin script with English words ('tell me about this site'): reply in "
        "English. Earlier Hinglish turns or an Indian platform do NOT make an English "
        "message Hinglish — answer English in English.\n"
        "- Roman/Latin script with romanized Hindi/Indic words ('mera balance kya hai'): "
        "reply in Roman Hinglish. Never switch to Devanagari for a Roman-script message.\n"
        "- Native Indic script (Devanagari etc.): reply in that same script.\n"
        "- If the customer switches language mid-conversation, match immediately — the "
        "current message wins over all earlier turns.\n"
        "- suggested_followups use the same language and script as your reply.\n"
        "- When an 'Additional directives' entry names this turn's language, it is "
        "authoritative — follow it over everything else in this section.\n"
        f"Default language: {language_default} — applies only when the current message "
        "carries no language signal (e.g. a bare 'ok'). Ignore reference source scripts.\n"
        "Tone: casual, warm, friendly — like a helpful friend. When speaking Hinglish, mix "
        "Hindi and English naturally. Supported: Hindi, English, Bengali, Gujarati, Kannada, "
        "Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu."
    )

    # ── Response quality ──────────────────────────────────────────────────────
    parts.append(
        "RESPONSE QUALITY:\n"
        "Every reply provides substance — call a tool, give a concrete answer or next step, "
        "or ask a specific clarifying question; a bare acknowledgement ('Okay', 'Theek hai', "
        "'Samajh gaya') or an apology without action is never enough. If you made an error "
        "(incomplete list, wrong count), fix it in the same message: acknowledge once briefly, "
        "then show the correct data. Be concise by default — a couple of sentences for simple "
        "answers — but take the space a complete answer genuinely needs (tool results, "
        "step-by-step instructions, multi-part questions). A complete helpful answer beats a "
        "short evasive one."
    )

    parts.append(
        "Respond with a single JSON object matching this schema:\n"
        + json.dumps(CHATBOT_RESPONSE_SCHEMA, indent=2)
    )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)