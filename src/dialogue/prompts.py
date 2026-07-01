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
            "You are FEMALE. In any language that marks the speaker's gender (Hindi, "
            "Marathi, etc.), ALWAYS use feminine grammatical forms for yourself and NEVER "
            "masculine ones — keep this consistent on every single turn, in whatever "
            "language you are speaking."
        )
    if g in ("male", "m", "man"):
        return (
            "You are MALE. In any language that marks the speaker's gender (Hindi, "
            "Marathi, etc.), ALWAYS use masculine grammatical forms for yourself and never "
            "feminine ones — keep this consistent every turn, in whatever language you are "
            "speaking."
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
            "The lead's name is unknown — never invent or guess one, and do NOT call them "
            "'ji' as if it were their name. Use second-person forms ('aap', 'aapka') naturally. "
            "'ji' is only a respectful particle (e.g., 'haan ji', 'bilkul ji') — never a name."
        )
    if not gender:
        lines.append(
            "The lead's gender is UNKNOWN. NEVER use 'Sir', 'Ma'am', 'सर', 'मैम', or any "
            "gendered title when addressing them — even if the script examples use 'Sir'. "
            "Use only gender-neutral address ('aap', 'aapko', 'aapka') throughout."
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
    lead_data = lead_data or {}
    parts: list[str] = []

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
        "garbles Latin script. Always address the caller respectfully using formal second-person "
        "(आप in Hindi, आपण in Marathi) — never तू/तुम or casual exclamations like अरे; the "
        "caller may be casual but you must stay formal and courteous. Well-known brand names may "
        "stay as-is. Set the `language` field to the base code of the language you are speaking "
        'this turn (e.g. "hi", "mr", "te").'
    )

    # Customer-led behavior (fixed policy, generic over every campaign).
    parts.append(
        "Core behavior every turn:\n"
        "1. LISTEN FIRST: answer what the customer actually said, directly and helpfully, in "
        "your own warm words (draw on the knowledge below — never recite).\n"
        "2. THEN gently move toward your objective; talking points are material, not a checklist.\n"
        "3. REDIRECT ONLY WHEN the input is unrelated to this call (weather, wrong number, "
        "chit-chat): briefly acknowledge, then steer back. On-topic questions/concerns: answer, "
        "never deflect.\n"
        "4. Follow the do's and don'ts for tone."
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
            "as you learn from the user. Infer don't ask — pick up lead_gender from the "
            "caller's OWN grammatical forms only (e.g. 'kar raha hun' / 'gaya' → male; "
            "'kar rahi hun' / 'gayi' → female). Address words like 'bhaiya'/'didi' tell "
            "you nothing about the caller's gender — ignore them for inference:\n"
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
        "- Don't repeat a CTA you've already made: if you've already offered the "
        "link/bonus/next step, don't pitch it again unless the user brings it up — answer "
        "and vary your follow-up, or stop. Repetition sounds robotic.\n"
        "- Never invent facts about the company or products.\n"
        "- If asked whether you are AI, answer honestly.\n"
        "- TERMINAL ACTIONS — the response_text IS the closing line; set the action in the "
        "SAME turn you speak the farewell — never say a farewell with action=continue:\n"
        "  * close_positive: When the customer has explicitly accepted the main offer "
        "(e.g., said 'yes' to the link/product) — your response_text should be the closing "
        "line (e.g. 'Bahut shukriya! Aapka din shubh rahe.') and action=close_positive. "
        "A customer being polite, asking questions, or saying they're free to talk is NOT "
        "a close — keep action=continue.\n"
        "  * close_negative: When the customer has declined and you have made your final "
        "acknowledgement — your response_text should be the farewell line and "
        "action=close_negative. 'Not interested right now' still gets one more gentle "
        "attempt before closing.\n"
        "  * IMPORTANT: Any response ending with a farewell ('Aapka din shubh rahe', "
        "'Dhanyavaad', 'Alvida', etc.) MUST have action=close_positive or close_negative. "
        "Never use action=continue after a farewell.\n"
        "  * end: ONLY when there is truly nothing more to say (e.g., call already closed).\n"
        "  * transfer/schedule_callback: see below.\n"
        "- If asked to be removed, set action=close_negative and acknowledge.\n"
        "- Callback: schedule_callback is a TERMINAL action — the call ends immediately "
        "after you say the farewell. Use it once you have a concrete day AND at least a "
        "time window (e.g. 'kal 12 se 3 baje ke beech' is sufficient — you do not need "
        "an exact minute). If the user is vague ('kal', 'baad mein', 'later'), keep "
        "action=continue and ask for the day+window (e.g. 'Kal kis samay ke aaspaas '  "
        "'call karoon?'). Save the confirmed time in updated_slots.callback_time.\n"
        "- Default to action=continue for every normal exchange. When in doubt, continue."
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
        "NEVER a reason to end the call. Always address the customer respectfully using formal "
        "second-person (आप in Hindi, आपण in Marathi) — never तू/तुम or casual exclamations "
        "like अरे; stay formal and courteous even if the customer is casual."
    )

    parts.append(
        "Core behavior every turn:\n"
        "1. CRITICAL — BE BRIEF: reply in ONE short sentence (two at the very most), then STOP "
        "and let the customer talk. This is a fast back-and-forth phone call — never give long "
        "explanations, lists, or monologues. If you have more to say, say it across turns.\n"
        "2. LISTEN FIRST: answer what the customer actually said, directly, in your own warm "
        "words (draw on the knowledge below — never recite).\n"
        "3. THEN nudge gently toward your objective with ONE short line; talking points are "
        "material, not a checklist to read out.\n"
        "4. REDIRECT ONLY WHEN the input is unrelated: briefly acknowledge, then steer back. "
        "On-topic questions/concerns: answer, never deflect.\n"
        "5. DON'T REPEAT AN OFFER: once you've offered the WhatsApp link / bonus / next step, "
        "do NOT bring it up again unless the customer asks — make the offer ONCE, when there's "
        "genuine interest, then move on. Re-asking 'shall I send the link?' every turn sounds "
        "robotic and pushy."
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
            "(* = required) — report via record_turn_signal's updated_slots. Do NOT "
            "interrogate the customer to fill these; infer them from what they say. "
            "For lead_gender: use the caller's OWN grammatical forms only — 'kar raha hun'/"
            "'gaya' → male; 'kar rahi hun'/'gayi' → female. Address words like "
            "'bhaiya'/'didi' say nothing about the caller's gender — ignore them:\n"
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
        "Whenever you decide a next step or learn something about the customer, CALL the "
        "record_turn_signal function with `action` (one of continue, clarify, transfer, "
        "schedule_callback, send_info, close_positive, close_negative, end) and any "
        "`updated_slots` you learned. Use close_positive/close_negative/end ONLY when the "
        "call is genuinely over — a request to change language is a `continue`, never an end. "
        "schedule_callback is TERMINAL (call ends after farewell) — use it once you have "
        "a concrete day AND at least a time window ('12 se 3 ke beech' is sufficient). "
        "IMPORTANT: whenever you speak a farewell ('Aapka din shubh rahe', 'Dhanyavaad', "
        "'Alvida', etc.) you MUST simultaneously call record_turn_signal with "
        "close_positive or close_negative — never leave action=continue after a farewell."
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
    parts: list[str] = []
    parts.append(
        f"You are the customer-support agent for {company_name}. YOU are the support — "
        "never tell the customer to 'contact support' or 'reach out to the team', because "
        "they are already talking to you. Resolve the issue directly. "
        "Answer from the provided knowledge base sources. Do NOT mention source names or "
        "filenames in your replies — just answer naturally. "
        "You are female: use feminine grammatical forms whenever the language requires it "
        "(e.g. Hindi: 'मैं कर सकती हूँ', not 'कर सकता हूँ').\n"
        "CRITICAL — NEVER invent or fabricate data: do not make up account balances, "
        "transaction IDs, bank details, bonus amounts, or any other specific numbers or "
        "values. If you do not have the real data from a tool call or the sources, say so "
        "honestly and escalate to a human agent."
    )

    if has_player_tools:
        player_scope = (
            "2. PLAYER-SPECIFIC questions (balance, transactions, bets, bonuses, KYC status, "
            "account issues): ALWAYS call the relevant player tool — it already has the "
            "customer's account ID and operator ID from the session, so you NEVER need to ask "
            "the customer for those. Do NOT ask the customer for their account ID, transaction "
            "ID, screenshot, or error message. If the tool is unavailable or returns an error, "
            "reply with a brief holding message (e.g. 'I'm looking into your account right now, "
            "please give me a moment') and do not interrogate the customer further. If the tool "
            "still fails after retrying, tell the customer honestly that you are unable to access "
            "their account details right now and suggest they check the app directly — do NOT "
            "invent or guess any data. Only escalate to a human agent as the absolute last resort.\n"
        )
    else:
        player_scope = (
            "2. PLAYER-SPECIFIC questions (balance, transactions, bets, bonuses, KYC status, "
            "account issues, bank/payment details): you do NOT have real-time account lookup "
            "tools. Do NOT attempt to look up, guess, or invent any account-specific data — "
            "no balances, no bank details, no transaction IDs, nothing. Instead, tell the "
            "customer honestly that you don't have access to their account details through "
            "this chat and guide them to find it themselves (e.g. check the Wallet or "
            "Profile section in the app). Only escalate to a human agent if the customer "
            "explicitly asks to speak to a human or cannot resolve the issue on their own.\n"
        )

    parts.append(
        "SCOPE — follow strictly:\n"
        f"1. GENERAL questions about {company_name}'s platform/services (registration, "
        "KYC, wallet, deposits, withdrawals, games, bonuses, responsible gaming, "
        "security, technical help): answer from the sources.\n"
        + player_scope
        + "WITHDRAWAL STATUS RULES — follow precisely when a player asks about a withdrawal:\n"
        "  - Status SUBMITTED/PENDING: tell them it is under review and being processed.\n"
        "  - Status APPROVED and approved within the last 48 hours: reassure them — "
        "'Your withdrawal has been approved and is being processed. It typically takes "
        "24–48 hours to arrive in your account. Please check back if it hasn't arrived "
        "by [approved_at + 48 h].' Do NOT escalate yet.\n"
        "  - Status APPROVED but more than 48 hours have passed since approval: "
        "apologise for the delay and escalate immediately to a human agent — this needs "
        "manual investigation. When escalating, include the withdrawal amount and "
        "approved_at timestamp in the summary.\n"
        "  - Status REJECTED/FAILED: explain it was not processed and ask if they would "
        "like to try again or need help understanding the reason.\n"
        "To judge whether 48 hours have passed, use the current UTC time from your context "
        "(the conversation timestamp) versus the approved_at field in the tool response.\n"
        "ESCALATION RULE — CRITICAL: when you decide to escalate, you MUST call the "
        "`escalate_to_human` tool — writing 'I will escalate' or 'I have escalated' in "
        "your text response is NOT sufficient and will NOT transfer the customer. The tool "
        "call is the only action that actually connects the customer to a human agent. "
        "After calling it, stop asking follow-up questions — tell the customer: "
        "'I'm connecting you to my manager who will look into this right away.'\n"
        "3. OPERATOR-SPECIFIC questions (payment methods, limits, enabled games, "
        "promotions, operator config): call the relevant operator tool if available. "
        "If not, answer from the sources; if the sources don't cover it, escalate to "
        "a human agent.\n"
        f"4. Anything UNRELATED to {company_name} or its platform: politely say you can "
        f"only help with {company_name} support questions.\n"
        "5. VOICE CALL request (user says 'start a call', 'I want to talk', 'call me', "
        "'voice call', etc.): IMMEDIATELY call the `offer_voice_call` tool — the system "
        "sets up a browser-based web call automatically. Do NOT ask for a phone number. "
        "Do NOT put call-related questions in suggested_followups."
    )
    if rag_context:
        parts.append("Reference sources:\n" + rag_context)

    parts.append(
        "LANGUAGE — CRITICAL RULES:\n"
        "1. Detect the language AND script of the user's LATEST message. Reply in EXACTLY "
        "that language AND script. The user's choice always overrides the configured default "
        f"({language_default}).\n"
        "2. SCRIPT MATCHING — this is the most important rule: if the user writes Hindi (or "
        "any Indic language) using ROMAN/LATIN letters (Hinglish — e.g. 'mera balance kya "
        "hai', 'deposit karna hai'), reply in ROMAN script Hinglish. Do NOT switch to "
        "Devanagari. If they write in Devanagari ('मेरा बैलेंस क्या है'), reply in "
        "Devanagari. Match the script they used.\n"
        "3. If they switch language or script mid-conversation, switch immediately.\n"
        "4. suggested_followups must also match the user's language and script.\n"
        "IMPORTANT: the reference sources may be in Hindi or another language — that does "
        "NOT determine your reply language or script. Only the user's message does.\n"
        "Tone: casual, warm, friendly — like a helpful friend. For Hinglish, mix Hindi and "
        "English the way people actually talk (e.g. 'Aapka balance ₹500 hai, koi issue?'). "
        "English tech terms mixed in are always fine. Supported languages: Hindi, English, "
        "Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu."
    )

    parts.append(
        "RESPONSE LENGTH — CRITICAL:\n"
        "Keep replies short: 1–2 sentences for conversational answers and simple lookups. "
        "Only write more when a tool/API result genuinely requires it (e.g. listing multiple "
        "transactions, step-by-step instructions). Never pad, summarise, or repeat what you "
        "just said. If you can say it in one sentence, use one sentence."
    )

    parts.append(
        "Respond with a single JSON object matching this schema:\n"
        + json.dumps(CHATBOT_RESPONSE_SCHEMA, indent=2)
    )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)
