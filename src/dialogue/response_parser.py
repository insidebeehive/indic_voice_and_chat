"""Structured LLM response parser.

The LLM is asked to emit a JSON object matching VOICEBOT_RESPONSE_SCHEMA
(PRD §12.2). In practice models sometimes:
- wrap the JSON in markdown fences (```json ... ```)
- prepend a stray sentence ("Here's the JSON:")
- emit truncated JSON when ``max_tokens`` is hit

``parse_voicebot_response`` is forgiving: it extracts the largest valid JSON
object it can find and validates it against the schema. On failure it
returns a fallback object with action=clarify so the conversation keeps
moving instead of crashing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class VoiceBotResponse:
    response_text: str
    language: str = "hi"
    conversation_phase: Optional[str] = None
    updated_slots: dict[str, Any] = field(default_factory=dict)
    action: str = "continue"
    action_reason: str = ""
    sentiment: str = "neutral"
    internal_notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    parse_error: Optional[str] = None


@dataclass
class ChatBotResponse:
    response_text: str
    language: str = "en"
    sources_used: list[str] = field(default_factory=list)
    confidence: str = "medium"
    action: str = "none"
    suggested_followups: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    parse_error: Optional[str] = None


_VOICEBOT_ACTIONS = {
    "continue", "clarify", "transfer", "schedule_callback",
    "send_info", "close_positive", "close_negative", "end",
}
_VOICEBOT_PHASES = {"opening", "pitch", "qualification", "objection", "closing"}
_SENTIMENTS = {"positive", "neutral", "negative", "frustrated"}

_CHATBOT_ACTIONS = {"none", "schedule_callback", "send_info", "create_ticket", "escalate", "resolved"}
_CONFIDENCES = {"high", "medium", "low"}


def parse_voicebot_response(text: str) -> VoiceBotResponse:
    """Tolerantly extract a VoiceBotResponse from LLM output."""
    obj, error = _extract_json(text)
    if obj is None:
        return VoiceBotResponse(
            response_text=_fallback_text(text, speakable=True),
            action="clarify",
            parse_error=error,
        )

    response_text = _str(obj.get("response_text"), "")
    if not response_text:
        return VoiceBotResponse(
            response_text="Maaf kijiye, main samjha nahi. Kya aap dobara bata sakte hain?",
            action="clarify",
            parse_error="missing response_text",
            raw=obj,
        )

    return VoiceBotResponse(
        response_text=response_text,
        language=_str(obj.get("language"), "hi"),
        conversation_phase=_enum(obj.get("conversation_phase"), _VOICEBOT_PHASES),
        updated_slots=obj.get("updated_slots") if isinstance(obj.get("updated_slots"), dict) else {},
        action=_enum(obj.get("action"), _VOICEBOT_ACTIONS) or "continue",
        action_reason=_str(obj.get("action_reason"), ""),
        sentiment=_enum(obj.get("sentiment"), _SENTIMENTS) or "neutral",
        internal_notes=_str(obj.get("internal_notes"), ""),
        raw=obj,
    )


def parse_chatbot_response(text: str) -> ChatBotResponse:
    obj, error = _extract_json(text)
    if obj is None:
        return ChatBotResponse(
            response_text=_fallback_text(text),
            parse_error=error,
        )

    response_text = _str(obj.get("response_text"), "")
    if not response_text:
        return ChatBotResponse(
            response_text="Sorry, I couldn't formulate an answer. Could you rephrase?",
            parse_error="missing response_text",
            raw=obj,
        )

    return ChatBotResponse(
        response_text=response_text,
        language=_str(obj.get("language"), "en"),
        sources_used=[str(s) for s in (obj.get("sources_used") or []) if s],
        confidence=_enum(obj.get("confidence"), _CONFIDENCES) or "medium",
        action=_enum(obj.get("action"), _CHATBOT_ACTIONS) or "none",
        suggested_followups=[
            str(s) for s in (obj.get("suggested_followups") or []) if s
        ],
        raw=obj,
    )


# --- helpers --------------------------------------------------------------


def _extract_json(text: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not text or not text.strip():
        return None, "empty response"

    # Try fenced code block first.
    fence = _FENCE_RE.search(text)
    candidates: list[str] = []
    if fence:
        candidates.append(fence.group(1))
    # Try the whole text.
    candidates.append(text.strip())
    # Try the largest balanced {...} block.
    block = _largest_balanced_block(text)
    if block:
        candidates.append(block)

    last_error = "no JSON object found"
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj, None
            last_error = "JSON was not an object"
        except json.JSONDecodeError as e:
            last_error = f"json decode: {e}"
            continue
    return None, last_error


def _largest_balanced_block(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    return text[start : end + 1]


def _fallback_text(text: str, speakable: bool = False) -> str:
    """When parsing fails, salvage *something* from the raw LLM output.

    speakable=True (voicebot): cut to the first sentence so JSON is never
    read aloud.  speakable=False (chatbot): return the full cleaned text.
    """
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return "Maaf kijiye, ek minute de dijiye."
    if not speakable:
        return cleaned
    # Voice path: take the first sentence-ish chunk so JSON isn't spoken.
    for sep in (". ", "! ", "? ", "।"):
        idx = cleaned.find(sep)
        if idx != -1:
            return cleaned[: idx + len(sep)].strip()
    return cleaned[:200]


def _str(v: Any, default: str) -> str:
    return v if isinstance(v, str) and v else default


def _enum(v: Any, allowed: set[str]) -> Optional[str]:
    if isinstance(v, str) and v in allowed:
        return v
    return None
