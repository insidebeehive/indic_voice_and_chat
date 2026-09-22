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
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.logging import debug_event

log = logging.getLogger(__name__)

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

_CHATBOT_FALLBACK_TEXT = "Sorry, I couldn't formulate an answer. Could you rephrase?"
_VOICEBOT_FALLBACK_TEXT = "Maaf kijiye, main samjha nahi. Kya aap dobara bata sakte hain?"
# Shown when the LLM's raw output was completely empty (even after the one
# built-in retry). Must NOT imply a follow-up is coming — this is a
# turn-based request/response system, not a live session; nothing else will
# be sent unless the customer asks again. Split by bot type/language like
# the other two fallbacks above (a prior single hardcoded-Hindi version was
# shown even in English chatbot conversations).
_CHATBOT_EMPTY_RESPONSE_TEXT = "Sorry, I wasn't able to put together an answer — could you ask that again?"
_VOICEBOT_EMPTY_RESPONSE_TEXT = "Maaf kijiye, main jawab nahi de payi. Kya aap dobara pooch sakte hain?"


def is_unusable_response(response_text: str) -> bool:
    """True when the parser salvaged nothing real and fell back to one of its
    canned lines — the caller should consider retrying rather than showing
    this to the customer as-is.

    Covers all four canned-fallback branches (empty input for each bot type,
    missing response_text field, and a still-JSON-looking leftover after a
    failed parse — see ``_fallback_text``) via a single check against their
    known literal text, so it stays correct automatically if a fifth branch
    is ever added here as long as it also returns one of these constants.
    """
    return response_text in (
        _CHATBOT_FALLBACK_TEXT, _VOICEBOT_FALLBACK_TEXT,
        _CHATBOT_EMPTY_RESPONSE_TEXT, _VOICEBOT_EMPTY_RESPONSE_TEXT,
    )


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
        # The envelope parsed cleanly (unlike the _extract_json failure above)
        # but the model didn't put anything in response_text -- a distinct
        # outcome from a malformed envelope. agents/voicebot.py's own
        # "voicebot turn parse_recovered" event already covers this at both
        # of its call sites (it fires on any truthy parse_error, including
        # this one), but that event only exists there -- log it here too so
        # any OTHER caller of this parser isn't silent about it.
        debug_event(log, "response_parser voicebot_response empty", raw=text, parsed=obj)
        return VoiceBotResponse(
            response_text=_VOICEBOT_FALLBACK_TEXT,
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
        # agents/chatbot.py's _handle_with_tools logs this via its own
        # "chatbot response_envelope parse_fallback" event (any truthy
        # parse_error, including this one), but its sibling _single_shot path
        # calls this same function with no parse_error logging of its own --
        # so this is the only trace that call site leaves. Reported as a gap
        # rather than fixed there: out of this package's scope.
        debug_event(log, "response_parser chatbot_response empty", raw=text, parsed=obj)
        return ChatBotResponse(
            response_text=_CHATBOT_FALLBACK_TEXT,
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
    candidates: list[tuple[str, str]] = []
    if fence:
        candidates.append(("fenced_block", fence.group(1)))
    # Try the whole text.
    candidates.append(("whole_text", text.strip()))
    # Try the largest balanced {...} block.
    block = _largest_balanced_block(text)
    if block:
        candidates.append(("balanced_block", block))

    last_error = "no JSON object found"
    for source, c in candidates:
        try:
            obj = json.loads(c, strict=False)
            if isinstance(obj, dict):
                if source != "whole_text":
                    # Direct parse of the whole text (the first candidate
                    # tried when there's no fence) is the happy path and
                    # would fire on every clean turn -- not worth a line.
                    # Reaching a later candidate means the raw text needed
                    # salvaging (a code fence stripped, or a stray prose
                    # prefix/suffix cut away by the balanced-brace scan),
                    # which is exactly the "what did the model actually
                    # emit vs. what we recovered" question an operator asks.
                    debug_event(
                        log, "response_parser extract_json recovered",
                        source=source, raw=text, recovered=c,
                    )
                return obj, None
            last_error = "JSON was not an object"
        except json.JSONDecodeError as e:
            last_error = f"json decode: {e}"
            continue
    # Every candidate failed -- shared by both parse_voicebot_response and
    # parse_chatbot_response (their callers' own recovery events log the
    # fallback TEXT substituted, not this decoder-level detail: which
    # candidates were tried and the literal JSONDecodeError) -- and by
    # src/analysis/call_outcome.py, which calls this function directly with
    # no logging of its own around the result at all.
    debug_event(log, "response_parser extract_json failed", raw=text, error=last_error)
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

    If the leftover text still looks like an unparsed JSON envelope, return
    a safe generic message instead (never expose/speak raw JSON). Otherwise:
    speakable=True (voicebot) cuts to the first sentence so partial JSON
    isn't read aloud; speakable=False (chatbot) returns the full cleaned
    text verbatim.
    """
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    if not cleaned:
        return _VOICEBOT_EMPTY_RESPONSE_TEXT if speakable else _CHATBOT_EMPTY_RESPONSE_TEXT
    if cleaned.startswith("{") and '"response_text"' in cleaned:
        # Still looks like an unparsed JSON envelope — never expose raw LLM
        # JSON to the customer (or speak it via TTS). Fall back to the same
        # safe generic message used for a missing response_text field.
        return _VOICEBOT_FALLBACK_TEXT if speakable else _CHATBOT_FALLBACK_TEXT
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
