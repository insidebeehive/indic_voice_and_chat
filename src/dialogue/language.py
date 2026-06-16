"""Dynamic conversation-language resolution for the layered voice pipeline.

The agent starts in the campaign's default language and switches to the caller's
language when they speak it or ask for it. The decision is made here as a pure
function so it's easy to test and reason about; the dialogue layer
(``VoiceBotAgent``) owns the ``active_language`` state and feeds it the per-turn
signals (the LLM's self-reported ``language`` and the STT-detected language).

Codes are kept in a canonical **base form** internally (``"mr"``, ``"hi"``);
``to_bcp47`` formats them for providers that want a region (``"mr-IN"``).
"""

from __future__ import annotations

from typing import Optional


def normalize_lang(code: Optional[str]) -> str:
    """Canonical base language code: lowercased, region stripped.

    ``"hi-IN"`` → ``"hi"``, ``"MR"`` → ``"mr"``, ``None``/``""`` → ``""``.
    """
    if not code:
        return ""
    return str(code).strip().lower().split("-")[0]


def to_bcp47(code: Optional[str]) -> str:
    """Format a base code for providers that want a region (Sarvam: ``xx-IN``).

    ``"mr"`` → ``"mr-IN"``; an already-regioned code passes through; ``""`` → ``""``.
    """
    if not code:
        return ""
    code = str(code).strip()
    if "-" in code:
        return code
    return f"{code.lower()}-IN"


def resolve_active_language(
    current: str,
    *,
    stt_lang: Optional[str] = None,
    llm_lang: Optional[str] = None,
) -> str:
    """Decide the conversation's active language for the next turn (sticky).

    The LLM is the arbiter — it sees both what the caller spoke and any explicit
    request ("can you speak Marathi?"), and is prompted to report the language it
    is now speaking. So an explicit LLM-reported language wins. When the LLM
    omits it, a confident STT-detected language that differs from the current one
    triggers the switch. With no signal, the current language sticks.
    """
    cur = normalize_lang(current)
    llm = normalize_lang(llm_lang)
    if llm:
        return llm
    stt = normalize_lang(stt_lang)
    if stt and stt != cur:
        return stt
    return cur
