"""Dynamic language switching: the pure resolver that decides the conversation's
active language each turn from the LLM-reported and STT-detected signals."""

from __future__ import annotations

from src.dialogue.language import (
    normalize_lang,
    resolve_active_language,
    to_bcp47,
)


def test_normalize_lang_strips_region_and_lowercases():
    assert normalize_lang("hi-IN") == "hi"
    assert normalize_lang("MR-IN") == "mr"
    assert normalize_lang("mr") == "mr"
    assert normalize_lang("  te-IN ") == "te"
    assert normalize_lang("") == ""
    assert normalize_lang(None) == ""


def test_to_bcp47_adds_region_for_bare_codes():
    assert to_bcp47("mr") == "mr-IN"
    assert to_bcp47("hi") == "hi-IN"
    assert to_bcp47("hi-IN") == "hi-IN"       # already regioned, unchanged
    assert to_bcp47("") == ""


def test_keeps_current_when_no_signal():
    # No LLM/STT signal → sticky, stays put.
    assert resolve_active_language("hi", stt_lang=None, llm_lang=None) == "hi"
    assert resolve_active_language("mr", stt_lang="", llm_lang="") == "mr"


def test_llm_reported_language_switches():
    # The LLM is the arbiter — when it reports a different language, switch.
    assert resolve_active_language("hi", llm_lang="mr") == "mr"
    assert resolve_active_language("hi", llm_lang="mr-IN") == "mr"   # normalized


def test_llm_can_revert_to_default():
    # Switching back to Hindi when the LLM goes back is allowed.
    assert resolve_active_language("mr", llm_lang="hi") == "hi"


def test_stt_detected_switch_when_llm_silent():
    # LLM omitted language → fall back to a confident STT detection.
    assert resolve_active_language("hi", stt_lang="mr", llm_lang=None) == "mr"


def test_llm_wins_over_stt():
    # Both present → LLM (the arbiter) wins.
    assert resolve_active_language("hi", stt_lang="en", llm_lang="mr") == "mr"


def test_same_language_is_a_no_op():
    assert resolve_active_language("hi", stt_lang="hi", llm_lang="hi") == "hi"
    assert resolve_active_language("mr", stt_lang="mr-IN", llm_lang="mr") == "mr"


def test_normalize_lang_rejects_spelled_out_language_name():
    # Groq/Whisper rejects a spelled-out name outright ("unsupported language:
    # hindi") when this reaches an STT config unnormalized — the LLM
    # occasionally self-reports "Hindi" instead of the code "hi" despite being
    # asked for a code; that must not corrupt the active language.
    assert normalize_lang("hindi") == ""
    assert normalize_lang("Hindi") == ""
    assert normalize_lang("Marathi-IN") == ""


def test_llm_spelled_out_language_name_does_not_switch_active_language():
    # An invalid LLM-reported value must not silently become the new active
    # language — falls through to STT, then to the current language, exactly
    # as if the LLM had omitted the field.
    assert resolve_active_language("hi", llm_lang="Hindi") == "hi"
    assert resolve_active_language("hi", stt_lang="mr", llm_lang="Hindi") == "mr"
