from __future__ import annotations

import logging

from src.pipeline.text_normalize import (
    DEFAULT_PRONUNCIATIONS,
    apply_pronunciations,
    normalize_for_tts,
)


def test_rewrites_known_terms_to_devanagari() -> None:
    out = apply_pronunciations("WhatsApp par link bhejun? Casino bhi hai.")
    # Mispronounced English terms are gone, replaced by Devanagari.
    assert "WhatsApp" not in out and "Casino" not in out and "link" not in out
    assert DEFAULT_PRONUNCIATIONS["WhatsApp"] in out
    assert DEFAULT_PRONUNCIATIONS["Casino"] in out
    assert DEFAULT_PRONUNCIATIONS["link"] in out
    # Surrounding text is preserved.
    assert "bhejun" in out and "bhi hai" in out


def test_case_insensitive_whole_word_only() -> None:
    out = apply_pronunciations("whatsapp WHATSAPP Whatsapp")
    assert out.count(DEFAULT_PRONUNCIATIONS["WhatsApp"]) == 3
    # 'app' must not match inside another word, only standalone.
    assert apply_pronunciations("happy apple") == "happy apple"
    assert DEFAULT_PRONUNCIATIONS["app"] in apply_pronunciations("open the app")


def test_empty_and_no_match_passthrough() -> None:
    assert apply_pronunciations("") == ""
    assert apply_pronunciations("sirf hindi text hai") == "sirf hindi text hai"


def test_extra_overrides_merge_over_defaults() -> None:
    out = apply_pronunciations("Khelo Aviator aur ZyxBrand", extra={"ZyxBrand": "ज़िक्सब्रांड"})
    assert "ज़िक्सब्रांड" in out
    assert DEFAULT_PRONUNCIATIONS["Aviator"] in out


def test_rewrites_newly_added_sports_and_gaming_terms() -> None:
    # These specific words were observed un-transliterated in real Stage test
    # transcripts (still in Latin script when reaching TTS) before this
    # expansion — this test guards against a future accidental removal.
    out = apply_pronunciations(
        "Cricket aur Football dono hai, aap join karke explore kar sakte hain, "
        "koi risk nahi. Matka bhi khel sakte hain."
    )
    assert "Cricket" not in out and DEFAULT_PRONUNCIATIONS["Cricket"] in out
    assert "Football" not in out and DEFAULT_PRONUNCIATIONS["Football"] in out
    assert "join" not in out and DEFAULT_PRONUNCIATIONS["join"] in out
    assert "explore" not in out and DEFAULT_PRONUNCIATIONS["explore"] in out
    assert "risk" not in out and DEFAULT_PRONUNCIATIONS["risk"] in out
    assert "Matka" not in out and DEFAULT_PRONUNCIATIONS["Matka"] in out
    # Surrounding Hindi/Hinglish text is preserved untouched.
    assert "aur" in out and "dono hai" in out and "koi" in out and "nahi" in out


def test_rewrites_caller_name_manoj() -> None:
    # The dev-console defaults the caller name to "Manoj" for IndicF5/ElevenLabs
    # (see static/dev_console.html applyTTSProviderDefaults) — it must be in the
    # dictionary or IndicF5 mispronounces the agent's own self-introduction.
    out = apply_pronunciations("Main Manoj baat kar raha hoon.")
    assert "Manoj" not in out and DEFAULT_PRONUNCIATIONS["Manoj"] in out


def test_normalize_for_tts_warns_on_devanagari_language_gap(caplog) -> None:
    # Devanagari-dominant text with one genuine gap word -- "ZyxUnknownBrand"
    # is guaranteed not to be in DEFAULT_PRONUNCIATIONS. Script-dominant text
    # (>=60% Devanagari) with a residual Latin word is exactly the case this
    # mechanism must catch.
    with caplog.at_level(logging.WARNING):
        normalize_for_tts(
            "व्हाट्सऐप पर लिंक भेजता हूं ZyxUnknownBrand", language="hi-IN"
        )
    assert any(
        r.levelno == logging.WARNING and "ZyxUnknownBrand" in str(r.__dict__.get("words"))
        for r in caplog.records
    )


def test_normalize_for_tts_no_warning_when_fully_covered(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        normalize_for_tts("व्हाट्सऐप पर लिंक भेजता हूं", language="hi-IN")
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_normalize_for_tts_no_warning_for_romanized_hinglish(caplog) -> None:
    # Romanized Hinglish ("par", "bhejun", "aur", "khelo" are ordinary Hindi
    # words correctly left in Latin script) must NOT be flagged -- this is
    # expected input shape, not a pronunciation-dictionary gap. This is the
    # exact false positive an earlier version of this mechanism produced.
    with caplog.at_level(logging.WARNING):
        normalize_for_tts("WhatsApp par link bhejun aur Cricket khelo", language="hi-IN")
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_normalize_for_tts_warns_for_script_dominant_unsupported_language(caplog) -> None:
    # Telugu ("te") has zero normalization today. This is real Telugu-script
    # text (generated via indic_transliteration from a known-correct
    # Devanagari sentence, so the Telugu itself is linguistically valid) with
    # one residual English word -- exactly the "no normalization exists yet
    # for this language" gap this task must surface.
    with caplog.at_level(logging.WARNING):
        result = normalize_for_tts(
            "WhatsApp మైం ఆపకో లింక భేజతా హూం", language="te-IN"
        )
    assert result == "WhatsApp మైం ఆపకో లింక భేజతా హూం"  # unchanged, as before this task
    assert any(
        r.levelno == logging.WARNING and "WhatsApp" in str(r.__dict__.get("words"))
        for r in caplog.records
    )


def test_normalize_for_tts_empty_text_no_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = normalize_for_tts("", language="hi-IN")
    assert result == ""
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
