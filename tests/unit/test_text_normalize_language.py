"""Language-aware TTS normalization: the Devanagari pronunciation + currency
rewrites must only apply to Devanagari-script languages, so a switch to Telugu/
Malayalam doesn't inject Devanagari into the wrong script."""

from __future__ import annotations

from src.pipeline.text_normalize import normalize_for_tts


def test_devanagari_languages_get_pronunciations_and_currency():
    for lang in ("hi", "hi-IN", "mr", "mr-IN"):
        out = normalize_for_tts("Pay ₹100 bonus on the app", lang)
        assert "रुपये" in out          # ₹100 -> "100 रुपये"
        assert "ऐप" in out              # app -> Devanagari spelling


def test_non_devanagari_languages_are_left_untouched():
    text = "Pay ₹100 bonus on the app"
    assert normalize_for_tts(text, "te-IN") == text   # Telugu — no Devanagari injected
    assert normalize_for_tts(text, "ml") == text       # Malayalam
    assert normalize_for_tts(text, "ta-IN") == text    # Tamil


def test_unknown_or_empty_language_keeps_legacy_behaviour():
    # When the language is unknown we keep the previous behaviour (apply).
    assert "ऐप" in normalize_for_tts("the app", "")
    assert "ऐप" in normalize_for_tts("the app", None)


def test_extra_overrides_still_apply_for_devanagari():
    out = normalize_for_tts("the FooBrand", "mr", extra={"FooBrand": "फूब्रांड"})
    assert "फूब्रांड" in out
