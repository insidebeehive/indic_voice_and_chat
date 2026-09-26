"""URLs are replaced with "website" in the text's own script before TTS.

Production (ticket 10944): a Hindi voice reply was sent
"आप हमारी वेबसाइट https://rama567.com पर जाकर…" and TTS read the URL aloud."""
from __future__ import annotations

import pytest

from src.pipeline.text_normalize import normalize_for_tts, replace_urls_for_tts


@pytest.mark.parametrize("text, language, expected", [
    # The production sentence: "website" already precedes the URL, so it's dropped.
    ("आप हमारी वेबसाइट https://rama567.com पर जाकर सीधे ऐप डाउनलोड कर सकते हैं।", "hi",
     "आप हमारी वेबसाइट पर जाकर सीधे ऐप डाउनलोड कर सकते हैं।"),
    ("ऐप डाउनलोड करने के लिए https://rama567.com/app खोलें।", "hi",
     "ऐप डाउनलोड करने के लिए वेबसाइट खोलें।"),
    # Hinglish arrives as "hi" but is Latin script: Latin placeholder.
    ("Aap rama567.com par jaakar app download kar sakte hain.", "hi",
     "Aap website par jaakar app download kar sakte hain."),
    ("Please visit our website www.rama567.com.", "en", "Please visit our website."),
    ("Open https://rama567.com, then log in.", "en", "Open website, then log in."),
    ("ನಮ್ಮ ವೆಬ್‌ಸೈಟ್ https://rama567.com ನಲ್ಲಿ ನೋಡಿ", "kn", "ನಮ್ಮ ವೆಬ್‌ಸೈಟ್ ನಲ್ಲಿ ನೋಡಿ"),
    ("இங்கே https://rama567.com பார்க்கவும்", "ta", "இங்கே இணையதளம் பார்க்கவும்"),
])
def test_urls_become_website_in_the_texts_script(text, language, expected) -> None:
    assert replace_urls_for_tts(text, language) == expected


@pytest.mark.parametrize("text", [
    "No link here, just ₹500.",
    "Balance is 1.5 lakh and version 2.0 works",
    "Minimum deposit ₹200.50 hai.",
    "",
])
def test_text_without_urls_is_unchanged(text) -> None:
    assert replace_urls_for_tts(text, "en") == text


def test_normalize_for_tts_replaces_urls_for_every_language() -> None:
    """Both branches: Devanagari languages (full normalization) and the others
    (returned otherwise unchanged) must drop the URL."""
    hi = normalize_for_tts("आप हमारी वेबसाइट https://rama567.com पर जाकर डाउनलोड करें।", "hi")
    assert "https" not in hi and "rama567" not in hi
    kn = normalize_for_tts("ನಮ್ಮ ವೆಬ್‌ಸೈಟ್ https://rama567.com ನಲ್ಲಿ ನೋಡಿ", "kn")
    assert "https" not in kn and "rama567" not in kn


def test_url_no_longer_triggers_the_untransliterated_warning(caplog) -> None:
    import logging
    caplog.set_level(logging.WARNING, logger="src.pipeline.text_normalize")
    normalize_for_tts("आप हमारी वेबसाइट https://rama567.com पर जाकर सीधे ऐप डाउनलोड कर सकते हैं।", "hi")
    assert not any("un-transliterated" in r.getMessage() for r in caplog.records)
