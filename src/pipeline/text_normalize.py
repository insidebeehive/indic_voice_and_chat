"""Pronunciation fixes for Indic TTS.

Indic TTS (e.g. Sarvam) mispronounces Latin-script English / brand words
embedded in Hindi text — "WhatsApp" comes out as "ThatsApp", "Casino" as
"Hasino". Rewriting those words to a Devanagari phonetic spelling before
synthesis makes the TTS pronounce them correctly.

The map is plain data: edit/extend ``DEFAULT_PRONUNCIATIONS`` or pass
campaign-specific overrides via ``apply_pronunciations(text, extra=...)``.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# English / brand term -> Devanagari phonetic spelling. Matched whole-word and
# case-insensitively. Keep entries high-confidence; a wrong spelling just trades
# one mispronunciation for another.
#
# Generic (industry-neutral) only -- betting/gambling-vertical vocabulary
# (Casino, Aviator, Cricket, market, etc.) lives in a CRM's own
# ``Crm.pronunciation_overrides`` instead (see ``apply_pronunciations``'s
# ``extra`` param and src/auth/db_resolver.py), so a non-gambling CRM doesn't
# inherit gambling-flavored substitutions by default.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "WhatsApp": "व्हाट्सऐप",
    "app": "ऐप",
    "link": "लिंक",
    "bonus": "बोनस",
    "cash": "कैश",
    "commission": "कमीशन",
    "registration": "रजिस्ट्रेशन",
    "deposit": "डिपॉज़िट",
    "withdrawal": "विड्रॉल",
    "instant": "इंस्टंट",
    "account": "अकाउंट",
    "support": "सपोर्ट",
    "update": "अपडेट",
    "trusted": "ट्रस्टेड",
    "safe": "सेफ",
    "official": "ऑफिशियल",
    "free": "फ्री",
    "automatic": "ऑटोमैटिक",
    "minimum": "मिनिमम",
    "join": "जॉइन",
    "download": "डाउनलोड",
    "platform": "प्लेटफॉर्म",
    "live": "लाइव",
    "guide": "गाइड",
    "explore": "एक्सप्लोर",
    "opportunity": "अपॉर्चुनिटी",
    "risk": "रिस्क",
    "perfect": "परफेक्ट",
    "Sales": "सेल्स",
    "team": "टीम",
    "refer": "रेफर",
    "international": "इंटरनेशनल",
    "fan": "फैन",
}


def apply_pronunciations(text: str, extra: dict[str, str] | None = None) -> str:
    """Rewrite known mispronounced terms to Devanagari so TTS says them right.

    Whole-word, case-insensitive. ``extra`` (e.g. a campaign's own overrides)
    is merged over the defaults and wins on conflict.
    """
    if not text:
        return text
    table = {**DEFAULT_PRONUNCIATIONS, **(extra or {})}
    if not table:
        return text
    lower = {k.lower(): v for k, v in table.items()}
    # Longest keys first so multi-word / longer terms win over their substrings.
    keys = sorted(table.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE
    )
    return pattern.sub(lambda m: lower[m.group(0).lower()], text)


# Currency: Sarvam TTS doesn't vocalize the ₹ symbol or a bare "Rs", so amounts
# like "₹100" / "Rs 100" get dropped. Rewrite to spoken Hindi: "100 रुपये".
_CURRENCY_RE = re.compile(r"(?:₹|\bRs\.?)\s*([\d][\d,]*)", re.IGNORECASE)


def normalize_currency(text: str) -> str:
    """Rewrite ``₹100`` / ``Rs 100`` / ``Rs. 1,000`` to ``100 रुपये`` so the
    amount is actually spoken. Spelled-out forms (``100 रुपये``) are untouched."""
    if not text:
        return text
    return _CURRENCY_RE.sub(lambda m: f"{m.group(1).replace(',', '')} रुपये", text)


# Indian languages written in Devanagari. The pronunciation + currency rewrites
# above are Devanagari, so they're correct for these and wrong (wrong script) for
# others (Telugu, Malayalam, Tamil, …).
DEVANAGARI_LANGS = frozenset({"hi", "mr", "ne", "sa", "kok", "mai", "bho", "doi"})

# Any Latin-script word run. Used only for gap visibility (see
# _warn_if_untransliterated below) -- never to decide what to rewrite.
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']*\b")

# Unicode block start/end for each script, keyed by base language code. Used
# only to judge whether TTS-bound text is actually written in the active
# language's native script before treating a residual Latin word as a
# genuine gap -- romanized Hinglish/English is expected to stay in Latin
# script for languages that don't have their own normalization yet, and must
# not be flagged as a mispronunciation risk (see the "Revision note" above).
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "hi": (0x0900, 0x097F),  # Devanagari -- also covers mr, ne, sa, kok, mai, bho, doi
    "bn": (0x0980, 0x09FF),  # Bengali -- also used as a stand-in for Assamese ("as")
    "as": (0x0980, 0x09FF),
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "od": (0x0B00, 0x0B7F),  # Odia
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
}


def _is_script_dominant(text: str, language: str) -> bool:
    """True when at least 60% of the text's alphabetic characters are in the
    active language's native script -- i.e. this looks like real native-script
    text with isolated Latin words in it, not romanized Hinglish/English
    written wholesale in Latin script (a different, out-of-scope problem).
    """
    lang_key = "hi" if language in DEVANAGARI_LANGS else language
    script_range = _SCRIPT_RANGES.get(lang_key)
    if script_range is None:
        return False
    lo, hi = script_range
    native_count = latin_count = 0
    for ch in text:
        if ch.isascii() and ch.isalpha():
            latin_count += 1
        elif lo <= ord(ch) <= hi:
            native_count += 1
    total = native_count + latin_count
    return total > 0 and (native_count / total) >= 0.6


def _warn_if_untransliterated(text: str, language: str) -> None:
    """Log a warning for every Latin-script word still present in
    script-dominant TTS-bound text. For a DEVANAGARI_LANGS language this
    means a DEFAULT_PRONUNCIATIONS gap (a real word TTS will likely
    mispronounce); for any other language it means every word in the call,
    since no normalization runs for that language yet. Text that isn't
    script-dominant (e.g. romanized Hinglish) is skipped entirely -- that's
    expected input shape, not a gap. Purely observational -- never raises,
    never changes the text that reaches TTS.
    """
    if not _is_script_dominant(text, language):
        return
    words = _LATIN_WORD_RE.findall(text)
    if words:
        log.warning(
            "tts text has un-transliterated Latin-script word(s); TTS will "
            "likely mispronounce them",
            extra={"language": language, "words": words, "text_sample": text[:160]},
        )


def normalize_for_tts(
    text: str, language: str | None = None, extra: dict[str, str] | None = None
) -> str:
    """Language-aware TTS text normalization.

    Applies the Devanagari currency + pronunciation rewrites only for
    Devanagari-script languages (Hindi, Marathi, …). For other scripts (Telugu,
    Malayalam) injecting Devanagari would render the wrong script, so the text is
    returned unchanged until per-language maps exist. An unknown/empty language
    keeps the legacy behaviour (apply — assumes Hindi).

    Either way, any Latin-script word still present in genuinely
    script-dominant text this function returns is logged as a warning (see
    ``_warn_if_untransliterated``) — a DEFAULT_PRONUNCIATIONS gap for a
    Devanagari-script language, or simply "no normalization exists yet for
    this language" for any other one. Romanized Hinglish/English is not
    flagged, since that's expected input shape, not a gap. This makes
    coverage gaps visible in logs instead of requiring someone to notice a
    mispronounced word by ear.
    """
    if not text:
        return text
    base = (language or "").strip().lower().split("-")[0]
    if base and base not in DEVANAGARI_LANGS:
        _warn_if_untransliterated(text, base)
        return text
    result = apply_pronunciations(normalize_currency(text), extra=extra)
    _warn_if_untransliterated(result, base or "hi")
    return result
