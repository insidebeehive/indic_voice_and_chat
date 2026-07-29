"""Pronunciation fixes for Indic TTS.

Indic TTS (e.g. Sarvam) mispronounces Latin-script English / brand words
embedded in Hindi text — "WhatsApp" comes out as "ThatsApp", "Casino" as
"Hasino". Rewriting those words to a Devanagari phonetic spelling before
synthesis makes the TTS pronounce them correctly.

The map is plain data: edit/extend ``DEFAULT_PRONUNCIATIONS`` or pass
campaign-specific overrides via ``apply_pronunciations(text, extra=...)``.
"""

from __future__ import annotations

import re

# English / brand term -> Devanagari phonetic spelling. Matched whole-word and
# case-insensitively. Keep entries high-confidence; a wrong spelling just trades
# one mispronunciation for another.
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "WhatsApp": "व्हाट्सऐप",
    "Casino": "कसीनो",
    "Aviator": "एविएटर",
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
    "betting": "बेटिंग",
    "Cricket": "क्रिकेट",
    "Football": "फुटबॉल",
    "Matka": "मटका",
    "download": "डाउनलोड",
    "platform": "प्लेटफॉर्म",
    "Sports": "स्पोर्ट्स",
    "Tennis": "टेनिस",
    "Basketball": "बास्केटबॉल",
    "live": "लाइव",
    "guide": "गाइड",
    "explore": "एक्सप्लोर",
    "opportunity": "अपॉर्चुनिटी",
    "risk": "रिस्क",
    "perfect": "परफेक्ट",
    "Sales": "सेल्स",
    "team": "टीम",
    "refer": "रेफर",
    "market": "मार्केट",
    "match": "मैच",
    "matches": "मैचेस",
    "international": "इंटरनेशनल",
    "fan": "फैन",
    "IPL": "आईपीएल",
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


def normalize_for_tts(
    text: str, language: str | None = None, extra: dict[str, str] | None = None
) -> str:
    """Language-aware TTS text normalization.

    Applies the Devanagari currency + pronunciation rewrites only for
    Devanagari-script languages (Hindi, Marathi, …). For other scripts (Telugu,
    Malayalam) injecting Devanagari would render the wrong script, so the text is
    returned unchanged until per-language maps exist. An unknown/empty language
    keeps the legacy behaviour (apply — assumes Hindi).
    """
    if not text:
        return text
    base = (language or "").strip().lower().split("-")[0]
    if base and base not in DEVANAGARI_LANGS:
        return text
    return apply_pronunciations(normalize_currency(text), extra=extra)
