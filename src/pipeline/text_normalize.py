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
import unicodedata

from src.utils.logging import debug_event

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
    # This runs on every sentence handed to TTS (every turn, often several
    # times), so the substitution expression below is written once and used
    # at both levels; only the tracking around it (the `substitutions` list
    # and the debug_event call) is built when DEBUG is actually on --
    # unguarded it would pay for a list nobody is reading on every live
    # call, forever (see docs/debug-logging.md "Cost when DEBUG is off").
    # A wrong substitution changes what the customer HEARS, so the thing
    # worth logging is the specific term matched and what it became. Keeping
    # one copy of the expression also means a future fix to it can't
    # diverge silently by log level -- the two branches used to each hold
    # their own copy of `lower[m.group(0).lower()]`.
    track = log.isEnabledFor(logging.DEBUG)
    substitutions: list[tuple[str, str]] = []

    def _sub(m: re.Match) -> str:
        replacement = lower[m.group(0).lower()]
        if track:
            substitutions.append((m.group(0), replacement))
        return replacement

    result = pattern.sub(_sub, text)
    if track and substitutions:
        debug_event(
            log, "tts_normalize pronunciation substituted",
            substitutions=substitutions, before=text, after=result,
        )
    return result


# Currency: Sarvam TTS doesn't vocalize the ₹ symbol or a bare "Rs", so amounts
# like "₹100" / "Rs 100" get dropped. Rewrite to spoken Hindi: "100 रुपये".
# The decimal group requires digits immediately after the dot with no
# intervening space -- that's what keeps a sentence-terminating "." (e.g.
# "Rs 100. Aapka agla kadam") from being swallowed as a decimal point: when
# the character after the dot isn't a digit, the optional group simply fails
# to match and the dot is left untouched for the rest of the sentence.
_CURRENCY_RE = re.compile(r"(?:₹|\bRs\.?)\s*([\d][\d,]*)(\.\d+)?", re.IGNORECASE)

# Scale words that turn "1.5 lakh" into "1.5 lakh रुपये", not "1 रुपये 50
# पैसे lakh" -- a paise breakdown is meaningless against a lakh/crore
# multiplier, so a match immediately followed by one of these is rendered as
# "number, scale word (untouched), रुपये" instead of going through the normal
# paise path. See ``_format_currency_scaled``.
#
# Trailing boundary is a negative lookahead for a word character rather than
# `\b`: a couple of the Devanagari words end in a combining nukta mark
# (करोड़ = क+र+ो+ड+़), and Python's `\w` doesn't count combining marks as word
# characters -- so `\b` right after one sees "non-word, non-word" (the nukta,
# then end-of-string/whitespace) and reports no boundary at all, silently
# failing to match. That same gap cuts the other way for inflected
# Devanagari forms (लाखों, करोड़ों, हजारों): the plural suffix's first
# character is *also* a combining vowel sign (Mc), not a `\w` character, so
# `(?!\w)` alone would falsely accept a boundary right after the bare word
# and match only its stem -- see ``_scale_word_match``, which adds the check
# `(?!\w)` can't express, so the inflected forms correctly fall through to
# the normal (non-scale) rewrite instead of having their suffix truncated.
#
# `arab`/`arabs` (100 crore) deliberately excluded: as an English scale word
# it's vanishingly rare in this product's copy, while "Arab" as an ordinary
# proper adjective ("Arab countries") is common -- so including it produced a
# false positive ("Rs 500 Arab countries mein" -> "500 Arab रुपये countries
# mein") far more often than it correctly caught a real scale word.
_CURRENCY_SCALE_WORD_RE = re.compile(
    r"\s*(lakh|lakhs|lac|lacs|crore|crores|million|billion|"
    r"thousand|thousands|hazaar|hazaars|hazar|"
    r"हज़ार|हजार|लाख|करोड़)(?!\w)",
    re.IGNORECASE,
)


# A spelled-out currency word already sitting after the amount. "₹100 रुपये"
# and "Rs 2 लाख रुपये" are redundant but common in LLM-written Hindi, and
# without this the rewrite appends a second one -- TTS then says "sau rupaye
# rupaye". Longest alternatives first so no alternative is shadowed by a
# shorter prefix of itself -- in particular "रु\.?" has to sort last among
# the Devanagari spellings since it's a literal prefix of all the others.
# Includes the long-ū spellings (रूपये, रूपए) and other misspellings/variants
# an LLM actually emits (रुपइया, रुपैया, rupaiya, INR), alongside the
# standard ones. Pre-dates the decimal fix (HEAD turns "₹100 रुपये" into
# "100 रुपये रुपये" too); the amount is consumed along with the word so the
# formatter's own canonical form is the only one that survives.
_TRAILING_CURRENCY_WORD_RE = re.compile(
    r"\s*(?:रुपयों|रुपैया|रुपइया|रुपये|रूपये|रुपया|रुपए|रूपए|रु\.?|"
    r"rupaiya|rupees|rupaye|rupya|INR)(?!\w)",
    re.IGNORECASE,
)


def _scale_word_match(text: str, pos: int) -> re.Match | None:
    r"""Match ``_CURRENCY_SCALE_WORD_RE`` at ``pos``, additionally rejecting a
    match immediately followed by a Unicode combining mark (category
    ``Mn``/``Mc``).

    Without this, an inflected form like "लाखों" (लाख + the plural vowel
    sign ों) would match on just the "लाख" stem: the regex's `(?!\w)`
    boundary checks for a *word* character, but the first character of "ों"
    is a combining vowel sign, which Python's `\w` doesn't count as one
    either. Left unchecked, that false "boundary" would fire the scale-word
    rewrite on the stem and strand the rest of the suffix in the output
    (e.g. "5 लाख रुपयेों"). Checking the Unicode category directly catches
    what `(?!\w)` structurally cannot.
    """
    m = _CURRENCY_SCALE_WORD_RE.match(text, pos)
    if m is None:
        return None
    end = m.end()
    if end < len(text) and unicodedata.category(text[end])[0] == "M":
        return None
    return m


# A further ``<number>`` group continuing a scale-word chain (see
# ``_scale_chain_end``), e.g. the " 50" in "1 lakh 50 hazaar". Requires
# actual whitespace before the digits -- a scale word running directly into
# digits with no separator isn't how these are ever written.
_CHAIN_NUMBER_RE = re.compile(r"\s+([\d][\d,]*)")


def _scale_chain_end(text: str, pos: int) -> int | None:
    """Find the end offset of the *whole* chain of ``<number> <scale word>``
    groups starting at ``pos`` (the position right after the currency
    amount's own number). Returns ``None`` when there's no scale word at
    ``pos`` at all -- i.e. the scale path doesn't apply here.

    Indian amounts are routinely written as a chain ("1 lakh 50 hazaar" =
    150,000, not two separate amounts), so once the first scale word
    matches, this keeps consuming further ``<number> <scale word>`` groups
    for as long as they appear immediately in sequence. It stops as soon as
    a number isn't immediately followed by a recognised scale word -- that
    number (and everything after it) is left as ordinary surrounding text
    instead, e.g. in "1 lakh 50 rupees" the trailing "50" has no scale word
    after it, so the chain is just "1 lakh".

    Each group boundary is checked with ``_scale_word_match``, so the
    combining-mark rejection it documents (rejecting an inflected form like
    "लाखों") is applied at whichever group turns out to be the chain's
    actual last one, rather than only at the first.
    """
    match = _scale_word_match(text, pos)
    if match is None:
        return None
    end = match.end()
    while True:
        number = _CHAIN_NUMBER_RE.match(text, end)
        if number is None:
            break
        next_match = _scale_word_match(text, number.end())
        if next_match is None:
            break
        end = next_match.end()
    return end


def _consume_trailing_currency_word(text: str, end: int) -> int:
    """Advance ``end`` past a spelled-out currency word already following the
    amount, so the rewrite does not add a second one.

    Both rewrite paths end in ``रुपये``, so "₹100 रुपये" would otherwise come
    out as "100 रुपये रुपये". Swallowing the source's own word rather than
    suppressing ours keeps one canonical spelling in the output whichever of
    the several spellings the model wrote.

    This match's boundary is the same ``(?!\\w)`` that ``_scale_word_match``
    documents as insufficient for a word immediately followed by a
    Devanagari combining mark (category Mn/Mc) -- the identical gap exists
    here: "₹100 रुपयें" (रुपये + anusvara) matches only the "रुपये" prefix
    and strands the anusvara. Unlike the scale path, that gap is NOT closed
    here: adding the same category-M rejection was tried and reverted, since
    it changes -- not just fails to improve -- that exact input. Rejecting
    the match leaves the *whole* trailing word (mark included) unconsumed
    rather than falling back to the shorter safe match, so "₹100 रुपयें"
    goes from "100 रुपयें" (the stranded mark harmlessly reattaches to our
    emitted रुपये) to "100 रुपये रुपयें" (a doubled currency word) -- worse,
    not better. So this stays the same latent gap ``_scale_word_match``
    documents, left unguarded here on purpose.
    """
    trailing = _TRAILING_CURRENCY_WORD_RE.match(text, end)
    return trailing.end() if trailing else end


def _format_currency_amount(rupees_group: str, frac_group: str | None) -> str:
    """Render a captured ``(rupees, .frac)`` pair as spoken Hindi.

    ``frac_group`` includes the leading dot (e.g. ``".50"``) or is ``None``
    when no decimal part was captured at all.
    """
    rupees_str = rupees_group.replace(",", "")
    if frac_group is None:
        return f"{rupees_str} रुपये"
    frac_digits = frac_group[1:]
    if len(frac_digits) >= 3:
        # Not a paise amount (paise only ever has 1-2 digits) -- keep the
        # number whole with the word after it so nothing is stranded, e.g.
        # "99.567" -> "99.567 रुपये".
        return f"{rupees_str}.{frac_digits} रुपये"
    # 1-2 fractional digits are paise. "₹99.5" means 50 paise, not 5, so a
    # single digit is padded on the right (not the left) to two digits.
    paise = int(frac_digits.ljust(2, "0"))
    rupees_zero = int(rupees_str) == 0
    paise_zero = paise == 0
    if rupees_zero and paise_zero:
        return "0 रुपये"
    if paise_zero:
        # All-zero fraction (".00") -- drop it rather than say "0 पैसे".
        return f"{rupees_str} रुपये"
    if rupees_zero:
        # Zero rupees -- omit the rupee part entirely.
        return f"{paise} पैसे"
    return f"{rupees_str} रुपये {paise} पैसे"


def _format_currency_scaled(
    rupees_group: str, frac_group: str | None, chain_text: str
) -> str:
    """Render a currency amount immediately followed by a chain of one or
    more scale words (``lakh``, ``crore``, ...) as ``number, chain, रुपये``
    -- the natural spoken order in both Hindi and English.

    ``chain_text`` is the raw source text spanning the *whole* scale-word
    chain (from ``_scale_chain_end``) -- e.g. " lakh 50 hazaar" for "1 lakh
    50 hazaar". Taking it verbatim, rather than reconstructing it piece by
    piece, is what preserves every scale word's original whitespace, case
    and script (translating them is out of scope, and unneeded --
    ``crore``/``करोड़`` etc. are already commonly spoken mid-sentence in
    Hindi) and what puts ``रुपये`` after the *entire* chain instead of
    splicing it in after just the first word ("1 lakh 50 hazaar रुपये", not
    "1 lakh रुपये 50 hazaar"). The fractional part, if any, stays attached
    to the number as a decimal rather than being converted to paise --
    paise are meaningless against a lakh/crore multiplier ("1.5 lakh
    रुपये", not "1 रुपये 50 पैसे lakh").
    """
    rupees_str = rupees_group.replace(",", "")
    number = rupees_str if frac_group is None else f"{rupees_str}{frac_group}"
    return f"{number}{chain_text} रुपये"


def normalize_currency(text: str) -> str:
    """Rewrite ``₹100`` / ``Rs 100`` / ``Rs. 1,000`` to ``100 रुपये`` so the
    amount is actually spoken. Spelled-out forms (``100 रुपये``) are untouched.

    Decimal amounts are rewritten to paise (``Rs 99.50`` -> ``99 रुपये 50
    पैसे``); see ``_format_currency_amount`` for the exact rules. A match
    immediately followed by a scale word (``lakh``, ``crore``, ...) is instead
    rewritten to ``number, scale word(s), रुपये`` (``Rs 1.5 lakh`` -> ``1.5
    lakh रुपये``), and a whole chain of them is consumed together
    (``Rs 1 lakh 50 hazaar`` -> ``1 lakh 50 hazaar रुपये``, not रुपये spliced
    in after just the first word) -- see ``_format_currency_scaled``,
    ``_scale_chain_end`` and ``_CURRENCY_SCALE_WORD_RE``.
    """
    if not text:
        return text
    # Same reasoning as apply_pronunciations above: this runs on every
    # sentence bound for TTS, so the substitution expressions are written
    # once and used at both levels; only the match-list tracking is built
    # when DEBUG is actually on.
    #
    # This can't be done with `_CURRENCY_RE.sub(...)` the way the plain-match
    # path is: a scale-word match consumes text beyond `m.end()` (the scale
    # word itself), which a `.sub()` callback has no way to also delete from
    # the source -- it can only replace what the pattern itself matched. So
    # matches are walked by hand and the output is assembled by hand too,
    # advancing past the scale word's span when one was consumed.
    track = log.isEnabledFor(logging.DEBUG)
    matches: list[str] = []
    scaled: list[dict[str, str]] = []
    parts: list[str] = []
    last_end = 0

    for m in _CURRENCY_RE.finditer(text):
        if m.start() < last_end:
            continue
        parts.append(text[last_end:m.start()])
        chain_end = _scale_chain_end(text, m.end())
        if chain_end is not None:
            chain_text = text[m.end():chain_end]
            replacement = _format_currency_scaled(m.group(1), m.group(2), chain_text)
            parts.append(replacement)
            last_end = _consume_trailing_currency_word(text, chain_end)
            if track:
                # The DEBUG-only fields below carry the FULL consumed source
                # span (currency + whole scale chain + any redundant
                # trailing currency word swallowed by the dedup), not just
                # the currency-regex span -- see _consume_trailing_currency_word's
                # caller-side reasoning: an operator asking "where did the
                # customer's text go" needs everything this rewrite deleted,
                # not only the first amount.
                scaled.append({
                    "matched": text[m.start():last_end],
                    "scale_word": _scale_word_match(text, m.end()).group(1),
                    "result": replacement,
                })
        else:
            replacement = _format_currency_amount(m.group(1), m.group(2))
            parts.append(replacement)
            last_end = _consume_trailing_currency_word(text, m.end())
            if track:
                matches.append(text[m.start():last_end])

    parts.append(text[last_end:])
    result = "".join(parts)
    if track and matches:
        debug_event(
            log, "tts_normalize currency rewritten",
            matches=matches, before=text, after=result,
        )
    if track and scaled:
        debug_event(
            log, "tts_normalize currency scaled",
            scaled=scaled, before=text, after=result,
        )
    return result


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
