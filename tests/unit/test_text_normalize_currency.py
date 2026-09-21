import logging

from src.pipeline.text_normalize import normalize_currency, normalize_for_tts


def test_rupee_symbol():
    assert normalize_currency("आप सिर्फ ₹100 से शुरू करें") == "आप सिर्फ 100 रुपये से शुरू करें"


def test_rupee_symbol_with_space():
    assert normalize_currency("₹ 500 का बोनस") == "500 रुपये का बोनस"


def test_rs_word_and_dot():
    assert normalize_currency("Rs 100 se shuru") == "100 रुपये se shuru"
    assert normalize_currency("Rs. 250") == "250 रुपये"


def test_commas_stripped():
    assert normalize_currency("₹1,000 deposit") == "1000 रुपये deposit"


def test_spelled_out_untouched():
    # Already-spoken forms must not be altered.
    assert normalize_currency("100 रुपये से शुरू") == "100 रुपये से शुरू"
    assert normalize_currency("कोई amount नहीं") == "कोई amount नहीं"


def test_empty():
    assert normalize_currency("") == ""


# --- decimal / paise handling (behaviour table) ----------------------------


def test_no_decimal_unchanged_from_today():
    assert normalize_currency("Rs 100") == "100 रुपये"


def test_two_fractional_digits_become_paise():
    # "Rs 99.50" must not strand ".50" after the rupee word -- both parts
    # need to be spoken, so the fraction becomes a separate "50 पैसे".
    assert normalize_currency("Rs 99.50") == "99 रुपये 50 पैसे"


def test_single_fractional_digit_is_padded_right_not_treated_as_units():
    # "Rs 99.5" means ninety-nine rupees and FIFTY paise (99.50), not five
    # paise -- a single decimal digit is a tenths digit, so it must be padded
    # on the right ("5" -> "50"), never on the left ("5" -> "05").
    assert normalize_currency("Rs 99.5") == "99 रुपये 50 पैसे"


def test_all_zero_fraction_is_dropped_not_spoken_as_zero_paise():
    # "Rs 1,234.00" has no paise at all -- saying "0 पैसे" would be noise a
    # human speaker would never actually say, so the fraction is dropped.
    assert normalize_currency("Rs 1,234.00") == "1234 रुपये"


def test_zero_rupees_omits_the_rupee_part_entirely():
    # "Rs 0.50" is fifty paise, full stop -- nobody says "0 रुपये 50 पैसे".
    assert normalize_currency("Rs 0.50") == "50 पैसे"


def test_zero_rupees_and_zero_paise():
    assert normalize_currency("Rs 0.00") == "0 रुपये"


def test_three_or_more_fractional_digits_kept_whole_not_treated_as_paise():
    # Paise only ever has 1-2 digits (hundredths of a rupee). "Rs 99.567" is
    # not a paise amount, so instead of guessing, the number is kept whole
    # with "रुपये" after it -- avoiding both a wrong paise value and a
    # stranded fragment.
    assert normalize_currency("Rs 99.567") == "99.567 रुपये"


# The ₹ symbol is the spelling that actually arrives from the LLM in
# practice -- every case above uses "Rs", so the paise behaviour needs its
# own coverage on the ₹ spelling too, not just an assumption that both
# spellings share a code path.
def test_rupee_symbol_two_fractional_digits_become_paise():
    assert normalize_currency("₹99.50") == "99 रुपये 50 पैसे"


def test_rupee_symbol_all_zero_fraction_is_dropped():
    assert normalize_currency("₹1,234.00") == "1234 रुपये"


def test_rupee_symbol_zero_rupees_omits_the_rupee_part():
    assert normalize_currency("₹0.50") == "50 पैसे"


def test_rupee_symbol_three_or_more_fractional_digits_kept_whole():
    assert normalize_currency("₹100.567") == "100.567 रुपये"


def test_sentence_terminating_period_is_not_swallowed_as_a_decimal_point():
    # This is the critical regression case: "Rs 100." followed by a space and
    # a new sentence is a full stop, NOT a decimal point. A naive `\.\d*`
    # would eat that dot and merge "Aapka agla kadam" into the currency
    # match's tail. The decimal group must require a digit immediately after
    # the dot with no space, so it simply fails to match here and the
    # sentence boundary survives untouched.
    assert normalize_currency("Rs 100. Aapka agla kadam") == "100 रुपये. Aapka agla kadam"


def test_scale_word_puts_currency_after_the_scale_word():
    # "1.5 lakh" cannot be rewritten to paise -- a paise breakdown is
    # meaningless against a lakh multiplier ("1 रुपये 50 पैसे lakh" would be
    # nonsense). Number, scale word, currency is the natural spoken order in
    # both Hindi and English, so "रुपये" moves to after the (untranslated)
    # scale word instead, and the fraction stays attached to the number as a
    # decimal rather than becoming paise.
    assert normalize_currency("Rs 1.5 lakh") == "1.5 lakh रुपये"


def test_scale_word_variants_all_rewritten_with_currency_after():
    for phrase, expected in (
        ("₹2 crore ka target", "2 crore रुपये ka target"),
        ("Rs 10 lakhs tak", "10 lakhs रुपये tak"),
        ("Rs 5 crores ka fund", "5 crores रुपये ka fund"),
        ("₹1 million ka fund", "1 million रुपये ka fund"),
        ("Rs 3 करोड़", "3 करोड़ रुपये"),
        # The scale word is carried through verbatim, case and script
        # untouched -- "Crore" is not normalised to "crore".
        ("Rs 2 Crore ka bonus", "2 Crore रुपये ka bonus"),
    ):
        assert normalize_currency(phrase) == expected


def test_scale_word_gap_words_thousand_hazaar():
    # These romanised scale words were missing from the guard entirely, so
    # they used to fall through to the normal (non-scale) rewrite and get
    # garbled: "Rs 50 thousand" -> "50 रुपये thousand". They must now be
    # recognised the same way lakh/crore already are.
    for phrase, expected in (
        ("Rs 50 thousand", "50 thousand रुपये"),
        ("Rs 20 thousands baad", "20 thousands रुपये baad"),
        ("Rs 50 hazaar", "50 hazaar रुपये"),
        ("Rs 20 hazaars milenge", "20 hazaars रुपये milenge"),
        ("Rs 5 hazar", "5 hazar रुपये"),
    ):
        assert normalize_currency(phrase) == expected


def test_arab_is_not_treated_as_a_scale_word():
    # "arab" (100 crore) was removed from the scale-word list: as an English
    # scale word it's vanishingly rare in this product's copy, while "Arab"
    # as an ordinary proper adjective ("Arab countries") is common. The
    # normal (non-scale) rewrite applies and "Arab" is left as ordinary
    # surrounding text.
    assert normalize_currency("Rs 500 Arab countries mein") == "500 रुपये Arab countries mein"
    assert normalize_currency("₹5 arab ka investment") == "5 रुपये arab ka investment"
    assert normalize_currency("Rs 2 arabs") == "2 रुपये arabs"


def test_scale_word_does_not_match_bare_k_or_m():
    # Bare "k"/"m" abbreviations are deliberately not treated as scale
    # words -- too ambiguous against ordinary single letters/words -- so
    # these fall through to the normal rewrite untouched by the scale path.
    assert normalize_currency("Rs 50k") == "50 रुपयेk"
    assert normalize_currency("Rs 50 m") == "50 रुपये m"


def test_scale_word_inflected_devanagari_forms_fall_through_correctly():
    # लाखों/करोड़ों/हजारों are inflected (plural) forms, not the bare scale
    # word the guard matches on -- they must NOT be treated as a scale-word
    # match (which would incorrectly fire on just the "लाख"/"करोड़"/"हजार"
    # stem and strand the rest of the suffix, e.g. "5 लाख रुपयेों"). The
    # normal (non-scale) currency rewrite applies instead, leaving the
    # inflected word itself untouched.
    assert normalize_currency("Rs 5 लाखों") == "5 रुपये लाखों"
    assert normalize_currency("Rs 5 करोड़ों") == "5 रुपये करोड़ों"
    assert normalize_currency("Rs 5 हजारों") == "5 रुपये हजारों"


def test_scale_word_rupee_symbol_with_devanagari_scale_word():
    # Both currency spellings (₹ and Rs) combined with a Devanagari scale
    # word, decimal amount included -- the fraction stays attached to the
    # number as a decimal, not converted to paise.
    assert normalize_currency("₹2 करोड़") == "2 करोड़ रुपये"
    assert normalize_currency("Rs 50 लाख") == "50 लाख रुपये"
    assert normalize_currency("₹1.5 लाख ka fund") == "1.5 लाख रुपये ka fund"


def test_scale_word_must_be_immediately_adjacent():
    # "lakh" appears in the sentence but not right after the number -- this
    # must NOT be treated as a scale-word match, so the normal rewrite
    # applies to the amount and "lakh" is left as ordinary surrounding text.
    assert normalize_currency("Rs 100 par lakh baad mein") == "100 रुपये par lakh baad mein"


def test_compound_scale_chain_gets_currency_word_at_the_end():
    # Indian amounts are routinely written as a chain of scale words ("1
    # lakh 50 hazaar" = 150,000). रुपये must land after the WHOLE chain, not
    # spliced in after just the first word -- "1 lakh रुपये 50 hazaar" would
    # be a materially different (and wrong) amount when spoken.
    for phrase, expected in (
        ("Rs 1 lakh 50 hazaar ka prize", "1 lakh 50 hazaar रुपये ka prize"),
        ("₹2 करोड़ 50 लाख", "2 करोड़ 50 लाख रुपये"),
        ("Rs 5 crore 25 lakh ka jackpot", "5 crore 25 lakh रुपये ka jackpot"),
    ):
        assert normalize_currency(phrase) == expected


def test_compound_scale_chain_with_redundant_trailing_currency_word():
    # The dedup must still fire after the whole chain, not just after the
    # first scale word.
    assert normalize_currency("₹1 lakh 20 hazaar rupaye") == "1 lakh 20 hazaar रुपये"


def test_compound_scale_chain_stops_when_second_number_has_no_scale_word():
    # "50" here is not followed by a scale word, so it does not extend the
    # chain -- the chain is just "1 lakh", and "50 rupees" is left as
    # ordinary trailing text (untouched by the dedup, since it isn't
    # immediately adjacent to the chain's end).
    assert normalize_currency("Rs 1 lakh 50 rupees") == "1 lakh रुपये 50 rupees"


def test_single_scale_word_still_works_after_chain_support_added():
    # Single scale words (no chain) must keep working exactly as before.
    assert normalize_currency("₹2 crore") == "2 crore रुपये"


def test_multiple_amounts_mixed_decimal_and_not():
    # Two amounts in one string, one with paise and one a clean whole number
    # followed by a sentence-terminating period -- both must be handled
    # independently and correctly in the same pass.
    assert (
        normalize_currency("Rs 100.50 aur Rs 200. Phir")
        == "100 रुपये 50 पैसे aur 200 रुपये. Phir"
    )


def test_rs_dot_prefix_with_decimal_amount():
    # "Rs." (prefix dot) immediately followed by a decimal amount -- two dots
    # in play, only the second one is a decimal point.
    assert normalize_currency("Rs.250.75") == "250 रुपये 75 पैसे"


# --- normalize_for_tts still routes the decimal fix correctly --------------


def test_normalize_for_tts_applies_paise_rewrite_for_devanagari_language():
    # A Devanagari-script language (Hindi here) must still route through
    # normalize_currency, so the paise fix actually reaches real calls and
    # isn't only reachable by calling normalize_currency directly.
    out = normalize_for_tts("Rs 99.50 mein milega", "hi")
    assert out == "99 रुपये 50 पैसे mein milega"


def test_normalize_for_tts_leaves_non_devanagari_language_untouched():
    # Telugu (and other non-Devanagari scripts) must not get the Devanagari
    # currency rewrite at all -- decimal or not, the text passes through
    # verbatim until a per-language normalizer exists.
    text = "Rs 99.50 mein milega"
    assert normalize_for_tts(text, "te-IN") == text


# --- DEBUG-only instrumentation: silent at INFO, named+shaped at DEBUG -----


def test_currency_tracking_silent_at_info_named_and_shaped_at_debug(caplog):
    # One input that produces both a normal paise rewrite and a scale-word
    # rewrite in the same call, to pin both DEBUG events at once: this must
    # do zero tracking work and emit nothing at INFO (DEBUG-only
    # instrumentation has zero cost -- and zero log noise -- when DEBUG is
    # off), and both events must fire at DEBUG under their expected names
    # and carry their documented fields.
    text = "Rs 99.50 aur Rs 5 lakh milega"

    with caplog.at_level(logging.INFO):
        normalize_currency(text)
    assert not any(r.levelno >= logging.INFO for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        result = normalize_currency(text)

    rewritten = [r for r in caplog.records if r.getMessage() == "tts_normalize currency rewritten"]
    scaled = [r for r in caplog.records if r.getMessage() == "tts_normalize currency scaled"]

    assert len(rewritten) == 1
    assert rewritten[0].matches == ["Rs 99.50"]
    assert rewritten[0].before == text
    assert rewritten[0].after == result

    assert len(scaled) == 1
    assert scaled[0].scaled == [
        {"matched": "Rs 5 lakh", "scale_word": "lakh", "result": "5 lakh रुपये"}
    ]
    assert scaled[0].before == text
    assert scaled[0].after == result


def test_debug_fields_carry_the_full_consumed_span_including_dedup(caplog):
    # matched/matches used to carry only the currency-regex span, silently
    # dropping the trailing spelled-out currency word the dedup also
    # deleted -- an operator asking "where did the customer's रुपये go" got
    # nothing from this field. Both fields must now carry the FULL consumed
    # source span.
    with caplog.at_level(logging.DEBUG):
        normalize_currency("Rs 2 लाख रुपये")
        normalize_currency("Rs 99.50 रुपयों में")

    scaled_events = [r for r in caplog.records if r.getMessage() == "tts_normalize currency scaled"]
    rewritten_events = [r for r in caplog.records if r.getMessage() == "tts_normalize currency rewritten"]

    assert scaled_events[0].scaled == [
        {"matched": "Rs 2 लाख रुपये", "scale_word": "लाख", "result": "2 लाख रुपये"}
    ]
    assert rewritten_events[0].matches == ["Rs 99.50 रुपयों"]


# --- A currency word already present after the amount ----------------------
#
# Both rewrite paths end in "रुपये", so an amount the model already spelled
# out ("₹100 रुपये") would otherwise gain a second one and TTS would say
# "sau rupaye rupaye". This pre-dates the decimal work -- HEAD turns
# "₹100 रुपये" into "100 रुपये रुपये" as well -- so these pin a fix, not a
# regression guard.


def test_redundant_currency_word_is_not_doubled():
    assert normalize_currency("₹100 रुपये") == "100 रुपये"
    assert normalize_currency("₹100 रुपये से शुरू") == "100 रुपये से शुरू"


def test_redundant_currency_word_after_a_scale_word():
    """The scale path ends in रुपये too, so it needs the same treatment."""
    assert normalize_currency("Rs 2 लाख रुपये") == "2 लाख रुपये"
    assert normalize_currency("Rs 2 crore रुपये") == "2 crore रुपये"


def test_redundant_currency_word_with_a_paise_amount():
    """The rupee word sits mid-string here ("99 रुपये 50 पैसे"), so the
    source's own trailing word is swallowed rather than ours suppressed."""
    assert normalize_currency("₹99.50 रुपये") == "99 रुपये 50 पैसे"


def test_redundant_currency_word_spelling_variants():
    """Whichever spelling the model reached for, one canonical form survives."""
    for written in ("रुपये", "रुपए", "रुपया", "रु", "rupees", "rupaye"):
        assert normalize_currency(f"₹100 {written}") == "100 रुपये", written


def test_spelled_out_amount_with_no_symbol_is_still_untouched():
    """The dedup must not give the function a reason to touch text that has
    no ₹/Rs marker at all -- that is the case it has always left alone."""
    assert normalize_currency("100 रुपये से शुरू") == "100 रुपये से शुरू"
    assert normalize_currency("कुछ रुपये बचे हैं") == "कुछ रुपये बचे हैं"


def test_redundant_currency_word_newly_added_spelling_variants():
    """Misspellings/variants an LLM actually emits, newly added to the dedup
    list: the long-ū रूपये/रूपए, रुपइया/रुपैया, rupaiya, and INR. Each must
    be recognised and deduped just like the pre-existing spellings, not
    doubled ("100 रुपये रूपये")."""
    for written in ("रूपये", "रूपए", "रुपइया", "रुपैया", "rupaiya", "INR"):
        assert normalize_currency(f"₹100 {written}") == "100 रुपये", written
