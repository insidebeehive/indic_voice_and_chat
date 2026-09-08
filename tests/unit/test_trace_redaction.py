"""Unit tests for src/observability/trace_redaction.py -- the PII-redaction
boundary for LLM conversation tracing (Phase 0 of
docs/superpowers/plans/2026-09-07-llm-conversation-tracing.md).

Standalone: this module is not wired into any call site yet (that's a later
phase), so these tests exercise it in isolation against realistic
betting-platform payload shapes taken from docs/crm-api-contract.md (wallet
balances, transactions, KYC/profile bank+UPI details,
get_player_latest_deposit_order's PgsOrderId, matka bid data) rather than
against any live request path.

Includes regression coverage for an independent review pass that found real
bugs in an earlier version of this module: a ReDoS in the email/UPI
patterns, a type-level fail-open gap in the recursive walk, and an
exact-match-only deny-list that missed realistic key spellings. See the
module docstring's "Design notes" for the fixes; the tests below pin them.
"""

import time
from datetime import datetime
from decimal import Decimal

import pytest

from src.observability.trace_redaction import (
    REDACTED_KEY_PLACEHOLDER,
    REDACTION_FAILED_PLACEHOLDER,
    redact_structure,
    redact_text,
    redaction_post_condition_ok,
)


def _flatten_to_text(value: object) -> str:
    """Render a redacted structure to one string so post-condition checks
    can be run over the whole thing at once, the same way a human skimming
    an exported trace would see it."""
    return repr(value)


# ---------------------------------------------------------------------------
# Individual pattern tests -- each one in isolation.
# ---------------------------------------------------------------------------


class TestIndividualPatterns:
    def test_uuid(self):
        assert redact_text("id 6c1a77a6-1234-4abc-9def-8add58aba9ef here") == (
            "id <UUID> here"
        )

    def test_email(self):
        assert redact_text("contact player@example.com now") == (
            "contact <EMAIL> now"
        )

    def test_upi_vpa(self):
        # Non-email-shaped: local@domain with no dot in the domain part.
        assert redact_text("pay to ram@paytm please") == "pay to <UPI> please"
        assert redact_text("upi: player@upi") == "upi: <UPI>"

    def test_indian_mobile_number(self):
        assert redact_text("call 9876543210 now") == "call <PHONE> now"
        assert redact_text("call +91-98765-43210 now") == "call <PHONE> now"

    def test_currency_amount(self):
        assert redact_text("balance is ₹4,250.75 today") == (
            "balance is <AMOUNT> today"
        )
        assert redact_text("paid Rs. 1500.00 yesterday") == "paid <AMOUNT> yesterday"
        assert redact_text("limit INR 50,000 monthly") == "limit <AMOUNT> monthly"

    def test_ifsc_code(self):
        assert redact_text("ifsc HDFC0001234 branch") == "ifsc <IFSC> branch"

    def test_pan(self):
        assert redact_text("pan ABCDE1234F on file") == "pan <PAN> on file"

    def test_aadhaar_shaped(self):
        assert redact_text("aadhaar 1234 5678 9012 verified") == (
            "aadhaar <AADHAAR> verified"
        )
        assert redact_text("aadhaar 1234-5678-9012 verified") == (
            "aadhaar <AADHAAR> verified"
        )
        assert redact_text("aadhaar 123456789012 verified") == (
            "aadhaar <AADHAAR> verified"
        )

    def test_generic_digit_run_catch_all(self):
        # A digit run with no more specific shape (not a phone/aadhaar/etc
        # length or format) still gets swept by the blunt >= 4 catch-all.
        assert redact_text("order ref 778899 pending") == "order ref <NUM> pending"
        assert redact_text("code 12345678 issued") == "code <NUM> issued"
        # Below the >= 4 threshold: left alone.
        assert redact_text("only 123 left") == "only 123 left"

    def test_date(self):
        assert redact_text("placed at 2026-06-20T10:30:00Z today") == (
            "placed at <DATE> today"
        )
        assert redact_text("on 2026-06-20 it happened") == "on <DATE> it happened"
        assert redact_text("due 21/06/2026 sharp") == "due <DATE> sharp"

    def test_specific_pattern_wins_over_catch_all(self):
        # A phone number and a currency amount both contain digit runs
        # >= 4, but must come out tagged with their specific placeholder,
        # not the blunt <NUM>.
        text = redact_text("mobile 9876543210 amount ₹4,250.75")
        assert "<PHONE>" in text
        assert "<AMOUNT>" in text
        assert "<NUM>" not in text


# ---------------------------------------------------------------------------
# Fix 6 (review) -- currency pattern must not eat word tails.
# ---------------------------------------------------------------------------


class TestCurrencyWordBoundaryFix:
    def test_rs_does_not_match_mid_word(self):
        # Before the fix: the case-insensitive "Rs" alternative matched the
        # "rs" inside "hours", producing "hou<AMOUNT> logged" -- eating half
        # of an unrelated word. The digit run is still swept by the
        # catch-all regardless, so the number is still gone either way;
        # this test is specifically about NOT mangling "hours".
        result = redact_text("hours 2400 logged")
        assert result == "hours <NUM> logged"
        assert "hou<AMOUNT>" not in result

    def test_rs_still_matches_as_a_real_prefix(self):
        assert redact_text("cost Rs2400 total") == "cost <AMOUNT> total"


# ---------------------------------------------------------------------------
# Fix 7 (review) -- UUID hex-boundary lookarounds instead of \b.
# ---------------------------------------------------------------------------


class TestUuidHexBoundaryFix:
    def test_uuid_glued_to_preceding_non_hex_letter_is_caught(self):
        # "r" (from "user") is not a hex character, so the new
        # (?<![0-9a-fA-F]) lookbehind allows the match to start here, where
        # the old \b-anchored pattern would not (no transition between two
        # \w characters).
        result = redact_text(
            "user550e8400-e29b-41d4-a716-446655440000 tail"
        )
        assert result == "user<UUID> tail"

    def test_uuid_glued_to_following_non_hex_char_is_caught(self):
        result = redact_text(
            "id:550e8400-e29b-41d4-a716-446655440000end"
        )
        # "d" from "id:" is not adjacent (there's a colon), and "end"
        # starts with "e" which IS hex-valid -- ambiguous on the right, so
        # per the documented limitation this one does not fully collapse
        # to <UUID>, but MUST NOT leave a bare full UUID string exposed
        # either.
        assert "550e8400-e29b-41d4-a716-446655440000" not in result

    def test_fully_delimited_uuid_still_matches(self):
        # Sanity: the common, unambiguous case is unaffected by the
        # boundary change.
        assert redact_text("id 550e8400-e29b-41d4-a716-446655440000 done") == (
            "id <UUID> done"
        )

    def test_uuid_embedded_in_hyphen_joined_identifier_is_caught(self):
        # Round-2 fix (regression from round-1's first attempt): the
        # trailing lookahead used to also exclude "-", which broke the very
        # common case of a UUID embedded in an ordinary hyphen-joined
        # identifier -- a trailing hyphen right after the last hex digit is
        # exactly what a correctly-terminated UUID looks like here, not a
        # sign of ambiguity.
        assert redact_text("order-550e8400-e29b-41d4-a716-446655440000-v2") == (
            "order-<UUID>-v2"
        )
        assert redact_text("trace-550e8400-e29b-41d4-a716-446655440000-span") == (
            "trace-<UUID>-span"
        )


# ---------------------------------------------------------------------------
# Fix 1 (review, HIGH) -- ReDoS in the email/UPI patterns.
# ---------------------------------------------------------------------------


class TestReDoSFix:
    def test_adversarial_email_shaped_text_completes_quickly(self):
        # Before the fix (unbounded quantifiers in _EMAIL_RE/_UPI_RE): this
        # exact shape -- local@domain with many dots and no valid trailing
        # TLD -- measured ~38s of blocking CPU on 100KB of input via
        # catastrophic backtracking. redact_text is specified (per the
        # plan) to eventually run on attacker-influenced user-message text,
        # so this is a real availability bug, not a theoretical one.
        adversarial = "a@" + "a." * 20000  # ~40KB
        start = time.perf_counter()
        redact_text(adversarial)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, (
            f"adversarial email-shaped input took {elapsed:.3f}s -- "
            "possible ReDoS regression in _EMAIL_RE/_UPI_RE"
        )

    def test_adversarial_upi_shaped_text_completes_quickly(self):
        adversarial = "a" * 20000 + "@" + "a" * 20000
        start = time.perf_counter()
        redact_text(adversarial)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, (
            f"adversarial UPI-shaped input took {elapsed:.3f}s -- "
            "possible ReDoS regression in _UPI_RE"
        )

    def test_large_realistic_text_still_completes_quickly(self):
        # A large but realistic block of text (not adversarial) should
        # also be fast -- guards against a fix that "solves" the
        # adversarial case by making the common case slow instead.
        text = "customer says mobile 9876543210 and amount Rs4250.75 " * 2000
        start = time.perf_counter()
        redact_text(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# Fix 2 (review, HIGH) -- type-level fail-open gap.
# ---------------------------------------------------------------------------


class TestTypeFallthroughFix:
    def test_set_does_not_leak_raw_content(self):
        payload = {"note": {"9876543210", "ram@paytm"}}
        result = redact_structure(payload)
        rendered = repr(result)
        assert "9876543210" not in rendered
        assert "ram@paytm" not in rendered
        assert "@" not in rendered
        assert result["note"] == REDACTED_KEY_PLACEHOLDER

    def test_bytes_does_not_leak_raw_content(self):
        payload = {"blob": b"phone 9876543210"}
        result = redact_structure(payload)
        assert result["blob"] == REDACTED_KEY_PLACEHOLDER
        assert b"9876543210" not in repr(result).encode()

    def test_bytearray_does_not_leak_raw_content(self):
        payload = {"blob": bytearray(b"otp 1234")}
        result = redact_structure(payload)
        assert result["blob"] == REDACTED_KEY_PLACEHOLDER

    def test_decimal_does_not_leak_raw_content(self):
        # A real shape: DB-numeric wallet balance fields commonly
        # deserialize as decimal.Decimal, not float.
        payload = {"wallet_amount": Decimal("4250.75")}
        result = redact_structure(payload)
        assert result["wallet_amount"] == REDACTED_KEY_PLACEHOLDER
        assert "4250" not in repr(result)

    def test_datetime_does_not_leak_raw_content(self):
        payload = {"event_time": datetime(2026, 6, 20, 10, 30, 0)}
        result = redact_structure(payload)
        assert result["event_time"] == REDACTED_KEY_PLACEHOLDER

    def test_custom_object_does_not_leak_its_dict(self):
        class _PlayerRecord:
            def __init__(self):
                self.mobile = "9876543210"
                self.email = "player@example.com"

            def __repr__(self):
                return f"_PlayerRecord(mobile={self.mobile!r}, email={self.email!r})"

        payload = {"record": _PlayerRecord()}
        result = redact_structure(payload)
        rendered = repr(result)
        assert result["record"] == REDACTED_KEY_PLACEHOLDER
        assert "9876543210" not in rendered
        assert "player@example.com" not in rendered


# ---------------------------------------------------------------------------
# Fix 4 (review, Medium) -- redact_text must fail closed, not silently
# downgrade, on non-str input.
# ---------------------------------------------------------------------------


class TestRedactTextNonStringInputFailsClosed:
    def test_dict_input_fails_closed(self):
        assert redact_text({"mobile": "9876543210"}) == REDACTION_FAILED_PLACEHOLDER

    def test_none_input_fails_closed(self):
        assert redact_text(None) == REDACTION_FAILED_PLACEHOLDER

    def test_list_input_fails_closed(self):
        assert redact_text(["9876543210"]) == REDACTION_FAILED_PLACEHOLDER


# ---------------------------------------------------------------------------
# Fix 5 (review, Medium) -- dict keys themselves must be pattern-scrubbed.
# ---------------------------------------------------------------------------


class TestDictKeyScrubbing:
    def test_phone_shaped_key_is_scrubbed(self):
        payload = {"9876543210": "some note about this player"}
        result = redact_structure(payload)
        assert "9876543210" not in result
        assert "<PHONE>" in result

    def test_uuid_shaped_key_is_scrubbed(self):
        payload = {"6c1a77a6-1234-4abc-9def-8add58aba9ef": "a note"}
        result = redact_structure(payload)
        assert "6c1a77a6-1234-4abc-9def-8add58aba9ef" not in result
        assert "<UUID>" in result

    def test_ordinary_key_is_unaffected(self):
        payload = {"status": "open"}
        result = redact_structure(payload)
        assert "status" in result
        assert result["status"] == "open"

    def test_int_key_is_scrubbed_not_passed_through_raw(self):
        # Round-2 fix (regression from round-1's dict-key fix): a
        # non-string key used to fall through an `else k` branch entirely
        # unscrubbed. An int key that is itself phone-shaped must not leak.
        payload = {9876543210: "some note about this player"}
        result = redact_structure(payload)
        rendered = repr(result)
        assert "9876543210" not in rendered
        assert "<PHONE>" in rendered

    def test_bytes_key_is_scrubbed_not_passed_through_raw(self):
        payload = {b"9876543210": "v"}
        result = redact_structure(payload)
        rendered = repr(result)
        assert "9876543210" not in rendered
        assert "<PHONE>" in rendered

    def test_tuple_key_is_scrubbed_not_passed_through_raw(self):
        payload = {("mobile", "9876543210"): "v"}
        result = redact_structure(payload)
        rendered = repr(result)
        assert "9876543210" not in rendered
        assert "<PHONE>" in rendered

    def test_scrubbed_key_collision_does_not_silently_drop_data(self):
        # Two distinct phone-number-shaped keys both scrub to "<PHONE>" --
        # the second must not silently overwrite the first entry.
        payload = {"9876543210": "first", "9123456780": "second"}
        result = redact_structure(payload)
        assert len(result) == 2
        assert set(result.values()) == {"first", "second"}
        assert "<PHONE>" in result
        assert any(k.startswith("<PHONE>~") for k in result if k != "<PHONE>")


# ---------------------------------------------------------------------------
# Fix 3 / Fix B (review, HIGH) -- deny-list must be TOKEN-aware matching,
# neither exact-match-only NOR unanchored substring matching.
# ---------------------------------------------------------------------------


_EXACT_MATCH_DENY_KEYS = [
    "mobile", "phone", "email", "upi", "vpa",
    "account_number", "account_no", "ifsc", "pan", "aadhaar", "aadhar",
    "kyc", "dob", "date_of_birth", "address", "bank_name", "beneficiary",
    "card_number", "balance", "name", "first_name", "last_name", "full_name",
    "customer_id", "player_id", "user_id", "token", "password", "otp",
]

# Realistic key spellings that a drifting CRM contract could plausibly
# produce, which an EXACT-match deny-list misses entirely but which are
# unambiguously PII-adjacent by their key alone. Reproduced by an
# independent review against an earlier version of this module.
_WILDCARD_ONLY_DENY_KEYS = [
    "customer_name",
    "account_holder_name",
    "beneficiary_name",
    "address_line1",
    "city",
    "kyc_rejection_reason",
    "auth_token",          # a real key in docs/crm-api-contract.md
    "api_key",
    "ip_address",
    "Mobile_Number",
    "UPI_ID",
    "bank_saved",
    "bet_id",               # *id suffix -- also the tool_executor.py divergence
    "transaction_id",
]


# Round-2 regression: unanchored substring matching (a normalized key
# CONTAINS a deny term anywhere) wrongly denied this platform's own real,
# non-PII fields and its own tracing attribute names. Every one of these
# is a real key in docs/crm-api-contract.md or a well-known
# OpenInference/OTel attribute name (span_kind, span_name, ...), plus a
# handful of plain English words chosen specifically because they contain
# a deny term as a substring while being one single, unsplit token.
#
# NOTE: "span_id" is deliberately NOT in this list. Unlike "span_kind"/
# "span_name" (which only ever matched due to the round-2 "pan"/"name"
# substring bug, now fixed), "span_id" tokenizes to ["span", "id"] and
# still denies today via the ordinary, INTENTIONAL "id" simple-token rule
# -- the same rule that denies bet_id/transaction_id/PgsOrderId. That's a
# known, accepted, documented limitation (see the module docstring's
# "Known limitations" section), not a false positive this list covers.
_TOKEN_BOUNDARY_FALSE_POSITIVES = [
    # "pan" substring inside tracing/attribute vocabulary and real words.
    "span_kind", "openinference_span_kind", "span_name",
    "japan", "company", "expand", "panel", "spaniel",
    # "name" substring/suffix inside business-object config, not a person.
    "market_name", "game_name", "event_name", "tool_name",
    # "id" substring inside ordinary single-token English words.
    "paid", "void", "bid", "avoid", "grid", "valid", "rapid", "acid",
    # "phone" substring inside single-token words (a VOICE platform).
    "microphone", "telephone", "saxophone",
    # "key" substring inside ordinary engineering plumbing terms.
    "cache_key", "idempotency_key", "keyword", "monkey", "turkey",
    # "card" substring inside single-token words.
    "scorecard", "discard", "cardiac", "postcard", "wildcard",
    # "city" substring inside single-token words.
    "capacity", "velocity", "electricity",
    # "upi" substring inside single-token words.
    "occupied", "groupies",
    # Real, non-customer-specific operator/config fields.
    "supported_banks", "blocked_banks", "upi_supported", "mobile_app",
    "android", "kyc_documents_required",
    # camelCase/kebab-case variants of the same business-object "name"
    # false positive, to exercise the tokenizer's boundary detection
    # itself, not just underscore-separated forms.
    "marketName", "MarketName", "market-name", "market.name",
]

# Synthetic single-token strings built by gluing letters directly onto
# EVERY deny term with no separator, so the term is present only as a
# mid-token substring. None of these should ever be denied -- if any of
# them is, unanchored substring matching has regressed back in. This is
# the "probe token boundaries broadly, not just the named examples" half
# of the round-2 review's request.
_ALL_DENY_TERMS_FOR_SUBSTRING_PROBE = [
    "mobile", "phone", "email", "upi", "vpa", "ifsc", "pan", "aadhaar",
    "aadhar", "kyc", "dob", "address", "bank", "beneficiary", "card",
    "balance", "name", "token", "password", "otp", "city", "key", "id",
    "account",
]
_SYNTHETIC_SUBSTRING_PROBES = [
    f"zz{term}xx" for term in _ALL_DENY_TERMS_FOR_SUBSTRING_PROBE
] + [
    f"{term}xx" for term in _ALL_DENY_TERMS_FOR_SUBSTRING_PROBE
] + [
    f"zz{term}" for term in _ALL_DENY_TERMS_FOR_SUBSTRING_PROBE
]


class TestKeyBasedHardDrop:
    @pytest.mark.parametrize("key", _EXACT_MATCH_DENY_KEYS)
    def test_exact_match_deny_keys_still_redacted(self, key):
        # Exact-match keys must still work under the new token-based
        # matching (a bare term is a one-token key, which is exactly what
        # whole-token matching catches).
        result = redact_structure({key: "not-pii-shaped-value"})
        assert result[key] == REDACTED_KEY_PLACEHOLDER

    @pytest.mark.parametrize("key", _WILDCARD_ONLY_DENY_KEYS)
    def test_realistic_key_spellings_now_redacted(self, key):
        # None of these values are pattern-shaped on their own -- if this
        # test passes, it's proof the deny-list is catching them by KEY,
        # not by accidentally matching their value's content.
        result = redact_structure({key: "plain-text-value-with-no-pii-shape"})
        assert result[key] == REDACTED_KEY_PLACEHOLDER, (
            f"key {key!r} should be caught by token-aware deny-list "
            "matching but was not"
        )

    @pytest.mark.parametrize("key", _TOKEN_BOUNDARY_FALSE_POSITIVES)
    def test_token_boundary_false_positives_are_not_denied(self, key):
        # Round-2 regression pin: these are real keys (this platform's own
        # CRM contract and tracing attribute vocabulary) that an earlier,
        # unanchored-substring-matching version of this module wrongly
        # denied. None of their values are pattern-shaped, so a pass here
        # is proof the key itself is correctly NOT triggering the deny
        # list under token-aware matching.
        result = redact_structure({key: "plain-text-value-with-no-pii-shape"})
        assert result[key] == "plain-text-value-with-no-pii-shape", (
            f"key {key!r} was wrongly denied -- token-boundary regression"
        )

    @pytest.mark.parametrize("key", _SYNTHETIC_SUBSTRING_PROBES)
    def test_synthetic_mid_token_substrings_are_not_denied(self, key):
        # Broader, systematic version of the above: for EVERY deny term,
        # a synthetic single-token string containing that term only as a
        # substring (never as its own whole token) must never be denied.
        result = redact_structure({key: "plain-text-value-with-no-pii-shape"})
        assert result[key] == "plain-text-value-with-no-pii-shape", (
            f"synthetic key {key!r} was wrongly denied -- a deny term "
            "matched as a mid-token substring instead of a whole token"
        )

    @pytest.mark.parametrize("key", ["balance", "otp", "user_id", "phone"])
    def test_deny_key_redacted_for_numeric_value(self, key):
        result = redact_structure({key: 4250})
        assert result[key] == REDACTED_KEY_PLACEHOLDER

    @pytest.mark.parametrize("key", ["bank_name", "address"])
    def test_deny_key_redacted_for_dict_value(self, key):
        result = redact_structure({key: {"nested": "value", "count": 12345}})
        assert result[key] == REDACTED_KEY_PLACEHOLDER

    @pytest.mark.parametrize("key", ["name", "aadhaar"])
    def test_deny_key_redacted_for_list_value(self, key):
        result = redact_structure({key: ["Aadhaar", "PAN", 123456]})
        assert result[key] == REDACTED_KEY_PLACEHOLDER

    def test_key_matching_is_case_and_separator_insensitive(self):
        payload = {
            "Mobile_Number": "irrelevant",
            "ACCOUNT_NUMBER": "irrelevant",
            "Account-Number": "irrelevant",
            "dateOfBirth": "irrelevant",
        }
        result = redact_structure(payload)
        for v in result.values():
            assert v == REDACTED_KEY_PLACEHOLDER

    def test_hard_drop_happens_instead_of_value_scrubbing_not_after(self):
        # A value that WOULD also match a pattern must come out as the
        # key-based placeholder, not the pattern's placeholder -- proves the
        # hard drop runs first and short-circuits, rather than merely
        # running before an otherwise-identical scrub.
        result = redact_structure({"email": "player@example.com"})
        assert result["email"] == REDACTED_KEY_PLACEHOLDER
        assert result["email"] != "<EMAIL>"

    def test_non_deny_key_is_not_affected(self):
        # A key containing none of the deny terms as a WHOLE TOKEN must
        # not be swept up -- value-pattern-scrubbing still applies
        # normally.
        result = redact_structure({"favorite_game": "player@example.com"})
        assert result["favorite_game"] == "<EMAIL>"

    def test_id_is_a_whole_token_match_not_a_substring_match(self):
        # "id" matches only as its OWN token (produced by a real separator
        # or camelCase boundary) -- "bet_id" tokenizes to ["bet", "id"]
        # and is denied; "bid"/"grid"/"avoid"/"guidance"/"video" never
        # produce a separate "id" token (nothing inside them to split on)
        # and are not denied.
        assert redact_structure({"bet_id": "x"})["bet_id"] == REDACTED_KEY_PLACEHOLDER
        for safe_key in ["video", "guidance", "bid", "grid", "avoid", "void", "paid"]:
            assert redact_structure({safe_key: "x"})[safe_key] == "x"


class TestAmbiguousTokenTerms:
    """Terms that are genuinely ambiguous at the whole-token level on this
    platform (the same token means different things depending on its
    neighbors) get a qualifier rule instead of bare presence -- see
    _AMBIGUOUS_TOKEN_TERMS in the module and its docstring's "Design
    notes". These tests exercise both directions for each ambiguous term
    explicitly, beyond the single pass/fail example each got in the
    round-2 review.

    IMPORTANT (round-3 review finding): "name" and "key" are DEFAULT-DENY
    ("except_with") -- the safe/allowed side is a small enumerated set and
    everything else denies. The deny-side test parameters below are
    deliberately chosen words that are NOT drawn from the implementation's
    own allow-list frozensets (_AMBIGUOUS_TOKEN_TERMS in the module) --
    using the implementation's own allow-list to test its own default
    would be tautological and structurally unable to catch an enumeration
    gap, which is exactly the round-1 regression class (rotated: round 1's
    bug was an unenumerated DENY-list miss, this would be an unenumerated
    ALLOW-list "test" that can't see a default-polarity bug). Instead, the
    deny-side parameters are independently generated qualifiers,
    including every leaked key an independent review reproduced against
    an earlier "only_with" (default-ALLOW) version of these two rules."""

    @pytest.mark.parametrize("key", ["mobile", "Mobile_Number", "mobile_no", "mobile_num"])
    def test_mobile_denied_when_bare_or_with_number_qualifier(self, key):
        assert redact_structure({key: "x"})[key] == REDACTED_KEY_PLACEHOLDER

    @pytest.mark.parametrize("key", ["mobile_app", "mobile_verified", "mobile_os"])
    def test_mobile_not_denied_without_number_qualifier(self, key):
        assert redact_structure({key: "x"})[key] == "x"

    # "upi"/"kyc" independent (non-allow-list) deny-side probes -- neither
    # "reference"/"transaction_note"/"handle_alt" nor "notes"/"comment"/
    # "score" appear anywhere in the upi/kyc companion frozensets.
    @pytest.mark.parametrize(
        "key",
        [
            "upi", "upi_id", "upi_vpa", "upi_reference", "upi_transaction_note",
            "upi_handle_alt",
        ],
    )
    def test_upi_denied_by_default_independent_probes(self, key):
        assert redact_structure({key: "x"})[key] == REDACTED_KEY_PLACEHOLDER

    @pytest.mark.parametrize(
        "key", ["upi_supported", "upi_enabled", "upi_available", "upi_allowed"]
    )
    def test_upi_not_denied_with_config_qualifier(self, key):
        assert redact_structure({key: "x"})[key] == "x"

    @pytest.mark.parametrize(
        "key",
        [
            "kyc", "kyc_status", "kyc_documents", "kyc_rejection_reason",
            "kyc_notes", "kyc_comment", "kyc_score",
        ],
    )
    def test_kyc_denied_by_default_independent_probes(self, key):
        assert redact_structure({key: "x"})[key] == REDACTED_KEY_PLACEHOLDER

    @pytest.mark.parametrize(
        "key", ["kyc_documents_required", "kyc_supported", "kyc_enabled"]
    )
    def test_kyc_not_denied_with_config_qualifier(self, key):
        assert redact_structure({key: "x"})[key] == "x"

    # "key" round-3 fix (F2): flipped from "only_with" (default-allow,
    # leaked "signing_key"/"master_key"/... which weren't in the old
    # enumerated companion list) to "except_with" (default-deny). Deny-side
    # probes below are independently chosen -- NOT drawn from the module's
    # non-secret-key-kind allow-list ({"cache", "idempotency", "sort",
    # "primary", "partition", "composite", "foreign", "unique", "hash",
    # "index"}) -- specifically including the leaked keys an independent
    # review reproduced.
    @pytest.mark.parametrize(
        "key",
        [
            "key", "api_key", "secret_key", "auth_key", "access_key",
            "signing_key", "master_key", "rotation_key", "backup_key",
            "encryption_key", "private_key", "license_key",
        ],
    )
    def test_key_denied_by_default_independent_probes(self, key):
        assert redact_structure({key: "x"})[key] == REDACTED_KEY_PLACEHOLDER, (
            f"key {key!r} should deny under key's default-deny polarity "
            "but did not -- F2 regression"
        )

    @pytest.mark.parametrize(
        "key",
        [
            "cache_key", "idempotency_key", "sort_key", "primary_key",
            "partition_key", "hash_key", "composite_key",
        ],
    )
    def test_key_not_denied_for_non_secret_key_kinds(self, key):
        assert redact_structure({key: "x"})[key] == "x"

    # "name" round-3 fix (F1): flipped from "only_with" (default-allow,
    # leaked "agent_name" -- a real key in docs/crm-api-contract.md -- plus
    # middle_name/father_name/payer_name/winner_name/... none of which
    # were in the old enumerated person-qualifier list) to "except_with"
    # (default-deny). Deny-side probes below are independently chosen --
    # NOT drawn from the module's business/system-object-noun allow-list
    # -- specifically including every key an independent review
    # reproduced as leaking, plus extras.
    @pytest.mark.parametrize(
        "key",
        [
            "name", "first_name", "last_name", "full_name", "customer_name",
            "account_holder_name", "beneficiary_name", "player_name", "user_name",
            # F1 leak list, reproduced by an independent review against
            # the old default-allow version of this rule:
            "agent_name", "middle_name", "maiden_name", "surname", "sur_name",
            "given_name", "legal_name", "display_name", "login_name",
            "nick_name", "father_name", "mother_name", "spouse_name",
            "payer_name", "payee_name", "depositor_name", "remitter_name",
            "sender_name", "receiver_name", "winner_name", "punter_name",
            "bettor_name", "caller_name", "subscriber_name", "member_name",
            "owner_name", "client_name",
            # "cardholder" is a closed compound (no separator/case
            # boundary to split on) -- tokenizes to a single token and is
            # listed directly in _SIMPLE_TOKEN_TERMS rather than reached
            # via the "name" conjunction rule at all, but it must deny
            # either way.
            "cardholder",
        ],
    )
    def test_name_denied_by_default_independent_probes(self, key):
        assert redact_structure({key: "x"})[key] == REDACTED_KEY_PLACEHOLDER, (
            f"key {key!r} should deny under name's default-deny polarity "
            "but did not -- F1 regression"
        )

    @pytest.mark.parametrize(
        "key",
        [
            "market_name", "game_name", "event_name", "tool_name", "span_name",
            "brand_name", "product_name", "provider_name", "session_name",
            "file_name", "host_name", "tenant_name",
        ],
    )
    def test_name_not_denied_for_business_object_nouns(self, key):
        assert redact_structure({key: "x"})[key] == "x"


# ---------------------------------------------------------------------------
# F2 (round-3 review) -- credential/secret terms with no value-level
# pattern backstop (a random alphanumeric secret has no digit-run to
# catch it), same failure class as F1's person names.
# ---------------------------------------------------------------------------


class TestCredentialAndSecretTerms:
    @pytest.mark.parametrize(
        "key",
        [
            "secret", "client_secret", "webhook_secret", "credential",
            "credentials", "api_credentials", "cvv", "cvc", "passcode",
            "pwd", "pin", "atm_pin", "signature", "jwt", "jwt_token",
            "bearer", "bearer_token", "cookie", "session_cookie",
            "privatekey",
        ],
    )
    def test_credential_shaped_key_denied(self, key):
        result = redact_structure({key: "x"})
        assert result[key] == REDACTED_KEY_PLACEHOLDER, (
            f"key {key!r} should deny (credential/secret term) but did "
            "not -- F2 regression"
        )

    def test_webhook_secret_realistic_scenario(self):
        # webhook_secret is plausible live config given commit
        # ad6f597 (fix(security): remove PII from logs, add webhook
        # signature-verification infra).
        payload = {"webhook_secret": "whsec_abcdef1234567890"}
        result = redact_structure(payload)
        assert result["webhook_secret"] == REDACTED_KEY_PLACEHOLDER


# ---------------------------------------------------------------------------
# The PgsOrderId divergence -- pinned explicitly.
# ---------------------------------------------------------------------------


class TestPgsOrderIdDivergesFromToolExecutor:
    def test_pgs_order_id_value_is_redacted_here(self):
        """src/chatbot/tool_executor.py::_redact_internal_ids() has a named,
        deliberate EXEMPTION for "PgsOrderId" (see
        _PGS_ORDER_ID_KEY_NORMALIZED there): that function forwards the
        payment-gateway order id to the LLM unmodified because the LLM
        needs it to raise a deposit-verification ticket with the vendor.

        This module must NOT carry that exemption over. A trace redactor's
        job is the opposite of tool_executor's: maximum redaction of
        anything PII/business-identifier-shaped for a human-readable
        debugging trace, not preservation of a value a downstream LLM
        consumer needs. "PgsOrderId" normalizes to "pgsorderid", which ends
        in "id" -- so this is now caught by the key-based hard-drop (an
        even more direct mechanism than the value-pattern-scrub that used
        to catch it), and comes out as "<REDACTED>" rather than "PGS<NUM>".
        Either way, if a future reader is tempted to "fix" this to match
        tool_executor.py's behavior -- don't. That would reintroduce
        exactly the kind of payload leak this whole module exists to
        prevent (see the module docstring's "Relationship to other
        redaction code" section).
        """
        payload = {
            "PgsOrderId": "PGS20260621143000123",
            "datetime": "2026-06-21T14:30:00Z",
            "amount": 1500.00,
            "status": "failed",
        }
        result = redact_structure(payload)

        assert result["PgsOrderId"] != "PGS20260621143000123"
        assert redaction_post_condition_ok(result["PgsOrderId"])

    def test_pgs_order_id_shaped_value_is_redacted_even_under_a_non_id_key(self):
        # Defense-in-depth check: even if a future CRM contract renames the
        # field to something that does NOT end in "id" (so the key-based
        # hard-drop wouldn't fire), the value-level digit-run catch-all
        # still strips the order reference's digits on its own.
        result = redact_structure({"gateway_reference": "PGS20260621143000123"})
        assert result["gateway_reference"] != "PGS20260621143000123"
        assert redaction_post_condition_ok(result["gateway_reference"])


# ---------------------------------------------------------------------------
# Realistic nested payload -- wallet + transactions + KYC/profile + matka.
# ---------------------------------------------------------------------------


def _realistic_payload() -> dict:
    """Modeled directly on docs/crm-api-contract.md's documented response
    shapes for get_player_wallet, get_player_transactions,
    get_player_profile, get_matka_bids, and
    get_player_latest_deposit_order."""
    return {
        "wallet": {
            "real_balance": 4250.75,
            "bonus_balance": 500.00,
            "total_available": 4750.75,
            "currency": "INR",
            "pending_withdrawal": {"amount": 1000.00, "status": "processing"},
        },
        "transactions": {
            "transactions": [
                {
                    "id": "txn_001",
                    "type": "deposit",
                    "amount": 2000.00,
                    "status": "success",
                    "timestamp": "2026-06-20T10:30:00Z",
                    "method": "UPI",
                },
                {
                    "id": "txn_002",
                    "type": "casino",
                    "amount": -350.00,
                    "status": "settled",
                    "timestamp": "2026-06-20T11:15:00Z",
                    "game": "Teen Patti",
                },
            ],
            "total": 25,
        },
        "profile": {
            "vip_tier": "Silver",
            "kyc_status": "verified",
            "kyc_documents": ["Aadhaar", "PAN"],
            "mobile": "+91-98765-43210",
            "email": "player@example.com",
            "bank_saved": {
                "bank": "HDFC",
                "account_last4": "7890",
                "upi": "player@upi",
            },
            "created_at": "2025-03-15T00:00:00Z",
            "last_login": "2026-06-21T08:45:00Z",
        },
        "matka_bids": {
            "bids": [
                {
                    "id": "bid_001",
                    "market": "Kalyan",
                    "bet_type": "Jodi",
                    "number": "45",
                    "stake": 100.00,
                    "status": "open",
                    "placed_at": "2026-08-04T13:10:00Z",
                },
                {
                    "id": "bid_002",
                    "market": "Rajdhani Day",
                    "bet_type": "Single",
                    "number": "6",
                    "stake": 50.00,
                    "status": "won",
                    "result": "123-6",
                    "payout": 450.00,
                    "credited_at": "2026-08-03T17:40:00Z",
                    "settled_at": "2026-08-03T17:36:00Z",
                },
            ],
            "pnl": {"this_week": 350.00, "this_month": -120.00},
        },
        "latest_deposit_order": {
            "PgsOrderId": "PGS20260621143000123",
            "datetime": "2026-06-21T14:30:00Z",
            "amount": 1500.00,
            "status": "failed",
        },
        "payment_config": {
            "deposit_destination": {
                "bank": "HDFC",
                "account_number": "50100123456789",
                "ifsc": "HDFC0001234",
                "upi_id": "operatorpay@hdfcbank",
            },
        },
    }


class TestRealisticNestedPayload:
    def test_post_conditions_hold_over_full_payload(self):
        payload = _realistic_payload()
        result = redact_structure(payload)
        rendered = _flatten_to_text(result)

        # The shared, module-owned post-condition check (no '@', no digit
        # run >= 4, no UUID shape) -- the same helper the plan's later
        # §2.5 exporter-level re-assertion is specified to reuse.
        assert redaction_post_condition_ok(rendered), rendered

        # No deny-set key's ORIGINAL value survives anywhere in the output.
        original_pii_values = [
            "+91-98765-43210",
            "player@example.com",
            "player@upi",
        ]
        for original_value in original_pii_values:
            assert original_value not in rendered, (original_value, rendered)

    def test_business_relevant_ids_still_get_redacted_here(self):
        # Divergence from tool_executor.py, re-affirmed in the context of a
        # full realistic payload (see TestPgsOrderIdDivergesFromToolExecutor
        # for the isolated, explicitly-pinned version of this).
        payload = _realistic_payload()
        result = redact_structure(payload)
        assert (
            result["latest_deposit_order"]["PgsOrderId"]
            != "PGS20260621143000123"
        )
        # bet/transaction-style ids are swept too now ("*id" suffix rule) --
        # the opposite of tool_executor.py's deliberate preservation of
        # bet_id/transaction_id.
        assert result["transactions"]["transactions"][0]["id"] == (
            REDACTED_KEY_PLACEHOLDER
        )
        assert result["matka_bids"]["bids"][0]["id"] == REDACTED_KEY_PLACEHOLDER

    def test_walkthrough_before_after_is_meaningfully_different(self):
        # Guards against a vacuous post-condition pass (e.g. an
        # accidentally-empty output would also satisfy "no @, no long
        # digit run..."). Assert real redaction actually happened AND that
        # non-PII structural/business fields survive untouched.
        payload = _realistic_payload()
        result = redact_structure(payload)

        assert result["profile"]["mobile"] == REDACTED_KEY_PLACEHOLDER
        assert result["profile"]["email"] == REDACTED_KEY_PLACEHOLDER
        # "bank_saved" normalizes to "banksaved", which now matches the
        # "bank" contains-rule -- the WHOLE nested object is hard-dropped,
        # not recursed into (matches the module's own docstring claim,
        # which an earlier exact-match version of the deny-list
        # contradicted).
        assert result["profile"]["bank_saved"] == REDACTED_KEY_PLACEHOLDER
        # Likewise the wallet's "real_balance"/"bonus_balance" keys contain
        # "balance" and are now hard-dropped wholesale rather than only
        # having their numeric value digit-scrubbed.
        assert result["wallet"]["real_balance"] == REDACTED_KEY_PLACEHOLDER
        assert result["wallet"]["bonus_balance"] == REDACTED_KEY_PLACEHOLDER
        # "total_available" does NOT contain "balance" (or any other deny
        # term) -- it survives as a key, and its numeric value still goes
        # through the digit-run catch-all.
        assert result["wallet"]["total_available"] != 4750.75
        assert redaction_post_condition_ok(result["wallet"]["total_available"])
        # "kyc_status"/"kyc_documents" both contain "kyc" and are now
        # hard-dropped wholesale.
        assert result["profile"]["kyc_status"] == REDACTED_KEY_PLACEHOLDER
        assert result["profile"]["kyc_documents"] == REDACTED_KEY_PLACEHOLDER

        # Non-PII structural data survives verbatim: status enums, product
        # labels, small counts, short bet-type strings, none of which
        # contain any deny term or match any content pattern.
        assert result["transactions"]["transactions"][0]["status"] == "success"
        assert result["transactions"]["transactions"][0]["type"] == "deposit"
        assert result["matka_bids"]["bids"][0]["market"] == "Kalyan"
        assert result["matka_bids"]["bids"][0]["bet_type"] == "Jodi"
        assert result["profile"]["vip_tier"] == "Silver"
        assert result["payment_config"]["deposit_destination"]["bank"] == (
            REDACTED_KEY_PLACEHOLDER
        )


# ---------------------------------------------------------------------------
# Idempotence.
# ---------------------------------------------------------------------------


class TestIdempotence:
    def test_structure_idempotent(self):
        payload = _realistic_payload()
        once = redact_structure(payload)
        twice = redact_structure(once)
        assert once == twice

    def test_text_idempotent(self):
        text = (
            "customer's mobile 9876543210 says amount ₹4,250.75 wasn't "
            "credited to UPI ram@paytm"
        )
        once = redact_text(text)
        twice = redact_text(once)
        assert once == twice

    def test_not_idempotent_when_a_dict_key_scrubs_to_a_deny_token(self):
        """Known limitation (F3, round-3 review), deliberately NOT fixed --
        see the module docstring's "Known limitations" section. When a
        dict key is itself PII-shaped (e.g. a literal phone number used as
        a mapping key), the FIRST pass scrubs the key's shape
        ("9876543210" -> "<PHONE>") but leaves its value alone (the key
        wasn't recognized as a deny-list TERM, only pattern-scrubbed by
        shape). A SECOND pass then sees a key that normalizes to "phone"
        (stripped of its angle brackets) and hard-drops the value via the
        ordinary deny-list mechanism. The direction is always safe --
        strictly MORE redacted on the second pass, never less -- so this
        is accepted rather than fixed. Pinned here so the behavior is
        visible and tested rather than a silent surprise for the next
        reader.
        """
        payload = {"9876543210": {"a": 1}}
        once = redact_structure(payload)
        twice = redact_structure(once)

        assert once == {"<PHONE>": {"a": 1}}
        assert twice == {"<PHONE>": REDACTED_KEY_PLACEHOLDER}
        assert once != twice

        # A third pass is stable (this is where idempotence actually
        # kicks back in, one step later than usual).
        thrice = redact_structure(twice)
        assert twice == thrice


# ---------------------------------------------------------------------------
# Fail-closed path.
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_circular_reference_fails_closed(self):
        # A circular structure would otherwise recurse until Python raises
        # RecursionError -- a realistic way a naive recursive-walk
        # implementation breaks internally. The result must be the literal
        # failure placeholder, never a partial structure and never the raw
        # input.
        circular: dict = {"a": 1, "b": {"mobile": "9876543210"}}
        circular["self"] = circular
        result = redact_structure(circular)
        assert result == REDACTION_FAILED_PLACEHOLDER

    def test_monkeypatched_internal_failure_fails_closed(self, monkeypatch):
        import src.observability.trace_redaction as trace_redaction

        def _boom(_value):
            raise RuntimeError("simulated internal failure")

        monkeypatch.setattr(trace_redaction, "_redact_value", _boom)
        result = trace_redaction.redact_structure({"mobile": "9876543210"})
        assert result == REDACTION_FAILED_PLACEHOLDER

    def test_text_fail_closed_on_internal_failure(self, monkeypatch):
        import src.observability.trace_redaction as trace_redaction

        def _boom(_text):
            raise RuntimeError("simulated internal failure")

        monkeypatch.setattr(trace_redaction, "_scrub_text", _boom)
        result = trace_redaction.redact_text("mobile 9876543210")
        assert result == REDACTION_FAILED_PLACEHOLDER

    def test_failure_placeholder_is_never_a_partial_or_raw_value(self):
        circular: dict = {}
        circular["self"] = circular
        result = redact_structure(circular)
        # Exactly the literal string, not e.g. a dict containing it.
        assert isinstance(result, str)
        assert result == "<redaction-failed>"


# ---------------------------------------------------------------------------
# Free text (not just structured data).
# ---------------------------------------------------------------------------


class TestFreeText:
    def test_sentence_with_multiple_pii_types_redacted_readably(self):
        text = (
            "customer's mobile 9876543210 says amount ₹4,250.75 wasn't "
            "credited to UPI ram@paytm"
        )
        result = redact_text(text)

        assert "9876543210" not in result
        assert "4,250.75" not in result
        assert "ram@paytm" not in result
        assert redaction_post_condition_ok(result)

        assert "<PHONE>" in result
        assert "<AMOUNT>" in result
        assert "<UPI>" in result

        # Surrounding sentence structure stays readable -- not replaced
        # wholesale, not turned into gibberish.
        assert result == (
            "customer's mobile <PHONE> says amount <AMOUNT> wasn't "
            "credited to UPI <UPI>"
        )

    def test_plain_sentence_with_no_pii_is_unchanged(self):
        text = "the bot answered the question about game rules"
        assert redact_text(text) == text


# ---------------------------------------------------------------------------
# Public post-condition helper -- used above, and pinned directly here so a
# drift between the helper and the tests that rely on it is caught fast.
# ---------------------------------------------------------------------------


class TestRedactionPostConditionHelper:
    def test_flags_at_sign(self):
        assert redaction_post_condition_ok("ram@paytm") is False

    def test_flags_long_digit_run(self):
        assert redaction_post_condition_ok("account 12345") is False

    def test_flags_uuid_shape(self):
        assert redaction_post_condition_ok(
            "id 6c1a77a6-1234-4abc-9def-8add58aba9ef"
        ) is False

    def test_passes_clean_text(self):
        assert redaction_post_condition_ok("status is <REDACTED> and open") is True

    def test_accepts_non_string_via_repr(self):
        assert redaction_post_condition_ok({"a": 1}) is True
        assert redaction_post_condition_ok({"a": "ram@paytm"}) is False

    def test_flags_upi_shaped_at_sign_without_dot(self):
        # A bare local@domain-shaped '@' with no dot (a VPA, not caught by
        # the email pattern) must still be flagged -- flanked by
        # alphanumerics on both sides.
        assert redaction_post_condition_ok("ram@paytm") is False
        assert redaction_post_condition_ok("still has player@upi in it") is False

    def test_does_not_flag_odds_notation_at_sign(self):
        # Fix D (round-2 review): a bare '@' with a SPACE (or other
        # non-alphanumeric) on at least one side is ordinary betting
        # platform text, not a missed email/UPI -- must not false-alarm.
        assert redaction_post_condition_ok("back @ 1.85") is True
        assert redaction_post_condition_ok("odds @2.50 now") is True

    def test_does_not_flag_social_handle_at_sign(self):
        assert redaction_post_condition_ok("follow us @kalyanmatka") is True
        assert redaction_post_condition_ok("@kalyanmatka posted an update") is True

    def test_still_flags_a_genuinely_missed_email_or_upi(self):
        # Sanity: the narrowing to "flanked on both sides" must not lose
        # the ability to catch what it's actually meant to catch.
        assert redaction_post_condition_ok("contact ram@example.com now") is False
        assert redaction_post_condition_ok("pay to ram@paytm please") is False
