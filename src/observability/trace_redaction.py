r"""PII-redaction boundary for LLM conversation tracing (Phase 0).

## What this module is

This is the load-bearing security boundary described in
``docs/superpowers/plans/2026-09-07-llm-conversation-tracing.md`` (§2). That
plan proposes sending tool arguments/responses, LLM completions, and user
messages to a conversation-tracing backend (self-hosted Arize Phoenix, per
the plan) so support staff can replay what the bot actually did on a given
turn. This platform has a deliberate, load-bearing policy that those exact
payloads are never logged at any level — hardened after an active production
PII leak (see commit ``ad6f597 fix(security): remove PII from logs``). A
tracing tool that captures raw tool/LLM payloads would reopen that leak by
construction unless something sits between "what the LLM/tools saw" and
"what leaves the process." This module is that something.

**This module does not send anything anywhere.** It has no network calls, no
file I/O, no logging side effects. It is a pure-function scrubber: nested
structure or free text goes in, a redacted version comes out (or, on any
internal failure, the literal string ``"<redaction-failed>"`` comes out —
see "Fail closed" below). Nothing in this codebase imports or calls this
module yet (Phase 0 is deliberately standalone, per the plan's §7 phased
rollout) — it exists to be reviewed and tested on its own merits before any
tracing backend is wired up at all.

## Zero dependency on any tracing backend

This module has **no import of OpenTelemetry, Phoenix, Langfuse, or any
other tracing/observability SDK**, and must never gain one. Its only imports
are the Python standard library (``re``). That is intentional and is the
entire point of doing this as Phase 0: the plan explicitly calls out that
"if the tracing backend is ever rejected entirely, or swapped, this phase
still stands alone." A future maintainer wiring in a tracing facade should
import *this* module and call its functions — never the other way around,
and never fold backend-specific logic in here.

## Relationship to other redaction code in this repo — read before touching

- ``src/utils/redact.py`` (``redact_url()``) does not fit this job: it is a
  single 28-line function that strips userinfo/query/fragment from a URL
  string. It has no notion of structured payloads or PII categories at all,
  so it was not extended for this purpose.
- ``src/chatbot/tool_executor.py::_redact_internal_ids()`` is where the
  recursive dict/list/str walk *shape* used below was copied from (the
  normalized-key-matching idiom, in particular). **Its ruleset was
  deliberately NOT copied.** That function is a system-routing redactor: it
  strips internal plumbing keys (``operator_id``, ``user_id``, ...) from CRM
  responses before they reach the LLM, but it *deliberately preserves*
  ``bet_id``/``transaction_id`` and has an explicit, narrow exemption that
  lets ``PgsOrderId`` (the payment-gateway order reference) through even
  though it is UUID-shaped — because the LLM needs those specific values to
  do its job (place a bet reference in a reply, forward an order id to a
  verification vendor, etc).

  **This module has the opposite goal.** It sits on the *trace* path, not
  the LLM path — nothing that passes through here is going back to a model
  or a customer, it is going to a debugging backend that a human might read
  later. There is no "the consumer needs this exact value" argument to be
  made for a trace, so there is no equivalent exemption here.
  ``get_player_latest_deposit_order``'s ``PgsOrderId`` value **is** redacted
  by this module (see the test suite for an explicit pinned test on this) —
  do not "fix" that to match ``tool_executor.py``'s behavior; the divergence
  is intentional and is the entire reason this is a separate module instead
  of a shared helper. So are ``bet_id``/``transaction_id``/any other
  business-object id: this module's key-matching treats any key that
  starts-or-ends with "id" as PII-adjacent-enough to drop (see
  ``_is_denied_key`` below), which is a deliberately more aggressive stance
  than ``tool_executor.py``'s narrow, explicit routing-key list.

## Fail closed, not open

Elsewhere in this codebase, "best-effort, never raise" means "don't crash
the call, the LLM/customer still needs an answer" (see the extensive
comments in ``tool_executor.py`` about why a failed CRM call still produces
a usable error string rather than propagating an exception). **That
convention is inverted here.** If anything in this module raises, silence or
redaction is always the safe outcome and passing the raw, unredacted value
through is never acceptable — a bug in a regex must not become a PII leak.
Both public entry points (`redact_structure` and `redact_text`) wrap their
entire body in a bare ``try/except Exception`` that returns the literal
string ``"<redaction-failed>"`` on any failure, including things like
runaway recursion on a circular reference. Downstream, "<redaction-failed>"
should be treated as "drop this span/field," never as data.

## Design notes / judgment calls (documented per the "err toward redacting
too much" instruction rather than resolved silently)

- **Deny-by-default, most-specific-pattern-first, blunt catch-all last.**
  Every string is run through an ordered pipeline of substitutions: UUID,
  email, UPI/VPA, Indian mobile number, currency amount, IFSC, PAN,
  Aadhaar-shaped, then a catch-all "any digit run of length >= 4" pattern,
  with date patterns applied just before that catch-all (see below for why
  that's not the literal order the plan's prose lists them in).
- **Dates are scrubbed just before the digit-run catch-all, not after it, a
  deliberate deviation from the plan's §2.4 prose ordering.** The catch-all
  is a blunt `\d{4,}` sweep. If it ran before the date pattern, it would
  already have eaten the 4-digit year out of e.g. "2026-06-20T10:30:00Z"
  (leaving "<NUM>-06-20T10:30:00Z"), and the date pattern would then have
  nothing left to match — timestamps would come out as a `<NUM>` fragment
  plus raw leftover 2-digit month/day/hour/minute/second components instead
  of one clean `<DATE>` tag. Running the date pattern first produces the
  intended, cleaner `<DATE>` tag and is strictly more redacted, not less, so
  it is consistent with this module's "when in doubt, redact more" bias.
- **UPI/VPA vs. email disambiguation.** A UPI VPA (`ram@paytm`,
  `player@upi`, `operatorpay@hdfcbank`) has the same `local@domain` shape as
  an email but the "domain" is a bank/PSP handle with no TLD dot. The email
  pattern requires a dot-separated TLD; UPI matching runs second and only
  fires on `local@domain`-shaped text that the email pattern did not already
  consume, so a real email is never double-tagged as `<UPI>` and a VPA is
  never left unredacted for lack of a dot. The UPI pattern's character
  class (`[a-zA-Z]`) already spans both cases directly -- there is no
  `re.IGNORECASE` flag on it (an earlier version of this comment claimed
  one that was never actually set), it simply doesn't need one.
- **Both the email and UPI patterns use BOUNDED quantifiers, not `+`/`*`,
  specifically to prevent catastrophic-backtracking (ReDoS) on adversarial
  input.** An earlier version of this module used unbounded quantifiers
  (`[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`); on `local@domain`-shaped text with many
  dots and no valid trailing TLD (e.g. `"a@" + "a." * 5000`), the regex
  engine's backtracking search for a valid split point was quadratic in
  input length, measured at ~38s of blocking CPU on 100KB of adversarial
  text. Since `redact_text` is specified (per the plan) to run on
  attacker-influenced user-message text once a tracing facade is wired in,
  an unbounded pattern here is a real availability bug, not a theoretical
  one. The fix bounds the local part to 64 characters, each domain label to
  63 characters, the label count to 8, and the final TLD to 24 characters
  (all generous versus real email/VPA lengths) so that the maximum amount
  of backtracking work per candidate match position is a small constant,
  independent of the overall input length — see `test_trace_redaction.py`
  for a timing test pinned against this regression class.
- **Indian mobile numbers** are matched as an optional `+91`/`91` prefix
  plus 10 digits starting 6-9, allowing a single separator (space or hyphen)
  after the first 5 digits, e.g. "+91-98765-43210" or "9876543210". The
  pattern is bounded by digit-adjacency lookarounds (`(?<!\d)` / `(?!\d)`),
  NOT `\b` word boundaries (an earlier version of this comment claimed
  "word boundaries," which is inaccurate: `\b` would not stop a match from
  running into a differently-shaped digit run either, since digits are all
  `\w`) -- this means a 10-digit run embedded inside a longer digit run
  (e.g. part of a 12-digit Aadhaar number or a 14-digit account number) is
  correctly *not* claimed by the phone pattern, and falls through to the
  Aadhaar pattern or the digit-run catch-all instead, per the "most specific
  first" ordering.
- **The currency pattern requires that the `Rs`/`Rs.`/`INR`/`₹` prefix not
  be immediately preceded by a letter** (`(?<![A-Za-z])`). Without this, the
  case-insensitive `Rs` alternative matches inside ordinary words that
  happen to contain the two-letter substring "rs" followed by a number
  later in the same text (e.g. "hours 2400 logged" previously mis-matched
  as "hou" + "<AMOUNT>", eating half of "hours"). The digit run itself is
  still caught by the catch-all pattern regardless, so this fix only
  affects which placeholder a false-positive currency match would have
  produced and stops it from also eating letters that aren't part of any
  number.
- **The UUID pattern uses hex-character lookarounds
  (`(?<![0-9a-fA-F])` / `(?![0-9a-fA-F])`) instead of `\b` word
  boundaries.** `\b` does not fire between two word characters (a letter and
  a digit are both `\w`), so a UUID glued directly to a preceding word
  character with no separator (e.g. `"user550e8400-e29b-..."`) would not
  even be recognized as starting a UUID under a `\b`-anchored pattern, and
  only its pure-digit sub-runs would get swept by the later digit-run
  catch-all — leaving hex fragments like `"e29b-41d4-a716"` in the output.
  Anchoring on "is not a hex character" instead of "is not a word
  character" catches this: a preceding letter outside `[a-fA-F]` (e.g.
  `"user"`'s trailing `"r"`) does not block the match. The trailing
  lookahead deliberately checks only `[0-9a-fA-F]`, NOT also `-`: an earlier
  version of this pattern also excluded a following `-` from the lookahead,
  which broke the common case of a UUID embedded in an ordinary
  hyphen-joined identifier (e.g. `"order-<uuid>-v2"`, `"trace-<uuid>-span"`)
  -- a hyphen immediately after the last hex digit is exactly what a
  correctly-terminated UUID looks like in that context, not a sign of
  ambiguity, so it must not block the match. **The actual, precise residual
  limitation** (state it exactly, not the narrower "glued on both sides"
  claim an earlier version of this note made): a UUID is NOT fully matched
  whenever a hex-alphabet character (`0-9a-fA-F`) sits immediately before
  the first hex digit or immediately after the last one, on *either single*
  side independently -- e.g. `"id:550e8400-...-440000end"` fails to
  collapse to `<UUID>` because trailing `"end"` starts with the hex-valid
  letter `e`, even though the left side (`"id:"`, ending in a
  non-hex-valid `:`) is perfectly unambiguous. This is a genuine, accepted
  limitation of any regex-only hex-boundary approach (there is no way to
  tell, from the text alone, whether that adjacent hex character belongs to
  the UUID or to whatever follows it), not something fixable without
  reintroducing a `\b`-style boundary and its own worse gap.
- **The digit-run catch-all is deliberately blunt by design, not an
  oversight** (per the plan): account numbers, transaction/order/bid ids,
  card fragments, and OTPs are all just digit runs, and enumerating each
  business object's id format separately would be a losing, ever-drifting
  game against real (and documented-as-drifting) CRM response shapes. Any
  digit run of length >= 4 that survives every more-specific pattern above
  becomes `<NUM>`.
- **Numeric (int/float) leaf values are scrubbed too, not just strings.**
  Real payloads in this codebase carry PII-shaped data as JSON numbers, not
  strings (e.g. `"real_balance": 4250.75`). A leaf that is an `int` or
  `float` (and not a `bool`) is rendered to its string form and run through
  the same pattern pipeline; if any pattern fired, the scrubbed *string* is
  written back in place of the original number, deliberately changing its
  type. If nothing fired, the original number is preserved unchanged.
- **Deny-by-default at the TYPE level, not just the string-pattern level.**
  `_redact_value` explicitly recognizes exactly `dict`, `list`, `tuple`,
  `str`, `bool`, `NoneType`, `int`, and `float`. Anything else — `set`,
  `frozenset`, `bytes`, `bytearray`, `decimal.Decimal`, `datetime`/`date`, a
  custom object, a dataclass instance, ... — is NOT run through the string
  pipeline and returned as-is; it is unconditionally replaced with
  `"<REDACTED>"`. An earlier version of this module fell through to `return
  value` for any unrecognized type, which meant a `set` containing a phone
  number, a `Decimal` wallet balance (a real shape for DB-numeric fields),
  or a custom object's entire `__dict__` (once later stringified by
  whatever eventually serializes the trace) would leak completely
  unredacted. Since this module cannot enumerate every type a future CRM
  response or ORM row might hand it, the safe default for "a type I don't
  recognize" is to redact it wholesale, not to guess at scrubbing it.
- **The key-based hard-drop is TOKEN-aware matching, neither exact-match
  nor unanchored substring matching** (`_is_denied_key` / `_tokenize_key`).
  This went through two failed designs before landing here, and a future
  maintainer should understand both failures before "simplifying" this
  again:

  1. *Exact match* (`"name"` only matches a key that normalizes to
     literally `"name"`) does not generalize to real-world key spellings:
     `"customer_name"`, `"account_holder_name"`, `"beneficiary_name"`,
     `"address_line1"`, `"kyc_rejection_reason"`, `"auth_token"`,
     `"Mobile_Number"`, `"UPI_ID"`, and `"bank_saved"` all failed to match
     despite being unambiguously PII-adjacent by their key alone.
  2. *Unanchored substring match* (does the normalized, separator-stripped
     key **contain** a deny term anywhere) fixed #1 but is far too broad:
     against this platform's own real field vocabulary
     (`docs/crm-api-contract.md`) and its own tracing attribute names, it
     wrongly matched `"pan"` inside `"span_kind"` / `"span_name"` /
     `"openinference_span_kind"`, `"name"` inside `"market_name"` /
     `"game_name"` / `"event_name"` / `"tool_name"`, `"id"` inside
     `"valid"` / `"paid"` / `"void"` / `"bid"` / `"avoid"` / `"grid"` (`bid`
     is literally this platform's own noun for a Matka wager), `"phone"`
     inside `"microphone"` / `"telephone"` (a **voice** platform),
     `"key"` inside `"cache_key"` / `"idempotency_key"`, `"card"` inside
     `"scorecard"` / `"discard"`, `"city"` inside `"capacity"` /
     `"velocity"`, and `"upi"` inside `"occupied"` -- and it wrongly denied
     real, non-PII operator-config fields `"supported_banks"`,
     `"blocked_banks"`, `"upi_supported"`, `"mobile_app"`, `"android"`, and
     `"kyc_documents_required"`. This is exactly the metadata the plan says
     must survive at full fidelity for a trace to be useful for debugging
     control flow.

  The fix (`_tokenize_key`) splits each key into whole tokens on
  separators (`_`, `-`, `.`, space, ...), camelCase boundaries, and
  letter/digit boundaries (`"UPI_ID"` -> `["upi", "id"]`,
  `"PgsOrderId"` -> `["pgs", "order", "id"]`,
  `"account_holder_name"` -> `["account", "holder", "name"]`), and every
  deny rule matches a **whole token**, never a substring of one. This alone
  fixes every false positive above: `"span"`, `"bid"`, `"valid"`,
  `"microphone"`, `"scorecard"`, `"capacity"`, `"occupied"` are each a
  single, unsplit token that is simply not equal to any deny term.

  A handful of terms are still genuinely ambiguous even at the whole-token
  level, because the SAME token means different things depending on its
  neighbors on this specific platform, so they get a small qualifier rule
  each (see `_AMBIGUOUS_TOKEN_TERMS`) instead of a bare presence check:

  - `"name"` -- denied only when it's the entire key (bare `"name"`) or
    paired with a person-referring token (`customer`, `account`, `holder`,
    `beneficiary`, `player`, `user`, `first`, `last`, `full`, `nominee`,
    `contact`, `applicant`, `guardian`). This is what lets
    `"customer_name"` be denied while `"market_name"` / `"game_name"` /
    `"tool_name"` / `"span_name"` are not -- both shapes end in the same
    `"name"` token, and only the human-identifying reading is in scope.
  - `"mobile"` -- denied only when bare or paired with `number`/`no`/`num`.
    `"Mobile_Number"` is denied; `"mobile_app"` (a feature-flag field, real
    key in the CRM contract) is not.
  - `"upi"` / `"kyc"` -- denied unless paired with a capability/config
    qualifier (`supported`, `enabled`, `available`, `allowed`, and for
    `kyc` also `required`). `"upi_id"` and bare `"kyc"` are denied;
    `"upi_supported"` and `"kyc_documents_required"` (both real,
    non-customer-specific config keys in the CRM contract) are not.
    Plain `"kyc_status"` / `"kyc_documents"` (a SPECIFIC customer's KYC
    state/submitted-document-types) have no such qualifier and stay
    denied, same as before this fix.
  - `"key"` -- denied only when bare or paired with a
    secret-indicating token (`api`, `secret`, `auth`, `access`, `private`,
    `license`, `encryption`). `"api_key"` is denied; `"cache_key"` /
    `"idempotency_key"` (ordinary engineering plumbing, not a credential)
    are not.
  - Date-of-birth spelled out as separate words (`"date_of_birth"`,
    `"dateOfBirth"`) is handled as a two-token conjunction (`"date"` AND
    `"birth"` both present) rather than folding `"date"` alone into the
    deny set, which would sweep up every ordinary `*_date`/`*_at` timestamp
    field on the platform (`placed_at`, `credited_at`, `settled_at`, ...).

  All non-ambiguous terms (`phone`, `email`, `vpa`, `ifsc`, `pan`,
  `aadhaar`, `aadhar`, `dob`, `address`, `bank`, `beneficiary`, `card`,
  `balance`, `token`, `password`, `otp`, `city`, `account`, `id`) are
  matched by simple whole-token presence -- no qualifier needed, and no
  false positive against this platform's real vocabulary was found for any
  of them. Note `"id"` needs no special prefix/suffix handling anymore
  (an earlier version of this module special-cased it that way): once
  matching is token-based, `"bet_id"` naturally tokenizes to
  `["bet", "id"]` with `"id"` as its own token, while `"bid"`/`"grid"`/
  `"avoid"` never produce an `"id"` token at all because there's no
  separator or case-boundary inside them to split on. `"id"` alone still
  sweeps `bet_id`/`transaction_id`/any other business-object id, on top of
  the `PgsOrderId`/`bank_saved` points made elsewhere in this docstring --
  entirely consistent with, and actually reinforcing, this module's
  "opposite of tool_executor.py" design goal. This has real teeth in the
  test suite: `test_trace_redaction.py` checks both directions (every
  round-1 target must still be caught, every token-boundary false positive
  found above must not be) rather than only the "must catch" direction,
  which is what let the substring-matching regression through undetected
  the first time.
- **Dict KEYS are pattern-scrubbed too, not just values -- and this applies
  regardless of the key's original type, not just `str` keys.** A payload
  keyed by a literal PII-shaped value (e.g. a mapping keyed by a raw mobile
  number, a UUID, or even a non-string key like an `int` or `bytes` value)
  would otherwise leak that key verbatim even though every value gets full
  treatment. Every dict key is stringified (if it isn't already a `str`)
  and then run through the same `_scrub_text` pipeline used for values,
  independent of (and in addition to) the deny-list hard-drop check above,
  which operates on the key's *meaning* (is this a field name we know is
  PII-adjacent) rather than the key's *shape* (does this specific key's
  text look like PII). An earlier version of this module only scrubbed
  keys that were already `str` and passed any other key type through
  `else k` unchanged -- a `{9876543210: "note"}` or `{b"9876543210":
  "note"}` mapping leaked its key completely. Stringifying every non-`str`
  key before scrubbing closes that gap entirely rather than only adding
  more special-cased type checks. If two distinct original keys happen to
  scrub to the same output key (e.g. two different phone numbers both
  becoming `"<PHONE>"`), a stable `"~2"`/`"~3"`/... counter suffix is
  appended on collision so the second value doesn't silently overwrite the
  first in the output dict.
- **`redact_text` raises loudly, not silently, on non-`str` input.** An
  earlier version silently coerced any non-string input via `str(text)`
  before scrubbing it as free text — which, for something like a `dict`
  argument passed by mistake, means it gets the *weaker* free-text pipeline
  (no key-based protection at all) instead of an error or the
  structure-aware pipeline. `redact_text` now raises `TypeError` on
  non-`str` input, which the fail-closed wrapper immediately turns into the
  loud, safe `"<redaction-failed>"` result rather than a quiet, weaker one.

## Known limitations (deliberately accepted, not fixed -- read before
"fixing" any of these; each was a recorded decision, not an oversight)

- **Not idempotent when a dict key itself scrubs to a placeholder that is
  also a deny token.** `{"9876543210": {"a": 1}}` redacts once to
  `{"<PHONE>": {"a": 1}}`; redacting THAT again denies the whole subtree,
  because `"<PHONE>"` (stripped of its angle brackets by `_tokenize_key`'s
  separator split) tokenizes to `["phone"]`, which is itself a deny term.
  The direction is always safe (strictly MORE redacted on a second pass,
  never less), so this is accepted rather than fixed. `TestIdempotence`
  pins this behavior explicitly with a PII-shaped-key payload so it's
  visible and tested rather than a silent surprise.
- **`span_id`/`trace_id`-shaped keys still deny via the blunt `id` simple
  term**, same as `bet_id`/`transaction_id`/any other business-object id
  (see the `_is_denied_key` design notes above) -- this was never on the
  round-2 or round-3 false-positive fix list, and real impact is limited
  in practice (a tracing backend sets its own `span_id`/`trace_id` via the
  OTel SDK, not by passing a dict through `redact_structure`), but it's
  worth a future maintainer knowing this module will redact those key
  names too if it's ever handed one directly.
- **`_PHONE_RE` allows only ONE separator**, positioned after the first 5
  digits (covering "9876543210" and "98765-43210"/"+91-98765-43210"). A
  3-3-4 grouping like `"987-654-3210"` is not recognized as one phone
  number and partially survives as `"987-654-<NUM>"` (the digit-run
  catch-all still claims the last group). 3-3-4 is not how Indian mobile
  numbers are conventionally grouped (5-5 and ungrouped are both far more
  common), so this is accepted as a narrow, low-likelihood gap rather than
  broadening the pattern's separator tolerance and risking new
  false-positive matches elsewhere.
- **The key-based deny-list is structural, not content-aware, by design.**
  It inspects dict KEYS during the recursive walk; a JSON object carried
  as a serialized STRING leaf (e.g. a tool response body that embeds
  `'{"first_name": "Rajesh"}'` as one big string value rather than a
  parsed nested object) gets only the value-pattern pipeline, which has no
  backstop for a bare human name (see the `name`/`key` polarity notes
  above -- this is exactly the failure mode those exist to prevent for
  PARSED structures, and it does not extend to unparsed ones). Callers
  should parse a JSON-shaped payload into a real nested structure before
  handing it to `redact_structure` rather than passing it through as a
  string.
- **Non-ASCII keys tokenize to nothing and can never deny.** `_tokenize_key`
  only recognizes `[A-Za-z0-9]` as token characters; a key like `"मोबाइल"`
  (Hindi for "mobile") produces an empty token list, so `_is_denied_key`
  returns `False` unconditionally for it (see the early-return on empty
  `tokens`). This platform's actual CRM contract and internal field names
  are consistently ASCII/English (see `docs/crm-api-contract.md`), so this
  is accepted as out of scope rather than adding Unicode-aware
  tokenization for a case with no known real occurrence.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------

# Returned by both public entry points when anything in this module raises.
# Never partial, never the raw input -- see the "Fail closed" section above.
REDACTION_FAILED_PLACEHOLDER = "<redaction-failed>"

# Placeholder written in place of:
#   (a) any value whose *key* falls in the deny set below, regardless of
#       the value's type, and
#   (b) any leaf value whose *type* is not one of the explicitly recognized
#       safe types (see _redact_value's fallback branch).
REDACTED_KEY_PLACEHOLDER = "<REDACTED>"

# ---------------------------------------------------------------------------
# Key-based hard drop -- applied BEFORE value scrubbing on the recursive walk
# ---------------------------------------------------------------------------

# Tokenizer: splits a key into whole tokens on separators, camelCase
# boundaries, and letter/digit boundaries. See module docstring's "Design
# notes" for why token-aware matching replaced both exact matching (too
# narrow) and unanchored substring matching (too broad -- it matched "pan"
# inside "span_kind", "id" inside "bid"/"grid"/"avoid", etc).
#
# _KEY_SEPARATOR_SPLIT_RE splits on any run of non-alphanumeric characters
# (_, -, ., space, ...). _KEY_SUBTOKEN_RE then splits each separator-free
# part on camelCase and letter/digit boundaries: an all-caps run followed
# by a Titlecase word (acronym boundary, e.g. "ID" in "IDCard"), a
# Titlecase-or-lowercase run, an all-caps run on its own, or a digit run.
_KEY_SEPARATOR_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
_KEY_SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _tokenize_key(key: object) -> list:
    """Split a dict key into lowercase whole tokens, e.g.
    "account_holder_name" -> ["account", "holder", "name"],
    "UPI_ID" -> ["upi", "id"], "PgsOrderId" -> ["pgs", "order", "id"]."""
    s = str(key)
    tokens: list = []
    for part in _KEY_SEPARATOR_SPLIT_RE.split(s):
        if not part:
            continue
        tokens.extend(m.lower() for m in _KEY_SUBTOKEN_RE.findall(part))
    return tokens


# Terms matched by simple whole-token presence -- no known false positive
# against this platform's real key vocabulary (see module docstring).
# Includes a small set of closed compounds ("surname", "cardholder",
# "privatekey") that are single, unsplittable tokens on their own -- the
# tokenizer cannot decompose them into their constituent words (there is
# no separator or case boundary inside "surname" to split on), so they are
# listed here directly rather than relying on any token-conjunction rule
# to reach them. See module docstring's "Design notes" (round-3 fix).
_SIMPLE_TOKEN_TERMS = {
    "phone", "email", "vpa", "ifsc", "pan", "aadhaar", "aadhar",
    "dob", "address", "bank", "beneficiary", "card", "balance",
    "token", "password", "otp", "city", "account", "id",
    # Credential/secret terms (round-3 fix F2) -- see module docstring.
    "secret", "credential", "credentials", "cvv", "cvc", "passcode",
    "pwd", "pin", "signature", "jwt", "bearer", "cookie", "privatekey",
    # Person-name closed compounds (round-3 fix F1) -- see module
    # docstring.
    "surname", "cardholder",
}

# Terms that are ambiguous at the whole-token level on THIS platform (the
# same token means different things depending on its neighbors -- see
# module docstring's "Design notes" for the concrete collisions each rule
# resolves). Each entry is (term, mode, companion_tokens):
#   "only_with"  -- denied ONLY when the key is exactly this token alone,
#                   OR paired with one of companion_tokens. A DEFAULT-ALLOW
#                   design: safe only when companion_tokens can be
#                   exhaustively enumerated, because anything NOT in
#                   companion_tokens silently passes through.
#   "except_with" -- denied UNLESS paired with one of companion_tokens. A
#                   DEFAULT-DENY design: safe by construction against an
#                   unanticipated qualifier, because only a SPECIFIC,
#                   deliberately-curated set of companions is exempted --
#                   everything else denies. Preferred whenever the "safe"
#                   side is the smaller, more enumerable one.
#
# "name" and "key" were BOTH originally "only_with" (default-allow) and
# both leaked in production use: an independent review round found real
# CRM-contract keys like "agent_name" (docs/crm-api-contract.md) and
# "middle_name"/"father_name"/"payer_name"/... sailing straight through
# because they simply weren't among the 13 enumerated person-referring
# qualifiers -- and human names have NO value-level pattern backstop
# (no regex matches "Rajesh Kumar"), so a miss here reaches a trace
# backend completely unredacted with nothing downstream to catch it.
# Likewise "signing_key"/"master_key" leaked under "key"'s old
# default-allow list. Both are now "except_with": the SAFE side (business
# object nouns for "name"; known non-secret key kinds for "key") is the
# one that's actually practical to enumerate, so an unanticipated
# `*_name`/`*_key` qualifier now fails closed (denied) instead of open.
_AMBIGUOUS_TOKEN_TERMS = (
    ("mobile", "only_with", frozenset({"number", "no", "num"})),
    ("upi", "except_with", frozenset({"supported", "enabled", "available", "allowed"})),
    ("kyc", "except_with", frozenset({"supported", "enabled", "available", "allowed", "required"})),
    # Non-secret key KINDS -- ordinary engineering plumbing, not a
    # credential. Anything else paired with "key" (a bare "key", or any
    # qualifier not in this set -- "api", "secret", "signing", "master",
    # "private", "encryption", "rotation", "backup", ...) denies.
    ("key", "except_with", frozenset({
        "cache", "idempotency", "sort", "primary", "partition",
        "composite", "foreign", "unique", "hash", "index",
    })),
    # Business/system-object nouns -- "the market's name", "the span's
    # name" -- not a human's. Anything else paired with "name" (a bare
    # "name", or any qualifier not in this set -- "customer", "father",
    # "payer", "winner", "agent", ...) denies.
    ("name", "except_with", frozenset({
        "market", "game", "event", "tool", "span", "brand", "product",
        "provider", "session", "file", "host", "tenant", "field",
        "column", "table", "class", "function", "module", "package",
        "job", "queue", "topic", "channel", "template", "variant",
        "tier", "plan", "sport", "league", "tournament", "team",
        "category", "currency", "language", "timezone", "country",
        "state", "method",
    })),
)

# Date-of-birth spelled out as separate words -- a two-token conjunction
# rather than folding "date" alone into _SIMPLE_TOKEN_TERMS (which would
# sweep up every ordinary *_date/*_at timestamp field on the platform).
_DOB_PHRASE_TOKENS = frozenset({"date", "birth"})


def _is_denied_key(key: object) -> bool:
    tokens = _tokenize_key(key)
    if not tokens:
        return False
    tokset = frozenset(tokens)

    if _DOB_PHRASE_TOKENS <= tokset:
        return True

    if tokset & _SIMPLE_TOKEN_TERMS:
        return True

    standalone = len(tokens) == 1
    for term, mode, companions in _AMBIGUOUS_TOKEN_TERMS:
        if term not in tokset:
            continue
        if mode == "only_with":
            if standalone or (companions & tokset):
                return True
        else:  # "except_with"
            if not (companions & tokset):
                return True
    return False


# ---------------------------------------------------------------------------
# String-value scrubber patterns -- longest/most-specific first, blunt
# digit-run catch-all second-to-last, per plan §2.4.
# ---------------------------------------------------------------------------

# UUID: hex-character lookarounds instead of \b -- see module docstring's
# "Design notes" for why \b is insufficient here (it doesn't fire between
# two word characters, e.g. a letter directly followed by a digit).
_UUID_RE = re.compile(
    r"(?<![0-9a-fA-F])"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"(?![0-9a-fA-F])"
)

# Standard email: requires a dot-separated TLD, which is exactly what
# distinguishes it from a UPI/VPA handle below (see module docstring).
# Bounded quantifiers throughout (local part <= 64 chars, each domain label
# <= 63 chars, <= 8 labels, TLD <= 24 chars) -- see module docstring's
# "Design notes" on the ReDoS fix. These bounds are generous versus any
# real email address and only exist to cap backtracking work per candidate
# match position to a small constant, independent of input length.
_EMAIL_LOCAL = r"[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}"
_EMAIL_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_EMAIL_RE = re.compile(
    rf"\b{_EMAIL_LOCAL}@(?:{_EMAIL_DOMAIN_LABEL}\.){{1,8}}[A-Za-z]{{2,24}}\b"
)

# UPI/VPA: same local@domain shape as an email but with no dot in the
# domain part (e.g. "ram@paytm", "player@upi", "operatorpay@hdfcbank").
# Runs strictly after the email pattern above, so a real email is never
# double-tagged. Its character class ([a-zA-Z]) already spans both cases
# directly, with no re.IGNORECASE flag needed or set. Bounded quantifiers
# for the same ReDoS reason as the email pattern above.
_UPI_RE = re.compile(r"\b[\w.\-]{1,64}@[a-zA-Z]{3,30}\b")

# Indian mobile number: optional country-code prefix (+91 / 91), then 10
# digits starting 6-9, with at most one separator (space or hyphen) allowed
# after the first 5 digits (covers both "9876543210" and the
# "+91-98765-43210" / "98765-43210" grouped forms seen in real profile
# payloads). Bounded by digit-adjacency lookarounds ((?<!\d) / (?!\d)), NOT
# \b word boundaries, so this never partially claims digits out of a
# longer run (a 12-digit Aadhaar number, a 14-digit account number) -- see
# module docstring.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}(?!\d)")

# Currency amount: ₹ / Rs / Rs. / INR prefix, optional whitespace, then a
# comma/decimal-grouped number. The negative lookbehind for a preceding
# letter stops the case-insensitive "Rs" alternative from matching inside
# ordinary words (e.g. "hours 2400" previously mis-matched starting at the
# "rs" inside "hours") -- see module docstring's "Design notes".
_CURRENCY_RE = re.compile(
    r"(?<![A-Za-z])(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?", re.IGNORECASE
)

# IFSC code: 4 letters, a literal '0', 6 more alphanumeric characters (the
# real format banks issue, e.g. "HDFC0001234"). Case-insensitive: erring
# toward redacting more, not less.
_IFSC_RE = re.compile(r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b")

# PAN: 5 letters, 4 digits, 1 letter (e.g. "ABCDE1234F").
_PAN_RE = re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")

# Aadhaar-shaped: 12 digits, optionally grouped in 4s with a single space or
# hyphen separator between groups (e.g. "1234 5678 9012", "1234-5678-9012",
# or bare "123456789012"). "Shaped" deliberately, not validated (no Verhoeff
# checksum) -- this module has no way to know if a 12-digit number is a real
# Aadhaar, and per the plan's bias, a false positive here (over-redaction)
# is the acceptable failure mode, not a false negative.
_AADHAAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

# Dates: ISO-8601 date/datetime (e.g. "2026-06-20" or
# "2026-06-20T10:30:00Z"/with fractional seconds) and common DD/MM/YYYY or
# DD-MM-YYYY forms. Deliberately applied BEFORE the digit-run catch-all
# below -- see the module docstring's "Design notes" for why.
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
)

# Catch-all: any remaining digit run of length >= 4. Deliberately blunt and
# deliberately LAST (after every pattern above, including dates) -- account
# numbers, transaction ids, order/bid refs, card fragments, and OTPs are all
# just digit runs, and this is intentionally not trying to enumerate each
# business object's id format. Anything more specific above has already
# claimed its match by the time this runs.
_DIGIT_RUN_RE = re.compile(r"\d{4,}")

# Ordered pipeline: (pattern, replacement) tuples applied in sequence to a
# single string. Order matters -- see comments above and the module
# docstring's "Design notes" section.
_STRING_SCRUB_PIPELINE = (
    (_UUID_RE, "<UUID>"),
    (_EMAIL_RE, "<EMAIL>"),
    (_UPI_RE, "<UPI>"),
    (_PHONE_RE, "<PHONE>"),
    (_CURRENCY_RE, "<AMOUNT>"),
    (_IFSC_RE, "<IFSC>"),
    (_PAN_RE, "<PAN>"),
    (_AADHAAR_RE, "<AADHAAR>"),
    (_DATE_RE, "<DATE>"),
    (_DIGIT_RUN_RE, "<NUM>"),
)


def _scrub_text(text: str) -> str:
    """Run a single string through the ordered pattern pipeline. Shared by
    both public entry points -- this is the only place pattern order is
    defined."""
    for pattern, placeholder in _STRING_SCRUB_PIPELINE:
        text = pattern.sub(placeholder, text)
    return text


# ---------------------------------------------------------------------------
# Recursive structure walk -- shape copied from
# src/chatbot/tool_executor.py::_redact_internal_ids (dict/list/str
# recursion, normalized-key matching), ruleset NOT copied. See module
# docstring for why.
# ---------------------------------------------------------------------------


def _redact_value(value: object) -> object:
    if isinstance(value, dict):
        result: dict = {}
        for k, v in value.items():
            # Dict keys are pattern-scrubbed too (a key that is itself
            # PII-shaped, e.g. a raw phone number or UUID used as a
            # mapping key, must not leak) -- independent of the deny-list
            # check below, which looks at the key's MEANING, not its
            # shape. See module docstring's "Design notes". Every key is
            # stringified first (not just `str` keys) so a non-string key
            # (an int, bytes, tuple, ...) gets the same treatment instead
            # of passing through raw.
            out_key = _scrub_text(k if isinstance(k, str) else str(k))
            if out_key in result:
                # Two distinct original keys scrubbed to the same output
                # key (e.g. two different phone-number-shaped keys both
                # becoming "<PHONE>") would otherwise silently collide,
                # with the second one overwriting the first in `result`.
                # Disambiguate with a stable counter suffix rather than
                # dropping data silently.
                suffix = 2
                candidate = f"{out_key}~{suffix}"
                while candidate in result:
                    suffix += 1
                    candidate = f"{out_key}~{suffix}"
                out_key = candidate
            if _is_denied_key(k):
                # Key-based hard drop: replace the VALUE regardless of its
                # type, and do NOT recurse into it or run it through the
                # string scrubber -- see module docstring. The deny check
                # runs against the ORIGINAL key (not out_key) since
                # normalization for meaning-matching is independent of
                # shape-scrubbing.
                result[out_key] = REDACTED_KEY_PLACEHOLDER
            else:
                result[out_key] = _redact_value(v)
        return result
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v) for v in value)
    if isinstance(value, str):
        return _scrub_text(value)
    if value is None or isinstance(value, bool):
        # Checked before the int/float branch below: bool is a subclass of
        # int in Python, and neither bool nor None carries PII.
        return value
    if isinstance(value, (int, float)):
        # Numeric leaves carry PII-shaped data too (e.g. a raw
        # "real_balance": 4250.75 float) -- see module docstring's "Design
        # notes". Only rewrite the leaf if scrubbing actually changed
        # something; otherwise preserve the original numeric type so a
        # harmless "total": 25 doesn't get needlessly stringified.
        as_str = str(value)
        scrubbed = _scrub_text(as_str)
        return scrubbed if scrubbed != as_str else value
    # Deny-by-default at the TYPE level: anything not explicitly recognized
    # above (set, frozenset, bytes, bytearray, Decimal, datetime/date, a
    # custom object, a dataclass instance, ...) is fully redacted rather
    # than ever returned raw or guessed at -- see module docstring's
    # "Design notes".
    return REDACTED_KEY_PLACEHOLDER


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def redact_structure(value: object) -> object:
    """Redact an arbitrary nested structure (dict/list/tuple/str/number/etc)
    for use in a trace -- the entry point for tool arguments and tool
    response bodies.

    Applies the key-based hard drop (see _is_denied_key) before any value
    scrubbing, scrubs dict keys themselves for PII-shaped content, and runs
    every remaining string/number leaf through the pattern pipeline in
    _scrub_text. Any value whose type isn't explicitly recognized (dict,
    list, tuple, str, bool, None, int, float) is fully redacted rather than
    returned raw -- see the module docstring's "Design notes".

    Fails closed: any internal exception (including runaway recursion on a
    circular reference) is caught and this function returns the literal
    string "<redaction-failed>" instead of the raw input or a partially
    redacted structure. Callers must treat that return value as "drop this
    data," never as data itself.
    """
    try:
        return _redact_value(value)
    except Exception:
        return REDACTION_FAILED_PLACEHOLDER


def redact_text(text: str) -> str:
    """Redact free text (an LLM completion, a user message) for use in a
    trace. Same pattern pipeline as redact_structure, but with no key-based
    logic -- free text has no keys.

    Raises internally (caught by the fail-closed wrapper below) if `text`
    is not a `str` -- silently coercing a non-string argument here would
    mean it goes through the WEAKER free-text pipeline (no key-based
    protection) by accident rather than by the caller's intent. See the
    module docstring's "Design notes".

    Fails closed: any internal exception is caught and this function
    returns the literal string "<redaction-failed>" instead of the raw or
    partially redacted text.
    """
    try:
        if not isinstance(text, str):
            raise TypeError(
                f"redact_text() requires a str, got {type(text).__name__}; "
                "use redact_structure() for non-string values instead of "
                "letting this silently degrade to the free-text pipeline."
            )
        return _scrub_text(text)
    except Exception:
        return REDACTION_FAILED_PLACEHOLDER


# A bare '@' is common, ordinary betting-platform text with no PII
# implication at all -- odds notation ("back @ 1.85"), a social handle
# ("follow us @kalyanmatka"). Flagging every bare '@' would make
# redaction_post_condition_ok fire constantly on totally clean text once a
# later phase wires it into the plan's §2.5 exporter (which is specified to
# log an ERROR on every trip, treating it as a bug report) -- see module
# docstring. Only flag '@' when it's flanked by an alphanumeric character
# on BOTH sides, i.e. still shaped like the local@domain part of a missed
# email/UPI match ("ram@paytm" flags; "back @ 1.85" and "@kalyanmatka",
# each missing a flanking alphanumeric on at least one side, do not).
_FLANKED_AT_RE = re.compile(r"[A-Za-z0-9]@[A-Za-z0-9]")


def redaction_post_condition_ok(value: object) -> bool:
    """Check whether a value's text form is free of the pattern-based
    sensitive shapes this module targets: no email/UPI-shaped '@' (an
    alphanumeric character flanking it on both sides -- see
    _FLANKED_AT_RE), no digit run of length >= 4, and no UUID-shaped
    substring.

    This is the exact check this module's own test suite uses to validate
    the realistic-payload and free-text redaction tests, factored out here
    as the single source of truth specifically so that a later phase's
    defense-in-depth re-assertion (the plan's §2.5 wrapping SpanExporter,
    which re-checks every span attribute value right before export) can
    import and reuse it rather than re-deriving an equivalent check that
    could silently drift out of sync with what this module actually
    guarantees.

    Deliberately NOT a check for "no deny-set key's original value" -- that
    is a property of the mapping from an ORIGINAL input to its redacted
    output, not something recoverable from an output value in isolation. A
    later stage (like the exporter above) only ever sees the
    already-redacted value it is about to export, never what it started
    as, so this only checks what such a stage could plausibly re-verify.
    """
    text = value if isinstance(value, str) else repr(value)
    if _FLANKED_AT_RE.search(text):
        return False
    if _DIGIT_RUN_RE.search(text):
        return False
    if _UUID_RE.search(text):
        return False
    return True
