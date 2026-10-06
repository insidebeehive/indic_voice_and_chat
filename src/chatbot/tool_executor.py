"""Execute a tenant-registered CRM tool as an HTTP call (PRD §4.6).

Parameter sources: ``"llm"`` params come from the model's tool-call arguments;
``"session"`` params come from the chat session context (e.g. customer_id).
``{param}`` placeholders in the endpoint are substituted; the rest become query
params (GET) or a JSON body (other methods). Auth is bearer / api-key with a
token the caller resolves (decrypted from tenant_secrets) — never logged.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

# A value destined for a `{placeholder}` in the endpoint PATH must not be
# able to change the shape of the path itself. `/` or `\` lets a value splice
# in extra path segments (or a full second path after a host-relative
# escape); `?`/`#` lets it terminate the path early and inject query params
# or a fragment the server-side router never intended; `..` is the classic
# traversal segment. This is checked BEFORE quote() below runs, not instead
# of it — quote() alone would happily percent-encode a literal ".." into
# "..%2F" or similar and some routers/proxies normalize percent-decoded path
# segments before matching, so encoding is not a substitute for rejecting the
# structurally dangerous shapes outright. Tenant-registered chat_tools rows
# (arbitrary endpoint templates, not just the two known-bad built-in catalog
# entries) make this a general placeholder-substitution guard, not a
# per-endpoint patch.
_PATH_VALUE_DANGEROUS_RE = re.compile(r"[/\\?#]|\.\.")

# A value that is EMPTY, or consists solely of one or more dots (".", "..",
# "...", ...), also reshapes the path even though a single "." doesn't match
# the two-dot traversal pattern above and quote() leaves a lone "." alone
# (it's in urllib's always-safe set). httpx normalises a "/./" path segment
# away client-side before the request is sent, so a tenant-registered
# template like ".../orders/{order_id}" with an LLM-sourced order_id="." is
# rewritten to ".../orders" — a different, real endpoint the template never
# described, reachable from a model-supplied argument alone. As of the
# current built-in catalog (src/chatbot/catalog.py), every default_path
# placeholder is {user_id} or {operator_id}, and both are source="session"
# on every catalog tool — never filled from an LLM-supplied argument — so
# this guard's only LIVE exposure today is a tenant-registered chat_tools
# row with an arbitrary endpoint template and an LLM-sourced path param, not
# the built-in catalog. An empty value collapses to "//", which some
# upstream routers normalise the same way. This is a check on the value AS A
# WHOLE (`^\.*$`), not on the dot character appearing anywhere in it — a
# legitimate value like "teen-patti.v2" contains a dot and must still be
# accepted.
_PATH_VALUE_EMPTY_OR_ALL_DOTS_RE = re.compile(r"^\.*$")

# Detects a `{name}` placeholder still present in the endpoint AFTER the
# substitution loop below has run. Normally that loop replaces every
# placeholder the row's `parameters` dict declares a value for; one surviving
# here means a declared source="session" (or "llm") param resolved to None
# (e.g. a catalog tool's {operator_id} when the per-session crm_context has no
# operator_id — see C1/W1, src/bootstrap.py's factory no longer fabricates a
# tenant.id fallback there) and was therefore skipped by the `v is not None`
# guard in that loop. Sending the request with a literal, percent-encoded
# "{operator_id}" path segment would be silently wrong (a guaranteed 404/403
# at the CRM) rather than loudly rejected here.
_UNRESOLVED_PATH_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Placeholder substituted for any internal-id-shaped value this module
# redacts (by key, in _redact_internal_ids, or by UUID pattern, in both
# _redact_internal_ids and _redact_url). Shared as a constant — rather than
# an inline literal repeated at each call site — so that
# src/chatbot/deposit_verification.py's "[redacted]" == missing-order-id
# guard (and its test) can import the same value and can never silently
# drift out of sync with what this module actually emits.
REDACTED_PLACEHOLDER = "[redacted]"

# Internal identifier keys stripped from every CRM tool response before it
# reaches the LLM. These are pure system-routing plumbing (never something a
# customer needs or should see) — echoed back by some CRM endpoints per their
# own contract (e.g. games-config includes "operator_id"), and real-world
# CRM responses can drift from their documented contract without our code
# knowing. The system prompt already has a "keep internals internal" rule,
# but that's an LLM instruction, not a guarantee — this makes leaking these
# specific fields structurally impossible instead of just discouraged.
# Matched case-insensitively (and ignoring underscores) against the key, so
# casing variants like "operatorID"/"USER_ID"/"Operator_Id" are also caught —
# see _REDACTED_KEYS_NORMALIZED below. Exact (normalized) key matching only:
# deliberately NOT a substring/pattern matcher, since legitimate
# customer-facing fields like bet_id/event_id/transaction_id must NOT be
# swept up by anything matching e.g. ".*_id$" or "id" in k.
_REDACTED_RESPONSE_KEYS = {
    "operator_id", "user_id", "tenant_id", "crm_id", "session_id",
}
_REDACTED_KEYS_NORMALIZED = {k.replace("_", "") for k in _REDACTED_RESPONSE_KEYS}

# Narrow, EXACT-match exemption from the UUID value-scrub below — not a
# general allowlist. These are payment-gateway order references that the
# platform must forward on to the deposit-verification vendor unmodified:
# "PgsOrderId" was the field name assumed before CRM PR #3963 shipped;
# get_player_latest_deposit_order's actual response (per that PR) carries
# the same reference as "order_id" (PgsIntegration.id, always present) with
# "external_transaction_id" as the gateway's own secondary reference (null
# on the INITIATED/BS_PENDING rows a deposit dispute is usually about). Both
# are UUID-shaped in practice, so without this exemption they would hit the
# UUID scrub below and come back as REDACTED_PLACEHOLDER, which is exactly
# what deposit_verification.py's missing_order_id guard treats as "no order
# id" — silently dead-ending every deposit dispute. "order_id" and
# "external_transaction_id" are the same class of customer-facing payment
# reference as the already-exempt bet_id/event_id/transaction_id class (see
# _REDACTED_RESPONSE_KEYS's comment above) — not internal platform ids this
# scrub was hardened against — so even a UUID-shaped value under one of
# these exact (normalized) keys must reach the LLM as-is instead of being
# redacted. Normalized the same way as _REDACTED_KEYS_NORMALIZED above
# (lowercased, underscores stripped).
# String params whose value is safe to log on the "crm tool call" line (see
# filter_params there): fixed filter words, never customer data.
_LOGGABLE_FILTER_PARAMS = ("type",)
_FILTER_VALUE_RE = re.compile(r"[A-Za-z_]{1,20}")

_ORDER_ID_EXEMPT_KEYS_NORMALIZED = {"pgsorderid", "orderid", "externaltransactionid"}

# Belt-and-suspenders value-level scrub: a UUID can leak through a key name
# that isn't in _REDACTED_RESPONSE_KEYS (e.g. "player_id", "customer_id", a
# bare "id") or embedded inside a free-text string value (e.g. a CRM error
# message like "No player found with id 6c1a77a6-...-8add58aba9ef"). Runs on
# every string value encountered during the recursive walk, in addition to
# (not instead of) the key-based redaction above.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# _ORDER_ID_EXEMPT_KEYS_NORMALIZED above exists to preserve a bare id value
# unmodified. But nothing stops a CRM from putting a free-text diagnostic
# under one of these same keys instead of a bare id (e.g.
# {"order_id": "no order for player 6c1a77a6-...-8add58aba9ef"}) — an
# unconditional exemption would let an embedded internal-id UUID inside that
# sentence leak straight through, defeating the UUID scrub above for exactly
# the keys this module cares most about. So the exemption only applies when
# the value actually LOOKS like a bare identifier — no whitespace, and a
# plausible id length — not merely because the key matches; anything else
# (free text, whitespace, absurdly long) falls through to the normal scrub,
# which still strips an embedded UUID via _UUID_RE.
#
# "No whitespace" alone is a not-prose test, not an is-an-identifier test: a
# hyphen- or colon-joined token like "no-order-for-6c1a77a6-...-8add58aba9ef"
# carries an embedded UUID and contains no whitespace, so it would still be
# forwarded verbatim. Hence the second condition at the call site: the value
# may be a UUID outright (the common case — order_id IS a UUID), or contain
# no UUID at all (gw-abc-123, PGS20260621143000123, UPI/2026/06/21/abc), but
# a value that merely *embeds* one is not a bare id and gets scrubbed. No
# charset allowlist, since hyphens/slashes/colons are all legal in real
# gateway references.
_BARE_ID_RE = re.compile(r"^\S{1,100}$")


def _redact_internal_ids(value: object) -> object:
    """Recursively strip _REDACTED_RESPONSE_KEYS from a CRM response body,
    plus scrub any UUID-shaped substring from string values.

    Applied once here so it covers every CRM endpoint automatically, rather
    than relying on each tool's response shape being individually audited.
    """
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            normalized_k = k.lower().replace("_", "")
            if normalized_k in _REDACTED_KEYS_NORMALIZED:
                result[k] = REDACTED_PLACEHOLDER
            elif (
                normalized_k in _ORDER_ID_EXEMPT_KEYS_NORMALIZED
                and isinstance(v, str)
                and _BARE_ID_RE.match(v)
                and (_UUID_RE.fullmatch(v) or not _UUID_RE.search(v))
            ):
                # EXACT-match exemption — see _ORDER_ID_EXEMPT_KEYS_NORMALIZED
                # and _BARE_ID_RE above. Preserve the payment-gateway order
                # reference unmodified, even if it happens to be UUID-shaped
                # — but only when it actually looks like a bare id; free text
                # under the same key still falls through to the scrub below.
                result[k] = v
            else:
                result[k] = _redact_internal_ids(v)
        return result
    if isinstance(value, list):
        return [_redact_internal_ids(v) for v in value]
    if isinstance(value, str):
        return _UUID_RE.sub(REDACTED_PLACEHOLDER, value)
    return value


def _redact_url(url: str) -> str:
    """Scrub UUID-shaped substrings from a resolved CRM URL before logging.

    ``{param}`` placeholders are substituted with real values before the call
    (e.g. /players/{user_id}/profile -> /players/6c1a77a6-.../profile), so the
    resolved URL carries a real player identifier that must not land in logs.
    Deliberately reuses _UUID_RE so URL scrubbing and response-body scrubbing
    can never drift apart.
    """
    return _UUID_RE.sub(REDACTED_PLACEHOLDER, url or "")


def _iter_list_locations(node: object):
    """Yield ``(container, key)`` for every list value found anywhere in
    *node*, however deeply nested — ``container[key]`` is that list, so
    assigning ``container[key] = ...`` replaces it in place.

    Walks into dict values and list items alike, so a list nested inside
    another list's items (e.g. a per-market ``bet_types`` list, one level
    inside the top-level ``markets`` list) is found too, not just top-level
    lists — see _apply_result_size_budget's "nested lists" handling.
    ``container`` is always a dict: a list value is only ever reachable
    through the dict key that holds it, which is also what lets
    _apply_result_size_budget attach a same-key-scoped truncation marker
    next to it.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, list):
                yield (node, k)
            yield from _iter_list_locations(v)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_list_locations(item)


# Field names a CRM list item carries its own timestamp under, tried in this
# order (see docs/crm-api-contract.md: "timestamp" on transactions,
# "placed_at" on bets/matka bids, "created_at" on a deposit order, ...). Not
# every list shape in the contract is sorted the same way — the transactions
# example is oldest-first — so a budget cut can't assume "the head is the
# newest" and must actually look at the data.
_TIMESTAMP_KEYS = ("timestamp", "created_at", "placed_at", "date", "updated_at", "approved_at")


def _parse_timestamp(value: object) -> Optional[float]:
    """Parse *value* to a comparable epoch float, or None if it isn't a
    recognisable timestamp.

    Accepts an ISO 8601 string (``fromisoformat`` handles an offset like
    ``+05:30``; a trailing ``Z`` is normalised to ``+00:00`` first, since
    ``fromisoformat`` on the Python versions this runs under doesn't accept
    ``Z`` itself) or an epoch number (int/float, or a numeric string).
    Anything else — unparseable text, None, a bool, a nested structure —
    returns None so the caller can fall back to keeping the head rather than
    sorting on a guess.
    """
    if isinstance(value, bool):
        return None  # bool is an int subclass; never treat True/False as an epoch
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        iso = s[:-1] + "+00:00" if s.endswith("Z") else s
        try:
            return datetime.fromisoformat(iso).timestamp()
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _newest_first(items: list) -> Optional[list]:
    """Rank *items* newest-first, for choosing which ones to KEEP when
    trimming a list, using a timestamp field every item shares -- or None
    if no such ordering can be trusted.

    Returns a list of ``(original_index, item)`` pairs ranked by
    ``(timestamp, original_index)`` descending -- newest first, ties
    preferring the item later in the original list. This ranking picks
    WHICH items survive a trim; it is deliberately not the output order.
    The caller takes the first N pairs to keep, then sorts THOSE by
    original_index to restore the list's original relative order before
    writing it back — a trimmed list must come out in the same order as an
    untrimmed one, just shorter.

    Returns None (caller falls back to keeping the head, as for a list with
    no timestamp at all) when: the items aren't all dicts, no single
    _TIMESTAMP_KEYS field is present on every item, or any one item's value
    under that field fails to parse — a partial/best-effort sort here would
    risk silently keeping the WRONG items, which is worse than the old
    behavior this replaces.
    """
    if not items or not all(isinstance(it, dict) for it in items):
        return None
    ts_key = next((k for k in _TIMESTAMP_KEYS if all(k in it for it in items)), None)
    if ts_key is None:
        return None
    parsed = []
    for idx, it in enumerate(items):
        ts = _parse_timestamp(it.get(ts_key))
        if ts is None:
            return None
        parsed.append((idx, ts, it))
    parsed.sort(key=lambda triple: (triple[1], triple[0]), reverse=True)
    return [(idx, it) for idx, _ts, it in parsed]


def _apply_result_size_budget(result: dict, max_chars: int) -> dict:
    """Shrink *result*'s longest list(s) until its JSON serialization fits
    within *max_chars* — never cutting through an object, only ever
    dropping whole trailing items; a result already under budget is
    returned byte-identical (no mutation at all).

    A shrunk list isn't always safe to cut from the head: docs/crm-api-
    contract.md's own get_player_transactions example is oldest-first, so
    keeping the head there would drop the newest item — the one a dispute
    is usually about. So when a shrunk list's items are dicts that all carry
    one of a small set of recognised timestamp fields (_TIMESTAMP_KEYS), the
    kept items are the N NEWEST (see _newest_first), but written back out
    in their ORIGINAL relative order — a trimmed list looks like the same
    list with items missing, not reshuffled. Otherwise (no recognisable/
    consistent timestamp field, or a value that fails to parse) the HEAD is
    kept, same as before — the only ordering available without guessing,
    and fine where order isn't meaningful anyway (e.g. matka markets
    config).

    Candidate lists (see _iter_list_locations — this covers nested lists
    too) are tried largest-serialized-size first, since trimming the
    biggest list makes the most progress per item dropped. Each one gets a
    same-key-scoped marker, e.g. ``_bids_truncated``, so the model can tell
    the customer there's more instead of silently seeing a partial list.
    At least one item is always kept per shrunk list — this budget is
    allowed to overshoot *max_chars* on a single huge item/list, but it
    must never look like "no data" when there is some.
    """
    if len(json.dumps(result)) <= max_chars:
        return result
    candidates = sorted(
        _iter_list_locations(result),
        key=lambda c: len(json.dumps(c[0][c[1]])), reverse=True,
    )
    for container, key in candidates:
        if len(json.dumps(result)) <= max_chars:
            break
        items = container[key]
        original_len = len(items)
        if original_len <= 1:
            continue  # nothing to cut without emptying the one item it has
        ranked = _newest_first(items)  # (original_index, item) pairs, newest first
        marker_key = f"_{key}_truncated"
        for keep_n in range(original_len - 1, 0, -1):
            if ranked is not None:
                # Keep the N newest, then restore original relative order.
                kept = sorted(ranked[:keep_n], key=lambda pair: pair[0])
                container[key] = [it for _idx, it in kept]
            else:
                container[key] = items[:keep_n]  # no trustworthy ordering -- keep the head
            container[marker_key] = (
                f"showing {keep_n} of {original_len} items; if what you need "
                f"isn't here, call this tool again with a narrower filter "
                f"(e.g. market_name, type or a smaller limit)"
            )
            if len(json.dumps(result)) <= max_chars:
                break
    return result


# Read timeout for one CRM HTTP call. Was tightened to 10s during the 2026-08
# relay-timeout incident fix, on the theory that a short timeout would keep
# the chat WS from going quiet long enough for the CRM's downstream relay to
# declare the socket dead (1006) and auto-close the ticket mid-conversation.
# In practice, production monitoring showed real CRM calls routinely
# saturating the OLD 30s ceiling across multiple endpoints in a single
# window (/players/{id}/transactions: 50 occurrences, /wallet: 41, /bets: 2,
# /profile: 1) -- these were slow-but-working calls, not hung connections, and
# a short timeout was converting them into content-free holding answers
# instead of real ones. The relay-silence problem this timeout interacts with
# is now handled at the WS layer by a visible periodic interim message
# (src/api/chat.py: _interim_wait_keepalive) rather than by cutting the tool
# budget short, so the budget can be widened back out. A timeout here is
# still not a turn failure: the except below converts it to {"error": ...},
# the model gets that as the tool result and still answers (a holding
# message), so a longer budget only improves the answer's richness -- it
# never risks the turn itself.
#
# This constant is only a FALLBACK default for callers that don't pass an
# explicit per-call budget (e.g. a direct/manual call to execute_crm_tool).
# The real bound on tool time within a turn is enforced by the caller
# (src/agents/chatbot.py) via a per-turn CUMULATIVE tool budget
# (_TOOL_BUDGET_S), because a turn can involve more than 2 tool calls total —
# up to _max_tool_rounds rounds, each of which can itself carry multiple
# function_call parts in a single LLM response (Gemini does this). A fixed
# "2 calls total" assumption doesn't hold, so this default is kept in step
# with _TOOL_CALL_CEILING_S (the per-call ceiling chatbot.py enforces via
# asyncio.wait_for) rather than derived from the turn-level cap
# _TURN_TIMEOUT_S (src/api/chat.py) the way it previously was.
_DEFAULT_CRM_TOOL_TIMEOUT_S = 35.0

# Fallback default for callers (e.g. a direct/manual call to
# execute_crm_tool) that don't pass an explicit per-tenant budget — mirrors
# _DEFAULT_CRM_TOOL_TIMEOUT_S's role just above. The real, settings-driven
# value is ChatToolsConfig.crm_result_max_chars (src/config.py), threaded
# through by src/bootstrap.py's crm_executor on every production call.
# Prod data (2026-09) showed get_matka_bids alone averaging ~10KB/call with
# a p95 of ~32KB across 682 calls/15d, re-sent uncached on every later round
# of the same turn — see _apply_result_size_budget below.
_DEFAULT_CRM_RESULT_MAX_CHARS = 6000


async def execute_crm_tool(
    *,
    endpoint: str,
    method: str,
    parameters: dict,
    auth_type: Optional[str],
    token: Optional[str],
    args: dict,
    x_api_key: Optional[str] = None,
    context: Optional[dict] = None,
    session_id: Optional[str] = None,
    ticket_id: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    http_client: object = None,
    timeout_s: float = _DEFAULT_CRM_TOOL_TIMEOUT_S,
    result_max_chars: int = _DEFAULT_CRM_RESULT_MAX_CHARS,
) -> dict:
    context = context or {}
    values: dict = {}
    for pname, spec in (parameters or {}).items():
        source = (spec or {}).get("source", "llm")
        values[pname] = context.get(pname) if source == "session" else args.get(pname)

    def _reject_invalid_path_param(param_name: str) -> dict:
        # Reject, don't raise. The turn would survive either way — the caller
        # in src/agents/chatbot.py already catches Exception around the
        # executor — but a raise would be classified as "transport_error",
        # which is a lie: nothing was ever sent. Returning lets the model see
        # an accurate invalid_parameter failure, and avoids a log.exception
        # traceback whose message would echo the offending value back into
        # the logs. Log the param NAME only — never the value, which is
        # exactly the attacker-controlled (or coincidentally PII-shaped)
        # string this check exists to keep out of logs.
        log.warning("crm tool call rejected invalid path parameter", extra={
            "ticket_id": ticket_id, "session_id": session_id, "param_name": param_name,
        })
        return {
            "error": "One of the request parameters had an invalid value.",
            "failure": "invalid_parameter",
        }

    url = endpoint
    path_used = set()
    for k, v in values.items():
        placeholder = "{" + k + "}"
        if placeholder in url and v is not None:
            str_v = str(v)
            if (
                _PATH_VALUE_DANGEROUS_RE.search(str_v)
                or _PATH_VALUE_EMPTY_OR_ALL_DOTS_RE.match(str_v)
            ):
                return _reject_invalid_path_param(k)
            # safe="" (not the urllib default safe="/") because a `/` in the
            # ENCODED output would re-introduce exactly the path-splicing
            # this substitution must prevent — the default exists for
            # callers building whole paths, not single path segments.
            try:
                encoded = quote(str_v, safe="")
            except UnicodeEncodeError:
                # A lone UTF-16 surrogate (e.g. "\ud800") can arrive through
                # json.loads() of the model's tool-call arguments — Python's
                # JSON parser does not reject lone surrogates, but quote()
                # cannot encode one to UTF-8 and raises. Treat it as just
                # another rejected path value instead of letting it escape
                # this function as an unhandled exception.
                return _reject_invalid_path_param(k)
            url = url.replace(placeholder, encoded)
            path_used.add(k)
    # A `{name}` placeholder that survives the loop above means its resolved
    # value was None (e.g. no operator_id in session context) -- reject
    # rather than send a literal placeholder in the request path.
    _unresolved = _UNRESOLVED_PATH_PLACEHOLDER_RE.search(url)
    if _unresolved:
        return _reject_invalid_path_param(_unresolved.group(1))
    rest = {k: v for k, v in values.items() if k not in path_used and v is not None}

    headers: dict = dict(extra_headers or {})
    if token and auth_type == "bearer":
        headers["Authorization"] = f"Bearer {token}"
    elif token and auth_type == "api_key":
        headers["X-API-Key"] = token
    if x_api_key:
        # Independent of auth_type — always sent alongside whatever the
        # token/auth_type logic above produced (the live CRM requires both
        # Authorization and X-API-Key together). Runs unconditionally AFTER
        # the if/elif above so it deliberately wins if a tenant has both the
        # old-style api_key auth_type/token AND this new dedicated field
        # configured: x_api_key is the more specific, newer mechanism.
        headers["X-API-Key"] = x_api_key

    method = (method or "GET").upper()
    client = http_client
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=5.0))
    # Values for NUMERIC params only, alongside the keys-only list below.
    # A type check, not an allowlist of names: a new count-shaped filter
    # (page_size, offset, top_k) is captured the day it is added, where a
    # name allowlist would silently miss it. Strings are excluded BY
    # CONSTRUCTION, which is the whole point -- a resolved param value can be
    # a mobile, an email, or a player id, and this module already logs
    # param_keys rather than param values for exactly that reason (see the
    # comment on param_keys below). Do not "improve" this into logging
    # values generally.
    #
    # bools are excluded deliberately: isinstance(True, int) is True in
    # Python, so a flag would ride in unnoticed rather than by decision, and
    # a flag is not a size driver -- what this captures is how many records
    # were asked for, to be read against result_chars on the response line
    # below.
    numeric_params = {
        k: v for k, v in sorted(rest.items())
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    # The one string param whose VALUE is logged: `type` is a fixed filter
    # word (deposit | withdrawal | casino | sports) that decides which
    # records the CRM returns, so it is needed to read result_chars and the
    # model's answer against what was actually asked for. Logged only when it
    # looks like a short filter word -- anything else (a model stuffing free
    # text into it) stays keys-only like every other string.
    filter_params = {
        k: rest[k] for k in _LOGGABLE_FILTER_PARAMS
        if isinstance(rest.get(k), str) and _FILTER_VALUE_RE.fullmatch(rest[k])
    }
    log.info("crm tool call", extra={
        "ticket_id": ticket_id, "session_id": session_id,
        "url": _redact_url(url), "method": method,
        "filter_params": filter_params,
        # keys only — resolved param VALUES can be customer PII (mobile,
        # email) or a player id; same rule as header_keys below.
        "param_keys": sorted(rest.keys()),
        "numeric_params": numeric_params,
        "header_keys": list(headers.keys()),  # keys only — never log token values
    })
    try:
        if method == "GET":
            resp = await client.get(url, params=rest, headers=headers)
        else:
            resp = await client.request(method, url, json=rest, headers=headers)
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 — non-JSON response
            body = {"text": resp.text}
        # The response body is never logged, at any level: CRM bodies carry
        # real customer PII (get_player_profile returns mobile, email,
        # kyc_documents, bank_saved). Only url (UUID-scrubbed) + status_code
        # go to the log, below. Redact internal ids before the body reaches
        # the LLM's context — see _REDACTED_RESPONSE_KEYS.
        result: dict = {"status_code": resp.status_code, "data": _redact_internal_ids(body)}
        if resp.status_code >= 400:
            # A 4xx/5xx is a real failure even though the HTTP call itself
            # succeeded (no exception) — previously this had no "error" key
            # at all and was invisible to any failure check downstream.
            result["failure"] = "http_error"
            result["error"] = f"The upstream service returned an error (HTTP {resp.status_code})."
        # Size budget, AFTER redaction so the budget is spent on real content
        # rather than on ids this result never sends anyway, and BEFORE the
        # result_chars log below so that number (and
        # src/agents/chatbot.py's out_json/chat_tool_metrics.result_chars,
        # which re-serializes this same returned dict) both measure what the
        # model actually receives, not the pre-trim size.
        result = _apply_result_size_budget(result, result_max_chars)
        log.info("crm tool response", extra={
            "ticket_id": ticket_id, "session_id": session_id,
            "url": _redact_url(url), "status_code": resp.status_code,
            # Measures the same dict that becomes the tool message content
            # sent to the model (src/agents/chatbot.py's out_json), i.e.
            # this result AFTER _redact_internal_ids, the failure-key
            # additions above, AND the size-budget trim just above -- not the
            # raw wire body -- so it's directly comparable to
            # chat_tool_metrics.result_chars. The body itself is still never
            # logged, only its length.
            "result_chars": len(json.dumps(result)),
        })
        return result
    except Exception as e:  # noqa: BLE001 — a failing CRM call must not kill the turn
        log.exception("crm tool http call failed", extra={
            "ticket_id": ticket_id, "session_id": session_id, "endpoint": endpoint,
        })
        # Ticket #1762 root cause #1: str(httpx.ReadTimeout()) is "" for a bare
        # timeout with no message — the caller (src/agents/chatbot.py) fed that
        # literal empty string to the LLM as the tool's error, giving the "say
        # so honestly" prompt rule nothing to react to. Always produce a real,
        # non-empty phrase, and add a machine-readable "failure" discriminator
        # so the caller can classify without string-sniffing.
        if isinstance(e, httpx.TimeoutException):
            failure = "timeout"
            message = f"The request to fetch this data timed out ({type(e).__name__})."
        elif isinstance(e, httpx.HTTPError):
            failure = "transport_error"
            message = str(e) or f"A network error occurred ({type(e).__name__})."
        else:
            failure = "transport_error"
            message = str(e) or f"An unexpected error occurred ({type(e).__name__})."
        # A future raise_for_status() (or similar) could embed a UUID-bearing
        # URL/id in the exception text — scrub it before it reaches the LLM.
        return {
            "error": _UUID_RE.sub(REDACTED_PLACEHOLDER, message),
            "failure": failure,
        }
    finally:
        if own:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
