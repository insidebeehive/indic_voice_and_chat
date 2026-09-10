"""Execute a tenant-registered CRM tool as an HTTP call (PRD §4.6).

Parameter sources: ``"llm"`` params come from the model's tool-call arguments;
``"session"`` params come from the chat session context (e.g. customer_id).
``{param}`` placeholders in the endpoint are substituted; the rest become query
params (GET) or a JSON body (other methods). Auth is bearer / api-key with a
token the caller resolves (decrypted from tenant_secrets) — never logged.
"""

from __future__ import annotations

import logging
import re
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
# away client-side before the request is sent, so
# ".../markets/{market_name}/holiday-schedule" with market_name="." is
# rewritten to ".../markets/holiday-schedule" — a different, real endpoint
# the template never described, reachable from a model-supplied argument
# alone. An empty value collapses to "//", which some upstream routers
# normalise the same way. This is a check on the value AS A WHOLE (`^\.*$`),
# not on the dot character appearing anywhere in it — a legitimate value like
# "teen-patti.v2" contains a dot and must still be accepted.
_PATH_VALUE_EMPTY_OR_ALL_DOTS_RE = re.compile(r"^\.*$")

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
# general allowlist. "PgsOrderId" is the payment gateway's own order
# reference (see get_player_latest_deposit_order), which the platform must
# forward on to the deposit-verification vendor unmodified. It is not one of
# the internal platform ids this scrub was hardened against, so even a
# UUID-shaped value under this exact (normalized) key must reach the LLM
# as-is instead of being redacted. Normalized the same way as
# _REDACTED_KEYS_NORMALIZED above (lowercased, underscores stripped).
_PGS_ORDER_ID_KEY_NORMALIZED = "pgsorderid"

# Belt-and-suspenders value-level scrub: a UUID can leak through a key name
# that isn't in _REDACTED_RESPONSE_KEYS (e.g. "player_id", "customer_id", a
# bare "id") or embedded inside a free-text string value (e.g. a CRM error
# message like "No player found with id 6c1a77a6-...-8add58aba9ef"). Runs on
# every string value encountered during the recursive walk, in addition to
# (not instead of) the key-based redaction above.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


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
            elif normalized_k == _PGS_ORDER_ID_KEY_NORMALIZED and isinstance(v, str):
                # EXACT-match exemption — see _PGS_ORDER_ID_KEY_NORMALIZED
                # above. Preserve the payment-gateway order id unmodified,
                # even if it happens to be UUID-shaped.
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
    log.info("crm tool call", extra={
        "ticket_id": ticket_id, "session_id": session_id,
        "url": _redact_url(url), "method": method,
        # keys only — resolved param VALUES can be customer PII (mobile,
        # email) or a player id; same rule as header_keys below.
        "param_keys": sorted(rest.keys()),
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
        log.info("crm tool response", extra={
            "ticket_id": ticket_id, "session_id": session_id,
            "url": _redact_url(url), "status_code": resp.status_code,
        })
        # The response body is never logged, at any level: CRM bodies carry
        # real customer PII (get_player_profile returns mobile, email,
        # kyc_documents, bank_saved). Only url (UUID-scrubbed) + status_code
        # go to the log, above. Redact internal ids before the body reaches
        # the LLM's context — see _REDACTED_RESPONSE_KEYS.
        result: dict = {"status_code": resp.status_code, "data": _redact_internal_ids(body)}
        if resp.status_code >= 400:
            # A 4xx/5xx is a real failure even though the HTTP call itself
            # succeeded (no exception) — previously this had no "error" key
            # at all and was invisible to any failure check downstream.
            result["failure"] = "http_error"
            result["error"] = f"The upstream service returned an error (HTTP {resp.status_code})."
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
