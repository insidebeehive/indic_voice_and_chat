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
from typing import Optional

log = logging.getLogger(__name__)

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
        return {
            k: (
                "[redacted]"
                if k.lower().replace("_", "") in _REDACTED_KEYS_NORMALIZED
                else _redact_internal_ids(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_internal_ids(v) for v in value]
    if isinstance(value, str):
        return _UUID_RE.sub("[redacted]", value)
    return value

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

    url = endpoint
    path_used = set()
    for k, v in values.items():
        placeholder = "{" + k + "}"
        if placeholder in url and v is not None:
            url = url.replace(placeholder, str(v))
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
        import httpx
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=5.0))
    log.info("crm tool call", extra={
        "ticket_id": ticket_id, "session_id": session_id,
        "url": url, "method": method, "params": rest,
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
            "url": url, "status_code": resp.status_code,
        })
        # TEMP DEBUG (remove after the matka-availability investigation):
        # log the actual response body so we can see what the CRM is really
        # returning, not just the status code. Truncated — bodies here are
        # small config/market payloads, not player PII.
        log.info("crm tool response body [TEMP DEBUG]", extra={
            "ticket_id": ticket_id, "session_id": session_id,
            "url": url, "body": json.dumps(body, default=str)[:4000],
        })
        # Redact internal ids AFTER logging (the raw body above is for our
        # own investigation, not customer-facing) but BEFORE this reaches the
        # LLM's context — see _REDACTED_RESPONSE_KEYS.
        return {"status_code": resp.status_code, "data": _redact_internal_ids(body)}
    except Exception as e:  # noqa: BLE001 — a failing CRM call must not kill the turn
        log.exception("crm tool http call failed", extra={
            "ticket_id": ticket_id, "session_id": session_id, "endpoint": endpoint,
        })
        # A future raise_for_status() (or similar) could embed a UUID-bearing
        # URL/id in the exception text — scrub it before it reaches the LLM.
        return {"error": _UUID_RE.sub("[redacted]", str(e))}
    finally:
        if own:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
