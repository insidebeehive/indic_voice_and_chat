"""Execute a tenant-registered CRM tool as an HTTP call (PRD §4.6).

Parameter sources: ``"llm"`` params come from the model's tool-call arguments;
``"session"`` params come from the chat session context (e.g. customer_id).
``{param}`` placeholders in the endpoint are substituted; the rest become query
params (GET) or a JSON body (other methods). Auth is bearer / api-key with a
token the caller resolves (decrypted from tenant_secrets) — never logged.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


async def execute_crm_tool(
    *,
    endpoint: str,
    method: str,
    parameters: dict,
    auth_type: Optional[str],
    token: Optional[str],
    args: dict,
    context: Optional[dict] = None,
    extra_headers: Optional[dict] = None,
    http_client: object = None,
    timeout_s: float = 10.0,
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

    method = (method or "GET").upper()
    client = http_client
    own = client is None
    if own:
        import httpx
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=5.0))
    try:
        if method == "GET":
            resp = await client.get(url, params=rest, headers=headers)
        else:
            resp = await client.request(method, url, json=rest, headers=headers)
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 — non-JSON response
            body = {"text": resp.text}
        return {"status_code": resp.status_code, "data": body}
    except Exception as e:  # noqa: BLE001 — a failing CRM call must not kill the turn
        log.exception("crm tool http call failed", extra={"endpoint": endpoint})
        return {"error": str(e)}
    finally:
        if own:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
