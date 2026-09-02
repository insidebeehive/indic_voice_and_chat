"""CRM tool HTTP executor (Phase 3b)."""

from __future__ import annotations

import inspect
import logging

import httpx
import pytest
import respx

from src.chatbot import tool_executor
from src.chatbot.tool_executor import execute_crm_tool


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def get(self, url, params=None, headers=None):
        self.calls.append(("GET", url, params, headers))
        return _FakeResp(self._payload)

    async def request(self, method, url, json=None, headers=None):
        self.calls.append((method, url, json, headers))
        return _FakeResp(self._payload)


@pytest.mark.asyncio
async def test_get_substitutes_path_param_and_bearer_auth() -> None:
    client = _FakeClient({"status": "shipped"})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/orders/{order_id}/status",
        method="GET",
        parameters={"order_id": {"type": "string", "source": "llm"}},
        auth_type="bearer", token="secret-tok",
        args={"order_id": "ORD-9"}, http_client=client,
    )
    assert out == {"status_code": 200, "data": {"status": "shipped"}}
    method, url, params, headers = client.calls[0]
    assert url == "https://crm.example.com/api/orders/ORD-9/status"
    assert params == {}  # order_id consumed by the path
    assert headers["Authorization"] == "Bearer secret-tok"


@pytest.mark.asyncio
async def test_session_sourced_param_comes_from_context() -> None:
    client = _FakeClient({"history": []})
    await execute_crm_tool(
        endpoint="https://crm.example.com/api/customers/{customer_id}/history",
        method="GET",
        parameters={"customer_id": {"type": "string", "source": "session"}},
        auth_type=None, token=None,
        args={}, context={"customer_id": "cust_42"}, http_client=client,
    )
    assert client.calls[0][1].endswith("/customers/cust_42/history")


@pytest.mark.asyncio
async def test_non_path_args_become_query_params() -> None:
    client = _FakeClient({"ok": True})
    await execute_crm_tool(
        endpoint="https://crm.example.com/api/search",
        method="GET",
        parameters={"q": {"type": "string", "source": "llm"}},
        auth_type=None, token=None, args={"q": "phones"}, http_client=client,
    )
    assert client.calls[0][2] == {"q": "phones"}


@pytest.mark.asyncio
async def test_http_failure_returns_error_dict() -> None:
    class _Boom:
        async def get(self, *a, **k):
            raise RuntimeError("connection refused")

    out = await execute_crm_tool(
        endpoint="https://x/y", method="GET", parameters={},
        auth_type=None, token=None, args={}, http_client=_Boom(),
    )
    assert "error" in out


@pytest.mark.asyncio
async def test_bearer_token_and_x_api_key_both_sent_together() -> None:
    # The live CRM (apistage.betstudio.io) requires BOTH headers on every
    # call: Authorization: Bearer <token> AND X-API-Key: <x_api_key>. The new
    # x_api_key field is additive and independent of auth_type/token.
    client = _FakeClient({"ok": True})
    await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={},
        auth_type="bearer", token="bearer-tok",
        x_api_key="the-x-api-key",
        args={}, http_client=client,
    )
    headers = client.calls[0][3]
    assert headers["Authorization"] == "Bearer bearer-tok"
    assert headers["X-API-Key"] == "the-x-api-key"


@pytest.mark.asyncio
async def test_x_api_key_wins_over_old_api_key_auth_type() -> None:
    # Edge case: auth_type == "api_key" (the OLD single-token mode) already
    # sets X-API-Key from `token`. If the NEW x_api_key is also configured,
    # it must win (it's the more specific, newer mechanism).
    client = _FakeClient({"ok": True})
    await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={},
        auth_type="api_key", token="old-style-token",
        x_api_key="dedicated-x-api-key",
        args={}, http_client=client,
    )
    headers = client.calls[0][3]
    assert headers["X-API-Key"] == "dedicated-x-api-key"


# --- Timeout: 30s -> 10s (relay 1006-on-silence fix) -> 45s (widened back out
# now that relay silence is handled by the visible interim WS message) -> 35s
# (this constant is now only a fallback default for callers that don't pass
# an explicit per-call budget; the real bound within a turn is the per-turn
# cumulative budget enforced by src/agents/chatbot.py's _handle_with_tools,
# kept in step with its _TOOL_CALL_CEILING_S) ------------------------------


def test_default_tool_timeout_is_35s() -> None:
    sig = inspect.signature(execute_crm_tool)
    default = sig.parameters["timeout_s"].default
    assert default == tool_executor._DEFAULT_CRM_TOOL_TIMEOUT_S == 35.0


@pytest.mark.asyncio
async def test_owned_client_gets_the_default_timeout(monkeypatch) -> None:
    """No http_client override -> execute_crm_tool builds its own httpx.AsyncClient.
    That client must be constructed with the module's default timeout budget."""
    captured: dict = {}

    class _RecordingClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        async def get(self, url, params=None, headers=None):
            return _FakeResp({"ok": True})

        async def request(self, method, url, json=None, headers=None):
            return _FakeResp({"ok": True})

        async def aclose(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", _RecordingClient)

    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={}, auth_type=None, token=None, args={},
    )
    assert out == {"status_code": 200, "data": {"ok": True}}
    timeout = captured["timeout"]
    assert timeout.read == 35.0
    assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_read_timeout_returns_error_dict_not_raise() -> None:
    """Regression guard for the safety property the whole timeout reduction
    depends on: a timed-out CRM call must become {"error": ...}, never raise
    out of execute_crm_tool (the caller — _handle_with_tools — treats a raise
    as a hard turn failure, not a degraded-but-successful tool result)."""

    class _TimingOutClient:
        async def get(self, *a, **k):
            raise httpx.ReadTimeout("boom")

        async def request(self, *a, **k):
            raise httpx.ReadTimeout("boom")

    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={}, auth_type=None, token=None, args={},
        http_client=_TimingOutClient(),
    )
    assert out == {"error": "boom"}


@pytest.mark.asyncio
async def test_internal_ids_are_redacted_from_response_body() -> None:
    """The real incident this guards: a CRM response echoing operator_id/
    user_id (per the authoritative contract, e.g. games-config) must never
    reach the LLM verbatim — a customer asked the bot to confirm a fake id
    and it read the real one straight out of a tool response and stated it."""
    client = _FakeClient({
        "operator_id": "6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef",
        "user_id": "f38bc464-d319-41b3-b1f0-22c4fa1b4aaf",
        "matka": {"enabled": True},
        "nested": {"user_id": "should-also-be-redacted"},
        "list_field": [{"operator_id": "should-be-redacted-too"}, {"keep": "me"}],
    })
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/operators/{operator_id}/games-config",
        method="GET",
        parameters={"operator_id": {"type": "string", "source": "session"}},
        auth_type=None, token=None,
        args={}, context={"operator_id": "6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef"},
        http_client=client,
    )
    data = out["data"]
    assert data["operator_id"] == "[redacted]"
    assert data["user_id"] == "[redacted]"
    assert data["nested"]["user_id"] == "[redacted]"
    assert data["list_field"][0]["operator_id"] == "[redacted]"
    assert data["list_field"][1] == {"keep": "me"}
    # Non-identifier data must pass through untouched.
    assert data["matka"] == {"enabled": True}


def test_redact_internal_ids_helper_is_recursive_and_leaves_other_keys_alone() -> None:
    out = tool_executor._redact_internal_ids({
        "operator_id": "x", "tenant_id": "y", "crm_id": "z", "session_id": "w",
        "safe": "value",
        "nested": {"user_id": "leak-me-not", "ok": 1},
        "items": [{"operator_id": "a"}, "plain-string", 42],
    })
    assert out == {
        "operator_id": "[redacted]", "tenant_id": "[redacted]",
        "crm_id": "[redacted]", "session_id": "[redacted]",
        "safe": "value",
        "nested": {"user_id": "[redacted]", "ok": 1},
        "items": [{"operator_id": "[redacted]"}, "plain-string", 42],
    }
    # Non-identifier strings/bools must pass through byte-for-byte -- only
    # actual UUID substrings (and known-id keys) get touched.
    assert tool_executor._redact_internal_ids(
        {"matka": {"enabled": True}, "status": "Open", "type": "casino"}
    ) == {"matka": {"enabled": True}, "status": "Open", "type": "casino"}


def test_uuid_embedded_in_arbitrary_string_value_is_scrubbed() -> None:
    """A UUID leaking under an unlisted key name (player_id, bare id) or
    embedded inside free text (a CRM error message) must be scrubbed even
    though the key itself isn't in the redacted-keys set."""
    out = tool_executor._redact_internal_ids({
        "message": "No player found with id 6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef",
        "player_id": "f38bc464-d319-41b3-b1f0-22c4fa1b4aaf",
        "id": "0f7d5e4a-1111-2222-3333-444455556666",
    })
    assert out == {
        "message": "No player found with id [redacted]",
        "player_id": "[redacted]",
        "id": "[redacted]",
    }


@pytest.mark.asyncio
async def test_uuid_in_exception_message_is_scrubbed_from_error_path() -> None:
    """A future raise_for_status() (or similar) could embed a UUID-bearing
    URL/id in the exception text reaching the {"error": ...} path -- that
    must be scrubbed the same as a normal response body."""

    class _Boom:
        async def get(self, *a, **k):
            raise RuntimeError(
                "404 for url https://crm.example.com/players/"
                "6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef"
            )

    out = await execute_crm_tool(
        endpoint="https://x/y", method="GET", parameters={},
        auth_type=None, token=None, args={}, http_client=_Boom(),
    )
    assert out == {
        "error": "404 for url https://crm.example.com/players/[redacted]"
    }


@respx.mock
@pytest.mark.asyncio
async def test_response_body_never_logged_pii_regression(caplog) -> None:
    """The real incident: a [TEMP DEBUG] log line used to dump the RAW CRM
    response body at INFO level. get_player_profile-shaped responses carry
    mobile/email/kyc_documents/bank_saved — none of that (nor the resolved
    UUID in the URL) may ever reach the logs, at any level."""
    user_id = "6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef"
    payload = {
        "mobile": "+919876543210",
        "email": "real.customer@example.com",
        "kyc_documents": ["Aadhaar-XXXX", "PAN-XXXX"],
        "bank_saved": {"bank": "HDFC", "account_last4": "1234", "upi": "customer@upi"},
        "user_id": user_id,
    }
    route = respx.get(f"https://crm.example.com/players/{user_id}/profile").mock(
        return_value=httpx.Response(200, json=payload)
    )

    with caplog.at_level(logging.DEBUG, logger="src.chatbot.tool_executor"):
        out = await execute_crm_tool(
            endpoint="https://crm.example.com/players/{user_id}/profile",
            method="GET",
            parameters={"user_id": {"type": "string", "source": "llm"}},
            auth_type=None, token=None,
            args={"user_id": user_id},
        )

    assert route.call_count == 1
    assert out["status_code"] == 200

    blob = "\n".join(
        r.getMessage() + " " + repr(r.__dict__) for r in caplog.records
    )
    for leaked in (
        "+919876543210",
        "real.customer@example.com",
        "customer@upi",
        '"account_last4": "1234"',
        user_id,
    ):
        assert leaked not in blob, f"PII/id leaked into logs: {leaked!r}"


@pytest.mark.asyncio
async def test_crm_tool_call_log_has_param_keys_not_values(caplog) -> None:
    client = _FakeClient({"ok": True})
    mobile = "+919876543210"

    with caplog.at_level(logging.INFO, logger="src.chatbot.tool_executor"):
        await execute_crm_tool(
            endpoint="https://crm.example.com/api/search",
            method="GET",
            parameters={"mobile": {"type": "string", "source": "llm"}},
            auth_type=None, token=None,
            args={"mobile": mobile}, http_client=client,
        )

    call_records = [r for r in caplog.records if r.getMessage() == "crm tool call"]
    assert len(call_records) == 1
    record = call_records[0]

    assert record.__dict__.get("param_keys") == ["mobile"]
    assert "params" not in record.__dict__
    assert mobile not in repr(record.__dict__)


def test_redact_url_scrubs_uuid() -> None:
    assert tool_executor._redact_url(
        "https://crm.example.com/players/6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef/profile"
    ) == "https://crm.example.com/players/[redacted]/profile"

    no_uuid = "https://crm.example.com/api/search"
    assert tool_executor._redact_url(no_uuid) == no_uuid


def test_case_variant_keys_are_also_redacted() -> None:
    """Casing variants of the canonical snake_case id keys must be caught by
    normalized (lowercased, underscore-stripped) key matching -- not just the
    exact casings previously hardcoded."""
    out = tool_executor._redact_internal_ids({
        "operatorID": "x", "USER_ID": "y", "TenantId": "z",
        "bet_id": "keep-me", "event_id": "keep-me-too",
    })
    assert out == {
        "operatorID": "[redacted]", "USER_ID": "[redacted]", "TenantId": "[redacted]",
        "bet_id": "keep-me", "event_id": "keep-me-too",
    }
