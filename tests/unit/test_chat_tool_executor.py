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
    assert out["failure"] == "transport_error"


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
    # Ticket #1762 fix: a bare httpx timeout carries an EMPTY str(e) -- the
    # message must always be a real, non-empty phrase (never that literal
    # empty string), plus a machine-readable "failure" discriminator.
    assert out == {
        "error": "The request to fetch this data timed out (ReadTimeout).",
        "failure": "timeout",
    }


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
        "error": "404 for url https://crm.example.com/players/[redacted]",
        "failure": "transport_error",
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


def test_pgs_order_id_is_exempt_from_redaction() -> None:
    """PgsOrderId is the payment gateway's own order reference, forwarded
    verbatim to the deposit-verification vendor -- it must reach the LLM
    unmodified, unlike the internal platform ids this scrub targets."""
    out = tool_executor._redact_internal_ids({
        "PgsOrderId": "PGS20260621143000123",
        "datetime": "2026-06-21T14:30:00Z",
        "status": "failed",
    })
    assert out == {
        "PgsOrderId": "PGS20260621143000123",
        "datetime": "2026-06-21T14:30:00Z",
        "status": "failed",
    }


def test_pgs_order_id_exemption_survives_uuid_shaped_value_but_is_narrow() -> None:
    """A UUID-shaped value under the exact PgsOrderId key must NOT be
    redacted (proving the exemption works even for the case the general
    UUID scrub would otherwise catch), while a UUID-shaped value under any
    other, non-exempt key in the SAME payload must still be redacted -- this
    pins the exemption's narrowness so it can't silently regress into a
    general allowlist."""
    out = tool_executor._redact_internal_ids({
        "PgsOrderId": "550e8400-e29b-41d4-a716-446655440000",
        "note": "ref 550e8400-e29b-41d4-a716-446655440000",
    })
    assert out == {
        "PgsOrderId": "550e8400-e29b-41d4-a716-446655440000",
        "note": f"ref {tool_executor.REDACTED_PLACEHOLDER}",
    }


def test_pgs_order_id_exemption_is_exact_match_not_substring() -> None:
    """A key that CONTAINS the normalized substring "pgsorderid" but does not
    normalize to exactly "pgsorderid" must NOT be exempt -- this is the actual
    proof the exemption is exact-match. A hypothetical buggy substring-based
    exemption (e.g. `if "pgsorderid" in normalized_k`) would wrongly pass this
    key through unredacted; the "note" key used in the test above never
    contains that substring, so it can't catch that bug on its own."""
    out = tool_executor._redact_internal_ids({
        "realPgsOrderId_internalPlayerRef": "550e8400-e29b-41d4-a716-446655440000",
    })
    assert out == {
        "realPgsOrderId_internalPlayerRef": tool_executor.REDACTED_PLACEHOLDER,
    }


def test_pgs_order_id_non_string_value_is_not_blindly_passed_through() -> None:
    """The exact PgsOrderId exemption only short-circuits for a string value
    (`isinstance(v, str)`) -- a non-string value under that exact key (a
    nested dict/list, as some CRM responses might send) must still recurse
    through the normal redaction path rather than being exempted wholesale."""
    out = tool_executor._redact_internal_ids({
        "PgsOrderId": {"user_id": "should-still-be-redacted", "keep": "me"},
    })
    assert out == {
        "PgsOrderId": {"user_id": tool_executor.REDACTED_PLACEHOLDER, "keep": "me"},
    }


# --- Step 1 (ticket #1762 fix): failure legibility ------------------------


@pytest.mark.asyncio
async def test_timeout_error_message_is_non_empty_with_failure_discriminator() -> None:
    """The exact ticket #1762 shape: a BARE httpx timeout with no message at
    all -- str(httpx.ReadTimeout()) is "". The error text reaching the LLM
    must never be that empty string, and must carry the "timeout" discriminator."""

    class _TimingOutClient:
        async def get(self, *a, **k):
            raise httpx.ReadTimeout("")  # empty message -- str(e) == ""

    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={}, auth_type=None, token=None, args={},
        http_client=_TimingOutClient(),
    )
    assert out["error"]  # non-empty
    assert out["error"] != ""
    assert out["failure"] == "timeout"


@pytest.mark.asyncio
async def test_status_code_4xx_is_detected_as_failure() -> None:
    class _404Client(_FakeClient):
        async def get(self, url, params=None, headers=None):
            self.calls.append(("GET", url, params, headers))
            return _FakeResp(self._payload, status=404)

    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={}, auth_type=None, token=None, args={},
        http_client=_404Client({"message": "player not found"}),
    )
    assert out["status_code"] == 404
    assert out["failure"] == "http_error"
    assert out["error"]
    assert out["data"] == {"message": "player not found"}


@pytest.mark.asyncio
async def test_status_code_5xx_is_detected_as_failure() -> None:
    class _503Client(_FakeClient):
        async def get(self, url, params=None, headers=None):
            self.calls.append(("GET", url, params, headers))
            return _FakeResp(self._payload, status=503)

    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={}, auth_type=None, token=None, args={},
        http_client=_503Client({"error": "unavailable"}),
    )
    assert out["status_code"] == 503
    assert out["failure"] == "http_error"
    assert out["error"]


@pytest.mark.asyncio
async def test_status_code_2xx_has_no_failure_key() -> None:
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/api/wallet",
        method="GET", parameters={}, auth_type=None, token=None, args={},
        http_client=client,
    )
    assert out == {"status_code": 200, "data": {"ok": True}}
    assert "failure" not in out
    assert "error" not in out


# --- Phase 0 task 2: path-parameter injection (bare string-replace into the
# endpoint URL had no encoding or validation — an LLM-sourced value like
# game_name/market_name could carry "/", "..", "?", or "#" and rewrite the
# request path/query against the tenant's real CRM host+credentials) -------


@pytest.mark.asyncio
async def test_path_traversal_in_llm_param_is_rejected_without_http_call() -> None:
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/casino/{operator_id}/players/{user_id}/games/{game_name}/bet-limit",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "user_id": {"type": "string", "source": "session"},
            "game_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"game_name": "../../operators/x/platform-config"},
        context={"operator_id": "op1", "user_id": "u1"},
        http_client=client,
    )
    assert out == {
        "error": "One of the request parameters had an invalid value.",
        "failure": "invalid_parameter",
    }
    assert client.calls == []  # rejected before any HTTP call was attempted


@pytest.mark.asyncio
async def test_query_and_fragment_chars_in_llm_param_are_rejected() -> None:
    client = _FakeClient({"ok": True})
    for bad_value in ("foo?admin=true", "foo#frag"):
        out = await execute_crm_tool(
            endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
            method="GET",
            parameters={
                "operator_id": {"type": "string", "source": "session"},
                "market_name": {"type": "string", "source": "llm"},
            },
            auth_type=None, token=None,
            args={"market_name": bad_value},
            context={"operator_id": "op1"},
            http_client=client,
        )
        assert out["failure"] == "invalid_parameter"
    assert client.calls == []


@pytest.mark.asyncio
async def test_legitimate_path_value_is_percent_encoded_and_call_is_made() -> None:
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/casino/{operator_id}/players/{user_id}/games/{game_name}/bet-limit",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "user_id": {"type": "string", "source": "session"},
            "game_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"game_name": "Teen Patti & Co"},
        context={"operator_id": "op1", "user_id": "u1"},
        http_client=client,
    )
    assert "error" not in out
    method, url, params, headers = client.calls[0]
    assert url == (
        "https://crm.example.com/casino/op1/players/u1/games/"
        "Teen%20Patti%20%26%20Co/bet-limit"
    )
    assert params == {}


@pytest.mark.asyncio
async def test_session_sourced_uuid_path_param_still_works_unchanged() -> None:
    """Regression guard: the common path (a server-derived session param, not
    LLM-controlled) must still resolve exactly as before -- no rejection, no
    unexpected re-encoding of characters a UUID never contains anyway."""
    client = _FakeClient({"profile": "ok"})
    user_id = "6c1a77a6-20b0-4fc9-ba4c-8add58aba9ef"
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/players/{user_id}/profile",
        method="GET",
        parameters={"user_id": {"type": "string", "source": "session"}},
        auth_type=None, token=None,
        args={}, context={"user_id": user_id}, http_client=client,
    )
    assert "error" not in out
    assert client.calls[0][1] == f"https://crm.example.com/players/{user_id}/profile"


@pytest.mark.asyncio
async def test_transport_error_uuid_still_scrubbed_with_failure_discriminator() -> None:
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
    assert out["failure"] == "transport_error"
    assert "6c1a77a6" not in out["error"]
    assert "[redacted]" in out["error"]


# --- Review findings: bare "." / empty path values, encode-failure surrogate,
# and the untested edges of the check-then-encode path-value guard ---------


@pytest.mark.asyncio
async def test_bare_dot_path_value_is_rejected_without_http_call() -> None:
    """A single "." doesn't match the two-dot traversal regex and quote()
    leaves it alone (it's in urllib's always-safe set) — but httpx normalises
    a "/./" segment away client-side, so market_name="." would resolve to a
    real endpoint the template never described."""
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": "."},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert out == {
        "error": "One of the request parameters had an invalid value.",
        "failure": "invalid_parameter",
    }
    assert client.calls == []


@pytest.mark.asyncio
async def test_empty_string_path_value_is_rejected_without_http_call() -> None:
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": ""},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert out["failure"] == "invalid_parameter"
    assert client.calls == []


@pytest.mark.asyncio
async def test_dotted_but_not_all_dots_value_still_works() -> None:
    """Regression guard for over-tightening: a legitimate value that merely
    CONTAINS a dot (not solely dots) must still be accepted and encoded."""
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": "teen-patti.v2"},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert "error" not in out
    method, url, params, headers = client.calls[0]
    assert url == (
        "https://crm.example.com/matka/op1/markets/teen-patti.v2/holiday-schedule"
    )


@pytest.mark.asyncio
async def test_whitespace_only_path_value_is_accepted_and_encoded() -> None:
    """Whitespace-only is DELIBERATELY accepted, not an oversight.

    Unlike "." (which httpx normalises away, reaching a different real
    endpoint), " " percent-encodes to a literal "%20" segment that no RFC 3986
    or httpx rule removes — so it is a distinct, non-existent path segment that
    404s at the CRM rather than a change to the path's SHAPE. Pinned here so a
    future "tidy-up" of _PATH_VALUE_EMPTY_OR_ALL_DOTS_RE into something like
    ``^[\\s.]*$`` has to justify itself against a failing test.
    """
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": " "},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert "error" not in out
    method, url, params, headers = client.calls[0]
    assert url == (
        "https://crm.example.com/matka/op1/markets/%20/holiday-schedule"
    )


@pytest.mark.asyncio
async def test_backslash_path_value_is_rejected() -> None:
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": "foo\\bar"},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert out["failure"] == "invalid_parameter"
    assert client.calls == []


@pytest.mark.asyncio
async def test_session_sourced_slash_value_is_also_rejected() -> None:
    """The path-value guard is source-agnostic by design (defence in depth) —
    a "session" param carrying "/" must be rejected exactly like an "llm" one,
    proving the check isn't scoped to source == "llm"."""
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/customers/{customer_id}/history",
        method="GET",
        parameters={"customer_id": {"type": "string", "source": "session"}},
        auth_type=None, token=None,
        args={}, context={"customer_id": "cust/42"}, http_client=client,
    )
    assert out["failure"] == "invalid_parameter"
    assert client.calls == []


@pytest.mark.asyncio
async def test_non_str_path_value_is_stringified_then_rejected() -> None:
    """Pins the stringify-then-check ordering: a non-str value (e.g. a list)
    is str()'d before the danger check runs, so its str() form ("['..', 'x']")
    is what gets rejected — it must not bypass the check by virtue of not
    being a string to begin with."""
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": ["..", "x"]},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert out["failure"] == "invalid_parameter"
    assert client.calls == []


@pytest.mark.asyncio
async def test_pre_encoded_traversal_is_neutralised_by_double_encoding() -> None:
    """The subtlest property of check-then-encode: "%2e%2e%2f" doesn't match
    the raw-character danger regex (no literal "/", "\\", "?", "#", or ".."),
    so it's accepted — but quote() then percent-encodes the literal "%"
    characters themselves, turning it into "%252e%252e%252f". That double
    encoding is inert: no proxy or router decodes twice, so it can never
    collapse back into a real ".." traversal segment server-side."""
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": "%2e%2e%2f"},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert "error" not in out
    method, url, params, headers = client.calls[0]
    assert "%252e%252e%252f" in url


@pytest.mark.asyncio
async def test_lone_surrogate_path_value_returns_invalid_parameter_not_raise() -> None:
    """quote() raises UnicodeEncodeError on a lone UTF-16 surrogate, which can
    arrive through json.loads() of the model's tool-call arguments (Python's
    JSON parser does not reject lone surrogates). That must surface as the
    same invalid_parameter error dict as any other rejected value, not
    escape execute_crm_tool as an unhandled exception."""
    client = _FakeClient({"ok": True})
    out = await execute_crm_tool(
        endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
        method="GET",
        parameters={
            "operator_id": {"type": "string", "source": "session"},
            "market_name": {"type": "string", "source": "llm"},
        },
        auth_type=None, token=None,
        args={"market_name": "\ud800"},
        context={"operator_id": "op1"},
        http_client=client,
    )
    assert out == {
        "error": "One of the request parameters had an invalid value.",
        "failure": "invalid_parameter",
    }
    assert client.calls == []


@pytest.mark.asyncio
async def test_rejection_log_does_not_contain_the_param_value(caplog) -> None:
    """Explicit requirement of the original fix: only the param NAME may be
    logged on rejection, never the (potentially attacker-controlled or
    PII-shaped) value itself."""
    client = _FakeClient({"ok": True})
    secret_value = "../../operators/x/platform-config"

    with caplog.at_level(logging.WARNING, logger="src.chatbot.tool_executor"):
        out = await execute_crm_tool(
            endpoint="https://crm.example.com/matka/{operator_id}/markets/{market_name}/holiday-schedule",
            method="GET",
            parameters={
                "operator_id": {"type": "string", "source": "session"},
                "market_name": {"type": "string", "source": "llm"},
            },
            auth_type=None, token=None,
            args={"market_name": secret_value},
            context={"operator_id": "op1"},
            http_client=client,
        )

    assert out["failure"] == "invalid_parameter"
    reject_records = [
        r for r in caplog.records
        if r.getMessage() == "crm tool call rejected invalid path parameter"
    ]
    assert len(reject_records) == 1
    record = reject_records[0]
    assert record.__dict__.get("param_name") == "market_name"
    assert secret_value not in repr(record.__dict__)
    assert secret_value not in record.getMessage()
