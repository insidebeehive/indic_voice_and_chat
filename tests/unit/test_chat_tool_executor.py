"""CRM tool HTTP executor (Phase 3b)."""

from __future__ import annotations

import inspect

import httpx
import pytest

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
