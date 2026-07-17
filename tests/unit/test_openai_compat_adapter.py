"""Tests for the OpenAI-compatible (vLLM) LLM adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage, ToolCall, ToolSpec
from src.providers.llm.openai_compat import OpenAICompatLLMAdapter


def _response(text: str | None = "ok", finish: str = "stop",
              tool_calls: list | None = None) -> SimpleNamespace:
    """Fake openai ChatCompletion response shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=tool_calls),
            finish_reason=finish,
        )],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _tool_call(name: str, args: dict, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _make_client(return_value: Any = None, side_effect: Any = None) -> SimpleNamespace:
    create = AsyncMock(return_value=return_value, side_effect=side_effect)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.asyncio
async def test_generate_returns_text_usage_and_finish() -> None:
    client = _make_client(return_value=_response('{"x": 1}'))
    adapter = OpenAICompatLLMAdapter({"client": client, "model": "test-model"})
    result = await adapter.generate(
        [LLMMessage(role="system", content="be terse"),
         LLMMessage(role="user", content="hi")],
        LLMConfig(temperature=0.4, max_tokens=128),
    )
    assert result.text == '{"x": 1}'
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][0] == {"role": "system", "content": "be terse"}
    assert kwargs["temperature"] == 0.4
    assert kwargs["max_tokens"] == 128


@pytest.mark.asyncio
async def test_generate_json_format_sets_response_format() -> None:
    client = _make_client(return_value=_response("{}"))
    adapter = OpenAICompatLLMAdapter({"client": client})
    await adapter.generate([LLMMessage(role="user", content="hi")],
                           LLMConfig(response_format="json"))
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_generate_with_tools_passes_openai_tools_and_omits_json() -> None:
    client = _make_client(return_value=_response("ok"))
    adapter = OpenAICompatLLMAdapter({"client": client})
    tools = [ToolSpec(name="search_kb", description="search",
                      parameters={"type": "object", "properties": {"q": {"type": "string"}}})]
    await adapter.generate([LLMMessage(role="user", content="hi")],
                           LLMConfig(response_format="json", tools=tools))
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["tools"][0]["function"]["name"] == "search_kb"
    assert "response_format" not in kwargs  # tools turns run in text mode


@pytest.mark.asyncio
async def test_generate_extracts_tool_calls() -> None:
    client = _make_client(return_value=_response(
        None, finish="tool_calls",
        tool_calls=[_tool_call("get_balance", {"user_id": "u1"})]))
    adapter = OpenAICompatLLMAdapter({"client": client})
    result = await adapter.generate([LLMMessage(role="user", content="balance?")],
                                    LLMConfig(tools=[ToolSpec("get_balance", "d", {})]))
    assert result.tool_calls[0].name == "get_balance"
    assert result.tool_calls[0].arguments == {"user_id": "u1"}
    assert result.finish_reason == "stop"  # agent reads .tool_calls, not finish
    assert result.text == ""


@pytest.mark.asyncio
async def test_tool_round_trip_message_mapping() -> None:
    client = _make_client(return_value=_response("done"))
    adapter = OpenAICompatLLMAdapter({"client": client})
    msgs = [
        LLMMessage(role="user", content="balance?"),
        LLMMessage(role="assistant", content="",
                   tool_calls=[ToolCall(id="c1", name="get_balance", arguments={"u": 1})]),
        LLMMessage(role="tool", name="get_balance", tool_call_id="c1",
                   content='{"balance": 100}'),
    ]
    await adapter.generate(msgs, LLMConfig())
    sent = client.chat.completions.create.await_args.kwargs["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["tool_calls"][0]["function"]["name"] == "get_balance"
    assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {"u": 1}
    assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": '{"balance": 100}'}


@pytest.mark.asyncio
async def test_unparseable_tool_arguments_degrade_to_empty() -> None:
    bad = SimpleNamespace(id="c1", type="function",
                          function=SimpleNamespace(name="t", arguments="{not json"))
    client = _make_client(return_value=_response(None, tool_calls=[bad]))
    adapter = OpenAICompatLLMAdapter({"client": client})
    result = await adapter.generate([LLMMessage(role="user", content="x")], LLMConfig())
    assert result.tool_calls[0].arguments == {}


@pytest.mark.asyncio
async def test_generate_stream_yields_deltas() -> None:
    async def _stream():
        for token in ("Hel", "lo"):
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=token))])
        yield SimpleNamespace(choices=[])  # keep-alive chunk without choices

    client = _make_client(return_value=_stream())
    adapter = OpenAICompatLLMAdapter({"client": client})
    tokens = [t async for t in adapter.generate_stream(
        [LLMMessage(role="user", content="hi")], LLMConfig())]
    assert tokens == ["Hel", "lo"]
    assert client.chat.completions.create.await_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_constructor_requires_base_url(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatLLMAdapter({})


def test_registered_in_provider_registry() -> None:
    from src.providers import LLM_PROVIDERS
    assert LLM_PROVIDERS["vllm"] is OpenAICompatLLMAdapter
