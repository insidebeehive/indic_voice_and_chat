from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.providers.llm.groq import GroqLLMAdapter


def _completion_response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        model_dump=lambda: {"choices": [{"message": {"content": content}}]},
    )


def _make_client(create_return: Any) -> SimpleNamespace:
    """Build a mock AsyncGroq-shaped client whose .chat.completions.create returns ``create_return``."""
    create = AsyncMock(return_value=create_return)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.mark.asyncio
async def test_generate_returns_text_and_usage() -> None:
    client = _make_client(_completion_response('{"response_text": "ok"}'))
    adapter = GroqLLMAdapter({"client": client})

    result = await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(model="llama-3.3-70b-versatile", temperature=0.5, max_tokens=100),
    )
    assert result.text == '{"response_text": "ok"}'
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}

    # Verify call args
    create = client.chat.completions.create
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "llama-3.3-70b-versatile"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 100
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_generate_text_format_omits_response_format() -> None:
    client = _make_client(_completion_response("plain text"))
    adapter = GroqLLMAdapter({"client": client})
    await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(response_format="text"),
    )
    kwargs = client.chat.completions.create.await_args.kwargs
    assert "response_format" not in kwargs


@pytest.mark.asyncio
async def test_generate_stream_yields_tokens() -> None:
    async def fake_stream():
        for tok in ("Hel", "lo, ", "world"):
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=tok))]
            )
        # An empty-choices chunk should be skipped
        yield SimpleNamespace(choices=[])

    client = _make_client(fake_stream())
    adapter = GroqLLMAdapter({"client": client})

    tokens: list[str] = []
    async for tok in adapter.generate_stream(
        [LLMMessage(role="user", content="hi")], LLMConfig()
    ):
        tokens.append(tok)
    assert tokens == ["Hel", "lo, ", "world"]


@pytest.mark.asyncio
async def test_constructor_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GroqLLMAdapter({})
