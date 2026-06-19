"""Groq LLM adapter.

Wraps the official ``groq`` SDK's ``AsyncGroq`` client. Streaming uses Groq's
SSE-backed ``stream=True`` mode and yields content tokens as they arrive.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from groq import AsyncGroq

from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult


DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqLLMAdapter(ILLMProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._default_model = config.get("model") or DEFAULT_MODEL
        client = config.get("client")
        if client is not None:
            self._client = client
            return
        api_key = config.get("api_key") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GroqLLMAdapter requires an API key (config 'api_key' or "
                "GROQ_API_KEY env var)"
            )
        self._client = AsyncGroq(api_key=api_key)

    def _build_kwargs(self, messages: list[LLMMessage], config: LLMConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": config.model or self._default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResult:
        kwargs = self._build_kwargs(messages, config)
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        return LLMResult(
            text=text,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(messages, config)
        kwargs["stream"] = True
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
