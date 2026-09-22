"""Groq LLM adapter.

Wraps the official ``groq`` SDK's ``AsyncGroq`` client. Streaming uses Groq's
SSE-backed ``stream=True`` mode and yields content tokens as they arrive.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from groq import AsyncGroq

from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

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
        debug_event(log, "groq generate request", **kwargs)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised unchanged; no retry in this adapter
            # This adapter has no retry/error logging of its own (unlike Gemini's),
            # so without this a Groq 4xx/5xx propagates with no record of the
            # model or request that produced it.
            debug_event(log, "groq generate failed", model=kwargs.get("model"), error=str(exc))
            raise
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        debug_event(log, "groq generate response", model=kwargs.get("model"), text=text,
                    finish_reason=choice.finish_reason or "stop", usage=usage, raw_response=raw)
        return LLMResult(
            text=text,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            raw_response=raw,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(messages, config)
        kwargs["stream"] = True
        debug_event(log, "groq generate_stream request", **kwargs)
        collecting = log.isEnabledFor(logging.DEBUG)  # see gemini.py's generate_stream for why
        chunks: list[str] = []
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    if collecting:
                        chunks.append(content)
                    yield content
        except Exception as exc:  # noqa: BLE001 - re-raised unchanged
            debug_event(log, "groq generate_stream failed", model=kwargs.get("model"),
                        error=str(exc))
            raise
        if collecting:
            debug_event(log, "groq generate_stream response", model=kwargs.get("model"),
                        text="".join(chunks), chunks=len(chunks))
