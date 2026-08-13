"""OpenAI-compatible LLM adapter — self-hosted vLLM (and any /v1 clone).

Registered as provider ``"vllm"``. Points the official ``openai`` client at a
configurable ``base_url`` (config ``base_url`` or the ``VLLM_BASE_URL`` env
var, e.g. ``https://<pod-id>-8001.proxy.runpod.net/v1``). vLLM ignores the
API key, so a placeholder is used unless one is configured.

Supports the full ILLMProvider surface the chat agent needs: tool calling
(ToolSpec ↔ OpenAI ``tools``; assistant tool_calls and ``tool`` role messages
round-trip), ``response_format="json"`` (vLLM guided decoding), and token
streaming. Multimodal content parts are text-only here — images stay on
Gemini; a list content's text parts are concatenated and inline media is
dropped with a warning.

Pod launch command: see docs/vllm-on-indicf5-pod.md.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

from src.interfaces.llm import (
    ILLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResult,
    ToolCall,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
_DEFAULT_TIMEOUT_S = 60.0

_FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "content_filter": "blocked",
    # "tool_calls" → the agent inspects result.tool_calls, not finish_reason.
    "tool_calls": "stop",
}


def _text_of(content: Any) -> str:
    """Flatten our message content (str or ContentPart list) to plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        text = getattr(p, "text", None)
        if getattr(p, "type", None) == "text" and text:
            parts.append(text)
        elif getattr(p, "inline_data", None):
            log.warning("openai-compat adapter is text-only; dropping inline media part")
    return "\n".join(parts)


def _to_openai_messages(messages: list[LLMMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": _text_of(m.content),
            })
            continue
        if m.role == "assistant" and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": _text_of(m.content) or None,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments or {})},
                } for tc in m.tool_calls],
            })
            continue
        out.append({"role": m.role, "content": _text_of(m.content)})
    return out


class OpenAICompatLLMAdapter(ILLMProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._default_model = config.get("model") or DEFAULT_MODEL
        client = config.get("client")
        if client is not None:
            # Tests inject a fake; bypass real SDK construction.
            self._client = client
            return
        base_url = config.get("base_url") or os.environ.get("VLLM_BASE_URL")
        if not base_url:
            raise ValueError(
                "OpenAICompatLLMAdapter requires a server URL (config 'base_url' "
                "or VLLM_BASE_URL env var, e.g. https://<pod>-8001.proxy.runpod.net/v1)"
            )
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=base_url,
            # vLLM ignores the key but the client requires one.
            api_key=config.get("api_key") or os.environ.get("VLLM_API_KEY") or "EMPTY",
            timeout=config.get("timeout", _DEFAULT_TIMEOUT_S),
        )

    def _request_kwargs(self, messages: list[LLMMessage], config: LLMConfig) -> dict:
        kwargs: dict[str, Any] = {
            "model": config.model or self._default_model,
            "messages": _to_openai_messages(messages),
        }
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        if config.tools:
            kwargs["tools"] = [{
                "type": "function",
                "function": {"name": t.name, "description": t.description,
                             "parameters": t.parameters},
            } for t in config.tools]
        elif config.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    async def generate(self, messages: list[LLMMessage], config: LLMConfig) -> LLMResult:
        resp = await self._client.chat.completions.create(
            **self._request_kwargs(messages, config))
        choice = resp.choices[0]
        tool_calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                log.warning("vllm tool call had unparseable arguments", extra={
                    "tool": tc.function.name})
                args = {}
            tool_calls.append(ToolCall(id=tc.id or f"{tc.function.name}-{len(tool_calls)}",
                                       name=tc.function.name, arguments=args))
        # LLMResult.usage is typed as a dict (never None) — the other three
        # adapters always populate it, so mirror that here instead of leaving
        # every consumer to defensively handle a None.
        usage: dict[str, int] = {}
        if resp.usage is not None:
            usage = {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens}
        return LLMResult(
            text=choice.message.content or "",
            finish_reason=_FINISH_REASONS.get(choice.finish_reason or "stop", "stop"),
            usage=usage,
            tool_calls=tool_calls,
        )

    async def generate_stream(
        self, messages: list[LLMMessage], config: LLMConfig,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._request_kwargs(messages, config), stream=True)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
