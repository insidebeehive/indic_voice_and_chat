"""Anthropic Claude LLM adapter.

Wraps the official ``anthropic`` SDK (async client). Mirrors the Gemini
adapter's shape so it drops straight into ``get_llm_provider``.

Two wrinkles versus Gemini, both handled here so the pipeline stays unchanged:

1. JSON envelope. The pipeline runs the LLM in ``response_format=json`` mode and
   the engine pulls ``response_text`` out of the streamed envelope. Claude has no
   ``response_mime_type`` switch, so we force a clean ``{...}`` object by
   *prefilling* the assistant turn with ``{``. The prefilled character is not
   echoed back in the stream, so ``generate_stream`` yields it as the first token
   itself — that keeps the engine's accumulated text valid JSON and lets
   ``_SpokenTextExtractor`` locate ``"response_text"``.

2. Per-model request surface. Assistant prefill 400s on Opus 4.6/4.7/4.8 and
   Sonnet 4.6; ``temperature`` 400s on Opus 4.7/4.8. Both are guarded by model
   name so a future model swap doesn't break. The default model is Haiku 4.5 —
   the fast, cheap tier, which is the reason to reach for Claude here (latency).

Indic-language handling: like Gemini, Claude handles Hindi/Devanagari natively;
the Devanagari directive lives in the agent system prompt, not here.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Optional

from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 1024


class AnthropicClaudeAdapter(ILLMProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._default_model = config.get("model") or DEFAULT_MODEL
        client = config.get("client")
        if client is not None:
            # Tests inject a fake; bypass real SDK construction.
            self._client = client
            return
        api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "AnthropicClaudeAdapter requires an API key (config 'api_key' or "
                "ANTHROPIC_API_KEY env var)"
            )
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "AnthropicClaudeAdapter requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from e
        self._client = AsyncAnthropic(api_key=api_key)

    # --- Message conversion ---------------------------------------------

    @staticmethod
    def _split(messages: list[LLMMessage]) -> tuple[Optional[str], list[dict]]:
        """Split our (role, content) messages into Claude's shape.

        Claude takes a separate ``system`` string plus a ``messages`` list whose
        roles are only ``user``/``assistant``. We map:
            our "system"    -> system (concatenated if many)
            our "user"      -> {role: "user", content: ...}
            our "assistant" -> {role: "assistant", content: ...}
        """
        system_parts: list[str] = []
        conv: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = "assistant" if m.role == "assistant" else "user"
            conv.append({"role": role, "content": m.content})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, conv

    @staticmethod
    def _allows_prefill(model: str) -> bool:
        # Last-assistant-turn prefill 400s on the 4.6 family and Opus 4.7/4.8.
        m = model.lower()
        return not any(
            tag in m for tag in ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6")
        )

    @staticmethod
    def _supports_sampling(model: str) -> bool:
        # temperature/top_p/top_k 400 on Opus 4.7 and 4.8.
        m = model.lower()
        return not any(tag in m for tag in ("opus-4-7", "opus-4-8"))

    def _build_request(
        self, messages: list[LLMMessage], config: LLMConfig
    ) -> tuple[dict[str, Any], str]:
        """Return ``(kwargs, prefill)`` for the SDK call.

        ``prefill`` is the assistant-turn prefix used to force a JSON object
        ('' when not prefilling). It is also what ``generate_stream`` must emit
        first, since the SDK does not echo prefilled text back in the stream.
        """
        system, conv = self._split(messages)
        model = config.model or self._default_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": config.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": conv,
        }
        if system:
            kwargs["system"] = system
        if self._supports_sampling(model):
            kwargs["temperature"] = config.temperature

        prefill = ""
        if config.response_format == "json" and self._allows_prefill(model):
            prefill = "{"
            conv.append({"role": "assistant", "content": prefill})
        return kwargs, prefill

    # --- Public API ----------------------------------------------------

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResult:
        kwargs, prefill = self._build_request(messages, config)
        # `kwargs` is the exact wire body (model/system/messages/temperature)
        # this method is about to send — logging it directly is the real
        # request, not a reconstruction of it.
        debug_event(log, "anthropic generate request", **kwargs)
        try:
            message = await self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised unchanged; this adapter has no retry
            # Nothing else in this adapter logs a failed call at any level —
            # unlike the Gemini adapter's retry/give-up logging, a Claude 4xx/5xx
            # would otherwise propagate with no record of which model/request
            # produced it. Re-raised as-is; this is purely observational.
            debug_event(log, "anthropic generate failed", model=kwargs.get("model"),
                        error=str(exc))
            raise
        text = prefill + self._extract_text(message)
        finish_reason = self._map_stop_reason(getattr(message, "stop_reason", None))
        usage = self._extract_usage(message)
        raw = _dump(message)
        debug_event(log, "anthropic generate response", model=kwargs.get("model"),
                    text=text, finish_reason=finish_reason, usage=usage, raw_response=raw)
        return LLMResult(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=raw,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        kwargs, prefill = self._build_request(messages, config)
        debug_event(log, "anthropic generate_stream request", **kwargs)
        # Emit the prefilled '{' first: the SDK streams only the *generated*
        # text, so without this the accumulated envelope would be missing its
        # opening brace and fail to parse.
        if prefill:
            yield prefill
        collecting = log.isEnabledFor(logging.DEBUG)  # see gemini.py's generate_stream for why
        chunks: list[str] = [prefill] if (prefill and collecting) else []
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if text:
                        if collecting:
                            chunks.append(text)
                        yield text
        except Exception as exc:  # noqa: BLE001 - re-raised unchanged
            debug_event(log, "anthropic generate_stream failed", model=kwargs.get("model"),
                        error=str(exc))
            raise
        if collecting:
            debug_event(log, "anthropic generate_stream response", model=kwargs.get("model"),
                        text="".join(chunks), chunks=len(chunks))

    # --- Response shape helpers (resilient to SDK changes) -------------

    @staticmethod
    def _extract_text(message: Any) -> str:
        parts: list[str] = []
        for block in getattr(message, "content", None) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts)

    @staticmethod
    def _extract_usage(message: Any) -> dict:
        u = getattr(message, "usage", None)
        if u is None:
            return {}
        return {
            "prompt_tokens": getattr(u, "input_tokens", 0) or 0,
            "completion_tokens": getattr(u, "output_tokens", 0) or 0,
        }

    @staticmethod
    def _map_stop_reason(raw: Any) -> str:
        if raw is None:
            return "stop"
        name = str(raw)
        if name == "end_turn":
            return "stop"
        if name == "max_tokens":
            return "length"
        if name == "refusal":
            return "blocked"
        return name


def _dump(message: Any) -> dict:
    """Best-effort serialization so the raw_response can be inspected."""
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(message, "to_dict"):
        try:
            return message.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return {}
