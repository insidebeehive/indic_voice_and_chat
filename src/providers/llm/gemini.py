"""Gemini LLM adapter (Google Generative AI).

Wraps the official ``google-genai`` SDK. Both batch and streaming flows
are supported — streaming uses ``aio.generate_content_stream`` and yields
text deltas as they arrive.

Indic-language handling: Gemini understands Hindi/Devanagari natively, so
the adapter doesn't need to do any pre-processing. The agent's system
prompt declares the language directive (see ``build_voicebot_system_prompt``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Callable, Optional

import json

from src.interfaces.llm import (
    ContentPart,
    ILLMProvider,
    LLMConfig,
    LLMMessage,
    LLMResult,
    ToolCall,
)


log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"

# Gemini intermittently returns transient backend errors — most notably
# ``500 INTERNAL`` ("An internal error has occurred. Please retry...") — even
# for well-formed requests. Google's own error body tells callers to retry, so
# we transparently retry these before any audio is spoken. A live turn must not
# die on a provider blip.
_RETRIABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 2  # total attempts = 1 + _MAX_RETRIES
_BACKOFF_BASE_S = 0.4


class GeminiLLMAdapter(ILLMProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._default_model = config.get("model") or DEFAULT_MODEL
        client = config.get("client")
        if client is not None:
            # Tests inject a fake; bypass real SDK construction.
            self._client = client
            return
        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GeminiLLMAdapter requires an API key (config 'api_key' or "
                "GEMINI_API_KEY env var)"
            )
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "GeminiLLMAdapter requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from e
        self._client = genai.Client(api_key=api_key)

    # --- Message conversion ---------------------------------------------

    @staticmethod
    def _to_gemini_contents(messages: list[LLMMessage]) -> tuple[Optional[str], list[dict]]:
        """Split our (role, content) messages into Gemini's shape.

        Gemini takes a separate ``system_instruction`` plus a list of
        ``contents`` entries with ``role`` in {user, model}. We map:
            our "system"    -> system_instruction (concatenated if many)
            our "user"      -> {role: "user", parts: [{text: ...}]}
            our "assistant" -> {role: "model", parts: [{text: ...}]}
        """
        system_parts: list[str] = []
        contents: list[dict] = []
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                continue
            # Tool result (function_response) → a "user"-role function_response part.
            if m.role == "tool":
                try:
                    response = json.loads(m.content) if isinstance(m.content, str) else {}
                except (json.JSONDecodeError, TypeError):
                    response = {"result": m.content}
                if not isinstance(response, dict):
                    response = {"result": response}
                contents.append({"role": "user", "parts": [
                    {"function_response": {"name": m.name or "", "response": response}}]})
                continue
            role = "model" if m.role == "assistant" else "user"
            # Assistant message that emitted tool calls → function_call parts.
            if m.tool_calls:
                contents.append({"role": role, "parts": [
                    {"function_call": {"name": tc.name, "args": tc.arguments}}
                    for tc in m.tool_calls]})
                continue
            contents.append({"role": role, "parts": _content_parts(m.content)})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, contents

    def _build_config(self, config: LLMConfig) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
            # Gemini 2.5 Flash thinking tokens count against max_output_tokens.
            # Cap thinking at 1024 so the bulk of the budget goes to actual output.
            "thinking_config": {"thinking_budget": 1024},
        }
        if config.tools:
            # Function declarations as plain dicts — the SDK coerces them.
            cfg["tools"] = [{"function_declarations": [{
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }]} for t in config.tools]
            # Gemini rejects response_mime_type=application/json together with
            # tools, so JSON format is suppressed for tool turns (the agent does a
            # separate no-tools JSON finalize turn).
        elif config.response_format == "json":
            cfg["response_mime_type"] = "application/json"
        return cfg

    @staticmethod
    async def _call_with_retry(fn: Callable[[], Any], *, what: str) -> Any:
        """Await ``fn()``, retrying transient 5xx/429 errors with backoff.

        Only errors whose HTTP status is in ``_RETRIABLE_STATUS`` are retried;
        everything else (and ``CancelledError``, which is a ``BaseException``)
        propagates immediately. The caller must ensure no side effects have been
        committed yet — for streaming, that means no token has been yielded.
        """
        attempt = 0
        while True:
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001 - re-raised unless retriable
                code = getattr(exc, "code", None)
                if code in _RETRIABLE_STATUS and attempt < _MAX_RETRIES:
                    attempt += 1
                    log.warning(
                        "gemini transient %s on %s; retry %d/%d",
                        code, what, attempt, _MAX_RETRIES,
                    )
                    await asyncio.sleep(_BACKOFF_BASE_S * attempt)
                    continue
                raise

    # --- Public API ----------------------------------------------------

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResult:
        system, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(config)
        if system:
            gen_config["system_instruction"] = system

        response = await self._call_with_retry(
            lambda: self._client.aio.models.generate_content(
                model=config.model or self._default_model,
                contents=contents,
                config=gen_config,
            ),
            what="generate",
        )
        text = self._extract_text(response)
        usage = self._extract_usage(response)
        finish_reason = self._extract_finish_reason(response)
        return LLMResult(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=_dump(response),
            tool_calls=self._extract_tool_calls(response),
        )

    async def transcribe_audio(self, audio: bytes, mime_type: str = "audio/mpeg") -> str:
        """Transcribe a complete audio recording (e.g. a call mp3) to text via
        Gemini multimodal — strong on Indian languages + Hindi/English code-switch,
        and handles long, mono recordings the live turn-STT can't. Returns "" on
        failure (analysis then falls back, never crashes)."""
        prompt = (
            "Transcribe this phone call audio verbatim. Preserve the original "
            "languages exactly as spoken (Hindi/English/other code-switching). "
            "Output ONLY the transcript text — no commentary, labels, or translation."
        )
        contents = [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": audio}},
        ]}]
        try:
            response = await self._call_with_retry(
                lambda: self._client.aio.models.generate_content(
                    model=self._default_model, contents=contents),
                what="transcribe")
            return self._extract_text(response) or ""
        except Exception:  # noqa: BLE001 - transcription failure must not crash finalize
            log.exception("gemini audio transcription failed")
            return ""

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        system, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(config)
        if system:
            gen_config["system_instruction"] = system
        model = config.model or self._default_model

        # Transient 5xx can surface either when opening the stream or on the
        # first chunk, so the retry must cover both — but only up to the first
        # yielded token, after which audio may already be playing and a retry
        # would re-speak. Re-opening starts a fresh stream each attempt.
        async def _open_and_first() -> tuple[Any, Any]:
            stream = await self._client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=gen_config,
            )
            agen = stream.__aiter__()
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                first = None
            return agen, first

        agen, first = await self._call_with_retry(
            _open_and_first, what="generate_stream",
        )

        if first is None:
            return
        text = self._extract_text(first)
        if text:
            yield text
        async for chunk in agen:
            text = self._extract_text(chunk)
            if text:
                yield text

    # --- Response shape helpers (resilient to SDK changes) -------------

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[ToolCall]:
        """Pull ``function_call`` parts out of the response into ToolCalls.
        Gemini calls carry no id, so we synthesize one from name + index."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []
        content = getattr(candidates[0], "content", None)
        parts = (getattr(content, "parts", None) or []) if content is not None else []
        calls: list[ToolCall] = []
        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc is None:
                continue
            name = getattr(fc, "name", "") or ""
            args = getattr(fc, "args", None) or {}
            calls.append(ToolCall(
                id=getattr(fc, "id", None) or f"{name}-{len(calls)}",
                name=name,
                arguments=dict(args),
            ))
        return calls

    @staticmethod
    def _extract_text(response: Any) -> str:
        # The SDK exposes a ``.text`` convenience accessor; fall back to
        # walking ``candidates[0].content.parts[*].text`` if that fails. The
        # ``.text`` property RAISES when the only part is a function_call, so
        # guard it.
        try:
            text = getattr(response, "text", None)
        except Exception:  # noqa: BLE001 - function-call-only responses raise here
            text = None
        if text:
            return text
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        content = getattr(candidates[0], "content", None)
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or []
        return "".join((getattr(p, "text", "") or "") for p in parts)

    @staticmethod
    def _extract_usage(response: Any) -> dict:
        u = getattr(response, "usage_metadata", None)
        if u is None:
            return {}
        return {
            "prompt_tokens": getattr(u, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(u, "candidates_token_count", 0) or 0,
        }

    @staticmethod
    def _extract_finish_reason(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "stop"
        raw = getattr(candidates[0], "finish_reason", None)
        if raw is None:
            return "stop"
        # Gemini returns an enum or a string depending on SDK version.
        name = getattr(raw, "name", None) or str(raw)
        # Map Gemini's vocabulary to ours.
        if "STOP" in name:
            return "stop"
        if "MAX_TOKENS" in name:
            return "length"
        if "SAFETY" in name or "RECITATION" in name:
            return "blocked"
        return name.lower()


def _content_parts(content) -> list[dict]:
    """Turn an ``LLMMessage.content`` (str OR list[ContentPart]) into Gemini parts.
    A plain string → a single text part (unchanged from before). A list →
    text parts + ``inline_data`` parts (the shape proven in ``transcribe_audio``)."""
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict] = []
    for part in content or []:
        if isinstance(part, ContentPart):
            if part.type == "image" and part.inline_data:
                parts.append({"inline_data": part.inline_data})
            elif part.text is not None:
                parts.append({"text": part.text})
        elif isinstance(part, dict):  # tolerate raw dicts
            parts.append(part)
    return parts or [{"text": ""}]


def _dump(response: Any) -> dict:
    """Best-effort serialization so the raw_response can be inspected."""
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return {"text": getattr(response, "text", "")}
