"""LLM provider interface (PRD §4.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Union


@dataclass
class ContentPart:
    """One part of a multimodal message. ``type`` is "text" or "image";
    image parts carry ``inline_data = {"mime_type": ..., "data": <bytes>}``
    (the shape Gemini accepts for inline media)."""
    type: str
    text: Optional[str] = None
    inline_data: Optional[dict] = None


@dataclass
class ToolSpec:
    """A function the model may call (provider-agnostic). ``parameters`` is a
    JSON-Schema-ish dict (type/properties/required/enum/items)."""
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    """A function call the model emitted."""
    id: str
    name: str
    arguments: dict = field(default_factory=dict)
    # Gemini 3.x attaches an opaque signature to each function-call part and
    # REQUIRES it back when the call is replayed in the follow-up request
    # (400 INVALID_ARGUMENT "missing a thought_signature" otherwise). Opaque
    # provider bytes — carried, never inspected.
    thought_signature: Any = None


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    # Plain string (the common case) OR a list of multimodal parts.
    content: Union[str, list[ContentPart]] = ""
    # Set on an assistant message that emitted tool calls.
    tool_calls: Optional[list[ToolCall]] = None
    # Set on a "tool" message carrying a tool result back to the model.
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # tool name (on a "tool" result message)


@dataclass
class LLMConfig:
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    response_format: Optional[str] = "json"  # "json" | "text"
    # Function-calling tools. When set, the JSON response-format is suppressed
    # (Gemini rejects response_mime_type=application/json together with tools).
    tools: Optional[list[ToolSpec]] = None


@dataclass
class LLMResult:
    text: str
    finish_reason: str
    usage: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)


class ILLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResult:
        """Generate a complete response."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""


def is_llm_spending_cap_error(exc: Exception) -> bool:
    """True if ``exc`` is a 429 caused by a monthly *spending cap*, as opposed
    to an ordinary rate/quota-exhaustion 429.

    Google's Gemini error body has no machine-checkable field for this case —
    it's prose that includes the literal phrase "spending cap" (observed live:
    "429 RESOURCE_EXHAUSTED. Your project has exceeded its monthly spending
    cap. See https://ai.studio/spend for details."). Ordinary quota 429s
    (per-minute/per-day RESOURCE_EXHAUSTED, FreeTier, RetryInfo.retryDelay)
    don't use this wording, so the match stays narrow: it flags a billing
    ceiling a human must raise, never a transient quota window that clears on
    its own.

    Provider-agnostic by design (matches on message text, not a Gemini-typed
    exception) so both the provider layer (fail fast instead of retrying) and
    the chat layer (surface a distinct ``llm_billing`` reason) can share this
    one definition instead of each hand-rolling the same substring check and
    silently drifting apart.
    """
    return "spending cap" in str(exc).lower()
