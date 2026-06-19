"""LLM provider interface (PRD §4.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Union


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
