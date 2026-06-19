"""Builtin ChatBot tools (PRD §5.2).

These three are always available to the agent; tenant-registered CRM tools
(Phase 3b) are appended at runtime. The LLM sees all of them and decides which
to call per turn. ``ToolSpec.parameters`` is a JSON-Schema-ish dict.
"""

from __future__ import annotations

from src.interfaces.llm import ToolSpec

SEARCH_KB = "search_knowledge_base"
ESCALATE = "escalate_to_human"
OFFER_CALL = "offer_voice_call"

BUILTIN_TOOL_NAMES = frozenset({SEARCH_KB, ESCALATE, OFFER_CALL})

BUILTIN_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name=SEARCH_KB,
        description=("Search the company's product docs, FAQs, and policies for "
                     "information relevant to the customer's query. Use this BEFORE "
                     "answering factual questions — never guess."),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The search query, in the customer's language"},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name=ESCALATE,
        description=("Transfer the conversation to a human support agent. Use when "
                     "the customer is frustrated, explicitly requests a human, or the "
                     "issue requires account-level actions you can't perform."),
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "summary": {"type": "string",
                            "description": "A short summary of the issue for the human agent"},
            },
            "required": ["reason", "summary"],
        },
    ),
    ToolSpec(
        name=OFFER_CALL,
        description=("Offer the customer a voice call when the issue would be easier "
                     "to resolve by talking."),
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    ),
]
