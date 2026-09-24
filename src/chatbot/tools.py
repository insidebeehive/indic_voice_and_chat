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
SUBMIT_DEPOSIT_VERIFICATION = "submit_deposit_verification"

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
                     "issue requires account-level actions you can't perform. Do not "
                     "call or offer this while the customer is still trying suggested "
                     "troubleshooting steps."),
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
        description=(
            "Offer the customer a browser-based voice call so they can speak with "
            "you directly — no phone number needed. The system automatically opens "
            "a web audio call in their browser. Use when the issue is complex enough "
            "that talking would be faster than typing. Do NOT ask the customer for "
            "a phone number; just call this tool with the reason."
        ),
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    ),
]

# Not in BUILTIN_TOOLS: only registered per-tenant, when the tenant has
# deposit_verification enabled, a webhook_url configured and a resolvable
# webhook signing secret (see src/bootstrap.py's
# deposit_verification_registrable and make_chatbot_factory).
#
# order_id vs. external_transaction_id: nothing at runtime signals back "the
# vendor rejected order_id" (the webhook ack is fire-and-forget — see
# src/chatbot/deposit_verification.py's module docstring), so the
# external_transaction_id fallback described in
# the order_id parameter below can't actually be triggered by observed
# behavior today. The CRM team explicitly asked us to try order_id first and
# report back which one their verification service accepts — that's an open
# question owed back to them on their ticket 86eywvpbt, not yet resolved.
SUBMIT_DEPOSIT_VERIFICATION_TOOL_SPEC = ToolSpec(
    name=SUBMIT_DEPOSIT_VERIFICATION,
    description=(
        "Submit the customer's deposit for manual verification against their proof "
        "screenshot. Call it for any disputed deposit that is NOT already confirmed "
        "successful: a failed one the customer insists went through, and equally a "
        "PENDING one — a deposit can be charged and left uncredited while its status "
        "still reads pending, so there is no need to wait for it to turn failed. A "
        "deposit already showing as successful is the only case this does not cover. "
        "Call get_player_latest_deposit_order first "
        "and pass its order_id as order_id — never invent one. Requires a "
        "screenshot to already be uploaded in this conversation — do not call this "
        "before the customer has sent one, ask them to upload it first. This takes "
        "a few minutes; the result will be delivered later in this same chat, not "
        "immediately — do not call this tool again while a submission is already "
        "pending."
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": (
                    "The order_id from get_player_latest_deposit_order's response "
                    "(always present). Fall back to that response's "
                    "external_transaction_id only if order_id is rejected — it is "
                    "null on exactly the pending/failed deposits a dispute is about."
                ),
            },
        },
        "required": ["order_id"],
    },
)
