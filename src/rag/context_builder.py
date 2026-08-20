"""Build RAG context for the LLM, and gate output via a hallucination guard.

The context builder takes ``RetrievedChunk`` results and formats them into
the ``rag_context`` block the ChatBot system prompt expects. Each chunk is
labeled with a stable source tag (``filename:section`` or just ``id``) so
the LLM can cite it via the ``sources_used`` field in its JSON response.

The hallucination guard is a small post-LLM check:
- If retrieval found nothing AND the response is non-empty AND ``confidence``
  isn't ``low``, override with a graceful "I don't know" answer.
- If the LLM cited a source we never gave it, downgrade ``confidence`` to
  ``low`` and strip the bogus citation. (The LLM gets the JSON schema
  embedded in the system prompt; this catches its lapses.)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.dialogue.response_parser import ChatBotResponse
from src.interfaces.vector_store import Document
from src.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from src.rag.retriever import HybridRetriever

log = logging.getLogger(__name__)


@dataclass
class RAGContext:
    text: str
    source_tags: list[str] = field(default_factory=list)
    chunk_count: int = 0


def build_rag_context(
    chunks: list[RetrievedChunk],
    max_chars: int = 4000,
) -> RAGContext:
    """Format retrieved chunks into a numbered context block.

    Truncates at ``max_chars`` so we don't blow past the LLM's input window
    on long retrievals. Truncation is on chunk boundaries — we never split
    mid-chunk, so the cited source is always either fully present or absent.
    """
    if not chunks:
        return RAGContext(text="(no relevant sources found)", source_tags=[], chunk_count=0)

    parts: list[str] = []
    tags: list[str] = []
    used_chars = 0
    for i, c in enumerate(chunks, start=1):
        tag = _source_tag(c)
        body = c.document.content.strip()
        block = f"[{i}] {tag}\n{body}"
        if used_chars + len(block) > max_chars and parts:
            break
        parts.append(block)
        tags.append(tag)
        used_chars += len(block) + 2  # +2 for separator

    return RAGContext(
        text="\n\n".join(parts),
        source_tags=tags,
        chunk_count=len(parts),
    )


def _source_tag(chunk: RetrievedChunk) -> str:
    md = chunk.document.metadata or {}
    filename = md.get("filename") or md.get("source")
    section = md.get("section") or md.get("page")
    if filename and section is not None:
        return f"{filename}:{section}"
    if filename:
        return str(filename)
    return chunk.document.id


# --- Multi-retriever helpers ---------------------------------------------


async def search_combined(
    query: str,
    retrievers: list["HybridRetriever"],
    top_k: int = 5,
    filters: Optional[dict] = None,
) -> list[RetrievedChunk]:
    """Query multiple retrievers in parallel and merge results by score."""
    if not retrievers:
        return []
    results_per = await asyncio.gather(
        *[r.search(query, top_k=top_k, filters=filters) for r in retrievers]
    )
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for chunk in sorted(
        (c for batch in results_per for c in batch),
        key=lambda c: c.score, reverse=True,
    ):
        if chunk.document.id not in seen:
            seen.add(chunk.document.id)
            merged.append(chunk)
    return merged[:top_k]


# Priority order for the voicebot's one-shot KB dump (src/agents/voicebot.py has
# no per-turn retrieval — this context is built once at call start, before the
# customer has said anything, so there's no query to rank against; this is a
# manually curated stand-in for "what's likely to come up on a sales call").
# Product/game content comes first since a customer asking "what casino games
# do you have" needs an actual answer, not just a link — followed by common
# trust objections, then lower-priority account-admin topics. Docs not listed
# here (e.g. a newly added KB file) still get included, ranked after these.
_VOICE_KB_PRIORITY: list[str] = [
    "06-casino-games.md",
    "07-sports-betting.md",
    "08-matka-lottery-games.md",
    "09-bonuses-and-promotions.md",
    "05-withdrawals.md",
    "04-deposits.md",
    "11-account-security.md",
    "10-responsible-gaming.md",
    "01-account-registration-login.md",
    "03-wallet-and-transactions.md",
    "02-kyc-identity-verification.md",
]
# Troubleshooting/support content — not a sales-call topic, excluded outright
# rather than left to crowd out product content by filename-sort luck.
_VOICE_KB_EXCLUDED: set[str] = {"12-technical-help.md"}


async def _kb_docs_for(r: "HybridRetriever") -> list[Document]:
    """Prefer the persistent-store-backed enumeration (sees chunks ingested by
    any process/worker); fall back to plain ``list_all()`` for retrievers that
    don't have ``list_all_persistent`` (e.g. fakes used in tests)."""
    lister = getattr(r, "list_all_persistent", None)
    if lister is not None:
        return await lister()
    return r.list_all()


async def build_voicebot_kb_context(
    retrievers: list["HybridRetriever"],
    max_chars: int = 15000,
) -> str:
    """Build a static KB context string from all docs in the given retrievers.

    Intended for injection into the voicebot system prompt at call start so the
    agent has factual KB content for the full call without per-turn retrieval.
    Docs are ordered by _VOICE_KB_PRIORITY (not filename/insertion order) so
    higher-value content is guaranteed to fit within max_chars.
    """
    seen: set[str] = set()
    all_docs: list[Document] = []
    for r in retrievers:
        for doc in await _kb_docs_for(r):
            if doc.id not in seen:
                seen.add(doc.id)
                all_docs.append(doc)
    if not all_docs:
        return ""

    def _filename(doc: Document) -> str:
        return (doc.metadata or {}).get("filename") or doc.id

    def _section(doc: Document) -> int:
        try:
            return int((doc.metadata or {}).get("section") or 0)
        except (TypeError, ValueError):
            return 0

    all_docs = [d for d in all_docs if _filename(d) not in _VOICE_KB_EXCLUDED]
    # Stable base sort on (filename, section) so pgvector's unordered rows
    # don't scramble intra-document chunk order. Must run BEFORE the priority
    # sort below — the priority sort's stability depends on this ordering
    # already being in place.
    all_docs.sort(key=lambda d: (_filename(d), _section(d)))
    all_docs.sort(key=lambda d: (
        _VOICE_KB_PRIORITY.index(_filename(d))
        if _filename(d) in _VOICE_KB_PRIORITY else len(_VOICE_KB_PRIORITY)
    ))

    parts: list[str] = []
    total = 0
    for doc in all_docs:
        fn = _filename(doc)
        entry = f"[{fn}]\n{doc.content.strip()}"
        if total + len(entry) + 2 > max_chars:
            break
        parts.append(entry)
        total += len(entry) + 2
    return "\n\n".join(parts)


# --- Hallucination guard -------------------------------------------------


@dataclass
class GuardConfig:
    require_sources_when_retrieved: bool = True
    fallback_text_en: str = (
        "I'm not able to find that in our documentation. "
        "Could you rephrase, or would you like me to connect you with a person?"
    )
    fallback_text_hi: str = (
        "Mujhe yeh humari documentation mein nahi mil raha. "
        "Kya aap dobara puch sakte hain ya main aapko team se connect karaun?"
    )


def apply_hallucination_guard(
    response: ChatBotResponse,
    rag_context: RAGContext,
    config: Optional[GuardConfig] = None,
) -> ChatBotResponse:
    """Audit the LLM's response against the retrieved sources.

    Returns the (possibly modified) response. The original is not mutated.
    """
    cfg = config or GuardConfig()

    # Defensive copy — caller may keep using the original.
    new = ChatBotResponse(
        response_text=response.response_text,
        language=response.language,
        sources_used=list(response.sources_used),
        confidence=response.confidence,
        action=response.action,
        suggested_followups=list(response.suggested_followups),
        raw=dict(response.raw),
        parse_error=response.parse_error,
    )

    # Drop bogus citations the LLM invented.
    available = set(rag_context.source_tags)
    if available and new.sources_used:
        valid = [s for s in new.sources_used if s in available]
        if len(valid) != len(new.sources_used):
            new.sources_used = valid
            new.confidence = "low"

    # No retrieval results -> reject and return the language-appropriate fallback.
    if rag_context.chunk_count == 0:
        if new.response_text:
            new.response_text = (
                cfg.fallback_text_hi if (new.language or "").startswith("hi")
                else cfg.fallback_text_en
            )
        new.confidence = "low"
        new.sources_used = []
        return new

    # Retrieval worked but the LLM gave a high-confidence answer with no
    # citations — a classic hallucination tell. Substitute the fallback.
    if (
        cfg.require_sources_when_retrieved
        and not new.sources_used
        and new.confidence == "high"
        and new.response_text
    ):
        new.response_text = (
            cfg.fallback_text_hi if (new.language or "").startswith("hi")
            else cfg.fallback_text_en
        )
        new.confidence = "low"
        return new

    return new


_TIME_OF_DAY_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b|\b\d{1,2}\.\d{2}\s*(am|pm|AM|PM)\b"
)
_CURRENCY_FIGURE_PATTERN = re.compile(r"(?:₹|\bRs\.?|\bINR)\s?\d[\d,]*\b", re.IGNORECASE)


def apply_no_grounding_guard(
    response: ChatBotResponse,
    retrieved_any: bool,
    tool_calls_made: list[str],
) -> ChatBotResponse:
    """Backstop for turns with NO grounding source at all — neither RAG
    retrieval nor a tool call. ``apply_hallucination_guard`` only runs when
    RAG retrieval happened, so a pure-parametric turn (no search, no tool)
    gets no code-level check today.

    Deliberately narrow: only downgrades confidence (never rewrites text)
    when a high-confidence, fully-ungrounded response matches a small set
    of lexically-detectable risk patterns (a stated time-of-day, a currency
    figure). It cannot catch fluent, well-formed prose fabrication that
    carries no such lexical tell (e.g. inventing specific game/variant
    names) — that class of failure is defended by prompt-level grounding
    rules instead, not this guard.
    """
    if retrieved_any or tool_calls_made:
        return response
    if response.confidence != "high" or not response.response_text:
        return response
    text = response.response_text
    if not (_TIME_OF_DAY_PATTERN.search(text) or _CURRENCY_FIGURE_PATTERN.search(text)):
        return response
    new = ChatBotResponse(
        response_text=response.response_text,
        language=response.language,
        sources_used=list(response.sources_used),
        confidence="low",
        action=response.action,
        suggested_followups=list(response.suggested_followups),
        raw=dict(response.raw),
        parse_error=response.parse_error,
    )
    log.warning(
        "no-grounding guard: high-confidence response with no retrieval/tool call "
        "matched a risk pattern",
        extra={"response_text": text[:200]},
    )
    return new
