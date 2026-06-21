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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.dialogue.response_parser import ChatBotResponse
from src.interfaces.vector_store import Document
from src.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from src.rag.retriever import HybridRetriever


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


def build_voicebot_kb_context(
    retrievers: list["HybridRetriever"],
    max_chars: int = 8000,
) -> str:
    """Build a static KB context string from all docs in the given retrievers.

    Intended for injection into the voicebot system prompt at call start so the
    agent has factual KB content for the full call without per-turn retrieval.
    """
    seen: set[str] = set()
    all_docs: list[Document] = []
    for r in retrievers:
        for doc in r.list_all():
            if doc.id not in seen:
                seen.add(doc.id)
                all_docs.append(doc)
    if not all_docs:
        return ""
    parts: list[str] = []
    total = 0
    for doc in all_docs:
        md = doc.metadata or {}
        fn = md.get("filename") or doc.id
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
