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
    # Unverified-data guard fallbacks (ticket #1762 fix, Step 5 / review Fix 4
    # + Fix 5). Language-selected the same way as fallback_text_en/hi above —
    # not hardcoded English. Phrased as an OFFER requiring the customer's
    # confirmation ("would you like me to connect you...?"), not a
    # declarative promise ("let me connect you now") — consistent with the
    # existing ESCALATION prompt rule (offer first, wait for a yes; never
    # promise a handoff that isn't actually triggered yet). The "_with_figure"
    # variant takes one ``{figure}`` placeholder for the customer's own
    # disputed figure (e.g. "₹19,600").
    unverified_data_fallback_en: str = (
        "I'm not able to verify that right now — would you like me to connect you "
        "to someone who can check?"
    )
    unverified_data_fallback_en_with_figure: str = (
        "I can see you've reported {figure}, but I'm not able to verify it right "
        "now — would you like me to connect you to someone who can check?"
    )
    unverified_data_fallback_hi: str = (
        "Mujhe abhi ispe confirm karne mein dikkat ho rahi hai. Kya aap chahenge ki "
        "main aapko team se connect karaun jo ise check kar sake?"
    )
    unverified_data_fallback_hi_with_figure: str = (
        "Aapne {figure} bataya hai, lekin mujhe abhi ispe confirm karne mein dikkat "
        "ho rahi hai. Kya aap chahenge ki main aapko team se connect karaun jo ise "
        "check kar sake?"
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
_CURRENCY_FIGURE_PATTERN = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s?\d[\d,]*(?:\.\d+)?\b", re.IGNORECASE
)


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


# --- Unverified-data guard (ticket #1762 fix, Step 5) ---------------------

_NUMERIC_TOKEN_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _normalize_number(token: str) -> str:
    """Canonical, floating-point-safe form of a numeric string.

    '4,250.75', '4250.75', and '4250.750' must all normalize to the SAME
    value ('4250.75'); '2,075' and '2075.00' must both normalize to '2075'.
    Deliberately string-based (never float()/Decimal()) — a float round-trip
    can itself introduce representation drift (e.g. 4250.75 -> 4250.7499999),
    which would reintroduce the exact class of false-positive this fixes.

    Review Fix 1 (ticket #1762): the reply-side figure (matched via
    _CURRENCY_FIGURE_PATTERN) and the grounded-text side (matched via
    _NUMERIC_TOKEN_PATTERN) must run through this SAME normalization, or a
    real tool amount like 4250.75 fails to match a reply that states it.
    """
    cleaned = token.replace(",", "").strip()
    if "." in cleaned:
        integer_part, _, frac_part = cleaned.partition(".")
        frac_part = frac_part.rstrip("0")
        integer_part = integer_part.lstrip("0") or "0"
        return f"{integer_part}.{frac_part}" if frac_part else integer_part
    return cleaned.lstrip("0") or "0"


def _normalize_currency_match(match: str) -> str:
    """Strip a currency-figure regex match (e.g. '₹8,100' or '₹4,250.75')
    down to its normalized numeric token ('8100' / '4250.75') for comparison
    against grounded text."""
    m = _NUMERIC_TOKEN_PATTERN.search(match)
    return _normalize_number(m.group()) if m else match


def _numeric_tokens(text: str) -> set[str]:
    """Every bare numeric token in *text*, normalized the SAME way as
    _normalize_currency_match (see _normalize_number) — both sides of the
    comparison must treat equal values as equal regardless of comma/decimal
    rendering. Deliberately not restricted to currency-prefixed numbers: a
    tool result's raw JSON often carries an amount as a plain number (e.g.
    {"amount": 19600}) with no ₹/Rs prefix at all, and that must still count
    as grounding.

    Review Fix 1 (round 3, ticket #1762): a grounded decimal value (e.g.
    4250.75) also grounds its whole-rupee truncation and round-to-nearest-
    rupee forms ('4250' / '4251'), since a reply is allowed to state "about
    ₹4,250" for a real ₹4,250.75 balance without being flagged. Only ever
    ADDS forms derived from an already-grounded value — cannot loosen the
    guard against a value that was never grounded in the first place."""
    out: set[str] = set()
    for t in _NUMERIC_TOKEN_PATTERN.findall(text or ""):
        n = _normalize_number(t)
        out.add(n)
        if "." in n:
            whole, _, frac = n.partition(".")
            out.add(whole)
            rounded_up = str(int(whole) + 1) if frac and int(frac[0]) >= 5 else whole
            out.add(rounded_up)
    return out


def apply_unverified_data_guard(
    response: ChatBotResponse,
    grounded_text: str,
    customer_text: str = "",
    config: Optional[GuardConfig] = None,
) -> ChatBotResponse:
    """Model-independent last-line-of-defense guard (ticket #1762 fix, Step 5).

    Checks every currency figure (₹ / Rs / INR ...) in the reply against
    ``grounded_text`` — the caller assembles this from THIS turn's actually
    grounded sources: successful tool results, RAG context, the customer's
    query, and the customer's own prior turns (see the call site in
    src/agents/chatbot.py for exactly what goes in). If any figure in the
    reply is not backed by a matching numeric token anywhere in
    ``grounded_text``, the ENTIRE reply is replaced with a safe fallback
    (never partially edited — an unverified number is a signal the whole
    reply may not be trustworthy), confidence is downgraded to "low", and an
    error is logged.

    Deliberately independent of whether a tool was *called* this turn —
    unlike ``apply_no_grounding_guard``, which skips its check whenever any
    tool call happened THIS turn, even a FAILED one (the second concrete bug
    behind ticket #1762: a called-but-failed tool is exactly the case with no
    real grounding). This guard only asks whether the number is actually
    present in grounded text, never whether machinery ran.

    ``customer_text`` — the customer-authored portion of the conversation
    (their current query plus their own prior turns), used ONLY to pick a
    disputed figure to name in the fallback (e.g. "I can see you've reported
    a ₹19,600 withdrawal..." rather than a generic "I can't verify this"),
    per the incident's actual resolution wording. Deliberately not sourced
    from tool results or RAG content, so the guard can never end up
    "confirming" a number back to the customer that came from the model's own
    unverified claim.

    Known limitations (documented, not fixed — product decisions for the
    project owner, not defaults to change unilaterally):
    - ``customer_text`` only ever includes string-content turns. A figure the
      customer supplied via an IMAGE (e.g. a screenshot of their own
      transaction history, as in the original #1762 dispute) is invisible to
      the disputed-figure fallback wording, and a reply that correctly reads
      a number off that screenshot is blocked exactly like a fabricated one
      (fails safe, but isn't ideal) — there's no OCR path today.
    - Only digit-string figures are recognized. Hinglish multiplier phrasing
      ("2 lakh", "50 hazaar") is not parsed and will be treated as ungrounded
      even when correct.
    """
    if not response.response_text:
        return response
    matches = _CURRENCY_FIGURE_PATTERN.findall(response.response_text)
    if not matches:
        return response
    grounded_numbers = _numeric_tokens(grounded_text)
    unverified = [m for m in matches if _normalize_currency_match(m) not in grounded_numbers]
    if not unverified:
        return response

    log.error(
        "unverified-data guard: reply contained a currency figure with no grounding "
        "in this turn's tool results/RAG context/query/history — reply replaced",
        extra={"unverified_figures": unverified, "response_text": response.response_text[:200]},
    )
    cfg = config or GuardConfig()
    is_hindi = (response.language or "").startswith("hi")
    disputed = _CURRENCY_FIGURE_PATTERN.search(customer_text or "")
    if disputed:
        template = cfg.unverified_data_fallback_hi_with_figure if is_hindi \
            else cfg.unverified_data_fallback_en_with_figure
        text_out = template.format(figure=disputed.group())
    else:
        text_out = cfg.unverified_data_fallback_hi if is_hindi else cfg.unverified_data_fallback_en
    return ChatBotResponse(
        response_text=text_out,
        language=response.language,
        sources_used=[],
        confidence="low",
        action=response.action,
        suggested_followups=list(response.suggested_followups),
        raw=dict(response.raw),
        parse_error=response.parse_error,
    )
