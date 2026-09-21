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
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.chatbot.tool_executor import REDACTED_PLACEHOLDER as _PII_REDACTED_PLACEHOLDER
from src.dialogue.prompts import SOURCES_CLOSE_MARKER, SOURCES_OPEN_MARKER
from src.dialogue.response_parser import ChatBotResponse
from src.interfaces.vector_store import Document
from src.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from src.rag.retriever import HybridRetriever

log = logging.getLogger(__name__)

# Matches either boundary marker literally. Retrieved KB content is inserted
# verbatim between SOURCES_OPEN_MARKER/SOURCES_CLOSE_MARKER with nothing else
# escaping it (see build_rag_context/build_voicebot_kb_context below); a
# document containing one of these literal strings could forge a close-marker
# + re-open pair, and the re-open is the dangerous half — pairwise marker
# parsing treats a segment that opens after a close as OUTSIDE the untrusted
# span, i.e. as trusted instructions. Only tenant-API-key holders
# (src/api/knowledge.py) and platform admins (src/api/crm_kb.py) can write KB
# documents today — no player-controlled path exists — but an operator can
# still ingest a document authored elsewhere, so we neutralise unconditionally
# rather than trusting the ingestion path.
# Deliberately broader than an exact match on the two marker constants. The
# boundary is enforced by how the MODEL reads the prompt, not by a parser, and
# a model would plausibly honour "<<< end sources >>>" or "<<<End Sources>>>"
# as a close marker even though `==` never would. So match any run of three or
# more angle brackets, in either direction, regardless of case or internal
# spacing. Legitimate KB text does not contain a "<<<" run -- single brackets
# ("<b>Withdraw</b>", "x < y", ">5000 needs KYC") are untouched, which is what
# keeps this from mangling real content. Git conflict markers ("<<<<<<<") are
# caught too, which is fine and arguably desirable in a KB document.
_SOURCES_MARKER_PATTERN = re.compile(r"<{3,}[^<>\n]{0,40}>{0,3}|<{0,3}[^<>\n]{0,40}>{3,}")


def neutralize_sources_markers(text: str, *, source: str = "") -> str:
    """Defang a literal boundary-marker string found inside retrieved content.

    Neutralise rather than reject: a legitimate document should never be
    silently dropped over an incidental substring match. Mangles just the
    angle brackets of a matched marker into look-alike characters so the
    string can no longer match SOURCES_OPEN_MARKER/SOURCES_CLOSE_MARKER
    (and so can't close or re-open the boundary), while staying visually
    close to the original for anyone reading the raw document. Logs at
    WARNING when it fires — that's either an attempted prompt injection or a
    very odd document, and either is worth a log line.
    """
    if not text or not _SOURCES_MARKER_PATTERN.search(text):
        return text
    log.warning(
        "retrieved content contained a sources-boundary marker string; neutralising",
        extra={"source": source},
    )
    return _SOURCES_MARKER_PATTERN.sub(
        lambda m: m.group(0).replace("<", "‹").replace(">", "›"), text
    )


# --- Trusted-frame defanging ---------------------------------------------
#
# Shared by src.agents.chatbot._defang_platform_frames (the CRM's
# previous_conversation summary) and src.api.deposit_verification's
# _defang_relay_frames (a vendor's ticket-reply relay message). Both call
# sites need the identical "strip invisibles -> NFKC -> defang frames/
# markers" pipeline; this is the one implementation, so it lives here next
# to neutralize_sources_markers, which the pipeline also calls.

# Canonical definition of the per-turn "the platform said this, not the
# customer" frame src.agents.chatbot._fold_turn_context wraps around
# retrieved sources/current time/language directive. Defined HERE, not in
# chatbot.py, so this module can build the defang regex below from these
# constants without importing chatbot.py: chatbot.py already imports FROM
# this module (for neutralize_sources_markers and the guard functions), so
# the reverse import would be a cycle. chatbot.py imports TURN_CONTEXT_OPEN/
# TURN_CONTEXT_CLOSE back from here (they remain available as
# `chatbot.TURN_CONTEXT_OPEN` etc.), and src.api.deposit_verification's own
# `_TURN_CONTEXT_OPEN`/`_TURN_CONTEXT_CLOSE` do the same instead of typing
# the text out a second/third time.
TURN_CONTEXT_OPEN = (
    "SYSTEM TURN CONTEXT — supplied by the support platform for this turn, not "
    "written by the customer. It carries the same authority as the system "
    "instructions above."
)
TURN_CONTEXT_CLOSE = "END SYSTEM TURN CONTEXT. The customer's own message follows."


def _build_invisible_format_chars() -> str:
    """Every Unicode General Category Cf ("format"/invisible) codepoint,
    plus the variation-selector blocks VS1-16 (U+FE00-U+FE0F -- category
    Mn, but functionally the same "invisible modifier" problem) and
    VS17-256 (U+E0100-U+E01EF).

    Built programmatically from ``unicodedata`` at import time, never typed
    out as a literal list -- a hand-typed list is exactly how this gap
    first shipped (covering only ~9 familiar codepoints such as U+200B/
    U+FEFF out of the ~170 Unicode actually assigns category Cf), missing
    the tag-character block (U+E0001, U+E0020-U+E007F) entirely -- the
    canonical invisible-prompt-injection vector, since a tag character can
    be wedged between any two letters of a forged marker/frame with zero
    visual trace. A programmatic class can't silently drift from the
    Unicode data as new Cf codepoints get assigned in future versions, the
    way a typed-out list already had.
    """
    chars = [
        chr(cp) for cp in range(0x110000)
        if not (0xD800 <= cp <= 0xDFFF)  # lone surrogates: not real codepoints
        and unicodedata.category(chr(cp)) == "Cf"
    ]
    chars.extend(chr(cp) for cp in range(0xFE00, 0xFE10))  # VS1-16
    chars.extend(chr(cp) for cp in range(0xE0100, 0xE01F0))  # VS17-256
    return "".join(chars)


_INVISIBLE_FORMAT_RE = re.compile("[" + re.escape(_build_invisible_format_chars()) + "]")


def _frame_regex(literal: str) -> re.Pattern[str]:
    """Compile *literal* (one of the two ``TURN_CONTEXT_*`` constants above)
    into a case-insensitive, whitespace-tolerant regex, instead of matching
    it exact-literal.

    Mirrors this module's own reasoning for ``_SOURCES_MARKER_PATTERN``
    above: "a model would plausibly honour ``<<< end sources >>>`` even
    though ``==`` never would". The same argument applies here -- a model
    asked to honour "SYSTEM TURN CONTEXT" would plausibly also honour
    "system turn context" or "System  Turn  Context" (extra/irregular
    whitespace), even though Python's ``==`` (what a plain ``str.replace``
    relies on) treats those as entirely unrelated strings. That argument
    covers approximate matching ONLY -- it says nothing about case folding
    or invisible characters, which change nothing a human or model
    perceives; those are closed instead by ``defang_trusted_frames``'s
    invisible-strip and NFKC steps below, run before this regex ever sees
    the text.
    """
    parts = re.split(r"\s+", literal.strip())
    pattern = r"\s+".join(re.escape(part) for part in parts)
    return re.compile(pattern, re.IGNORECASE)


_TURN_CONTEXT_OPEN_RE = _frame_regex(TURN_CONTEXT_OPEN)
_TURN_CONTEXT_CLOSE_RE = _frame_regex(TURN_CONTEXT_CLOSE)


def _has_forged_frame_or_marker(view: str) -> bool:
    """True if *view* contains a forged sources-marker run (``_SOURCES_MARKER_
    PATTERN``, the same pattern ``neutralize_sources_markers`` acts on) or a
    forged ``TURN_CONTEXT_OPEN``/``TURN_CONTEXT_CLOSE`` frame
    (case-insensitive, whitespace-tolerant). Detection-only -- never mutates
    or logs; ``defang_trusted_frames`` below only escalates to the logging,
    mutating helpers once this has already said yes on some view."""
    return bool(
        _SOURCES_MARKER_PATTERN.search(view)
        or _TURN_CONTEXT_OPEN_RE.search(view)
        or _TURN_CONTEXT_CLOSE_RE.search(view)
    )


def defang_trusted_frames(text: str, *, source: str = "") -> str:
    """Defang a trusted-content frame/marker forgery, WITHOUT rewriting text
    that never attempted one.

    Rationale for the two-tier structure: the old, unconditional pipeline
    (strip invisibles -> NFKC -> neutralize/defang) ran on every caller's
    text, including ordinary clean prose. ``_INVISIBLE_FORMAT_RE`` strips
    real, semantically-load-bearing characters in legitimate Indic text
    (ZWNJ in a Hindi conjunct, ZWJ in a Bengali ro-phola or an emoji ZWJ
    sequence) and NFKC can rewrite superscripts/fractions/lookalike glyphs
    into different, sometimes misleading, plain characters ("1½ lakh" ->
    "11⁄2 lakh", which *reads* as "11 lakh"). Both are fine to do to an
    attacker's payload; neither is acceptable to do to a customer's own
    message or a clean CRM/vendor summary. So: normalize only to LOOK for a
    forgery, and only pay the corruption cost on text that actually
    contains one.

    1. Build a DETECTION VIEW: strip invisible/format characters
       (``_INVISIBLE_FORMAT_RE``) then NFKC-normalize. This view is used
       only for the check below -- it is never returned.
    2. Check for a forged marker/frame in BOTH the original ``text`` and the
       detection view (``_has_forged_frame_or_marker``, which runs
       ``neutralize_sources_markers``'s marker pattern and both
       ``TURN_CONTEXT_*`` frame regexes -- all three already
       case-insensitive and whitespace-tolerant, so a verbatim, case-varied,
       or odd-whitespace forgery is already caught against the RAW text with
       no normalization at all). Checking both views catches strictly more
       than either alone: a verbatim/case/whitespace forgery appears
       unnormalized in the original text already; an obfuscated one (an
       invisible character wedged inside the frame text breaking contiguity,
       or a fullwidth/small-form bracket lookalike that only reads as ``<``/
       ``>`` after NFKC) appears only in the detection view once that
       obfuscation is undone. There is therefore no payload that is
       dangerous in the original text yet clean in BOTH views -- anything
       that could forge a frame downstream is caught by one check or the
       other here first.
    3. Nothing found in either view: return ``text`` completely UNCHANGED --
       no strip, no NFKC, byte-identical. This is the common case (clean
       prose, including the Indic/emoji/numeric-lookalike examples above)
       and it must be exact.
    4. Something found in either view: fall back to today's aggressive
       pipeline, run on the ALREADY-COMPUTED detection view (stripped +
       NFKC'd) rather than recomputing it -- ``neutralize_sources_markers``
       mangles any run of 3+ angle brackets, then the two ``TURN_CONTEXT_*``
       regexes catch the plain-English frame in any case/whitespace variant.
       Each match is replaced with the non-empty sentinel ``"[removed]"``,
       NEVER ``""`` -- an empty replacement would delete the frame but let
       the text immediately before and after it join up, which can itself
       assemble into a fresh forged marker/frame nothing here explicitly
       detects (e.g. ``"<<"`` immediately followed by ``"<SOURCES>>>"`` with
       the forged middle span removed empty would leave ``"<<<SOURCES>>>"``
       intact). The sentinel breaks that adjacency deliberately -- this is
       NOT a simplification opportunity; do not change it to ``""``.
       Corrupting an attacker's own prose on this path is fine; only the
       no-forgery path (step 3) needs to be byte-preserving.
    5. No-growth guarantee: NFKC can EXPAND text (e.g. U+FDFA is 1 character
       but NFKC-normalizes to an 18-character string), so step 4's output
       can be longer than the input even though every other transform here
       only shrinks or holds length steady. Callers such as
       ``src.agents.chatbot._fold_previous_conversation`` rely on this
       function never growing its input past a caller-enforced cap (see
       ``PREVIOUS_CONVERSATION_MAX_CHARS``) -- that cap is applied to the
       INPUT, not re-applied after this call, so if this function could grow
       the text, the effective bound folded into every LLM turn would not be
       the cap the caller thinks it is. Truncating to ``len(text)`` here
       makes "never longer than the input" a property of this shared layer
       itself, holding for every caller, present and future, rather than
       something each caller must separately re-enforce after the fact.
    """
    if not text:
        return text
    detection_view = unicodedata.normalize("NFKC", _INVISIBLE_FORMAT_RE.sub("", text))
    if not _has_forged_frame_or_marker(text) and not _has_forged_frame_or_marker(detection_view):
        return text
    cleaned = neutralize_sources_markers(detection_view, source=source)
    for pattern in (_TURN_CONTEXT_OPEN_RE, _TURN_CONTEXT_CLOSE_RE):
        cleaned = pattern.sub("[removed]", cleaned)
    # Step 5 above: NFKC (folded into detection_view) can expand text; this
    # is the only place that can happen on this path, so truncating here is
    # sufficient to guarantee the whole function never returns something
    # longer than its input.
    if len(cleaned) > len(text):
        cleaned = cleaned[: len(text)]
    return cleaned


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
        body = neutralize_sources_markers(c.document.content.strip(), source=tag)
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
        entry = f"[{fn}]\n{neutralize_sources_markers(doc.content.strip(), source=fn)}"
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
        # Review Fix 1: redact PII out of the logged slice BEFORE truncating
        # to 200 chars -- this guard runs before apply_pii_guard ever sees
        # this text (see apply_pii_guard's own docstring for why the guard
        # ORDER itself isn't changing), so without this a mobile/email/
        # account number sitting in an otherwise-risk-flagged reply landed
        # in this WARNING log completely unredacted.
        extra={"response_text": _redact_pii_for_log(text)[:200]},
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
        # Review Fix 1: same redact-before-truncate treatment as
        # apply_no_grounding_guard's WARNING above -- this guard runs before
        # apply_pii_guard too, so a mobile/email/account number co-occurring
        # with the unverified figure that triggered this ERROR used to reach
        # the log completely unredacted. `unverified_figures` itself is a
        # list of currency-figure substrings (amounts, not customer PII) and
        # is left as-is.
        extra={
            "unverified_figures": unverified,
            "response_text": _redact_pii_for_log(response.response_text)[:200],
        },
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


# --- PII guard (outbound counterpart to tool_executor._redact_internal_ids) -

# Reused/adapted from the (unrelated, deliberately NOT imported) aggressive
# trace-redaction module at src/observability/trace_redaction.py, which
# solves a different problem: it is a blunt, deny-by-default scrubber for
# payloads headed to a debugging backend, where over-redaction is the
# explicitly preferred failure mode. This guard runs on text a customer is
# about to receive; mangling a legitimate reply happens on every clean turn,
# so it is tuned in the OPPOSITE direction -- catch the shapes below with
# high confidence, leave everything else (amounts, dates, bet/transaction
# references, OTPs) alone. Defined locally rather than imported so a future
# change to the tracing module's aggressive ruleset can never silently
# change what reaches a customer.

# Email: same bounded-quantifier shape as trace_redaction.py's _EMAIL_RE
# (64-char local part, 63-char domain labels, up to 8 labels, 24-char TLD) --
# copied for the same reason it exists there: unbounded quantifiers on
# `local@domain`-shaped adversarial text are a measured ReDoS (~38s on 100KB
# of crafted input in that module's own regression test). \b on both ends
# means a genuine email is always fully bounded by non-word characters.
_PII_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._%+\-]{0,63}"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.){1,8}[A-Za-z]{2,24}\b"
)

# Indian mobile number: optional +91/91/0 prefix, then 10 digits starting
# 6-9, with EITHER no separator, or a single space/hyphen separator at one
# of the three groupings realistically seen in this product's replies: 5-5
# ("98765 43210"), 3-3-4 ("987 654 3210", review Fix 5), or 4-3-3
# ("9876 543 210", review Fix 5) -- 3-3-4/4-3-3 were a documented gap before
# Fix 5; only 5-5 (and no separator) were matched.
# Uses \b WORD boundaries, not the digit-adjacency lookarounds
# ((?<!\d)/(?!\d)) trace_redaction.py uses for its own phone pattern. This is
# a deliberate divergence, not an oversight: \b also refuses to match when
# the character on either side is a LETTER (letters are \w too), so a
# transaction/bet reference formatted as "TXN9876543210" or "BET9123456780"
# -- both realistic in this product's replies -- is correctly left alone,
# where a digit-adjacency lookaround would still fire on the embedded
# digits (verified empirically against both patterns before choosing this
# one). \b still rejects a 10-digit-shaped substring embedded in a longer
# PURE digit run (a 12-digit Aadhaar, a 14-digit account number) the same as
# the lookaround would, since digit-to-digit is also a \w-to-\w, non-boundary
# transition. The one thing \b gives up versus the lookaround: a "+"
# immediately before the digits is dropped from the match when preceded by
# whitespace (\W-to-\W is not a boundary), e.g. "+91-98765-43210" matches as
# "91-98765-43210" -- harmless, since the "+" itself is not sensitive and the
# digits are still fully redacted.
_PII_MOBILE_PATTERN = re.compile(
    r"\b(?:\+?91[-\s]?|0)?"
    r"(?:[6-9]\d{9}"                      # no separator: 9876543210
    r"|[6-9]\d{4}[-\s]\d{5}"              # 5-5: 98765 43210
    r"|[6-9]\d{2}[-\s]\d{3}[-\s]\d{4}"    # 3-3-4: 987 654 3210
    r"|[6-9]\d{3}[-\s]\d{3}[-\s]\d{3})"   # 4-3-3: 9876 543 210
    r"\b"
)

# Bank account number: any contiguous digit run 9-18 characters long, found
# via a maximal-match \d+ scan (not a fixed-width \d{9,18} regex) so a run
# LONGER than 18 digits is rejected in full rather than partially matched at
# whatever 18-digit window happens to sit next to a non-digit boundary --
# "reject runs that are part of a longer sequence," not just the
# embedded-in-something-else case the mobile pattern handles above.
#
# \b on both ends (review Fix 3): without it, a digit run embedded in a
# larger alphanumeric token -- "TXN9876543210", a realistic deposit/
# transaction reference -- was extracted as its own bare 10-digit run and,
# if an account anchor word happened to sit nearby (e.g. "...while
# depositing to your bank account"), wrongly redacted as a bank account
# number. \b closes exactly the gap the mobile pattern's own \b already
# closes for mobile-shaped runs (see its comment above): a digit-to-letter
# transition is \w-to-\w, never a boundary, so \b\d+\b simply never matches
# starting mid-token. A run bounded by non-word characters on both sides
# (the real bank-account case) is unaffected.
#
# This is the highest false-positive-risk category in this guard, so unlike
# mobile/email it is NOT emitted on shape alone. It additionally requires an
# "account"-flavoured word within _PII_ACCOUNT_ANCHOR_WINDOW characters on
# either side -- the same anchor-word discipline the voice sentence guard
# uses for currency detection ("a guard that fires on 'press 1 for support'
# gets switched off"). Without this, a 9-18 digit transaction/bet reference
# (realistic in this product: "TXN9876543210", a PGS order number) would be
# indistinguishable from a real bank account number by shape alone.
# Known limitation (documented, not fixed): a bank account number relayed
# with NO account-ish wording anywhere nearby is not caught -- the
# alternative (shape-only matching) was measured to false-positive on
# exactly the transaction/reference content this guard must not touch.
_PII_DIGIT_RUN_PATTERN = re.compile(r"\b\d+\b")
# Bank-account anchor words. The optional "no./number" suffix is a SHARED
# trailing group applied to whichever prefix branch matches (not duplicated
# per-branch) so that "bank account number" is captured as ONE match instead
# of stopping at "bank account" and leaving "number" as an unclaimed gap --
# that gap was previously wide enough for a disqualifier word appearing
# right after the digit run (e.g. "...account number is 5010...789; the UTR
# will follow") to end up NEARER to the digit run than the anchor match,
# defeating the nearest-wins logic below. ifsc/beneficiary have no such
# suffix and are kept as separate bare alternatives.
_PII_ACCOUNT_ANCHOR_PATTERN = re.compile(
    r"\b(?:bank\s*accounts?|a/c|acct|acc(?:ounts?)?)\.?\s*(?:no\.?|number)?\b"
    r"|\bifsc\b|\bbeneficiary\b",
    re.IGNORECASE,
)
# Review Fix 5: "acct" (e.g. "Your acct no. is ...") is common shorthand this
# product's replies can plausibly use and was previously not recognized at
# all -- note it needs its own alternative rather than relying on
# "acc(?:ounts?)?": that branch requires "acc" to be followed by either
# nothing or "ount(s)", so "acct" fails its own trailing \b (the "t" sits
# right before the shared "no./number" suffix group with no boundary before
# it) and the whole alternative fails outright for this input, same as it
# would for any other non-"acc(ount(s))" word starting with "acc".

# Review Fix 3: "on account of" is ordinary English ("On account of the
# delay, your refund id ... was reissued") that has nothing to do with a
# bank account, but its bare "account" token still satisfies
# _PII_ACCOUNT_ANCHOR_PATTERN. Checked explicitly (not just left to the
# disqualifier list above, which is about a CO-OCCURRING reference word, not
# about the anchor word itself being idiomatic) so this specific idiom's
# "account" token is never treated as a real anchor no matter how close it
# sits to an unrelated digit run.
_PII_ACCOUNT_ANCHOR_IDIOM_PATTERN = re.compile(r"\bon\s+account\s+of\b", re.IGNORECASE)
# Reference-type words that mean the nearby digit run is a transaction/order
# reference rather than a bank account number, even if an account-ish word
# ALSO happens to be in the same window (e.g. "credited to your account;
# reference number 123456789012" -- "account" and "reference" both present,
# but this is a reference number, not an account number). Checked FIRST,
# and wins over a positive anchor match if both are present -- consistent
# with this guard's overall bias toward under- rather than over-redacting.
# Deliberately narrow (reference/order/UTR/ticket-shaped words only, not
# "transaction" or "bet") -- a broader list risks suppressing a genuine
# leaked account number in phrasing like "...used for this transaction" or
# "your bet win goes to account 501234567890", which are realistic in this
# product and must still be redacted.
_PII_ACCOUNT_DISQUALIFIER_PATTERN = re.compile(
    r"\b(?:order|utr|reference|pgsorderid|ticket|confirmation|invoice|receipt)\b",
    re.IGNORECASE,
)
# Review Fix 5: widened from 40 to 60. Reproduced gap: "Your bank account
# details are as follows for the pending refund case: 50100123456789" -- the
# anchor ("bank account") sits 53 chars from the digit run, comfortably past
# the old 40-char window, and "...as follows for..." (or "...as follows:...")
# is ordinary phrasing this product's replies can plausibly use before
# stating a number. 60 was chosen as the smallest round number that covers
# this reproduction with headroom, not the largest window that still avoids
# false positives -- the trade re-checked against the full existing test
# suite before landing: widening does not flip any currently-passing
# "must stay unredacted" case (e.g. "on account of ... refund id ...", the
# unrelated-anchor/reference-number cases) into a false positive, because
# those are excluded by the disqualifier/idiom checks above, not by window
# size. A window this size does make it more likely for an unrelated
# disqualifier word further down the same sentence to now fall inside the
# window too -- but _pii_account_matches's nearest-wins comparison (not
# any-match-vetoes) already handles that: only a disqualifier STRICTLY
# CLOSER than the anchor overrides it.
_PII_ACCOUNT_ANCHOR_WINDOW = 60
_PII_ACCOUNT_MIN_DIGITS = 9
_PII_ACCOUNT_MAX_DIGITS = 18

# Currency amount with the currency word AFTER the number instead of a
# ₹/Rs/INR prefix before it (e.g. "8100 rupees", "123456789 rupees") --
# _CURRENCY_FIGURE_PATTERN only covers the prefixed form. Needed here (not
# added to _CURRENCY_FIGURE_PATTERN itself, which the OTHER guards in this
# module also use and were not asked to change) because a bare amount in
# this suffix form is exactly the kind of number this guard must not treat
# as a leaked bank account digit run.
# Bounded to 24 repeats of `[\d,]` (generous headroom for any realistic
# currency amount, thousands separators included) rather than the unbounded
# `[\d,]*` this used to be -- an unbounded leading-\d, unanchored quantifier
# here is a measured ReDoS: ~13.8s on a 16,000-digit string with no trailing
# currency word, since the engine backtracks over every possible split point
# once the required (?:rupees?|rs\.?|inr)\b tail fails to match at the end of
# a long pure-digit run. Same fix shape as _PII_EMAIL_PATTERN's own bounded
# quantifiers above, and the same class of bug src/observability/
# trace_redaction.py's TestReDoSFix regression tests guard against.
_PII_BARE_WORD_AMOUNT_PATTERN = re.compile(
    r"\d[\d,]{0,24}(?:\.\d+)?\s?(?:rupees?|rs\.?|inr)\b", re.IGNORECASE
)

# _PII_REDACTED_PLACEHOLDER is imported at module top as an alias for
# src/chatbot/tool_executor.py's REDACTED_PLACEHOLDER ("so it can never
# silently drift out of sync" per that module's own comment) -- no circular
# import risk, since tool_executor.py imports nothing from src.rag.


def _normalize_digits(value: str) -> str:
    """Strip every character except ASCII digits from *value*.

    Used only by apply_pii_guard's safe_to_state matching, to compare a
    value's digit identity across cosmetic formatting differences
    (spaces/hyphens) between a CRM payload and the reply text; deliberately
    distinct from _normalize_number above, which is decimal/comma-aware for
    CURRENCY AMOUNT comparison and strips leading zeros -- an account/UPI
    number has no fractional part, and a leading zero is part of its
    identity, not noise to strip.
    """
    return "".join(ch for ch in value if ch.isdigit())


def _nearest_match_distance(
    pattern: re.Pattern, window: str, run_start: int, run_end: int,
) -> Optional[int]:
    """Character distance from the closest match of *pattern* in *window* to
    the digit-run span [run_start, run_end) (both window-relative offsets
    into *window*). 0 if some match overlaps the run itself; None if
    *pattern* has no match in *window* at all."""
    best: Optional[int] = None
    for cand in pattern.finditer(window):
        if cand.end() <= run_start:
            dist = run_start - cand.end()
        elif cand.start() >= run_end:
            dist = cand.start() - run_end
        else:
            dist = 0
        if best is None or dist < best:
            best = dist
    return best


def _account_anchor_distance(window: str, run_start: int, run_end: int) -> Optional[int]:
    """Like ``_nearest_match_distance(_PII_ACCOUNT_ANCHOR_PATTERN, ...)``, but
    a candidate anchor match that falls entirely inside an "on account of"
    idiom span (_PII_ACCOUNT_ANCHOR_IDIOM_PATTERN) is skipped -- review
    Fix 3. Kept as its own function rather than an extra parameter on
    _nearest_match_distance, since the idiom exclusion is specific to the
    account anchor (the disqualifier-word lookup a few lines below still
    uses the generic function unmodified)."""
    idiom_spans = [m.span() for m in _PII_ACCOUNT_ANCHOR_IDIOM_PATTERN.finditer(window)]
    best: Optional[int] = None
    for cand in _PII_ACCOUNT_ANCHOR_PATTERN.finditer(window):
        if any(cand.start() >= s and cand.end() <= e for s, e in idiom_spans):
            continue
        if cand.end() <= run_start:
            dist = run_start - cand.end()
        elif cand.start() >= run_end:
            dist = cand.start() - run_end
        else:
            dist = 0
        if best is None or dist < best:
            best = dist
    return best


def _pii_account_matches(
    text: str, exclude_spans: list[tuple[int, int]],
) -> list[re.Match]:
    """Bank-account-shaped digit runs in *text*, anchor-gated and
    disqualifier-aware (see _PII_ACCOUNT_ANCHOR_PATTERN /
    _PII_ACCOUNT_DISQUALIFIER_PATTERN above), excluding any span already
    claimed by *exclude_spans* -- this guard's own mobile matches, plus
    money-figure spans from _CURRENCY_FIGURE_PATTERN and
    _PII_BARE_WORD_AMOUNT_PATTERN, so a legitimate amount is never mistaken
    for an account number just because it normalizes to a long digit run.

    NEAREST-WINS, not any-match-vetoes: a candidate is redacted only if an
    account-anchor word exists in the window AND no disqualifier word is
    STRICTLY CLOSER to the digit run than the nearest anchor. An earlier
    version skipped redaction whenever ANY disqualifier word appeared
    anywhere in the (±40 char) window, which correctly fixed the
    reference-number false positives (see the disqualifier pattern's own
    docstring) but over-corrected: it also suppressed genuine leaked account
    numbers in realistic phrasing like "Your bank account number is
    5010...789; the UTR will follow shortly" -- the UTR mention, though
    clearly unrelated to the account number two clauses earlier, still fell
    inside the same window. Comparing distances instead of presence keeps
    both fixed: a disqualifier word sitting right next to the digit run
    (the true reference-number case) still wins, but a disqualifier word
    elsewhere in the same sentence no longer overrides an anchor word
    sitting immediately next to the number.
    """
    out: list[re.Match] = []
    for m in _PII_DIGIT_RUN_PATTERN.finditer(text):
        run_len = len(m.group())
        if not (_PII_ACCOUNT_MIN_DIGITS <= run_len <= _PII_ACCOUNT_MAX_DIGITS):
            continue
        if any(m.start() < e and m.end() > s for s, e in exclude_spans):
            continue
        window_start = max(0, m.start() - _PII_ACCOUNT_ANCHOR_WINDOW)
        window_end = min(len(text), m.end() + _PII_ACCOUNT_ANCHOR_WINDOW)
        window = text[window_start:window_end]
        run_start_rel = m.start() - window_start
        run_end_rel = m.end() - window_start
        anchor_dist = _account_anchor_distance(window, run_start_rel, run_end_rel)
        if anchor_dist is None:
            continue
        disqualifier_dist = _nearest_match_distance(
            _PII_ACCOUNT_DISQUALIFIER_PATTERN, window, run_start_rel, run_end_rel,
        )
        if disqualifier_dist is not None and disqualifier_dist < anchor_dist:
            continue
        out.append(m)
    return out


def _find_pii_hits(text: str) -> list[tuple[int, int, str]]:
    """Every PII-shaped span in *text*: ``(start, end, type)`` with type in
    ``{"mobile", "email", "account"}``, sorted start-ascending / longest-
    first on a tie (see apply_pii_guard's own sort comment below for why).

    This is the shared detection core, factored out so it can be applied
    to a bare log string (review Fix 1, see ``_redact_pii_for_log`` below)
    or to a ``suggested_followups`` entry (review Fix 2) without dragging in
    the full ``ChatBotResponse``-shaped machinery of ``apply_pii_guard``
    itself.

    Money-span handling (review Fix 4): a PREFIX currency figure (``₹``/
    ``Rs``/``INR`` before the digits, `_CURRENCY_FIGURE_PATTERN`) is
    unambiguous -- the currency marker literally precedes the number, so it
    can only be an amount -- and continues to suppress an overlapping mobile
    match, e.g. "Rs 9876543210" stays an amount, never a mobile number. A
    SUFFIX currency word trailing the digits (``rupees``/``rs``/``inr``
    AFTER the number, `_PII_BARE_WORD_AMOUNT_PATTERN`) does NOT suppress an
    overlapping mobile match any more: "9876543210 INR" / "9876543210 rs"
    (the latter is everyday Hinglish, not just formal currency notation)
    were leaking a real 10-digit mobile-shaped run because the trailing
    currency word happened to make it look like an amount. Shape (10 digits
    starting 6-9) is a stronger and more dangerous signal than an ambiguous
    trailing unit word, so the mobile interpretation wins.
    Accepted, documented tradeoff: a genuine 10-digit amount starting 6-9
    immediately followed by a currency word (e.g. "9500000000 rupees", a
    literal 950-crore payout) is now indistinguishable from a mobile number
    by shape alone and gets misclassified/redacted as one -- see
    ``test_pii_guard_large_bare_word_amount_shaped_like_mobile_is_redacted_as_mobile_documented_tradeoff``.
    This is deliberately accepted: a false positive on an amount that large
    is vanishingly rare for this product, whereas the false negative it
    replaces (a real mobile number reaching the customer) is the exact class
    of leak this guard exists to prevent. A PREFIX-form amount of the same
    size ("Rs 9500000000") is unaffected, since that overlap check is
    unchanged.

    Suffix-form money spans are still used, unchanged, to keep an account-
    shaped bare amount ("123456789 rupees") from being mistaken for a bank
    account number -- only the MOBILE overlap check narrowed to prefix-only.
    """
    prefix_money_spans = [m.span() for m in _CURRENCY_FIGURE_PATTERN.finditer(text)]
    suffix_money_spans = [m.span() for m in _PII_BARE_WORD_AMOUNT_PATTERN.finditer(text)]
    money_spans = prefix_money_spans + suffix_money_spans

    mobile_matches = [
        m for m in _PII_MOBILE_PATTERN.finditer(text)
        if not any(m.start() < e and m.end() > s for s, e in prefix_money_spans)
    ]
    email_matches = list(_PII_EMAIL_PATTERN.finditer(text))
    mobile_spans = [m.span() for m in mobile_matches]
    account_matches = _pii_account_matches(text, money_spans + mobile_spans)

    hits: list[tuple[int, int, str]] = (
        [(m.start(), m.end(), "mobile") for m in mobile_matches]
        + [(m.start(), m.end(), "email") for m in email_matches]
        + [(m.start(), m.end(), "account") for m in account_matches]
    )
    # Sort by start ascending, longest-first on a tie, so a span that fully
    # CONTAINS another (e.g. an email whose local part happens to also be
    # mobile-shaped, "9876543210@gmail.com") is the one applied, rather than
    # the smaller contained span leaving the rest of the containing match
    # (the "@gmail.com" domain) unredacted.
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    return hits


def _pii_hit_is_exempt(matched_text: str, typ: str, safe_digits: set[str]) -> bool:
    """True if a single PII hit is covered by ``safe_to_state`` (see
    apply_pii_guard's own docstring for the full contract). Restricted to
    account/mobile hits of a realistic minimum length -- never "email" --
    and shared between response_text redaction and suggested_followups
    filtering (review Fix 2) so both apply the exact same exemption rule."""
    return (
        bool(safe_digits)
        and typ in ("account", "mobile")
        and len(_normalize_digits(matched_text)) >= _PII_ACCOUNT_MIN_DIGITS
        and _normalize_digits(matched_text) in safe_digits
    )


def _redact_pii_hits(
    text: str, hits: list[tuple[int, int, str]], safe_digits: set[str],
) -> tuple[str, set[str]]:
    """Rebuild *text* with every hit in *hits* either left untouched (if
    exempt via ``safe_digits``, see ``_pii_hit_is_exempt``) or replaced with
    ``_PII_REDACTED_PLACEHOLDER``. Returns ``(rebuilt_text, types_found)``,
    where ``types_found`` excludes exempted hits -- the same bookkeeping
    apply_pii_guard itself used to do inline, now shared with
    ``_redact_pii_for_log``."""
    out_parts: list[str] = []
    last = 0
    types_found: set[str] = set()
    for start, end, typ in hits:
        if start < last:
            continue  # already covered by a preceding, larger/earlier span
        matched_text = text[start:end]
        if _pii_hit_is_exempt(matched_text, typ, safe_digits):
            # Caller has vouched for this exact value (e.g. the operator's
            # own deposit account from get_payment_config this turn) --
            # leave it untouched, and don't count it toward types_found, so
            # a reply made up entirely of exempted matches doesn't trigger
            # the confidence downgrade below. Preserve the gap since the
            # previous span (or string start) too, exactly like the redact
            # branch below does -- without this, the untouched text between
            # spans was silently dropped from the rebuilt string.
            out_parts.append(text[last:start])
            out_parts.append(matched_text)
            last = end
            continue
        out_parts.append(text[last:start])
        out_parts.append(_PII_REDACTED_PLACEHOLDER)
        last = end
        types_found.add(typ)
    out_parts.append(text[last:])
    return "".join(out_parts), types_found


def _redact_pii_for_log(text: str) -> str:
    """Redact PII shapes out of *text* before it is embedded in a log
    payload (review Fix 1).

    apply_no_grounding_guard's WARNING and apply_unverified_data_guard's
    ERROR log calls both include a slice of the model's raw response_text
    for diagnostics. apply_pii_guard -- the guard that would otherwise catch
    a mobile/email/account number in that text -- runs LAST in the pipeline,
    deliberately AFTER both of these (see apply_pii_guard's own "Known
    limitations": the ordering itself is not changing, since running PII
    first would force confidence="low" and disable the other guards'
    confidence=="high" gates). Without this, a reply that tripped one of
    those sibling guards WHILE also containing real PII put the raw PII
    value straight into an ERROR/WARNING log, never reaching apply_pii_guard
    at all.

    This is the SAME detection core apply_pii_guard uses (``_find_pii_hits``
    / ``_redact_pii_hits``), applied here purely as a log-safety pass -- it
    never touches the customer-facing response_text, and it deliberately
    ignores safe_to_state entirely (passing an empty exemption set): a log
    line has no notion of a per-turn allow-set, and over-redacting an
    operator's own account number in a diagnostic log line costs nothing.
    Keeps logging the redacted text rather than dropping it -- it's still
    diagnostically useful to see the shape/length of the reply that tripped
    the guard, just never the PII itself.
    """
    if not text:
        return text
    hits = _find_pii_hits(text)
    if not hits:
        return text
    redacted, _types = _redact_pii_hits(text, hits, safe_digits=set())
    return redacted


def apply_pii_guard(
    response: ChatBotResponse, safe_to_state: Optional[set[str]] = None,
) -> ChatBotResponse:
    """Outbound counterpart to tool_executor._redact_internal_ids.

    That function makes leaking operator_id/user_id/tenant_id/crm_id/
    session_id "structurally impossible instead of just discouraged" on the
    way IN to the model. Nothing plays the same role on the way OUT: the
    system prompt (src/dialogue/prompts.py's "keep internals internal" /
    DATA RULE / IDENTITY CONFIRMATION rules) tells the model never to state a
    customer's contact details even when a tool genuinely returns one -- but
    that is an LLM instruction, not a guarantee, exactly the gap
    _redact_internal_ids's own docstring calls out for the inbound side.
    This is that same guarantee applied outbound, for customer PII instead
    of internal plumbing.

    Detects (see the patterns above for the false-positive reasoning behind
    each): Indian mobile numbers, email addresses, and anchor-gated bank
    account numbers. Each match in response_text is redacted IN PLACE
    (_PII_REDACTED_PLACEHOLDER) -- unlike apply_unverified_data_guard, a
    leaked contact detail is a specific token in an otherwise
    possibly-correct answer, not a signal the whole reply is untrustworthy,
    so only the offending span is removed and the rest of the reply
    survives. confidence is downgraded to "low" (a turn that needed this
    guard is not one to trust blindly) and an ERROR is logged with the
    matched TYPE(s) only (`pii_types`) -- deliberately NOT response_text or
    the matched value itself, unlike the sibling guards in this module,
    because logging the PII to defend against leaking it would be
    self-defeating (matches tool_executor.py's own `param_keys`-only
    logging precedent).

    ``suggested_followups`` (review Fix 2) are scanned with the exact same
    detection core, since they are customer-visible on every surface
    response_text is: ``src/api/chat.py``'s REST ``_to_message_response``
    (``suggested_followups=r.suggested_followups``) and its WebSocket
    ``"suggestions"`` frame, and ``src/api/external_chat.py``'s
    ``ExternalMessageResponse.suggestions``. Checked and confirmed NOT
    customer-visible on any of those three surfaces: ``response.raw`` --
    none of them read it, so it needs no scanning. A followup that leaks PII
    is DROPPED from the list entirely, never redacted-in-place like
    response_text is: a followup is a SUGGESTED REPLY the customer might
    tap/say verbatim, and "Should I send the OTP to [redacted]?" is not a
    coherent thing to hand a customer as a one-tap suggestion -- unlike a
    redacted span sitting inside an otherwise-normal sentence, a redacted
    span standing in for the entire payload of a short suggested question
    reads as broken, not merely cautious. A dropped/redacted followup or
    response_text span, in either case, downgrades confidence to "low" and
    logs the same ERROR (pii_types covers hits from both response_text and
    dropped followups; a `followups_dropped` count is included whenever at
    least one followup was dropped).

    No GuardConfig parameter -- unlike apply_hallucination_guard/
    apply_unverified_data_guard, this guard has no fallback text to
    template (it redacts a span, it doesn't replace the whole reply), so
    there is nothing in GuardConfig for it to consume. Matches
    apply_no_grounding_guard's precedent of the same shape for the same
    reason.

    Runs regardless of whether the customer supplied the value themselves
    first (e.g. "is my number 98XXXXXXXX?" echoed back as "Yes, ...
    9812345678 ...") -- the prompt's IDENTITY CONFIRMATION rule forbids
    confirming it that way for exactly this reason, and this guard has no
    way to know provenance even if it wanted to make that distinction.

    safe_to_state is an optional allow-set of values the CALLER has already
    verified are safe to state to the user this turn -- e.g. the operator's
    own deposit account number, returned by a CRM tool call earlier in the
    same turn, which is not customer PII at all and should not be redacted
    out of a deposit-instructions reply. Matching is on NORMALIZED DIGITS
    (see _normalize_digits), not raw text, so cosmetic formatting
    differences (spaces/hyphens) between how a CRM payload spells the number
    and how the model's reply spells it don't defeat the exemption. This
    function has no concept of "turn" itself -- safe_to_state is trusted
    as-is, whatever it contains; scoping it to the current turn (and to the
    right tool's own output) is entirely the caller's responsibility.

    PAN and Aadhaar were considered and deliberately excluded:
    - PAN's shape (5 letters + 4 digits + 1 letter, e.g. "ABCDE1234F") is
      distinctive versus this product's dates/amounts/bet-ids, but collides
      with a plausible promo/bonus/referral code shape ("PROMO1234X") that
      this product's replies could legitimately contain -- and unlike the
      three patterns above, this wasn't verified against this product's
      actual reply content, so the risk wasn't taken.
    - Aadhaar's most distinctive form (12 digits spaced 4-4-4) was left out
      for the same reason. A CONTIGUOUS 12-digit Aadhaar number is still
      caught by the anchor-gated account-number pattern above if it happens
      to sit near account-ish wording, but the spaced form, or one with no
      nearby anchor at all, is a known gap.

    Known limitations (documented, not fixed):
    - This guard runs after the other reply guards (hallucination/
      no-grounding/unverified-data) in the pipeline, by design, so their own
      confidence gates and canned-fallback substitutions aren't disturbed by
      this guard's confidence downgrade. FIXED (review Fix 1, was
      previously documented here as an accepted gap covering only the
      no-grounding WARNING path and not mentioning the ERROR path at all):
      both ``apply_no_grounding_guard``'s WARNING log call and
      ``apply_unverified_data_guard``'s ERROR log call now redact PII out of
      the ``response_text`` slice they log (via ``_redact_pii_for_log``,
      the same detection core this guard uses) before this guard ever runs
      -- so a reply that trips one of those sibling guards while also
      containing real PII no longer puts the raw value into an ERROR/WARNING
      log line. The guard ORDER itself is unchanged and is not the fix: PII
      guard still runs last, for the reason above.
    - A bank account or mobile number rendered with spaces/hyphens IN THE
      REPLY TEXT ITSELF is not detected at all (the account/mobile patterns
      require a contiguous digit run to fire in the first place) --
      ``safe_to_state``'s digit-normalization only ever helps match a
      contiguous reply-side span against a differently-formatted CRM value;
      it does not make a spaced/hyphenated reply-side number detectable when
      it wasn't already.
    - The ``safe_to_state`` exemption only ever applies on the tool-loop path
      (``_handle_with_tools``), where a CRM tool call's own result is
      available to build the allow-set from. On the single-shot/no-tool-loop
      path, or if an operator's deposit account is ever sourced from a
      knowledge-base chunk rather than a live ``get_payment_config`` call,
      this guard has no way to know the value is safe and will redact it
      like any other account-shaped number -- a caller who needs the
      exemption in another code path must build and pass its own
      ``safe_to_state`` set the same way.
    - A bank account number stated with no account-ish word anywhere within
      _PII_ACCOUNT_ANCHOR_WINDOW characters is not caught (see
      _pii_account_matches).
    - _pii_account_matches's anchor-vs-disqualifier tie-break is nearest-wins
      (see there): on an exact distance tie the anchor wins (redacts), and a
      disqualifier word landing strictly closer to the digit run than any
      anchor word can still suppress a genuine leak in an unusual phrasing.
      This is rare in practice -- real account-number sentences almost
      always place the anchor word immediately adjacent to the number.
    - NOT limited to ASCII-digit numerals as previously (incorrectly)
      documented here: Python's regex ``\\d`` matches any Unicode decimal
      digit, Devanagari included. The ACCOUNT path (_PII_DIGIT_RUN_PATTERN,
      a bounded bare-digit-run regex) DOES match an anchor-adjacent
      Devanagari-digit run (e.g. "बैंक account number है ९८७६५४३२१०१२") like
      any other digit run. The MOBILE pattern is unaffected in practice: its
      leading character class is the literal ASCII range ``[6-9]``, which
      does not match Devanagari digit codepoints, so a fully-Devanagari
      10-digit run is never recognized as a mobile number. Email is
      unaffected (no digit matching involved in the local/domain shape
      check beyond what `[A-Za-z0-9...]` already restricts to ASCII). None
      of this was verified/tested before this pass, so treat the account-
      path behavior as incidental rather than a designed guarantee -- chat
      is text-in/text-out with no numeral-normalization pass, unlike voice,
      so a model rendering digits in Devanagari at all is not expected in
      practice, tested or not.
    - A 10-digit run starting 6-9 immediately followed by a currency word
      ("9500000000 rupees") is now classified as a mobile number rather than
      an amount, even though it could in principle be a genuine (very large)
      rupee amount -- see _find_pii_hits's own docstring (review Fix 4) for
      the full reasoning; this is an accepted, deliberate tradeoff, not an
      oversight.
    - The guard cannot tell WHOSE account a digit run belongs to from the
      text alone -- correctness of safe_to_state depends entirely on the
      caller supplying the right allow-set for the right scope. A caller
      that widened it to "any tool's output this turn" (instead of one
      specific tool's own result) would wrongly exempt a customer's own
      leaked bank account number just because some other tool happened to
      return it in the same turn.
    """
    text = response.response_text or ""
    safe_digits = {d for v in (safe_to_state or ()) if (d := _normalize_digits(v))}

    # Exemption is restricted to account/mobile hits of a realistic minimum
    # length (_PII_ACCOUNT_MIN_DIGITS, reused rather than a new magic
    # number) -- never "email". _walk_digit_bearing_values (in chatbot.py)
    # deliberately harvests EVERY digit-bearing leaf value from a CRM
    # payload, by design, so safe_digits can contain a short, incidental
    # value (a 3-digit SLA-hours field, a deposit limit) that is not itself
    # an account/mobile number. Without the length floor, a short
    # safe_digits entry could coincidentally match inside an EMAIL hit's
    # local part (e.g. safe_to_state={"100"} exempting
    # "support100@operator.com") purely by digit coincidence -- an email
    # address should never be exempt via this mechanism at all. See
    # _pii_hit_is_exempt, shared between response_text and followups below.
    hits = _find_pii_hits(text) if text else []
    redacted_text, body_types_found = (
        _redact_pii_hits(text, hits, safe_digits) if hits else (text, set())
    )

    # Review Fix 2: suggested_followups get the exact same scan. A followup
    # that leaks PII is DROPPED from the list entirely rather than redacted
    # in place -- see this function's own docstring for why a placeholder
    # inside a one-tap suggested reply is worse than just not offering it.
    kept_followups: list[str] = []
    followup_types_found: set[str] = set()
    followups_dropped = 0
    for followup in response.suggested_followups:
        f_hits = _find_pii_hits(followup)
        leaked_types = {
            typ for start, end, typ in f_hits
            if not _pii_hit_is_exempt(followup[start:end], typ, safe_digits)
        }
        if leaked_types:
            followups_dropped += 1
            followup_types_found |= leaked_types
        else:
            kept_followups.append(followup)

    types_found = body_types_found | followup_types_found
    if not types_found:
        # Nothing survived to report: either there were no hits anywhere
        # (response_text or followups), or every hit that did exist was
        # exempted via safe_to_state. Return the original response
        # untouched: no rebuild, no confidence downgrade, no error log,
        # since nothing was actually redacted or dropped.
        return response

    log_extra: dict[str, object] = {"pii_types": sorted(types_found)}
    if followups_dropped:
        log_extra["followups_dropped"] = followups_dropped
    log.error(
        "pii guard: reply contained a customer PII pattern; span(s) redacted"
        + (" and suggested_followups dropped" if followups_dropped else ""),
        extra=log_extra,
    )
    return ChatBotResponse(
        response_text=redacted_text,
        language=response.language,
        sources_used=list(response.sources_used),
        confidence="low",
        action=response.action,
        suggested_followups=kept_followups,
        raw=dict(response.raw),
        parse_error=response.parse_error,
    )
