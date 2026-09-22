"""Document parsing + chunking for RAG ingestion.

Parsers map ``(filename, bytes)`` to plain text. They're dispatched by file
extension; unknown extensions fall through to a UTF-8 text decode so we
don't blow up on stray inputs.

Chunkers split a parsed document into overlapping text windows. The default
``RecursiveChunker`` walks down a hierarchy of separators (``\\n\\n`` →
``\\n`` → ``. `` → ``? `` → ``! `` → ``।`` → space) so chunks land on
natural boundaries when possible. ``FixedChunker`` is a simpler fallback
for benchmarking.

Both chunkers operate on character counts. ``chunk_size`` and ``overlap``
in PRD config are nominal "tokens"; we treat 1 token ≈ 4 chars (a coarse but
predictable approximation that doesn't pull in a tokenizer dependency).
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from src.utils.logging import debug_event

log = logging.getLogger(__name__)

_TOKEN_TO_CHAR = 4  # rough rule of thumb; we don't ship a tokenizer here

# Cap on per-page "pdf_page_extract_failed" DEBUG events for a single
# document. A pathological PDF can fail on every page; LokiPushHandler's
# queue is bounded (src/utils/logging.py) and drops silently on overflow, so
# an uncapped per-page event stream from one bad document can evict unrelated
# DEBUG lines from an active debug session. The aggregate
# "ingestion pdf_pages_blank" event below still carries the true total, so
# nothing is lost diagnostically by capping the per-page ones.
_MAX_PDF_PAGE_FAILURE_EVENTS = 20


# --- Parsers --------------------------------------------------------------


class IDocumentParser(Protocol):
    extensions: tuple[str, ...]

    def parse(self, filename: str, data: bytes) -> str: ...


class MarkdownParser:
    extensions = (".md", ".markdown", ".txt")

    def parse(self, filename: str, data: bytes) -> str:
        return data.decode("utf-8", errors="replace")


class PDFParser:
    extensions = (".pdf",)

    def parse(self, filename: str, data: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("pypdf is required for PDF parsing") from e
        reader = PdfReader(io.BytesIO(data))
        pages = []
        page_failures_logged = 0
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:
                # Swallowed by design (one bad page must not fail the whole
                # document) -- but silent otherwise, so a page that never
                # yields text is indistinguishable from a page that was
                # genuinely blank unless this is logged. Capped at
                # _MAX_PDF_PAGE_FAILURE_EVENTS; the "ingestion pdf_pages_blank"
                # aggregate below carries the full count regardless.
                if page_failures_logged < _MAX_PDF_PAGE_FAILURE_EVENTS:
                    debug_event(log, "ingestion pdf_page_extract_failed",
                                document_filename=filename, page=i, error=str(exc))
                    page_failures_logged += 1
                pages.append("")
        kept = [p for p in pages if p.strip()]
        if len(kept) != len(pages) and log.isEnabledFor(logging.DEBUG):
            debug_event(log, "ingestion pdf_pages_blank", document_filename=filename,
                        total_pages=len(pages), kept_pages=len(kept),
                        blank_pages=[i for i, p in enumerate(pages) if not p.strip()])
        return "\n\n".join(kept)


class DOCXParser:
    extensions = (".docx",)

    def parse(self, filename: str, data: bytes) -> str:
        try:
            import docx
        except ImportError as e:
            raise RuntimeError("python-docx is required for DOCX parsing") from e
        doc = docx.Document(io.BytesIO(data))
        paragraphs = doc.paragraphs
        kept = [p.text for p in paragraphs if p.text.strip()]
        debug_event(log, "ingestion docx_parsed", document_filename=filename,
                    total_paragraphs=len(paragraphs), kept_paragraphs=len(kept))
        return "\n\n".join(kept)


_DEFAULT_PARSERS: list[IDocumentParser] = [MarkdownParser(), PDFParser(), DOCXParser()]


def parse_document(filename: str, data: bytes, parsers: Optional[list[IDocumentParser]] = None) -> str:
    """Dispatch to a parser by extension. Falls back to UTF-8 text."""
    parsers = parsers or _DEFAULT_PARSERS
    ext = Path(filename).suffix.lower()
    for parser in parsers:
        if ext in parser.extensions:
            text = parser.parse(filename, data)
            debug_event(log, "ingestion parse_document", document_filename=filename, ext=ext,
                        parser=type(parser).__name__, input_bytes=len(data),
                        output_chars=len(text))
            return text
    # Fallback: best-effort utf-8 decode.
    text = data.decode("utf-8", errors="replace")
    debug_event(log, "ingestion parse_document", document_filename=filename, ext=ext,
                parser="utf8_fallback", input_bytes=len(data), output_chars=len(text))
    return text


# --- Chunkers ------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    index: int  # 0-based position in the source document
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkConfig:
    chunk_size: int = 500          # tokens
    chunk_overlap: int = 100       # tokens
    strategy: str = "recursive"    # recursive | fixed

    @property
    def chunk_chars(self) -> int:
        return self.chunk_size * _TOKEN_TO_CHAR

    @property
    def overlap_chars(self) -> int:
        return self.chunk_overlap * _TOKEN_TO_CHAR


class FixedChunker:
    """Cuts at fixed character intervals with overlap. Predictable but
    splits mid-sentence."""

    def __init__(self, config: ChunkConfig) -> None:
        self._cfg = config

    def chunk(self, text: str, base_metadata: Optional[dict] = None) -> list[Chunk]:
        if not text:
            return []
        size = self._cfg.chunk_chars
        overlap = self._cfg.overlap_chars
        if size <= 0:
            raise ValueError("chunk_size must be > 0")
        step = max(1, size - overlap)

        out: list[Chunk] = []
        i = 0
        idx = 0
        while i < len(text):
            piece = text[i : i + size].strip()
            if piece:
                out.append(Chunk(
                    text=piece,
                    index=idx,
                    start_char=i,
                    end_char=min(i + size, len(text)),
                    metadata=dict(base_metadata or {}),
                ))
                idx += 1
            i += step
        if log.isEnabledFor(logging.DEBUG):
            debug_event(log, "ingestion chunk fixed", input_chars=len(text), chunk_chars=size,
                        overlap_chars=overlap, chunk_count=len(out),
                        sizes=[len(c.text) for c in out])
        return out


_RECURSIVE_SEPARATORS: list[str] = ["\n\n", "\n", "। ", ". ", "? ", "! ", "; ", " "]


class RecursiveChunker:
    """Splits on a hierarchy of separators, falling through when fragments
    are still too large. Prefers natural boundaries (paragraph > sentence
    > word). Adds character overlap between successive chunks."""

    def __init__(
        self,
        config: ChunkConfig,
        separators: Optional[list[str]] = None,
    ) -> None:
        self._cfg = config
        self._separators = separators or _RECURSIVE_SEPARATORS

    def chunk(self, text: str, base_metadata: Optional[dict] = None) -> list[Chunk]:
        if not text:
            return []
        target = self._cfg.chunk_chars
        overlap = self._cfg.overlap_chars
        meta = dict(base_metadata or {})

        pieces = self._recursive_split(text, target)
        # Merge adjacent pieces while we're still under target, then add overlap.
        merged: list[str] = []
        cur = ""
        for p in pieces:
            if not cur:
                cur = p
                continue
            if len(cur) + len(p) + 1 <= target:
                cur = cur + " " + p
            else:
                merged.append(cur)
                cur = p
        if cur:
            merged.append(cur)

        # Add character-level overlap by prepending the tail of the previous chunk.
        out: list[Chunk] = []
        cursor = 0
        for i, m in enumerate(merged):
            text_chunk = m
            if i > 0 and overlap > 0:
                tail = merged[i - 1][-overlap:]
                if tail and not text_chunk.startswith(tail):
                    text_chunk = tail + " " + text_chunk
            start = cursor
            end = cursor + len(m)
            out.append(Chunk(
                text=text_chunk.strip(),
                index=i,
                start_char=start,
                end_char=end,
                metadata=dict(meta),
            ))
            cursor = end
        if log.isEnabledFor(logging.DEBUG):
            debug_event(log, "ingestion chunk recursive", input_chars=len(text),
                        target_chars=target, overlap_chars=overlap, chunk_count=len(out),
                        sizes=[len(c.text) for c in out])
        return out

    def _recursive_split(self, text: str, target: int) -> list[str]:
        if len(text) <= target:
            return [text]
        for sep in self._separators:
            if sep in text:
                parts = _split_keeping_boundary(text, sep)
                # If splitting on this separator made any piece smaller, recurse
                # on each over-target piece and flatten.
                out: list[str] = []
                for p in parts:
                    if len(p) <= target:
                        out.append(p)
                    else:
                        out.extend(self._recursive_split(p, target))
                return [p for p in out if p.strip()]
        # No separator helps — hard cut every ``target`` chars.
        return [text[i : i + target] for i in range(0, len(text), target)]


def _split_keeping_boundary(text: str, sep: str) -> list[str]:
    """Split on ``sep`` but reattach the separator to the left piece so we
    don't lose it (so ``"A. B."`` -> ``["A.", "B."]``)."""
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    out: list[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append((p + sep).strip())
        else:
            if p.strip():
                out.append(p.strip())
    return [p for p in out if p]


# --- Top-level convenience -----------------------------------------------


def get_chunker(config: ChunkConfig) -> Callable[[str, Optional[dict]], list[Chunk]]:
    if config.strategy == "fixed":
        chunker = FixedChunker(config)
    elif config.strategy == "recursive":
        chunker = RecursiveChunker(config)
    else:
        raise ValueError(f"unknown chunking strategy: {config.strategy}")
    return chunker.chunk


def detect_language(text: str) -> Optional[str]:
    """Cheap heuristic — Devanagari -> hi, otherwise None.

    Real language ID belongs in Phase 5+; this is enough to tag chunks."""
    if not text:
        return None
    if re.search(r"[ऀ-ॿ]", text):
        return "hi"
    return None
