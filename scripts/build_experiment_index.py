#!/usr/bin/env python3
"""Build a parallel (non-live) pgvector index to A/B retrieval hypotheses.

Ingests the same bundled KB source files the live seeder uses
(``data/kb/packs/betting-default/**`` by default — see
``src.main._seed_crm_kb``) into a SEPARATE ``crm_id`` scope, with
configurable chunking and an optional Gemini embedding ``task_type``, so the
result can be benchmarked against the live index without ever touching it.

Two independent knobs, both opt-in:
  --chunk-size / --chunk-overlap   nominal tokens, ~4 chars/token (matches
                                    ``src.rag.ingestion.ChunkConfig``)
  --document-task-type             e.g. "RETRIEVAL_DOCUMENT" (see
                                    ``src.rag.embeddings.GeminiEmbedder``)

They are tested TOGETHER in one parallel index because changing either one
requires a full re-index: a different chunk size changes chunk ids and
content outright, and a different document task type must be paired with
the SAME task type at query time (see ``--document-task-type``'s docstring
in ``src.rag.embeddings``) — querying a mixed-encoding index measures
nothing meaningful.

Safety
------
- Refuses outright if ``--crm-id betstudio`` — that is the live scope
  serving staging traffic.
- Refuses to write if the target scope already has chunks, unless
  ``--overwrite`` is passed (which first deletes the scope's existing
  chunks, then re-ingests — needed because a different chunk size produces
  different chunk ids, so stale chunks from a previous run would otherwise
  never be cleaned up by a plain upsert).
- Dry-run by default. Pass ``--apply`` to actually write. The dry-run path
  never opens a database connection and never calls the embedding API — it
  only parses + chunks the source files (pure local IO) and reports what an
  ``--apply`` run would do.

Example (dry run):
  python scripts/build_experiment_index.py --crm-id betstudio-exp-small \\
      --chunk-size 150 --chunk-overlap 30 --document-task-type RETRIEVAL_DOCUMENT

Example (write it):
  python scripts/build_experiment_index.py --crm-id betstudio-exp-small \\
      --chunk-size 150 --chunk-overlap 30 --document-task-type RETRIEVAL_DOCUMENT \\
      --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.ingestion import Chunk, ChunkConfig, detect_language, get_chunker, parse_document  # noqa: E402

LIVE_CRM_ID = "betstudio"
DEFAULT_KB_DIR = REPO_ROOT / "data" / "kb" / "packs" / "betting-default"
# Same extension/hidden-file filter as src.main._seed_crm_kb.
_KB_EXTS = {".md", ".txt", ".pdf", ".docx", ".csv"}


class GuardError(RuntimeError):
    """Raised when a safety guard refuses to proceed."""


def guard_target_crm_id(crm_id: str) -> None:
    """Refuse outright if the target is the live scope."""
    if crm_id == LIVE_CRM_ID:
        raise GuardError(
            f"refusing to target crm_id={crm_id!r} -- that is the LIVE index "
            f"serving staging traffic. Pick a different --crm-id (e.g. "
            f"'{crm_id}-exp')."
        )


def guard_overwrite(existing_count: int, overwrite: bool, crm_id: str) -> None:
    """Refuse to write into a non-empty target scope without --overwrite."""
    if existing_count > 0 and not overwrite:
        raise GuardError(
            f"target crm_id={crm_id!r} already has {existing_count} chunk(s). "
            f"Pass --overwrite to replace them, or pick an empty --crm-id."
        )


def guard_no_live_id_collision(doc_ids: list[str]) -> None:
    """Defense-in-depth beyond ``guard_target_crm_id``.

    ``voicebot.knowledge_chunks.id`` is a single PRIMARY KEY shared across
    every ``crm_id``/``tenant_id`` scope (see docs/pgvector_setup.sql), and
    ``PGVectorAdapter.index()`` upserts on that id with ``ON CONFLICT ...
    SET ... crm_id = EXCLUDED.crm_id`` -- an id collision doesn't just
    overwrite content, it can reassign a row's scope outright. Doc ids are
    built as ``crm_kb_{crm_id}_{file_stem}``, and because both ``crm_id`` and
    the stem can themselves contain underscores, two different (crm_id,
    stem) pairs can in principle concatenate to the same string (e.g.
    crm_id="betstudio_extra" + stem="x" == crm_id="betstudio" +
    stem="extra_x"). ``guard_target_crm_id`` already blocks ``crm_id ==
    "betstudio"`` outright, but not a crm_id that merely starts with
    "betstudio_" combined with an unlucky stem. Belt-and-suspenders: refuse
    if any planned id would land in the live seeder's exact id namespace.
    """
    live_prefix = f"crm_kb_{LIVE_CRM_ID}_"
    colliding = [d for d in doc_ids if d.startswith(live_prefix)]
    if colliding:
        raise GuardError(
            f"{len(colliding)} planned document id(s) fall inside the LIVE "
            f"index's id namespace ({live_prefix!r}...), e.g. {colliding[0]!r} "
            f"-- refusing to risk overwriting a live chunk. Pick a --crm-id "
            f"that does not start with {LIVE_CRM_ID!r}."
        )


@dataclass
class PlannedFile:
    """One source file's chunking plan -- no embeddings yet."""

    path: Path
    doc_id: str
    filename: str
    language: Optional[str]
    chunks: list[Chunk] = field(default_factory=list)


def discover_files(kb_dir: Path) -> list[Path]:
    """Same recursive, extension-filtered, hidden-file-excluding sweep as
    ``src.main._seed_crm_kb``."""
    return sorted(
        p for p in kb_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _KB_EXTS and not p.name.startswith(".")
    )


def plan_index(kb_dir: Path, crm_id: str, chunk_config: ChunkConfig) -> list[PlannedFile]:
    """Parse + chunk every source file under ``kb_dir``. Pure local IO --
    no network calls, no database, no embedding. Doc id / chunk id / metadata
    conventions match ``src.main._seed_crm_kb`` exactly (with ``crm_id``
    swapped for the target experimental scope) so the eval harness's
    ``file_hit`` (which keys off the ``filename`` basename in metadata)
    works unchanged against this index."""
    chunker = get_chunker(chunk_config)
    planned: list[PlannedFile] = []
    for f in discover_files(kb_dir):
        text = parse_document(f.name, f.read_bytes())
        if not text.strip():
            continue
        language = detect_language(text)
        doc_id = f"crm_kb_{crm_id}_{f.stem}"
        raw_chunks = chunker(text, {
            "filename": f.name, "document_id": doc_id, "language": language,
        })
        if not raw_chunks:
            continue
        planned.append(PlannedFile(
            path=f, doc_id=doc_id, filename=f.name, language=language, chunks=raw_chunks,
        ))
    return planned


def to_documents(planned: PlannedFile) -> list:
    """Build ``Document`` objects (no embedding set yet) matching
    ``src.main._seed_crm_kb``'s id/metadata shape:
    ``{doc_id}::chunk-{index}`` with ``section``/``page`` set to the chunk
    index, layered over whatever metadata the chunker propagated
    (``filename``, ``document_id``, ``language``)."""
    from src.interfaces.vector_store import Document

    return [
        Document(
            id=f"{planned.doc_id}::chunk-{c.index}",
            content=c.text,
            metadata={**c.metadata, "section": c.index, "page": c.index},
        )
        for c in planned.chunks
    ]


def resolve_database_url() -> str:
    """src.config settings first (reads .env directly via pydantic-settings,
    independent of whether .env is exported into the process environment),
    then a plain os.environ fallback.

    Deliberately only ``settings.secrets.DATABASE_URL`` -- NOT
    ``settings.database.url`` -- matching ``scripts/run_benchmark.py``'s
    ``_build_retriever_for_cli`` convention. ``settings.database.url`` has a
    non-``None`` YAML default (a local ``postgresql+asyncpg://vox:vox@...``
    URL, see ``DatabaseConfig`` in ``src/config.py``), so falling back to it
    would silently mask a genuinely unset ``DATABASE_URL`` behind a
    plausible-looking connection string to a database that isn't the one the
    operator meant, instead of surfacing a clear error.
    """
    try:
        from src.config import get_settings

        settings = get_settings()
        if settings.secrets.DATABASE_URL:
            return settings.secrets.DATABASE_URL
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "")


def resolve_gemini_api_key() -> str:
    try:
        from src.config import get_settings

        settings = get_settings()
        if settings.secrets.GEMINI_API_KEY:
            return settings.secrets.GEMINI_API_KEY
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def _print_plan(
    args: argparse.Namespace,
    kb_dir: Path,
    chunk_config: ChunkConfig,
    planned: list[PlannedFile],
) -> None:
    total_chunks = sum(len(pf.chunks) for pf in planned)
    print(f"Target crm_id:        {args.crm_id!r}")
    print(f"KB source dir:        {kb_dir}")
    print(
        f"Chunk config:         chunk_size={args.chunk_size} tokens "
        f"(~{chunk_config.chunk_chars} chars), chunk_overlap={args.chunk_overlap} "
        f"tokens (~{chunk_config.overlap_chars} chars)"
    )
    print(
        "Document task type:   "
        + (args.document_task_type or "(none -- symmetric, same as the live index)")
    )
    print(f"Embedding dim:        {args.embedding_dim}")
    print(f"Overwrite existing:   {args.overwrite}")
    print()
    print(f"Files discovered:     {len(planned)}")
    for pf in planned:
        print(f"  {pf.filename:<45} {len(pf.chunks):>3} chunk(s)  lang={pf.language}")
    print()
    print(f"Planned chunks total:        {total_chunks}")
    print(f"Planned embedding API calls: {len(planned)} (one batched call per file)")


async def _apply(
    args: argparse.Namespace,
    planned: list[PlannedFile],
) -> int:
    from src.providers import get_vector_store
    from src.rag.embeddings import GeminiEmbedder

    database_url = resolve_database_url()
    if not database_url:
        raise GuardError(
            "no DATABASE_URL resolved (checked src.config settings, then the "
            "DATABASE_URL env var) -- cannot open a connection to write."
        )
    gemini_api_key = resolve_gemini_api_key()
    if not gemini_api_key:
        raise GuardError(
            "no GEMINI_API_KEY resolved (checked src.config settings, then "
            "the GEMINI_API_KEY env var) -- cannot call the embedding API."
        )

    vector_store = get_vector_store({
        "provider": "pgvector",
        "crm_id": args.crm_id,
        "embedding_dim": args.embedding_dim,
        "database_url": database_url,
    })

    existing_count = await vector_store.count()
    guard_overwrite(existing_count, args.overwrite, args.crm_id)
    if args.overwrite and existing_count > 0:
        existing_docs = await vector_store.list_documents(limit=1_000_000)
        stale_ids = [d.id for d in existing_docs]
        if stale_ids:
            deleted = await vector_store.delete(stale_ids)
            print(f"overwrite: deleted {deleted} pre-existing chunk(s) from crm_id={args.crm_id!r}")

    embedder = GeminiEmbedder(
        dim=args.embedding_dim,
        api_key=gemini_api_key,
        document_task_type=args.document_task_type,
    )

    total_indexed = 0
    for pf in planned:
        docs = to_documents(pf)
        texts = [d.content for d in docs]
        vectors = await asyncio.to_thread(embedder.embed_documents, texts)
        # Fail fast and loud: a short response from embed_documents would
        # otherwise silently truncate via zip() here, leaving the tail chunks
        # unembedded (caught later, confusingly, as a ValueError inside
        # PGVectorAdapter.index -- and only after earlier files in this loop
        # were already written).
        if len(vectors) != len(docs):
            raise RuntimeError(
                f"{pf.filename}: embed_documents returned {len(vectors)} "
                f"vector(s) for {len(docs)} chunk(s) -- refusing to write "
                f"partially-embedded documents."
            )
        for d, v in zip(docs, vectors):
            d.embedding = v
        n = await vector_store.index(docs)
        total_indexed += n
        print(f"  indexed {pf.filename}: {n} chunk(s)")

    print(f"\ndone: {total_indexed} chunk(s) written to crm_id={args.crm_id!r}")
    return total_indexed


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--crm-id", required=True,
        help="target experimental scope, e.g. betstudio-exp-small (must not be 'betstudio')",
    )
    ap.add_argument(
        "--kb-dir", default=str(DEFAULT_KB_DIR),
        help=f"source KB directory (default: {DEFAULT_KB_DIR})",
    )
    ap.add_argument(
        "--chunk-size", type=int, default=ChunkConfig().chunk_size,
        help="nominal tokens (~4 chars/token), matches ChunkConfig.chunk_size",
    )
    ap.add_argument(
        "--chunk-overlap", type=int, default=ChunkConfig().chunk_overlap,
        help="nominal tokens (~4 chars/token), matches ChunkConfig.chunk_overlap",
    )
    ap.add_argument(
        "--document-task-type", default=None,
        help='Gemini embed_content task_type for documents, e.g. "RETRIEVAL_DOCUMENT" '
             "(opt-in; omit to match the live index's symmetric, no-task-type encoding)",
    )
    ap.add_argument("--embedding-dim", type=int, default=384)
    ap.add_argument(
        "--overwrite", action="store_true",
        help="allow writing into a non-empty target scope (deletes its existing chunks first)",
    )
    ap.add_argument(
        "--apply", action="store_true",
        help="actually write (open a DB connection, call the embedding API). "
             "Default is dry-run: report the plan and exit.",
    )
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        guard_target_crm_id(args.crm_id)
    except GuardError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        print(f"error: --kb-dir not found or not a directory: {kb_dir}", file=sys.stderr)
        return 2

    chunk_config = ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    planned = plan_index(kb_dir, args.crm_id, chunk_config)
    if not planned:
        print(f"error: no chunkable files found under {kb_dir}", file=sys.stderr)
        return 1

    try:
        all_ids = [doc.id for pf in planned for doc in to_documents(pf)]
        guard_no_live_id_collision(all_ids)
    except GuardError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    _print_plan(args, kb_dir, chunk_config, planned)

    if not args.apply:
        print()
        print("DRY RUN -- no database connection opened, no embedding API calls made.")
        print("Re-run with --apply to write.")
        return 0

    try:
        asyncio.run(_apply(args, planned))
    except GuardError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
