#!/usr/bin/env python3
"""Standalone verifier for data/eval/rag_eval.jsonl.

Not part of the benchmark package (scripts/run_benchmark.py) — this is a
content-integrity check for the eval dataset itself. It loads every record,
whitespace-normalises each expected span and each named KB file's contents,
and asserts the span occurs verbatim (modulo whitespace) in at least one of
the record's expected_files.

That whole-file check alone is not enough: a span can legitimately appear
somewhere in the file while still being useless as a retrieval label, if it
lands in a chunk of the file that no sensible query would ever retrieve (for
example, a negative aside chunked apart from the definition it disambiguates
against). So this script also runs the REAL chunker
(``src.rag.ingestion.RecursiveChunker`` via ``get_chunker(ChunkConfig())``,
matching the ``config/default.yaml`` chunking defaults) over every referenced
file and asserts each span lands wholly inside at least one emitted chunk.
A span that fails this check would silently score a correct retrieval as a
miss, so any failure is reported and the script exits non-zero.

The chunk-level check can't tell whether a span is the RIGHT passage for the
query (that requires reading the file), but it can flag the risky shape: a
span that matches only one chunk of a multi-chunk file is one accidental
chunk-boundary change away from becoming unretrievable, and is exactly the
shape defect that has previously slipped past review. Those are reported as
WARNINGs for manual audit, not failures.

Usage:
    python3 scripts/verify_rag_eval.py [path/to/rag_eval.jsonl]
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "rag_eval.jsonl"
KB_ROOT = REPO_ROOT / "data" / "kb"

# Defensive: make sure the repo root is importable as `src.*` even if this
# script isn't run from an environment with the package installed editable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.ingestion import ChunkConfig, get_chunker  # noqa: E402

REQUIRED_FIELDS = {
    "id",
    "query",
    "lang",
    "intent",
    "tier",
    "expected_files",
    "expected_spans",
    "unanswerable",
}

TIER_VALUES = {"pack", "module", "layout"}
# Most restrictive first: a record whose expected_files span multiple tiers
# should be labelled with the most restrictive one it depends on.
TIER_RANK = {"layout": 0, "module": 1, "pack": 2}


def normalise_ws(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def find_kb_files() -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Map basename -> path for every markdown file under data/kb/, plus the
    subset of basenames that are ambiguous (appear under more than one
    directory).

    The resolved map still picks a first candidate for any ambiguous name
    (so a caller that only cares about non-ambiguous files keeps working
    unmodified), but callers that verify dataset records must additionally
    hard-fail any record whose expected_files references an ambiguous
    basename -- see the caller in ``main()``. Silently picking "the first
    match" for a record would verify that record against whichever file
    happened to sort first, not necessarily the one the record's author
    meant.
    """
    by_basename: dict[str, list[Path]] = {}
    for path in KB_ROOT.rglob("*.md"):
        by_basename.setdefault(path.name, []).append(path)

    resolved: dict[str, Path] = {}
    ambiguous: dict[str, list[Path]] = {}
    for name, paths in by_basename.items():
        resolved[name] = paths[0]
        if len(paths) > 1:
            ambiguous[name] = paths
    return resolved, ambiguous


def infer_tier(path: Path) -> str | None:
    """Classify a KB file path by which tier of the corpus it lives in.

    Only ``data/kb/packs/**`` is auto-seeded into every tenant at startup;
    ``data/kb/modules/**`` is ingested only when a CRM opts into that module,
    and ``data/kb/layouts/**`` only when a tenant is on that specific
    frontend layout — a tenant has exactly one layout. Returns None if the
    path isn't under any recognised tier directory.
    """
    parts = path.parts
    if "packs" in parts:
        return "pack"
    if "modules" in parts:
        return "module"
    if "layouts" in parts:
        return "layout"
    return None


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
            missing = REQUIRED_FIELDS - obj.keys()
            if missing:
                raise SystemExit(f"{path}:{line_no}: record missing fields {missing}")
            records.append(obj)
    return records


def main() -> int:
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    records = load_dataset(dataset_path)
    kb_files, ambiguous_kb_files = find_kb_files()
    chunker = get_chunker(ChunkConfig())

    # Cache normalised file contents.
    normalised_cache: dict[str, str] = {}

    def normalised_contents(basename: str) -> str | None:
        if basename not in kb_files:
            return None
        if basename not in normalised_cache:
            normalised_cache[basename] = normalise_ws(
                kb_files[basename].read_text(encoding="utf-8")
            )
        return normalised_cache[basename]

    # Cache chunks (from the real chunker, ChunkConfig() defaults) per file.
    chunk_cache: dict[str, list] = {}

    def chunks_for(basename: str) -> list | None:
        if basename not in kb_files:
            return None
        if basename not in chunk_cache:
            text = kb_files[basename].read_text(encoding="utf-8")
            chunk_cache[basename] = chunker(text, {})
        return chunk_cache[basename]

    failures: list[str] = []
    warnings: list[str] = []
    chunk_report: list[str] = []
    ids_seen: Counter[str] = Counter()

    lang_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    unanswerable_count = 0

    for rec in records:
        rid = rec["id"]
        ids_seen[rid] += 1
        lang_counts[rec["lang"]] += 1
        intent_counts[rec["intent"]] += 1
        topic_prefix = rid.split("-")[0] if "-" in rid else rid
        topic_counts[topic_prefix] += 1
        if rec["unanswerable"]:
            unanswerable_count += 1

        tier = rec["tier"]
        if tier not in TIER_VALUES:
            failures.append(f"{rid}: tier {tier!r} is not one of {sorted(TIER_VALUES)}")
        else:
            tier_counts[tier] += 1

        expected_files = rec["expected_files"]
        expected_spans = rec["expected_spans"]

        if rec["unanswerable"]:
            if expected_files or expected_spans:
                failures.append(
                    f"{rid}: unanswerable=true but expected_files/expected_spans "
                    f"are non-empty ({expected_files!r}, {expected_spans!r})"
                )
            if tier != "pack":
                failures.append(
                    f"{rid}: unanswerable=true but tier is {tier!r}, expected 'pack' "
                    "(unanswerable records carry no corpus dependency, by convention "
                    "labelled 'pack')"
                )
            continue

        if not expected_files:
            failures.append(f"{rid}: unanswerable=false but expected_files is empty")
            continue
        if not expected_spans:
            failures.append(f"{rid}: unanswerable=false but expected_spans is empty")
            continue

        # Verify every named file actually exists in the corpus.
        missing_files = [f for f in expected_files if f not in kb_files]
        for mf in missing_files:
            failures.append(f"{rid}: expected_files references unknown file {mf!r}")

        # A basename that resolves to more than one file under data/kb/ is
        # not something this verifier (or the retriever, which also keys by
        # filename -- see score_retrieval_spans's file_hit) can resolve
        # uniquely. Silently picking "the first match" would verify this
        # record's span against whichever file happens to sort first, not
        # necessarily the one the record's author meant -- a hard failure,
        # not a warning, because a dataset must not reference a file the
        # verifier can't uniquely resolve.
        ambiguous_refs = [f for f in expected_files if f in ambiguous_kb_files]
        for af in ambiguous_refs:
            paths = [str(p.relative_to(REPO_ROOT)) for p in ambiguous_kb_files[af]]
            failures.append(
                f"{rid}: expected_files references {af!r}, which is an "
                f"AMBIGUOUS basename across the KB tree ({paths}) -- rename "
                "one of the files or make expected_files unambiguous."
            )

        # Cross-check the declared tier against where expected_files actually
        # live. If a record depends on files in more than one tier, the most
        # restrictive tier is the correct label (layout > module > pack).
        actual_tiers = [
            infer_tier(kb_files[f]) for f in expected_files if f in kb_files
        ]
        actual_tiers = [t for t in actual_tiers if t is not None]
        if actual_tiers:
            expected_tier = min(actual_tiers, key=lambda t: TIER_RANK[t])
            if tier in TIER_VALUES and tier != expected_tier:
                failures.append(
                    f"{rid}: tier is {tier!r} but expected_files "
                    f"{expected_files!r} resolve to tier {expected_tier!r} "
                    f"(observed tiers: {sorted(set(actual_tiers))})"
                )

        available_texts = {
            f: normalised_contents(f) for f in expected_files if f in kb_files
        }

        for span in expected_spans:
            norm_span = normalise_ws(span)
            found_in = [
                f for f, text in available_texts.items() if text and norm_span in text
            ]
            if not found_in:
                failures.append(
                    f"{rid}: span not found in {expected_files} -> {span!r}"
                )
                continue

            # The span is somewhere in the file(s) verbatim. Now check it
            # actually lands wholly inside at least one chunk the real
            # chunker would emit — this is the check that whole-file
            # matching alone cannot make.
            per_file_chunk_hits: dict[str, list[int]] = {}
            for f in found_in:
                chunks = chunks_for(f) or []
                hits = [
                    c.index for c in chunks if norm_span in normalise_ws(c.text)
                ]
                per_file_chunk_hits[f] = hits

            any_hits = any(hits for hits in per_file_chunk_hits.values())
            if not any_hits:
                failures.append(
                    f"{rid}: span matched file text but landed in NO chunk emitted "
                    f"by the real chunker (ChunkConfig defaults) for "
                    f"{list(per_file_chunk_hits)} -> {span!r}"
                )
                continue

            for f, hits in per_file_chunk_hits.items():
                total_chunks = len(chunks_for(f) or [])
                chunk_report.append(
                    f"{rid}: {f} span -> chunk(s) {hits} (of {total_chunks} total)"
                )
                if hits and total_chunks > 1 and len(hits) == 1:
                    # A file split into 3+ chunks has more than one merge
                    # boundary running through it, which compounds the odds
                    # that a re-chunk moves the span away from the chunk a
                    # retriever would actually surface — flag those louder.
                    severity = "HIGH RISK" if total_chunks >= 3 else "risky shape"
                    warnings.append(
                        f"{rid}: [{severity}] span in {f} matches only chunk "
                        f"{hits[0]} of {total_chunks} — one chunk-boundary change "
                        "away from becoming unretrievable; verify this is the "
                        "passage that actually answers the query"
                    )

    dup_ids = [rid for rid, count in ids_seen.items() if count > 1]
    for rid in dup_ids:
        failures.append(f"duplicate id: {rid!r} appears {ids_seen[rid]} times")

    # ---- Report ----
    print("=" * 72)
    print("RAG eval dataset verification")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Total records: {len(records)}")
    print()
    print("By language:")
    for lang, count in sorted(lang_counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / len(records)
        print(f"  {lang:10s} {count:4d}  ({pct:.1f}%)")
    print()
    print(f"By intent ({len(intent_counts)} distinct):")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent:40s} {count}")
    print()
    print("By topic prefix (id slug):")
    for topic, count in sorted(topic_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {topic:10s} {count}")
    print()
    print("By tier:")
    for t, count in sorted(tier_counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / len(records)
        print(f"  {t:10s} {count:4d}  ({pct:.1f}%)")
    print()
    print(f"Unanswerable records: {unanswerable_count}")
    followups = sum(1 for i in intent_counts if i.endswith("-followup"))
    followup_records = sum(c for i, c in intent_counts.items() if i.endswith("-followup"))
    print(f"Follow-up intents: {followups} distinct ({followup_records} records)")
    followup_lang_counts = Counter(
        r["lang"] for r in records if r["intent"].endswith("-followup")
    )
    print(f"  by language: {dict(followup_lang_counts)}")
    tri_lingual = [
        i
        for i in intent_counts
        if not i.endswith("-followup")
        and len(
            {
                r["lang"]
                for r in records
                if r["intent"] == i
            }
        )
        == 3
    ]
    print(f"Intent groups covered in all 3 languages: {len(tri_lingual)}")
    for i in sorted(tri_lingual):
        print(f"  {i}")
    print()

    print("=" * 72)
    print("Per-record chunk placement (real chunker, ChunkConfig defaults):")
    print("=" * 72)
    for line in chunk_report:
        print(f"  {line}")
    print()

    if warnings:
        high_risk = sum(1 for w in warnings if "[HIGH RISK]" in w)
        print("=" * 72)
        print(
            f"WARNINGS ({len(warnings)}) — single-chunk-of-multi-chunk-file risk "
            f"({high_risk} of which are [HIGH RISK]: the file splits into 3+ "
            "chunks, so multiple merge boundaries run through it):"
        )
        print("=" * 72)
        for w in warnings:
            print(f"  - {w}")
        print()

    if failures:
        print("=" * 72)
        print(f"FAILURES ({len(failures)}):")
        print("=" * 72)
        for f in failures:
            print(f"  - {f}")
        print()
        print(f"RESULT: FAIL ({len(failures)} problems)")
        return 1

    print(
        "RESULT: PASS — every expected_span matched verbatim (whitespace-normalised) "
        "and lands inside at least one real chunk"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
