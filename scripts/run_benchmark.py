"""CLI to execute one slice of the benchmark suite.

Designed for the deferred-API path (no live keys):
    python scripts/run_benchmark.py stt --dataset data/stt.jsonl --out data/stt_results.csv

Sub-commands:
    stt          run STT benchmark against a JSONL dataset
    tts          run TTS benchmark against a JSONL dataset
    latency      run latency matrix across N replays of one audio sample
    rag          run RAG benchmark against a JSONL dataset
    retrieval    run retrieval-only benchmark (no LLM/agent) against a JSONL dataset

The CLI deliberately wires fake provider clients when ``--mock`` is given so
the suite is exercisable end-to-end without API keys. With real keys present
in the environment, the existing provider factories pick them up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.benchmarks.datasets import (
    load_rag_dataset,
    load_stt_dataset,
    load_tts_dataset,
)
from src.benchmarks.export import (
    retrieval_csv_meta_path,
    write_latency_csv,
    write_rag_csv,
    write_retrieval_csv,
    write_stt_csv,
    write_tts_csv,
)
from src.benchmarks.rag_benchmark import run_retrieval_benchmark
from src.benchmarks.stt_benchmark import run_stt_benchmark
from src.benchmarks.tts_benchmark import run_tts_benchmark


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vox-agent benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stt = sub.add_parser("stt", help="Run STT benchmark")
    p_stt.add_argument("--dataset", required=True, type=Path)
    p_stt.add_argument("--out", required=True, type=Path)
    p_stt.add_argument("--provider", default="sarvam")
    p_stt.add_argument("--mock", action="store_true", help="Use a deterministic fake provider")

    p_tts = sub.add_parser("tts", help="Run TTS benchmark")
    p_tts.add_argument("--dataset", required=True, type=Path)
    p_tts.add_argument("--out", required=True, type=Path)
    p_tts.add_argument("--provider", default="sarvam")
    p_tts.add_argument("--mock", action="store_true")

    p_rag = sub.add_parser("rag", help="Run RAG benchmark")
    p_rag.add_argument("--dataset", required=True, type=Path)
    p_rag.add_argument("--out", required=True, type=Path)

    p_retrieval = sub.add_parser(
        "retrieval",
        help="Run retrieval-only benchmark (no LLM/agent) -- for tuning fusion weights",
    )
    p_retrieval.add_argument("--dataset", required=True, type=Path)
    p_retrieval.add_argument("--out", required=True, type=Path)
    p_retrieval.add_argument(
        "--top-k", type=int, default=None,
        help="override rag.retrieval.top_k from config for this run "
             "(defaults to the config value when omitted)")
    p_retrieval.add_argument(
        "--strategy", choices=["dense", "hybrid"], default=None,
        help="override rag.retrieval.strategy from config for this run. "
             "NOTE: when strategy is 'dense', --bm25-weight/--dense-weight are "
             "never read by HybridRetriever.search (src/rag/retriever.py) -- "
             "the CLI warns if you pass them together with a dense strategy.")
    p_retrieval.add_argument(
        "--bm25-weight", type=float, default=None,
        help="override rag.retrieval.bm25_weight from config for this run "
             "(no-op if the strategy in force is 'dense' -- see --strategy)")
    p_retrieval.add_argument(
        "--dense-weight", type=float, default=None,
        help="override rag.retrieval.dense_weight from config for this run "
             "(no-op if the strategy in force is 'dense' -- see --strategy)")
    p_retrieval.add_argument(
        "--similarity-threshold", type=float, default=None,
        help="override rag.retrieval.similarity_threshold from config for this run")
    p_retrieval.add_argument(
        "--fp-threshold", type=float, default=None,
        help="bar a RAW dense_score (never the fused score -- see "
             "src/benchmarks/rag_benchmark.py's is_false_positive docstring) "
             "must clear for a retrieval on an unanswerable sample to count "
             "as a false positive. No default: omitting this flag leaves "
             "false_positive_rate unset (reported as 'not computed') rather "
             "than silently applying a guessed cutoff that would be wrong "
             "for most embedders. Use the printed "
             "unanswerable_dense_score_mean/_max vs. answerable_dense_score_"
             "mean/_max to pick a bar empirically before setting this.")
    p_retrieval.add_argument(
        "--tier", default=None,
        help="score only samples whose dataset 'tier' field matches this value "
             "(e.g. 'pack') -- use when your index only holds a subset of the "
             "corpus tiers the dataset covers")
    p_retrieval.add_argument(
        "--max-chunks", type=int, default=2000,
        help="cap on chunks enumerated from the persistent store for BM25 "
             "hydration and unindexed-sample detection. A logged WARNING "
             "means this run returned exactly this many chunks, i.e. the "
             "corpus may be larger and got truncated by the cap")
    p_retrieval.add_argument(
        "--tenant-id", default=None,
        help="scope the retriever to one tenant's KB (pgvector only). "
             "Effectively MANDATORY along with --crm-id: with neither set, "
             "pgvector scopes to 'tenant_id IS NULL AND crm_id IS NULL' "
             "(src/providers/vector_store/pgvector_store.py) while every "
             "seeder writes rows under a real tenant_id or crm_id, so the "
             "query matches 0 chunks and the run is refused below.")
    p_retrieval.add_argument(
        "--crm-id", default=None,
        help="scope the retriever to one CRM's shared KB (pgvector only). "
             "See --tenant-id above -- one of the two is effectively required.")
    p_retrieval.add_argument(
        "--index-path", default=None,
        help="override the FAISS index_path from config (faiss provider only)")

    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.command == "stt":
        samples = load_stt_dataset(args.dataset)
        if args.mock:
            stt = _MockSTT()
        else:
            from src.config import load_settings
            from src.providers import get_stt_provider
            stt = get_stt_provider({"provider": args.provider, **load_settings().pipeline.stt.model_dump()})
        result = await run_stt_benchmark(args.provider, stt, samples)
        n = write_stt_csv(args.out, result)
        print(f"wrote {n} rows to {args.out}; wer_mean={result.overall.wer_mean:.4f}")
        return 0

    if args.command == "tts":
        samples = load_tts_dataset(args.dataset)
        if args.mock:
            tts = _MockTTS()
        else:
            from src.config import load_settings
            from src.providers import get_tts_provider
            tts = get_tts_provider({"provider": args.provider, **load_settings().pipeline.tts.model_dump()})
        result = await run_tts_benchmark(args.provider, tts, samples)
        n = write_tts_csv(args.out, result)
        print(f"wrote {n} rows to {args.out}; avg_chars_per_second={result.avg_chars_per_second:.2f}")
        return 0

    if args.command == "rag":
        # RAG path requires a wired retriever + agent. The CLI form is a
        # thin compatibility shim that emits a clear message — full RAG runs
        # currently happen through ``tests/integration/test_benchmark_e2e.py``
        # or the API (Phase 6+).
        print(
            "rag CLI requires app bootstrap; use the API or the integration test for now",
            file=sys.stderr,
        )
        return 2

    if args.command == "retrieval":
        samples = load_rag_dataset(args.dataset)
        try:
            retriever = _build_retriever_for_cli(args)
        except Exception as e:  # noqa: BLE001 -- surfaced to the operator, not re-raised
            print(f"retrieval CLI: could not build retriever: {e}", file=sys.stderr)
            return 2
        # _build_retriever_for_cli never ingests, so its BM25 arm starts
        # empty -- a --bm25-weight sweep over an empty sparse index would
        # silently degrade to dense-only for every sample and report
        # meaningless numbers with no error. Hydrate from the persistent
        # dense store and refuse to run if that comes back empty. Gated on
        # strategy == "hybrid": HybridRetriever.search never touches the BM25
        # arm in dense mode (src/rag/retriever.py), so hydrating it (or
        # refusing to run when it can't be hydrated) for a --strategy dense
        # run rejects a legitimate run for a reason that doesn't apply to it.
        if retriever.config.strategy == "dense" and (
            args.bm25_weight is not None or args.dense_weight is not None
        ):
            print(
                "retrieval CLI: WARNING -- strategy in force is 'dense'; "
                "--bm25-weight/--dense-weight are supplied but HybridRetriever."
                "search never reads either in dense mode (src/rag/retriever.py) "
                "-- every point in this sweep will return identical numbers.",
                file=sys.stderr,
            )
        if retriever.config.strategy == "hybrid":
            try:
                hydrated = await retriever.hydrate_sparse_from_persistent(max_chunks=args.max_chunks)
            except Exception as e:  # noqa: BLE001
                print(f"retrieval CLI: BM25 hydration failed: {e}", file=sys.stderr)
                return 2
            if hydrated == 0:
                print(
                    "retrieval CLI: BM25 hydration loaded 0 chunks from the persistent "
                    "store -- refusing to run a hybrid/bm25 benchmark against an empty "
                    "sparse index (results would silently read as dense-only). Check "
                    "--tenant-id/--crm-id/--index-path and that the store has been "
                    "ingested into.",
                    file=sys.stderr,
                )
                return 2
            print(f"retrieval CLI: hydrated BM25 index with {hydrated} chunks from the persistent store")
        # Use the retriever's effective top_k (YAML default unless --top-k
        # overrode it in _build_retriever_for_cli), not args.top_k directly
        # -- args.top_k is None whenever the flag was omitted.
        run_kwargs: dict[str, Any] = {
            "top_k": retriever.config.top_k, "tier_filter": args.tier, "max_chunks": args.max_chunks,
        }
        if args.fp_threshold is not None:
            run_kwargs["fp_threshold"] = args.fp_threshold
        try:
            result = await run_retrieval_benchmark([retriever], samples, **run_kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"retrieval CLI: benchmark run failed: {e}", file=sys.stderr)
            return 2
        n = write_retrieval_csv(args.out, result)
        meta_path = retrieval_csv_meta_path(args.out)
        print(
            f"wrote {n} rows to {args.out} (config/provenance in {meta_path}); "
            f"strategy={result.strategy} top_k={result.top_k} "
            f"bm25_weight={result.bm25_weight} dense_weight={result.dense_weight} "
            f"similarity_threshold={result.similarity_threshold} "
            f"fp_threshold={result.fp_threshold}"
        )
        fp_rate_str = (
            "not computed (pass --fp-threshold to compute it)"
            if result.false_positive_rate is None
            else f"{result.false_positive_rate:.4f}"
        )
        print(
            f"precision_mean={result.precision_mean:.4f} "
            f"recall_mean={result.recall_mean:.4f} "
            f"mrr_mean={result.mrr_mean:.4f} "
            f"file_hit_rate={result.file_hit_rate:.4f} "
            f"false_positive_rate={fp_rate_str} "
            f"unanswerable_dense_score_mean={result.unanswerable_dense_score_mean:.4f} "
            f"unanswerable_dense_score_max={result.unanswerable_dense_score_max:.4f} "
            f"answerable_dense_score_mean={result.answerable_dense_score_mean:.4f} "
            f"answerable_dense_score_max={result.answerable_dense_score_max:.4f}"
        )
        if result.unindexed_detection_truncated:
            print(
                "WARNING: unindexed-sample detection hit --max-chunks "
                f"({args.max_chunks}) while enumerating the index -- the "
                "unindexed classification below (and therefore which samples "
                "were excluded from the means above) CANNOT be trusted: a real "
                "retrieval miss on a late-sorting file would be misclassified "
                "as 'unindexed' and silently drop out of the means instead of "
                "counting against them. Re-run with a higher --max-chunks.",
                file=sys.stderr,
            )
        if result.unindexed_sample_count:
            print(
                f"WARNING: {result.unindexed_sample_count} sample(s) excluded from "
                f"the means above -- their expected_files were never found in the "
                f"index: {result.unindexed_sample_ids}"
            )
        if result.recall_by_lang:
            print(f"recall_by_lang={result.recall_by_lang}")
        if result.recall_by_intent:
            print(f"recall_by_intent={result.recall_by_intent}")
        if result.recall_by_tier:
            print(f"recall_by_tier={result.recall_by_tier}")
        return 0

    return 1


def _build_retriever_for_cli(args: argparse.Namespace):
    """Build one ``HybridRetriever`` from the app's own settings + provider
    factories, following the same wiring ``src/bootstrap.py`` uses for the
    tenant-runtime retriever (see ``build_runtime_registry``'s ``_retriever``
    closure there) and that ``src/api/knowledge.py`` relies on for the chat
    query path.

    Deliberately does NOT fall back to an empty in-memory index when the
    configured backend can't actually be reached (e.g. pgvector with no
    DATABASE_URL / no live DB) -- an empty index would silently report zero
    recall on every sample, which reads as "retrieval regressed to zero"
    rather than "couldn't reach the KB". Any failure here (or on the first
    search, for backends like pgvector that connect lazily) is left to
    propagate with its own message; the caller wraps both in a clear
    operator-facing print.
    """
    from src.config import load_settings
    from src.providers import get_vector_store
    from src.rag.embeddings import GeminiEmbedder
    from src.rag.retriever import HybridRetriever, retrieval_config_from_settings

    settings = load_settings()
    vs_cfg: dict[str, Any] = settings.pipeline.vector_store.model_dump()
    # PGVectorAdapter falls back to the raw DATABASE_URL *process env var*
    # when its config carries no database_url. That works for the API server
    # (its env is populated by the container/compose file) but not for a CLI
    # run, where DATABASE_URL usually lives only in .env — which pydantic
    # reads into settings.secrets and never exports to os.environ. Pass it
    # through explicitly so the CLI resolves the database the same way the
    # rest of the app does, from settings rather than from the shell.
    if not vs_cfg.get("database_url") and settings.secrets.DATABASE_URL:
        vs_cfg["database_url"] = settings.secrets.DATABASE_URL
    if args.index_path:
        vs_cfg["index_path"] = args.index_path
    if args.tenant_id:
        vs_cfg["tenant_id"] = args.tenant_id
    if args.crm_id:
        vs_cfg["crm_id"] = args.crm_id

    retrieval_cfg = retrieval_config_from_settings(settings.rag.retrieval)
    if args.strategy is not None:
        retrieval_cfg.strategy = args.strategy
    if args.bm25_weight is not None:
        retrieval_cfg.bm25_weight = args.bm25_weight
    if args.dense_weight is not None:
        retrieval_cfg.dense_weight = args.dense_weight
    if args.similarity_threshold is not None:
        retrieval_cfg.similarity_threshold = args.similarity_threshold
    if args.top_k is not None:
        retrieval_cfg.top_k = args.top_k

    vector_store = get_vector_store(vs_cfg)
    return HybridRetriever(
        # Same reasoning as database_url above: GeminiEmbedder's own fallback
        # reads the GEMINI_API_KEY process env var, which a CLI run started
        # from a shell that only has .env does not have. Take it from
        # settings so config resolution is identical to the app's.
        embedder=GeminiEmbedder(
            dim=vs_cfg.get("embedding_dim", 384),
            api_key=settings.secrets.GEMINI_API_KEY,
        ),
        vector_store=vector_store,
        config=retrieval_cfg,
    )


# --- Mock providers for --mock mode -------------------------------------


class _MockSTT:
    async def transcribe(self, audio, config):
        from src.interfaces.stt import STTResult
        return STTResult(text="mock transcription", confidence=0.5, language=config.language)

    async def transcribe_stream(self, audio_stream, config):
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi-IN"]


class _MockTTS:
    async def synthesize(self, text, config):
        from src.interfaces.tts import TTSResult
        return TTSResult(audio=b"\x00\x00" * 100, duration_ms=80.0 * len(text), sample_rate=config.sample_rate)

    async def synthesize_stream(self, text_stream, config):
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language):
        return []


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
