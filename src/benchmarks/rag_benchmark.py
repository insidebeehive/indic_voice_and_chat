"""RAG benchmark.

Concerns scored independently:

1. Retrieval quality (id-based) — precision@k, recall@k, MRR computed
   against ``expected_chunks`` (set of ground-truth chunk ids). Kept for
   backward compatibility; fragile against re-chunking (see
   ``src/benchmarks/datasets.py`` docstring).

2. Retrieval quality (span-based) — precision@k, recall@k, MRR, and a
   coarser ``file_hit`` diagnostic, computed against ``expected_spans`` /
   ``expected_files`` (verbatim KB text + source filename). Survives a
   re-chunk / re-ingest since it's resolved against whatever chunks are
   actually in the index at scoring time, not a chunk id.

3. Answer faithfulness — does the agent's ``response_text`` rely on chunks
   the retriever actually returned? Light citation-coverage check: every
   citation in ``sources_used`` must be present in the retrieved set
   (catches the LLM inventing references), and at least one citation must
   exist when retrieval returned anything (catches dropped grounding).

4. Answer recall — does the agent's text contain key terms from the
   expected answer? Token-overlap heuristic so we don't pull in a heavy
   semantic evaluator. Not perfect, but actionable for regression tracking.

``run_rag_benchmark`` drives the full agent (LLM calls included) and scores
1/3/4. ``run_retrieval_benchmark`` scores only 2, going straight through
``search_combined`` with no LLM/agent involved — the cheap loop for tuning
fusion weights.
"""

from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Optional

from src.benchmarks.datasets import RAGSample
from src.benchmarks.metrics import LatencyStats, latency_stats
from src.rag.context_builder import search_combined
from src.rag.embeddings import _tokenize
from src.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    # ChatBotAgent is only ever used as a type annotation on run_rag_benchmark
    # below (the LLM-driven path) -- never instantiated or isinstance-checked
    # in this module. Keeping it out of the module-level import list means
    # importing this file (e.g. for the retrieval-only loop in
    # run_retrieval_benchmark, or from scripts/run_benchmark.py's cheap
    # "retrieval" subcommand) no longer drags in the agent/LLM module graph.
    # Safe under `from __future__ import annotations` (PEP 563): the
    # annotation is a string at runtime, never resolved unless something
    # calls typing.get_type_hints() on this module, which nothing here does.
    from src.agents.chatbot import ChatBotAgent
    from src.rag.retriever import HybridRetriever


# --- Retrieval scoring --------------------------------------------------


@dataclass
class RetrievalScore:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: bool


def score_retrieval(
    expected_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    *,
    k: Optional[int] = None,
) -> RetrievalScore:
    """Compute precision@k / recall@k / MRR for one query.

    If ``k`` is None it defaults to ``len(retrieved_chunk_ids)``.
    """
    expected = set(expected_chunk_ids)
    if not expected:
        # Convention: empty ground truth -> trivially correct
        return RetrievalScore(precision_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0, hit=True)
    top_k = retrieved_chunk_ids if k is None else retrieved_chunk_ids[:k]
    hits = [r for r in top_k if r in expected]
    precision = len(hits) / max(len(top_k), 1)
    recall = len(hits) / len(expected)
    mrr = 0.0
    for i, r in enumerate(retrieved_chunk_ids, start=1):
        if r in expected:
            mrr = 1.0 / i
            break
    return RetrievalScore(
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=mrr,
        hit=bool(hits),
    )


# --- Span-based retrieval scoring ----------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    """Collapse whitespace runs (incl. newlines) and strip.

    The chunker joins pieces with " " and prepends overlap tails (see
    src/rag/ingestion.py), so a span copied verbatim from the source doc can
    land in a chunk with different whitespace around it. Deliberately NOT
    lowercasing here (unlike src/benchmarks/metrics.py's normalize_text):
    Devanagari has no case, and lowercasing the Latin side risks conflating
    brand/product names that only differ by case.
    """
    return _WHITESPACE_RE.sub(" ", text or "").strip()


@dataclass
class SpanRetrievalScore:
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    file_hit: bool


def score_retrieval_spans(
    expected_spans: list[str],
    expected_files: list[str],
    retrieved: list[RetrievedChunk],
    *,
    k: Optional[int] = None,
) -> SpanRetrievalScore:
    """Score one query's retrieval against verbatim text spans + filenames.

    - precision@k: (chunks in top-k containing >=1 expected span) / min(k, len(retrieved)).
    - recall@k: (distinct expected spans covered by top-k) / len(expected_spans).
      The denominator is SPANS, not chunks -- one chunk can cover several
      spans, and the chunker's overlap means one span can legitimately show
      up in several chunks; counting chunk-hits would double count.
    - MRR: 1/rank of the first chunk (over the FULL retrieved list, not just
      top-k) containing any expected span. Deliberately NOT scoped to top-k
      (matches the pre-existing score_retrieval behaviour above) -- MRR's
      whole point is "how far down the full ranking did we have to look".
    - file_hit: whether any chunk IN THE TOP-K's metadata["filename"] is in
      expected_files -- separates "right document, wrong passage" from
      "wrong document entirely". Scoped to top-k (not the full retrieved
      list) because file_hit exists to diagnose recall_at_k/precision_at_k,
      which are themselves top-k-scoped -- a filename match at rank 7 with
      k=5 is not something top-k retrieval actually surfaced, so file_hit
      must not claim it did.

    If ``k`` is None it defaults to ``len(retrieved)`` (mirrors
    ``score_retrieval`` above). Callers with ``sample.unanswerable`` set
    must NOT call this -- score those with ``is_false_positive`` instead.
    """
    k_eff = len(retrieved) if k is None else k
    top_k = retrieved[:k_eff]

    norm_spans = [_normalize_ws(s) for s in expected_spans if _normalize_ws(s)]
    covered: set[int] = set()
    relevant_chunks = 0
    for rc in top_k:
        chunk_text = _normalize_ws(rc.document.content)
        hit_this_chunk = False
        for i, span in enumerate(norm_spans):
            if span in chunk_text:
                covered.add(i)
                hit_this_chunk = True
        if hit_this_chunk:
            relevant_chunks += 1

    precision = relevant_chunks / max(min(k_eff, len(retrieved)), 1)
    recall = (len(covered) / len(norm_spans)) if norm_spans else 0.0

    mrr = 0.0
    for i, rc in enumerate(retrieved, start=1):
        chunk_text = _normalize_ws(rc.document.content)
        if any(span in chunk_text for span in norm_spans):
            mrr = 1.0 / i
            break

    expected_file_set = set(expected_files)
    file_hit = bool(expected_file_set) and any(
        (rc.document.metadata or {}).get("filename") in expected_file_set
        for rc in top_k
    )

    return SpanRetrievalScore(
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=mrr,
        file_hit=file_hit,
    )


def is_false_positive(retrieved: list[RetrievedChunk], dense_score_threshold: float) -> bool:
    """For an ``unanswerable`` sample: did retrieval return anything whose
    RAW ``dense_score`` (absolute cosine similarity -- see ``RetrievedChunk.
    dense_score`` and ``src/rag/retriever.py``) is at or above
    ``dense_score_threshold``?

    IMPORTANT -- this must be checked against ``dense_score``, never against
    ``RetrievedChunk.score``. In hybrid mode ``.score`` is the FUSED score,
    which ``_fuse`` (src/rag/retriever.py) min-max normalizes per query
    before combining. ``_minmax`` always maps the winning candidate on each
    arm to exactly 1.0 (and to 1.0 for every candidate on a tie, including
    all-zero input), which means the top fused score is bounded BELOW by
    ``dense_weight`` whenever the dense arm returns at least one candidate --
    regardless of how irrelevant that candidate actually is. Measured over
    20,000 randomised trials with the shipped ``dense_weight=0.7``: minimum
    observed top-1 fused score was exactly 0.7000, including for a gibberish
    query ("zxqv plorm garbleflax") whose raw cosine similarity was 0.0000.
    A threshold compared against ``.score`` is therefore not just tautological
    against the retriever's own ``similarity_threshold`` floor (the original
    Finding-1 bug) but tautological against ``dense_weight`` itself -- the
    exact knob a fusion-weight sweep varies -- which flips
    false_positive_rate 1.0->0.0 at ``dense_weight=0.5`` as a pure
    normalization artifact, independent of retrieval quality.

    A chunk contributed only by the BM25 arm has ``dense_score is None`` (see
    ``_fuse``) and never counts toward this check -- ``None`` is not "a low
    score", it is "no dense signal for this chunk at all".

    This function stays a thin, generic ``any(...)`` check -- the caller
    (``run_retrieval_benchmark``'s ``fp_threshold`` parameter, operator-
    supplied with no baked-in default -- see its docstring) decides what bar
    counts as "confidently wrong". The explicit parameter (rather than
    reading it off a retriever) keeps this testable against hand-built
    ``RetrievedChunk`` lists that never went through a real retriever.
    """
    return any(
        rc.dense_score is not None and rc.dense_score >= dense_score_threshold
        for rc in retrieved
    )


def _top_dense_score(retrieved: list[RetrievedChunk]) -> float:
    """Max RAW ``dense_score`` across ``retrieved``, ignoring chunks with no
    dense signal (BM25-only hits, where ``dense_score is None``).

    Deliberately NOT ``retrieved[0].dense_score`` -- ``retrieved`` is sorted
    by the FUSED score, and a chunk found only by the BM25 arm can outrank
    every dense hit in that fused ordering while carrying ``dense_score =
    None`` itself (reachable whenever ``bm25_weight > dense_weight`` with
    disjoint arms), which would make ``retrieved[0].dense_score`` ``None``
    despite other chunks in the list carrying real dense scores.
    """
    return max(
        (rc.dense_score for rc in retrieved if rc.dense_score is not None),
        default=0.0,
    )


# --- Answer scoring -----------------------------------------------------


@dataclass
class AnswerScore:
    answer_recall: float        # token overlap with expected answer
    faithful: bool              # all citations supported by retrieved
    citations_supported: int
    citations_total: int


def score_answer(
    expected_answer: Optional[str],
    response_text: str,
    cited_sources: list[str],
    retrieved_source_tags: list[str],
) -> AnswerScore:
    available = set(retrieved_source_tags)
    supported = [c for c in cited_sources if c in available]
    faithful = (len(supported) == len(cited_sources)) if cited_sources else True

    if not expected_answer:
        answer_recall = 1.0 if response_text.strip() else 0.0
    else:
        expected_tokens = set(_tokenize(expected_answer))
        response_tokens = set(_tokenize(response_text))
        if not expected_tokens:
            answer_recall = 1.0
        else:
            answer_recall = len(expected_tokens & response_tokens) / len(expected_tokens)

    return AnswerScore(
        answer_recall=answer_recall,
        faithful=faithful,
        citations_supported=len(supported),
        citations_total=len(cited_sources),
    )


# --- Full RAG benchmark -------------------------------------------------


@dataclass
class RAGSampleResult:
    sample_id: str
    query: str
    retrieved_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    answer_recall: float
    faithful: bool
    latency_ms: float
    response_text: str = ""


@dataclass
class RAGRunResult:
    sample_count: int
    precision_mean: float
    recall_mean: float
    mrr_mean: float
    faithfulness_rate: float
    answer_recall_mean: float
    latency: LatencyStats
    per_sample: list[RAGSampleResult] = field(default_factory=list)


async def run_rag_benchmark(
    agent: "ChatBotAgent",
    samples: list[RAGSample],
    *,
    top_k: int = 5,
) -> RAGRunResult:
    """Run each sample through ``agent.handle_message`` + score.

    ``ChatBotAgent`` is imported here (function-local), not at module scope
    -- see the TYPE_CHECKING import note near the top of this file. This is
    the one path in the module that's actually LLM/agent-driven; keeping the
    import local to it means nothing above (in particular
    run_retrieval_benchmark's cheap, LLM-free sweep loop) pays for pulling in
    the agent/LLM module graph just to import this module.
    """
    from src.agents.chatbot import ChatBotAgent  # noqa: F401 -- local import, see docstring above
    rows: list[RAGSampleResult] = []
    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    faithful_count = 0
    answer_recalls: list[float] = []
    latencies: list[float] = []

    for sample in samples:
        t0 = time.perf_counter()
        result = await agent.handle_message(sample.query)
        dt = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [r.document.id for r in result.retrieved]
        retrieved_tags = []
        for r in result.retrieved:
            md = r.document.metadata or {}
            fn = md.get("filename") or md.get("source")
            section = md.get("section") or md.get("page")
            if fn and section is not None:
                retrieved_tags.append(f"{fn}:{section}")
            elif fn:
                retrieved_tags.append(str(fn))
            else:
                retrieved_tags.append(r.document.id)

        retr = score_retrieval(sample.expected_chunks, retrieved_ids, k=top_k)
        ans = score_answer(
            expected_answer=sample.expected_answer,
            response_text=result.response.response_text,
            cited_sources=result.response.sources_used,
            retrieved_source_tags=retrieved_tags,
        )

        precisions.append(retr.precision_at_k)
        recalls.append(retr.recall_at_k)
        mrrs.append(retr.reciprocal_rank)
        if ans.faithful:
            faithful_count += 1
        answer_recalls.append(ans.answer_recall)
        latencies.append(dt)

        rows.append(RAGSampleResult(
            sample_id=sample.id,
            query=sample.query,
            retrieved_ids=retrieved_ids,
            precision_at_k=retr.precision_at_k,
            recall_at_k=retr.recall_at_k,
            reciprocal_rank=retr.reciprocal_rank,
            answer_recall=ans.answer_recall,
            faithful=ans.faithful,
            latency_ms=dt,
            response_text=result.response.response_text,
        ))

    n = max(len(samples), 1)
    return RAGRunResult(
        sample_count=len(samples),
        precision_mean=float(mean(precisions)) if precisions else 0.0,
        recall_mean=float(mean(recalls)) if recalls else 0.0,
        mrr_mean=float(mean(mrrs)) if mrrs else 0.0,
        faithfulness_rate=faithful_count / n,
        answer_recall_mean=float(mean(answer_recalls)) if answer_recalls else 0.0,
        latency=latency_stats(latencies),
        per_sample=rows,
    )


# --- Retrieval-only benchmark (no LLM, no agent) -------------------------


@dataclass
class RetrievalSampleResult:
    sample_id: str
    query: str
    lang: Optional[str]
    intent: Optional[str]
    unanswerable: bool
    retrieved_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    file_hit: bool
    latency_ms: float
    tier: Optional[str] = None
    # True when this sample's expected_files were never found in the index at
    # all (see run_retrieval_benchmark's unindexed-sample detection below).
    # precision/recall/mrr/file_hit are forced to 0.0/False on this row --
    # not measurements, just placeholders -- because there was nothing this
    # retrieval configuration could have done differently.
    unindexed: bool = False
    # Only meaningful when unanswerable is True. None when the run's
    # fp_threshold was never supplied (see RetrievalRunResult.fp_threshold) --
    # "not computed", not "not a false positive". A bare CSV/JSON reader sees
    # None render as "0"/null either way, but the aggregate
    # false_positive_rate on the run result makes the distinction explicit.
    false_positive: Optional[bool] = None


@dataclass
class RetrievalRunResult:
    sample_count: int
    answerable_count: int
    unanswerable_count: int
    precision_mean: float
    recall_mean: float
    mrr_mean: float
    file_hit_rate: float
    latency: LatencyStats
    # Mean recall@k grouped by RAGSample.lang / .intent / .tier -- an
    # aggregate mean hides exactly the question this dataset exists to
    # answer (does a Hinglish/Devanagari phrasing of the same intent
    # retrieve as well as the English one; does recall hold up on
    # module/layout-tier documents), so these breakdowns are first-class,
    # not an afterthought computed later from per_sample.
    recall_by_lang: dict[str, float] = field(default_factory=dict)
    recall_by_intent: dict[str, float] = field(default_factory=dict)
    recall_by_tier: dict[str, float] = field(default_factory=dict)
    # Mean/max of the top-1 RAW dense_score (absolute cosine similarity,
    # never the fused/normalized .score -- see is_false_positive's docstring
    # for why) across UNANSWERABLE and answerable (non-unindexed) samples
    # respectively. Reported as a PAIR of distributions rather than a
    # pass/fail rate against a guessed cutoff: real embedders rarely emit
    # cosine below ~0.5 even for unrelated text, so any hardcoded absolute
    # threshold shipped here would be wrong for someone's embedder. The
    # operator reads the GAP between the two distributions -- a config that
    # separates answerable from unanswerable queries on this score is a good
    # config; one where the two ranges overlap is not, independent of
    # whatever similarity_threshold/fp_threshold happens to be configured.
    # 0.0 when the respective sample set is empty, matching the other
    # empty-input conventions in this module.
    unanswerable_dense_score_mean: float = 0.0
    unanswerable_dense_score_max: float = 0.0
    answerable_dense_score_mean: float = 0.0
    answerable_dense_score_max: float = 0.0
    # Fraction of unanswerable samples whose top RAW dense_score cleared
    # ``fp_threshold``. ``None`` -- not 0.0 -- when ``fp_threshold`` was never
    # supplied: unlike the fused-score version this replaced (pinned at 1.0
    # for every hybrid config -- see is_false_positive's docstring), this one
    # is honest against dense_score, but there is still no universally-correct
    # default cutoff to bake in (see the distributions above), so a rate is
    # only ever reported when an operator explicitly chose a bar via the
    # CLI's --fp-threshold. An always-present-but-frequently-meaningless
    # number is worse than an explicit "not computed".
    false_positive_rate: Optional[float] = None
    # The bar in force for false_positive_rate above, or None if it wasn't
    # computed. Carried alongside the rate so a report never has to guess
    # which is true.
    fp_threshold: Optional[float] = None
    # Samples whose expected_files were never found in the index at all --
    # NOT a retrieval failure (the target document doesn't exist for this
    # retrieval configuration to find), so excluded from every mean above.
    # See run_retrieval_benchmark's docstring for how "in the index" is
    # determined and why it's a best-effort, capability-gated check.
    unindexed_sample_count: int = 0
    unindexed_sample_ids: list[str] = field(default_factory=list)
    # True when the enumeration behind the unindexed-sample check
    # (_collect_indexed_filenames) hit its max_chunks cap -- meaning
    # indexed_filenames is very likely INCOMPLETE, and the same corpus rows
    # were deterministically excluded (list_documents/list_all both return a
    # stable order) rather than randomly sampled. When True,
    # unindexed_sample_count/_ids -- and therefore which samples were
    # excluded from precision/recall/mrr/file_hit above -- cannot be trusted:
    # a real retrieval miss on a late-sorting file would misreport as
    # "unindexed" and silently vanish from the means instead of dragging them
    # down. Callers (the CLI) must surface this loudly rather than let a
    # truncated run report clean numbers.
    unindexed_detection_truncated: bool = False
    # Provenance: the retrieval config actually in force for this run.
    # Without this, two CSVs from different sweep points are indistinguishable
    # after the fact -- see write_retrieval_csv, which also carries these
    # into a sidecar JSON file.
    strategy: str = ""
    top_k: int = 0
    bm25_weight: float = 0.0
    dense_weight: float = 0.0
    similarity_threshold: float = 0.0
    per_sample: list[RetrievalSampleResult] = field(default_factory=list)


def _collect_indexed_filenames(
    retrievers: list["HybridRetriever"], max_chunks: int,
) -> tuple[Optional[set[str]], bool]:
    """Best-effort set of every ``metadata["filename"]`` any retriever in
    ``retrievers`` currently has indexed (drawn from ``list_all``, i.e. the
    hydrated in-memory BM25 corpus -- see HybridRetriever.hydrate_sparse_
    from_persistent, which the CLI calls before running a sweep).

    Returns ``(None, False)`` -- meaning "unknown, skip the unindexed-sample
    check" -- when NONE of the retrievers expose ``list_all`` at all. This is
    a fail-open design deliberately distinct from "the index is empty": a
    test double that never implements list_all (several exist in this
    codebase's test suite) must not have every sample it's asked about
    misreported as unindexed. A real HybridRetriever always has list_all, so
    this only ever degrades detection for hand-rolled fakes, never for
    production usage.

    Otherwise returns ``(filenames, truncated)``, where ``truncated`` is True
    if any retriever's ``list_all(max_chunks)`` returned a count >= max_chunks
    -- indistinguishable from "the corpus has exactly max_chunks rows" but far
    more often meaning the corpus is BIGGER and got capped (see
    HybridRetriever.list_all_persistent's identical caveat). Now that
    ``list_documents`` enumerates in a deterministic order, a truncated
    enumeration doesn't just possibly miss files -- it deterministically
    excludes the same late-sorting ones on every run, which is exactly the
    condition Finding 3 requires callers to treat as unreliable rather than
    silently trusting.
    """
    listers = [getattr(r, "list_all", None) for r in retrievers]
    if not any(listers):
        return None, False
    filenames: set[str] = set()
    truncated = False
    for lister in listers:
        if lister is None:
            continue
        docs = lister(max_chunks)
        if len(docs) >= max_chunks:
            truncated = True
        for doc in docs:
            fn = (doc.metadata or {}).get("filename")
            if fn:
                filenames.add(fn)
    return filenames, truncated


async def run_retrieval_benchmark(
    retrievers: list["HybridRetriever"],
    samples: list[RAGSample],
    *,
    top_k: int = 5,
    fp_threshold: Optional[float] = None,
    tier_filter: Optional[str] = None,
    max_chunks: int = 2000,
) -> RetrievalRunResult:
    """Score retrieval quality alone, via ``search_combined`` -- no LLM call,
    no ``ChatBotAgent``. This is the loop meant to run dozens of times while
    tuning fusion weights, so it deliberately never constructs an LLM
    provider or an agent.

    ``unanswerable`` samples are scored separately (``unanswerable_dense_
    score_mean``/``_max``, and optionally ``false_positive_rate``) and
    excluded from the precision/recall/MRR means entirely -- folding them in
    would silently inflate those means (an empty/weak retrieval on an
    unanswerable query is the CORRECT outcome, not a miss).

    ``fp_threshold`` -- the bar a RAW ``dense_score`` (never the fused
    ``.score`` -- see ``is_false_positive``'s docstring) must clear for a
    retrieval on an ``unanswerable`` sample to count as a false positive.
    Defaults to ``None``, meaning "not computed": ``false_positive_rate`` and
    the per-sample ``false_positive`` flag both come back ``None`` unless an
    operator explicitly supplies a value (the CLI's ``--fp-threshold`` flag).
    There is deliberately NO baked-in default threshold here any more --
    real embedders rarely emit cosine similarity below roughly 0.5 even for
    completely unrelated text, so any single guessed cutoff would be wrong
    for whichever embedder is actually in use. Use
    ``unanswerable_dense_score_mean``/``_max`` alongside ``answerable_dense_
    score_mean``/``_max`` on the result to set the bar empirically from the
    gap between the two distributions, then pass it explicitly if a
    pass/fail rate is still wanted.

    ``tier_filter`` -- when set, scores only samples whose ``RAGSample.tier``
    equals this value (exact match; samples with ``tier=None``, including
    ordinary ``unanswerable`` samples with no target document, are excluded
    by any non-None filter same as a genuine tier mismatch). Lets an operator
    score only what their index actually holds -- see the corpus-tier note
    below and the CLI's ``--tier`` flag.

    Unindexed-sample handling -- a sample whose ``expected_files`` were never
    ingested into ANY retriever's index at all (e.g. a KB-module or per-tenant
    layout file that a default-seeded corpus never loaded) is not a retrieval
    failure: no configuration of this retriever could have found a document
    that was never indexed. Such samples are detected via
    ``_collect_indexed_filenames`` (best-effort; see its docstring), reported
    separately as ``unindexed_sample_count`` / ``unindexed_sample_ids``, and
    excluded from precision/recall/MRR/file_hit -- otherwise a default-seeded
    corpus floors those means for corpus-composition reasons that have
    nothing to do with retrieval quality, with no warning that it's happening.

    If that enumeration itself hit ``max_chunks`` (see ``_collect_indexed_
    filenames``), the unindexed classification -- and therefore which samples
    got excluded from the means above -- is NOT trustworthy: a real miss on a
    late-sorting file would be misclassified as "unindexed" and silently drop
    out rather than count against the means. This is reported via
    ``unindexed_detection_truncated`` rather than by raising, so a caller
    that only wants the numbers for samples it knows are unaffected can still
    proceed, but the CLI treats it as a loud warning, not a footnote.
    """
    if tier_filter is not None:
        samples = [s for s in samples if s.tier == tier_filter]

    indexed_filenames, indexed_enumeration_truncated = _collect_indexed_filenames(
        retrievers, max_chunks)

    rows: list[RetrievalSampleResult] = []
    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    file_hit_count = 0
    false_positive_count = 0
    latencies: list[float] = []
    answerable = 0
    unanswerable = 0
    unanswerable_dense_scores: list[float] = []
    answerable_dense_scores: list[float] = []
    unindexed_ids: list[str] = []
    recall_by_lang: dict[str, list[float]] = defaultdict(list)
    recall_by_intent: dict[str, list[float]] = defaultdict(list)
    recall_by_tier: dict[str, list[float]] = defaultdict(list)

    for sample in samples:
        t0 = time.perf_counter()
        retrieved = await search_combined(sample.query, retrievers, top_k=top_k)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        retrieved_ids = [r.document.id for r in retrieved]

        if sample.unanswerable:
            unanswerable += 1
            unanswerable_dense_scores.append(_top_dense_score(retrieved))
            fp: Optional[bool] = None
            if fp_threshold is not None:
                fp = is_false_positive(retrieved, fp_threshold)
                if fp:
                    false_positive_count += 1
            rows.append(RetrievalSampleResult(
                sample_id=sample.id, query=sample.query, lang=sample.lang,
                intent=sample.intent, unanswerable=True,
                retrieved_ids=retrieved_ids, precision_at_k=0.0, recall_at_k=0.0,
                reciprocal_rank=0.0, file_hit=False, false_positive=fp,
                latency_ms=dt, tier=sample.tier,
            ))
            continue

        is_unindexed = (
            indexed_filenames is not None
            and bool(sample.expected_files)
            and not (set(sample.expected_files) & indexed_filenames)
        )
        if is_unindexed:
            unindexed_ids.append(sample.id)
            rows.append(RetrievalSampleResult(
                sample_id=sample.id, query=sample.query, lang=sample.lang,
                intent=sample.intent, unanswerable=False,
                retrieved_ids=retrieved_ids, precision_at_k=0.0, recall_at_k=0.0,
                reciprocal_rank=0.0, file_hit=False, false_positive=None,
                latency_ms=dt, tier=sample.tier, unindexed=True,
            ))
            continue

        answerable += 1
        answerable_dense_scores.append(_top_dense_score(retrieved))
        score = score_retrieval_spans(
            sample.expected_spans, sample.expected_files, retrieved, k=top_k)
        precisions.append(score.precision_at_k)
        recalls.append(score.recall_at_k)
        mrrs.append(score.reciprocal_rank)
        if score.file_hit:
            file_hit_count += 1
        if sample.lang:
            recall_by_lang[sample.lang].append(score.recall_at_k)
        if sample.intent:
            recall_by_intent[sample.intent].append(score.recall_at_k)
        if sample.tier:
            recall_by_tier[sample.tier].append(score.recall_at_k)

        rows.append(RetrievalSampleResult(
            sample_id=sample.id, query=sample.query, lang=sample.lang,
            intent=sample.intent, unanswerable=False,
            retrieved_ids=retrieved_ids, precision_at_k=score.precision_at_k,
            recall_at_k=score.recall_at_k, reciprocal_rank=score.reciprocal_rank,
            file_hit=score.file_hit, false_positive=None, latency_ms=dt,
            tier=sample.tier,
        ))

    cfg = retrievers[0].config if retrievers else None
    return RetrievalRunResult(
        sample_count=len(samples),
        answerable_count=answerable,
        unanswerable_count=unanswerable,
        precision_mean=float(mean(precisions)) if precisions else 0.0,
        recall_mean=float(mean(recalls)) if recalls else 0.0,
        mrr_mean=float(mean(mrrs)) if mrrs else 0.0,
        file_hit_rate=(file_hit_count / answerable) if answerable else 0.0,
        false_positive_rate=(
            None if fp_threshold is None
            else (false_positive_count / unanswerable) if unanswerable else 0.0
        ),
        fp_threshold=fp_threshold,
        unanswerable_dense_score_mean=(
            float(mean(unanswerable_dense_scores)) if unanswerable_dense_scores else 0.0
        ),
        unanswerable_dense_score_max=(
            max(unanswerable_dense_scores) if unanswerable_dense_scores else 0.0
        ),
        answerable_dense_score_mean=(
            float(mean(answerable_dense_scores)) if answerable_dense_scores else 0.0
        ),
        answerable_dense_score_max=(
            max(answerable_dense_scores) if answerable_dense_scores else 0.0
        ),
        unindexed_sample_count=len(unindexed_ids),
        unindexed_sample_ids=unindexed_ids,
        unindexed_detection_truncated=indexed_enumeration_truncated,
        latency=latency_stats(latencies),
        recall_by_lang={k: float(mean(v)) for k, v in recall_by_lang.items()},
        recall_by_intent={k: float(mean(v)) for k, v in recall_by_intent.items()},
        recall_by_tier={k: float(mean(v)) for k, v in recall_by_tier.items()},
        strategy=cfg.strategy if cfg else "",
        top_k=top_k,
        bm25_weight=cfg.bm25_weight if cfg else 0.0,
        dense_weight=cfg.dense_weight if cfg else 0.0,
        similarity_threshold=cfg.similarity_threshold if cfg else 0.0,
        per_sample=rows,
    )
