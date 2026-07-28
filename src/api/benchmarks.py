"""Benchmark endpoints (PRD §7.7).

- GET /benchmarks/latency   per-provider-combo latency stats from the latest runs
- GET /benchmarks/accuracy  per-provider STT WER/CER summaries
- GET /benchmarks/runs      list recorded suite runs
- GET /benchmarks/runs/{id} fetch a specific run
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import require_admin
from src.benchmarks.runner import SuiteRunner
from src.models.turn_metrics import TurnMetric

router = APIRouter(
    prefix="/benchmarks",
    tags=["benchmarks"],
    dependencies=[Depends(require_admin)],
)


_runner: Optional[SuiteRunner] = None


def set_runner(runner: Optional[SuiteRunner]) -> None:
    global _runner
    _runner = runner


def _require_runner() -> SuiteRunner:
    if _runner is None:
        raise HTTPException(status_code=503, detail="benchmark runner not initialized")
    return _runner


class LatencyEntry(BaseModel):
    combo: dict[str, str]
    p95_ms: float
    mean_ms: float
    samples: int


class LatencyResponse(BaseModel):
    runs_considered: int
    entries: list[LatencyEntry]


class AccuracyEntry(BaseModel):
    provider: str
    wer_mean: float
    cer_mean: float
    sample_count: int


class AccuracyResponse(BaseModel):
    runs_considered: int
    entries: list[AccuracyEntry]


class RunSummary(BaseModel):
    id: str
    name: str
    description: str
    language: str
    dataset: str
    created_at: str


class RunListResponse(BaseModel):
    runs: list[RunSummary]
    total: int


class TurnMetricsComboEntry(BaseModel):
    """One (mode, stt_provider, llm_provider, tts_provider) combo's averaged
    per-turn latencies.

    NOT directly comparable across ``mode``: for ``mode="s2s"`` (Gemini
    Live), there is no bridge-level VAD/utterance-end signal, so
    ``avg_total_latency_ms`` is a turn-window duration (session-connect or
    prior-turn-complete to now) that includes the caller's own speaking
    time, not a cascade-style utterance-end-to-response latency — and
    ``avg_tts_first_chunk_ms`` is reinterpreted as time-to-first-spoken-audio
    from the realtime model, not a TTS-specific measurement. Compare
    ``mode="layered"`` rows against each other, and ``mode="s2s"`` rows
    against each other, but not the two against one another.
    """

    mode: str
    stt_provider: Optional[str]
    llm_provider: str
    tts_provider: Optional[str]
    samples: int
    avg_stt_latency_ms: float
    avg_llm_ttft_ms: float
    avg_llm_total_ms: float
    avg_tts_first_chunk_ms: float
    avg_tts_total_ms: float
    avg_total_latency_ms: float
    avg_tts_segments_dropped: float


class TurnMetricsSummaryResponse(BaseModel):
    entries: list[TurnMetricsComboEntry]


@router.get("/latency", response_model=LatencyResponse)
async def latency() -> LatencyResponse:
    runner = _require_runner()
    entries: list[LatencyEntry] = []
    runs = 0
    for record in runner.records:
        latency_payload = (record.results or {}).get("latency")
        if not latency_payload:
            continue
        runs += 1
        for run in (latency_payload.get("results") or []):
            e2e = run.get("end_to_end") or {}
            entries.append(LatencyEntry(
                combo=run.get("combo", {}),
                p95_ms=float(e2e.get("p95_ms", 0.0)),
                mean_ms=float(e2e.get("mean_ms", 0.0)),
                samples=int(run.get("sample_count", 0)),
            ))
    return LatencyResponse(runs_considered=runs, entries=entries)


@router.get("/accuracy", response_model=AccuracyResponse)
async def accuracy() -> AccuracyResponse:
    runner = _require_runner()
    entries: list[AccuracyEntry] = []
    runs = 0
    for record in runner.records:
        stt = (record.results or {}).get("stt") or []
        if not stt:
            continue
        runs += 1
        for r in stt:
            overall = r.get("overall") or {}
            entries.append(AccuracyEntry(
                provider=r.get("provider", "?"),
                wer_mean=float(overall.get("wer_mean", 0.0)),
                cer_mean=float(overall.get("cer_mean", 0.0)),
                sample_count=int(r.get("sample_count", 0)),
            ))
    return AccuracyResponse(runs_considered=runs, entries=entries)


@router.get("/runs", response_model=RunListResponse)
async def list_runs() -> RunListResponse:
    runner = _require_runner()
    items = [
        RunSummary(
            id=r.id,
            name=r.name,
            description=r.description,
            language=r.language,
            dataset=r.dataset,
            created_at=r.created_at.isoformat(),
        )
        for r in runner.records
    ]
    return RunListResponse(runs=items, total=len(items))


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    runner = _require_runner()
    for r in runner.records:
        if r.id == run_id:
            return {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "language": r.language,
                "dataset": r.dataset,
                "pipeline_config": r.pipeline_config,
                "results": r.results,
                "created_at": r.created_at.isoformat(),
            }
    raise HTTPException(status_code=404, detail="run not found")


@router.get("/turn-metrics/summary", response_model=TurnMetricsSummaryResponse)
async def turn_metrics_summary(
    db: AsyncSession = Depends(get_db_session),
) -> TurnMetricsSummaryResponse:
    stmt = (
        select(
            TurnMetric.mode,
            TurnMetric.stt_provider,
            TurnMetric.llm_provider,
            TurnMetric.tts_provider,
            func.count().label("samples"),
            func.avg(TurnMetric.stt_latency_ms).label("avg_stt_latency_ms"),
            func.avg(TurnMetric.llm_ttft_ms).label("avg_llm_ttft_ms"),
            func.avg(TurnMetric.llm_total_ms).label("avg_llm_total_ms"),
            func.avg(TurnMetric.tts_first_chunk_ms).label("avg_tts_first_chunk_ms"),
            func.avg(TurnMetric.tts_total_ms).label("avg_tts_total_ms"),
            func.avg(TurnMetric.total_latency_ms).label("avg_total_latency_ms"),
            func.avg(TurnMetric.tts_segments_dropped).label("avg_tts_segments_dropped"),
        )
        .group_by(TurnMetric.mode, TurnMetric.stt_provider, TurnMetric.llm_provider, TurnMetric.tts_provider)
    )
    rows = (await db.execute(stmt)).all()
    entries = [
        TurnMetricsComboEntry(
            mode=r.mode,
            stt_provider=r.stt_provider,
            llm_provider=r.llm_provider,
            tts_provider=r.tts_provider,
            samples=r.samples,
            avg_stt_latency_ms=float(r.avg_stt_latency_ms or 0.0),
            avg_llm_ttft_ms=float(r.avg_llm_ttft_ms or 0.0),
            avg_llm_total_ms=float(r.avg_llm_total_ms or 0.0),
            avg_tts_first_chunk_ms=float(r.avg_tts_first_chunk_ms or 0.0),
            avg_tts_total_ms=float(r.avg_tts_total_ms or 0.0),
            avg_total_latency_ms=float(r.avg_total_latency_ms or 0.0),
            avg_tts_segments_dropped=float(r.avg_tts_segments_dropped or 0.0),
        )
        for r in rows
    ]
    return TurnMetricsSummaryResponse(entries=entries)
