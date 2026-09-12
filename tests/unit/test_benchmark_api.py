from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import benchmarks as bench_routes
from src.auth.middleware import set_admin_tokens
from src.benchmarks.latency_benchmark import LatencyMatrixResult, LatencyRunResult
from src.benchmarks.metrics import AggregateAccuracy, latency_stats
from src.benchmarks.runner import SuiteResults, SuiteRunner
from src.benchmarks.stt_benchmark import STTRunResult


ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest.fixture
def app():
    runner = SuiteRunner()
    bench_routes.set_runner(runner)
    set_admin_tokens(["admin-token"])
    a = FastAPI()
    a.include_router(bench_routes.router)
    yield a, runner
    bench_routes.set_runner(None)
    set_admin_tokens([])


@pytest.mark.asyncio
async def test_record_persists_in_memory() -> None:
    runner = SuiteRunner()
    results = SuiteResults(stt=[
        STTRunResult(
            provider="sarvam", sample_count=2,
            overall=AggregateAccuracy(sample_count=2, wer_mean=0.1, cer_mean=0.05, wer_p95=0.2, cer_p95=0.1),
            latency=latency_stats([100.0, 120.0]),
        )
    ])
    record = await runner.record(
        name="initial",
        description="smoke",
        pipeline_config={"stt": "sarvam"},
        language="hi-IN",
        dataset="data/stt.jsonl",
        results=results,
    )
    assert record.id.startswith("br_")
    assert len(runner.records) == 1


@pytest.mark.asyncio
async def test_latency_endpoint_returns_recent_runs(app) -> None:
    fastapi_app, runner = app
    await runner.record(
        name="r1", description="x", pipeline_config={}, language="hi-IN", dataset="d",
        results=SuiteResults(latency=LatencyMatrixResult(results=[
            LatencyRunResult(
                combo={"stt": "sarvam", "llm": "groq", "tts": "sarvam"},
                sample_count=5,
                stt=latency_stats([100.0]),
                llm_ttft=latency_stats([50.0]),
                llm_total=latency_stats([200.0]),
                tts_first_chunk=latency_stats([100.0]),
                tts_total=latency_stats([200.0]),
                end_to_end=latency_stats([400.0, 500.0]),
            )
        ])),
    )
    client = TestClient(fastapi_app)
    body = client.get("/benchmarks/latency", headers=ADMIN_HEADERS).json()
    assert body["runs_considered"] == 1
    assert len(body["entries"]) == 1
    assert body["entries"][0]["combo"]["llm"] == "groq"


@pytest.mark.asyncio
async def test_accuracy_endpoint(app) -> None:
    fastapi_app, runner = app
    await runner.record(
        name="r1", description="x", pipeline_config={}, language="hi-IN", dataset="d",
        results=SuiteResults(stt=[STTRunResult(
            provider="sarvam", sample_count=10,
            overall=AggregateAccuracy(sample_count=10, wer_mean=0.12, cer_mean=0.05, wer_p95=0.2, cer_p95=0.08),
            latency=latency_stats([100.0]),
        )]),
    )
    client = TestClient(fastapi_app)
    body = client.get("/benchmarks/accuracy", headers=ADMIN_HEADERS).json()
    assert body["runs_considered"] == 1
    assert body["entries"][0]["provider"] == "sarvam"
    assert body["entries"][0]["wer_mean"] == pytest.approx(0.12)


@pytest.mark.asyncio
async def test_list_runs_and_get_run(app) -> None:
    fastapi_app, runner = app
    record = await runner.record(
        name="r1", description="x", pipeline_config={"k": "v"},
        language="hi-IN", dataset="d", results=SuiteResults(),
    )
    client = TestClient(fastapi_app)
    listing = client.get("/benchmarks/runs", headers=ADMIN_HEADERS).json()
    assert listing["total"] == 1
    assert listing["runs"][0]["id"] == record.id

    detail = client.get(f"/benchmarks/runs/{record.id}", headers=ADMIN_HEADERS).json()
    assert detail["pipeline_config"] == {"k": "v"}


@pytest.mark.asyncio
async def test_get_run_404(app) -> None:
    fastapi_app, _ = app
    client = TestClient(fastapi_app)
    assert client.get("/benchmarks/runs/missing", headers=ADMIN_HEADERS).status_code == 404


def test_routes_503_when_runner_unset() -> None:
    bench_routes.set_runner(None)
    set_admin_tokens(["admin-token"])
    a = FastAPI()
    a.include_router(bench_routes.router)
    client = TestClient(a)
    assert client.get("/benchmarks/latency", headers=ADMIN_HEADERS).status_code == 503
    set_admin_tokens([])


def test_routes_401_without_admin_token() -> None:
    bench_routes.set_runner(SuiteRunner())
    set_admin_tokens(["admin-token"])
    a = FastAPI()
    a.include_router(bench_routes.router)
    client = TestClient(a)
    assert client.get("/benchmarks/latency").status_code == 401
    set_admin_tokens([])
    bench_routes.set_runner(None)


# --- CLI ----------------------------------------------------------------


def test_cli_stt_mock_round_trip(tmp_path: Path) -> None:
    from scripts.run_benchmark import main

    dataset = tmp_path / "stt.jsonl"
    dataset.write_text(
        json.dumps({"id": "s1", "transcript": "mock transcription", "language": "hi"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "results.csv"
    rc = main(["stt", "--dataset", str(dataset), "--out", str(out), "--mock"])
    assert rc == 0
    body = out.read_text()
    assert "sample_id" in body
    assert "s1" in body


def test_cli_tts_mock_round_trip(tmp_path: Path) -> None:
    from scripts.run_benchmark import main

    dataset = tmp_path / "tts.jsonl"
    dataset.write_text(
        json.dumps({"id": "t1", "text": "Namaste dosto", "language": "hi-IN"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "tts_results.csv"
    rc = main(["tts", "--dataset", str(dataset), "--out", str(out), "--mock"])
    assert rc == 0
    assert "t1" in out.read_text()


def test_cli_rag_returns_2_until_app_bootstrap(tmp_path: Path, capsys) -> None:
    from scripts.run_benchmark import main

    rag_dataset = tmp_path / "rag.jsonl"
    rag_dataset.write_text("{}\n", encoding="utf-8")
    rc = main(["rag", "--dataset", str(rag_dataset), "--out", str(tmp_path / "x.csv")])
    assert rc == 2


def test_cli_retrieval_accepts_rrf_strategy_and_rrf_k() -> None:
    from scripts.run_benchmark import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args(
        ["retrieval", "--dataset", "x", "--out", "y", "--strategy", "rrf", "--rrf-k", "30"]
    )
    assert args.strategy == "rrf"
    assert args.rrf_k == 30

    args_no_k = parser.parse_args(["retrieval", "--dataset", "x", "--out", "y"])
    assert args_no_k.rrf_k is None


def test_cli_retrieval_rejects_unknown_strategy() -> None:
    from scripts.run_benchmark import _build_arg_parser

    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["retrieval", "--dataset", "x", "--out", "y", "--strategy", "bm25"]
        )


def test_cli_retrieval_validates_config_after_applying_overrides(monkeypatch) -> None:
    """``_build_retriever_for_cli`` must call ``validate_retrieval_config``
    AFTER applying the CLI overrides (--strategy/--similarity-threshold/etc.),
    not before -- otherwise it would validate the pre-override settings and
    let an invalid combination like --strategy rrf --similarity-threshold 0.3
    build a retriever instead of raising. This pins that ordering so a future
    reshuffle of _build_retriever_for_cli regresses this test instead of
    shipping a silently-inert validation.
    """
    import src.config as config_module
    from scripts.run_benchmark import _build_arg_parser, _build_retriever_for_cli
    from src.config import RetrievalSettings, VectorStoreConfig

    # A valid settings object on its own (strategy="hybrid", threshold=0.0) --
    # only the CLI overrides below make it invalid.
    fake_settings = SimpleNamespace(
        pipeline=SimpleNamespace(vector_store=VectorStoreConfig(provider="faiss")),
        secrets=SimpleNamespace(DATABASE_URL=None, GEMINI_API_KEY=None),
        rag=SimpleNamespace(
            retrieval=RetrievalSettings(
                strategy="hybrid",
                top_k=5,
                bm25_weight=0.3,
                dense_weight=0.7,
                rrf_k=60,
                similarity_threshold=0.0,
            )
        ),
    )
    monkeypatch.setattr(config_module, "load_settings", lambda *a, **k: fake_settings)

    args = _build_arg_parser().parse_args(
        ["retrieval", "--dataset", "x", "--out", "y",
         "--strategy", "rrf", "--similarity-threshold", "0.3"]
    )

    with pytest.raises(ValueError, match="similarity_threshold"):
        _build_retriever_for_cli(args)


def test_export_results_script_writes_per_kind_files(tmp_path: Path) -> None:
    from scripts.export_results import main

    payload = {
        "id": "br_x",
        "results": {
            "stt": [{"provider": "p", "overall": {"wer_mean": 0.1}}],
            "latency": {"results": [{"combo": {"llm": "groq"}}]},
        },
    }
    src = tmp_path / "in.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out"
    rc = main(["--results-json", str(src), "--out-dir", str(out)])
    assert rc == 0
    assert (out / "stt.json").exists()
    assert (out / "latency.json").exists()
