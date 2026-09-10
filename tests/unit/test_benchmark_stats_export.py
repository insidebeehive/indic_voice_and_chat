from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from src.benchmarks.export import (
    recommend_providers,
    retrieval_csv_meta_path,
    to_jsonable,
    write_json,
    write_latency_csv,
    write_rag_csv,
    write_retrieval_csv,
    write_stt_csv,
    write_task_csv,
    write_tts_csv,
)
from src.benchmarks.latency_benchmark import LatencyMatrixResult, LatencyRunResult
from src.benchmarks.metrics import AggregateAccuracy, LatencyStats, latency_stats
from src.benchmarks.rag_benchmark import (
    RAGRunResult,
    RAGSampleResult,
    RetrievalRunResult,
    RetrievalSampleResult,
)
from src.benchmarks.stats import one_way_anova
from src.benchmarks.stt_benchmark import STTRunResult, STTSampleResult
from src.benchmarks.task_benchmark import TaskRunResult, TaskScenarioResult
from src.benchmarks.tts_benchmark import TTSRunResult, TTSSampleResult


# --- ANOVA --------------------------------------------------------------


def test_anova_clearly_different_groups_flags_significant() -> None:
    g1 = [0.01, 0.02, 0.015, 0.012, 0.018] * 10   # very low WER
    g2 = [0.50, 0.55, 0.52, 0.48, 0.51] * 10      # very high WER
    r = one_way_anova([g1, g2])
    assert r.f_statistic > 10
    assert r.significance == "<0.001"
    assert r.df_between == 1
    assert r.df_within == len(g1) + len(g2) - 2


def test_anova_similar_groups_not_significant() -> None:
    g1 = [0.10, 0.12, 0.11, 0.09, 0.10] * 5
    g2 = [0.11, 0.10, 0.12, 0.10, 0.11] * 5
    r = one_way_anova([g1, g2])
    assert r.significance == ">=0.05"


def test_anova_requires_at_least_2_groups() -> None:
    with pytest.raises(ValueError):
        one_way_anova([[0.1, 0.2]])


def test_anova_rejects_empty_group() -> None:
    with pytest.raises(ValueError):
        one_way_anova([[0.1], []])


# --- CSV writers --------------------------------------------------------


def _empty_agg() -> AggregateAccuracy:
    return AggregateAccuracy(sample_count=0, wer_mean=0.0, cer_mean=0.0, wer_p95=0.0, cer_p95=0.0)


def test_write_stt_csv_to_buffer() -> None:
    r = STTRunResult(
        provider="p",
        sample_count=1,
        overall=_empty_agg(),
        latency=latency_stats([100.0]),
        per_sample=[STTSampleResult(
            sample_id="s1", reference="hi", hypothesis="hi", language="hi",
            code_switch=False, confidence=0.9, stt_latency_ms=100.0, wer=0.0, cer=0.0,
        )],
    )
    buf = io.StringIO()
    n = write_stt_csv(buf, r)
    assert n == 1
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert rows[0][0] == "sample_id"
    assert rows[1][0] == "s1"


def test_write_stt_csv_to_path(tmp_path: Path) -> None:
    r = STTRunResult(
        provider="p",
        sample_count=0,
        overall=_empty_agg(),
        latency=latency_stats([]),
    )
    out = tmp_path / "out" / "stt.csv"
    n = write_stt_csv(out, r)
    assert n == 0
    assert out.exists()


def test_write_tts_csv() -> None:
    r = TTSRunResult(
        provider="p", sample_count=1, avg_duration_ms=100.0, avg_chars_per_second=12.0,
        expected_sample_rate=16000, sample_rate_violations=0,
        latency=latency_stats([5.0]),
        mos_mean=4.0,
        per_sample=[TTSSampleResult(
            sample_id="t1", text="Hi", language="en", audio_bytes=10,
            duration_ms=100, sample_rate=16000, chars_per_second=20.0,
            synth_latency_ms=5.0, mos=4.0,
        )],
    )
    buf = io.StringIO()
    n = write_tts_csv(buf, r)
    assert n == 1
    assert "t1" in buf.getvalue()
    assert "4.00" in buf.getvalue()


def test_write_latency_csv() -> None:
    result = LatencyMatrixResult(results=[
        LatencyRunResult(
            combo={"stt": "sarvam", "llm": "groq", "tts": "sarvam"},
            sample_count=10,
            stt=latency_stats([100.0]),
            llm_ttft=latency_stats([50.0]),
            llm_total=latency_stats([400.0]),
            tts_first_chunk=latency_stats([200.0]),
            tts_total=latency_stats([300.0]),
            end_to_end=latency_stats([900.0]),
        )
    ])
    buf = io.StringIO()
    n = write_latency_csv(buf, result)
    assert n == 1
    body = buf.getvalue()
    assert "sarvam" in body
    assert "groq" in body


def test_write_rag_csv() -> None:
    r = RAGRunResult(
        sample_count=1, precision_mean=1.0, recall_mean=1.0, mrr_mean=1.0,
        faithfulness_rate=1.0, answer_recall_mean=1.0,
        latency=latency_stats([10.0]),
        per_sample=[RAGSampleResult(
            sample_id="r1", query="q", retrieved_ids=["c1", "c2"],
            precision_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0,
            answer_recall=1.0, faithful=True, latency_ms=10.0, response_text="ok",
        )],
    )
    buf = io.StringIO()
    n = write_rag_csv(buf, r)
    assert n == 1
    assert "c1;c2" in buf.getvalue()


def test_write_retrieval_csv_is_plain_rectangular_no_preamble_row() -> None:
    # Finding 5: an earlier version prepended a "#"-prefixed config row ahead
    # of the header. Both csv.DictReader and pandas.read_csv()'s default
    # parsing mis-parse that (see _open_writer's docstring for the measured
    # specifics) -- so the CSV itself must now be plain header + data rows,
    # nothing else. Config provenance moved to a sidecar JSON -- see
    # test_write_retrieval_csv_writes_sidecar_meta_and_dict_reader_aligns.
    r = RetrievalRunResult(
        sample_count=1, answerable_count=1, unanswerable_count=0,
        precision_mean=1.0, recall_mean=1.0, mrr_mean=1.0,
        file_hit_rate=1.0, false_positive_rate=0.0,
        latency=latency_stats([10.0]),
        strategy="hybrid", top_k=5, bm25_weight=0.3, dense_weight=0.7,
        similarity_threshold=0.1, fp_threshold=0.5,
        per_sample=[RetrievalSampleResult(
            sample_id="r1", query="q", lang="en", intent="deposit",
            unanswerable=False, retrieved_ids=["c1"],
            precision_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0,
            file_hit=True, false_positive=False, latency_ms=10.0,
            tier="pack", unindexed=False,
        )],
    )
    buf = io.StringIO()
    n = write_retrieval_csv(buf, r)
    assert n == 1
    lines = buf.getvalue().splitlines()
    # Row 0 is the real header -- no leading "#"-prefixed config row.
    assert not lines[0].startswith("#")
    header = lines[0].split(",")
    assert "tier" in header
    assert "unindexed" in header
    assert "r1" in lines[1]
    assert "pack" in lines[1]
    assert len(lines) == 2  # header + exactly one data row, nothing else


def test_write_retrieval_csv_writes_sidecar_meta_and_dict_reader_aligns(tmp_path: Path) -> None:
    # The missing test flagged by Finding 5: prove csv.DictReader (a standard
    # reader, not this suite's own line-index assertions) parses the file
    # with the correct fieldnames and correct per-column alignment, and that
    # the run's config/provenance survives via the sidecar JSON.
    r = RetrievalRunResult(
        sample_count=1, answerable_count=1, unanswerable_count=0,
        precision_mean=1.0, recall_mean=1.0, mrr_mean=1.0,
        file_hit_rate=1.0, false_positive_rate=0.0,
        latency=latency_stats([10.0]),
        strategy="hybrid", top_k=5, bm25_weight=0.3, dense_weight=0.7,
        similarity_threshold=0.1, fp_threshold=0.5,
        per_sample=[RetrievalSampleResult(
            sample_id="r1", query="q", lang="en", intent="deposit",
            unanswerable=False, retrieved_ids=["c1"],
            precision_at_k=1.0, recall_at_k=1.0, reciprocal_rank=1.0,
            file_hit=True, false_positive=False, latency_ms=10.0,
            tier="pack", unindexed=False,
        )],
    )
    out = tmp_path / "sweep.csv"
    n = write_retrieval_csv(out, r)
    assert n == 1

    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == [
            "sample_id", "query", "lang", "intent", "tier", "unanswerable",
            "unindexed", "retrieved_ids", "precision_at_k", "recall_at_k",
            "reciprocal_rank", "file_hit", "false_positive", "latency_ms",
        ]
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    # No restkey (None) column -- every field landed under its real name.
    assert None not in row
    assert row["sample_id"] == "r1"
    assert row["tier"] == "pack"
    assert row["precision_at_k"] == "1.0000"
    assert row["latency_ms"] == "10.00"

    meta_path = retrieval_csv_meta_path(out)
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["strategy"] == "hybrid"
    assert meta["top_k"] == 5
    assert meta["bm25_weight"] == 0.3
    assert meta["dense_weight"] == 0.7
    assert meta["similarity_threshold"] == 0.1
    assert meta["fp_threshold"] == 0.5


def test_write_retrieval_csv_marks_unindexed_rows() -> None:
    r = RetrievalRunResult(
        sample_count=1, answerable_count=0, unanswerable_count=0,
        precision_mean=0.0, recall_mean=0.0, mrr_mean=0.0,
        file_hit_rate=0.0, false_positive_rate=0.0,
        latency=latency_stats([5.0]),
        unindexed_sample_count=1, unindexed_sample_ids=["r1"],
        per_sample=[RetrievalSampleResult(
            sample_id="r1", query="q", lang=None, intent=None,
            unanswerable=False, retrieved_ids=[],
            precision_at_k=0.0, recall_at_k=0.0, reciprocal_rank=0.0,
            file_hit=False, false_positive=False, latency_ms=5.0,
            unindexed=True,
        )],
    )
    buf = io.StringIO()
    write_retrieval_csv(buf, r)
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    header = rows[0]
    data_row = rows[1]
    assert data_row[header.index("unindexed")] == "1"


def test_write_task_csv() -> None:
    r = TaskRunResult(
        scenario_count=1, completion_rate=1.0, disposition_match_rate=1.0,
        avg_slot_fill_rate=1.0, avg_slot_value_match_rate=1.0,
        latency=latency_stats([10.0]),
        per_scenario=[TaskScenarioResult(
            scenario_id="s1", completed=True, disposition_match=True,
            expected_disposition="close_positive", actual_action="close_positive",
            required_slot_fill_rate=1.0, slot_value_match_rate=1.0,
            turn_count=2, duration_ms=10.0, final_state="ended",
        )],
    )
    buf = io.StringIO()
    n = write_task_csv(buf, r)
    assert n == 1
    assert "close_positive" in buf.getvalue()


# --- JSON dump ----------------------------------------------------------


def test_to_jsonable_handles_nested_dataclass() -> None:
    r = STTRunResult(provider="p", sample_count=0, overall=_empty_agg(), latency=latency_stats([]))
    out = to_jsonable(r)
    assert isinstance(out, dict)
    assert out["provider"] == "p"
    assert isinstance(out["overall"], dict)


def test_write_json_round_trip(tmp_path: Path) -> None:
    payload = {"x": 1, "items": [{"a": 2}]}
    path = tmp_path / "dump" / "results.json"
    write_json(path, payload)
    loaded = json.loads(path.read_text())
    assert loaded == payload


# --- Recommendation matrix --------------------------------------------


def test_recommend_picks_low_wer_stt() -> None:
    bad = STTRunResult(
        provider="bad",
        sample_count=0,
        overall=AggregateAccuracy(sample_count=0, wer_mean=0.5, cer_mean=0.3, wer_p95=0.6, cer_p95=0.4),
        latency=latency_stats([200.0]),
    )
    good = STTRunResult(
        provider="good",
        sample_count=0,
        overall=AggregateAccuracy(sample_count=0, wer_mean=0.05, cer_mean=0.02, wer_p95=0.1, cer_p95=0.04),
        latency=latency_stats([250.0]),
    )
    rec = recommend_providers([bad, good], tts_results=[])
    assert rec["stt_recommendation"] == "good"
    # Ranking is sorted ascending by score so 'good' is first
    assert rec["stt_ranking"][0][0] == "good"


def test_recommend_picks_lowest_latency_combo() -> None:
    matrix = LatencyMatrixResult(results=[
        LatencyRunResult(
            combo={"stt": "sarvam", "llm": "groq", "tts": "sarvam"},
            sample_count=5,
            stt=latency_stats([100.0]),
            llm_ttft=latency_stats([50.0]),
            llm_total=latency_stats([200.0]),
            tts_first_chunk=latency_stats([100.0]),
            tts_total=latency_stats([200.0]),
            end_to_end=latency_stats([500.0]),
        ),
        LatencyRunResult(
            combo={"stt": "sarvam", "llm": "gemini", "tts": "sarvam"},
            sample_count=5,
            stt=latency_stats([100.0]),
            llm_ttft=latency_stats([300.0]),
            llm_total=latency_stats([800.0]),
            tts_first_chunk=latency_stats([100.0]),
            tts_total=latency_stats([200.0]),
            end_to_end=latency_stats([1100.0]),
        ),
    ])
    rec = recommend_providers([], [], latency=matrix)
    assert rec["latency_combo_recommendation"]["llm"] == "groq"
    assert rec["latency_combo_ranking"][0][0]["llm"] == "groq"
