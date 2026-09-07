"""Tests for the Phase-2 TurnMetric -> Grafana Cloud Prometheus push job.

Mirrors tests/unit/test_turn_metrics_model.py's sqlite-in-memory sessionmaker
fixture, and tests/unit/test_logging_loki.py's respx-mocked-HTTP style for
the outbound push.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx
import pytest_asyncio
import respx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.turn_metrics import TurnMetric
from src.observability.turn_metrics_push import (
    _percentile,
    aggregate_and_push_turn_metrics,
)

PUSH_URL = "https://prometheus-prod-example.grafana.net/api/prom/push"
PUSH_ROUTE = f"{PUSH_URL}/metrics/job/vox_turn_metrics"


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()


async def _seed(sessionmaker, **overrides) -> None:
    defaults = dict(
        tenant_id="dev", session_id="call1", campaign_id=None, mode="voice",
        stt_provider="GroqSTTAdapter", llm_provider="GeminiLLMAdapter",
        tts_provider="SarvamTTSAdapter", action="continue",
        stt_latency_ms=0, llm_ttft_ms=0, llm_total_ms=0, tts_first_chunk_ms=0,
        tts_total_ms=0, total_latency_ms=0, tts_segments_dropped=0,
    )
    defaults.update(overrides)
    async with sessionmaker() as db:
        db.add(TurnMetric(**defaults))
        await db.commit()


class _ExplodingSessionmaker:
    """A sessionmaker stand-in that raises if ever invoked -- used to assert
    the aggregation function returns before touching the DB at all."""

    def __call__(self):
        raise AssertionError("sessionmaker must not be invoked when push_url is unset")


async def test_noop_when_push_url_unset() -> None:
    n = await aggregate_and_push_turn_metrics(_ExplodingSessionmaker(), None, None)
    assert n == 0


async def test_noop_when_push_url_empty_string() -> None:
    n = await aggregate_and_push_turn_metrics(_ExplodingSessionmaker(), "", None)
    assert n == 0


def test_percentile_nearest_rank() -> None:
    values = [100, 200, 300]
    assert _percentile(values, 50) == 200
    assert _percentile(values, 95) == 300
    assert _percentile([42], 50) == 42
    assert _percentile([42], 95) == 42


@respx.mock
async def test_aggregation_computes_count_and_percentiles_and_excludes_out_of_window_rows(
    sessionmaker,
) -> None:
    route = respx.put(PUSH_ROUTE).mock(return_value=httpx.Response(200))
    now = datetime.utcnow()

    # Group A (Groq/Gemini/Sarvam, voice): 3 rows inside the window.
    for i, v in enumerate([100, 200, 300]):
        await _seed(
            sessionmaker, session_id=f"a{i}",
            stt_latency_ms=v, llm_ttft_ms=v, llm_total_ms=v,
            tts_first_chunk_ms=v, tts_total_ms=v, total_latency_ms=v,
            created_at=now - timedelta(seconds=10 * (i + 1)),
        )
    # Group B: different llm_provider -> separate group, 1 row inside the window.
    await _seed(
        sessionmaker, session_id="b0", llm_provider="GroqLLMAdapter",
        stt_latency_ms=999, created_at=now - timedelta(seconds=5),
    )
    # Outside the window entirely -- must be excluded from both count and percentiles.
    await _seed(
        sessionmaker, session_id="old", stt_latency_ms=123456,
        created_at=now - timedelta(seconds=500),
    )

    n = await aggregate_and_push_turn_metrics(sessionmaker, PUSH_URL, "user:key", window_s=100)

    assert n == 4  # excludes the row outside the 100s window
    assert route.called
    body = route.calls.last.request.content.decode()

    # Group A: count=3, p50=200, p95=300 for every latency stage.
    assert (
        'vox_turn_metric_count{llm_provider="GeminiLLMAdapter",mode="voice",'
        'stt_provider="GroqSTTAdapter",tts_provider="SarvamTTSAdapter"} 3.0' in body
    )
    assert (
        'vox_turn_metric_latency_ms{llm_provider="GeminiLLMAdapter",mode="voice",'
        'quantile="p50",stage="stt_latency_ms",stt_provider="GroqSTTAdapter",'
        'tts_provider="SarvamTTSAdapter"} 200.0' in body
    )
    assert (
        'vox_turn_metric_latency_ms{llm_provider="GeminiLLMAdapter",mode="voice",'
        'quantile="p95",stage="stt_latency_ms",stt_provider="GroqSTTAdapter",'
        'tts_provider="SarvamTTSAdapter"} 300.0' in body
    )
    # Group B: count=1, p50==p95==999 (single-row group).
    assert (
        'vox_turn_metric_count{llm_provider="GroqLLMAdapter",mode="voice",'
        'stt_provider="GroqSTTAdapter",tts_provider="SarvamTTSAdapter"} 1.0' in body
    )
    assert (
        'vox_turn_metric_latency_ms{llm_provider="GroqLLMAdapter",mode="voice",'
        'quantile="p50",stage="stt_latency_ms",stt_provider="GroqSTTAdapter",'
        'tts_provider="SarvamTTSAdapter"} 999.0' in body
    )
    # The out-of-window row's distinctive value must not appear anywhere.
    assert "123456" not in body


@respx.mock
async def test_push_sends_basic_auth_and_puts_to_job_path(sessionmaker) -> None:
    route = respx.put(PUSH_ROUTE).mock(return_value=httpx.Response(200))
    await _seed(sessionmaker, stt_latency_ms=100)

    n = await aggregate_and_push_turn_metrics(sessionmaker, PUSH_URL, "grafana_user:api_key")

    assert n == 1
    assert route.called
    request = route.calls.last.request
    assert request.method == "PUT"
    assert request.headers["authorization"].startswith("Basic ")


@respx.mock
async def test_push_without_auth_sends_no_authorization_header(sessionmaker) -> None:
    route = respx.put(PUSH_ROUTE).mock(return_value=httpx.Response(200))
    await _seed(sessionmaker, stt_latency_ms=100)

    n = await aggregate_and_push_turn_metrics(sessionmaker, PUSH_URL, None)

    assert n == 1
    request = route.calls.last.request
    assert "authorization" not in {k.lower() for k in request.headers.keys()}


@respx.mock
async def test_no_rows_in_window_still_pushes_an_empty_registry(sessionmaker) -> None:
    """No data in the window must NOT skip the push. Prometheus Pushgateway
    retains the LAST pushed value for a job/group indefinitely until
    explicitly overwritten -- skipping here would leave the dashboard
    showing stale, no-longer-true counts/latencies once traffic stops
    (overnight, or a real outage). The fix: still push, with an empty
    registry -- a Pushgateway PUT replaces the job/group's entire prior
    state, so an empty push correctly clears the stale data."""
    route = respx.put(PUSH_ROUTE).mock(return_value=httpx.Response(200))
    # A row exists, but well outside the window -> aggregated count is 0.
    await _seed(
        sessionmaker, stt_latency_ms=1,
        created_at=datetime.utcnow() - timedelta(seconds=99999),
    )

    n = await aggregate_and_push_turn_metrics(sessionmaker, PUSH_URL, None, window_s=10)

    assert n == 0
    assert route.called  # the push DOES happen, unlike the old skip-on-empty behavior
    body = route.calls.last.request.content.decode()
    # No samples in the pushed body -- just metric HELP/TYPE headers, if any.
    assert "vox_turn_metric_count{" not in body
    assert "vox_turn_metric_latency_ms{" not in body


@respx.mock
async def test_push_http_failure_is_caught_and_does_not_raise(sessionmaker) -> None:
    respx.put(PUSH_ROUTE).mock(return_value=httpx.Response(500))
    await _seed(sessionmaker, stt_latency_ms=100)

    n = await aggregate_and_push_turn_metrics(sessionmaker, PUSH_URL, None)

    # Aggregation itself succeeded (1 row seen); only the push failed.
    assert n == 1


@respx.mock
async def test_push_network_error_is_caught_and_does_not_raise(sessionmaker) -> None:
    respx.put(PUSH_ROUTE).mock(side_effect=httpx.ConnectError("boom"))
    await _seed(sessionmaker, stt_latency_ms=100)

    n = await aggregate_and_push_turn_metrics(sessionmaker, PUSH_URL, None)

    assert n == 1  # must not raise


@respx.mock
async def test_push_failure_never_logs_embedded_url_credentials(sessionmaker, caplog) -> None:
    """httpx's raise_for_status() embeds the full request URL -- including any
    userinfo -- in its exception message (confirmed separately: a 401 against
    a 'https://user:pass@host/...' URL raises
    "Client error '401 Unauthorized' for url 'https://user:pass@host/...'").
    If GRAFANA_PROMETHEUS_PUSH_URL ever carried embedded basic-auth
    credentials, logging that exception (message or traceback) would leak
    them -- and since Phase 1's Loki shipping may be active, potentially ship
    them to an external log aggregator too. Assert the secret never reaches
    the log output at all."""
    secret_push_url = "https://vox_user:s3cr3t_pw@prometheus-prod-example.grafana.net/api/prom/push"
    secret_route = f"{secret_push_url}/metrics/job/vox_turn_metrics"
    respx.put(secret_route).mock(return_value=httpx.Response(401))
    await _seed(sessionmaker, stt_latency_ms=100)

    with caplog.at_level(logging.WARNING):
        n = await aggregate_and_push_turn_metrics(sessionmaker, secret_push_url, None)

    assert n == 1  # aggregation itself succeeded; only the push failed
    assert "s3cr3t_pw" not in caplog.text
    assert "vox_user" not in caplog.text


async def test_db_query_failure_is_caught_and_does_not_raise() -> None:
    class _BrokenSessionmaker:
        def __call__(self):
            raise RuntimeError("db unavailable")

    with respx.mock:
        # No route registered -- a network call here would also fail the test.
        n = await aggregate_and_push_turn_metrics(_BrokenSessionmaker(), PUSH_URL, None)
        assert n == 0
