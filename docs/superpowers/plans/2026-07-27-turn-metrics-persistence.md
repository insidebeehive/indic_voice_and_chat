# Turn-Metrics Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every voice turn's STT/LLM/TTS provider names and latency numbers get written to a durable Postgres table (not just stdout), so benchmarking data survives process restarts/redeploys and can be queried per provider-combo.

**Architecture:** A new `turn_metrics` table + `TurnMetric` model, written via a best-effort async helper (`record_turn_metric`) called from `VoiceBotAgent.apply_signal` alongside the existing `"voice turn metrics"` log line. `VoiceBotAgent` gets an optional `record_metric` callback (default `None`, backward compatible) wired at all 6 construction call sites. A new admin-only `GET /benchmarks/turn-metrics/summary` endpoint aggregates rows grouped by `(stt_provider, llm_provider, tts_provider)`.

**Tech Stack:** Python 3, FastAPI, SQLAlchemy 2.x async (`Mapped`/`mapped_column`), Alembic, pytest/pytest-asyncio, `httpx.AsyncClient` for route tests, `aiosqlite` in-memory DB for route tests (matches `tests/unit/test_crm_routes.py`).

## Global Constraints

- Must never add latency to a live call: the write is fire-and-forget from the caller's perspective (an `await`, but wrapped so failures never propagate) — it must never block or delay TTS audio delivery.
- Must never raise into the turn-handling path on a DB failure. Defense in depth: `record_turn_metric` itself catches and logs; the `VoiceBotAgent.apply_signal` call site also catches and logs (two independent layers, matching `src/agents/base.py`'s existing `persist_turn`/`persist_state` pattern).
- `record_metric` defaults to `None` on `VoiceBotAgent.__init__` — every existing caller/test that doesn't pass it must keep working unchanged.
- Follow existing conventions exactly: `Mapped`/`mapped_column` style from `src/models/benchmark.py`; Alembic revision/down_revision header style from `alembic/versions/0010_crm_kb_documents.py`; router/test patterns from `src/api/crms.py` / `tests/unit/test_crm_routes.py`.
- `tenants.id` is `String(50)` (confirmed in `src/models/tenant.py:35`) — the new table's `tenant_id` FK column must match that type exactly.
- `TurnMetrics` (dataclass, `src/pipeline/engine.py:165-172`) has exactly these int fields, defaulting to `0`: `stt_latency_ms`, `llm_ttft_ms`, `llm_total_ms`, `tts_first_chunk_ms`, `tts_total_ms`, `total_latency_ms`. Use these exact names for the new table's metric columns — do not invent shorter aliases.
- The existing `"voice turn metrics"` log line in `src/agents/voicebot.py` (~line 357) only fires inside `if agent_text:` inside `if metrics_dict:` — this is the exact point the new DB write is added alongside, not a new gate.
- Raw JSON endpoint only — no backoffice UI work in this plan.

---

### Task 1: `TurnMetric` model + `record_turn_metric` helper + migration

**Files:**
- Create: `src/models/turn_metrics.py`
- Create: `alembic/versions/0011_turn_metrics.py`
- Test: `tests/unit/test_turn_metrics_model.py`

**Interfaces:**
- Produces: `TurnMetric` (SQLAlchemy model, `src/models/turn_metrics.py`), table name `turn_metrics`.
- Produces: `async def record_turn_metric(*, tenant_id: str, session_id: str, campaign_id: str | None, mode: str, stt_provider: str | None, llm_provider: str, tts_provider: str | None, action: str, metrics: dict[str, int]) -> None` (`src/models/turn_metrics.py`). Opens its own session via `src.models.database.get_sessionmaker()`, inserts one row, commits. Catches all exceptions internally and logs a warning — never raises.

- [ ] **Step 1: Write the failing model/migration round-trip test**

Create `tests/unit/test_turn_metrics_model.py`:

```python
"""Round-trip test for the TurnMetric model + record_turn_metric helper."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.models.database import Base
from src.models.turn_metrics import TurnMetric, record_turn_metric


@pytest_asyncio.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()


async def test_record_turn_metric_inserts_row(sessionmaker, monkeypatch) -> None:
    monkeypatch.setattr("src.models.turn_metrics.get_sessionmaker", lambda: sessionmaker)

    await record_turn_metric(
        tenant_id="dev",
        session_id="call_abc123",
        campaign_id="bharat_matka",
        mode="layered",
        stt_provider="GroqSTTAdapter",
        llm_provider="GeminiLLMAdapter",
        tts_provider="SarvamTTSAdapter",
        action="continue",
        metrics={
            "stt_latency_ms": 300,
            "llm_ttft_ms": 1200,
            "llm_total_ms": 4000,
            "tts_first_chunk_ms": 2000,
            "tts_total_ms": 2500,
            "total_latency_ms": 4300,
        },
    )

    async with sessionmaker() as db:
        rows = (await db.execute(select(TurnMetric))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.tenant_id == "dev"
    assert row.session_id == "call_abc123"
    assert row.campaign_id == "bharat_matka"
    assert row.mode == "layered"
    assert row.stt_provider == "GroqSTTAdapter"
    assert row.llm_provider == "GeminiLLMAdapter"
    assert row.tts_provider == "SarvamTTSAdapter"
    assert row.action == "continue"
    assert row.stt_latency_ms == 300
    assert row.llm_ttft_ms == 1200
    assert row.llm_total_ms == 4000
    assert row.tts_first_chunk_ms == 2000
    assert row.tts_total_ms == 2500
    assert row.total_latency_ms == 4300
    assert row.created_at is not None


async def test_record_turn_metric_swallows_db_errors(monkeypatch) -> None:
    def _broken_sessionmaker():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("src.models.turn_metrics.get_sessionmaker", _broken_sessionmaker)

    # Must not raise.
    await record_turn_metric(
        tenant_id="dev",
        session_id="call_x",
        campaign_id=None,
        mode="layered",
        stt_provider=None,
        llm_provider="GeminiLLMAdapter",
        tts_provider=None,
        action="continue",
        metrics={
            "stt_latency_ms": 0,
            "llm_ttft_ms": 0,
            "llm_total_ms": 0,
            "tts_first_chunk_ms": 0,
            "tts_total_ms": 0,
            "total_latency_ms": 0,
        },
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_metrics_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.turn_metrics'`

- [ ] **Step 3: Create the model + helper**

Create `src/models/turn_metrics.py`:

```python
"""Per-turn STT/LLM/TTS latency metrics, persisted for cross-provider
benchmarking (PRD §7.7). Written best-effort from VoiceBotAgent.apply_signal
so a DB hiccup never affects a live call — see record_turn_metric.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.database import Base, get_sessionmaker

log = logging.getLogger(__name__)


class TurnMetric(Base):
    __tablename__ = "turn_metrics"
    __table_args__ = (
        Index("idx_turn_metrics_combo", "stt_provider", "llm_provider", "tts_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    session_id: Mapped[str] = mapped_column(String(100), nullable=False)
    campaign_id: Mapped[Optional[str]] = mapped_column(String(100))
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    stt_provider: Mapped[Optional[str]] = mapped_column(String(100))
    llm_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    tts_provider: Mapped[Optional[str]] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    stt_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_ttft_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tts_first_chunk_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tts_total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


async def record_turn_metric(
    *,
    tenant_id: str,
    session_id: str,
    campaign_id: Optional[str],
    mode: str,
    stt_provider: Optional[str],
    llm_provider: str,
    tts_provider: Optional[str],
    action: str,
    metrics: dict[str, int],
) -> None:
    """Insert one turn-metrics row. Best-effort: never raises — a DB outage
    must degrade to no-persistence, not break a live call (see
    VoiceBotAgent.apply_signal, the only caller)."""
    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as db:
            db.add(TurnMetric(
                tenant_id=tenant_id,
                session_id=session_id,
                campaign_id=campaign_id,
                mode=mode,
                stt_provider=stt_provider,
                llm_provider=llm_provider,
                tts_provider=tts_provider,
                action=action,
                stt_latency_ms=metrics.get("stt_latency_ms", 0),
                llm_ttft_ms=metrics.get("llm_ttft_ms", 0),
                llm_total_ms=metrics.get("llm_total_ms", 0),
                tts_first_chunk_ms=metrics.get("tts_first_chunk_ms", 0),
                tts_total_ms=metrics.get("tts_total_ms", 0),
                total_latency_ms=metrics.get("total_latency_ms", 0),
            ))
            await db.commit()
    except Exception:  # noqa: BLE001 - must never break a live call
        log.warning("record_turn_metric failed; continuing without persistence", exc_info=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_metrics_model.py -v`
Expected: 2 passed

- [ ] **Step 5: Write the Alembic migration**

Create `alembic/versions/0011_turn_metrics.py`. First check the actual current head:

Run: `.venv/bin/alembic heads`
Expected output: `0010_crm_kb_documents (head)` — use this exact value as `down_revision` below (if a different revision is reported, use that value instead; do not guess).

```python
"""alembic/versions/0011_turn_metrics.py

Per-turn STT/LLM/TTS latency metrics table, for cross-provider benchmarking.
Written by VoiceBotAgent.apply_signal (src/agents/voicebot.py) via
record_turn_metric (src/models/turn_metrics.py) on every completed layered
(cascade) turn.

Revision: 0011_turn_metrics
Down: 0010_crm_kb_documents
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_turn_metrics"
down_revision = "0010_crm_kb_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "turn_metrics",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(50), sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("campaign_id", sa.String(100), nullable=True),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("stt_provider", sa.String(100), nullable=True),
        sa.Column("llm_provider", sa.String(100), nullable=False),
        sa.Column("tts_provider", sa.String(100), nullable=True),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("stt_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_ttft_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_total_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tts_first_chunk_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tts_total_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_turn_metrics_tenant", "turn_metrics", ["tenant_id"])
    op.create_index(
        "idx_turn_metrics_combo", "turn_metrics",
        ["stt_provider", "llm_provider", "tts_provider"],
    )


def downgrade() -> None:
    op.drop_index("idx_turn_metrics_combo", table_name="turn_metrics")
    op.drop_index("idx_turn_metrics_tenant", table_name="turn_metrics")
    op.drop_table("turn_metrics")
```

- [ ] **Step 6: Verify the migration applies cleanly against the model**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_metrics_model.py -v` (still passes — this step is a sanity check that the migration's column set exactly matches the model; there is no separate migration-runner test harness in this repo, so parity is verified by inspection: diff the migration's `sa.Column(...)` list against the model's `mapped_column(...)` list field-by-field before moving on).

- [ ] **Step 7: Commit**

```bash
git add src/models/turn_metrics.py alembic/versions/0011_turn_metrics.py tests/unit/test_turn_metrics_model.py
git commit -m "feat(benchmarking): add turn_metrics table + record_turn_metric helper"
```

---

### Task 2: Wire `record_metric` into `VoiceBotAgent`

**Files:**
- Modify: `src/agents/voicebot.py:62-98` (`__init__`), `src/agents/voicebot.py:290-366` (`apply_signal`)
- Test: `tests/unit/test_voicebot_handle_turn_text.py` (append; this file already has the `apply_signal` metrics-logging tests from the prior turn-metrics-logging feature — follow its existing fixture/agent-construction pattern exactly)

**Interfaces:**
- Consumes: `TurnMetric`/`record_turn_metric` are NOT imported here — `VoiceBotAgent` never imports `src/models/turn_metrics.py` directly. It only knows about a generic callback shape, so it has zero coupling to SQLAlchemy/DB concerns. This keeps the agent testable without a DB.
- Produces: `VoiceBotAgent.__init__(..., record_metric: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None)`. The payload dict passed to `record_metric` has exactly these keys: `session_id`, `campaign_id`, `mode`, `stt_provider`, `llm_provider`, `tts_provider`, `action`, `metrics` (the raw `metrics_dict`). Task 3's closures at each construction site consume this exact shape.

- [ ] **Step 1: Read the existing test file's fixture pattern first**

Open `tests/unit/test_voicebot_handle_turn_text.py` and find the existing tests `test_apply_signal_logs_metrics_when_metrics_dict_present` and `test_apply_signal_does_not_log_metrics_when_absent` (added in commit `337707c`). Reuse their exact agent-construction helper/fixture for the new tests below — do not invent a second construction pattern in the same file.

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit/test_voicebot_handle_turn_text.py`:

```python
async def test_apply_signal_calls_record_metric_when_present(make_agent) -> None:
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    agent = make_agent(record_metric=_record_metric)  # use this file's existing agent factory
    metrics_dict = {
        "stt_latency_ms": 300, "llm_ttft_ms": 1200, "llm_total_ms": 4000,
        "tts_first_chunk_ms": 2000, "tts_total_ms": 2500, "total_latency_ms": 4300,
    }

    await agent.apply_signal(
        user_text="hi", agent_text="hello", action="continue",
        metrics_dict=metrics_dict,
    )

    assert len(calls) == 1
    payload = calls[0]
    assert payload["session_id"] == agent.session.session_id
    assert payload["mode"] == "layered"
    assert payload["action"] == "continue"
    assert payload["metrics"] == metrics_dict
    assert payload["llm_provider"]  # non-empty class name string


async def test_apply_signal_record_metric_failure_does_not_break_turn(make_agent) -> None:
    async def _record_metric(payload):
        raise RuntimeError("db unavailable")

    agent = make_agent(record_metric=_record_metric)
    metrics_dict = {
        "stt_latency_ms": 0, "llm_ttft_ms": 0, "llm_total_ms": 0,
        "tts_first_chunk_ms": 0, "tts_total_ms": 0, "total_latency_ms": 0,
    }

    # Must not raise.
    await agent.apply_signal(
        user_text="hi", agent_text="hello", action="continue",
        metrics_dict=metrics_dict,
    )


async def test_apply_signal_no_record_metric_is_a_no_op(make_agent) -> None:
    agent = make_agent(record_metric=None)
    metrics_dict = {
        "stt_latency_ms": 0, "llm_ttft_ms": 0, "llm_total_ms": 0,
        "tts_first_chunk_ms": 0, "tts_total_ms": 0, "total_latency_ms": 0,
    }
    # Must not raise (default behavior, unchanged from before this task).
    await agent.apply_signal(
        user_text="hi", agent_text="hello", action="continue",
        metrics_dict=metrics_dict,
    )
```

If this file's existing agent-construction fixture is not named `make_agent` or does not accept extra `VoiceBotAgent` constructor kwargs, adapt the three tests above to whatever the file's real fixture/helper name and calling convention is — read the file first (Step 1) and match it exactly rather than introducing a second construction path.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k record_metric`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'record_metric'`

- [ ] **Step 4: Add the constructor param**

In `src/agents/voicebot.py`, add to the imports at the top of the file:

```python
from typing import Any, Awaitable, Callable, Optional
```

(replacing the existing `from typing import Any, Optional` at line 18 — keep `Any` and `Optional`, add `Awaitable` and `Callable`.)

Modify `VoiceBotAgent.__init__` (currently lines 63-73) to add the new parameter:

```python
    def __init__(
        self,
        session: AgentSession,
        state_machine: AgentStateMachine,
        slot_schema: SlotSchema,
        script: VoiceBotScript,
        engine: PipelineEngine,
        store=None,
        extra_directives: Optional[list[str]] = None,
        kb_context: Optional[str] = None,
        record_metric: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
```

Add this line in the body, near the other `self._foo = foo` assignments (after `self._kb_context = kb_context`, currently line 85):

```python
        self._record_metric = record_metric
```

- [ ] **Step 5: Call it from `apply_signal`**

In `src/agents/voicebot.py`, inside `apply_signal`'s existing `if metrics_dict:` block (currently lines 349-364, right after the `log.info("voice turn metrics", ...)` call), add:

```python
                if self._record_metric is not None:
                    try:
                        await self._record_metric({
                            "session_id": self.session.session_id,
                            "campaign_id": self.session.campaign_id,
                            "mode": "layered",
                            "stt_provider": type(getattr(self._engine, "_stt", None)).__name__,
                            "llm_provider": type(getattr(self._engine, "_llm", None)).__name__,
                            "tts_provider": type(getattr(self._engine, "_tts", None)).__name__,
                            "action": action,
                            "metrics": metrics_dict,
                        })
                    except Exception:  # noqa: BLE001 - never break a live call on a metrics-write failure
                        log.warning(
                            "record_metric failed; continuing without persistence",
                            exc_info=True,
                        )
```

(This mirrors the existing `log.info("voice turn metrics", ...)` call immediately above it — same fields, same source values — so this is why `apply_signal` never needs an import of anything from `src/models/turn_metrics.py`: the field values are computed once already for the log line and duplicated here.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v`
Expected: all tests in the file pass (previously-passing tests unaffected, 3 new ones pass)

- [ ] **Step 7: Run the full unit suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented CLAUDE.md baseline, plus the new passing tests (the two known pre-existing failures — `test_chat_routes.py::test_claim_session_and_agent_ws` and `test_prompts.py::test_chatbot_prompt_has_scope_guardrails` — are expected and unrelated)

- [ ] **Step 8: Commit**

```bash
git add src/agents/voicebot.py tests/unit/test_voicebot_handle_turn_text.py
git commit -m "feat(voicebot): call an optional record_metric hook from apply_signal"
```

---

### Task 3: Wire `record_metric` at all 6 `VoiceBotAgent` construction sites

**Files:**
- Modify: `src/bootstrap.py` (4 sites: ~line 545, ~line 699, ~line 861, ~line 936)
- Modify: `src/api/dev_console.py` (2 sites: ~line 868, ~line 961)
- Test: existing test suites covering `bootstrap.py` and `dev_console.py` (run in Step 4 below; no new test file — this task is pure wiring, verified by "does the existing suite still pass" plus a manual grep-based check in Step 3)

**Interfaces:**
- Consumes: `record_turn_metric` from `src/models/turn_metrics.py` (Task 1) — `async def record_turn_metric(*, tenant_id, session_id, campaign_id, mode, stt_provider, llm_provider, tts_provider, action, metrics)`. Consumes `VoiceBotAgent.__init__`'s new `record_metric` param (Task 2).
- Produces: nothing new for later tasks — this is the last wiring point before the admin endpoint (Task 4), which reads from the same `turn_metrics` table independently.

Every one of the 6 sites already has a `tenant` object in local scope (confirmed by reading each site: `src/bootstrap.py` reads `tenant.settings.pipeline...` immediately above each `VoiceBotAgent(...)` call; `src/api/dev_console.py` reads `_crm_retriever_for(tenant, crm_retrievers)` immediately above each call). `tenant.id` is the `String(50)` tenant identifier.

- [ ] **Step 1: Add the import to both files**

In `src/bootstrap.py`, add near the top with the other `src.` imports:

```python
from src.models.turn_metrics import record_turn_metric
```

In `src/api/dev_console.py`, add near the top with the other `src.` imports:

```python
from src.models.turn_metrics import record_turn_metric
```

- [ ] **Step 2: Add `record_metric=` to each of the 4 `src/bootstrap.py` call sites**

For each of the 4 `VoiceBotAgent(...)` constructor calls in `src/bootstrap.py` (~lines 545, 699, 861, 936), add one new keyword argument, `record_metric=`, set to a lambda that closes over that site's own local `tenant` variable:

```python
            record_metric=lambda payload: record_turn_metric(tenant_id=tenant.id, **payload),
```

Add it as the last keyword argument in each call (after whatever the last existing kwarg is — `store=store` for the first site at ~545, `kb_context=...`/`extra_directives=...` etc. for the others — check each call's actual current last line before appending, since the exact kwarg list differs slightly per site).

- [ ] **Step 3: Add `record_metric=` to each of the 2 `src/api/dev_console.py` call sites**

Same pattern, at ~lines 868 and 961:

```python
            record_metric=lambda payload: record_turn_metric(tenant_id=tenant.id, **payload),
```

- [ ] **Step 4: Grep-verify all 6 sites are wired**

Run: `grep -n "record_metric=lambda" src/bootstrap.py src/api/dev_console.py`
Expected: 6 matching lines total (4 in `bootstrap.py`, 2 in `dev_console.py`)

- [ ] **Step 5: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented CLAUDE.md baseline (no new failures — this task only adds a new optional kwarg to existing calls, it doesn't change any existing test's expected behavior)

- [ ] **Step 6: Commit**

```bash
git add src/bootstrap.py src/api/dev_console.py
git commit -m "feat(voicebot): wire record_turn_metric at all VoiceBotAgent construction sites"
```

---

### Task 4: `GET /benchmarks/turn-metrics/summary` admin endpoint

**Files:**
- Modify: `src/api/benchmarks.py`
- Test: `tests/unit/test_benchmarks_turn_metrics_route.py`

**Interfaces:**
- Consumes: `TurnMetric` model (Task 1, `src/models/turn_metrics.py`). Consumes `get_db_session` (`src/api/deps.py`) and `require_admin` (already applied at the router level in `src/api/benchmarks.py` via `dependencies=[Depends(require_admin)]`).
- Produces: `GET /benchmarks/turn-metrics/summary` → `TurnMetricsSummaryResponse` (new Pydantic model, this file), a list of per-combo aggregates.

- [ ] **Step 1: Write the failing route test**

Create `tests/unit/test_benchmarks_turn_metrics_route.py`:

```python
"""Route test for the turn-metrics benchmarking summary endpoint."""

from __future__ import annotations

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import benchmarks
from src.api.deps import get_db_session
from src.auth.middleware import set_admin_tokens
from src.models.database import Base
from src.models.turn_metrics import TurnMetric

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)

    async def _seed():
        async with sm() as session:
            session.add_all([
                TurnMetric(
                    tenant_id="dev", session_id="c1", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="GeminiLLMAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=300, llm_ttft_ms=1200,
                    llm_total_ms=4000, tts_first_chunk_ms=2000, tts_total_ms=2500,
                    total_latency_ms=4300,
                ),
                TurnMetric(
                    tenant_id="dev", session_id="c1", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="GeminiLLMAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=280, llm_ttft_ms=1400,
                    llm_total_ms=4200, tts_first_chunk_ms=2100, tts_total_ms=2600,
                    total_latency_ms=4500,
                ),
                TurnMetric(
                    tenant_id="dev", session_id="c2", campaign_id="bharat_matka",
                    mode="layered", stt_provider="GroqSTTAdapter",
                    llm_provider="AnthropicClaudeAdapter", tts_provider="SarvamTTSAdapter",
                    action="continue", stt_latency_ms=300, llm_ttft_ms=0,
                    llm_total_ms=5000, tts_first_chunk_ms=1900, tts_total_ms=2400,
                    total_latency_ms=5300,
                ),
            ])
            await session.commit()

    await _seed()

    async def _session_override():
        async with sm() as session:
            yield session

    set_admin_tokens(["admin-token"])
    app = FastAPI()
    app.include_router(benchmarks.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_admin_tokens([])
    await engine.dispose()


async def test_summary_groups_by_combo(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    combos = {(e["stt_provider"], e["llm_provider"], e["tts_provider"]): e for e in body["entries"]}
    assert len(combos) == 2

    gemini_combo = combos[("GroqSTTAdapter", "GeminiLLMAdapter", "SarvamTTSAdapter")]
    assert gemini_combo["samples"] == 2
    assert gemini_combo["avg_total_latency_ms"] == 4400.0  # (4300 + 4500) / 2

    claude_combo = combos[("GroqSTTAdapter", "AnthropicClaudeAdapter", "SarvamTTSAdapter")]
    assert claude_combo["samples"] == 1
    assert claude_combo["avg_total_latency_ms"] == 5300.0


async def test_summary_requires_admin(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary")
    assert resp.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_benchmarks_turn_metrics_route.py -v`
Expected: FAIL — 404 (route doesn't exist yet)

- [ ] **Step 3: Add the endpoint**

In `src/api/benchmarks.py`, add these imports at the top (alongside the existing ones):

```python
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.models.turn_metrics import TurnMetric
```

Add these Pydantic models near the other response models (after `RunListResponse`, before the route functions):

```python
class TurnMetricsComboEntry(BaseModel):
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


class TurnMetricsSummaryResponse(BaseModel):
    entries: list[TurnMetricsComboEntry]
```

Add the route at the end of the file:

```python
@router.get("/turn-metrics/summary", response_model=TurnMetricsSummaryResponse)
async def turn_metrics_summary(
    db: AsyncSession = Depends(get_db_session),
) -> TurnMetricsSummaryResponse:
    stmt = (
        select(
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
        )
        .group_by(TurnMetric.stt_provider, TurnMetric.llm_provider, TurnMetric.tts_provider)
    )
    rows = (await db.execute(stmt)).all()
    entries = [
        TurnMetricsComboEntry(
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
        )
        for r in rows
    ]
    return TurnMetricsSummaryResponse(entries=entries)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_benchmarks_turn_metrics_route.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented CLAUDE.md baseline, plus all new tests from Tasks 1-4 passing

- [ ] **Step 6: Commit**

```bash
git add src/api/benchmarks.py tests/unit/test_benchmarks_turn_metrics_route.py
git commit -m "feat(benchmarking): add GET /benchmarks/turn-metrics/summary admin endpoint"
```

---

## Verification (after all 4 tasks)

- `.venv/bin/python -m pytest tests/unit -q` — full suite green apart from the 2 documented pre-existing failures.
- `.venv/bin/alembic upgrade head` against a real (or throwaway local) Postgres DB — confirms the migration actually applies, not just the SQLite-based unit tests.
- Manually run a dev-console turn locally (`VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --env-file .env`, open `/dev/voice`, complete one turn) and confirm a row appears in `turn_metrics` for that session.
- `curl` the new endpoint (mounted at `/api/v1/benchmarks/turn-metrics/summary`, not bare `/benchmarks/...` — the API is mounted under an `/api/v1` prefix in `src/main.py`) with a valid admin token and confirm it returns the seeded/real data grouped correctly.
