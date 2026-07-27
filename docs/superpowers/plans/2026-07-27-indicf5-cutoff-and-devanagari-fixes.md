# IndicF5 Mid-Sentence Cutoff + Devanagari-Purity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three compounding root causes behind IndicF5's "audio cuts short mid-sentence" bug and a related Devanagari-purity gap: (C) make silent TTS-segment drops visible in the benchmarking pipeline, (A) let campaign-specific proper nouns (agent/company names) render in Devanagari phonetics instead of raw Latin text, and (B) restructure the turn-timeout architecture so a slow/hung TTS call can no longer orphan a background task that keeps speaking after the turn has already ended.

**Architecture:** (C) adds a `tts_segments_dropped` counter threaded from the in-memory `TurnMetrics` dataclass through to the persisted `turn_metrics` table and the admin summary endpoint. (A) adds an optional `pronunciations` dict to campaign scripts, threaded through `TTSConfig.extra_pronunciations` into the existing (currently-unused) `extra=` parameter of `normalize_for_tts`. (B) replaces one shared 20-second deadline wrapping an entire turn with per-component budgets (a per-sentence TTS watchdog, an LLM-generation deadline, and a last-resort outer backstop that signals cancellation cooperatively instead of cancelling a task out from under an independently-running background worker).

**Tech Stack:** Python 3, asyncio, SQLAlchemy 2.x async, Alembic, pytest/pytest-asyncio.

## Global Constraints

- Branch is `stage` (already checked out on disk — do not create a new branch, do not switch branches; commits go directly onto it).
- Run `.venv/bin/python -m pytest tests/unit -q` after every task. Baseline in effect at the start of this plan: `24 failed, 1122 passed, 1 skipped, 22 errors` (pre-existing, unrelated per CLAUDE.md — do not chase them).
- Task order matters: Task 1 (fix C, the `tts_segments_dropped` counter) must land before Task 3 (fix B), which increments that counter. Task 2 (fix A) is independent and can run in any order relative to the others.
- User has already approved editing `src/dialogue/prompts.py` for Task 2 (a dataclass/schema field addition, not a change to any LLM prompt text).
- Do not change the deliberate per-sentence streaming strategy (`SentenceDetector`, first-chunk-soft behavior) — the caller must still hear the first sentence while the LLM is still generating the rest. Nothing in this plan touches sentence-splitting logic itself, only the timeout/cancellation and observability wrapped around it.

---

### Task 1: Silent TTS-segment-drop counter, end to end (fix C)

**Files:**
- Modify: `src/pipeline/engine.py` (add `tts_segments_dropped` field to `TurnMetrics`, ~line 166-172)
- Modify: `src/models/turn_metrics.py` (add DB column + `record_turn_metric` param)
- Create: `alembic/versions/0012_turn_metrics_dropped_segments.py`
- Modify: `src/api/benchmarks.py` (add the field to the summary response + query)
- Test: `tests/unit/test_turn_metrics_model.py`, `tests/unit/test_benchmarks_turn_metrics_route.py`

**Interfaces:**
- Produces: `TurnMetrics.tts_segments_dropped: int = 0` (new field, `src/pipeline/engine.py`). Task 3 increments this field directly on the `metrics` object already in scope inside `run_turn_text`'s `tts_worker` closure — no new function needed, just a field that must exist first.
- Produces: `TurnMetric.tts_segments_dropped` DB column and `record_turn_metric(..., metrics: dict)`'s handling of that key (`src/models/turn_metrics.py`) — reads `metrics.get("tts_segments_dropped", 0)`, same pattern as the other 6 metric keys.
- Consumes: nothing new — `VoiceBotAgent.apply_signal` (`src/agents/voicebot.py`) already forwards the ENTIRE `metrics_dict = pipeline_result.metrics.__dict__` object generically to both the `"voice turn metrics"` log line and `record_metric`'s payload (confirmed by reading the current code — the payload's `"metrics": metrics_dict` key is a passthrough of the whole dataclass' `__dict__`). Adding a new field to `TurnMetrics` requires ZERO changes to `voicebot.py` itself — it flows through automatically.

- [ ] **Step 1: Write the failing test for the model + helper**

Open `tests/unit/test_turn_metrics_model.py`. It currently has two tests:
`test_record_turn_metric_inserts_row` and `test_record_turn_metric_swallows_db_errors`,
each passing a 6-key `metrics` dict. Update BOTH call sites' `metrics={...}` dict to add
`"tts_segments_dropped": 1,` (first test) / `"tts_segments_dropped": 0,` (second test) as a
new line inside the existing dict literal, and add a new assertion line
`assert row.tts_segments_dropped == 1` right after `assert row.total_latency_ms == 4300` in
`test_record_turn_metric_inserts_row`.

The full updated first test should read:

```python
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
            "tts_segments_dropped": 1,
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
    assert row.tts_segments_dropped == 1
    assert row.created_at is not None
```

And update the second test's `metrics={...}` dict to add `"tts_segments_dropped": 0,` after
`"total_latency_ms": 0,` (no new assertions needed there — it only checks that errors are
swallowed).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_metrics_model.py -v`
Expected: FAIL — `AttributeError: 'TurnMetric' object has no attribute 'tts_segments_dropped'`

- [ ] **Step 3: Add the field to `TurnMetrics`**

In `src/pipeline/engine.py`, find the `TurnMetrics` dataclass (currently):

```python
@dataclass
class TurnMetrics:
    stt_latency_ms: int = 0
    llm_ttft_ms: int = 0
    llm_total_ms: int = 0
    tts_first_chunk_ms: int = 0
    tts_total_ms: int = 0
    total_latency_ms: int = 0
```

Add one field at the end:

```python
@dataclass
class TurnMetrics:
    stt_latency_ms: int = 0
    llm_ttft_ms: int = 0
    llm_total_ms: int = 0
    tts_first_chunk_ms: int = 0
    tts_total_ms: int = 0
    total_latency_ms: int = 0
    tts_segments_dropped: int = 0
```

- [ ] **Step 4: Add the DB column + helper param**

In `src/models/turn_metrics.py`, add a column to `TurnMetric` (after `total_latency_ms`, before `created_at`):

```python
    tts_segments_dropped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
```

And in `record_turn_metric`, add the field to the `TurnMetric(...)` constructor call (after `total_latency_ms=metrics.get("total_latency_ms", 0),`):

```python
                tts_segments_dropped=metrics.get("tts_segments_dropped", 0),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_turn_metrics_model.py -v`
Expected: 2 passed

- [ ] **Step 6: Write the Alembic migration**

Run: `.venv/bin/alembic heads`
Expected output: `0011_turn_metrics (head)` — use this exact value as `down_revision` below (if a
different revision is reported, use that value instead; do not guess).

Create `alembic/versions/0012_turn_metrics_dropped_segments.py`:

```python
"""alembic/versions/0012_turn_metrics_dropped_segments.py

Add turn_metrics.tts_segments_dropped — a count of TTS sentences that
failed/timed out and were skipped during a turn, silently before this
change (a log line nobody greps for). Written by
VoiceBotAgent.apply_signal via record_turn_metric, same as every other
turn_metrics column.

Revision: 0012_turn_metrics_dropped_segments
Down: 0011_turn_metrics
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_turn_metrics_dropped_segments"
down_revision = "0011_turn_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "turn_metrics",
        sa.Column("tts_segments_dropped", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("turn_metrics", "tts_segments_dropped")
```

- [ ] **Step 7: Write the failing test for the admin endpoint**

Open `tests/unit/test_benchmarks_turn_metrics_route.py`. Add `tts_segments_dropped=0,` (or a
non-zero value for at least one seeded row, to make the aggregation meaningful) to each of the
3 `TurnMetric(...)` constructor calls in the `client` fixture — e.g. the first seeded row gets
`tts_segments_dropped=1,`, the other two get `tts_segments_dropped=0,`. Add this as a new
kwarg line right after each `total_latency_ms=...,` line in the fixture.

Then add one new assertion to `test_summary_groups_by_combo`, right after the existing
`assert gemini_combo["avg_total_latency_ms"] == 4400.0  # (4300 + 4500) / 2` line:

```python
    assert gemini_combo["avg_tts_segments_dropped"] == 0.5  # (1 + 0) / 2
```

And update `test_record_turn_metric_then_summary_e2e`'s `payload["metrics"]` dict to add
`"tts_segments_dropped": 2,` after `"total_latency_ms": 3300,`, and add a new assertion right
after `assert entry["avg_total_latency_ms"] == 3300.0`:

```python
    assert entry["avg_tts_segments_dropped"] == 2.0
```

- [ ] **Step 8: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_benchmarks_turn_metrics_route.py -v`
Expected: FAIL — `KeyError: 'avg_tts_segments_dropped'` (the response JSON doesn't have this
field yet)

- [ ] **Step 9: Add the field to the admin endpoint**

In `src/api/benchmarks.py`, add a field to `TurnMetricsComboEntry` (after `avg_total_latency_ms: float`):

```python
    avg_tts_segments_dropped: float
```

Add a column to the `select(...)` statement inside `turn_metrics_summary` (after the
`func.avg(TurnMetric.total_latency_ms).label("avg_total_latency_ms"),` line):

```python
            func.avg(TurnMetric.tts_segments_dropped).label("avg_tts_segments_dropped"),
```

And add a field to the `TurnMetricsComboEntry(...)` construction inside the list comprehension
(after `avg_total_latency_ms=float(r.avg_total_latency_ms or 0.0),`):

```python
            avg_tts_segments_dropped=float(r.avg_tts_segments_dropped or 0.0),
```

- [ ] **Step 10: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_benchmarks_turn_metrics_route.py -v`
Expected: 3 passed

- [ ] **Step 11: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented baseline, plus all new/updated tests passing
(no reduction in test count — this task only adds assertions/columns, doesn't remove any)

- [ ] **Step 12: Commit**

```bash
git add src/pipeline/engine.py src/models/turn_metrics.py alembic/versions/0012_turn_metrics_dropped_segments.py src/api/benchmarks.py tests/unit/test_turn_metrics_model.py tests/unit/test_benchmarks_turn_metrics_route.py
git commit -m "feat(benchmarking): add tts_segments_dropped counter end to end"
```

---

### Task 2: Devanagari pronunciation plumbing for campaign proper nouns (fix A)

**Files:**
- Modify: `src/dialogue/prompts.py` (add `pronunciations` field to `VoiceBotScript` + parse it in `from_campaign_yaml`)
- Modify: `src/interfaces/tts.py` (add `extra_pronunciations` to `TTSConfig`)
- Modify: `src/providers/tts/indicf5.py`, `src/providers/tts/sarvam.py` (pass it through to `normalize_for_tts`)
- Modify: `src/agents/voicebot.py` (thread `script.pronunciations` into both the opening-line TTS config and the per-turn engine config)
- Modify: `src/pipeline/engine.py` (`PipelineEngine` needs a way to carry `extra_pronunciations` on its configured `tts` config)
- Test: `tests/unit/test_text_normalize.py` (or wherever `apply_pronunciations`/`normalize_for_tts` tests currently live — check first), `tests/unit/test_voicebot_handle_turn_text.py` or a new focused test file for the threading

**Interfaces:**
- Produces: `VoiceBotScript.pronunciations: dict[str, str]` (default `{}`), populated by `from_campaign_yaml` from the campaign's `script.pronunciations` YAML key (if present).
- Produces: `TTSConfig.extra_pronunciations: Optional[dict[str, str]] = None` (`src/interfaces/tts.py`).
- Consumes: `apply_pronunciations(text, extra=...)` / `normalize_for_tts(text, language, extra=...)` — both already exist in `src/pipeline/text_normalize.py` and already merge `extra` into the lookup table (`table = {**DEFAULT_PRONUNCIATIONS, **(extra or {})}`, confirmed by reading the file). This task does NOT modify `text_normalize.py` — it only starts calling the already-existing `extra=` parameter, which today is never passed by any caller.

- [ ] **Step 1: Find where `apply_pronunciations`/`normalize_for_tts` are currently tested**

Run: `grep -rn "apply_pronunciations\|normalize_for_tts" tests/unit/`

Read whichever test file that finds (it exercises `src/pipeline/text_normalize.py`). Match its
existing style for the new test in Step 2.

- [ ] **Step 2: Write a failing test for `normalize_for_tts`'s `extra` passthrough being used with campaign-style keys**

This function already supports `extra` (confirmed: `text_normalize.py`'s `apply_pronunciations`
merges `extra` into its lookup table). The behavior this task adds is CAMPAIGN DATA reaching
that parameter, not new logic in `text_normalize.py` itself. So the test to add here is NOT
about `text_normalize.py` (it already has coverage for `extra` presumably, or if it doesn't,
that's a pre-existing gap outside this task's scope — check via the grep in Step 1 before
deciding whether a `text_normalize.py` test is even missing; if `apply_pronunciations` already
has a test exercising `extra`, skip straight to Step 3's `VoiceBotScript` test instead of
duplicating coverage).

Add this test to `tests/unit/test_prompts.py` (or wherever `VoiceBotScript`/`from_campaign_yaml`
is currently tested — run `grep -rn "from_campaign_yaml" tests/unit/` first and match that
file's existing style/fixtures):

```python
def test_from_campaign_yaml_parses_pronunciations():
    script = VoiceBotScript.from_campaign_yaml({
        "agent_name": "Priya",
        "company_name": "XYZ",
        "pronunciations": {"Anaaya": "अनाया", "XYZ": "एक्स वाय ज़ेड"},
    })
    assert script.pronunciations == {"Anaaya": "अनाया", "XYZ": "एक्स वाय ज़ेड"}


def test_from_campaign_yaml_pronunciations_defaults_empty():
    script = VoiceBotScript.from_campaign_yaml({"agent_name": "Priya", "company_name": "XYZ"})
    assert script.pronunciations == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts.py -v -k pronunciations` (adjust the
path to whatever file Step 2 actually added to)
Expected: FAIL — `TypeError: from_campaign_yaml() got an unexpected keyword argument
'pronunciations'` or `AttributeError: 'VoiceBotScript' object has no attribute
'pronunciations'`

- [ ] **Step 4: Add the field to `VoiceBotScript` + parse it**

In `src/dialogue/prompts.py`, add a field to the `VoiceBotScript` dataclass (after `max_turns: int = 0`):

```python
    pronunciations: dict[str, str] = field(default_factory=dict)
```

And in `from_campaign_yaml` (the `return cls(...)` call), add one line after
`max_turns=int(script.get("max_turns") or 0),`:

```python
            pronunciations=dict(script.get("pronunciations") or {}),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts.py -v -k pronunciations`
Expected: 2 passed

- [ ] **Step 6: Add `extra_pronunciations` to `TTSConfig`**

In `src/interfaces/tts.py`, add a field to `TTSConfig` (after `sample_rate: int = 16000`):

```python
    extra_pronunciations: Optional[dict[str, str]] = None
```

- [ ] **Step 7: Write a failing test for the adapters passing `extra_pronunciations` through**

Both `IndicF5TTSAdapter` and `SarvamTTSAdapter` are tested in ONE shared file,
`tests/unit/test_tts_adapters.py`, which mocks the HTTP layer with `respx` (not a hand-rolled
fake client) and already has an `indicf5` fixture, an `adapter` (Sarvam) fixture, a `_wav()`
helper, and `_INDICF5_URL`/`SARVAM_BASE_URL` constants — reuse all of these exactly, do not
invent new ones. Add these two tests to that file (place them near the other
`test_indicf5_*`/Sarvam tests, respecting the file's existing `# --- IndicF5 ---` section
comment):

```python
@pytest.mark.asyncio
@respx.mock
async def test_indicf5_passes_extra_pronunciations_to_normalize(
    indicf5: IndicF5TTSAdapter, monkeypatch,
) -> None:
    captured = {}

    def _fake_normalize(text, language=None, extra=None):
        captured["extra"] = extra
        return text

    monkeypatch.setattr("src.providers.tts.indicf5.normalize_for_tts", _fake_normalize)
    pcm = b"\x01\x02" * 100
    respx.post(f"{_INDICF5_URL}/tts").mock(return_value=Response(200, content=_wav(pcm, 16000)))

    config = TTSConfig(language="hi-IN", extra_pronunciations={"XYZ": "एक्स वाय ज़ेड"})
    await indicf5.synthesize("hello XYZ", config)

    assert captured["extra"] == {"XYZ": "एक्स वाय ज़ेड"}


@pytest.mark.asyncio
@respx.mock
async def test_sarvam_passes_extra_pronunciations_to_normalize(
    adapter: SarvamTTSAdapter, monkeypatch,
) -> None:
    captured = {}

    def _fake_normalize(text, language=None, extra=None):
        captured["extra"] = extra
        return text

    monkeypatch.setattr("src.providers.tts.sarvam.normalize_for_tts", _fake_normalize)
    pcm = b"\x01\x02\x03\x04" * 1000
    respx.post(f"{SARVAM_BASE_URL}/text-to-speech").mock(
        return_value=Response(200, json={"audios": [base64.b64encode(pcm).decode()]})
    )

    config = TTSConfig(language="hi-IN", extra_pronunciations={"XYZ": "एक्स वाय ज़ेड"})
    await adapter.synthesize("hello XYZ", config)

    assert captured["extra"] == {"XYZ": "एक्स वाय ज़ेड"}
```

- [ ] **Step 8: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_tts_adapters.py -v -k extra_pronunciations`
Expected: FAIL — `assert None == {"XYZ": "एक्स वाय ज़ेड"}` (today's call site doesn't pass `extra=`
at all, so the fake receives `extra=None`, its default)

- [ ] **Step 9: Wire the adapters**

In `src/providers/tts/indicf5.py`, change (inside `synthesize`):

```python
        text = normalize_for_tts(text, config.language)
```

to:

```python
        text = normalize_for_tts(text, config.language, extra=config.extra_pronunciations)
```

In `src/providers/tts/sarvam.py`, change (inside `synthesize`):

```python
        text = normalize_for_tts(text, config.language)
```

to:

```python
        text = normalize_for_tts(text, config.language, extra=config.extra_pronunciations)
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_tts_adapters.py -v -k extra_pronunciations`
Expected: both new tests pass (2 passed)

- [ ] **Step 11: Write a failing test for `play_opening` threading `script.pronunciations`**

Add to `tests/unit/test_voicebot_handle_turn_text.py` (reuse this file's existing
`_agent_with_slots`/`_agent` helper pattern — read the top of the file for the exact
construction signature before writing):

```python
@pytest.mark.asyncio
async def test_play_opening_passes_script_pronunciations_to_tts():
    from src.interfaces.tts import TTSConfig

    captured = {}

    class _CapturingTTS:
        async def synthesize(self, text, config):
            captured["extra_pronunciations"] = config.extra_pronunciations
            return TTSResult(audio=b"\x00\x00", duration_ms=1.0, sample_rate=16000)

    engine = PipelineEngine(
        stt=None, llm=None, tts=_CapturingTTS(),
        config=PipelineConfig(
            stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(language="hi-IN"),
        ),
    )
    script = VoiceBotScript(
        agent_name="Priya", agent_role="sales", company_name="XYZ",
        opening="Hello from {company_name}",
        pronunciations={"XYZ": "एक्स वाय ज़ेड"},
    )
    agent = VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(), slot_schema=SlotSchema(),
        script=script, engine=engine, store=None,
    )

    async def sink(audio: bytes):
        pass

    await agent.play_opening(sink)
    assert captured["extra_pronunciations"] == {"XYZ": "एक्स वाय ज़ेड"}
```

Add the necessary imports at the top of the test file if not already present:
`from src.interfaces.tts import TTSResult`, `from src.interfaces.llm import LLMConfig`,
`from src.interfaces.stt import STTConfig`, `from src.pipeline.engine import PipelineConfig,
PipelineEngine`. Check which of these are already imported before adding duplicates.

- [ ] **Step 12: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k pronunciations`
Expected: FAIL — `assert None == {"XYZ": "एक्स वाय ज़ेड"}` (today `play_opening` builds
`opening_tts` via `_replace(base_tts, language=opening_lang)`, which carries over whatever
`extra_pronunciations` `base_tts` already had — i.e. `None`, since nothing sets it yet)

- [ ] **Step 13: Wire `play_opening`**

In `src/agents/voicebot.py`, inside `play_opening` (currently ~line 160-168), change:

```python
        opening_tts = (
            _replace(base_tts, language=opening_lang)
            if base_tts is not None else _TTSConfig(language=opening_lang)
        )
```

to:

```python
        opening_tts = (
            _replace(
                base_tts, language=opening_lang,
                extra_pronunciations=self._script.pronunciations or None,
            )
            if base_tts is not None else
            _TTSConfig(language=opening_lang, extra_pronunciations=self._script.pronunciations or None)
        )
```

- [ ] **Step 14: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k pronunciations`
Expected: PASS (the opening-line test from Step 11)

- [ ] **Step 15: Write a failing test for the per-turn (non-opening) path**

The per-turn TTS config comes from `PipelineEngine._config.tts` (a plain `PipelineConfig`
set once at construction, `engine.py:199-209`), overridden per-call only for `language`
(`replace(self._config.tts, language=language) if language else self._config.tts`,
`engine.py:290`). The cleanest way to make campaign pronunciations reach that config is for
`VoiceBotAgent.__init__` to patch the engine's `_config.tts` once at construction time —
`VoiceBotAgent` already reaches into `self._engine`'s private attributes elsewhere in this
file (e.g. `self._engine._tts` in `play_opening`), so this matches an existing, accepted
pattern in this codebase, not a new violation. This test asserts on that construction-time
wiring directly (inspecting `engine._config.tts.extra_pronunciations` after constructing a
real `VoiceBotAgent`), rather than routing through a fake engine's internals. Add to the same
test file:

```python
@pytest.mark.asyncio
async def test_voicebot_agent_threads_script_pronunciations_onto_engine_config():
    engine = PipelineEngine(
        stt=None, llm=None, tts=None,
        config=PipelineConfig(
            stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(language="hi-IN"),
        ),
    )
    script = VoiceBotScript(
        agent_name="Priya", agent_role="sales", company_name="XYZ",
        pronunciations={"XYZ": "एक्स वाय ज़ेड"},
    )
    VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(), slot_schema=SlotSchema(),
        script=script, engine=engine, store=None,
    )
    assert engine._config.tts.extra_pronunciations == {"XYZ": "एक्स वाय ज़ेड"}


@pytest.mark.asyncio
async def test_voicebot_agent_leaves_engine_config_alone_when_no_pronunciations():
    original_tts = TTSConfig(language="hi-IN")
    engine = PipelineEngine(
        stt=None, llm=None, tts=None,
        config=PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=original_tts),
    )
    script = VoiceBotScript(agent_name="Priya", agent_role="sales", company_name="XYZ")
    VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(), slot_schema=SlotSchema(),
        script=script, engine=engine, store=None,
    )
    assert engine._config.tts is original_tts
```

- [ ] **Step 16: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k engine_config`
Expected: FAIL — `assert None == {"XYZ": "एक्स वाय ज़ेड"}`

- [ ] **Step 17: Wire `VoiceBotAgent.__init__`**

In `src/agents/voicebot.py`, in `VoiceBotAgent.__init__`, find the line `self._engine = engine`
(currently ~line 83) and add immediately after it:

```python
        self._engine = engine
        if script.pronunciations:
            from dataclasses import replace as _replace_cfg
            engine._config = _replace_cfg(
                engine._config,
                tts=_replace_cfg(engine._config.tts, extra_pronunciations=script.pronunciations),
            )
```

- [ ] **Step 18: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k "engine_config or pronunciations"`
Expected: all pass

- [ ] **Step 19: Add the live data fix for the dev-console campaign**

This step is a DATA change, not a code change — flag it to the user rather than executing it
yourself as part of this task: once the plumbing above is merged, the campaign's
`config_yaml.script.pronunciations` needs entries like `{"Anaaya": "अनाया", "XYZ": "एक्स वाय
ज़ेड"}` added via the backoffice/DB for the actual live campaign this bug was reported against.
Do not guess at the transliteration yourself — note in your task report that this data step is
outstanding and needs the user's confirmation of the actual desired pronunciation before
anyone applies it.

- [ ] **Step 20: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented baseline, plus all new tests from this task
passing, no reduction in test count.

- [ ] **Step 21: Commit**

```bash
git add src/dialogue/prompts.py src/interfaces/tts.py src/providers/tts/indicf5.py src/providers/tts/sarvam.py src/agents/voicebot.py tests/unit/
git commit -m "feat(tts): thread campaign-specific pronunciations into TTS synthesis"
```

---

### Task 3: Turn-timeout / cancellation architecture restructure (fix B)

**Files:**
- Modify: `src/pipeline/engine.py` (per-sentence TTS watchdog, LLM-generation deadline, consecutive-failure abort)
- Modify: `src/agents/voicebot.py` (replace the shared outer `asyncio.wait_for` with a cooperative-cancellation backstop in both `handle_turn` and `handle_turn_text`)
- Modify: `src/providers/tts/sarvam.py` (update the stale comment referencing the old `TURN_TIMEOUT_S`)
- Test: `tests/unit/test_engine_run_turn_text.py`, `tests/unit/test_voicebot_handle_turn_text.py`

**Interfaces:**
- Consumes: `TurnMetrics.tts_segments_dropped` (Task 1) — incremented directly by `tts_worker` inside `run_turn_text`.
- Produces: `VoiceBotAgent._run_turn_with_backstop(self, coro, cancel_event: asyncio.Event) -> TurnResult` (new private method) — used by both `handle_turn` and `handle_turn_text`, nothing outside this class needs it.
- Produces: `src/pipeline/engine.py` module-level constants `LLM_TURN_TIMEOUT_S = 20.0`, `TTS_SENTENCE_TIMEOUT_S = 25.0`, `MAX_CONSECUTIVE_TTS_FAILURES = 2`.
- Produces: `src/agents/voicebot.py` module-level constants `HARD_TURN_TIMEOUT_S = 90.0`, `_BACKSTOP_GRACE_S = 30.0` — REPLACING the existing `TURN_TIMEOUT_S = 20.0` (delete that name; both existing call sites and the existing test that monkeypatches it must be updated, not left referencing a name that no longer exists).

- [ ] **Step 1: Write a failing test for the per-sentence TTS watchdog dropping a slow segment**

Open `tests/unit/test_engine_run_turn_text.py`. Read the existing `_FakeLLM`/`_FakeTTS`/`_engine()`
helper (already shown above in this plan's own research — `_FakeLLM.generate_stream` yields
3 tokens that assemble into one JSON envelope with response text `"नमस्ते जी।"`; `_FakeTTS`
always returns a fixed `TTSResult`). Add a new fake TTS that hangs on one call:

```python
class _SlowThenFastTTS:
    """First call sleeps past TTS_SENTENCE_TIMEOUT_S; later calls return instantly."""

    def __init__(self):
        self.calls = 0

    async def synthesize(self, text, config):
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(10)  # never returns within the (monkeypatched-small) timeout
            raise AssertionError("should have been cancelled by the per-sentence watchdog")
        return TTSResult(audio=b"\x00\x00" * 80, duration_ms=10.0, sample_rate=16000)


@pytest.mark.asyncio
async def test_run_turn_text_drops_one_slow_sentence_via_watchdog(monkeypatch):
    import src.pipeline.engine as engine_mod
    monkeypatch.setattr(engine_mod, "TTS_SENTENCE_TIMEOUT_S", 0.05)

    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    tts = _SlowThenFastTTS()
    engine = PipelineEngine(_FakeSTT(), _FakeLLM(), tts, cfg)
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?", history=[], audio_sink=sink,
    )
    assert result.metrics.tts_segments_dropped == 1
    assert result.cancelled is False  # one dropped sentence, not enough to abort the whole turn
```

Note: `_FakeLLM.generate_stream` yields only ONE sentence's worth of text (the whole
`'{"response_text": "नमस्ते जी।", "action": "continue"}'` envelope), so `tts.calls` will only
ever reach 1 in this specific fake — meaning this test only exercises the single-slow-sentence
path, not a multi-sentence sequence. This is intentional and sufficient for this step (Step 5
below covers the consecutive-failure/multi-sentence case with its own fake).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py -v -k drops_one_slow`
Expected: FAIL — `AttributeError: module 'src.pipeline.engine' has no attribute
'TTS_SENTENCE_TIMEOUT_S'`

- [ ] **Step 3: Add the constants and the LLM-generation deadline check**

In `src/pipeline/engine.py`, add these module-level constants near the top, right after the
existing imports and before the first dataclass (or alongside any other existing module-level
constants — check the file's current top section for where similar constants already live):

```python
LLM_TURN_TIMEOUT_S = 20.0  # legitimate end-to-end LLM-generation budget for one turn
TTS_SENTENCE_TIMEOUT_S = 25.0  # per-sentence rolling watchdog — covers IndicF5's 10s x 2
                                # attempts (indicf5.py's _DEFAULT_TIMEOUT_S=10.0,
                                # _TTS_ATTEMPTS=2) plus margin
MAX_CONSECUTIVE_TTS_FAILURES = 2
```

In `run_turn_text`, find the LLM token loop (currently):

```python
        try:
            async for token in self._llm.generate_stream(messages, self._config.llm):
                if cancel_event.is_set():
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
```

Add a deadline check right after the existing `cancel_event.is_set()` check:

```python
        try:
            async for token in self._llm.generate_stream(messages, self._config.llm):
                if cancel_event.is_set():
                    break
                if time.perf_counter() - t_llm_start > LLM_TURN_TIMEOUT_S:
                    log.error(
                        "LLM generation exceeded %.0fs budget; ending turn early",
                        LLM_TURN_TIMEOUT_S,
                    )
                    cancel_event.set()
                    break
                if first_token_at is None:
                    first_token_at = time.perf_counter()
```

- [ ] **Step 4: Add the per-sentence watchdog + drop-counter to `tts_worker`**

In `src/pipeline/engine.py`, find `tts_worker` (currently):

```python
        async def tts_worker() -> None:
            nonlocal first_audio_at, bytes_sent
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    return
                if cancel_event.is_set():
                    continue
                try:
                    result = await self._tts.synthesize(sentence, tts_cfg)
                except Exception as _tts_err:  # noqa: BLE001
                    log.error("TTS synthesize failed: %s", _tts_err)
                    continue
                if cancel_event.is_set():
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                bytes_sent += len(result.audio)
                sentences_spoken.append(sentence)
                await audio_sink(result.audio)
```

Replace it with:

```python
        async def tts_worker() -> None:
            nonlocal first_audio_at, bytes_sent
            consecutive_failures = 0
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    return
                if cancel_event.is_set():
                    continue
                try:
                    result = await asyncio.wait_for(
                        self._tts.synthesize(sentence, tts_cfg),
                        timeout=TTS_SENTENCE_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    log.error(
                        "TTS synthesize timed out after %.0fs: %r",
                        TTS_SENTENCE_TIMEOUT_S, sentence[:60],
                    )
                    metrics.tts_segments_dropped += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_TTS_FAILURES:
                        log.error(
                            "aborting turn: %d consecutive TTS failures",
                            consecutive_failures,
                        )
                        cancel_event.set()
                    continue
                except Exception as _tts_err:  # noqa: BLE001
                    log.error("TTS synthesize failed: %s", _tts_err)
                    metrics.tts_segments_dropped += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_TTS_FAILURES:
                        log.error(
                            "aborting turn: %d consecutive TTS failures",
                            consecutive_failures,
                        )
                        cancel_event.set()
                    continue
                consecutive_failures = 0
                if cancel_event.is_set():
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter()
                bytes_sent += len(result.audio)
                sentences_spoken.append(sentence)
                await audio_sink(result.audio)
```

`metrics` is `run_turn_text`'s own local variable (assigned earlier in the function via
`metrics = TurnMetrics()`); mutating `metrics.tts_segments_dropped += 1` from this nested
closure does not need a `nonlocal metrics` declaration — only reassigning the name `metrics`
itself would need that, and this code never does.

- [ ] **Step 5: Run the Step 1 test to verify it passes, then add + run the consecutive-failure test**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py -v -k drops_one_slow`
Expected: PASS

Add a second new test to the same file, using a fake LLM that yields TWO separate sentences
(so the consecutive-failure counter actually gets exercised across more than one TTS call):

```python
class _TwoSentenceLLM:
    async def generate_stream(self, messages, config):
        for tok in [
            '{"response_text": "पहला वाक्य। दूसरा वाक्य।", "action": "continue"}',
        ]:
            yield tok


class _AlwaysFailsTTS:
    async def synthesize(self, text, config):
        raise RuntimeError("provider down")


@pytest.mark.asyncio
async def test_run_turn_text_aborts_after_consecutive_tts_failures():
    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    engine = PipelineEngine(_FakeSTT(), _TwoSentenceLLM(), _AlwaysFailsTTS(), cfg)
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?", history=[], audio_sink=sink,
    )
    assert result.metrics.tts_segments_dropped >= 2
    assert result.cancelled is True
    assert sink_calls == []
```

Note: whether the fake LLM's single-token JSON envelope produces one or two sentences from
`SentenceDetector` depends on the detector's own clause-splitting behavior on
`"पहला वाक्य। दूसरा वाक्य।"` (two sentences separated by Devanagari danda-equivalent
punctuation) — if `result.metrics.tts_segments_dropped` comes back as exactly 1 instead of 2
because the detector treats it as one chunk, that is fine too: the important assertions are
`result.cancelled is True` (the 2-consecutive-failure abort fired) and `sink_calls == []` (no
audio got through). Adjust the `>=` assertion to match whatever the detector actually produces
if you observe a different real count when running the test — do not force the input text to
artificially produce exactly 2 sentences if the detector's natural behavior differs; report
what you observed in your task report either way.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py -v -k aborts_after_consecutive`
Expected: PASS

- [ ] **Step 7: Run the full existing engine test file to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py -v`
Expected: all tests pass (the 2 pre-existing tests + the 2 new ones from this task = 4 passed)

- [ ] **Step 8: Rewrite the existing hang-recovery test in voicebot.py's test file**

Open `tests/unit/test_voicebot_handle_turn_text.py`. Find
`test_handle_turn_text_recovers_on_provider_hang` (currently):

```python
@pytest.mark.asyncio
async def test_handle_turn_text_recovers_on_provider_hang(monkeypatch):
    """A hung provider call must not wedge the agent: the per-turn timeout
    walks the state machine back to LISTENING with a timeout error."""
    import src.agents.voicebot as vb
    monkeypatch.setattr(vb, "TURN_TIMEOUT_S", 0.05)

    class _HangingEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            await asyncio.sleep(5)  # never returns within the timeout
            raise AssertionError("should have timed out")

    agent = _agent(_HangingEngine())
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("कुछ", sink)
    assert agent.state.state is State.LISTENING
    assert "TimeoutError" in (outcome.response.parse_error or "")
```

Replace it with (the constant is renamed, and the semantics change: a hung engine call that
never checks `cancel_event` now gets force-cancelled after the backstop's hard-cap + grace
period elapse, producing a `cancelled=True` result that flows through the EXISTING
barge-in/cancelled-turn branch — ending with `parse_error == "barge-in"`, not a raised
`TimeoutError` string):

```python
@pytest.mark.asyncio
async def test_handle_turn_text_recovers_on_provider_hang(monkeypatch):
    """A hung provider call that never checks cancel_event must still not wedge
    the agent forever: the backstop's hard-cap + grace period force-cancel it,
    landing in the same cancelled-turn recovery path as a barge-in."""
    import src.agents.voicebot as vb
    monkeypatch.setattr(vb, "HARD_TURN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(vb, "_BACKSTOP_GRACE_S", 0.05)

    class _HangingEngine:
        async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
            await asyncio.sleep(5)  # never returns, never checks cancel_event
            raise AssertionError("should have been force-cancelled")

    agent = _agent(_HangingEngine())
    await agent.start()

    async def sink(a):
        pass

    outcome = await agent.handle_turn_text("कुछ", sink)
    assert agent.state.state is State.LISTENING
    assert outcome.response.parse_error == "barge-in"
```

- [ ] **Step 9: Run test to verify it fails (for the right reason)**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k recovers_on_provider_hang`
Expected: FAIL — either an `AttributeError` (no `HARD_TURN_TIMEOUT_S` attribute yet on the
`vb` module) or the test hanging/timing out at the real 20s+90s durations (since the old
`TURN_TIMEOUT_S` monkeypatch target no longer has any effect until Step 10 below is done) —
either failure mode confirms the test is exercising real, not-yet-updated code.

- [ ] **Step 10: Add the backstop helper and rewire `handle_turn`/`handle_turn_text`**

In `src/agents/voicebot.py`, replace the module-level constant (currently):

```python
TURN_TIMEOUT_S = 20.0
```

with:

```python
# LLM generation and each TTS sentence now enforce their own internal budgets
# (see LLM_TURN_TIMEOUT_S / TTS_SENTENCE_TIMEOUT_S in src/pipeline/engine.py), so
# these are a last-resort backstop against a truly wedged call, not the primary
# timeout mechanism — see VoiceBotAgent._run_turn_with_backstop.
HARD_TURN_TIMEOUT_S = 90.0
_BACKSTOP_GRACE_S = 30.0
```

Add this new private method to `VoiceBotAgent` (place it right before `handle_turn`, currently
~line 206):

```python
    async def _run_turn_with_backstop(self, coro, cancel_event: asyncio.Event) -> TurnResult:
        """Run a pipeline-turn coroutine with a hard backstop timeout.

        LLM generation and each TTS sentence now enforce their own internal
        budgets (PipelineEngine.run_turn_text), so this outer cap is a
        last-resort guard against a truly wedged call, not the primary timeout
        mechanism. On expiry we SET cancel_event (the same signal barge-in
        uses) rather than cancelling the task directly — a bare task-cancel
        can land inside `await tts_task` inside run_turn_text, which does NOT
        propagate to the independently-running TTS worker Task, letting it
        keep calling audio_sink after this turn has already returned control
        to LISTENING (the original mid-sentence cutoff bug). Only if the
        coroutine still hasn't wound down after a grace period do we fall
        back to a hard cancel.
        """
        task = asyncio.create_task(coro)
        done, _ = await asyncio.wait({task}, timeout=HARD_TURN_TIMEOUT_S)
        if task in done:
            return task.result()

        log.error("turn exceeded hard cap of %.0fs; signalling cancellation", HARD_TURN_TIMEOUT_S)
        cancel_event.set()
        done, _ = await asyncio.wait({task}, timeout=_BACKSTOP_GRACE_S)
        if task in done:
            return task.result()

        log.error("turn still wedged after %.0fs grace period; force-cancelling", _BACKSTOP_GRACE_S)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return TurnResult(
            user_text="", user_language=None, user_confidence=0.0,
            agent_text="", audio_bytes_sent=0, metrics=TurnMetrics(), cancelled=True,
        )
```

In `handle_turn` (currently ~line 206-254), replace:

```python
        try:
            pipeline_result = await asyncio.wait_for(
                self._engine.run_turn(
                    captured_audio=captured_audio,
                    history=self._history_window(),
                    audio_sink=audio_sink,
                    language=to_bcp47(self._active_language),
                ),
                timeout=TURN_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure (incl. timeout) must not drop the call
```

with:

```python
        cancel_event = asyncio.Event()
        try:
            pipeline_result = await self._run_turn_with_backstop(
                self._engine.run_turn(
                    captured_audio=captured_audio,
                    history=self._history_window(),
                    audio_sink=audio_sink,
                    cancel_event=cancel_event,
                    language=to_bcp47(self._active_language),
                ),
                cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure (incl. timeout) must not drop the call
```

`handle_turn` currently has NO handling for `pipeline_result.cancelled` (unlike
`handle_turn_text`) — since the backstop's rare force-cancel fallback now produces
`cancelled=True` instead of raising, add the same cancelled-turn branch right after the
`except Exception as exc:` block's `return TurnOutcome(...)`, before the final
`return await self._finish_turn(pipeline_result)` line:

```python
        if pipeline_result.cancelled:
            await self.state.fire(Event.LLM_RESPONSE_READY)
            await self.state.fire(Event.RESPONSE_DELIVERED)
            return TurnOutcome(
                response=VoiceBotResponse(
                    response_text="", action="continue", parse_error="barge-in"
                ),
                pipeline=pipeline_result,
            )

        return await self._finish_turn(pipeline_result)
```

In `handle_turn_text` (currently ~line 404-470), replace:

```python
        try:
            pipeline_result = await asyncio.wait_for(
                self._engine.run_turn_text(
                    user_text,
                    self._history_window(),
                    audio_sink,
                    cancel_event,
                    language=to_bcp47(self._active_language),
                ),
                timeout=TURN_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure (incl. timeout) must not drop the call
```

with:

```python
        cancel_event = cancel_event or asyncio.Event()
        try:
            pipeline_result = await self._run_turn_with_backstop(
                self._engine.run_turn_text(
                    user_text,
                    self._history_window(),
                    audio_sink,
                    cancel_event,
                    language=to_bcp47(self._active_language),
                ),
                cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure (incl. timeout) must not drop the call
```

The existing `if pipeline_result.cancelled:` branch further down in `handle_turn_text` (already
present, unchanged) now also correctly handles the backstop-triggered case in addition to
barge-in — no further change needed there.

- [ ] **Step 11: Run the rewritten test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k recovers_on_provider_hang`
Expected: PASS

- [ ] **Step 12: Update the stale comment in `sarvam.py`**

In `src/providers/tts/sarvam.py`, change the comment (currently):

```python
# A TTS request must fail well within the turn budget (TURN_TIMEOUT_S = 20s):
# otherwise a hung Sarvam request stalls the whole turn in "thinking" until the
# turn timeout cancels it mid-synthesis (no audio, 20s dead air). With a tight
# per-request timeout the hang fails fast and one retry can recover a transient
# blip — total worst case stays under the turn budget.
```

to:

```python
# A TTS request must fail well within the per-sentence watchdog
# (TTS_SENTENCE_TIMEOUT_S = 25s, src/pipeline/engine.py): otherwise a hung
# Sarvam request gets treated as a dropped segment instead of completing.
# With a tight per-request timeout the hang fails fast and one retry can
# recover a transient blip — total worst case (2 attempts) stays under the
# per-sentence watchdog.
```

- [ ] **Step 13: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented baseline, plus all new/updated tests from
this task passing, no reduction in test count (this task rewrites one test's assertions but
does not delete it).

- [ ] **Step 14: Commit**

```bash
git add src/pipeline/engine.py src/agents/voicebot.py src/providers/tts/sarvam.py tests/unit/test_engine_run_turn_text.py tests/unit/test_voicebot_handle_turn_text.py
git commit -m "fix(voicebot): route turn timeouts through cancel_event instead of orphaning the TTS worker task"
```

---

## Verification (after all 3 tasks)

- `.venv/bin/python -m pytest tests/unit -q` — full suite green apart from the documented
  pre-existing failures.
- `.venv/bin/alembic upgrade head` against a real (or throwaway local) Postgres DB — confirms
  Task 1's migration actually applies.
- Manual: run the dev console against the `t_dev` / "XYZ Official App" campaign (after Task 2's
  Step 19 data fix is separately applied with the user's confirmed pronunciations), trigger a
  multi-sentence LLM reply, and confirm (a) the opening line's proper nouns are spoken in
  Devanagari phonetics, not raw Latin, and (b) a longer multi-sentence reply plays to completion
  without an audible cutoff.
- Check `GET /api/v1/benchmarks/turn-metrics/summary` after a real call with at least one
  dropped TTS segment and confirm `avg_tts_segments_dropped` is non-zero for that combo.
