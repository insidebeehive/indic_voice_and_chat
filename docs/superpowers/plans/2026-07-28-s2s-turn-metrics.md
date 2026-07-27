# S2S Turn-Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture benchmarking data (`turn_metrics` rows) for S2S (Gemini Live) calls — today S2S never calls `apply_signal` with `metrics_dict` at all, so zero rows exist for S2S despite the persistence pipeline already working for layered mode.

**Architecture:** Give `VoiceBotAgent.apply_signal` an explicit mode/provider-override capability (defaulting to today's cascade behavior), track two real timing boundaries in the S2S event loop (turn start → first spoken audio, turn start → turn_complete), and pass both through `apply_signal` with `mode="s2s"` and the true realtime-session provider name. `record_metric` is already wired at every `VoiceBotAgent` construction site including both S2S ones — no new wiring needed there.

**Tech Stack:** Python 3, asyncio, pytest/pytest-asyncio, SQLAlchemy 2.x async.

## Global Constraints

- Branch is `stage`. **Before the first commit tomorrow, run `git rev-parse --abbrev-ref HEAD` and confirm it says `stage`** — this exact mistake (committing to `main` by accident) happened twice already in this session; do not repeat it. If it says anything else, `git checkout stage` first.
- Run `.venv/bin/python -m pytest tests/unit -q` after every task. **Do not trust any baseline number written in this plan** — re-measure fresh when execution actually begins (this plan was written a day before implementation; other work may have landed on `stage` in the meantime).
- No new Alembic migration. Reuse the existing `mode` / `tts_first_chunk_ms` / `total_latency_ms` columns on `TurnMetric` — document the S2S semantic reinterpretation in code comments (`tts_first_chunk_ms` means "time to first spoken audio from the realtime model," not TTS specifically, for `mode="s2s"` rows).
- `apply_signal`'s new parameters MUST default to exactly today's behavior when omitted — `_finish_turn` (the cascade/layered caller in `voicebot.py`) needs zero changes and must keep working identically.
- Task order: Task 1 (apply_signal capability) → Task 2 (S2S wiring, consumes Task 1) → Task 3 (benchmarks endpoint visibility, independent but logically last).

---

### Task 1: `apply_signal` mode/provider-override capability

**Files:**
- Modify: `src/agents/voicebot.py` (`apply_signal`, currently lines 350-460)
- Test: `tests/unit/test_voicebot_handle_turn_text.py`

**Interfaces:**
- Produces: `apply_signal(..., metrics_mode: str = "layered", metrics_provider_override: Optional[dict[str, Optional[str]]] = None)`. When `metrics_provider_override` is `None` (the default, used by every existing caller), behavior is IDENTICAL to today. When provided, it's a dict with keys `"stt_provider"`, `"llm_provider"`, `"tts_provider"` (values `str | None`) that replace the `type(getattr(self._engine, ...)).__name__` derivation for both the `"voice turn metrics"` log line and the `record_metric` payload. `metrics_mode` replaces the hardcoded `"mode": "layered"` string in the `record_metric` payload only (the log line doesn't currently include mode, and this plan doesn't add it there — only the persisted DB payload needs mode).
- Consumes: nothing new from other tasks — this task is the foundation.

- [ ] **Step 1: Read the full current method**

Read `src/agents/voicebot.py` lines 350-460 in full (the complete `apply_signal` method) before making any change, to confirm current line numbers match this brief (they may have shifted slightly since this plan was written).

- [ ] **Step 2: Write the failing tests**

Find this file's existing agent-construction helper (it's `_agent_with_slots` per this session's earlier work on this same file — confirm the exact name/signature by reading the top of `tests/unit/test_voicebot_handle_turn_text.py` before writing, since it already accepts a `record_metric` param from earlier work). Add these two tests near the existing `apply_signal`/metrics tests in that file:

```python
async def test_apply_signal_uses_default_mode_and_engine_providers_when_no_override(make_agent):
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
    assert calls[0]["mode"] == "layered"
    assert calls[0]["stt_provider"]  # derived from self._engine, non-empty string


async def test_apply_signal_uses_explicit_mode_and_provider_override(make_agent):
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    agent = make_agent(record_metric=_record_metric)
    metrics_dict = {
        "stt_latency_ms": 0, "llm_ttft_ms": 0, "llm_total_ms": 0,
        "tts_first_chunk_ms": 1400, "tts_total_ms": 0, "total_latency_ms": 3800,
    }

    await agent.apply_signal(
        user_text="hi", agent_text="hello", action="continue",
        metrics_dict=metrics_dict,
        metrics_mode="s2s",
        metrics_provider_override={
            "stt_provider": None,
            "llm_provider": "GeminiLiveSession",
            "tts_provider": None,
        },
    )

    assert len(calls) == 1
    payload = calls[0]
    assert payload["mode"] == "s2s"
    assert payload["stt_provider"] is None
    assert payload["llm_provider"] == "GeminiLiveSession"
    assert payload["tts_provider"] is None
    assert payload["metrics"]["tts_first_chunk_ms"] == 1400
```

If this file's real fixture/helper name or calling convention differs from `make_agent(record_metric=...)` shown above (per Step 1's earlier note, it may be `_agent_with_slots`), adapt these two tests to match the file's actual pattern exactly — do not introduce a second construction path.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v -k "default_mode_and_engine or explicit_mode_and_provider"`
Expected: FAIL — `TypeError: apply_signal() got an unexpected keyword argument 'metrics_mode'`

- [ ] **Step 4: Add the parameters and use them**

In `src/agents/voicebot.py`, change the `apply_signal` signature from:

```python
    async def apply_signal(
        self,
        *,
        user_text: str,
        agent_text: str,
        action: str,
        updated_slots: Optional[dict[str, Any]] = None,
        sentiment: Optional[str] = None,
        phase: Optional[str] = None,
        metrics_dict: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
```

to:

```python
    async def apply_signal(
        self,
        *,
        user_text: str,
        agent_text: str,
        action: str,
        updated_slots: Optional[dict[str, Any]] = None,
        sentiment: Optional[str] = None,
        phase: Optional[str] = None,
        metrics_dict: Optional[dict[str, Any]] = None,
        metrics_mode: str = "layered",
        metrics_provider_override: Optional[dict[str, Optional[str]]] = None,
    ) -> dict[str, Any]:
```

Then, inside the `if metrics_dict:` block, replace:

```python
                log.info("voice turn metrics", extra={
                    "session_id": self.session.session_id,
                    "campaign_id": self.session.campaign_id,
                    "stt_provider": type(getattr(self._engine, "_stt", None)).__name__,
                    "llm_provider": type(getattr(self._engine, "_llm", None)).__name__,
                    "tts_provider": type(getattr(self._engine, "_tts", None)).__name__,
                    "metrics": metrics_dict,
                })
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

with:

```python
                if metrics_provider_override is not None:
                    stt_provider = metrics_provider_override.get("stt_provider")
                    llm_provider = metrics_provider_override.get("llm_provider")
                    tts_provider = metrics_provider_override.get("tts_provider")
                else:
                    stt_provider = type(getattr(self._engine, "_stt", None)).__name__
                    llm_provider = type(getattr(self._engine, "_llm", None)).__name__
                    tts_provider = type(getattr(self._engine, "_tts", None)).__name__
                log.info("voice turn metrics", extra={
                    "session_id": self.session.session_id,
                    "campaign_id": self.session.campaign_id,
                    "stt_provider": stt_provider,
                    "llm_provider": llm_provider,
                    "tts_provider": tts_provider,
                    "metrics": metrics_dict,
                })
                if self._record_metric is not None:
                    try:
                        await self._record_metric({
                            "session_id": self.session.session_id,
                            "campaign_id": self.session.campaign_id,
                            "mode": metrics_mode,
                            "stt_provider": stt_provider,
                            "llm_provider": llm_provider,
                            "tts_provider": tts_provider,
                            "action": action,
                            "metrics": metrics_dict,
                        })
                    except Exception:  # noqa: BLE001 - never break a live call on a metrics-write failure
                        log.warning(
                            "record_metric failed; continuing without persistence",
                            exc_info=True,
                        )
```

Do not touch anything below this block (the `sentiment`/`_ESCALATION_ACTIONS`/`_END_ACTIONS`/`persist_state` tail, lines ~442-460) — it is unrelated and must stay exactly as-is.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_voicebot_handle_turn_text.py -v`
Expected: all tests in the file pass (pre-existing tests unaffected — they never pass `metrics_mode`/`metrics_provider_override`, so they exercise the default-behavior path, which is unchanged; the 2 new tests pass).

- [ ] **Step 6: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the freshly-measured baseline (see Global Constraints), plus 2 new passing tests, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/agents/voicebot.py tests/unit/test_voicebot_handle_turn_text.py
git commit -m "feat(voicebot): let apply_signal report an explicit mode + provider override for benchmarking"
```

---

### Task 2: S2S turn-boundary timing + wiring

**Files:**
- Modify: `src/api/live_bridge_base.py` (`__init__` ~lines 80-102, `_drive` ~lines 105-157, `_consume_events` ~lines 193-242, `_commit_turn` ~lines 244-283)
- Test: `tests/unit/test_gemini_live_bridge.py`

**Interfaces:**
- Consumes: `apply_signal(..., metrics_mode=..., metrics_provider_override=...)` from Task 1.
- Produces: nothing new for later tasks — Task 3 only reads from the `turn_metrics` table Task 2 starts populating, no direct code dependency.

- [ ] **Step 1: Read the full current file**

Read `src/api/live_bridge_base.py` in full (342 lines) to confirm current line numbers match this brief — earlier session work may have shifted them slightly.

- [ ] **Step 2: Write the failing test**

Read `tests/unit/test_gemini_live_bridge.py`'s existing `_agent()` helper (currently ~lines 53-62) and `_bridge()` helper (currently ~lines 65-75) in full first. Extend `_agent()` with an optional `record_metric` parameter defaulting to `None`, so every existing call site (`_agent()` with no args) keeps working unchanged:

```python
def _agent(record_metric=None):
    return VoiceBotAgent(
        session=AgentSession(session_id="t1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema.from_campaign_yaml(
            {"interest_level": {"type": "enum", "values": ["hot", "warm", "cold"]}}),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=object(),
        store=None,
        record_metric=record_metric,
    )
```

Extend `_bridge()` to accept and thread through an optional `record_metric`:

```python
def _bridge(events, llm=None, record_metric=None):
    agent = _agent(record_metric=record_metric)
    sess = _FakeSession(events)

    async def connect(cfg):
        return sess

    b = GeminiLiveBridge(websocket=_FakeWS(), agent=agent,
                         config=RealtimeConfig(model="m"), connect_session=connect, llm=llm)
    b._session = sess
    return b, sess, agent
```

Then add this test near the existing `test_consume_events_records_turn_and_slots` test:

```python
@pytest.mark.asyncio
async def test_consume_events_records_s2s_turn_metrics():
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    events = [
        RealtimeEvent(type="input_transcript", text="yeh app safe hai?"),
        RealtimeEvent(type="output_transcript", text="bilkul safe hai"),
        RealtimeEvent(type="audio", audio=b"\x01\x02" * 100, audio_rate=24000),
        RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                      tool_args={"action": "send_info", "updated_slots": {}},
                      tool_id="x1"),
        RealtimeEvent(type="turn_complete"),
    ]
    b, sess, agent = _bridge(events, record_metric=_record_metric)
    await agent.start()
    await b._consume_events()

    assert len(calls) == 1
    payload = calls[0]
    assert payload["mode"] == "s2s"
    assert payload["stt_provider"] is None
    assert payload["llm_provider"] == "_FakeSession"
    assert payload["tts_provider"] is None
    metrics = payload["metrics"]
    # tts_first_chunk_ms here means "time to first spoken audio" for S2S rows
    # (see the code comment in live_bridge_base.py) — the audio event fired
    # before turn_complete, so this must be a real, non-zero measurement, and
    # must not exceed the full turn's total_latency_ms.
    assert metrics["tts_first_chunk_ms"] >= 0
    assert metrics["total_latency_ms"] >= metrics["tts_first_chunk_ms"]


@pytest.mark.asyncio
async def test_consume_events_no_metrics_recorded_for_silent_turn():
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    events = [RealtimeEvent(type="turn_complete")]
    b, sess, agent = _bridge(events, record_metric=_record_metric)
    await agent.start()
    await b._consume_events()

    # No user/agent text this turn (model went silent) — no real turn
    # happened, so no metrics row should be recorded.
    assert calls == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_gemini_live_bridge.py -v -k "s2s_turn_metrics or no_metrics_recorded_for_silent"`
Expected: FAIL — `record_metric` is never called (S2S doesn't build/pass `metrics_dict` yet), so `calls` stays empty and the `len(calls) == 1` assertion fails.

- [ ] **Step 4: Add per-turn timing state**

In `src/api/live_bridge_base.py`, in `__init__` (currently ~lines 80-102), add two new attributes right after the other per-turn accumulators (after `self._speaking = False`, before `self._last_event_at = 0.0`):

```python
        self._turn_start_at = 0.0        # monotonic ts: start of the current turn
        self._first_audio_at: float | None = None  # monotonic ts: first "audio" event this turn
```

In `_drive()` (currently ~line 114), right after `self._mark_activity()`, add:

```python
            self._turn_start_at = time.monotonic()
```

- [ ] **Step 5: Track first-audio timing**

In `_consume_events()` (currently ~lines 198-202), change:

```python
                if ev.type == "audio":
                    if not self._dbg_model_audio:
                        self._dbg_model_audio = True
                        log.info("live: model is producing audio (responding)")
                    await self._send_audio_out(ev.audio, ev.audio_rate)
```

to:

```python
                if ev.type == "audio":
                    if not self._dbg_model_audio:
                        self._dbg_model_audio = True
                        log.info("live: model is producing audio (responding)")
                    if self._first_audio_at is None:
                        self._first_audio_at = time.monotonic()
                    await self._send_audio_out(ev.audio, ev.audio_rate)
```

- [ ] **Step 6: Build and pass the metrics_dict in `_commit_turn`**

In `_commit_turn()` (currently ~lines 244-283), change:

```python
            await self._agent.apply_signal(
                user_text=user, agent_text=agent, action=action,
                updated_slots=self._pending_slots)
```

to:

```python
            now = time.monotonic()
            # S2S has no discrete STT/LLM/TTS phases (one end-to-end audio
            # model) — tts_first_chunk_ms is reinterpreted here as "time to
            # first spoken audio from the realtime model" (0 if the model
            # never produced audio this turn), and every other cascade-only
            # field stays 0. Reuses the existing turn_metrics schema; no new
            # column, disambiguated by mode="s2s" below.
            metrics_dict = {
                "stt_latency_ms": 0,
                "llm_ttft_ms": 0,
                "llm_total_ms": 0,
                "tts_first_chunk_ms": (
                    int((self._first_audio_at - self._turn_start_at) * 1000)
                    if self._first_audio_at is not None else 0
                ),
                "tts_total_ms": 0,
                "total_latency_ms": int((now - self._turn_start_at) * 1000),
                "tts_segments_dropped": 0,
            }
            await self._agent.apply_signal(
                user_text=user, agent_text=agent, action=action,
                updated_slots=self._pending_slots,
                metrics_dict=metrics_dict,
                metrics_mode="s2s",
                metrics_provider_override={
                    "stt_provider": None,
                    "llm_provider": type(self._session).__name__ if self._session is not None else None,
                    "tts_provider": None,
                },
            )
```

Then, in the existing state-reset block right after (currently):

```python
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action = None
        self._pending_slots = {}
        self._speaking = False
```

add the two new resets so the NEXT turn's clock starts fresh:

```python
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action = None
        self._pending_slots = {}
        self._speaking = False
        self._turn_start_at = time.monotonic()
        self._first_audio_at = None
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_gemini_live_bridge.py -v`
Expected: all tests in the file pass (pre-existing tests unaffected — none of them pass `record_metric`, so `self._record_metric` stays `None` on the agent and the `if self._record_metric is not None:` guard inside `apply_signal` means nothing new is exercised for them; the 2 new tests pass).

- [ ] **Step 8: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same counts as Task 1's final state, plus 2 new passing tests, no regressions. Also explicitly run `tests/unit/test_telephony_live_bridge.py` and `tests/unit/test_live_bridge_teardown.py` (both exercise `_BaseLiveBridge` from other angles) to confirm they're unaffected: `.venv/bin/python -m pytest tests/unit/test_telephony_live_bridge.py tests/unit/test_live_bridge_teardown.py -v`.

- [ ] **Step 9: Commit**

```bash
git add src/api/live_bridge_base.py tests/unit/test_gemini_live_bridge.py
git commit -m "feat(s2s): capture turn_metrics for Gemini Live calls (mode=s2s)"
```

---

### Task 3: Surface `mode` in the benchmarks summary endpoint

**Files:**
- Modify: `src/api/benchmarks.py` (`TurnMetricsComboEntry`, `TurnMetricsSummaryResponse`, `turn_metrics_summary`)
- Test: `tests/unit/test_benchmarks_turn_metrics_route.py`

**Interfaces:**
- Consumes: `TurnMetric.mode` (existing column, `src/models/turn_metrics.py`) — already populated correctly by Task 2 for S2S rows and pre-existing for layered rows (hardcoded `"layered"` before this plan, now via `metrics_mode` from Task 1, same value).
- Produces: no new interface for later tasks — this is the last task in this plan.

- [ ] **Step 1: Read the current file**

Read `src/api/benchmarks.py` in full to confirm current line numbers for `TurnMetricsComboEntry`, `TurnMetricsSummaryResponse`, and `turn_metrics_summary` match this brief.

- [ ] **Step 2: Write the failing test**

Read `tests/unit/test_benchmarks_turn_metrics_route.py`'s existing `client` fixture (seeds 3 `TurnMetric` rows directly, all with `mode="layered"`) in full first. Add a 4th seeded row with `mode="s2s"` to the SAME fixture (so it coexists with the existing 3, testing that mode-grouping doesn't merge combos that happen to share `llm_provider` across modes — pick a distinct `llm_provider` value like `"GeminiLiveSession"` so this new row is trivially distinguishable regardless):

```python
                TurnMetric(
                    tenant_id="dev", session_id="s2s1", campaign_id="bharat_matka",
                    mode="s2s", stt_provider=None,
                    llm_provider="GeminiLiveSession", tts_provider=None,
                    action="continue", stt_latency_ms=0, llm_ttft_ms=0,
                    llm_total_ms=0, tts_first_chunk_ms=1400, tts_total_ms=0,
                    total_latency_ms=3800, tts_segments_dropped=0,
                ),
```

Then add this test:

```python
async def test_summary_includes_mode_and_separates_s2s_combo(client: AsyncClient) -> None:
    resp = await client.get("/benchmarks/turn-metrics/summary", headers=ADMIN_HEADERS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    combos = {
        (e["mode"], e["stt_provider"], e["llm_provider"], e["tts_provider"]): e
        for e in body["entries"]
    }
    assert len(combos) == 3  # the 2 pre-existing layered combos + 1 new s2s combo

    s2s_combo = combos[("s2s", None, "GeminiLiveSession", None)]
    assert s2s_combo["samples"] == 1
    assert s2s_combo["avg_tts_first_chunk_ms"] == 1400.0
    assert s2s_combo["avg_total_latency_ms"] == 3800.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_benchmarks_turn_metrics_route.py -v -k separates_s2s_combo`
Expected: FAIL — `KeyError` on `e["mode"]` (the response doesn't include `mode` yet).

- [ ] **Step 4: Add `mode` to the endpoint**

In `src/api/benchmarks.py`, add a field to `TurnMetricsComboEntry` (after `stt_provider: Optional[str]`, i.e. as the first field so `mode` reads naturally alongside the other combo-identifying fields):

```python
class TurnMetricsComboEntry(BaseModel):
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
```

In `turn_metrics_summary`'s `select(...)`, add `TurnMetric.mode` as the first selected column and add it to `group_by(...)`:

```python
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
```

And in the `TurnMetricsComboEntry(...)` construction inside the list comprehension, add `mode=r.mode,` as the first kwarg:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_benchmarks_turn_metrics_route.py -v`
Expected: all tests in the file pass (the 2 pre-existing combo-grouping assertions still hold since they don't check `mode` explicitly and both existing seeded rows share `mode="layered"`; the new test passes).

- [ ] **Step 6: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same counts as Task 2's final state, plus 1 new passing test, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/api/benchmarks.py tests/unit/test_benchmarks_turn_metrics_route.py
git commit -m "feat(benchmarking): surface mode in the turn-metrics summary endpoint"
```

---

## Verification (after all 3 tasks)

- `.venv/bin/python -m pytest tests/unit -q` — full suite green apart from documented pre-existing failures.
- Manual (once redeployed): place a real S2S call via the dev console, confirm a new `turn_metrics` row appears with `mode="s2s"`, `stt_provider`/`tts_provider` both null, `llm_provider` the real Gemini Live session class name, and a plausible non-zero `tts_first_chunk_ms`/`total_latency_ms`.
- `GET /api/v1/benchmarks/turn-metrics/summary` with a valid admin token — confirm S2S and layered combos both appear, clearly distinguished by `mode`.
