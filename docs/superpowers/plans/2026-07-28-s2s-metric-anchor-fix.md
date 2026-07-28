# S2S Turn-Metrics Anchor Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Anchor S2S (Gemini Live) turn-metrics latency fields from the caller's last detected speech this turn, instead of from turn-start, so they measure model responsiveness instead of also including however long the caller spoke.

**Architecture:** Track a new monotonic timestamp, `self._last_input_transcript_at`, updated on every `input_transcript` event in `_BaseLiveBridge._consume_events`. In `_commit_turn`, compute `tts_first_chunk_ms` and `total_latency_ms` from that timestamp (falling back to the existing `self._turn_start_at` only when no caller speech was heard this turn, e.g. the agent-greets-first kickoff turn) instead of always from `self._turn_start_at`. No new DB column, no new event type — purely an in-memory timestamp change in shared bridge infrastructure used by both the browser dev console (`GeminiLiveBridge`) and telephony (`TelephonyLiveBridge`).

**Tech Stack:** Python 3, asyncio, pytest + pytest-asyncio.

## Global Constraints

- Branch is `stage` (confirmed checked out, working tree clean of anything but unrelated untracked docs files). Do NOT create or switch branches. Direct commits are this project's established workflow.
- Run `.venv/bin/python -m pytest tests/unit -q` after the task; do not trust any pass/fail count written in this plan as current — re-measure fresh at execution time.
- No new Alembic migration — this is an in-memory timestamp only, not a schema change.
- Do not touch the 6 existing S2S `turn_metrics` rows on Stage's DB — the user is clearing those themselves.
- Never use "thesis"/academic framing in commit messages or comments.
- This touches the same subtle timing/state code (`_BaseLiveBridge`) reviewed carefully in the immediately-preceding S2S turn-metrics plan — apply the same rigor: multi-turn correctness, reset-block ordering, fallback-path correctness.

---

### Task 1: Anchor S2S turn-metrics from last-detected caller speech

**Files:**
- Modify: `src/api/live_bridge_base.py:80-104` (`__init__`), `:196-233`-ish (`_consume_events`'s `input_transcript` branch — re-locate exact lines at implementation time, since line numbers may have shifted), `:249-317`-ish (`_commit_turn` — same caveat)
- Modify: `src/api/benchmarks.py:82-95` (`TurnMetricsComboEntry` docstring)
- Test: `tests/unit/test_gemini_live_bridge.py`

**Interfaces:**
- Consumes: `_BaseLiveBridge.__init__`'s existing `self._turn_start_at: float`, `self._first_audio_at: float | None` (both already exist; this task only adds a third timestamp alongside them, does not change their own semantics or update points).
- Produces: `self._last_input_transcript_at: float | None` — a new instance attribute other bridge code (or a future fix) can read; no other task/file currently depends on it.

- [ ] **Step 1: Write the failing tests**

Add two new tests to `tests/unit/test_gemini_live_bridge.py`, placed directly after the existing `test_consume_events_records_s2s_turn_metrics` test (around line 137). These use a custom fake session whose `events()` generator inserts a real `asyncio.sleep` so the gap between turn-start and the last `input_transcript` is large enough to distinguish from the (short) gap between the last `input_transcript` and the `audio` event — proving which anchor the code actually used.

```python
@pytest.mark.asyncio
async def test_consume_events_anchors_metrics_from_last_input_transcript():
    # Simulate a turn where the caller spoke for a while (long gap between
    # turn-start and their last transcript chunk) and the model then responded
    # quickly after that (short gap from last transcript to first audio). If
    # metrics are still anchored from turn-start, tts_first_chunk_ms would be
    # dominated by the caller's speaking time (the long sleep below) instead of
    # the model's actual response time (the short sleep below).
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    class _SlowStartSession(_FakeSession):
        async def events(self):
            await asyncio.sleep(0.15)  # caller "speaking" before their first transcript chunk
            yield RealtimeEvent(type="input_transcript", text="yeh")
            await asyncio.sleep(0.15)  # more caller speech
            yield RealtimeEvent(type="input_transcript", text=" app safe hai?")
            await asyncio.sleep(0.02)  # short gap: model responds quickly after caller stops
            yield RealtimeEvent(type="output_transcript", text="bilkul safe hai")
            yield RealtimeEvent(type="audio", audio=b"\x01\x02" * 100, audio_rate=24000)
            yield RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                                 tool_args={"action": "send_info", "updated_slots": {}},
                                 tool_id="x1")
            yield RealtimeEvent(type="turn_complete")

    agent = _agent(record_metric=_record_metric)
    sess = _SlowStartSession([])

    async def connect(cfg):
        return sess

    b = GeminiLiveBridge(websocket=_FakeWS(), agent=agent,
                         config=RealtimeConfig(model="m"), connect_session=connect, llm=None)
    b._session = sess
    await agent.start()
    b._turn_start_at = asyncio.get_event_loop().time()
    import time as _time
    b._turn_start_at = _time.monotonic()
    await b._consume_events()

    assert len(calls) == 1
    metrics = calls[0]["metrics"]
    # Anchored from last input_transcript (~0.02s before audio): must be well
    # under the ~0.3s+0.02s it would be if still anchored from turn-start.
    assert metrics["tts_first_chunk_ms"] < 150
    assert metrics["total_latency_ms"] >= metrics["tts_first_chunk_ms"]


@pytest.mark.asyncio
async def test_consume_events_falls_back_to_turn_start_when_no_input_transcript():
    # Greeting-first turn: the model speaks without ever hearing the caller
    # this turn. There is no input_transcript to anchor from, so metrics must
    # fall back to turn_start_at (unchanged behavior for this edge case).
    calls = []

    async def _record_metric(payload):
        calls.append(payload)

    events = [
        RealtimeEvent(type="output_transcript", text="नमस्ते! मैं Anaaya बोल रही हूँ"),
        RealtimeEvent(type="audio", audio=b"\x01\x02" * 100, audio_rate=24000),
        RealtimeEvent(type="tool_call", tool_name="record_turn_signal",
                      tool_args={"action": "continue", "updated_slots": {}}, tool_id="x1"),
        RealtimeEvent(type="turn_complete"),
    ]
    b, sess, agent = _bridge(events, record_metric=_record_metric)
    await agent.start()
    await b._consume_events()

    assert len(calls) == 1
    metrics = calls[0]["metrics"]
    # No input_transcript this turn -> anchor is turn_start_at -> must still be
    # a small, non-negative, well-defined number (not a crash, not negative).
    assert metrics["tts_first_chunk_ms"] >= 0
    assert metrics["total_latency_ms"] >= metrics["tts_first_chunk_ms"]
```

Note: the first test constructs the bridge manually (not via the `_bridge()` helper) because it needs a custom `_FakeSession` subclass with a real `asyncio.sleep`-based `events()` generator — the shared `_bridge()` helper only accepts a flat list of pre-built events with no delays.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_gemini_live_bridge.py -k "anchors_metrics_from_last_input_transcript or falls_back_to_turn_start" -v`
Expected: `test_consume_events_anchors_metrics_from_last_input_transcript` FAILS (assertion `metrics["tts_first_chunk_ms"] < 150` fails because current code anchors from `_turn_start_at`, giving a value around 300+ms). `test_consume_events_falls_back_to_turn_start_when_no_input_transcript` PASSES already (this is the unchanged fallback path — it's included to lock in current behavior, not to prove a bug), which is fine; both tests must at least run without erroring.

- [ ] **Step 3: Add the new timestamp in `__init__`**

In `src/api/live_bridge_base.py`, in `_BaseLiveBridge.__init__`, add the new attribute directly after the existing `self._first_audio_at` line:

```python
        self._first_audio_at: float | None = None  # monotonic ts: first "audio" event this turn
        self._last_input_transcript_at: float | None = None  # monotonic ts: last "input_transcript" event this turn (proxy for "caller stopped talking")
        self._last_event_at = 0.0   # monotonic ts of the last model event (idle watchdog)
```

- [ ] **Step 4: Update `_consume_events`'s `input_transcript` branch**

In `src/api/live_bridge_base.py`, find the `elif ev.type == "input_transcript":` branch inside `_consume_events` (currently around line 208). It reads:

```python
                elif ev.type == "input_transcript":
                    if not self._dbg_heard_caller:
                        self._dbg_heard_caller = True
                        log.info("live: model heard the caller (input_transcript)",
                                 extra={"first_text": ev.text[:60]})
                    self._user_buf += ev.text
                    await self._emit_transcript("user", self._user_buf, partial=True)
```

Change it to unconditionally update the new timestamp on every event (not gated by `_dbg_heard_caller`, since we want the LAST one, not the first):

```python
                elif ev.type == "input_transcript":
                    if not self._dbg_heard_caller:
                        self._dbg_heard_caller = True
                        log.info("live: model heard the caller (input_transcript)",
                                 extra={"first_text": ev.text[:60]})
                    self._last_input_transcript_at = time.monotonic()
                    self._user_buf += ev.text
                    await self._emit_transcript("user", self._user_buf, partial=True)
```

- [ ] **Step 5: Anchor `_commit_turn`'s metrics from last-detected caller speech**

In `src/api/live_bridge_base.py`, in `_commit_turn` (currently around line 249), find:

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
```

Replace with:

```python
            now = time.monotonic()
            # Anchor from the last input_transcript event this turn — the best
            # available proxy for "the caller just finished speaking" (the
            # underlying Live session does its own VAD internally but never
            # surfaces a discrete end-of-speech event; see RealtimeEvent). This
            # can lag the true end-of-speech moment slightly, since incoming
            # audio isn't transcribed instantly, but it is far closer than
            # turn_start_at, which would include however long the caller spoke.
            # Falls back to turn_start_at only when no caller speech was heard
            # this turn at all (e.g. the agent-greets-first kickoff turn).
            anchor = (
                self._last_input_transcript_at
                if self._last_input_transcript_at is not None
                else self._turn_start_at
            )
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
                    max(0, int((self._first_audio_at - anchor) * 1000))
                    if self._first_audio_at is not None else 0
                ),
                "tts_total_ms": 0,
                "total_latency_ms": max(0, int((now - anchor) * 1000)),
                "tts_segments_dropped": 0,
            }
```

- [ ] **Step 6: Reset the new timestamp in the per-turn reset block**

Still in `_commit_turn`, find the per-turn reset block (currently around line 305):

```python
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action = None
        self._pending_slots = {}
        self._speaking = False
        self._turn_start_at = time.monotonic()
        self._first_audio_at = None
```

Change to:

```python
        self._user_buf = ""
        self._agent_buf = ""
        self._pending_action = None
        self._pending_slots = {}
        self._speaking = False
        self._turn_start_at = time.monotonic()
        self._first_audio_at = None
        self._last_input_transcript_at = None
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_gemini_live_bridge.py -v`
Expected: all tests in this file PASS, including both new ones.

- [ ] **Step 8: Update the now-inaccurate docstring in `benchmarks.py`**

In `src/api/benchmarks.py`, the `TurnMetricsComboEntry` class docstring (lines 82-95) currently reads:

```python
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
```

Replace with:

```python
class TurnMetricsComboEntry(BaseModel):
    """One (mode, stt_provider, llm_provider, tts_provider) combo's averaged
    per-turn latencies.

    For ``mode="s2s"`` (Gemini Live), there is no bridge-level VAD/utterance-end
    event, so both fields are anchored from the last ``input_transcript`` event
    seen this turn — a proxy for "the caller just finished speaking" — falling
    back to turn-start only on a turn with no caller speech at all (e.g. an
    opening greeting). ``avg_tts_first_chunk_ms`` is reinterpreted as
    time-to-first-spoken-audio from the realtime model rather than a
    TTS-specific measurement. This makes S2S rows broadly comparable in spirit
    to ``mode="layered"`` rows (both are utterance-end-anchored), but not
    byte-for-byte identical in methodology — layered mode's timestamp comes
    from a hard STT-completion signal, S2S's from a proxy. Compare
    ``mode="layered"`` rows against each other, and ``mode="s2s"`` rows against
    each other, and treat cross-mode comparisons as directional, not exact.
    """
```

- [ ] **Step 9: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail count as the pre-existing baseline (the two known pre-existing failures unrelated to this work — `test_chat_routes.py::test_claim_session_and_agent_ws` and `test_prompts.py::test_chatbot_prompt_has_scope_guardrails` — may still fail; no new failures).

- [ ] **Step 10: Commit**

```bash
git add src/api/live_bridge_base.py src/api/benchmarks.py tests/unit/test_gemini_live_bridge.py
git commit -m "fix(s2s): anchor turn-metrics from last caller speech, not turn-start

S2S latency numbers looked worse than layered mode despite the call feeling
faster, because tts_first_chunk_ms/total_latency_ms were measured from
turn-start — which includes however long the caller spoke. Anchor from the
last input_transcript event instead (falls back to turn-start only when no
caller speech was heard this turn, e.g. a greeting-first turn)."
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` green (modulo the two known pre-existing unrelated failures).
- `.venv/bin/python -m pytest tests/unit/test_gemini_live_bridge.py -v` — all pass, including the two new anchor tests.
- No Alembic migration added; `git diff --stat` for this task touches only `src/api/live_bridge_base.py`, `src/api/benchmarks.py`, and the one test file.
