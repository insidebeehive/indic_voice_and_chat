# llm_total_ms Attribution Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a confirmed measurement/attribution bug: `TurnMetrics.llm_total_ms` is captured after the TTS queue has fully drained instead of right when the LLM's own work finishes, silently inflating it by the trailing TTS-synthesis time for the last sentence(s) — worst with a slow TTS provider (confirmed live: Gemini TTS was making `llm_total_ms` read larger, even though the LLM itself wasn't slower).

**Architecture:** Move the `metrics.llm_total_ms` timestamp capture to the first statement inside `run_turn_text`'s existing `finally` block — i.e. immediately after the LLM's token-generation loop and flush logic conclude (normally, via cancellation, or via an exception), and *before* draining the TTS sentence queue. No other metric's computation changes.

**Tech Stack:** Python 3, asyncio, pytest/pytest-asyncio.

## Global Constraints

- Branch is `stage` (already checked out on disk — do not create or switch branches; direct commits are this project's established workflow).
- Run `.venv/bin/python -m pytest tests/unit -q` before and after. Baseline confirmed moments ago: `24 failed, 1146 passed, 1 skipped, 22 errors` (pre-existing, unrelated per CLAUDE.md — do not chase them).
- Do NOT change `tts_total_ms`, `llm_ttft_ms`, `tts_first_chunk_ms`, or `total_latency_ms`'s computations — all four are already correct. Only `llm_total_ms`'s timestamp-capture *placement* changes.
- This bug predates this session by months (original commit) and is unrelated to any other work done today — this is a standalone, one-task fix.

---

### Task 1: Move the `llm_total_ms` timestamp capture before the TTS drain

**Files:**
- Modify: `src/pipeline/engine.py` (`run_turn_text`, currently lines 371-405)
- Test: `tests/unit/test_engine_run_turn_text.py`

**Interfaces:**
- Consumes: `TurnMetrics` (existing dataclass, unchanged shape), `PipelineEngine`/`PipelineConfig` (existing, unchanged), the file's existing `_FakeSTT`/`_FakeLLM` test doubles.
- Produces: no new public interface — this task only changes when one existing field gets computed, not any signature.

- [ ] **Step 1: Write the failing test**

Open `tests/unit/test_engine_run_turn_text.py`. Add a new fake TTS double whose single sentence takes a deliberately noticeable delay, and a test proving `llm_total_ms` excludes that delay:

```python
class _SlowTTS:
    """TTS whose synthesis takes noticeably longer than the fake LLM's
    near-instant token stream, so a test can prove llm_total_ms excludes
    the trailing TTS-drain time instead of silently including it."""

    async def synthesize(self, text, config):
        await asyncio.sleep(0.2)
        return TTSResult(audio=b"\x00\x00" * 80, duration_ms=10.0, sample_rate=16000)


@pytest.mark.asyncio
async def test_run_turn_text_llm_total_ms_excludes_tts_drain_tail():
    cfg = PipelineConfig(
        stt=STTConfig(language="hi-IN"),
        llm=LLMConfig(response_format="json", max_tokens=256),
        tts=TTSConfig(language="hi-IN", sample_rate=16000),
    )
    engine = PipelineEngine(_FakeSTT(), _FakeLLM(), _SlowTTS(), cfg)
    sink_calls = []

    async def sink(audio: bytes):
        sink_calls.append(audio)

    result = await engine.run_turn_text(
        "और कुछ benefits हैं?", history=[], audio_sink=sink,
    )
    # The fake LLM yields its 3 tokens near-instantly; the 0.2s TTS delay
    # happens almost entirely AFTER the LLM's own stream finishes. If
    # llm_total_ms still included that drain (the bug), it would read
    # >= ~200ms. It must stay well under that.
    assert result.metrics.llm_total_ms < 100
    # The full turn DOES take >= 200ms once TTS is included.
    assert result.metrics.total_latency_ms >= 200
```

If `asyncio` is not already imported at the top of this test file, add `import asyncio`. Check the file's existing imports first — the earlier `test_run_turn_text_cancel_stops_before_audio` test already uses `asyncio.Event()`, so `asyncio` should already be imported; if it somehow isn't, add it rather than assuming.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py -v -k excludes_tts_drain_tail`
Expected: FAIL — `assert result.metrics.llm_total_ms < 100` fails because today's code computes `llm_total_ms` *after* `await tts_task`, so it reads ~200ms (including the fake TTS's artificial delay), not near-zero.

- [ ] **Step 3: Move the timestamp capture**

In `src/pipeline/engine.py`, inside `run_turn_text`, replace (currently lines 371-405):

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
                full_text_parts.append(token)
                speakable = extractor.feed(token) if extractor is not None else token
                if speakable:
                    spoke_anything = True
                    for sentence in detector.feed(speakable):
                        await sentence_queue.put(sentence)

            if not cancel_event.is_set():
                if is_json and not spoke_anything:
                    for sentence in detector.feed(
                        _speakable_from_json("".join(full_text_parts))
                    ):
                        await sentence_queue.put(sentence)
                for sentence in detector.flush():
                    await sentence_queue.put(sentence)
        finally:
            await sentence_queue.put(None)
            await tts_task

        metrics.llm_total_ms = int((time.perf_counter() - t_llm_start) * 1000)
        if first_token_at is not None:
            metrics.llm_ttft_ms = int((first_token_at - t_llm_start) * 1000)
        if first_audio_at is not None:
            metrics.tts_first_chunk_ms = int((first_audio_at - t_llm_start) * 1000)
            metrics.tts_total_ms = int((time.perf_counter() - first_audio_at) * 1000)
        metrics.total_latency_ms = int((time.perf_counter() - t_overall) * 1000)
```

with:

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
                full_text_parts.append(token)
                speakable = extractor.feed(token) if extractor is not None else token
                if speakable:
                    spoke_anything = True
                    for sentence in detector.feed(speakable):
                        await sentence_queue.put(sentence)

            if not cancel_event.is_set():
                if is_json and not spoke_anything:
                    for sentence in detector.feed(
                        _speakable_from_json("".join(full_text_parts))
                    ):
                        await sentence_queue.put(sentence)
                for sentence in detector.flush():
                    await sentence_queue.put(sentence)
        finally:
            # Captured here, BEFORE draining the TTS queue: this must reflect
            # only the LLM's own work (token generation + flush), not however
            # long the trailing TTS synthesis for the last sentence(s) takes
            # afterward — that time is already correctly counted in
            # tts_total_ms below. Capturing it after `await tts_task` (the
            # bug this fixes) silently inflated llm_total_ms by that TTS
            # tail, worst with a slow TTS provider.
            metrics.llm_total_ms = int((time.perf_counter() - t_llm_start) * 1000)
            await sentence_queue.put(None)
            await tts_task

        if first_token_at is not None:
            metrics.llm_ttft_ms = int((first_token_at - t_llm_start) * 1000)
        if first_audio_at is not None:
            metrics.tts_first_chunk_ms = int((first_audio_at - t_llm_start) * 1000)
            metrics.tts_total_ms = int((time.perf_counter() - first_audio_at) * 1000)
        metrics.total_latency_ms = int((time.perf_counter() - t_overall) * 1000)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_engine_run_turn_text.py -v`
Expected: all tests in the file pass (pre-existing tests unaffected, the new test passes)

- [ ] **Step 5: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented baseline (`24 failed, 1146 passed, 1 skipped, 22 errors`), plus the 1 new passing test, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/engine.py tests/unit/test_engine_run_turn_text.py
git commit -m "fix(engine): stop llm_total_ms from silently including the trailing TTS-drain tail"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — full suite green apart from the documented pre-existing failures.
- Manual (optional, once redeployed): run a turn with a deliberately slow TTS provider and confirm `llm_total_ms` in the resulting `turn_metrics` row stays proportional to actual LLM latency, while `tts_total_ms` absorbs the slow-TTS time instead.
