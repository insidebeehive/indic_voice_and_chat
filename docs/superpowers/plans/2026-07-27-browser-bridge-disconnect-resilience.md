# Browser Bridge Disconnect Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a confirmed production bug (real logs) where the dev-console browser WebSocket dies mid-turn while testing a slow TTS provider (IndicF5), crashing the entire session with an unhandled exception instead of ending gracefully — and add a heartbeat so the connection survives long-running turns in the first place.

**Architecture:** Centralize outbound-send failure handling in the bridge's two low-level send functions (`_send_json`, `_send_pcm`) so a dead socket is detected once and handled once, instead of crashing whichever call site happens to hit it. Add a background heartbeat task that runs alongside any in-flight turn, serialized against real sends via a lock so the two never corrupt the WebSocket message stream.

**Tech Stack:** Python 3, asyncio, Starlette WebSockets, pytest/pytest-asyncio.

## Global Constraints

- Scoped ONLY to `src/api/browser_bridge.py` (the dev-console-only browser bridge). Do NOT touch the Twilio/Exotel/Stringee telephony bridges — separate files, separate connection-liveness semantics, not reported as broken, out of scope for this plan.
- Branch is `stage` (already checked out on disk — do not create or switch branches; direct commits are this project's established workflow).
- Run `.venv/bin/python -m pytest tests/unit -q` after every task. Baseline in effect at the start of this plan: `24 failed, 1131 passed, 1 skipped, 22 errors` (pre-existing, unrelated per CLAUDE.md — do not chase them).
- Task order matters: Task 2 (heartbeat) depends on Task 1's `self._send_lock` existing to safely serialize concurrent sends.
- Reuse the existing test file's `FakeWebSocket`/`_bridge()`/`FakeAgent` helpers in `tests/unit/test_browser_bridge.py` exactly as they exist — do not invent a second WebSocket test-double pattern in the same file.

---

### Task 1: Guard outbound sends against a dead socket

**Files:**
- Modify: `src/api/browser_bridge.py` (`__init__` ~line 92-131, `_send_json` ~line 135-136, `_send_pcm` ~line 138-153)
- Test: `tests/unit/test_browser_bridge.py`

**Interfaces:**
- Produces: `self._send_lock: asyncio.Lock` (new instance attribute on `BrowserVoiceBridge`) — Task 2's heartbeat loop uses this same lock via `_send_json` (it doesn't need to touch the lock directly, since `_send_json` already acquires it internally).
- Consumes: `self._cancel_event` (existing attribute, already set to a real `asyncio.Event()` during an in-flight streaming turn, `None` otherwise — confirmed by reading `_dispatch_text_turn`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_browser_bridge.py`. First add `import asyncio` to the file's imports (it's currently missing — only `json` and `pytest` are imported at the top):

```python
import asyncio
```

Then add a websocket test-double whose sends raise (simulating a closed connection), right after the existing `FakeWebSocket` class definition:

```python
class _RaisingWebSocket(FakeWebSocket):
    """A websocket whose sends raise, simulating a closed/dead connection —
    the exact RuntimeError Starlette raises when the app tries to send after
    the connection has already closed."""

    async def send_text(self, data: str) -> None:
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
        )

    async def send_bytes(self, data: bytes) -> None:
        raise RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
        )
```

Add these 5 tests near the existing `test_send_json_emits_text_frame`/`test_send_pcm_writes_binary_frames` tests:

```python
@pytest.mark.asyncio
async def test_send_json_swallows_closed_socket_and_stops():
    ws = _RaisingWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_json({"type": "status", "status": "listening"})  # must not raise
    assert bridge._stopped is True


@pytest.mark.asyncio
async def test_send_pcm_swallows_closed_socket_and_stops():
    ws = _RaisingWebSocket([])
    bridge = _bridge(ws)
    await bridge._send_pcm(b"\x01\x02\x03\x04")  # must not raise
    assert bridge._stopped is True


@pytest.mark.asyncio
async def test_send_json_noop_once_stopped():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    bridge._stopped = True
    await bridge._send_json({"type": "status", "status": "listening"})
    assert ws.sent_text == []


@pytest.mark.asyncio
async def test_send_pcm_noop_once_stopped():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)
    bridge._stopped = True
    await bridge._send_pcm(b"\x01\x02\x03\x04")
    assert ws.sent_bytes == []
    assert ws.sent_text == []


@pytest.mark.asyncio
async def test_send_pcm_sets_cancel_event_on_closed_socket():
    ws = _RaisingWebSocket([])
    bridge = _bridge(ws)
    bridge._cancel_event = asyncio.Event()
    await bridge._send_pcm(b"\x01\x02")
    assert bridge._cancel_event.is_set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v -k "closed_socket or noop_once_stopped or cancel_event_on_closed"`
Expected: FAIL — the `_RaisingWebSocket`-based tests fail because the current `_send_json`/`_send_pcm` let the `RuntimeError` propagate instead of swallowing it; the `noop_once_stopped` tests fail because `_send_json`/`_send_pcm` don't currently check `self._stopped` at all.

- [ ] **Step 3: Add the lock and rewrite `_send_json`**

In `src/api/browser_bridge.py`, add `self._send_lock = asyncio.Lock()` to `__init__` — place it right after `self._stopped = False` (currently line 113):

```python
        self._stopped = False
        self._send_lock = asyncio.Lock()
```

Replace `_send_json` (currently lines 135-136):

```python
    async def _send_json(self, obj: dict) -> None:
        await self._ws.send_text(json.dumps(obj))
```

with:

```python
    async def _send_json(self, obj: dict) -> None:
        if self._stopped:
            return
        async with self._send_lock:
            try:
                await self._ws.send_text(json.dumps(obj))
            except Exception:  # noqa: BLE001 - client disconnected mid-turn; stop sending, don't crash the session
                log.info("browser bridge: send failed (client likely disconnected)")
                self._stopped = True
                if self._cancel_event is not None:
                    self._cancel_event.set()
```

- [ ] **Step 4: Rewrite `_send_pcm`**

Replace `_send_pcm` (currently lines 138-153):

```python
    async def _send_pcm(self, pcm16: bytes) -> None:
        """AudioSink: ship agent TTS audio to the browser as binary frames.

        Unlike Twilio there is no real-time pacing — the browser schedules
        gapless playback itself, so we just chunk and send.
        """
        if not pcm16:
            return
        await self._send_json({"type": "status", "status": "speaking"})
        for i in range(0, len(pcm16), _SEND_CHUNK):
            await self._ws.send_bytes(pcm16[i : i + _SEND_CHUNK])
        # Track when this audio will finish playing (16-bit mono PCM), mirroring
        # the browser's gapless scheduling, so a terminal turn can wait for it.
        duration_s = len(pcm16) / 2 / self._config.pcm_sample_rate
        self._play_until = max(self._play_until, time.monotonic()) + duration_s
        await self._send_json({"type": "status", "status": "listening"})
```

with:

```python
    async def _send_pcm(self, pcm16: bytes) -> None:
        """AudioSink: ship agent TTS audio to the browser as binary frames.

        Unlike Twilio there is no real-time pacing — the browser schedules
        gapless playback itself, so we just chunk and send.
        """
        if not pcm16 or self._stopped:
            return
        await self._send_json({"type": "status", "status": "speaking"})
        for i in range(0, len(pcm16), _SEND_CHUNK):
            if self._stopped:
                return
            async with self._send_lock:
                try:
                    await self._ws.send_bytes(pcm16[i : i + _SEND_CHUNK])
                except Exception:  # noqa: BLE001 - client disconnected mid-turn; stop sending, don't crash the session
                    log.info("browser bridge: send_bytes failed (client likely disconnected)")
                    self._stopped = True
                    if self._cancel_event is not None:
                        self._cancel_event.set()
                    return
        # Track when this audio will finish playing (16-bit mono PCM), mirroring
        # the browser's gapless scheduling, so a terminal turn can wait for it.
        duration_s = len(pcm16) / 2 / self._config.pcm_sample_rate
        self._play_until = max(self._play_until, time.monotonic()) + duration_s
        await self._send_json({"type": "status", "status": "listening"})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v`
Expected: all tests in the file pass (the pre-existing ones unaffected, the 5 new ones green)

- [ ] **Step 6: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented baseline, plus the 5 new passing tests, no regressions. Also run `tests/unit/test_browser_bridge_streaming.py` explicitly since it exercises the same class from a different angle: `.venv/bin/python -m pytest tests/unit/test_browser_bridge_streaming.py -v` — expected: unchanged, all still passing.

- [ ] **Step 7: Commit**

```bash
git add src/api/browser_bridge.py tests/unit/test_browser_bridge.py
git commit -m "fix(browser-bridge): stop sends gracefully on a dead socket instead of crashing the session"
```

---

### Task 2: Heartbeat during in-flight turns

**Files:**
- Modify: `src/api/browser_bridge.py` (new constant near `_SEND_CHUNK` ~line 41, two new methods near `_send_pcm`/`_send_filler` ~line 155-166, two call-site changes: `_dispatch_utterance` ~line 329, `_dispatch_text_turn` ~lines 521-523)
- Test: `tests/unit/test_browser_bridge.py`

**Interfaces:**
- Consumes: `self._send_lock`, the guarded `_send_json` (Task 1) — the heartbeat's own sends go through the same protected path, so a dead socket during a heartbeat tick is handled identically (sets `self._stopped`, no crash).
- Produces: `BrowserVoiceBridge._run_with_heartbeat(self, coro) -> Any` — wraps any turn-processing coroutine; both `_dispatch_utterance` and `_dispatch_text_turn` call through it instead of awaiting `self._agent.handle_turn(...)`/`self._agent.handle_turn_text(...)` directly.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_browser_bridge.py`:

```python
@pytest.mark.asyncio
async def test_run_with_heartbeat_sends_periodic_status(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "_HEARTBEAT_INTERVAL_S", 0.05)
    ws = FakeWebSocket([])
    bridge = _bridge(ws)

    async def _slow_coro():
        await asyncio.sleep(0.17)
        return "done"

    result = await bridge._run_with_heartbeat(_slow_coro())
    assert result == "done"
    statuses = [
        json.loads(t).get("status") for t in ws.sent_text
        if json.loads(t).get("type") == "status"
    ]
    assert statuses.count("thinking") >= 2  # at least 2 heartbeats during the 0.17s sleep at 0.05s interval


@pytest.mark.asyncio
async def test_run_with_heartbeat_stops_after_coro_completes(monkeypatch):
    import src.api.browser_bridge as bb
    monkeypatch.setattr(bb, "_HEARTBEAT_INTERVAL_S", 0.05)
    ws = FakeWebSocket([])
    bridge = _bridge(ws)

    async def _fast_coro():
        return "done"

    await bridge._run_with_heartbeat(_fast_coro())
    count_after = len(ws.sent_text)
    await asyncio.sleep(0.15)  # would produce more heartbeat sends if the task weren't cancelled
    assert len(ws.sent_text) == count_after


@pytest.mark.asyncio
async def test_run_with_heartbeat_propagates_coro_exception():
    ws = FakeWebSocket([])
    bridge = _bridge(ws)

    async def _failing_coro():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await bridge._run_with_heartbeat(_failing_coro())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v -k run_with_heartbeat`
Expected: FAIL — `AttributeError: 'BrowserVoiceBridge' object has no attribute '_run_with_heartbeat'`

- [ ] **Step 3: Add the constant and the two new methods**

In `src/api/browser_bridge.py`, add a constant right after `_SEND_CHUNK = 8192` (currently line 41):

```python
_SEND_CHUNK = 8192
_HEARTBEAT_INTERVAL_S = 10.0  # keep the WS connection active during long-running turns — a
                                # slow self-hosted TTS provider (e.g. IndicF5) can take longer
                                # than a network intermediary's idle-connection timeout
```

Add two new methods to `BrowserVoiceBridge`, right after `_send_filler` (currently ends at line 166, right before the `# --- entrypoint ---` comment at line 168):

```python
    async def _heartbeat_loop(self) -> None:
        """Send a periodic status ping while a turn is in flight, so no
        intermediary (load balancer, proxy) treats a long-running turn as an
        idle connection and closes it — see _run_with_heartbeat."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            if self._stopped:
                return
            await self._send_json({"type": "status", "status": "thinking"})

    async def _run_with_heartbeat(self, coro):
        """Run `coro` (a turn-processing call) while a background heartbeat
        keeps the WebSocket connection visibly active — see _heartbeat_loop."""
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            return await coro
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except BaseException:  # noqa: BLE001 - cancellation during teardown, matches this file's existing pattern (e.g. run()'s own stream_task cleanup)
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v -k run_with_heartbeat`
Expected: 3 passed

- [ ] **Step 5: Wire it at both turn-dispatch call sites**

In `_dispatch_utterance` (currently ~line 329), change:

```python
        outcome = await self._agent.handle_turn(captured, self._send_pcm)
```

to:

```python
        outcome = await self._run_with_heartbeat(
            self._agent.handle_turn(captured, self._send_pcm)
        )
```

In `_dispatch_text_turn` (currently ~lines 521-523), change:

```python
            outcome = await self._agent.handle_turn_text(
                text, self._send_pcm, cancel_event=self._cancel_event
            )
```

to:

```python
            outcome = await self._run_with_heartbeat(
                self._agent.handle_turn_text(
                    text, self._send_pcm, cancel_event=self._cancel_event
                )
            )
```

- [ ] **Step 6: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented baseline, plus the 8 new passing tests from Tasks 1-2 total, no regressions. Existing tests exercising `_dispatch_utterance`/`_dispatch_text_turn` (e.g. `test_run_handshake_plays_opening_and_processes_a_turn`) must still pass unchanged — they complete fast enough that the default 10-second heartbeat interval never fires during the test, so wrapping the call in `_run_with_heartbeat` should not change their behavior or assertions.

- [ ] **Step 7: Commit**

```bash
git add src/api/browser_bridge.py tests/unit/test_browser_bridge.py
git commit -m "feat(browser-bridge): heartbeat during in-flight turns to survive slow TTS providers"
```

---

## Verification (after both tasks)

- `.venv/bin/python -m pytest tests/unit -q` — full suite green apart from the documented pre-existing failures.
- Manual: run the dev console against IndicF5 again (once redeployed) and confirm the connection survives a slow turn without crashing, and if it does eventually disconnect for an unrelated reason, the server logs a clean `"browser bridge: send failed (client likely disconnected)"` instead of `"browser voice bridge crashed"`.
