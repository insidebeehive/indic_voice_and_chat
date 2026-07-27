# Dev-Console Fail-Loud Provider Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a confirmed, user-reported bug class: when a user explicitly selects an STT/LLM/TTS provider in the dev console's dropdowns and that provider fails to construct (missing API key/URL), the console silently substitutes the tenant's default provider and only logs a WARNING — the user believed a specific provider ("vllm") was being tested for weeks when it never actually was. Make these fail loudly instead.

**Architecture:** Remove the try/except-and-silently-substitute wrapper around each of the 4 explicit provider-override construction calls in `src/api/dev_console.py`, letting the real construction exception propagate up to the existing bridge-build error handler — then fix that handler's hardcoded WebSocket close reason so it reflects the real failure, and surface that reason as a visible error banner in the dev console's frontend (reusing the existing error-banner pattern already used for other error messages).

**Tech Stack:** Python 3, FastAPI WebSockets, pytest/pytest-asyncio, vanilla JS (no framework/build step for `static/dev_console.html`).

## Global Constraints

- Branch is `stage` (already checked out on disk — do not create or switch branches; direct commits are this project's established workflow).
- Run `.venv/bin/python -m pytest tests/unit -q` after every task. Baseline in effect at the start of this plan: `24 failed, 1139 passed, 1 skipped, 22 errors` (pre-existing, unrelated per CLAUDE.md — do not chase them).
- Do NOT touch the "no override requested" branches (the `else`/non-override paths that call `providers.get_stt(tenant)` / `providers.get_llm(tenant)` / `providers.get_tts(tenant)` / `_build_stream_provider(tenant)`) — those are the tenant's normal default-provider selection, not a silent fallback, and must be left exactly as they are.
- Do NOT touch any of the OTHER fallback patterns found in the audit this plan is scoped from (VAD algorithm selection, streaming-STT-to-batch degradation, webhook signing, softphone/call-analysis transcriber fallback, `_extract_pcm`'s silent WAV-parsing fallback) — those are explicitly out of scope, pending separate decisions.
- `static/dev_console.html` has no JS test framework in this repo — Task 2 is verified by careful code review, not an automated test.

---

### Task 1: Backend — fail loudly on a failed provider override

**Files:**
- Modify: `src/api/dev_console.py` (`factory()` inside `make_browser_bridge_factory`, currently lines ~717-746; `run_browser_voice`, currently lines ~77-100)
- Test: `tests/unit/test_factory_slots.py`

**Interfaces:**
- Consumes: `_providers()`, `_tenant()` (existing test helpers in `tests/unit/test_factory_slots.py`) and `make_browser_bridge_factory` (existing function, unchanged signature).
- Produces: no new public interface — this task only changes error-propagation behavior inside `factory()` and the close-reason string inside `run_browser_voice`. Nothing downstream depends on new names.

- [ ] **Step 1: Write the failing tests**

Open `tests/unit/test_factory_slots.py` and add these 4 tests near the existing `test_browser_factory_*` tests (reuse the file's existing `_providers()`/`_tenant()` helpers and `pytest.mark.asyncio`-style async test functions — check the top of the file for whether `pytest.mark.asyncio` is applied per-function or via `asyncio_mode = auto` in project config before deciding whether to add the decorator; match whatever the existing tests in this file already do):

```python
import pytest


async def test_browser_factory_raises_when_llm_override_fails(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    ws = SimpleNamespace(query_params={"llm": "vllm"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="VLLM_BASE_URL"):
        await factory(websocket=ws, tenant=_tenant())


async def test_browser_factory_raises_when_tts_override_fails(monkeypatch):
    monkeypatch.delenv("INDICF5_TTS_URL", raising=False)
    ws = SimpleNamespace(query_params={"tts": "indicf5"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="INDICF5_TTS_URL"):
        await factory(websocket=ws, tenant=_tenant())


async def test_browser_factory_raises_when_batch_stt_override_fails(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ws = SimpleNamespace(query_params={"stt": "groq"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        await factory(websocket=ws, tenant=_tenant())


async def test_browser_factory_raises_when_streaming_stt_override_fails(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    ws = SimpleNamespace(query_params={"stt": "deepgram"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        await factory(websocket=ws, tenant=_tenant())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_factory_slots.py -v -k raises_when`
Expected: FAIL — today's code catches the construction error and silently falls back to the tenant's default provider (a `Mock()` from `_providers()`), so `await factory(...)` returns a bridge instead of raising; `pytest.raises(ValueError, ...)` fails with "DID NOT RAISE".

- [ ] **Step 3: Remove the 4 silent-fallback blocks**

In `src/api/dev_console.py`, inside `factory()` (`make_browser_bridge_factory`), replace the current block (currently ~lines 717-746):

```python
        # STT override — deepgram is streaming; sarvam/groq are batch.
        _stream_override = None
        if stt_sel in STREAMING_STT_PROVIDERS:
            stt = providers.get_stt(tenant)
            try:
                _stream_override = get_streaming_stt_provider({"provider": stt_sel})
            except Exception as e:  # noqa: BLE001 - missing key etc.
                log.warning("dev console: streaming STT override '%s' failed (%s); using default", stt_sel, e)
                _stream_override = _build_stream_provider(tenant)
        elif stt_sel in STT_PROVIDERS:
            try:
                stt = get_stt_provider({"provider": stt_sel})
            except Exception as e:  # noqa: BLE001
                log.warning("dev console: STT override '%s' failed (%s); using default", stt_sel, e)
                stt = providers.get_stt(tenant)
            _stream_override = None   # batch selected — disable streaming path
        else:
            stt = providers.get_stt(tenant)
            _stream_override = _build_stream_provider(tenant)

        try:
            llm = get_llm_provider({"provider": llm_sel}) if llm_sel in LLM_PROVIDERS else providers.get_llm(tenant)
        except Exception as e:  # noqa: BLE001
            log.warning("dev console: LLM override '%s' failed (%s); using default", llm_sel, e)
            llm = providers.get_llm(tenant)
        try:
            tts = get_tts_provider({"provider": tts_sel}) if tts_sel in TTS_PROVIDERS else providers.get_tts(tenant)
        except Exception as e:  # noqa: BLE001
            log.warning("dev console: TTS override '%s' failed (%s); using default", tts_sel, e)
            tts = providers.get_tts(tenant)
```

with:

```python
        # STT override — deepgram is streaming; sarvam/groq are batch. An
        # explicit override that fails to construct (missing key/URL) raises —
        # it must NOT silently substitute the tenant's default, which would
        # make the console lie about which provider is actually running.
        _stream_override = None
        if stt_sel in STREAMING_STT_PROVIDERS:
            stt = providers.get_stt(tenant)
            _stream_override = get_streaming_stt_provider({"provider": stt_sel})
        elif stt_sel in STT_PROVIDERS:
            stt = get_stt_provider({"provider": stt_sel})
            _stream_override = None   # batch selected — disable streaming path
        else:
            stt = providers.get_stt(tenant)
            _stream_override = _build_stream_provider(tenant)

        llm = get_llm_provider({"provider": llm_sel}) if llm_sel in LLM_PROVIDERS else providers.get_llm(tenant)
        tts = get_tts_provider({"provider": tts_sel}) if tts_sel in TTS_PROVIDERS else providers.get_tts(tenant)
```

- [ ] **Step 4: Fix the hardcoded close-reason string**

In `src/api/dev_console.py`, inside `run_browser_voice` (currently ~lines 77-89), replace:

```python
    try:
        bridge = await _browser_bridge_factory(websocket, tenant)
    except Exception as e:  # noqa: BLE001 - e.g. tenant has no campaign configured
        log.warning("browser voice bridge build failed: %s", e)
        await websocket.close(code=1011, reason="no campaign configured for tenant")
        return
```

with:

```python
    try:
        bridge = await _browser_bridge_factory(websocket, tenant)
    except Exception as e:  # noqa: BLE001 - e.g. no campaign configured, or a provider override failed to construct
        log.warning("browser voice bridge build failed: %s", e)
        # WebSocket close reasons are capped at 123 UTF-8 bytes (RFC 6455) —
        # truncate defensively. Must reflect the REAL failure (not a generic
        # guess) so a failed provider override is visible, not silently wrong.
        await websocket.close(code=1011, reason=str(e)[:120])
        return
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_factory_slots.py -v`
Expected: all tests in the file pass (pre-existing tests unaffected, 4 new ones pass)

- [ ] **Step 6: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as the documented baseline, plus the 4 new passing tests, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/api/dev_console.py tests/unit/test_factory_slots.py
git commit -m "fix(dev-console): fail loudly on a failed provider override instead of silently substituting the default"
```

---

### Task 2: Frontend — surface the failure as a visible error banner

**Files:**
- Modify: `static/dev_console.html` (near line 307 for the new helper, lines ~508-514 and ~517 for the two call sites)

**Interfaces:**
- Consumes: Task 1's fix — the WebSocket now closes with a `reason` string containing the real error message when a bridge build fails (instead of a hardcoded, misleading string).
- Produces: `showError(text)` (new small JS helper function) — used by both the existing `{"type": "error"}` message handler and the new `ws.onclose` check. Nothing outside this file needs it.

- [ ] **Step 1: Read the current code**

Open `static/dev_console.html` and locate: the `setStatus` helper (currently ~line 307: `const setStatus = (s) => { $("status").textContent = s; };`), the `msg.type === "error"` handler inside `ws.onmessage` (currently ~lines 508-514), and `ws.onclose` (currently ~line 517).

- [ ] **Step 2: Add the `showError` helper**

Right after the `setStatus` line (~307), add:

```javascript
const showError = (text) => {
  const div = document.createElement("div");
  div.className = "turn"; div.style.color = "#fca5a5";
  div.textContent = "⚠ " + text;
  $("transcript").appendChild(div);
  $("transcript").scrollTop = $("transcript").scrollHeight;
};
```

- [ ] **Step 3: Use it in the existing error-message handler**

Replace (currently ~lines 508-514):

```javascript
    else if (msg.type === "error") {
      const div = document.createElement("div");
      div.className = "turn"; div.style.color = "#fca5a5";
      div.textContent = "⚠ " + msg.message;
      $("transcript").appendChild(div);
      $("transcript").scrollTop = $("transcript").scrollHeight;
    }
```

with:

```javascript
    else if (msg.type === "error") { showError(msg.message); }
```

- [ ] **Step 4: Surface the close reason**

Replace (currently ~line 517):

```javascript
  ws.onclose = (e) => { stop(false); setStatus("closed"); };
```

with:

```javascript
  ws.onclose = (e) => {
    if (e.reason) { showError("connection closed: " + e.reason); }
    stop(false); setStatus("closed");
  };
```

- [ ] **Step 5: Verify by reading the diff carefully**

There is no JS test framework in this repo for this file. Verify by re-reading the full modified `ws.onmessage`/`ws.onclose` block end-to-end and confirming: (a) `showError` is defined exactly once, before its first use; (b) the `msg.type === "error"` branch and the `ws.onclose` handler both call it with the exact same rendering behavior the original inline code had (a `div.turn` element, color `#fca5a5`, `"⚠ "` prefix, appended to `#transcript`, scrolled into view); (c) `e.reason` being empty/undefined (a normal, expected close) does NOT call `showError` — only a non-empty reason does.

- [ ] **Step 6: Run the full unit suite (regression check — this file isn't covered by tests, but confirm nothing elsewhere broke)**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same pass/fail counts as Task 1's final state (this task touches no Python code, so the counts must be identical to Task 1's Step 6 result).

- [ ] **Step 7: Commit**

```bash
git add static/dev_console.html
git commit -m "feat(dev-console): show the real error when the connection closes unexpectedly"
```

---

## Verification (after both tasks)

- `.venv/bin/python -m pytest tests/unit -q` — full suite green apart from the documented pre-existing failures.
- Manual: open `/dev/voice`, select a provider override known to be unconfigured on whatever environment you're testing against (e.g. `llm=vllm` if `VLLM_BASE_URL` isn't set there), start a call, and confirm the console immediately shows a visible `⚠ connection closed: ...` banner naming the real missing config, instead of silently running on a different provider.
