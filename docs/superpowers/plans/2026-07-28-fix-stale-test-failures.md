# Fix Stale Test Failures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 13 remaining stale/misconfigured test failures on `stage` (out of 15 currently failing — 2 are known pre-existing failures, explicitly out of scope) by aligning each test's expectations with confirmed-correct current behavior, or fixing a test-fixture timing gap. No production code changes anywhere in this plan.

**Architecture:** Every item was independently verified against current source-of-truth code across two dedicated investigations (one root-causing `test_browser_bridge.py`, one confirming exact current values for the other 8 stale tests). Each fix is either: (a) updating a test's expected value to match confirmed-intentional current behavior, or (b) widening a test fixture's fake audio payload to remove a timing race — never a change to production code.

**Tech Stack:** Python 3, pytest.

## Global Constraints

- Branch is `stage` — re-verify with `git rev-parse --abbrev-ref HEAD` immediately before committing (this session had an incident where a commit landed on `main` because the working tree had silently drifted off `stage`). Do not create a new branch.
- Run `.venv/bin/python -m pytest tests/unit -q` after the task. Baseline immediately before this task (freshly measured): 15 failed, 1151 passed, 1 skipped, 0 errors. Expected after: 2 failed (the two CLAUDE.md-documented pre-existing failures below), passed count rises accordingly, zero new failures.
- Do NOT touch `tests/unit/test_chat_routes.py::test_claim_session_and_agent_ws` or `tests/unit/test_prompts.py::test_chatbot_prompt_has_scope_guardrails` — both are known, pre-existing, documented failures unrelated to this task (per `CLAUDE.md`).
- No Alembic migration. No production source-code changes anywhere in this plan — every change is test-file-only.

---

### Task 1: Fix all 13 stale test assertions and the browser-bridge fixture timing gap

**Files:**
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_health.py`
- Modify: `tests/unit/test_dev_call_control.py`
- Modify: `tests/unit/test_dev_console.py`
- Modify: `tests/unit/test_tenants_routes.py`
- Modify: `tests/unit/test_prompts.py`
- Modify: `tests/unit/test_browser_bridge.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the only task). No production code is read or modified — every change targets test-file assertions or test-fixture data.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Confirm you're on the right branch**

```bash
git rev-parse --abbrev-ref HEAD
```
Expected: `stage`. If it prints anything else, stop and report back — do not proceed or switch branches yourself.

- [ ] **Step 2: Fix `tests/unit/test_config.py::test_loads_default_yaml`**

Current:
```python
def test_loads_default_yaml() -> None:
    s = load_settings()
    assert s.app.name == "vox-agent"
    assert s.app.version == "1.0.0"
    assert s.pipeline.stt.provider == "sarvam"
    assert s.pipeline.llm.provider == "groq"
    assert s.pipeline.tts.provider == "sarvam"
    assert s.pipeline.telephony.provider == "twilio"
    assert s.pipeline.vector_store.provider == "faiss"
    assert s.pipeline.vector_store.embedding_dim == 384
```

`config/default.yaml`'s `llm.provider` is now `"gemini"` (changed by commit `fe56ed9`) and `vector_store.provider` is now `"pgvector"` (changed by commit `7acb491` — the yaml even has `# provider: faiss` commented out directly below the live `pgvector` line). Change to:

```python
def test_loads_default_yaml() -> None:
    s = load_settings()
    assert s.app.name == "vox-agent"
    assert s.app.version == "1.0.0"
    assert s.pipeline.stt.provider == "sarvam"
    assert s.pipeline.llm.provider == "gemini"
    assert s.pipeline.tts.provider == "sarvam"
    assert s.pipeline.telephony.provider == "twilio"
    assert s.pipeline.vector_store.provider == "pgvector"
    assert s.pipeline.vector_store.embedding_dim == 384
```

- [ ] **Step 3: Fix `tests/unit/test_health.py::test_health_reports_provider_names`**

Find this block (it's inside `test_health_reports_provider_names`):
```python
    assert body["platform_defaults"] == {
        "stt": "sarvam",
        "llm": "groq",
        "tts": "sarvam",
        "telephony": "twilio",
        "vector_store": "faiss",
    }
```

Change to:
```python
    assert body["platform_defaults"] == {
        "stt": "sarvam",
        "llm": "gemini",
        "tts": "sarvam",
        "telephony": "twilio",
        "vector_store": "pgvector",
    }
```

- [ ] **Step 4: Fix `tests/unit/test_dev_call_control.py::test_override_set_and_pop_is_one_shot`**

Current:
```python
def test_override_set_and_pop_is_one_shot():
    dcc.set_override("dev", mode="s2s", voice="Kore", lead_name="Raju")
    assert dcc.pop_override("dev") == {"mode": "s2s", "voice": "Kore", "lead_name": "Raju"}
    assert dcc.pop_override("dev") is None
    assert dcc.pop_override("never-set") is None
```

`set_override`/`pop_override` (`src/api/dev_call_control.py`) gained 3 new always-present fields since this test was written — `caller_name`, `lead_gender`, `transfer_webhook_url` (all default `""`). Change to:

```python
def test_override_set_and_pop_is_one_shot():
    dcc.set_override("dev", mode="s2s", voice="Kore", lead_name="Raju")
    assert dcc.pop_override("dev") == {
        "mode": "s2s", "voice": "Kore", "caller_name": "", "lead_name": "Raju",
        "lead_gender": "", "transfer_webhook_url": "",
    }
    assert dcc.pop_override("dev") is None
    assert dcc.pop_override("never-set") is None
```

- [ ] **Step 5: Fix `tests/unit/test_dev_console.py::test_dev_voice_page_served`**

Current:
```python
def test_dev_voice_page_served():
    app = FastAPI()
    app.include_router(dev_router)
    client = TestClient(app)
    resp = client.get("/dev/voice")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Voice Dev Console" in resp.text
```

`static/dev_console.html`'s title/heading is now "Voice Demo Console" (both `<title>` and `<h1>`). Change the last line:

```python
def test_dev_voice_page_served():
    app = FastAPI()
    app.include_router(dev_router)
    client = TestClient(app)
    resp = client.get("/dev/voice")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Voice Demo Console" in resp.text
```

- [ ] **Step 6: Fix `tests/unit/test_dev_console.py::test_dev_voices_per_mode`**

Find the final 4 assertions in this test:
```python
        assert data["layered"]["voices"] == ["anushka", "karun"]
        assert data["layered"]["default"] == "anushka"
        assert data["s2s"]["voices"] == ["Aoede", "Kore", "Leda"]
        assert data["s2s"]["default"] == "Aoede"
```

`/dev/voices`'s `dev_voices` handler (`src/api/dev_console.py`) deliberately always returns the FULL Gemini Live voice catalog for `s2s.voices` now (its own docstring: "S2S: always show the full 30-voice catalog so devs can try any voice"), ignoring the tenant's `allowed_voices` config this test sets. `layered.*` and `s2s.default` are unaffected. Change to:

```python
        assert data["layered"]["voices"] == ["anushka", "karun"]
        assert data["layered"]["default"] == "anushka"
        assert data["s2s"]["voices"] == [
            "Aoede", "Kore", "Leda", "Puck", "Charon", "Fenrir", "Orus", "Zephyr",
            "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Autonoe",
            "Callirrhoe", "Despina", "Enceladus", "Erinome", "Gacrux", "Iapetus",
            "Laomedeia", "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager",
            "Schedar", "Sulafat", "Umbriel", "Vindemiatrix", "Zubenelgenubi",
        ]
        assert data["s2s"]["default"] == "Aoede"
```

- [ ] **Step 7: Fix `tests/unit/test_dev_console.py::test_place_call_uses_selected_provider_and_its_caller_id`**

Find the final assertion in this test:
```python
        assert dev_call_control.pop_override("dev") == {
            "mode": "s2s", "voice": "Kore", "lead_name": "Raju"}
```

Same root cause as Step 4 (shared `pop_override` dict shape). Change to:

```python
        assert dev_call_control.pop_override("dev") == {
            "mode": "s2s", "voice": "Kore", "caller_name": "", "lead_name": "Raju",
            "lead_gender": "", "transfer_webhook_url": "",
        }
```

- [ ] **Step 8: Fix `tests/unit/test_tenants_routes.py::test_list_tenants_shows_mode_and_models`**

Find:
```python
    assert acme["mode"] == "layered"
```

`RegisterTenantRequest.mode`'s default was deliberately changed to `"s2s"` in commit `b9fa799` ("Mode defaults to s2s"). This test's `_body()` helper never passes `mode` explicitly, so it picks up the new default. Change to:

```python
    assert acme["mode"] == "s2s"
```

Leave every other assertion in this test unchanged (the `llm`/`tts` lines are already correct — `_body()` passes those explicitly).

- [ ] **Step 9: Fix `tests/unit/test_tenants_routes.py::test_tenant_analytics_and_billing`**

Find these two lines:
```python
    assert an["by_channel"] == {"voice": 2, "webconsole": 1, "softphone": 1}
    assert an["by_provider"]["twilio"] == 3 and an["by_provider"]["none"] == 1
```

`tenant_analytics` (`src/api/tenants.py`) deliberately remaps `channel == "webconsole"` to `"voice"` in `by_channel` (inline comment: `# webconsole is a browser transport, not a channel — treat as "voice"`), and buckets it under `"webconsole"` (not `"none"`) in `by_provider` (inline comment: `# webconsole has no telephony provider — bucket it explicitly so totals match`). Given this test's seed data (c1/c2 = voice+twilio, c3 = webconsole+no provider, c4 = softphone+twilio), change to:

```python
    assert an["by_channel"] == {"voice": 3, "softphone": 1}
    assert an["by_provider"]["twilio"] == 3 and an["by_provider"]["webconsole"] == 1
```

Leave the rest of the test unchanged — the sum-to-`total_calls` invariant check later in the same test (`for key in (...): assert sum(an[key].values()) == an["total_calls"]`) already covers `by_channel`/`by_provider` generically and needs no change (3+1=4 and 3+1=4 both still sum correctly).

- [ ] **Step 10: Fix `tests/unit/test_prompts.py::test_gender_directive_enforces_feminine_in_both_prompts`**

Current:
```python
def test_gender_directive_enforces_feminine_in_both_prompts() -> None:
    # Language-agnostic now: holds in whatever language the agent is speaking,
    # not just Hindi (so it survives a switch to Marathi etc.).
    from src.dialogue.prompts import build_s2s_system_instruction
    script = VoiceBotScript.from_campaign_yaml({**SCRIPT, "gender": "female"})
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    for instr in (build_s2s_system_instruction(script, schema),
                  build_voicebot_system_prompt(script, schema)):
        assert "FEMALE" in instr
        assert "feminine grammatical forms" in instr
        assert "masculine" in instr        # named as the form to avoid
```

The Jul 9 prompt redesign intentionally downgraded "FEMALE" (all-caps) to "female" (regular case), and the redesigned female-gender directive text no longer mentions "masculine" as a contrast at all (verified by running this test directly against current code). Change to:

```python
def test_gender_directive_enforces_feminine_in_both_prompts() -> None:
    # Language-agnostic now: holds in whatever language the agent is speaking,
    # not just Hindi (so it survives a switch to Marathi etc.).
    from src.dialogue.prompts import build_s2s_system_instruction
    script = VoiceBotScript.from_campaign_yaml({**SCRIPT, "gender": "female"})
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    for instr in (build_s2s_system_instruction(script, schema),
                  build_voicebot_system_prompt(script, schema)):
        assert "female" in instr
        assert "feminine grammatical forms" in instr
```

- [ ] **Step 11: Fix the `test_browser_bridge.py` fixture timing gap (fixes all 4 failures)**

Find `FakeAgent.play_opening` (near the top of the file):
```python
    async def play_opening(self, sink):
        self.opening_played = True
        self.session.turns.append(type("Msg", (), {"role": "assistant", "content": "Namaste! Main Priya."})())
        await sink(b"\x10\x11")  # fake opening audio
```

Root cause (already investigated, confirmed NOT a regression — this bug predates this session, from an old commit `8f531b0`): `BrowserVoiceBridge._on_pcm_frame` drops inbound mic frames while `time.monotonic() < self._play_until` (an echo gate against the agent's own TTS). `_play_until` is set in `_send_pcm` to `now + len(pcm16)/2/pcm_sample_rate` (16000 Hz) for ANY non-empty payload. **Correction from an earlier draft of this step:** widening the fake clip does NOT fix this — it makes it worse. Any non-empty payload sets `_play_until` to a real future timestamp; since this test's entire 35-message flood executes in a handful of microseconds of real wall-clock time (a plain list `.pop(0)`, no artificial delay), a LARGER `_play_until` window only guarantees MORE of the flood lands before it expires, not less (empirically verified: 24/35 frames blocked with a 2-byte clip, but 35/35 blocked with a 16000-byte clip). The actual fix is to make the fake clip **empty** — `_send_pcm` has `if not pcm16 or self._stopped: return` before it ever touches `_play_until`, so an empty payload is a true no-op for the echo gate:

```python
    async def play_opening(self, sink):
        self.opening_played = True
        self.session.turns.append(type("Msg", (), {"role": "assistant", "content": "Namaste! Main Priya."})())
        # Empty payload: BrowserVoiceBridge._send_pcm's `if not pcm16: return` makes
        # this a true no-op, so it never touches _play_until (the post-opening echo
        # gate). Verified empirically that ANY non-empty fake clip — including a
        # larger one — sets _play_until to a real future timestamp that this test's
        # zero-wall-clock message flood cannot outrun (all 35 follow-on frames get
        # dropped as false-positive echo before the gate ever expires), which is
        # what was silently swallowing the test's speech frames.
        await sink(b"")  # fake opening audio (empty; no echo-gate window)
```

This is the only change needed in this file. `TerminalFakeAgent` and `ErroringAgent` both subclass `FakeAgent` and inherit `play_opening` unchanged, so this single change fixes all 4 failing tests (`test_run_handshake_plays_opening_and_processes_a_turn`, `test_terminal_agent_stops_run_loop`, `test_subframe_chunks_are_assembled_then_endpointed`, `test_bridge_emits_error_event_on_turn_error`). Do NOT modify `src/api/browser_bridge.py` — the source code is correct as-is.

- [ ] **Step 12: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 2 failed, 1 skipped, 0 errors — only `test_chat_routes.py::test_claim_session_and_agent_ws` and `test_prompts.py::test_chatbot_prompt_has_scope_guardrails` remain failing (both known, pre-existing, out of scope). Passed count rises from 1151 by roughly 13 (one per fixed test). Zero new failures anywhere.

Also specifically confirm `tests/unit/test_browser_bridge.py`'s previously-passing test `test_subframe_chunks_do_not_endpoint_too_early` still passes:

Run: `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v`
Expected: all tests in this file pass, including `test_subframe_chunks_do_not_endpoint_too_early` (it should now pass for the correct reason — its 10 silence frames genuinely fall below the 600ms endpoint threshold — rather than accidentally passing because the old echo-gate bug happened to drop its speech frames too).

- [ ] **Step 13: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # must print "stage" — stop if it doesn't
git add tests/unit/test_config.py tests/unit/test_health.py tests/unit/test_dev_call_control.py tests/unit/test_dev_console.py tests/unit/test_tenants_routes.py tests/unit/test_prompts.py tests/unit/test_browser_bridge.py
git status --short
```

Confirm the status output shows exactly these 7 modified test files, no unrelated changes. Then commit:

```bash
git commit -m "$(cat <<'EOF'
test: fix stale assertions and a fixture timing gap across 7 test files

Every one of these tests was asserting an old expected value from before a
legitimate, already-shipped behavior change (LLM default groq->gemini,
vector store faiss->pgvector, dev-console page rename, S2S voice catalog
now showing all 30 voices, tenant mode default layered->s2s, webconsole
channel/provider remapping, dev-console override dict gaining 3 fields,
and the Jul 9 prompt redesign's FEMALE->female downgrade) plus one
test-fixture timing race in test_browser_bridge.py (a 2-byte fake opening
clip gave the echo gate a ~62us window that raced against the test's
zero-delay frame flood). No production code changed.
EOF
)"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — 2 failed (both known pre-existing, out of scope), 0 errors, passed count up from 1151.
- `.venv/bin/python -m pytest tests/unit/test_browser_bridge.py -v` — all pass.
- `git diff --stat` (before committing) touches only the 7 test files listed above — no production source files.
- `git status --short` clean except the pre-existing, unrelated untracked `docs/voice-recording-scripts*.md` files — do not touch those.
