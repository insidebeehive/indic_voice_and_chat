# S2S bridge transcript persistence

## Problem

For a live S2S (Gemini Live) call — whether the call reaches us via a CRM
handoff (`POST /api/v1/telephony/register-call` or `.../handoff`) or via the
webconsole `/dev/bridge` test page — the per-turn caller/agent transcript text
is already computed in memory (`agent.session.turns`) but is never written to
the database. It's only used transiently to build the post-call LLM outcome
analysis, then discarded when the process moves on. There is no durable
record of what was actually said on a call.

This does **not** cover the cascade (STT→LLM→TTS) direct-dial telephony
bridges (`OutcomeRecorderMixin`, `TwilioMediaBridge`/`ExotelMediaBridge`
cascade paths, `StringeeIvrBridge`) — that call-origination code is being
removed as telephony ownership moves to the CRM. Scope is limited to the S2S
bridge path (`_BaseLiveBridge` / `TelephonyLiveBridge`), which is what
survives regardless of who initiates or hands off the call.

## Current state

- `agent.session.turns` (a list of `LLMMessage`) is populated by
  `_BaseLiveBridge._commit_turn` → `VoiceBotAgent.apply_signal`
  (`src/api/live_bridge_base.py:251`, `src/agents/voicebot.py:350`) — the same
  accumulation mechanism cascade mode uses.
- At teardown, `_emit_outcome` (`src/api/live_bridge_base.py:336`) builds an
  LLM-generated outcome summary from that same turn list and hands a payload
  dict to `self._deliver_outcome(...)`.
- `TelephonyLiveBridge._deliver_outcome` (`src/api/telephony_live_bridge.py:86`)
  forwards that payload to `call_store.deliver_to_persister(call_sid, payload)`,
  which invokes the app-level `_persist_call_outcome` closure
  (`src/main.py:283`), which calls `record_outcome()`
  (`src/api/call_store.py:245`) — this writes outcome/cost fields to the
  `conversations` row (looked up by `provider_call_sid`), but never touches
  turns.
- `GeminiLiveBridge._deliver_outcome` (`src/api/gemini_live_bridge.py:97`)
  handles the *same* payload dict for the browser dev-console mic path by
  JSON-serializing it straight to the client over the WebSocket.
- A `turns` table already exists for exactly this purpose
  (`src/models/conversation.py`, `Turn`, FK'd to `conversations`) and is
  already written to — but only from the unrelated dev-console mic flow
  (`save_turns()` called from `src/api/dev_console.py:605`).

## Design

Thread the already-computed turn list through the existing outcome-delivery
pipe and write it wherever `record_outcome()` already resolves the
conversation row. No new table, no migration — reuses `turns`/`conversations`
exactly as the dev-console path already does.

### Preconditions

`record_outcome()` only does anything if it can resolve a `conversations` row
by `provider_call_sid` (`call_store.py:262-267`) — if it can't, it logs
`"no conversation for call sid"` and returns `None` before ever reaching
`save_turns`. The row must already exist by call-teardown time. Entry points
that create it today:

- `POST /api/v1/telephony/register-call` (`telephony_crm.py:115`) — row
  created before the CRM dials.
- `POST /api/v1/telephony/handoff` (`telephony_crm.py:233`) — row created
  right after `adapter.redirect_to_stream` succeeds (`:223`); the window
  before the row exists is milliseconds and `record_outcome` only runs at
  call end, so there's no practical race.
- `dev_console.py:449` (the older single-tenant dev "place call" flow).

**`/dev/bridge` (`bridge_console.py:81` `bridge_place_call`) does NOT create
this row today** — there is no `insert_call` anywhere in `bridge_console.py`.
Since `/dev/bridge` is explicitly the surface this design is meant to cover
(the webconsole simulation of a CRM handoff), this is a prerequisite fix, not
just a caveat: add an `insert_call` call in `bridge_place_call` right after
`adapter.initiate_call(cfg)` succeeds (`bridge_console.py:170`), mirroring
`dev_console.py:445-457` — `call_id = f"call_{uuid.uuid4().hex[:16]}"`,
`provider_call_sid=session.session_id`, `mode=req.mode`,
`voice=req.voice.strip() or None`, wrapped in the same best-effort
try/except so a DB hiccup never fails the already-placed call. Without this,
turns (and outcomes) silently drop to zero on the exact path used to verify
this feature.

### Changes

1. **`bridge_console.bridge_place_call`** (`src/api/bridge_console.py:81`) —
   add the `insert_call` described above.

2. **`call_store.record_outcome()`** (`src/api/call_store.py:245`) gains an
   optional `turns: list | None = None` parameter. Immediately after the
   `row is None` guard (`call_store.py:267`) — before the outcome fields are
   mutated or `compute_call_cost`/`emit_tenant_event` run, so a later failure
   in those steps can't cost us the transcript — if `turns` is non-empty:
   `await save_turns(session, conversation_id=row.id, turns=turns)`. Also set
   `row.total_turns = <count save_turns returns>` while we're there — the
   column already exists on `Conversation` (`conversation.py:52`) and is
   currently never written.

3. **`main.py:_persist_call_outcome`** (`src/main.py:283`) passes
   `turns=payload.get("turns")` through to `record_outcome(...)`.

4. **`TelephonyLiveBridge._deliver_outcome`** (`src/api/telephony_live_bridge.py:86`)
   builds `turns = list(getattr(getattr(self._agent, "session", None), "turns", []))`
   and passes a **new, non-mutated** dict to the persister call —
   `await call_store.deliver_to_persister(self._call_sid, {**payload, "turns": turns})`
   — leaving the original `payload` object untouched. This matters because
   the same `payload` object is *also* handed to
   `dev_call_control.monitor.set_outcome(self._call_sid, payload)` on the
   line above (`telephony_live_bridge.py:88-89`), which stores it by
   reference; `GET /dev/call-status/{call_sid}` (`dev_console.py:539-546`)
   returns that stored dict straight to the browser, which polls it
   repeatedly. Mutating `payload` in place would leak raw `LLMMessage`
   dataclasses into that JSON response — and since `LLMMessage` carries an
   opaque `bytes` field (`thought_signature`, `interfaces/llm.py:39`),
   FastAPI's `jsonable_encoder` can hard-fail on it, not just render oddly.
   This is separate from the reason `turns` doesn't go into the shared
   `_emit_outcome` payload in the first place (next point) — this is about
   not letting it leak into the *monitor* copy either.

5. **Why `turns` is added in `TelephonyLiveBridge`, not in the shared
   `_emit_outcome`** (`src/api/live_bridge_base.py:336-359`): that same
   payload dict is also JSON-serialized straight to the browser by
   `GeminiLiveBridge._deliver_outcome` (`gemini_live_bridge.py:97-101`) for
   the dev-console mic path. Raw `LLMMessage` objects aren't JSON-serializable
   and `json.dumps` would raise inside a broad `except Exception` there,
   surfacing only as a misleading `"outcome computed but not delivered
   (socket closed)"` log line — a real, silent regression to the dev console.
   Keeping the `turns` merge inside `TelephonyLiveBridge` only avoids that.

6. **Turn persistence must survive an outcome-analysis failure.**
   `_emit_outcome` (`live_bridge_base.py:336-349`) wraps
   `analyze_call(...)` in a try/except that returns *before*
   `_deliver_outcome` is ever called if analysis throws (e.g. an LLM
   timeout/quota error — this project has hit exactly that failure mode
   before). `_outcome_emitted` is already set `True` by that point, so the
   `_drive` finally block won't retry. Today that only costs the outcome
   analysis; once this change ships, it would also silently cost the
   transcript. Fix: on the `except` path, still call
   `self._deliver_outcome({"type": "outcome_failed", "turns": <session turns>})`
   (or equivalent) so the transcript makes it to the persister even when
   analysis fails. `TelephonyLiveBridge._deliver_outcome` doesn't need
   `outcome`/`summary` fields to be present — `record_outcome`'s existing
   `outcome`/`summary`/`notes` params all default to `None` and are only
   applied `if not None` (`call_store.py:270-277`), so a turns-only payload
   is already handled correctly by the existing code, no further change
   needed there.

### Data flow

```
caller/agent audio
  -> _BaseLiveBridge._commit_turn (per turn, unchanged)
  -> VoiceBotAgent.apply_signal -> agent.session.turns  (unchanged)
  -> [call ends] _emit_outcome() builds LLM outcome analysis
       success -> _deliver_outcome({..., "turns": [...]})       (NEW field)
       analyze_call() throws -> _deliver_outcome({"turns": [...]}) anyway (NEW path)
  -> TelephonyLiveBridge._deliver_outcome(payload)
       monitor.set_outcome(call_sid, payload)             (unchanged, payload untouched)
       deliver_to_persister(call_sid, {**payload, "turns": turns})  (NEW, non-mutating)
  -> main.py:_persist_call_outcome(call_sid, payload)
       turns=payload.get("turns")                        (NEW)
  -> call_store.record_outcome(session, call_sid, ..., turns=turns)
       row = <lookup by provider_call_sid>                (unchanged)
       if row is not None and turns:
         save_turns(session, conversation_id=row.id, turns)   (NEW, right after the guard)
         row.total_turns = <count>                             (NEW)
  -> turns table rows, FK'd to the conversations row
```

### Error handling

Matches the existing pattern on this exact path — best-effort, never breaks
call teardown:

- `deliver_to_persister` already wraps the persister call in a broad
  `except Exception` + log.
- `save_turns` is only reachable after `record_outcome` has already resolved
  `row` by `provider_call_sid` (change #2 above places the call right after
  that guard), so the `Turn.conversation_id` FK is always satisfied. (The
  `save_turns` docstring currently claims orphaned rows are safe if the
  conversation doesn't exist — that's not actually true on Postgres given the
  `NOT NULL` FK; it only looked true because the unit-test fixture uses
  SQLite without foreign-key enforcement. Fix the docstring alongside this
  change so it doesn't mislead a future caller into invoking `save_turns`
  before resolving a row.)
- No new failure mode beyond what change #6 (analysis-failure path) already
  covers; a turn-save failure logs and the call still tears down normally.
- **Known pre-existing gap, not introduced by this change:** if the transfer
  action's coordination-server wait is cancelled
  (`TelephonyLiveBridge._on_transfer_hold`, `telephony_live_bridge.py:212-214`
  swallows `CancelledError` and returns), the subsequent `_emit_outcome` call
  in `_commit_turn` (`live_bridge_base.py:332`) has its first `await`
  re-raise that same cancellation, so `_deliver_outcome` never runs and both
  the outcome and (now) the turns for that call are lost. Worth a follow-up,
  out of scope here since it predates this change and affects the outcome
  path identically.

### Out of scope

- Cascade/direct-dial telephony bridges (`OutcomeRecorderMixin`,
  `TwilioMediaBridge`/`ExotelMediaBridge` cascade paths, `StringeeIvrBridge`)
  — being removed as call origination moves to the CRM.
- The browser dev-console mic path (`GeminiLiveBridge` + `dev_console.py`'s
  `/dev/voice` WebSocket) — already saves turns via its own
  `_run_billed_session` → `save_turns()` call; untouched by this change.
- The transfer-hold cancellation gap noted above.
- Any new schema, table, or migration — the existing `turns`/`conversations`
  tables already fit this data.

## Testing

- `tests/unit/test_call_store.py` (fixture `sm` at `:21`, helper `_conv` at
  `:42`): extend `record_outcome` coverage with
  - `turns=[...]` writes the expected `Turn` rows FK'd to the resolved
    conversation, and sets `total_turns` to match.
  - omitting `turns` (or passing `None`/`[]`) behaves exactly as it does
    today — no regression for the 5 existing call sites.
  - an **unknown** `provider_call_sid` with a non-empty `turns` list writes
    **zero** `Turn` rows (guards the precondition-ordering issue directly).
- `tests/unit/test_telephony_live_bridge.py`: extend the existing
  `_bridge()` harness (`:57`) / `test_teardown_and_outcome_publish_to_monitor`
  pattern (`:169`). Note `_bridge()` builds the bridge without `llm`, which
  short-circuits `_emit_outcome` at `live_bridge_base.py:337` — the new test
  needs a fake `llm` (or to drive `_deliver_outcome` directly) to reach the
  turns-delivery path, and a second case that forces `analyze_call` to raise
  to exercise the analysis-failure path from change #6.
- Same test: assert `monitor.get(call_sid)["outcome"]` (or equivalent) does
  **not** contain a `turns` key — pins the non-mutation fix from change #4,
  not just the browser-facing behavior.
- Add/extend a test that `GET /dev/call-status/{call_sid}` still serializes
  successfully end-to-end after a turns-bearing outcome is recorded (catches
  any FastAPI `jsonable_encoder` regression directly, rather than only
  asserting on the stored dict).
- Confirm `GeminiLiveBridge`'s outcome JSON message to the browser is
  unaffected (no `turns` key, no serialization error) — the shared
  `_emit_outcome` payload is untouched by this design, so this should already
  hold; a regression test pins it.
