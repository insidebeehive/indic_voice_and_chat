# ChatBot Turn-Metrics Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status: implemented, all three phases.** §11's open questions (child table vs. aggregate-only, retention, Prometheus scope) were settled with the project owner and are recorded in §11 below. See §10 for the phase-by-phase breakdown.

**Goal:** Give ChatBot the aggregate latency/failure insight VoiceBot already has. Resolves Open Question #5 of `docs/superpowers/plans/2026-09-07-llm-conversation-tracing.md`.

**The gap, precisely:** voice has `TurnMetric` → `/benchmarks/turn-metrics/summary` → Prometheus → Grafana. Chat has none of it — but the agent *already computes* per-LLM-call latencies, per-tool-call latencies with tool names, round count and retrieval count on every turn, and throws them away into a formatted log string. `ChatMessage.latency_ms` is a declared-but-never-written dead column. This is a persistence-and-aggregation problem, not an instrumentation-from-scratch problem.

---

## 1. Verified current state

- **`ChatMessage.latency_ms` is genuinely dead** (`src/models/chat.py:68`, `alembic/versions/0004_chat_module.py:52`). Repo-wide grep finds no writer, no reader, no API exposure.
- **Tokens/cost are already solved — do not duplicate.** `src/api/chat.py::_persist_turn` (2082) writes `input_tokens`/`output_tokens`/`cost`/`llm_provider`/`llm_model` per message and accumulates onto `ChatSession`, via `compute_chat_turn_cost` (`src/api/chat_cost.py:35`) on a deliberately separate DB session. `GET /tenants/{id}/billing` reads those totals. **The new tables carry no token or cost columns.**
- **`_persist_turn` is the single funnel for every chat entry point** (WS text/audio/image, `POST /chat/message`, `POST /chat/{id}/upload`, `process_message`). It holds `row.tenant_id`. It receives a `ChatTurnResult` that currently carries no timing.
- **WS layer and agent each hold timing the other lacks.** WS owns the per-turn `trace_id` (`src/api/chat.py:1709`, scoped 1727), the duplicate-turn guard (`_try_begin_turn`, 1711), the 90s `_TURN_TIMEOUT_S` wrapper (`_run_turn`, 148), and media fetch/transcription time. The agent owns per-LLM-call and per-tool-call timing, rounds, retrieval, guards, the failure directive.
- **Guards don't report whether they fired.** `apply_hallucination_guard` (`src/rag/context_builder.py:243`) always returns a fresh copy, so identity comparison is useless; the other two return the same object when inert. A "fired" signal needs a before/after `(response_text, confidence)` snapshot.
- **Alembic head is `0019_turn_metrics_created_idx`.** Migrations are hand-written by convention (autogenerate picks up unrelated tables from other apps sharing the Neon DB — documented in `0016`/`0017`/`0018`).
- **Two constraints not previously noted:** (a) `src/models/__init__.py` eagerly imports every model module — a new model **must** be registered there or `Base.metadata.create_all` in `tests/conftest.py:51` won't create the table for the unit suite; (b) `ChatSession.id` **is** the WebSocket capability, so no read endpoint may ever return a raw `session_id`.

## 2. Table design: two new sibling tables, not an extended `TurnMetric`

**Recommended (high confidence): `chat_turn_metrics` (one row per turn) + `chat_tool_metrics` (one row per tool call).**

The decisive argument is **grain**. Chat's highest-value metric — per-tool-call latency with the tool name, given the documented `/players/{id}/transactions` timeout saturation and the #1762 incident — is 0..N rows *per turn*. There is no honest way to store a variable-length list of `(tool_name, latency, outcome, budget_slice)` in a per-turn row and still answer *"p95 latency and timeout rate for `get_player_transactions` this week"* with a `GROUP BY`. That question is the point of this work.

Supporting arguments:
- **Mixing products would silently corrupt the live voice read paths.** `src/api/benchmarks.py:203` groups by `(mode, stt_provider, llm_provider, tts_provider)` with **no** `WHERE` clause; chat rows would appear as a fictional "combo" with NULL STT/TTS and zeros in five of seven latency columns. `turn_metrics_push.py` has the same unfiltered shape and would inflate counts and emit STT gauges for chat rows. Both would need a `product` filter — so the claimed "reuse" evaporates either way, except in the shared-table version you edit production-validated voice code *under regression risk*.
- **The columns are wrong in both directions.** Dead for chat: `stt_provider`, `tts_provider`, `stt_latency_ms`, `tts_first_chunk_ms`, `tts_total_ms`, `tts_segments_dropped`, `campaign_id`, and `mode` in its voice sense. Missing for chat: rounds, per-tool detail, retrieval, guard verdicts, failure directive, escalation, budget exhaustion. Shared = 7 dead columns + ~14 nullable additions.
- **`llm_ttft_ms` has no chat meaning.** Chat calls `generate()` non-streaming everywhere; there is no first-token event. Sharing that column between a real measurement and a duplicate of `llm_total_ms` is actively misleading.
- **Migration risk.** Two `CREATE TABLE`s on tables nothing reads yet beats a ~14-column `ALTER` on a live, continuously-written, growing table.

**Why not a JSONB detail column:** it destroys the read path. `GROUP BY tool_name` becomes `jsonb_array_elements` gymnastics, percentiles over a JSON array are painful, and SQLite (which the entire unit suite runs on) has no comparable story. Every read path here is plain SQLAlchemy aggregation.

**Honest counter-argument, so it can be overruled knowingly:** two tables is more surface, and the parent is the *less* valuable half. The cheaper variant is a **single `chat_turn_metrics` with aggregate-only tool columns** (`tool_calls`, `tool_failures`, `tool_timeouts`, `tool_total_ms`, `slowest_tool_name`, `slowest_tool_ms`) — answers "how bad, per tenant" and "worst offender today," but *cannot* give per-tool p95 or per-tool timeout rate, which is exactly the number you'd want when arguing with the CRM team. Recommendation is to pay for the child table; the aggregate-only variant is a legitimate fallback, not a wrong answer. **This is open question #1.**

## 3. What to capture — and what to omit

### `chat_turn_metrics` (one row per agent turn)

`id` PK · `tenant_id` (FK `tenants.id` CASCADE, indexed) · `crm_id` (nullable — tool health is a CRM-level property; available as `tenant.settings.crm_id`) · `session_id` · **`trace_id`** (nullable, indexed) · `path` (`tools` | `single_shot` — deliberately *not* named `mode`) · `llm_provider` / `llm_model` · `action` · `total_ms` · `llm_total_ms` / `llm_calls` · `tool_total_ms` / `tool_calls` / `tool_failures` / `tool_timeouts` / `tool_calls_skipped` · `kb_search_ms` / `kb_searches` / `retrieved_chunks` · `rounds` · `rounds_exhausted` · `retry_fired` · `failure_directive_fired` · `failure_directive_escalated` · `guard_hallucination_fired` / `guard_no_grounding_fired` / `guard_unverified_data_fired` · `escalated` · `created_at` (indexed).

Notes on the non-obvious ones:
- **`trace_id` is the highest-leverage single column here.** Read from `src/utils/trace_id.py::current_trace_id()` — zero new plumbing, already a ContextVar scoped around the whole turn. Turns "this row is the slow one" into "here are its Loki log lines" (and later its Phoenix trace).
- **`kb_search_ms` stays separate from `tool_total_ms`**, mirroring the code's own deliberate budget separation (`_KB_SEARCH_TIMEOUT_S` vs `_TOOL_BUDGET_S`). Folding them would make `tool_total_ms` incomparable to the 45s budget.
- **`tool_calls_skipped`** captures `_exec_tool`'s `timeout_s < _TOOL_MIN_SLICE_S` branch — today only a log warning. "The model asked for a tool and we never even tried."
- **`rounds_exhausted`** is the `for...else` forced-final-answer branch: the model still wanted tools and got cut off. A real degradation mode, currently invisible.
- **`failure_directive_fired`** is the #1762 signal.
- Integer columns are `NOT NULL default 0`, copying `TurnMetric`'s convention so `AVG`/percentiles never need coalescing.
- **Index both `created_at` alone and `(tenant_id, created_at)` from day one.** Migration `0019` exists *because* the standalone `created_at` index was missed and the push loop full-scanned a growing table forever. Don't repeat it.

### `chat_tool_metrics` (one row per tool call)

`id` · `turn_id` (FK CASCADE, indexed) · `tenant_id` (denormalized, so tenant-scoped tool queries need no join — `TurnMetric` denormalizes the same way) · `tool_name` · `kind` (`crm` | `kb` | `local` | `deposit_verification`, so zero-I/O local builders are excludable) · `latency_ms` · `outcome` (`ok` | `timeout` | `transport_error` | `error` | `skipped_budget`) · `budget_slice_ms` · `round_index` · `created_at`. Indexes: `(turn_id)`, `(tool_name, created_at)`.

**`budget_slice_ms` paired with `latency_ms` is the most diagnostic pair in the design** — it distinguishes "given 35s and used all of it" from "given 3s because an earlier call ate the budget." Exactly what made #1762 painful to read out of logs.

Outcome is already derivable from `_tool_result_is_failure` plus the result dict (`failure == "timeout"`, `failure == "transport_error"`, `status == "error"` with neither key for deposit-verification's own shape, else `ok`).

### Explicitly omitted

`llm_ttft_ms` (no first-token event on this path) · tokens/cost (already persisted; two sources of truth that can disagree is worse than one join) · per-round LLM latency detail (`llm_ms_list` is ≤~4 entries; sum+count gives the mean) · STT/TTS anything · `call_offered` (product-funnel, derivable) · detected language / `response_chars` / parse-error kind (no ops question they answer that `retry_fired` and the guard booleans don't) · tool endpoint URL / HTTP status / response size (endpoint templates are tenant config and can embed identifiers; PII surface for no gain) · **duplicate-turn drops** — structurally impossible here, since `_try_begin_turn` rejects the frame *before* an agent turn exists. That stays a WS-layer concern; stated explicitly so nobody later "fixes" it by inventing a synthetic row.

## 4. Write path

**Recommended (high confidence): mirror the voice inversion-of-control pattern exactly** — inject a `record_metric` callback from `make_chatbot_factory`; the agent never imports a model.

```
src/agents/chatbot.py
  + @dataclass(frozen=True) ChatTurnMetrics       # next to ChatTurnResult (331)
  + ChatTurnResult.metrics: ChatTurnMetrics | None = None
  + ChatBotAgent.__init__(..., record_metric: Callable[[dict], Awaitable[None]] | None = None)
  ~ _single_shot (452) / _handle_with_tools (516) — assemble, emit, attach
```

Emit at the **two existing `log.info("chat turn done...")` sites** (503, 783) — already the "turn complete, everything in scope" point, and it keeps the log line and the metric row in lockstep. Wrap in the established idiom from `src/agents/voicebot.py:733-749`:

```python
if self._record_metric is not None:
    try:
        await self._record_metric({...})
    except Exception:  # noqa: BLE001 - never break a live turn on a metrics-write failure
        log.warning("record_metric failed; continuing without persistence", exc_info=True)
```

Two layers, same as voice: `record_chat_turn_metric` never raises internally, and the call site catches anyway.

**Why IoC rather than writing from `_persist_turn` or the WS layer:** `src/agents/chatbot.py` currently imports **zero** model modules — importing one would make its ~50 unit tests DB-aware. `tenant_id`/`crm_id` are facts of the factory closure, not the agent. All per-tool/per-round data exists only inside `_handle_with_tools`; writing from `_persist_turn` would mean widening `ChatTurnResult` with the whole structure anyway, and would miss turns where persistence returns early.

Wiring in `src/bootstrap.py::make_chatbot_factory` (405), parallel to the voice sites at 653/919/1085/1167:

```python
record_metric=lambda payload: record_chat_turn_metric(
    tenant_id=tenant.id, crm_id=getattr(tenant.settings, "crm_id", None), **payload),
```

`getattr`-defensive, matching `_crm_retriever_for` (152) and the `_llm_defaults` comment (558-560) about tests stubbing a bare `SimpleNamespace`.

**Guard-verdict detection:** snapshot `(response_text, confidence)` immediately before/after each of the three guard calls. Necessary because `apply_hallucination_guard` always returns a copy. Do **not** change the guards to return verdict objects in this pass — shared, heavily-tested pure functions; the snapshot is 2 lines per site. (Noted as a future cleanup, open question #6.)

**`failure_directive_fired`/`_escalated`** need turn-scoped flags hoisted out of the `if failed_category_names:` block (645/649) — deriving from `directive_index` at the end is wrong, since the loop can `break` at 577 before the directive block runs.

**Known Phase 2 gap, deliberately accepted: a turn that raises or times out writes no row** — the worst turns are invisible. Do **not** fix with `try/finally` inside the agent: `_run_turn`'s `asyncio.wait_for` *cancels* the coroutine, and awaiting a DB insert in a `finally` on the cancellation path is unreliable by construction. The correct fix is a minimal failure row from the **WS/HTTP layer's** timeout/error handler, which knows tenant/session/trace-id/elapsed and isn't itself being cancelled. That's Phase 3.

**`src/dialogue/prompts.py` is untouched** — the agent change adds a constructor param and observation only; `_build_failure_directive` is *observed*, never edited. No CLAUDE.md confirmation gate is triggered.

## 5. Read path

**Recommended: one new admin endpoint where an ops person already looks.** `GET /api/v1/tenants/{tenant_id}/chat-turn-metrics?window_h=24` in `src/api/tenants.py`, beside the existing `chat-analytics` (1061) and `billing` (1125) — same `Depends(require_admin)` + `_require_tenant` shape, same aggregate-only style. Two blocks:

- **turns**: `samples`, avg/p50/p95 `total_ms`, `avg_llm_total_ms`, `avg_tool_total_ms`, `avg_kb_search_ms`, `avg_rounds`, and rates for `rounds_exhausted` / `retry_fired` / `failure_directive_fired` / `guard_*_fired` / `escalated`, plus `turns_with_tool_failure`.
- **tools**: per `tool_name` — `calls`, p50/p95 `latency_ms`, `timeouts`, `transport_errors`, `skipped`, `avg_budget_slice_ms`, `failure_rate_pct`.

Aggregate count/sum/avg in SQL (portable to SQLite); compute percentiles in Python **reusing `turn_metrics_push.py::_percentile`** (already a tested pure function). Cap the fetch (~50k rows) so a large `window_h` can't be a DoS.

**Do not extend `/benchmarks/turn-metrics/summary`, and don't add a chat sibling to it.** Its response model is voice-shaped and its grouping key is the provider combo — and **chat has no provider combo to compare**: `make_chatbot_factory` always uses `get_platform_llm()` with global defaults and no per-tenant override (its own docstring says so). Cross-provider chat benchmarking has nothing to vary until someone deliberately A/B tests a model; add it then, with a real question to answer.

**Never return `session_id` or `trace_id` from a tenant-facing surface** — both are capabilities/correlators. Aggregates only, admin-gated, asserted in a test rather than left as a comment.

**Prometheus push: Phase 3, and a sibling module, not an extension.** `src/observability/chat_metrics_push.py`, job name `vox_chat_turn_metrics`. It must be separate because (a) `_GROUP_LABELS` is voice's combo tuple while chat wants `(tenant_id, path)` and `(tenant_id, tool_name, outcome)`; (b) stage names differ entirely; (c) decisively — a Pushgateway PUT **replaces the whole job/group's prior state**, which the existing module relies on to clear stale data, so a shared job name would let one product's empty window wipe the other's live data. Reuse by importing `_percentile` and parameterizing the hardcoded `_JOB_NAME` in `_push` (current value as default). Call the new aggregator from the *existing* `_push_turn_metrics_loop` (`src/main.py:183`) sequentially — no second task, no new config knob. Cardinality note: `tenant_id × tool_name × outcome` is fine at today's ~1 live tenant and ~23 catalog tools; drop `tenant_id` from the tool gauge first if tenant count grows.

Push is what makes this **alertable** ("`get_player_transactions` timeout rate > 20% over 15m") — the actual ops win over a JSON endpoint someone must remember to open.

## 6. Reconciliation with the tracing plan

**This work invalidates §3.4 reason #3 of `2026-09-07-llm-conversation-tracing.md`**, which reads *"ChatBot has zero latency-metrics pipeline today … tracing adds genuinely new coverage on chat."* After this, that sentence is false and must be struck, or the tracing plan will keep justifying itself with a gap that no longer exists.

**Tracing becomes less urgent but not redundant — and its justification gets sharper:**

| Question | Answered by |
|---|---|
| "Is `get_player_transactions` saturating its timeout for tenant X this week?" | **This plan.** Tracing answers it badly — sampled, retention-bounded, hand-counted in a UI |
| "p95 chat turn latency, and how much is tools vs. LLM?" | **This plan** |
| "How often does the TOOL FAILURE directive fire, and does a guard then rewrite the reply?" | **This plan** (booleans) — previously listed as tracing-only in its §2.3 |
| "What exactly did the bot say on #1762 turn 3, and what did the model see?" | **Tracing only.** This plan stores no content, permanently, by design |
| "Iterate the prompt against real captured traffic" (datasets, playground) | **Tracing only** |

So tracing's honest marginal value after this work is **per-conversation payload replay and the prompt-iteration workflow** — narrower, but far more defensible than "we have no latency data," and a *better* position for that plan, since tracing would have done the aggregate-latency job badly (sampled + 30-day retention + no SQL is a poor latency store).

The two are complementary **by construction, not by hope**, because of the `trace_id` column: one id gets you the metric row, its Loki lines, and later its Phoenix trace. Cheapest thing in this plan, most important for keeping the systems from drifting into overlap.

**Recommended doc-only edits to the tracing plan:** mark **Open Question #5 resolved** ("Yes — chat gets its own metrics pipeline, deliberately; tracing is not the answer to the aggregate-latency gap"); strike/rewrite §3.4 reason #3; add both tables to §3.5's keep-vs-retire list; note in §2.3 that guard verdicts are now SQL-queryable booleans so tracing's version is narrative context rather than measurement; reduce the *urgency* framing but not the plan — Phase 0 (`trace_redaction.py`, committed `6f3160a`) stands on its own merits regardless.

## 7. PII

Holds voice's line exactly. **Stored:** ids, tool *names*, provider/model names, a fixed set of enum strings, integers, booleans. **Never stored:** tool arguments, tool response bodies, message text, LLM completion text, system prompts, RAG chunk text, `customer_id`/player id, phone/email/UPI, media bytes or URLs, CRM endpoint URLs, and — importantly — **no free-text error strings**. `outcome` is a bounded enum *specifically* so a CRM error message (which can and does embed player data) never has a column to land in. Any future column that could hold free text must be rejected on that basis.

`src/observability/trace_redaction.py` is **not needed here, and importing it would be a design smell** — it would signal content is entering the table. The correct property is that there is nothing to redact.

One code-derived constraint for the model docstring: `session_id` is the live WS capability for that session. Fine to store (`chat_messages` and `TurnMetric` already store session ids) but it must never be returned by a read endpoint — hence §5's aggregate-only, admin-gated design and the route test asserting no session id appears in the body.

## 8. Migration

Single hand-written `alembic/versions/0020_chat_turn_metrics.py`, `down_revision = "0019_turn_metrics_created_idx"`. Two `create_table` + four `create_index`; `downgrade()` drops indexes then tables in reverse (child first). Follow `0011_turn_metrics.py` for column spelling (`nullable=False, server_default="0"` on integer counters; `server_default=sa.func.now()` on `created_at`) and include the standard hand-written-not-autogenerated note from `0016`/`0017`/`0018`.

Two implementation notes: `src/models/__init__.py` must import/re-export the new models (§1) or the test suite won't create the tables; and because `record_chat_turn_metric` swallows everything, a pre-migration deploy degrades to "no rows, one WARNING per turn" rather than an outage — but still tell the owner the ordering (apply `0020` manually, then deploy).

## 9. Testing

Baseline: re-verify `.venv/bin/python -m pytest tests/ -q` before starting (CLAUDE.md notes it drifts) and again at the end; no new failures. Do **not** `pip install -e ".[dev]"` to "fix" the pgvector failure — that changes the baseline mid-task.

1. **`test_chat_turn_metrics_model.py`** — mirrors `test_turn_metrics_model.py`. Round-trip parent + 3 children; CASCADE on parent delete; **never-raises** property via a monkeypatched exploding `get_sessionmaker`.
2. **`test_chatbot_metrics.py`** — uses `test_chatbot_tools.py`'s `ScriptedLLM` + `_agent()` with a `RecordingMetric` callback. Cases: 2-round turn → `rounds=2` + correct children; `tool_total_ms` **excludes** `search_knowledge_base` while `kb_search_ms` includes it; timing-out CRM executor → `outcome="timeout"` + `failure_directive_fired`; second consecutive failing turn on the same agent → `failure_directive_escalated`; budget-exhausted skip → `skipped_budget` + `tool_calls_skipped=1`; `for...else` → `rounds_exhausted`; unusable first response → `retry_fired`; ungrounded ₹ figure → `guard_unverified_data_fired`; single-shot → `path="single_shot"`, `kb_searches=0`.
3. **`test_chatbot_metrics_never_raises.py`** — exploding callback drives a full two-round turn; asserts the `ChatTurnResult` is field-for-field identical to the same run with `record_metric=None`. (The tracing plan's `ExplodingTracer` idea, applied here.)
4. **`test_chat_metrics_route.py`** — mirrors `test_benchmarks_turn_metrics_route.py`. 401 without admin; correct per-tool grouping/percentiles; `window_h` excludes out-of-window rows; **no `session_id`/`trace_id` substring anywhere in the response body**.
5. **`test_chat_metrics_push.py`** (Phase 3) — mirrors `test_turn_metrics_push.py` with `respx`. Asserts the PUT lands on `.../metrics/job/vox_chat_turn_metrics` (distinct from `vox_turn_metrics`), an empty window still pushes to clear stale gauges, and `_push`'s default job name is unchanged for the voice caller.
6. **Regression guard — the executable form of §2's central argument.** Seed `chat_turn_metrics` rows, then assert `GET /benchmarks/turn-metrics/summary` output is byte-identical to the un-seeded case.

## 10. Phased rollout

- [x] **Phase 1 — Structured turn telemetry, no schema change. Independently valuable, near-zero risk.** `ChatTurnMetrics` dataclass; collect on both paths; attach to `ChatTurnResult.metrics`. Convert the two `log.info("chat turn done...")` calls from formatted strings to structured `extra={...}` fields (keeping a readable message) — this alone turns the #1762 diagnosis workflow from "parse a string by eye" into a Loki field query, using data already in memory. Populate the dead `ChatMessage.latency_ms` in `_persist_turn` from `result.metrics.total_ms` — retires a dead column, gives the transcript UI per-turn latency, no migration. Tests run against the dataclass, no DB. (Committed `b01376b`.)
- [x] **Phase 2 — Tables and write path.** Migration `0020` (owner applies manually); `src/models/chat_turn_metrics.py` with `record_chat_turn_metric`; register in `src/models/__init__.py`; `record_metric` param on `ChatBotAgent`; wire from `make_chatbot_factory`. Tests 1-3. Accepted gap: timed-out/errored turns write no row. (Committed `f2eb969`.)
- [x] **Phase 3 — Read path and alerting.** `GET /tenants/{id}/chat-turn-metrics`; tests 4 and 6. WS-layer failure row (closes Phase 2's gap, correctly located outside the cancelled coroutine) — its rows are counted separately and excluded from the read endpoint's and push module's averages/latency gauges, since a turn that never ran carries no real timing signal. `chat_metrics_push.py` + `_push` job-name parameterization + hook into the existing loop; test 5. Retention prune job (§11.2), 90-day default. Docs: `docs/HANDOVER.md` at-a-glance row and schema list; the §6 tracing-plan edits.

## 11. Owner decisions (settled 2026-09-08)

Volume data gathered to inform these: chat ran **~4,733 agent turns in the last 30 days** (28,358 lifetime) against voice's **50 `turn_metrics` rows** in the same window (747 lifetime). Chat metrics are therefore ~90× voice's volume — roughly 4.7k parent rows/month plus ~12–15k child rows/month at the ~2–3 tool calls per turn observed in real traffic.

1. **Child table — YES.** `chat_tool_metrics` as specced in §2/§3. At this volume per-tool p95 and per-tool timeout rate are worth the extra surface; the aggregate-only fallback is not taken.
2. **Retention — prune, ~90 days.** A pruning job is part of this work (Phase 3). ~50k rows steady-state. Deliberately *not* following voice's unbounded precedent, because voice is unbounded at 747 lifetime rows while chat would add ~17k/month — the precedent doesn't transfer.
3. **Prometheus push — build it (Phase 3), plus fix the failure-log rate limiting.** Scoping rationale established from the code:
   - **Env vars unset** (current state): a genuinely free no-op. `aggregate_and_push_turn_metrics` checks `if not push_url` and returns `0` *before* the `try` block that queries the DB — no HTTP call, no DB query, logs at `DEBUG` only. Building the push module costs nothing while Grafana is unconfigured.
   - **URL set but unreachable**: cannot crash (caught inside the function *and* by the loop's `except Exception`, which exists so "the push loop must never die"), but would fail every `METRICS_PUSH_INTERVAL_S` (default 60s) and log a full traceback each time — **~1,440 tracebacks/day**. This module has *no* failure-log rate limiting, unlike `LokiPushHandler`'s warn-once-per-outage design. **Fix that as part of Phase 3**; it improves the existing voice pusher too, so review must confirm no voice regression.
   - **The actually-dangerous case** (what the module's docstring warns about): a URL that *accepts* the PUT without speaking classic Pushgateway protocol returns 2xx, `raise_for_status()` passes, success is logged — and no data reaches Grafana. Grafana Cloud's usual Prometheus path is **remote-write** (protobuf+snappy), a different wire format entirely. So verify the account actually exposes a Pushgateway-compatible endpoint **before relying on any dashboard built on this**, independent of shipping the code.
4. **Voice-note transcription time — leave out.** Keeps the agent-side callback clean; it can't see WS-layer timing. Revisit only if audio-turn latency becomes a complaint.
5. **`crm_id` on the child table — parent only** (coordinator's call, minor either way). Accept a join for CRM-level tool rollups; denormalize later if that query becomes common.
6. **Guards returning a verdict object — not in this pass** (coordinator's call). Before/after `(response_text, confidence)` snapshots as specced in §4. Recorded as a future cleanup so it isn't lost: it would be cleaner and less fragile, but touches shared, heavily-tested pure functions in `src/rag/context_builder.py` for no behavior gain here.
