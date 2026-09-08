# LLM Conversation-Flow Tracing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status: PLAN ONLY, not yet approved for implementation.** Design produced by an Opus planning pass on 2026-09-07, reworked the same day after a primary-source vendor re-verification pass (see §1). Section 9 (Open Questions) must be resolved with the project owner before Phase 0 is dispatched. Do not implement anything from this file without that sign-off.
>
> **Rework note:** the first draft of this plan assumed Langfuse. Verification against Arize's own documentation found that **Arize Phoenix resolves the retention problem that was this plan's biggest flagged weakness**, at a fraction of the infrastructure footprint. The recommendation changed to Phoenix. Everything backend-agnostic — the redaction boundary, the tiered capture modes, the trace hierarchy, the dual-seam architecture, the failure-mode design, the testing strategy, the product scoping — is unchanged, because none of it depended on the vendor.

**Goal:** Add replay-grade conversation tracing so a support engineer can answer "what did the bot actually do on ticket #1762 turn 3?" — which tools fired, in what order, what the model decided, where the guard tripped — without asking a customer for a screenshot.

**The real problem, stated plainly:** this platform has a deliberate, load-bearing policy that tool response bodies and full LLM reply text are never logged at any level, hardened after an active production PII leak (`ad6f597 fix(security): remove PII from logs…`). A conversation-tracing tool exists to capture exactly those payloads. **This is not "add a tracing tool," it's "build a redaction boundary, then put a tracing tool behind it."** The redaction boundary is the real deliverable; the tracing backend is a replaceable consumer behind it.

**Tech stack:** Python 3.11+, FastAPI, the existing `ILLMProvider` interface, OpenTelemetry via `arize-phoenix-otel` + `openinference-instrumentation` (self-hosted Arize Phoenix backend — see §1.3 for the decision).

---

## 1. Backend decision: Phoenix vs. Langfuse (verified 2026-09-07, primary sources only)

### 1.1 Why this section exists

The first draft of this plan chose Langfuse, and flagged in its own Open Questions that **Langfuse gates data-retention control behind an enterprise license even when self-hosting**. For a real-money betting operator holding customer conversation data, "self-host for free but you cannot control how long the data is kept" is close to a disqualifying property. That question has now been chased down properly. Alternatives were screened first:

| Candidate | Outcome |
|---|---|
| **LangSmith** | **Disqualified.** Self-hosting is Enterprise-only under a custom contract; no free/OSS self-host path exists at all. |
| **Helicone** | **Disqualified.** Effectively cloud-only. |
| **Braintrust** | **Disqualified.** Proprietary closed-source SaaS, not self-hostable. |
| **Arize Phoenix** | **Viable — and the recommendation.** See §1.2. |
| **Langfuse** | Viable, the incumbent proposal. See §1.2. |
| **OpenLLMetry → existing Grafana Cloud Tempo** | Viable as a *fallback / dual-export target*, not the primary. See §1.4. |

### 1.2 Head-to-head, primary-sourced

Confidence key: **[P]** = confirmed against the vendor's own docs or source code. **[S]** = secondary source only. **[?]** = unclear or contradictory.

| Axis | Arize Phoenix (self-hosted) | Langfuse (self-hosted OSS) | Winner |
|---|---|---|---|
| **Retention control** *(the deciding issue)* | **[P]** First-class and free. `PHOENIX_DEFAULT_RETENTION_POLICY_DAYS` set at deploy time. **Default is `0` = infinite**, so it must be set explicitly — treat this as a required config, not an optional one. Phoenix ≥9.0 adds named policies that are time-based *and/or* trace-count-based, each with a CRON enforcement schedule, plus **per-project overrides** in the UI. No paid gating anywhere in the docs. | **[P]** **EE-gated.** Retention policies require a license key. Self-hosting has exactly two tiers — OSS (free) and **Enterprise (custom pricing, "talk to sales")**; there is no published price at which retention can simply be bought. The only free path is hand-rolling ClickHouse TTLs against Langfuse's **internal, unowned schema**, which is unsupported and can break or silently mis-delete on any upgrade. | **Phoenix, decisively** |
| **Self-host infra footprint** | **[P]** **One container** + one SQL database. Postgres ≥14 (`PHOENIX_SQL_DATABASE_URL`) or SQLite for dev (`PHOENIX_WORKING_DIR`). No ClickHouse, no Redis, no blob store, no separate worker process. Ports 6006 (UI + OTLP/HTTP) and 4317 (OTLP/gRPC). | **[P]** `langfuse-web` + `langfuse-worker` + Postgres + **ClickHouse** + Redis/Valkey + an S3-compatible bucket. Six moving parts, each with its own failure mode, backup story, and upgrade path. | **Phoenix, by a wide margin** |
| **Masking / redaction hook** | **[P]** **Weaker.** OpenInference offers only coarse boolean hide-flags (`OPENINFERENCE_HIDE_INPUTS`, `HIDE_OUTPUT_MESSAGES`, `HIDE_INPUT_IMAGES`, `HIDE_LLM_PROMPTS`, `BASE64_IMAGE_MAX_LENGTH`, …) via env vars or a `TraceConfig` object. **There is no programmable callback.** An open feature request (Arize-ai/openinference#3203) asks for exactly this. Mitigation in §2.5. | **[P]** **Stronger.** `mask_otel_spans(func)` is a real client-side callback invoked on every span before export. (Server-side masking is separately EE-gated and unavailable to us either way.) | **Langfuse** — the only axis it wins on substance |
| **Python SDK ergonomics for hand-rolled adapters** | **[P]** Excellent fit. `arize-phoenix-otel` (Apache-2.0) `register(project_name=, endpoint=, batch=, api_key=, headers=, protocol=, auto_instrument=)` returns an OTel `TracerProvider`. `openinference-instrumentation` exposes `OITracer.start_as_current_span(name, openinference_span_kind="llm"|"tool"|"chain"|"agent"|"guardrail"|"retriever")`, `@tracer.llm` / `.tool` / `.chain` / `.agent` decorators, and **typed attribute builders** — `get_llm_attributes()`, `get_tool_attributes()`, `get_session_attributes()`, `get_user_id_attributes()`, `get_metadata_attributes()`, `get_input_attributes()`, `get_output_attributes()` — so we never hand-assemble semconv attribute strings. Verified by reading the package's exported `__all__`. | **[P]** Comparable. SDK v4 is OTel-based; manual work goes through `start_as_current_observation`. No native wrapper for Gemini or Anthropic; **none at all for Groq**. Manual instrumentation required regardless. | **Phoenix, slightly** — the typed attribute builders and the `using_*` contextvars are a closer match to this codebase (see next row) |
| **Ambient-context propagation** *(matters because of the shared LLM singleton, §3.1)* | **[P]** `using_session` / `using_user` / `using_metadata` / `using_tags` / `using_attributes` are **OTel-Context (contextvar) based** context managers *and* decorators — structurally identical to this repo's existing `src/utils/trace_id.py` idiom. **Caveat [P]:** they are read automatically only by *auto-instrumentors*; for our hand-created spans we must call `get_attributes_from_context()` ourselves. That is one line in the facade. | **[P]** Equivalent capability via SDK trace attributes, but not expressed as a contextvar primitive we can align with `trace_id.py`. | **Phoenix, slightly** |
| **Multi-tenancy primitives** | **[P]** `project` (top-level namespace, per-process via `register(project_name=)` / `PHOENIX_PROJECT_NAME`, or per-span via `dangerously_using_project`), `session.id`, `user.id`, arbitrary `metadata` dict, `tags` list. **[P] Gap:** no first-class `environment` attribute — separate prod/stage by using separate *projects*, which is arguably cleaner anyway. | **[P]** `project`, `session_id`, `userId`, `metadata`, `tags`, **plus a first-class `environment` filter**. | **Langfuse, marginally** (`environment`) — but see §4 for why separate projects are fine |
| **RBAC / SSO / audit** | **[P]** **Free.** Project-level RBAC (≥8.23, admin/member roles over project CRUD), OAuth2 (Google, Entra ID, Cognito, Auth0), LDAP group→role mapping, System/User API keys, `PHOENIX_ADMINS` deploy-time provisioning, `PHOENIX_ADMIN_SECRET`. | **[P]** **EE-gated.** Project-level RBAC, audit logs, org-management API and SCIM all require a license key. | **Phoenix** — and this matters for a regulated operator |
| **License** | **[P]** Server is **Elastic License 2.0** (source-available, not OSI). Its Limitations are exactly three clauses; only one is substantive — *"You may not provide the software to third parties as a hosted or managed service."* Internal self-hosting is explicitly *"free and fully permitted"* with *"no feature gates."* The license-key clause is vacuous (Phoenix has no license key). **[P] Important:** the app-side package `arize-phoenix-otel` is **Apache-2.0**; ELv2 never touches code we ship. | **[P]** Core is **MIT**. Cleanly OSI-approved. | **Langfuse** on license purity — but the ELv2 restriction does not bind this use case at all |
| **Ecosystem maturity** | **[P]** 11.4k stars / 1.1k forks / 979 open issues. Repo created **2022-11-09** — *older* than Langfuse. `arize-phoenix` v20.8.0 shipped 2026-09-04, commits landing daily. Backed by Arize AI, whose commercial product (Arize AX) makes the OSS funnel strategically sustainable. **Note:** comparisons routinely conflate self-hosted Phoenix with Arize AX; they are different products, and AX's paid tiers gate nothing in self-hosted Phoenix. | **[P]** 34.3k stars / 3.7k forks / 915 open issues; created 2023-05-18; also committing daily. **~3x the community.** | **Langfuse, clearly** — more StackOverflow answers, more blog posts, more people who have hit your bug first |
| **Prompt management / playground in OSS** | **[S]** Arize's own blog and the Phoenix 8.0 community release note claim prompt versioning + playground are in the OSS core. **Not confirmed against a self-hosting feature matrix.** Low stakes: this plan sends a prompt *hash*, never prompt text (§2.3). | **[P]** Present in OSS; *protected prompt labels* specifically are EE-gated. | Immaterial here |

### 1.3 Decision: **self-hosted Arize Phoenix. This is not a toss-up.**

Retention was the deciding issue and it inverts completely. Phoenix hands us exactly the control this platform needs — a bounded, enforced, per-project retention window — for one environment variable, free, documented, and supported. Langfuse offers the same capability only via an unpriced sales negotiation, or via a hand-rolled TTL against a database schema we neither own nor control across upgrades. For customer conversation data at a real-money operator, "supported deletion" versus "DIY deletion we hope keeps working" is not a close call.

Phoenix then compounds the win on two more axes that were secondary concerns in the first draft but are real: the infrastructure footprint drops from six moving parts to two, and RBAC/SSO — which a regulated operator will eventually be asked about in an audit — is free rather than EE-gated.

**Honest accounting of what we give up:**
- **The masking hook.** Langfuse's `mask_otel_spans` is genuinely better than anything Phoenix offers. This is the real cost of the decision. It is affordable specifically because of this plan's architecture: **all redaction happens at source, before any span attribute is ever set** (§2), and every span in this design is hand-created — there are no auto-instrumentors capturing payloads behind our back. The hook was always belt-and-suspenders, never the barrier. §2.5 restores an equivalent second net via a wrapping `SpanExporter`.
- **~3x smaller community.** Fewer people will have hit your bug first. This is a live risk, tracked as Open Question #6 rather than waved away.
- **ELv2 instead of MIT.** Needs a legal nod (Open Question #7), though the one binding clause — no reselling as a hosted service — is plainly irrelevant to internal use, and the code we actually ship is Apache-2.0.
- **No first-class `environment` attribute.** Solved by using one project per environment (§4), which is cleaner than a filter anyway.

### 1.4 The third option: OpenLLMetry → existing Grafana Cloud Tempo

Worth taking seriously, and its strongest argument is not technical.

**What's gained:**
- **Zero new infrastructure and zero new spend.** This platform already runs Grafana Cloud — Loki for logs (`src/utils/logging.py`) and Prometheus for metrics (`src/observability/turn_metrics_push.py`), both authenticated with the same `user:api_key` HTTP-basic pattern already in `src/config.py`. Tempo is in the same stack, and Grafana Cloud publishes a managed OTLP gateway.
- **It largely dissolves Open Question #1.** Grafana Cloud is *already an approved sub-processor receiving production telemetry from this platform.* Adding a span type to an existing approved processor is a categorically smaller legal ask than onboarding a brand-new one. If the legal answer on new sub-processors turns out to be "no," this is the only surviving option.
- Traces land next to the logs and metrics they correlate with, in a UI the team already uses.

**What's sacrificed — state this plainly to the owner:**
- **No purpose-built conversation-replay UI.** You read raw span trees in Grafana's trace panel. There is no side-by-side prompt/completion view, no session thread view, no "next turn in this conversation" navigation.
- **No session grouping as a first-class concept.** Tempo has traces and spans; the notion that 14 traces form one customer conversation has to be reconstructed by hand via TraceQL on an attribute.
- **No prompt playground, no datasets, no evals** — the entire "iterate on the prompt against real captured traffic" workflow is gone.
- **TraceQL is the wrong shape for the actual questions.** "Show me every turn where the TOOL FAILURE directive fired and a guard then rewrote the reply" is a saved view in an LLM-native tool and a bespoke query in Tempo.
- **[P] Retention is *less* controllable than Phoenix's, not more.** Grafana Cloud Traces has a **30-day minimum**, extendable in 30-day increments at ~$0.10/GB; going *below* 30 days requires contacting Support. For PII-adjacent data we would prefer the freedom to pick 7 days, and Phoenix gives it while Tempo does not.

**Verdict: fallback, not primary — but keep the door open.** Because Phoenix is plain OTLP with OpenInference semantic conventions, the *same* instrumentation can dual-export by attaching a second `BatchSpanProcessor` pointed at Grafana Cloud's OTLP gateway. That option did not exist under the more opinionated Langfuse SDK. **Choosing Phoenix does not foreclose Grafana; choosing Grafana forecloses Phoenix's UI.** That asymmetry is the argument.

---

## 2. Data-sensitivity design (the core of this plan — unchanged by the vendor decision)

> This entire section is backend-agnostic. It was the load-bearing design decision in the first draft and it remains so. Only §2.5 (the second net) and §2.6 (hosting) changed.

### 2.1 `src/utils/redact.py` does not fit — do not extend it
It's 28 lines, one function (`redact_url()`), strips userinfo/query/fragment from a URL. No notion of structured payloads or PII at all.

### 2.2 The reusable shape lives in `src/chatbot/tool_executor.py` — copy the shape, not the ruleset
`_redact_internal_ids()` (recursive dict/list/str walk), `_UUID_RE`, `REDACTED_PLACEHOLDER`, the normalized-key-matching idiom. **But its key set is wrong for this purpose** — it's a system-routing redactor that deliberately preserves `bet_id`/`transaction_id` and exempts `PgsOrderId` from scrubbing (the LLM needs those). A trace redactor must NOT carry those exemptions over.

### 2.3 Tiered capture policy
One env var, `VOX_TRACE_CAPTURE_MODE ∈ {off, redacted, full}`, **default `redacted`**, with `full` **hard-refused at config load whenever the environment is production-like** — fail-safe by construction, not by convention.

| Payload | `redacted` (prod) | `full` (non-prod only) |
|---|---|---|
| Tool name / argument keys | full | full |
| Tool argument values | dropped → type+shape only | full |
| Tool response body | **never sent** — only status/failure-category/top-level-keys/array-lengths/latency | full |
| LLM completion text | PII-scrubbed with typed placeholders (`<AMOUNT>`, `<PHONE>`, `<UPI>`, `<EMAIL>`, `<UUID>`, `<NUM>`) | full |
| User message text | same scrubber | full |
| System prompt | **not sent** — only `prompt_pack` name + hash + length | full |
| RAG chunk text | not sent — source labels + scores + count | full |
| Derived signals (below) | **always full fidelity, no PII in them** | full |
| Multimodal image/video bytes | **never sent in ANY mode** (deposit screenshots are the platform's most sensitive artifact) | mime + length only |

**Why `redacted` mode is still useful:** what you actually debug in this system is control flow and derived verdicts, not customer numbers. Always emit, unredacted: round count, tool-call ordering, `failed_category_names`, whether the TOOL FAILURE directive fired and escalated, `finish_reason`/token usage, which guard fired and whether it modified the reply, detected language, budget/timeout telemetry, and `grounded=true|false` per stated figure (the boolean, never the figure). That set answers "why did the bot say the wrong thing" for essentially every failure class this codebase has actually hit, with zero PII.

Guard verdicts (`guard_hallucination_fired`/`guard_no_grounding_fired`/`guard_unverified_data_fired`) and `rounds_exhausted`/`retry_fired`/`failure_directive_fired`/`failure_directive_escalated`/`escalated` are SQL-queryable booleans on `chat_turn_metrics` (`docs/superpowers/plans/2026-09-08-chatbot-turn-metrics.md`), aggregatable and rate-able (`GET /tenants/{id}/chat-turn-metrics`) without opening a trace at all. What tracing adds on top of that per-turn boolean is narrative context, not the measurement itself: *which* span the guard fired inside, what the pre/post `(response_text, confidence)` snapshot looked like, and how that turn's guard verdict sits alongside its tool calls and rounds in one place.

**Phoenix-specific hardening:** set the OpenInference hide-flags as a *third* independent layer, so that even a future accidentally-enabled auto-instrumentor cannot capture payloads: `OPENINFERENCE_HIDE_INPUT_IMAGES=true` and `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH=0` are set **unconditionally in every environment**, backing the "image bytes never leave the process in ANY mode" rule with a vendor-level guarantee rather than only our own code path.

### 2.4 Scrubber rules — new module `src/observability/trace_redaction.py`
Deny-by-default on strings, longest/most-specific pattern first: UUID → `<UUID>`; email → `<EMAIL>`; UPI/VPA → `<UPI>`; Indian mobile → `<PHONE>`; currency amounts (₹/Rs/INR) → `<AMOUNT>`; IFSC → `<IFSC>`; PAN → `<PAN>`; Aadhaar-shaped → `<AADHAAR>`; **catch-all: any digit run ≥ 4 → `<NUM>`** (deliberately blunt — account numbers, transaction IDs, order refs, card fragments, OTPs are all digit runs, not worth enumerating); dates → `<DATE>`. Plus a key-based hard drop (mobile/phone/email/upi/ifsc/pan/aadhaar/kyc/dob/address/bank*/card*/balance/name/*id/token/password/otp → `<REDACTED>` regardless of value type) applied before value scrubbing.

Two properties to build and test for:
- **Fail closed.** Whole scrubber wrapped in `try/except Exception` → literal `"<redaction-failed>"`. This is the one place in the codebase where "best-effort, never raise" must mean *drop the data*, not *pass it through raw*.
- **Idempotent and total** — a post-condition assertion (no `@`, no digit run ≥ 4, no UUID, no deny-set key in the output) is exactly what the test suite checks.

This module has **no OTel or Phoenix import** and must stay that way. It is the piece that survives any future backend change.

### 2.5 The second net, without a masking hook

Langfuse's `mask_otel_spans` gave a single choke point where every span could be re-scrubbed before export. Phoenix has no such callback. Restore the equivalent with a **wrapping `SpanExporter`** in `src/observability/trace_export.py` (new): it delegates to the real `HTTPSpanExporter`, but first re-runs the redactor's post-condition assertion over every attribute value and rebuilds any offending `ReadableSpan` with the value replaced by `<redaction-failed>`.

**Two things a reviewer must know here:**

1. **Do not implement this as a `SpanProcessor`.** The `PIIMaskingSpanProcessor` pattern widely copied from blog posts calls `span.set_attribute()` inside `on_end()`. In OpenTelemetry Python this **silently does nothing** — `Span.set_attribute` early-returns with a log warning once the span has ended. A processor-based masker would look correct in code review, produce no test failure that isn't specifically looking for it, and redact nothing. The exporter seam is the only one that actually works.
2. Attach it via the `TracerProvider` returned from `register()`, using `add_span_processor()` with our own `BatchSpanProcessor(RedactingExporter(...))`, rather than relying on `register(batch=True)`'s default processor.

This is defence-in-depth, matching the belt-and-suspenders idiom `tool_executor.py` already uses. **It is not the barrier** — source-side redaction (§2.4) is. Treat a trip of the exporter's assertion as a *bug report*, and have it emit a rate-limited `ERROR` log when it fires.

### 2.6 Hosting: self-hosted, and now cheap enough that it stops being a debate
The first draft's self-hosting recommendation was driven by retention and priced at ~$60–150/mo for a six-service Langfuse stack. With Phoenix the same recommendation costs far less and carries far less ops burden: **one container plus one Postgres database.** Sizing for an internal, off-the-call-path tool: ~1–2 vCPU / 2–4 GB for the Phoenix container, plus a small managed Postgres and a persistent volume. Single replica, no HA — with an explicit accepted consequence: **if Phoenix is down, traces are dropped and nothing else is affected** (§5).

---

## 3. Instrumentation architecture

### 3.1 Both a provider seam AND an agent seam — and why
> Unchanged reasoning; only the SDK calls differ.

`src/auth/registry.py::get_platform_llm()` memoizes **one shared `ILLMProvider` instance across every tenant and every chat session**. A provider-level wrapper cannot know which tenant/session/turn it's serving from constructor state — it must read ambient context, exactly the pattern `src/utils/trace_id.py`'s `ContextVar` already established.

- **Layer A — provider seam** (`src/observability/llm_tracing.py::TracingLLMProvider`, an `ILLMProvider` decorator wrapping both `generate()` and `generate_stream()` — the latter is what `src/pipeline/engine.py` uses for cascade voice, don't skip it). Wrapped at the composition root (`src/auth/registry.py`'s `llm_factory`), zero edits to the four adapter files.
- **Layer B — agent seam** (`src/agents/chatbot.py::_handle_with_tools`, line ~516) — the round loop, tool dispatch, budget slicing, and guard verdicts are only visible here; Layer A alone sees N independent calls with no idea they form one agentic turn.

Phoenix makes Layer A slightly cleaner than Langfuse would have: because `using_session` / `using_user` / `using_metadata` write into the OTel Context (contextvars), the ambient state the shared singleton needs is propagated by the same mechanism `trace_id_scope()` already uses. The facade calls `get_attributes_from_context()` when building each manual span — **required**, because the `using_*` helpers are auto-read only by auto-instrumentors, and every span here is hand-created.

### 3.2 Trace hierarchy, in this codebase's real units
> Unchanged. Only the primitive names are Phoenix's.

```
Phoenix session (session.id) = chat WS session (bare session_id)
└── trace = ONE TURN (one user message → one bot reply)
    root span kind=AGENT, name "chat.turn"
    metadata.vox_trace_id = the existing per-turn trace_id (Loki join key)
    ├── span kind=CHAIN name "round.1"
    │   ├── span kind=LLM   name "llm.generate"
    │   ├── span kind=TOOL  name "tool.search_knowledge_base"
    │   ├── span kind=TOOL  name "tool.get_player_wallet"
    │   └── span kind=CHAIN name "directive.injected" (labels only)
    ├── span kind=CHAIN name "round.2" ...
    ├── span kind=LLM name "llm.generate.forced_final"
    ├── span kind=LLM name "llm.generate.retry"
    └── span kind=GUARDRAIL name "guards" (which guard fired, did it modify the reply)
```

Turn-as-trace (not session-as-trace) is deliberate — matches the existing per-turn granularity, keeps traces small and cheap, and directly serves the duplicate-turn-guard debugging case (two traces, same session, same fingerprint, different `vox_trace_id`). Phoenix's session view recovers the full-conversation thread in the UI.

Phoenix's `openinference_span_kind` taxonomy (`AGENT` / `CHAIN` / `LLM` / `TOOL` / `GUARDRAIL` / `RETRIEVER`) maps onto this codebase's real units better than Langfuse's flatter span/generation split — in particular **`GUARDRAIL` is a first-class kind**, which matters because guard verdicts are one of the top things support needs to see.

**`vox_trace_id` is the integration point that makes Phoenix and the existing Loki pipeline complementary, not duplicative** — paste one id, get the Phoenix trace AND every Loki log line for that turn.

### 3.3 Specific file:function targets

| File | Change |
|---|---|
| `src/observability/tracing.py` **(new)** | The facade: `turn_trace()`, `record_generation()`, `tool_span()`, `round_span()`, `guard_span()`, `set_turn_attributes()`, `flush()`, `shutdown()`. No-op when unconfigured. **The only module that imports `phoenix.otel` / `openinference.*`.** Calls `register(project_name=…, endpoint=…, batch=True, set_global_tracer_provider=False, auto_instrument=False)` — `auto_instrument=False` is a **security control**, not a preference: it guarantees no library starts capturing payloads outside the redaction boundary. |
| `src/observability/trace_redaction.py` **(new)** | Scrubber + shape-summarizer + the post-condition assertion. Pure functions, no I/O, **no OTel/Phoenix import**. |
| `src/observability/trace_export.py` **(new)** | `RedactingSpanExporter` — the §2.5 second net. Wraps `HTTPSpanExporter`. |
| `src/observability/llm_tracing.py` **(new)** | `TracingLLMProvider`, an `ILLMProvider` decorator over `generate()` and `generate_stream()` (see `src/interfaces/llm.py`). |
| `src/observability/trace_context.py` **(new)** | Ambient `ContextVar` for `(tenant_id, crm_id, session_id, ticket_id, vox_trace_id, capture_mode)`, mirroring `trace_id.py`'s getter/setter/scope idiom. Bridges into OTel Context via `using_attributes(...)`. |
| `src/utils/trace_id.py` | **Keep**; rewrite its "gets deleted once real OpenTelemetry tracing lands" docstring. It becomes the permanent Loki↔Phoenix join key, not a stopgap. |
| `src/api/chat.py` **~line 1727** | Nest the turn trace inside the existing `with trace_id_scope(turn_trace_id):`. Single insertion point. |
| `src/agents/chatbot.py` | Round spans in `_handle_with_tools` (line ~516) loop; LLM span attrs at each `generate()` call site; tool spans around dispatch; a `GUARDRAIL` span around the guard-application block; lighter instrumentation on `_single_shot` (line ~452) and `summarize_session` (line ~1071). |
| `src/bootstrap.py::make_chatbot_factory` **(line ~405)** | Populate the trace context from the `tenant` closure (`tenant.id`, `tenant.settings.crm_id`, `tenant.settings.prompt_pack`) — **not** new `ChatBotAgent` constructor params. |
| `src/auth/registry.py::get_platform_llm` / `llm_factory` **(lines 43, 76, 83, 90)** | Wrap in `TracingLLMProvider`. |
| `src/main.py` lifespan | Initialize the facade at startup (installs the `RedactingSpanExporter` processor); `tracing.shutdown()` in the shutdown path. No new background loop — the OTel `BatchSpanProcessor` owns its own export thread. |
| `src/config.py::Secrets` | New env vars, placed beside the existing `GRAFANA_*` block (lines ~296–305). |
| `src/dialogue/prompts.py` | **Confirmed untouched.** No owner confirmation gate is triggered by this plan — tracing wraps calls, and the design sends a prompt hash + pack name, never prompt text. |

**Sequencing conflict — RESOLVED.** The first draft recommended blocking on `696cd1c` (the parallel observability-rollout work, then on `stage`). Verified 2026-09-07: `696cd1c` is now contained in `main` (`5af64a1 Merge branch 'stage' into main`), and `src/observability/` on `main` contains `__init__.py` + `turn_metrics_push.py`. **There is no longer a four-file conflict; implementation is unblocked.**

### 3.4 Product scope: ChatBot first, confidently
> Unchanged — every reason here is a fact about this codebase, not about the vendor.

1. It's the primary active workstream.
2. The agentic multi-round loop is where tracing earns its keep — voice cascade is one call per turn, existing logs already cover that.
3. **ChatBot has its own aggregate latency/failure pipeline** (`chat_turn_metrics`/`chat_tool_metrics`, `record_chat_turn_metric`, `GET /tenants/{id}/chat-turn-metrics`, Prometheus push — `docs/superpowers/plans/2026-09-08-chatbot-turn-metrics.md`), built deliberately rather than left to tracing (see Open Question #5's resolution). That pipeline answers "how slow/how often does it fail, aggregated." Tracing's ChatBot-first case rests on reasons 1, 2, and 4: the multi-round agentic loop and its per-conversation payload replay / prompt-iteration workflow, not a latency-coverage gap.
4. **S2S/Gemini Live cannot be traced through this seam at all** — it doesn't implement `ILLMProvider`, and audio-to-audio has no prompt/completion pair. **Permanently out of scope for this design, not deferred.**

### 3.5 What to keep vs. retire from existing observability
Keep everything (Loki logs, `TurnMetric` table, `chat_turn_metrics`/`chat_tool_metrics` tables, the Prometheus push loop covering both) — none of it does what conversation tracing does, and conversation tracing doesn't do what any of them does (Phoenix has no STT/TTS concept at all, and stores no per-turn content by design — see the table above). Only `trace_id.py`'s own "delete me later" framing is retired; it is a permanent join key, shared today by Loki and `chat_turn_metrics` via the same `trace_id` column, with Phoenix as a third consumer once this plan ships.

---

## 4. Multi-tenancy mapping

**One Phoenix instance; one project per environment** (`vox-prod`, `vox-stage`, `vox-dev`); tenant/CRM as span attributes, **not** a project per tenant. Per-tenant projects would mean per-tenant key management inside a process that shares one LLM singleton, and would fragment the cross-tenant view that makes this tool useful.

| This platform | Phoenix / OpenInference primitive |
|---|---|
| `tenant.id` | `user.id` (via `using_user`) — the high-cardinality dimension, used for **tenant**, never for the end customer |
| `tenant.settings.crm_id` | `metadata.crm_id` + a tag |
| `bare_session_id` | `session.id` (via `using_session`) |
| `ticket_id` | `metadata.ticket_id` |
| environment (prod/stage/dev) | **project name** (`PHOENIX_PROJECT_NAME` / `register(project_name=)`) — Phoenix has no first-class `environment` attribute, and a separate project per environment is a harder boundary than a filter anyway: retention policies are also per-project, so prod and stage can have genuinely different windows |
| `prompt_pack`, `llm_provider`, `llm_model` | tags |
| **End customer / player** | **nothing.** No player id or phone ever becomes a Phoenix dimension. If ever needed, use `token_fingerprint()` from `src/auth/audit.py`, never a raw id. |

Environment separation by project has a second benefit worth naming: `full` capture mode can *only* ever exist in the `vox-dev` / `vox-stage` projects, so a misconfiguration is visible as "unredacted data in the wrong project" rather than hidden inside a filter dropdown.

---

## 5. Failure-mode design & testing

**Never touch a live turn:** gate at import (unconfigured → no-op singleton, `phoenix.otel` never imported — same convention as `configure_logging` / the metrics push loop); every facade entry point wrapped in `try/except Exception`; rate-limited failure warnings (reuse `LokiPushHandler`'s "one per outage" pattern); a circuit breaker (N consecutive failures → disable for M seconds); `flush()`/`shutdown()` only in the lifespan `finally`, never on the request path; `BatchSpanProcessor` bounded queue with drop-on-overflow (tune via `OTEL_BSP_MAX_QUEUE_SIZE`); redaction is the one place that fails **closed** (drops data on error, doesn't pass it through); sampling **by session, not by turn**, so a sampled conversation is always complete and never half-traced.

**Testing — designed so the existing 1927+ tests need zero edits.** One `autouse=True` conftest fixture clears `PHOENIX_*` / `OPENINFERENCE_*` / `VOX_TRACE_*` env and resets the facade singleton, so every existing test runs the no-op path by default — deliberately the opposite of the existing conftest's `setdefault`-style provider-key fixture, because **tracing must default OFF in tests.**

The fake-based strategy is entirely tool-agnostic and carries over unchanged:

- `test_trace_redaction.py` — pure-function tests over realistic betting payloads (wallet, transactions, KYC, bank/UPI, `PgsOrderId`), asserting the no-`@` / no-digit-run-≥4 / no-UUID / no-deny-key post-condition, idempotence, and the fail-closed path.
- `test_tracing_facade.py` — a **`RecordingTracer`** fake modeled on the existing `ScriptedLLM`, asserting the §3.2 hierarchy shape (span names, `openinference_span_kind` values, parent/child nesting).
- `test_tracing_never_raises.py` — an **`ExplodingTracer`** driving a full two-round tool turn, asserting the resulting `ChatTurnResult` is byte-identical to the untraced run.
- `test_trace_redaction_integration.py` — drives a turn with a realistic PII payload and asserts the *tracer's own recorded output* contains none of it.
- `test_llm_tracing.py` — transparency of `TracingLLMProvider` for return values, exceptions, and streamed chunks.
- **`test_trace_export.py` (new, Phoenix-specific)** — asserts the `RedactingSpanExporter` actually mutates what reaches the delegate. This test exists specifically to catch the §2.5 trap: a naive `SpanProcessor`-based implementation would pass every other test in this list and redact nothing.

---

## 6. Ops & deploy

**Env vars.** Vendor-native names the SDK reads itself, plus `VOX_`-prefixed platform policy:

| Var | Purpose |
|---|---|
| `PHOENIX_COLLECTOR_ENDPOINT` | Self-hosted Phoenix OTLP endpoint |
| `PHOENIX_API_KEY` | Phoenix **system** API key (not a user key) for OTLP ingest auth |
| `PHOENIX_PROJECT_NAME` | `vox-prod` / `vox-stage` / `vox-dev` — carries environment separation (§4) |
| `OPENINFERENCE_HIDE_INPUT_IMAGES` | Hardcoded `true` in **all** environments (§2.3) |
| `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` | Hardcoded `0` in **all** environments (§2.3) |
| `VOX_TRACE_CAPTURE_MODE` | `off` / `redacted` / `full`, default `redacted` |
| `VOX_TRACE_SAMPLE_RATE` | Default `1.0`; sampled by session |
| `VOX_TRACE_ALLOW_FULL_TENANTS` | Non-prod escape hatch |

Two config-load-time validations, both fail-safe by construction: `full` mode is **rejected** when the environment is production-like; `VOX_TRACE_ALLOW_FULL_TENANTS` non-empty is **rejected** in production. `PHOENIX_API_KEY` is a plaintext platform-wide secret (like the STT/LLM/TTS keys), **not** Fernet-encrypted per-tenant. Note `VOX_TRACE_ENVIRONMENT` from the first draft is **dropped** — environment now lives in the project name.

**Self-hosted infra** — new Northflank services, **none load-bearing for app liveness**:

- **One** `arizephoenix/phoenix` container, 1 replica, ~1–2 vCPU / 2–4 GB. Expose 6006 (UI + OTLP/HTTP) and optionally 4317 (OTLP/gRPC).
- **One** small Postgres database via `PHOENIX_SQL_DATABASE_URL` — a **separate** database, **never** the shared `voicebot`-schema DB.
- A persistent volume for `PHOENIX_WORKING_DIR`.
- **`PHOENIX_DEFAULT_RETENTION_POLICY_DAYS=30`** — mandatory, not optional. The default is `0` (infinite), which is precisely the property this platform cannot accept. Treat an unset value as a deploy blocker. Tighten further per-project in the UI once real volumes are known.
- Auth on: `PHOENIX_ADMINS` for deploy-time admin provisioning, `PHOENIX_ADMIN_SECRET`, `PHOENIX_ENABLE_STRONG_PASSWORD_POLICY=true`, and **change `PHOENIX_DEFAULT_ADMIN_INITIAL_PASSWORD` off its `admin` default before the service is ever reachable.**
- `PHOENIX_TELEMETRY_ENABLED=false` and `PHOENIX_ALLOW_EXTERNAL_RESOURCES=false` — no phone-home, no external asset loading from a service holding conversation data.
- **No ClickHouse. No Redis. No S3 bucket.** (Explicitly called out because the first draft required all three, and the deploy doc should not carry their ghosts.)

**Docs to update:** `docs/deploy/northflank.md` (new "Stage 3 — Conversation tracing" section); `docs/HANDOVER.md` (a row in the at-a-glance table, plus a note in the security section that `VOX_TRACE_CAPTURE_MODE` is a **security control, not a verbosity knob**); a new `docs/observability.md` (which tool answers which question — Loki for logs, Grafana/Prometheus for latency, Phoenix for conversation replay, `vox_trace_id` as the join key); `.env.example`; and `src/utils/trace_id.py`'s docstring. **No Alembic migration** — no new tables in this repo's own schema.

---

## 7. Phased rollout

- **Phase 0 — Redaction boundary only, no tracing backend.** `trace_redaction.py` + its tests. No new dependency, no network, nothing wired in. Independently useful and reviewable on its own merits — **if the tracing backend is ever rejected entirely, or swapped, this phase still stands alone.** (Unchanged from the first draft, and the vendor rework is itself the proof that leading with this was right.)
- **Phase 1 — Facade + no-op wiring.** `tracing.py`, `trace_context.py`, `trace_export.py`, `llm_tracing.py`, the conftest fixture, all tests. Add `arize-phoenix-otel` + `openinference-instrumentation` (app-side only — **never** `arize-phoenix`, the 90-dependency server package). Re-verify the pytest baseline is unmoved. Ships to prod with tracing off.
- **Phase 2 — ChatBot instrumentation, non-prod only, `full` mode, synthetic data.** Run Phoenix locally or in stage — **one container, `docker run -p 6006:6006 arizephoenix/phoenix`, SQLite-backed, zero cost** — against `tools/mock_crm.py` data only, no real customer traffic. Purpose: validate hierarchy and ergonomics before committing infra. *(Phoenix improves on the first draft here: the old plan needed Langfuse Cloud Hobby for a zero-infra trial, which meant sending even synthetic data to a third party. Now the trial is fully local.)*
- **Phase 3 — Self-hosted stack + prod `redacted` mode, one tenant, low sample rate.** Deploy the §6 footprint with `PHOENIX_DEFAULT_RETENTION_POLICY_DAYS=30`. **Gate: a manual human audit of ~50 real prod traces for any leaked PII before widening — the single most important checkpoint in this entire plan**, since it's the only thing that validates the redactor against real (and documented-as-drifting) CRM response shapes.
- **Phase 4 — Widen prod** (more tenants, higher sample rate). Retire nothing else. Optionally evaluate attaching the §1.4 dual-export processor to Grafana Cloud Tempo, if trace-to-log correlation inside Grafana proves worth the extra egress.
- **Phase 5 — VoiceBot cascade, only if Phase 4 proves its worth.** S2S remains permanently out of scope.

**Do not add a "Phase 6: relax redaction once trust is established."** Trust in a regex is not a security control, and `redacted` mode is not meaningfully degraded for debugging (§2.3) — `full` stays non-prod-only, permanently, enforced at config load.

---

## 8. Resolved questions (were open in the first draft)

| Was | Resolution |
|---|---|
| Self-hosted vs. Cloud? | **Resolved: self-hosted Phoenix.** Retention was the deciding factor and Phoenix gives it away free. Self-hosting is now also the *cheaper and simpler* option, so the trade-off that made this a question has evaporated. |
| Budget for the hosting service (~$60–150/mo)? | **Largely resolved.** One small container + one small Postgres, versus six services. Materially cheaper; exact figure is a Northflank sizing detail, not an architectural question. |
| Sequencing against the `stage` observability work (`696cd1c`)? | **Resolved.** `696cd1c` is merged into `main` as of `5af64a1`. No conflict. Implementation is unblocked. |
| Is retention controllable without an enterprise license? | **Resolved — this is why the plan changed.** Not on Langfuse self-hosted OSS (EE-gated, custom-priced). Yes on Phoenix, free, one env var plus per-project CRON policies. |
| Does ChatBot need its own `TurnMetric`-equivalent latency pipeline (Open Question #5)? | **Resolved: yes — chat gets its own metrics pipeline, deliberately; tracing is not the answer to the aggregate-latency gap.** `docs/superpowers/plans/2026-09-08-chatbot-turn-metrics.md` (`chat_turn_metrics`/`chat_tool_metrics`, `GET /tenants/{id}/chat-turn-metrics`, Prometheus push, 90-day prune) is that pipeline, built and shipped. §3.4 reason #3 and §3.5 reflect it. |

## 9. Open questions for the project owner (must be resolved before Phase 0)

1. **Is a new third-party sub-processor for customer-conversation data contractually and legally permitted** for a real-money gaming operator, even fully redacted? Needs a legal answer, not an engineering one. **Note the shape of this question has changed:** self-hosted Phoenix on our own Northflank account is arguably *not* a new sub-processor at all — no conversation data leaves infrastructure we already control. That may make this a much easier "yes" than it was when the answer implied Langfuse Cloud. If the answer is still "no," §1.4 (Grafana Cloud Tempo, an already-approved processor) is the fallback.
2. **Scrubber aggressiveness** — the digit-run-≥4 catch-all will redact non-PII numbers too (bet limits, promo amounts). Confident this is the right prod trade-off; the exact boundary is the owner's call.
3. **Retention window** — 30 days suggested. Phoenix now makes this a genuinely free choice, so pick it on support-escalation reach rather than on cost. Note it must be set explicitly: the default is infinite.
4. **Who signs off on the Phase-3 PII audit?** Needs a named owner, not a checkbox.
5. ~~**Does ChatBot need its own `TurnMetric`-equivalent latency pipeline**, decided deliberately, rather than tracing becoming the accidental answer to a gap it wasn't built to fill?~~ **Resolved — see §8.**
6. **[NEW] Is Phoenix's smaller ecosystem an acceptable risk?** Phoenix has ~11.4k GitHub stars against Langfuse's ~34.3k, and roughly a third of the community. When self-hosted Phoenix breaks in an unusual way, there will be measurably fewer people who hit it first. Mitigating facts: the repo is *older* than Langfuse's (Nov 2022), ships releases multiple times per week (v20.8.0 on 2026-09-04), and is backed by a funded company whose commercial product depends on the OSS project staying healthy. **Countervailing fact that mostly settles it:** a one-container-plus-Postgres deployment has a far smaller surface area to break than a six-service ClickHouse stack, so the *need* for community rescue is correspondingly smaller. Recommendation is to accept, but the owner should accept it knowingly.
7. **[NEW] Is Elastic License 2.0 acceptable to whoever signs off on licensing?** ELv2 is source-available, not OSI-approved "open source." Its only substantive restriction is against providing the software to third parties as a hosted or managed service, which this internal deployment plainly does not do; Arize's own docs state self-hosting is *"free and fully permitted"* with *"no feature gates."* Also worth stating to the reviewer: **the code we ship is unaffected** — `arize-phoenix-otel` is Apache-2.0; ELv2 covers only the server container image we run internally. Expected to be a formality, but it is a different answer from "MIT" and should not be assumed.
8. **[NEW] Dual-export to Grafana Cloud Tempo — now, later, or never?** Technically a single extra `BatchSpanProcessor` (§1.4). Buys trace↔log correlation inside a UI the team already lives in, at the cost of Grafana egress/ingest charges and a 30-day-minimum retention floor on that copy. Recommendation: defer to Phase 4, decide with real volume data.
9. **Groq/vLLM test coverage** — the provider seam covers all four adapters, but only Gemini and Anthropic are in active use; confirm whether the others need Phase-1 test coverage or can be deferred.

---

## Sources

**Arize Phoenix (primary — vendor docs and source):**
- https://arize.com/docs/phoenix/self-hosting/license — ELv2; *"Self-hosting … is free and fully permitted"*; *"There are no feature gates."*
- https://arize.com/docs/phoenix/settings/data-retention — retention policies, time- and count-based, CRON schedules, per-project overrides, Phoenix ≥9.0
- https://arize.com/docs/phoenix/self-hosting/configuration — `PHOENIX_SQL_DATABASE_URL`, `PHOENIX_WORKING_DIR`, `PHOENIX_DEFAULT_RETENTION_POLICY_DAYS` (default `0` = infinite), `PHOENIX_PORT`, `PHOENIX_GRPC_PORT`, `PHOENIX_TELEMETRY_ENABLED`, `PHOENIX_ALLOW_EXTERNAL_RESOURCES`
- https://arize.com/docs/phoenix/self-hosting/deployment-options/docker — Postgres ≥14 or SQLite; container shape; ports 6006 / 4317
- https://arize.com/docs/phoenix/self-hosting/features/authentication — OAuth2 (Google, Entra ID, Cognito, Auth0), LDAP group→role mapping
- https://arize.com/docs/phoenix/settings/api-keys — System keys, User keys, Admin Secret
- https://arize.com/docs/phoenix/settings/access-control-rbac and https://docs.arize.com/phoenix/release-notes/04.09.2025-new-rest-api-for-projects-with-rbac — project RBAC, admin/member roles
- https://arize.com/docs/phoenix/self-hosting/features/provisioning — `PHOENIX_ADMINS`
- https://arize.com/docs/phoenix/tracing/how-to-tracing/advanced/masking-span-attributes — the `OPENINFERENCE_HIDE_*` flags and `TraceConfig`; **no programmable callback**
- https://arize.com/docs/phoenix/tracing/how-to-tracing/add-metadata/customize-spans — `using_session` / `using_user` / `using_metadata` / `using_tags` / `using_attributes`; contextvar-based; auto-read by auto-instrumentors only
- https://arize.com/docs/phoenix/cookbook/tracing/openinference-best-practices — manual spans for unsupported providers, `OPENINFERENCE_SPAN_KIND`, `SpanAttributes`
- https://arize.com/docs/phoenix/tracing/tutorial/sessions — session grouping via `session.id`
- https://github.com/Arize-ai/phoenix — README, ELv2, repo stats (11.4k★ / 1.1k forks, created 2022-11-09), release `arize-phoenix-v20.8.0` (2026-09-04)
- https://raw.githubusercontent.com/Arize-ai/phoenix/main/LICENSE — ELv2 Limitations, all three clauses
- https://raw.githubusercontent.com/Arize-ai/phoenix/main/packages/phoenix-otel/src/phoenix/otel/otel.py — `register()` full signature; env vars read; `TracerProvider.add_span_processor()`
- https://raw.githubusercontent.com/Arize-ai/openinference/main/python/openinference-instrumentation/src/openinference/instrumentation/__init__.py — exported `__all__`: `OITracer`, `TraceConfig`, `get_attributes_from_context`, `get_llm_attributes`, `get_tool_attributes`, `get_session_attributes`, `get_user_id_attributes`, `get_metadata_attributes`, `dangerously_using_project`
- https://github.com/Arize-ai/openinference/issues/3203 — open request for a masking/privacy preset (confirms the gap)
- PyPI: `arize-phoenix` 20.8.0 (Elastic-2.0, 90 deps), `arize-phoenix-otel` 0.17.1 (**Apache-2.0**), `openinference-instrumentation` 0.1.61, `openinference-semantic-conventions` 0.1.35

**Langfuse (primary — carried over from the first draft, re-verified 2026-09-07):**
- https://langfuse.com/self-hosting/license-key — EE-gated list: project RBAC, **data retention policies**, audit logs, **server-side data masking**, UI customization, org-management API/SCIM, instance-management API
- https://langfuse.com/pricing-self-host — **only two self-host tiers: OSS (free) and Enterprise (custom pricing, "talk to sales")**; retention is Enterprise-only
- https://langfuse.com/self-hosting, https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse, https://langfuse.com/self-hosting/deployment/infrastructure/containers — Postgres + ClickHouse + Redis/Valkey + S3 + web + worker
- https://langfuse.com/docs/observability/features/masking — `mask_otel_spans`, client-side
- https://langfuse.com/docs/administration/data-retention, https://langfuse.com/pricing, https://langfuse.com/docs/observability/sdk/overview, https://langfuse.com/docs/observability/sdk/python/instrumentation, https://langfuse.com/security/data-isolation
- https://langfuse.com/integrations/model-providers/anthropic, https://langfuse.com/integrations/model-providers/google-gemini — no native wrappers
- https://github.com/orgs/langfuse/discussions/6565, https://github.com/orgs/langfuse/discussions/5924
- https://github.com/langfuse/langfuse — repo stats (34.3k★ / 3.7k forks, created 2023-05-18)

**Grafana Cloud / OpenLLMetry (for §1.4):**
- https://grafana.com/docs/grafana-cloud/cost-management-and-billing/understand-your-invoice/traces-invoice/ — **30-day minimum retention**, +$0.10/GB per additional 30 days, sub-30-day requires contacting Support
- https://grafana.com/blog/a-complete-guide-to-llm-observability-with-opentelemetry-and-grafana-cloud/ — managed OTLP gateway
- https://www.traceloop.com/docs/openllmetry/integrations/grafana, https://github.com/traceloop/openllmetry — Apache-2.0 OTel instrumentation, Tempo export

**Screened and disqualified:** LangSmith (self-host is Enterprise-contract-only), Helicone (cloud-only), Braintrust (proprietary SaaS, not self-hostable).
