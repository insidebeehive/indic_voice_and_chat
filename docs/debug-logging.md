# Debug logging

DEBUG is a switch an operator flips when something is being investigated, and
it must then carry enough to reach a conclusion without reading source or
querying the database. It is off in normal running.

```bash
VOX_LOG_LEVEL=DEBUG     # validated against a whitelist in src/config.py; needs a restart
```

Everything below applies to DEBUG only. INFO and above keep the redaction and
keys-only conventions they already have — `_redact_pii_for_log`
(`src/rag/context_builder.py`), `param_keys` rather than values
(`src/chatbot/tool_executor.py`), fingerprints rather than identifiers. Do not
change those call sites.

## What a DEBUG line carries

**Full values.** Customer text, media URLs, identifiers, prompts, request and
response bodies, config values, the numbers a branch decided on. A redacted
debug line is useless: the reason the voice-reply failure below took a database
query to diagnose is that the skip carried nothing at all.

This is a deliberate operator decision, recorded here so nobody reverts it
after reading the redaction comments elsewhere in the tree: at DEBUG, PII is
logged. It is bounded by the level being off in normal running.

**Never credentials.** An API key, bearer token, signing secret or password is
not a diagnostic value and appears at no level. This is the one exception and
it has no cases.

**Note on retention:** when `GRAFANA_LOKI_PUSH_URL` is set, the root level
gates the Loki handler too, so DEBUG ships to Grafana Cloud and persists for
its retention window. "On for an hour" means "stored for the retention period".

## The helper

Use `debug_event` (`src/utils/logging.py`) rather than `log.debug` with an
f-string, so values land as structured fields a Loki query can filter on
instead of text a human has to grep:

```python
debug_event(log, "gemini generate request", model=model, system=system,
            contents=contents, config=gen, cache_name=cache_name)
```

Values are keyword arguments.

## Naming an event

`<component> <operation> <phase>` — space-separated, with the operation itself
snake_case where it needs more than one word:

```
gemini generate request          retriever dense_search response
elevenlabs tts response          ingestion pdf_page_extract_failed
```

The component prefix is the point: `event=~"retriever .*"` scopes a query to
one subsystem, which a flat name cannot do. Phase is `request`/`response` for a
boundary; for a decision or a skip, name the outcome (`..._dropped`,
`..._failed`, `..._triggered`).

A name is a query key. Once an event has shipped, treat it as an interface and
do not rename it — a rename silently splits any query spanning it inside Loki's
retention window, and breaks saved queries with no error.

Three shipped events in `src/api/chat.py` predate this convention and use flat
snake_case (`chat_frame_received`, `chat_reply_frame_sent`,
`tts_reply_synthesized`). They stay as they are, under the rule above. Match
the convention in new code rather than these three.

## Cost when DEBUG is off

`debug_event` is a no-op at INFO and above — but **its arguments are evaluated
before it is called**, so anything built to feed it is paid for in normal
running, forever:

```python
# Wrong: builds the list on every search, DEBUG off or not.
debug_event(log, "retriever search response", hits=[c.id for c in results])

# Right.
if log.isEnabledFor(logging.DEBUG):
    debug_event(log, "retriever search response", hits=[c.id for c in results])
```

Pass what you already hold. The guard is for work on a path that actually runs
— per request, per turn, per chunk, per document — where an unguarded
comprehension is paid by every request forever to build a line nobody is
reading. `search_combined` is the shape to watch: it runs on every chat turn,
so its events build `raw_total`, the tags and the scores inside the level
check, not outside it.

It is not an absolute rule, and applying it as one buries the code in guards.
A rare error path over a handful of items does not need one — the bogus-citation
event in `context_builder.py` fires only when the model invents a citation, and
guarding a few strings on a path that mostly never executes costs more
readability than it saves. Judge by how often the path runs and how big the
collection is, not by whether a comprehension is present.

## Key names collide with LogRecord

`logging` owns `name`, `module`, `filename`, `args`, `msg`, `levelname` and
friends on every record. Passing one as a keyword raises inside the handler and
takes out the caller — which is why `debug_event` renames collisions by
appending an underscore rather than letting them through.

That keeps the process up but the value lands somewhere nobody will look:
`filename="kb-manual.pdf"` is emitted as `filename_="kb-manual.pdf"`, while
`filename` still appears holding `logging.py`. So the rename is a backstop, not
a licence — qualify the key yourself (`document_filename`, `source_module`)
whenever the natural name is one of `logging`'s.

`event` is reserved one level up as well: it names the line itself, and it is
`debug_event`'s own second parameter. It is positional-only for that reason, so
`event="answered"` lands in the values and is renamed rather than raising
`TypeError: got multiple values for argument 'event'` — which is what it did
when the state-machine and voicebot instrumentation both reached for it, taking
out every state transition from inside a log line. Prefer a qualified name
(`sm_event`, `trigger`) over relying on the rename.

**Run the suite at `--log-level=DEBUG` as well as normally.** At the default
level `debug_event` returns before touching its arguments, so an ordinary run
exercises none of the code a pass adds — every collision above, and any
exception building a value, is invisible until the level is raised. That is
also precisely when it would first fire in production: during an incident.

## What to instrument

**Boundaries.** Every outbound HTTP call and its response, every database
write, every provider call — full request and response bodies, including
prompts. These are the largest lines in the system and the most useful.

**Decisions.** Any branch that changes a user-visible outcome, with the values
that decided it — not just which way it went.

**Skips.** Every early return, `continue` and swallowed exception that skips
work a customer or operator would notice, with the reason and the deciding
values. This is the category that motivated the standard.

**State transitions.** Session mode changes, escalations, handovers, cache
arming and eviction — before and after.

**Not:** ordinary control flow. A parser returning early, a helper with nothing
to do, a loop guard. Instrumenting those buries the lines that matter.

## Why the skip category exists

A tenant reported that voice-note replies were not working. The configuration
was correct, the API key was present, and the customer had sent a voice note.
The cause was in `_synthesize_reply_audio` (`src/api/chat.py`):

```python
if not text or len(text) > _TTS_MAX_REPLY_CHARS:   # 300
    return None
```

The reply exceeded 300 characters, so nothing was synthesised — silently, with
no log and no metric. Three conditions in that one function return `None` with
no trace, and from outside they are indistinguishable from the feature being
broken. Diagnosing it needed a database query and a code read.

## On a hot path, log transitions — not frames

`src/pipeline` runs 50 audio frames a second per open call, and parts of
`engine.py` run per LLM token. A line per frame there is not "verbose": one
call under investigation fills `LokiPushHandler`'s queue (bounded at 10,000,
`except queue.Full: pass`) and silently evicts the lines the operator turned
DEBUG on to read. The instrumentation destroys the evidence it was added for.

The answer is not to skip these paths — they are where voice problems live —
but to log the **edges**, carrying the accumulated state:

```python
def detect(self, pcm16: bytes) -> VADFrame:
    energy = rms_energy_pcm16(pcm16)
    is_speech = energy >= self._threshold
    if is_speech != self._last_is_speech:     # the flip, not the frame
        debug_event(log, "vad energy_threshold transition",
                    is_speech=is_speech, energy=energy, threshold=self._threshold)
        self._last_is_speech = is_speech
```

Two or three lines per utterance instead of fifty a second, and they answer
more: `energy` beside `threshold` at the moment of the flip is what
`BARGE_RMS` was tuned by hand against, with temporary diagnostics that were
then deleted. The same move works per token — `SentenceDetector` logs each
sentence EMITTED rather than each `feed()`, and `_SpokenTextExtractor` exposes
what it accumulated for one read at end of turn.

**Watch for level-triggered returns.** `EndpointDetector.feed` returns True on
every silent frame once the threshold is crossed, not just the first, so
logging on the return value floods the moment a caller is slow to `reset()`.
Both it and `turn_capture.accumulate_and_detect` latch instead. Ask of any
hot-path event: does the condition GO true here, or is it merely true here?

## Logging cannot observe its own configuration

`load_settings` runs before `configure_logging` and always will: the lifespan
resolves settings precisely to learn the log level, then configures logging
with it. So every `debug_event` on the config load path evaluates against an
unconfigured root logger and goes nowhere on a live server — while firing
normally under pytest, which raises the root level itself. Instrumentation that
looks healthy in tests and is silent in production is the worst of both.

It was not academic: the unknown-key sweep exists to surface a dead config key
like `config/default.yaml`'s inert `tts.model`, and it would have been silent
on every boot — the one place it was worth having.

The fix is not to reorder (that is circular) but to stash and replay:
`load_settings` keeps its findings in `_PENDING_LOAD_DIAGNOSTICS`, and
`main.py` emits them as one event immediately after `configure_logging`. The
replay is a single literal-named event carrying the payloads, rather than a
loop re-emitting each with a computed name — a computed name would defeat
`test_debug_event_call_sites.py`'s literal-name rule, and the rule is worth
more than the convenience.

The general shape: **anything decided before the logger exists needs somewhere
to wait.** Import-time work in `main.py` — router mounting, middleware, the
dev-console token gate — has the same problem and is deliberately left
uninstrumented rather than logging into the void.

## An event that misleads is worse than no event

The `rag` pass added `rag context chunk_dropped_for_budget` to
`build_rag_context`, naming every chunk cut to fit `max_context_chars`. It is
the right instrument for the incident that motivated this doc. It was also,
on production, wrong.

`build_rag_context` has two callers. `_single_shot` composes its `text` into
the system prompt. `_handle_with_tools` — the production chat path, per
`bootstrap.py` — does not: there `rag.text` is discarded and KB content reaches
the model as a `role="tool"` message instead, untruncated. The built context
survives only to give `apply_hallucination_guard` its citation scope.

So on production the event fired for a truncation that did not affect what the
model saw. An operator investigating a hallucination would read
`dropped_tags=["kyc.md#4"]`, conclude the model never received that chunk, and
spend the afternoon raising a budget that changes nothing. A missing log costs
an investigation; a confidently wrong one costs the investigation and sends the
next one the same way.

The fix is the `purpose` argument that call site now passes. The general rule:
when a value is computed in one place and consumed differently depending on the
caller, the event has to record which consumer it was built for. "What
happened" is not enough — an event must not be readable as a claim about
something it does not govern.

## Coverage

One pass per package, in the order an operator would need them. A package is
done when every site in it has been classified — instrumented, or justified as
control flow, or named as already covered by an existing log or a
`chat_turn_metrics` field.

58 of 182 files carry a `debug_event`. That count understates coverage a
little — `chatbot/tool_executor.py` is done and uses none, because every path
there already logs with a discriminator — but not by much.

What is done is every path a live conversation touches, on both channels: the
inbound webhook or websocket, tenant and provider config resolution, retrieval
and ranking, tool dispatch, the provider call, the pipeline, and the reply
going back out. A chat or voice turn can be followed end to end with full
request and response bodies.

What is left is mostly not on that path: the backoffice and reporting
endpoints, dialogue/campaign machinery, `models`, `observability`, and
`benchmarks`. The two worth doing next on operator value are `src/bootstrap.py`
(1,301 lines, where wiring decisions are made once at startup and are invisible
afterwards) and `src/api/dev_console.py` plus `external_chat.py`.

The conventions this doc describes are enforced by
`tests/unit/test_debug_event_call_sites.py`, which AST-walks every call site.
Reserved-key collisions and flat event names both survived being written down
here and restated in per-pass instructions, twice, before that test existed —
a convention that depends on remembering is not a convention.

| package | files | lines | status |
|---|---|---|---|
| `api` — chat request path (`chat.py`) | 1 | — | **done** |
| `api` — telephony webhooks (`telephony_hooks/twilio/exotel/stringee/crm`, `answer_paths`) | 6 | 1,923 | **done** |
| `api` — live media bridges (`browser_bridge`, `live_bridge_base`, `telephony_live_bridge`, `telephony_stringee_bridge`, livekit ×3, `gemini_live_bridge`) | 8 | 2,717 | **done** |
| `api` — tenant/CRM config (`tenants`, `crms`, `crm_kb`, `catalog`) | 4 | 2,877 | **done** |
| `api` — the remaining 22 files | 22 | ~5,400 | not started |
| `agents` — `chatbot.py` | 1 | 2,212 | **done** |
| `agents` — `voicebot.py`, `state_machine.py`, `base.py` | 3 | 1,337 | **done** |
| `chatbot` — `tool_executor.py` | 1 | — | **done** (needed nothing; every path already logs with a discriminator) |
| `chatbot` — `deposit_verification.py` | 1 | — | **done** |
| `chatbot` — the rest | 5 | — | not started |
| `auth` — `registry.py` (`get_chat_tts`) | 1 | — | **done** |
| `auth` — the rest | 8 | — | not started |
| `providers` | 33 | 5,045 | **done** (23 instrumented; 10 without — 8 empty `__init__`, plus `model_catalog.py` and `voice_catalog.py`, which are static tables. See 4a3be36) |
| `rag` | 5 | 2,586 | **done** (4 files instrumented, `__init__` empty) |
| `pipeline` | 8 | 1,379 | **done** (7 files instrumented, `__init__` empty) |
| `dialogue` | 11 | 2,017 | not started |
| `campaign` | 5 | 863 | not started |
| `models` | 11 | 1,297 | not started |
| `observability` | 4 | 1,432 | not started |
| `integration` | 6 | 662 | not started |
| `analysis` | 3 | 459 | not started |
| `utils` | 7 | 912 | not started |
| `interfaces` | 8 | 449 | not started |
| `benchmarks` | 11 | 2,472 | not started |
| `src/` root — all 7 files | 7 | 3,436 | **done** (`defaults.py`/`exceptions.py` are static data, nothing to classify) |

Tracked per file rather than per package where a pass covered only part of
one: the first pass followed the chat request path across four packages rather
than finishing any single directory, and recording it as "api: done" would
claim 40 untouched files.

`agents/chatbot.py` is the case that shows why "done" needs the definition
above. The first pass instrumented its silent skips and marked the file done on
that basis, while 2,212 lines of turn body, tool dispatch and KB tool-result
assembly stayed unclassified behind 3 `log.debug` calls. The `agents` pass
finished it. The lesson is in the status column, not the code: a file can be
genuinely improved by a pass and still be nowhere near done, and recording it
as done is how coverage goes dark while the checklist reads as finished.

What kept that gap narrow while it lasted is worth knowing, because it decides
what this package should NOT carry: `src/providers/llm/gemini.py` logs
`contents` in full on every request, and the `role="tool"` KB payload the
production path feeds the model rides in there. "What did the model see" and
"what did it say" are answered at the provider boundary. So the events added
here are the decisions BETWEEN those boundaries — which tool was selected and
with what arguments, why a requested tool was not dispatched, why the round
loop stopped — and not a second copy of the prompt.

The first pass (commit 1d97f8b) classified ~90 sites and instrumented ~20. It
also found a class the original framing missed: sites where the customer is
told something happened and the database never recorded it — a turn persisted
against a missing session row, a human agent's reply forwarded but never
written to the transcript, a mode change confirmed to the customer while the
row still said otherwise. Those were silent `if row is None: return` and
`if r:` with no else. Worth looking for specifically in later passes; they are
harder to spot than an early return because the code reads as a success path.

Each pass produces a classification table in its commit message: every site
examined, its category, and what covers it where it was left alone. That table
is what makes "is this package done" answerable by someone who was not there.
