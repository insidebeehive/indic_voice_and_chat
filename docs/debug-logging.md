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
debug_event(log, "tts_reply_skipped", reason="reply_too_long",
            length=len(text), cap=_TTS_MAX_REPLY_CHARS, tenant_id=tenant.id)
```

The event name is a stable identifier — snake_case, specific enough to query on
its own. Values are keyword arguments. It is a no-op when DEBUG is off, so
building the values must itself be cheap; pass what you already have rather
than computing something for the log.

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

## Coverage

One pass per package, in the order an operator would need them. A package is
done when every site in it has been classified — instrumented, or justified as
control flow, or named as already covered by an existing log or a
`chat_turn_metrics` field.

| package | files | lines | status |
|---|---|---|---|
| `api` (chat path first) | 41 | 16,394 | in progress |
| `agents` | 5 | 3,523 | not started |
| `chatbot` | 7 | 1,729 | not started |
| `providers` | 33 | 5,045 | not started |
| `rag` | 5 | 2,586 | not started |
| `pipeline` | 8 | 1,379 | not started |
| `auth` | 9 | 1,755 | not started |
| `dialogue` | 11 | 2,017 | not started |
| `campaign` | 5 | 863 | not started |
| `models` | 11 | 1,297 | not started |
| `observability` | 4 | 1,432 | not started |
| `integration` | 6 | 662 | not started |
| `analysis` | 3 | 459 | not started |
| `utils` | 7 | 912 | not started |
| `interfaces` | 8 | 449 | not started |
| `benchmarks` | 11 | 2,472 | not started |

Each pass produces a classification table in its commit message: every site
examined, its category, and what covers it where it was left alone. That table
is what makes "is this package done" answerable by someone who was not there.
