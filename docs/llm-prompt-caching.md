# LLM prompt caching (Gemini)

What Gemini's caching actually does to this codebase's chat prompts, measured
against the live API on 2026-09-17. Read this before optimising prompt size or
ordering — two plausible-sounding optimisations were measured and rejected, and
the reason they fail is not obvious from the code.

## The shape of a chat turn's cost

Measured on production `chat_turn_metrics`, n=3,165 turns:

| rounds | turns | mean input tokens | mean cached | cache fraction |
|---|---|---|---|---|
| 1 | 346 (11%) | 8,230 | 3,241 | 40.6% |
| 2 | 2,819 (89%) | 21,129 | 7,967 | 38.1% |

`input_tokens` and `cached_tokens` in that table are **summed across every LLM
round of a turn** (`src/agents/chatbot.py`, the `in_tok +=` sites), so a
2-round turn's 21,129 is not one prompt — it is the same conversation sent
about 2.5 times.

Single-round turns give the clean reading: **one prompt is ~8,230 tokens**.
Multi-round turns average ~2.47 `llm_calls`, because a turn that exhausts
`max_tool_rounds` still wanting tools makes a third call to force a plain
answer.

Consequence: the extra rounds account for `2,819 x (21,129 - 8,230)` = 36.4M of
the sample's 62.4M input tokens — **58% of the bill is round two and beyond**.

Rounds are not reducible by tidying. `_handle_with_tools` breaks out of its
loop the moment a response carries no tool calls, so a second round happens
only when the model actually asked for a tool. There are no wasted rounds.

## Why round two never reuses round one

Round two's request contains round one's as a strict prefix — verified both by
reading `_handle_with_tools` (the message list is built once and only appended
to) and by serialising both requests and confirming `startswith`. So prefix
caching ought to hit. It does not, and the reason is the useful finding here.

Measured against the live API:

- Delay sweep between the two calls at 0, 1, 2, 5, 15 and 60 seconds, 3
  repetitions each (18 pairs, 36 calls): round two's cached count was **exactly
  4,024 every single time**, never once reflecting round one's turn-specific
  content. Zero variance. Not a cache-write race.
- Eight identical back-to-back calls: cached held at exactly 4,028 and never
  grew. Not "needs more occurrences" either.
- The same content flattened into a single `contents` entry and repeated:
  cached 0 on the first call, then **4,078 (55%) from the second onward**.
- The same content as a real multi-entry `contents` array (21 entries), and
  again forced to a single role to rule out alternation: **cached stayed at 0,
  every repetition**.

**Gemini's implicit cache covers one contiguous text field — in practice
`system_instruction` — and does not accumulate across a multi-entry `contents`
array.** Every real chat turn's `contents` is inherently multi-entry (history,
then the user message, then the tool call and its result), so round two
structurally cannot reuse round one. Tool declarations live in `config.tools`
and never contributed to the cached count either way.

This explains the otherwise puzzling stability of the production number: the
~37-40% cache rate is simply `system_instruction / whole prompt`, and it is
capped there by construction, not by tuning.

## Explicit caching does work

`client.caches.create` on the Developer API (SDK 2.2.0, no Vertex required):

- minimum **1,024 tokens** (a 10-token probe is rejected with
  `min_total_token_count=1024`)
- creating a 7,881-token prefix took **1.77s**
- calls referencing it reported **7,881 of 7,904 cached — 99.7%**, against
  4,028 (51%) for identical content sent without the reference

1.77s of creation latency cannot sit on a live turn, so using this means one
cache per tenant over the static prefix, with a TTL and a refresh path.

## Two optimisations that were measured and rejected

**Moving retrieved sources out of the system prompt so history could cache.**
The premise was wrong twice over. First, `bootstrap.py` constructs the chat
agent with `enable_tools=True`, so production always runs `_handle_with_tools`,
which calls `_compose("")` — `rag_context` is always empty on that path and KB
content arrives as a tool result instead. Second, and decisively: moving
content out of `system_instruction` and into `contents` moves it out of the
only region that caches at all.

**Anchoring the history window so it stops sliding.** `MAX_HISTORY_TURNS = 10`
is replayed as a sliding tail, so turn N's history and turn N+1's differ at
their first message and the byte-prefix breaks at the start of history. Real,
but small: history measures ~4.3KB mean and ~700-2,100 tokens even at the
eviction boundary, against a 14-26K-token turn. And since `contents` never
caches, making history prefix-stable would not help regardless.

## Cost figures are understated

`ProviderCost` carries `cost_per_1k_input_tokens` and
`cost_per_1k_output_tokens` and **no cached-input rate**, so
`src/api/chat_cost.py` bills a cached token at the same rate as a fresh one.
Every cost number this platform reports is therefore wrong in the same
direction, and the size of the error is whatever discount the provider applies
to cached input. Modelling that rate is a precondition for any credible
projection of what caching work would save.

## Measurement notes

Production figures come from operator-run queries against the production
database; the token-count columns are populated from Gemini's own
`usage_metadata`. Live API measurements used the repo's real
`GeminiLLMAdapter` against `gemini-3.5-flash` with a production-shaped ~7,900
token prompt, not synthetic filler. Byte-prefix length between consecutive
requests, used in the rejected experiments above, turns out not to predict
Gemini cache hits at all — it is an upper bound on what a prefix-caching
provider could reuse, and this one reuses far less.
