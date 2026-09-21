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

## Cost figures account for cached tokens

`ProviderCost` carries a `cost_per_1k_cached_tokens` column alongside
`cost_per_1k_input_tokens` and `cost_per_1k_output_tokens` (migration
`alembic/versions/0022_provider_cost_cached_rate.py`), and `src/api/chat_cost.py`
bills a chat turn as `(input_tokens - cached_tokens) * in_rate + cached_tokens *
cached_rate + output_tokens * out_rate`.

The column is nullable with no server default, and that's deliberate: `NULL` means
"no cached rate configured for this `(provider, model)`," and `compute_chat_turn_cost`
falls back to billing cached tokens at the *full* input rate in that case, never at
`0.0`. A missing cached-rate row silently collapsing cached-token cost to near-free
would look like a spectacular saving and be a reporting bug, not a real one — the
safety direction is "never invented as too cheap," matching the platform's existing
figures until a rate is explicitly set. Every existing `provider_costs` row (and
every new row that doesn't set the column) is `NULL` on this migration, so upgrading
never changes a dollar figure the platform already reports; the discounted rate only
takes effect once a `(kind='llm', provider, model)` row has it set explicitly.

## Explicit caching as implemented

Built behind `GEMINI_EXPLICIT_CACHE` (unset/falsy = disabled, the shipped
default) plus `GEMINI_CACHE_TTL_S` (default 1800s = 30 minutes). The registry
lives entirely inside `GeminiLLMAdapter` (`src/providers/llm/gemini.py`) —
`src/interfaces/llm.py` carries no new fields, so the adapter derives
everything (system text, tool declarations) from what `generate()` is already
handed. TTL of 30 minutes was picked against two numbers in this doc: a
30-minute window is long enough that an active tenant's turns — which arrive
far more often than every 30 minutes — keep reusing one cache instead of
re-paying the 1.77s-plus-a-full-token-write creation cost every restart of
the window, and short enough to bound idle storage billing (explicit caches
bill storage per token-hour) to at most half an hour after a tenant goes
quiet.

`build_chatbot_system_prompt` gained `include_variable_tail: bool = True`
(default preserves today's output byte-for-byte) plus a standalone
`build_chatbot_variable_tail()`, so the static body (cacheable) and the
per-turn tail (sources / current time / language directive — minute-granular,
would invalidate a cache every minute if left in `system_instruction`) can be
built separately. `ChatBotAgent` only uses the split when `cache_split_prompt`
is on (wired in `src/bootstrap.py`, gated on both the platform LLM being the
Gemini adapter and the env flag) — when it's on, `_compose` folds the tail
into the user turn's `contents` message instead, framed so the model reads it
as platform-supplied context rather than something the customer said.

A cache is not created on the first sighting of a `(model, system, tools)`
key — only from the second sighting, so a one-shot or minute-varying system
prompt (voice, analysis) never burns a real `caches.create` call. Chat's
static body is stable across a turn's ~2.47 LLM calls, so it typically gets
created within the first one or two turns of traffic for a given tenant
config. Any prompt/tenant/tool-catalog/model change produces a different
sha256 key, so a stale cache is never reused for changed content — it just
ages out.

### Live verification, 2026-09-17

Verified against the live Developer API with a production-shaped prompt: the
real `build_chatbot_system_prompt` static body (betting pack, all tool flags
on, `include_variable_tail=False`, 20,138 chars) plus the FULL production tool
set a fully-configured tenant would actually send — 3 builtins +
`submit_deposit_verification` + the entire 22-tool CRM catalog
(`src/chatbot/catalog.py`'s `ALL_TOOLS`) — 26 tools total — plus ~10 turns of
realistic Hinglish chat history.

- **Generate-side contract, confirmed live**: a `generate_content` call
  combining `cached_content` with either `system_instruction` or `tools` is
  rejected — `400 INVALID_ARGUMENT`, verbatim: *"CachedContent can not be used
  with GenerateContent request setting system_instruction, tools or
  tool_config. Proposed fix: move those values to CachedContent from
  GenerateContent request."* The implementation matches this: on a hit it
  sends neither.
- **Tool-declaration token count (the size of the prize)**: caching the
  static system body alone costs **4,574 tokens**; adding the full 26-tool
  declaration set to the same cache costs **9,468 tokens** — the 26 tools
  alone are **4,894 tokens**, i.e. tool declarations are *larger* than the
  static system prompt they ride alongside. This is materially bigger than
  the ~4,024-4,028-token implicit-cache ceiling measured above — the implicit
  cache structurally cannot see this content at all (`config.tools` never
  contributed to the cached count, confirmed above), so this is real,
  previously-unreachable coverage, not a duplicate of the existing implicit
  hit.
- **Before/after `cached_content_token_count`, via the real adapter**: a cold
  uncached call against this prompt (9,683 total prompt tokens, tools
  included) reported **0 cached** (the implicit cache did not warm on a
  single one-off call in this run — consistent with it being best-effort, not
  guaranteed, and unrelated to the explicit-cache path). Once the adapter's
  own second-sighting rule created the cache and a third call hit it, the
  same 9,683-token prompt reported **9,468 cached (97.8%)** — matching the
  99.7% figure from the original explicit-cache experiment above, on a larger
  and fully tool-bearing prompt this time.
- **Tool-calling still works on a cache hit**: a follow-up cached call that
  should trigger `search_knowledge_base` did — `tool_calls=['get_player_wallet',
  'get_player_bonuses', 'search_knowledge_base']` — confirming caching the
  tool declarations does not silently disable function-calling.
- All caches created during this verification (2 in the tool-declaration
  measurement, 1 in the contract probe, 1 created by the adapter itself) were
  deleted immediately after use. Total spend: a handful of small
  `generate_content` calls plus four short-lived caches, well under a cent.

## Measurement notes

Production figures come from operator-run queries against the production
database; the token-count columns are populated from Gemini's own
`usage_metadata`. Live API measurements used the repo's real
`GeminiLLMAdapter` against `gemini-3.5-flash` with a production-shaped ~7,900
token prompt, not synthetic filler. Byte-prefix length between consecutive
requests, used in the rejected experiments above, turns out not to predict
Gemini cache hits at all — it is an upper bound on what a prefix-caching
provider could reuse, and this one reuses far less.
