# Decouple Betting/Gambling Vertical Assumptions From Core Platform — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** This platform was designed to be industry-agnostic (multi-tenant, pluggable CRM tools/KB), but three places in the *core* code — not CRM/tenant config — currently hardcode betting/gambling-vertical assumptions as the unconditional default for every tenant and every CRM: the chatbot's base system-prompt scope, the knowledge-base bundle auto-seeded at boot, and the TTS pronunciation dictionary. Onboarding a non-gambling CRM today would silently inherit all three. This plan decouples them: each becomes an explicit, opt-in, per-CRM choice, with the existing betting operator's behavior preserved byte-for-byte via explicit backfill. **Documentation is intentionally out of scope until the final task** — do not add "note: this is stale" caveats to any doc while Tasks 1–3 are in flight; docs get one comprehensive rewrite at the end, once, describing the finished system as its current, only state.

**Architecture:** No change to the multi-tenant DB-backed architecture itself. Adds one selector column each to `crms` for prompt pack, bundled-KB pack, and pronunciation overrides; splits the monolithic default content in `src/dialogue/prompts.py` and `src/pipeline/text_normalize.py` into a generic base + a betting-specific pack; changes `src/main.py`'s KB boot-seeder from "seed every CRM unconditionally" to "seed only CRMs that opted into a named pack."

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.x async, Alembic, pytest.

**Explicitly NOT in scope:**
- The casino/sports/matka module KB docs and their ingestion path (`_INGESTIBLE_DOCS` in `src/api/knowledge.py`, the per-tenant `/knowledge/ingest-layout`-style route) — these are correctly **tenant**-scoped already (different tenants under the same CRM enable different subsets of these verticals) and must stay tenant-scoped. Do not move them to CRM level, do not touch this mechanism at all.
- `src/chatbot/catalog.py` (the "standard CRM tool catalog") and `src/api/dev_console.py`'s default test prompt — both noted as lower-priority in discussion, deliberately deferred to a future pass, not part of this plan.
- Rewriting any documentation before Task 4.

## Global Constraints

- Branch is `stage`. **Do not commit anything unless the project owner explicitly asks** — this overrides any other project's convention of committing directly; confirm before every commit in this plan, not just the first.
- **`src/dialogue/prompts.py` is a prompt file.** Per this repo's `CLAUDE.md`: *"Always confirm with the user before editing any prompt file (`prompts.py` etc.), even in auto mode."* This is non-negotiable and applies even though this plan describes the intended edit — **stop and get explicit, fresh confirmation from the project owner immediately before making any edit to `src/dialogue/prompts.py` or the new `src/dialogue/packs/*.py` files**, in Task 1. Do not treat approval of this plan as approval of the edit itself.
- Run `.venv/bin/python -m pytest tests/ -q` after every task. Baseline as of 2026-09-04: **1927 passed, 3 failed** — `test_chat_routes.py::test_claim_session_and_agent_ws` (pre-existing, documented, unrelated), `test_dev_console.py::test_place_call_passes_tenant_creds_to_adapter` (new, looks like real `.env` Stringee creds leaking into a monkeypatched test — unrelated to this plan, do not chase), `test_pgvector_crm_scoping.py::test_crm_scoped_chunk_is_isolated_from_tenant_scoped_chunk` (missing `pgvector` package in `.venv`, fixable with `pip install -e ".[dev]"`, unrelated to this plan). Expect these three to persist unchanged; any *new* failure is this plan's responsibility to fix before moving on.
- Every task that changes seeded/default content for the **existing** production CRM must include an explicit backfill so that CRM's live behavior (prompt output, seeded KB docs, TTS pronunciation) does not change. This is not "nice to have" — a silent behavior change to the one CRM already in production is a regression, even if the new *code path* is correct.
- Verify current function/column/file names and line numbers with `grep`/`git grep` before editing — this plan was written from a point-in-time investigation and specifics may have shifted.
- No secrets in any commit, comment, or this plan's own working notes.

---

### Task 1: Split the ChatBot system prompt's SCOPE + examples into a generic core + a selectable "prompt pack"

**Scope note (narrower than originally estimated):** only `build_chatbot_system_prompt` in `src/dialogue/prompts.py` needs splitting. `build_voicebot_system_prompt` and `build_s2s_system_instruction` (the VoiceBot builders) are already fully campaign-script-driven (`VoiceBotScript` dataclass supplies all content) with no hardcoded vertical text to extract — do not touch them in this task. Within `build_chatbot_system_prompt`, most sections (identity/anti-leak core, DATA RULE's core enforcement mechanism, TOOL USE's reasoning framework, ESCALATION, `DEPOSIT DISPUTE VERIFICATION`'s gating on `has_deposit_verification_tool`, ACTION VALUES, RESOLVED's core logic, LANGUAGE, RESPONSE QUALITY, the JSON schema) are already industry-neutral and must stay in `prompts.py` unchanged. Only two things move into packs:
1. The **SCOPE section** (today's lines ~656–746): the tier-1 "GENERAL (...)" category list, `player_scope` (both the has-tools and no-tools variants), `operator_scope` (both variants), and the WITHDRAWAL STATUS sub-block.
2. A handful of **illustrative example phrases** embedded in otherwise-generic paragraphs: DEPTH-MATCHING's consequential-action examples (the quoted phrases in bullet 1, the "self-exclusion/cooling-off menu" label in bullet 3, and the self-exclusion-specific closing paragraph), TOOL USE's consequential-action example, and RESOLVED's "update your KYC/bank details" example.
3. **DATA RULE** (today's lines ~608–639) has betting vocabulary woven into what looked like core-mechanism sentences, not confined to one bracketed example as originally scoped — confirmed during Task 1's own investigation. Extract as **five** named pack constants, not the two originally planned, keeping the rule's actual logic (never invent specific numbers, tool-call-or-decline, escalate for real PII) identical between packs — only the noun phrases below differ:
   - `DATA_RULE_INVENT_EXAMPLES` — betting: `"account balances, transaction IDs, the player's own bank/UPI details, bonus amounts"`; generic: `"account balances, transaction IDs, the customer's own bank/payment details, credit amounts"`.
   - `DATA_RULE_CONCEPT_EXAMPLES` — the original spot #1: betting keeps "why KYC exists, why deposits can be delayed, how self-exclusion works in general"; generic gets a neutral equivalent (e.g. "why verification exists, why payments can be delayed, how account closure works in general").
   - `DATA_RULE_PROPER_NAME_EXAMPLES` — betting: `"a game, market, provider"`; generic: `"a product, plan, provider"`.
   - `DATA_RULE_FREE_TOPICS` — betting: `"responsible gaming tips, game rules, platform features, strategies, how betting works"`; generic: `"safety/usage tips, product rules, platform features, best practices, how the service works"`.
   - `DATA_RULE_CATALOG_SENTENCE` — the original spot #2, now including its noun list (not just the trailing quoted queries): betting's `"games, sports, leagues, tournaments, matka variants, providers, markets"` + example queries; generic's neutral equivalent (e.g. "products, categories, plans, providers") + example queries in a neutral register.

**Files:**
- New: `src/dialogue/packs/__init__.py`, `src/dialogue/packs/betting.py`, `src/dialogue/packs/generic.py` — each holding only plain string/text constants for the two categories above (SCOPE blocks + example phrases), no logic.
- Modify: `src/dialogue/prompts.py` — `build_chatbot_system_prompt` gains a `prompt_pack: str = "generic"` parameter; does a `PACKS = {"betting": betting, "generic": generic}` lookup with `.get(prompt_pack, generic)` (never raises on an unrecognized/missing key — always falls back to generic); pulls the SCOPE assembly and the example phrases from the resolved pack module instead of the inline text. Everything else in the function is untouched.
- Modify: `src/models/crm.py` — add `prompt_pack: Mapped[str | None]` column on `Crm`.
- New Alembic migration adding that column, **plus a data migration/backfill** setting `prompt_pack = 'betting'` explicitly on every existing `crms` row (do not rely on an implicit code-level default for the production row — verify the actual row count/ids before writing the backfill SQL, don't assume there's exactly one).
- Modify: `src/auth/db_resolver.py` — mirror the existing `crm_id=tenant.crm_id` denormalization (currently at line ~72) by also joining/fetching the linked `Crm` row's `prompt_pack` and copying it onto `TenantContext.settings.prompt_pack`, defaulting to `"generic"` when that value is NULL or the tenant has no linked CRM at all.
- Modify: `src/bootstrap.py::make_chatbot_factory` (the `ChatBotAgent(...)` construction around line ~561) — read `prompt_pack` off `tenant.settings` the same way `company_name=tenant.name` and `tenant_timezone=getattr(tenant.settings, "timezone", ...)` are already read there, and pass it into the constructor call.
- Modify: `src/agents/chatbot.py` — `ChatBotAgent.__init__` (~line 252) gains a `prompt_pack: str = "generic"` parameter, stored as `self._prompt_pack` (same pattern as `self._company`/`self._tenant_timezone`); the `build_chatbot_system_prompt(...)` call (~line 805) passes `prompt_pack=self._prompt_pack`.
- Modify/extend: whatever test file(s) currently cover `src/dialogue/prompts.py` and `src/agents/chatbot.py` (`grep -rl "build_chatbot_system_prompt" tests/unit/`).

**Interfaces:**
- Selection is pure data, not runtime inference: **which `Crm` row a tenant is linked to** decides the pack, resolved once at auth/tenant-resolution time (`db_resolver.py`), the same place `crm_id`, `operator_id`, and telephony keys are already resolved — nothing in the chat turn itself influences pack choice.
- `packs/generic.py` supplies a minimal, industry-neutral SCOPE (e.g. tier-1 "registration, account setup, billing, general features, security, tech help"; `player_scope`/`operator_scope` phrased as "account status, billing, profile info" / "how this business is configured — pricing, policies, hours, service area"; no WITHDRAWAL STATUS block; generic stand-ins for the DATA RULE/DEPTH-MATCHING/TOOL USE/RESOLVED example phrases with no gambling vocabulary — "a subscription cancellation" instead of "self-exclusion," etc.).
- `packs/betting.py` is today's SCOPE text and example phrases, moved verbatim (not rewritten) — this task relocates the betting-specific content unchanged, it does not redesign it.

- [x] **Step 1:** Re-read `src/dialogue/prompts.py::build_chatbot_system_prompt` in full (all ~367 lines) and mark the exact line ranges for the SCOPE section and each of the four example-phrase spots described above. Confirm the "Scope note" above still matches the live file — line numbers will have shifted.
- [x] **Step 2:** **STOP.** Confirm with the project owner, explicitly, in this session, that you are about to edit `src/dialogue/prompts.py` and create `src/dialogue/packs/*.py`, per the CLAUDE.md prompt-file rule above. Do not proceed on an earlier or implied approval — get a fresh yes right before this edit.
- [x] **Step 3:** Create `src/dialogue/packs/betting.py` (today's SCOPE text + example phrases, verbatim) and `src/dialogue/packs/generic.py` (the neutral equivalents) per Step 1's inventory.
- [x] **Step 4:** Modify `build_chatbot_system_prompt` to accept `prompt_pack`, resolve it via the `PACKS` dict, and substitute the pack's text into the SCOPE assembly and the four example spots — leave every other section of the function byte-for-byte unchanged.
- [x] **Step 5:** Add `prompt_pack` to `Crm` in `src/models/crm.py`; generate the Alembic migration (`alembic revision --autogenerate -m "add crm prompt_pack"`) and hand-edit it to backfill `UPDATE crms SET prompt_pack = 'betting'` for existing rows (check actual row count/ids first, don't assume).
- [x] **Step 6:** Wire `prompt_pack` through the resolution chain: `db_resolver.py` (denormalize onto `tenant.settings.prompt_pack`, mirroring `crm_id`) → `bootstrap.py::make_chatbot_factory` (read off `tenant.settings`, pass to `ChatBotAgent(...)`) → `agents/chatbot.py` (`self._prompt_pack`, passed to `build_chatbot_system_prompt`).
- [x] **Step 7:** Update/add unit tests: one asserting the betting CRM's prompt output is unchanged from before this change (byte-for-byte on the SCOPE section and the four example spots, or as close as the existing test's assertions allow), one asserting a CRM with `prompt_pack=NULL`/unset (or a tenant with no linked CRM) produces the neutral SCOPE with no gambling vocabulary, one asserting an unrecognized `prompt_pack` string falls back to generic rather than raising.
- [x] **Step 8:** Run `.venv/bin/python -m pytest tests/ -q`, confirm no new failures beyond the documented baseline.

**Status: DONE (2026-09-05).** Implemented by a Sonnet subagent, verified by two independent Opus review rounds (round 1 found 3 issues — a `PLAYER-SPECIFIC`/`ACCOUNT-SPECIFIC` label inconsistency in DATA RULE, a stray article in the generic pack's grammar, and a test-coverage gap in the "no gambling vocabulary" check — all fixed and re-verified in round 2, no third round needed). Betting-pack output confirmed byte-for-byte identical to pre-split across 32 flag/context permutations. Full suite: 1932 passed, 3 failed (documented pre-existing baseline), 5 deselected — no new failures. All changes are uncommitted in the working tree, pending the project owner's decision to commit.

**⚠️ Deploy blocker, not yet resolved:** `alembic/versions/0016_crm_prompt_pack.py` exists (adds `Crm.prompt_pack`, backfills `'betting'` for existing rows) but has **not been applied** to the live DB (`alembic current` still reports `0015_deposit_verification`). `src/auth/db_resolver.py`'s `DbTenantResolver.reload()` unconditionally selects `Crm.prompt_pack`, and that call is unguarded in `src/main.py` — deploying this code before running `alembic upgrade head` will crash tenant resolution at boot. **Apply the migration before or together with deploying this change, not after.**

**Known Low-priority residuals, left for the owner's call (none block Task 1):**
- `src/dialogue/prompts.py:608-610` (the identity/anti-leak paragraph, explicitly out of scope for Task 1) still contains the illustrative example `"Player Profile & Wallet APIs"` / `"Transactions & Bets APIs"` — meaning "Player," "Wallet," and "Bets" still appear in the generic-pack prompt output. Fixing this would mean reopening the identity paragraph, which Task 1 deliberately left alone as already industry-neutral in substance.
- `tests/unit/test_prompts.py`'s gambling-vocabulary-check comment doesn't name all the residuals above explicitly.
- No regression test pins the Fix-2 grammar text (`"heading into cancellation/downgrade territory"`) — a future edit could silently reintroduce the stray article.

---

### Task 2: Make bundled KB seeding an explicit per-CRM opt-in, not automatic-for-every-CRM

**Files:**
- Modify: `src/main.py` (`_seed_crm_kb` and its boot-time invocation)
- Modify: `src/models/crm.py` (add `bundled_kb_pack: Mapped[str | None]` column)
- New Alembic migration adding that column **plus backfill**: `UPDATE crms SET bundled_kb_pack = 'betting-default'` for existing rows.
- Rename (git mv, preserve history): `data/kb/global/` → `data/kb/packs/betting-default/`. Update every reference to the old path (`grep -rn "data/kb/global" src/ scripts/ docs/` — expect hits in `src/main.py`, `src/rag/context_builder.py::PRIORITY_ORDER`, `scripts/ingest_kb.py`'s docstring, possibly others).
- Modify: `src/rag/context_builder.py::PRIORITY_ORDER` — verify first whether this list drives ingestion or only reranking/priority of already-indexed content; update the path either way, and confirm it degrades gracefully (no crash, just lower priority / no boost) for a CRM that never seeded this pack.
- Do not touch: the tenant-level casino/sports/matka ingestion mechanism in `src/api/knowledge.py` (`_INGESTIBLE_DOCS`, the layout-ingest route) — explicitly out of scope, see Global Constraints.

**Interfaces:**
- `_seed_crm_kb`'s existing `kb_dir` parameter (default `Path("data/kb/global")`, soon `Path("data/kb/packs/betting-default")`) stays for testability, but the function's CRM-selection logic changes from "every `Crm` row" (`select(Crm.id)`) to "every `Crm` row where `bundled_kb_pack == '<pack matching this kb_dir>'`" — i.e., only CRMs that opted into that specific pack get seeded from it.
- The existing legacy-id reconciliation/pruning logic (`crm_kb_{crm_id}_*` vs stale `global_kb_*`) can stay as-is; this task does not need to also do the id-scheme cleanup mentioned in `scripts/purge_stale_kb_docs.py` unless it's trivial to include — if it adds real risk/scope, leave it as a separate followup and say so in the completion report.

- [ ] **Step 1:** Re-read `src/main.py::_seed_crm_kb` in full (function signature, doc comment, and its caller near the bottom of the file) to confirm current behavior matches this plan's description before changing anything.
- [ ] **Step 2:** Add `bundled_kb_pack` to `Crm`, generate + hand-edit the Alembic migration with the explicit backfill (check actual row count/CRM ids in the target DB before writing the backfill SQL — don't guess).
- [ ] **Step 3:** Change `_seed_crm_kb`'s CRM-selection query to filter by `bundled_kb_pack` matching the pack being seeded, instead of selecting all CRMs.
- [ ] **Step 4:** `git mv data/kb/global data/kb/packs/betting-default` and fix every reference found in Step-0 grep (`src/main.py`, `src/rag/context_builder.py`, `scripts/ingest_kb.py`, any others found).
- [ ] **Step 5:** Add/update a unit test for `_seed_crm_kb` (it already has a `kb_dir`/`auto_prune` override for testability per its docstring — use the same pattern) asserting: a CRM with `bundled_kb_pack` set gets seeded, a CRM with it unset does not.
- [ ] **Step 6:** Run `.venv/bin/python -m pytest tests/ -q`, confirm no new failures.

---

### Task 3: Move betting-specific TTS pronunciation vocabulary to per-CRM override data

**Files:**
- Modify: `src/pipeline/text_normalize.py` (trim `DEFAULT_PRONUNCIATIONS` to genuinely generic entries only; **also remove the stray `"Manoj": "मनोज"` entry** — a one-off/test artifact unrelated to CRM-scoping that has no business in any shared production default)
- Modify: `src/models/crm.py` (add `pronunciation_overrides: Mapped[dict | None]` as a JSON column)
- New Alembic migration adding that column **plus backfill**: populate the existing CRM's `pronunciation_overrides` with exactly the entries removed from `DEFAULT_PRONUNCIATIONS` in this task (Casino, Aviator, betting, Cricket, Football, Matka, Sports, Tennis, Basketball, market, match, matches, IPL — re-verify the exact list against Step 1's diff, don't transcribe from memory).
- Modify: `src/providers/tts/sarvam.py`, `src/providers/tts/indicf5.py` (thread CRM/tenant context through to `normalize_for_tts(text, extra=...)` — the `extra` parameter already exists per the module's docstring; this is wiring, not new mechanism). Check `src/interfaces/tts.py` and every other TTS call site (`grep -rn "normalize_for_tts\|apply_pronunciations" src/`) for whether CRM/tenant context is already available at that point in the call chain — if not, trace how far up the call stack it needs to be threaded and confirm that's a contained change before proceeding.

**Interfaces:**
- `normalize_for_tts(text, extra=crm.pronunciation_overrides)` at every TTS call site, where `extra` merges over (or is merged with, confirm precedence in the existing function) the trimmed generic `DEFAULT_PRONUNCIATIONS`.

- [ ] **Step 1:** Re-read `src/pipeline/text_normalize.py` in full; classify every entry in `DEFAULT_PRONUNCIATIONS` as generic-loanword vs betting-vertical. Write the split down (this becomes the backfill data for the migration in Step 3).
- [ ] **Step 2:** Trace every call site of `normalize_for_tts`/`apply_pronunciations` and confirm what context (tenant, CRM, or neither) is available there today.
- [ ] **Step 3:** Add `pronunciation_overrides` to `Crm`, generate + hand-edit the Alembic migration with the explicit backfill from Step 1's betting-specific list.
- [ ] **Step 4:** Trim `DEFAULT_PRONUNCIATIONS` in `text_normalize.py` to the generic-only entries; delete the `"Manoj"` entry.
- [ ] **Step 5:** Thread CRM context through the call sites found in Step 2 to `normalize_for_tts`'s `extra=` parameter.
- [ ] **Step 6:** Add/update a unit test asserting: the existing CRM's TTS output for a betting term (e.g. "Casino") is unchanged; a hypothetical CRM with no overrides does not get betting-term substitutions.
- [ ] **Step 7:** Run `.venv/bin/python -m pytest tests/ -q`, confirm no new failures.

---

### Task 4: Re-evaluate and rewrite documentation to match the finished architecture

**Only start this task once Tasks 1–3 are complete, tests are green, and the project owner has reviewed/accepted the code changes.**

**Files (verify this list is complete via `grep -rl "data/kb/global\|DEFAULT_PRONUNCIATIONS\|prompt_pack\|bundled_kb_pack" docs/` plus a read of each doc listed):**
- `docs/HANDOVER.md`
- `docs/ARCHITECTURE.md`
- `docs/PROJECT-STATUS.md`
- `docs/chatbot.md`
- `docs/chatbot-report.md`
- `docs/crm-api-contract.md`
- `docs/crm-chat-media-contract.md`
- `data/kb/modules/README.md`, `data/kb/layouts/README.md` (if they reference the old `global` directory or seeding behavior)
- `README.md`, `.env.example` if any new env vars/scripts changed

**Interfaces:** none — pure documentation task.

- [ ] **Step 1:** Read every file in the list above in full.
- [ ] **Step 2:** Rewrite each one to describe the system **as it now stands** — prompt packs (generic default, betting as one named pack), CRM-level opt-in KB bundling (with tenant-level casino/sports/matka module docs unchanged), CRM-level pronunciation overrides. Write these as plain, current-state facts. **Do not** add "this used to be global," "previously this was hardcoded," "note: stale," or any diff/history framing anywhere in the rewritten docs — a reader six months from now should not be able to tell this was ever any other way from the doc text itself. (Git history is where the "how it changed" story belongs, not the docs.)
- [ ] **Step 3:** Grep the whole repo for `data/kb/global` and confirm zero remaining references outside git history/this plan file itself.
- [ ] **Step 4:** Confirm no doc still asserts the RAG/ChatBot subsystem is "untouched scaffold" or similar outdated framing unrelated to this plan but caught in the same sweep — fix opportunistically if trivial, otherwise leave a separate note for the owner rather than scope-creeping this task.
