# Handover

**Written:** 2026-09-04. This is the current source of truth for handing this project to
a new owner. `docs/PROJECT-STATUS.md` and `docs/ARCHITECTURE.md` (last updated
2026-06-15/16) are **stale on several major points** — see [What's stale in the old
docs](#whats-stale-in-the-old-docs) before trusting them. `docs/chatbot.md` and
`docs/crm-api-contract.md` (Aug 2026) are current and worth reading directly.

## What this is

`vox-agent` is a multi-tenant AI platform for an online gaming/betting operator
(casino / sports betting / matka-lottery verticals), with two products sharing one
backend:

1. **VoiceBot** — outbound AI phone calls (originally the project's starting point).
2. **ChatBot** — a customer-support chat widget backed by the operator's live CRM APIs
   (account/wallet/KYC/withdrawal questions, deposit-dispute handling, escalation to
   human agents). **This is now the primary active workstream** — most commits in the
   last two months are here, not in voice.

Both run as one FastAPI app, multi-tenant, DB-backed (Postgres schema `voicebot`,
shared DB), deployed to Northflank with auto-deploy from git.

## At a glance (corrected, 2026-09-04)

| Area | Status |
|---|---|
| VoiceBot core (STT/LLM/TTS cascade + Gemini Live S2S) | done & validated |
| Twilio / Exotel telephony (cascade + S2S) | production; **S2S has native barge-in, cascade still doesn't** |
| **LiveKit (CRM-hosted SIP → WebRTC room join)** | **flagship integration, live-verified — not in old docs at all** |
| Stringee telephony (turn-based IVR) | built, still blocked live-side by Stringee (external, unresolved) |
| Telnyx / Infobip | **removed entirely** (old docs call it "scaffold" — it's gone) |
| SIP / DiDLogic (pyVoIP) | parked on a branch, not merged, superseded by LiveKit |
| **ChatBot + CRM tool-calling** | **mature, primary active area** — old docs call this "untouched scaffold," which is false |
| **RAG / Knowledge base** | **fully wired**: FAISS+BM25 hybrid, per-tenant + CRM-shared KB — old docs call this scaffold too |
| **Deposit verification** (screenshot-based dispute resolution) | new, actively evolving, two vendor webhook contracts — treat as unstable |
| Multi-tenant platform (5 core APIs, admin/console UIs) | live in production |
| **Security posture** | **remediation sprint just completed** (2026-09-01/02) — read the section below before assuming anything is safe |
| Tests | 1927 passing, 3 failing (1 known, 2 environmental — see [Testing](#testing)) |
| Campaign → live outbound calling | orchestration logic done; live dispatch/outcome wiring not fully validated |
| Benchmarking harness | still a basic skeleton |
| Code-switching / multilingual | not implemented — Hindi-only on voice |

## Architecture

The diagram in `docs/ARCHITECTURE.md` is still structurally correct for the *voice*
side but is missing LiveKit and says nothing about the chat/CRM side, which is now
comparably large. Read both `docs/ARCHITECTURE.md` (voice/platform shape) and
`docs/chatbot.md` (chat/CRM shape) together.

One FastAPI app (`src/main.py`), two tenant-facing surfaces:

- **Voice**: `POST /api/v1/campaigns/{id}/calls` places an outbound call; a carrier
  (Twilio/Exotel media-stream WS, Stringee turn-based webhook, or LiveKit room join)
  connects back to a media bridge; the bridge runs `VoiceBotAgent` in one of two
  pipeline modes — **cascade** (STT→LLM→TTS, `src/pipeline/engine.py`) or **S2S**
  (Gemini Live, audio-to-audio, native barge-in, shared across browser/Twilio/Exotel/
  LiveKit via `src/api/live_bridge_base.py::_BaseLiveBridge`).
- **Chat**: `POST /api/v1/chat/sessions` (tenant bearer) then `WS /chat/ws/{session_id}`
  (the session_id itself is the capability — no further socket auth). `ChatBotAgent`
  (`src/agents/chatbot.py`) runs an agentic tool-calling loop against per-tenant CRM
  tools, a KB search tool, and escalation/voice-handoff tools. A Chatwoot inbound
  webhook path also exists.

Both sides write to Postgres (`conversations`/`turns` for voice, `chat_sessions`/
`chat_messages` for chat) and are billed per-call/per-token against a
`provider_costs` catalog.

## Subsystem notes

### Voice / Telephony
- **Twilio + Exotel**: production, both pipeline modes. Cascade mode still has no
  barge-in (`handle_turn` isn't passed a `cancel_event`); S2S mode gets barge-in for
  free via `_BaseLiveBridge`.
- **LiveKit** (`src/api/livekit_bridge.py`, `livekit_runner.py`, `livekit_routes.py`):
  new since June, live-verified against a real LiveKit server. Model: the CRM fronts
  its own PSTN/SIP via LiveKit SIP, drops the call into a room, this app joins as a
  WebRTC participant — zero direct SIP/RTP handling here. No inbound HTTP leg; the
  only trigger is a `participant_joined` webhook. Read
  `docs/livekit-sip-integration-plan.md` and `docs/integrations/livekit-room-handoff.md`.
- **Stringee**: fully built turn-based IVR, but live calls still fail on Stringee's
  side — open issue with their support (`docs/stringee-support-writeup.md`), not
  something fixable from this codebase alone.
- **Telnyx/Infobip**: deleted (see `docs/superpowers/plans/2026-07-28-remove-telnyx-infobip.md`).
- **SIP/DiDLogic**: parked, not merged, deprioritized in favor of LiveKit.
- Key files: `src/api/live_bridge_base.py`, `src/api/livekit_bridge.py`,
  `src/api/telephony_twilio.py`/`telephony_exotel.py`, `src/pipeline/engine.py`.

### ChatBot / CRM
- Read `docs/chatbot.md` first — it's accurate and dense.
- `ChatBotAgent` runs plain-text tool-calling (Gemini can't do JSON+tools at once),
  max 2 tool rounds per turn, calling `search_knowledge_base`, `escalate_to_human`,
  `offer_voice_call`, or per-tenant CRM tools (tenant's own tools always win over the
  shared CRM catalog).
- A code-level hallucination guard prevents the bot from inventing player-specific
  data (balances, tx IDs) — not just a prompt instruction.
- **RAG/KB is real and wired**: FAISS+BM25 hybrid retrieval, per-tenant index, plus a
  CRM-level shared KB (`crm_kb_documents`) merged in at query time, plus opt-in
  "product module" KB layers (casino/sports/matka).
- Production tenants are DB-backed; `config/tenants/*.yaml` (`dev.yaml`,
  `example.yaml`) is legacy/dev-console-only fallback, not the live path.
- No open TODO/FIXME/NotImplementedError found in this subsystem — unusually clean,
  and test coverage is extensive (~35 unit + 2 integration files).
- Key files: `src/agents/chatbot.py`, `src/api/chat.py`, `src/api/chat_webhooks.py`,
  `src/rag/context_builder.py`, `docs/crm-api-contract.md`.

### Deposit verification
New and still moving (commits through 2026-09-03). Handles disputed failed-deposit
screenshots via **two independent, incompatible vendor webhook contracts**:
- "multipart_verdict" — synchronous, HMAC-verified, terminal callback.
- "json_ticket_relay" — capability-token-in-URL, multiple non-terminal progress
  messages, sliding timeout window.
Treat this feature as unstable/in-progress, not a settled contract. Files:
`src/api/deposit_verification.py`, `src/chatbot/deposit_verification.py`.

### Auth, security, multi-tenancy — read this before touching anything
The last six commits on `stage` (2026-09-01/02) are a genuine **security remediation
sprint**, not routine hardening. What they closed:
- Dev/bridge consoles were reachable without per-request auth → now admin-gated.
- A Twilio/Stringee recording-download webhook could be pointed at an attacker URL
  (SSRF) and leaked Twilio credentials → fixed with host allowlisting + SID binding.
- A call-transfer endpoint let one tenant hijack another tenant's in-flight transfer
  → transfer state now re-keyed by `(tenant_id, call_sid)`.
- Raw CRM tool responses (mobile/email/KYC/bank details) and Chatwoot payloads were
  logged **unredacted at INFO in production** → replaced with one-way fingerprinting.
- Chatwoot's webhook resolved the tenant from an enumerable `inbox_id` with zero auth
  → moved to unguessable per-tenant capability token + optional HMAC.
- `src/auth/middleware.py` had **zero log calls** — auth rejections and brute-force
  attempts left no trace → logging added (rate-limited to avoid log-flood).

All five are marked fixed+tested in their commit messages, but **verify independently
before treating them as closed** — this is exactly the kind of thing a departing team
says is fixed and a new owner should re-check.

**Known remaining gap**: webhook signature-verification infra
(`src/auth/webhook_auth.py`) exists but is **not wired into any live route yet** —
`VOX_WEBHOOK_SIGNATURE_MODE` defaults to `"enforce"` but nothing currently calls it.

Auth model: tenant bearer tokens (SHA-256 hashed, DB-backed), separate admin tokens
(`VOX_ADMIN_TOKENS`, labeled per-admin), and per-tenant unguessable capability tokens
for inbound webhooks. Only telephony credentials are Fernet-encrypted at rest
(`VOX_SECRET_KEY` — treat as a master credential, back it up, losing it orphans every
stored tenant secret). STT/LLM/TTS/S2S keys are shared platform-wide env vars, not
per-tenant.

Key files: `src/auth/middleware.py`, `src/auth/db_resolver.py`, `src/auth/secrets.py`,
`src/auth/webhook_auth.py`, `src/auth/audit.py`. The five security commit messages
(`git log --grep=security -i`) are unusually detailed and effectively double as a
mini security changelog — read them.

### Data & infrastructure
- **Deploy**: Docker image, Northflank, auto-deploy from git (confirm which branch —
  docs say `main`, but active work happens on `stage`; check Northflank's actual
  tracking branch before assuming). `/health` always returns 200 so the app boots
  without Redis/Postgres present (used for a no-addon dev-console-only stage before
  full production). Alembic runs on container start with a 60s non-blocking timeout —
  for multi-replica deploys, run migrations as a separate pre-deploy job instead
  (Alembic has no cross-process lock).
- **Datastores**: Postgres under schema `voicebot` (shared DB, `VOX_DB_SCHEMA`
  configurable), Redis for session storage.
- **Schema** (16 migrations, actively maintained): tenants/secrets/api-keys/phone
  numbers; `provider_costs`; campaigns/leads; conversations/turns/events (voice);
  chat_sessions/chat_messages/chat_tools; crms/crm_tools/crm_secrets/crm_kb_documents
  (CRM-partner-level, shared across tenants under that CRM); deposit_verification_requests;
  turn_metrics; benchmark_runs/kb_documents (early scaffold).
- **No CI** — no `.github/workflows`. Only automation is Northflank's git-triggered
  build/deploy. Worth flagging: nothing currently blocks a broken commit from
  auto-deploying.
- **Ops scripts** (`scripts/`): DB/session diagnostics, provider/secret checks,
  one-off data fixes, smoke-test callers per provider, KB ingestion, benchmark runner.
  `tools/mock_crm.py` is a mock CRM server for local dev.
- **Static browser UIs** (`static/`): `/admin` (tenant registration+costs), `/console`
  (campaigns/calls), `/admin/tenants` (backoffice: analytics/billing, human-agent
  handover UI), `/dev/voice` (live voice testing), embeddable chat widget.

### Testing
Current run: **1927 passed, 3 failed**, 58s.
1. `test_chat_routes.py::test_claim_session_and_agent_ws` — pre-existing, documented
   in `CLAUDE.md`, unrelated to most work. Still failing, unchanged.
2. `test_dev_console.py::test_place_call_passes_tenant_creds_to_adapter` — **new**,
   not previously known. Looks like real `.env` Stringee credentials are leaking into
   a test that expects a monkeypatched fake SID — a precedence bug between
   monkeypatched env and already-loaded settings/cache. Worth investigating before
   handover; not yet diagnosed to root cause.
3. `test_pgvector_crm_scoping.py::test_crm_scoped_chunk_is_isolated_from_tenant_scoped_chunk`
   — `pgvector` is declared in `pyproject.toml` but not installed in `.venv` (stale
   venv, not a code bug). Fix: `pip install -e ".[dev]"`.

`tests/unit` (158 files) is fully mocked. `tests/integration` (6 files) runs by
default against fixtures/mocks. `tests/live` (1 file) hits real provider APIs and is
excluded by default (`-m 'not live'`); opt in with `VOX_LIVE_TESTS=1`.

Verify command: `.venv/bin/python -m pytest tests/ -q`

## What's stale in the old docs

`docs/PROJECT-STATUS.md` and `docs/ARCHITECTURE.md` (2026-06-15/16) are wrong or
missing on:
- Calling RAG/ChatBot "untouched scaffold, not part of current work" — it's the
  opposite: the primary active area, with 170+ commits since.
- Calling Telnyx/Infobip "auth scaffold only" — they've been deleted entirely.
- Saying telephony barge-in is "pending across all telephony" — true for cascade
  mode, but S2S transports (browser, Twilio/Exotel S2S, LiveKit) have had native
  barge-in for a while.
- Not mentioning LiveKit at all — it's now the flagship telephony integration.
- Not mentioning deposit verification, the security remediation, or the CRM/KB
  layers at all.

Recommend either updating those two files or retiring them in favor of this one —
your call; I didn't overwrite them since they're not simply wrong, they're a
snapshot of a real prior state that may still have historical value.

## Suggested reading order for a new owner

1. This file.
2. `docs/chatbot.md` and `docs/crm-api-contract.md` (current, chat/CRM shape).
3. `docs/ARCHITECTURE.md` (voice/platform shape — mentally patch in LiveKit).
4. The 6 security commits: `git log --grep=security -i -p` (or just the messages).
5. `src/main.py` and `src/bootstrap.py` (how everything wires together).
6. Run `pytest tests/ -q` and `docker compose up -d` + `alembic upgrade head` +
   `uvicorn src.main:app --reload` locally per the README quick-start.
