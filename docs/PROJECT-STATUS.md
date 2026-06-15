# Project Status

**Last updated:** 2026-06-15

Ground-truth status of what has actually been **built, validated, and worked on** —
as opposed to what merely exists in the tree. Several modules were scaffolded during
the early PRD-phase generation and have **not** been touched since; those are listed
under "Not started" even though code for them exists.

The current focus is a **Hindi-only outbound VoiceBot** (Bharat Matka / "Anaaya"
campaign), tested through the browser dev console.

---

## At a glance

| Area | Status |
|---|---|
| Voice core: STT / LLM / TTS (cascade) | ✅ done & validated |
| **Speech-to-speech (Gemini Live)** | ✅ **selectable path, validated live (~1.4s first word)** |
| Dev console (browser) + barge-in | ✅ done — primary surface |
| Twilio / Exotel telephony (streaming) | ✅ media bridge done · barge-in ⬜ |
| Stringee telephony (turn-based IVR) | 🟡 built, live-blocked on Stringee's side · barge-in ⬜ |
| Telnyx / Infobip telephony | 🟡 auth scaffold only, media ⬜ |
| **API/DB multi-tenant platform (5 APIs)** | ✅ **built, deployed to prod** |
| **Provider cost catalog + per-call billing** | ✅ **model-level, telephony-as-tentative** |
| **Admin / Tenant / Backoffice consoles** | ✅ **built (browser UIs over the API)** |
| **Production deploy (Northflank)** | ✅ **live, auto-deploy from `main`** |
| **SIP trunk / DiDLogic (outbound)** | 🟡 built on a branch — pending live creds (not on `main`) |
| Campaign orchestration | ✅ logic done; DB-backed create/end + per-lead Call Lead |
| **Telephony barge-in** | ⬜ **pending across all telephony** (S2S path has native barge-in) |
| **RAG / ChatBot** | ⬜ **untouched scaffold — not worked on** |
| **Benchmarking** | ⬜ **very basic skeleton — much more to do** |
| **Code-switching / multilingual** | ⬜ **not considered — Hindi-only today** |

---

## 🚀 Since 2026-06-10 — Speech-to-speech + the API/DB platform

The work since the last report shifted from *latency experiments on the cascade* to
two big additions: a **speech-to-speech path** that breaks the latency floor, and a
**purely API-driven, DB-backed multi-tenant platform** (now deployed to production).

### Speech-to-speech (Gemini Live)
- **Selectable path beside the cascade** (`pipeline.mode: layered | s2s`): `GeminiLiveBridge`
  (browser) and `TelephonyLiveBridge` (Twilio/Exotel media streams) speak end-to-end audio
  in/out. Cascade stays the default + fallback. Validated live (multi-turn Hindi).
- **Latency: ~1.4 s to first word** on `gemini-3.1-flash-live-preview` (1389 ms median, real-time
  input), vs ~3.2 s for the cascade — Experiment 6 in the latency log. Brings **native barge-in /
  turn-taking** for free. (Reply *playback* still runs long, ~7–12 s — a brevity-tuning follow-up.)
- **Structured output / state in S2S:** Gemini Live emits **free-form audio, not JSON**. Control
  comes from a **function/tool call** the model makes while speaking —
  `record_turn_signal(action, updated_slots)` where `action ∈ {continue, clarify, transfer,
  schedule_callback, send_info, close_positive, close_negative, end}`. So the cascade's JSON
  envelope (`response_text` etc.) is **not** returned, but **slots, action, state machine, and
  end-of-call outcome are preserved** — `apply_signal` (extracted from the cascade) is driven by
  the tool call, and Live transcriptions feed the same outcome analysis.
- Gotchas found: `session.receive()` is a **per-turn** generator (must loop), and a **text
  kickoff breaks realtime-audio VAD** (v1 = user speaks first). Realtime-audio tokens are
  pricier than text — cost is tracked per call (below).

### API/DB-backed multi-tenant platform
Replaced YAML-on-boot + in-memory state with real DB tables and a clean API surface. **The 5 APIs**:
1. **Register Tenant** — `POST /api/v1/tenants` (admin): provider/model choices + Fernet-encrypted
   telephony keys; returns the tenant's API token once.
2. **Get Voice List** — `GET /api/v1/voices` (+ `GET /api/v1/providers` costs, `GET /api/v1/models`
   variants): the selectable rosters.
3. **Create Campaign** — `POST /api/v1/campaigns` (+ CSV lead upload).
4. **Call Lead** — `POST /api/v1/campaigns/{id}/calls` (async, 202): outbound call for a lead,
   concurrency-capped; returns `call_id`. `GET /api/v1/calls/{id}` polls status/outcome/cost.
5. **End Campaign** — `POST /api/v1/campaigns/{id}/end`.
   *(Admin backoffice: `GET /tenants`, `/tenants/{id}/analytics`, `/tenants/{id}/billing`.)*

- **Schema namespacing:** all tables live under a configurable schema (**`voicebot`**) so we share
  the existing Postgres DB; managed-Postgres URL normalization (`postgresql://…?sslmode=require`
  → async + ssl).
- **Per-tenant telephony keys encrypted at rest** (Fernet, `VOX_SECRET_KEY`); STT/LLM/TTS/S2S use
  shared master keys.
- **Provider cost catalog** — table `provider_costs` keyed `(kind, provider, model)`, **per-minute**
  (USD/min), **model-level** for STT/LLM/TTS/S2S (e.g. gemini flash vs flash-lite vs pro),
  per-provider for telephony. LLM/STT are token-priced upstream — modeled per-minute from typical
  dialogue volume. **Per-call cost** = Σ(components × duration); **telephony excluded from the
  platform total** (tenant's own trunk), shown as a *tentative* figure.
- **Browser consoles** (thin UIs over the API, so using them exercises the API): `/admin` (register
  with a live cost breakdown + cost editor), `/console` (campaigns/calls), `/admin/tenants`
  (backoffice: tenant list + analytics + billing). **Webconsole/browser test calls are recorded +
  billed** like real calls.
- **Deployed to production** (Northflank, auto-deploy from `main`) — live and healthy on the
  `voicebot` schema.

### In flight (NOT on `main`)
- **SIP trunk / DiDLogic outbound** — pure-Python (pyVoIP) in-app transport, S2S over RTP,
  implemented + unit-tested on `feature/sip-didlogic-outbound`; **pending live DiDLogic creds**
  before merge (see `docs/sip-didlogic-integration-plan.md`).

---

## ✅ Fully implemented (built & validated)

**Voice core**
- **STT:** Deepgram streaming (`nova-2 hi`, active dev-console path) + Groq Whisper batch fallback. Tuned and validated.
- **LLM:** Gemini 2.5-flash (active, with transient-error retry hardening) + Anthropic Claude (Haiku 4.5) as a tested one-line swap-in.
- **TTS:** Sarvam (`bulbul:v2`, voice `anushka`) — the only voice; batch + sentence-overlapped streaming.

**Dev console (browser) — the primary working surface**
- Full streaming pipeline (live Deepgram endpointing → Gemini token stream → overlapped Sarvam TTS).
- **Server-side barge-in** shipped (PR #15): sustained-interim detection, headphones-required, behind the "Allow interruptions" toggle. Validated live.
- Post-call outcome analysis wired on this path.

**Telephony — streaming bridges**
- **Twilio + Exotel** media-stream bridges built, wired via `bootstrap.py`, unit-tested. (Turn detection here is batch Silero VAD, not streaming endpointing.)

**Campaign orchestration — logic**
- Scheduler, concurrency cap, rate-limiting, calling-hours gate, retry/backoff, DND filtering, CRM/event-bus hooks. Tested.

---

## 🟡 Partially done

- **Stringee telephony** — turn-based IVR (no media streaming; record → webhook → batch turn → reply WAV) fully built and unit-tested, but **live calls fail on Stringee's side**; never completed end-to-end. Parked pending a Stringee fix.
- **Telnyx + Infobip** — adapters provide auth/JWT scaffolding only; `stream_audio_in`/`stream_audio_out` raise `NotImplementedError`. No media bridge.
- **Campaign → live calling** — the orchestration engine is done, but the live dispatch to a real telephony provider and per-call outcome recording on the campaign path are **not wired/validated** (only the dev console is wired for outcome analysis).

---

## ⬜ Not started / not touched

- **Telephony barge-in — PENDING for all telephony.** Barge-in is **dev-console only**. Twilio/Exotel streaming barge isn't built (their `handle_turn` has no `cancel_event` yet); Stringee has only the coarse SCCO `bargeIn` flag, not real detection. Documented fast-follow, not started.
- **RAG / ChatBot — untouched scaffold.** `src/rag/*`, `src/agents/chatbot.py`, `src/api/chat.py`, `src/api/knowledge.py`, and the FAISS vector store exist as early-generation scaffold but have **not** been worked on, wired to an active tenant, or validated. Not part of current work.
- **Benchmarking — very basic.** `src/benchmarks/*` is an early skeleton; substantial work is still required before it's a usable harness. Not "done."
- **Code-switching / multilingual — not considered.** The system is **Hindi-only** today. The "write `response_text` in Devanagari only" prompt rule merely makes the single-language Hindi path work with the Hindi TTS — it is **not** a multilingual or code-switch feature. No transliteration engine, no second language.
- **Other:** second/fallback TTS provider; multi-instance scale for the Stringee call registry (currently in-memory, single-instance by design).

---

## Known dialogue-quality items (future work)

- **CTA repetition.** The agent over-repeats its call-to-action — nearly every turn
  ends with the "WhatsApp link + 10% first-deposit bonus" push (most visible on
  `send_info` turns). This is a **prompt-tuning issue, not architectural**: the
  `bharat_matka` campaign over-weights that CTA and the system prompt nudges toward
  the objective every turn with no "don't repeat a CTA already made" guard. Fix is a
  campaign-script tweak plus a one-line prompt rule — logged as a dialogue-quality
  improvement, not a blocker.

## Notes on latency (context)

The dominant per-turn cost in the **cascade** is **LLM + TTS inference**, not STT or server
placement (see `docs/latency-llm-stt-experiments.md`). Cascade live medians (dev console, local):
LLM TTFT ~2.1 s, first spoken word ~3.3 s, full turn ~6.4 s. STT is effectively free on the
streaming path (overlapped with speech).

**Speech-to-speech removes that floor** (Experiment 6): `gemini-3.1-flash-live-preview` reaches
**~1.4 s to first word** (1389 ms median) vs ~3.2 s for the cascade — end-to-end audio, no
serialized STT→LLM→TTS, with native barge-in. S2S is the architectural answer; the cascade
remains the cheaper, more controllable default.
