# Architecture

A single view of the system: a **multi-tenant, API-driven voice-agent platform**. A
CRM (or an operator via the browser consoles) registers a tenant, creates a campaign,
and places outbound calls; each call runs an AI agent over telephony (or the browser),
in one of two pipeline modes, and is recorded + billed to the database.

```mermaid
flowchart TB
  classDef ext   fill:#11213a,stroke:#3b5374,color:#dbe7ff;
  classDef store fill:#0b1220,stroke:#334155,color:#93c5fd;
  classDef note  fill:#1c1407,stroke:#7c5e1e,color:#fde68a;

  CRM["CRM / API consumer"]:::ext
  OPS["Operators<br/>/admin · /console · /admin/tenants"]:::ext
  TESTER["Tester<br/>/dev/voice (browser)"]:::ext
  PSTN["Lead's phone (PSTN)"]:::ext

  subgraph APP["FastAPI app · multi-tenant · deployed on Northflank (from main)"]
    direction TB
    API["REST API — /api/v1<br/>tenants · catalog (providers/models/voices + costs)<br/>campaigns · calls (Call Lead async + status)"]
    AUTH["Auth &amp; tenancy<br/>DbTenantResolver → TenantContext · bearer/admin<br/>Fernet-encrypted per-tenant telephony keys"]
    BRIDGE["Media bridges — _BaseLiveBridge<br/>browser S2S · Twilio/Exotel (S2S + cascade)<br/>Stringee IVR · SIP/DiDLogic (RTP, on a branch)"]
    AGENT["VoiceBotAgent<br/>state machine · slots · prompts · outcome analysis"]
    MODES["Two pipeline modes<br/>① Cascade: STT → LLM → TTS<br/>② S2S: Gemini Live (audio ↔ audio, ~1.4s)"]:::note
    REG["Per-tenant provider registry"]
    COST["call_store + cost catalog<br/>insert_call · record_outcome · per-min cost<br/>(telephony shown tentative, excluded from total)"]
  end

  subgraph PROV["External providers"]
    AI["AI: Gemini (LLM + Live) · Sarvam (STT/TTS)<br/>Groq · Deepgram · Anthropic — shared master keys"]:::ext
    TEL["Telephony: Twilio · Exotel · Stringee<br/>SIP trunk (DiDLogic) — per-tenant keys"]:::ext
  end

  subgraph DATA["Data"]
    PG[("Postgres — schema 'voicebot'<br/>tenants · secrets · api_keys · phone_numbers<br/>campaigns · leads · conversations · turns · events<br/>provider_costs")]:::store
    REDIS[("Redis — session store")]:::store
  end

  CRM --> API
  OPS --> API
  TESTER --> BRIDGE
  API --> AUTH
  API -->|"Call Lead → dial out"| TEL
  TEL <-->|"media: WS (Twilio/Exotel) / RTP (SIP)"| BRIDGE
  TEL --- PSTN
  BRIDGE --> AGENT
  AGENT --> MODES
  MODES --> REG
  REG --> AI
  AGENT --> COST
  AUTH --> PG
  COST --> PG
  API --> PG
  AGENT --> REDIS
```

## How a call flows (Call Lead)
1. **Register** — `POST /api/v1/tenants` (admin) stores the tenant + provider/model
   choices; telephony keys are Fernet-encrypted into `tenant_secrets`. Returns an API token.
2. **Create campaign** — `POST /api/v1/campaigns` (+ CSV leads).
3. **Call Lead** — `POST /api/v1/campaigns/{id}/calls` (async, returns `call_id`): checks the
   campaign is active + the tenant's concurrency cap, places the outbound call on the tenant's
   telephony provider, and inserts an `in_progress` `conversations` row snapshotting the config.
4. **Run** — the carrier connects back to a **media bridge** (WS for Twilio/Exotel, in-process
   RTP for SIP). The bridge runs the **VoiceBotAgent** in the tenant's mode (cascade or S2S),
   using the per-tenant **provider registry**. Slots/action/state come from the LLM JSON envelope
   (cascade) or the `record_turn_signal` tool call (S2S).
5. **Teardown** — outcome analysis runs, and `record_outcome` writes status + outcome + **cost**
   (Σ provider cost/min × duration; telephony excluded as tentative) to the `conversations` row.
6. **Poll** — `GET /api/v1/calls/{id}` returns status/outcome/cost; the **backoffice**
   (`/admin/tenants`) aggregates per-tenant analytics + billing.

## Key properties
- **Multi-tenant, DB-backed.** All state lives in Postgres under the `voicebot` schema (shared
  DB); tenants resolve from the DB via bearer token / admin. No YAML at runtime.
- **Two pipeline modes.** Cascade (STT→LLM→TTS, the controllable default, ~3.2s first word) and
  S2S (Gemini Live, ~1.4s, native barge-in). Selectable per tenant (`pipeline.mode`).
- **Key isolation.** Only telephony keys are per-tenant (encrypted at rest); STT/LLM/TTS/S2S use
  shared platform master keys.
- **Cost model.** `provider_costs` keyed `(kind, provider, model)`, per-minute; per-call cost is
  the platform components only — telephony is the tenant's own trunk, shown as a *tentative* figure.
- **Browser consoles** are thin UIs over the same `/api/v1` (so using them exercises the API):
  `/admin` (register + costs), `/console` (campaigns/calls), `/admin/tenants` (analytics + billing),
  `/dev/voice` (live voice test — recorded + billed like a real call).

See `docs/PROJECT-STATUS.md` for component-by-component status and
`docs/sip-didlogic-integration-plan.md` for the SIP path.
