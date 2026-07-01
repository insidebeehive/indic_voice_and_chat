# Coordination Service — Integration Interface

**Audience:** CRM Architecture team + AI Platform team  
**Status:** Proposed — CS is conceptual; this document defines the interface so both sides can build independently  

---

## Approach

The Coordination Service (CS) is the **channel and coordination layer** between CRM Frontend (customers) and the AI Platform. Its Phase 1 deliverable is **chat relay**: session creation, WebSocket relay, media proxying, webhook forwarding, and human agent console proxy. Voice channel adapters (Twilio, SIP, WebRTC) are a Phase 2 skeleton.

This means:
- CRM Frontend never calls AI Platform directly — all chat traffic goes through CS.
- CRM Backend (CSS) never calls AI Platform directly — CS is the bridge for all AI Platform interactions.
- For direct-human sessions (`operator_flag=human`), CSS handles its own WebSocket; CS is not in the path.
- No disruption to current production integrations — CS is validated in staging before any cutover.
- Rollback is trivial — switch the tenant config back to direct CRM Backend → AI Platform integration.

---

## Roles

```
Customer (any channel)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│             Coordination Service  (CRM team)                │
│                                                             │
│  Chat relay (Phase 1):                                      │
│    Session create proxy (CSS → CS → AI Platform)            │
│    WebSocket relay (bidirectional, with media rewrite)      │
│    Webhook forwarder (AI Platform → CS → CSS)               │
│    Human agent console proxy (claim + agent-ws)             │
│                                                             │
│  Session router:                                            │
│    operator_flag = ai/hybrid  ────────────────────────────► │──┐
│    operator_flag = human  ──► CSS owns WS directly          │  │
│                               (CS not involved)             │  │
│                                                             │  │
│  Voice channel adapters (Phase 2 — skeleton only):          │  │
│    Intercepts call_offer (pstn transport)                    │  │
│    Dials out via provider; hands off to AI Platform          │  │
└────────────────────────────────────────────────────────────┘  │
        │                                     ┌───────────────────┘
        │ webhooks + escalation events         ▼
        │                         ┌─────────────────────────────┐
        ▼                         │       AI Platform (us)       │
┌──────────────────────┐          │                              │
│   CSS (CRM Backend)  │◄─────────│  webhooks (lifecycle events) │
│                      │          │                              │
│  Webhooks receiver   │          │  STT → LLM → TTS             │
│  Support agent console──────────│  RAG / knowledge base        │
│  Ticket system       │  claim + │  CRM tool integration        │
│  Analytics           │ agent-ws │  Escalation decision         │
│  Business logic      │ (via CS) │  Media storage (S3)          │
└──────────────────────┘          │  Summarization API           │
                                  └─────────────────────────────┘
```

**CS owns:** channel relay, webhook forwarding, session routing, voice channel adapters (Phase 2)  
**We own:** AI intelligence, speech processing, media storage, summarization  
**CSS owns:** webhooks receiver, support agent console, ticket system, analytics, business logic

---

## What Changes, What Stays

| Concern | Today | With CS |
|---|---|---|
| Chat WebSocket relay | CSS | CS (CSS retains webhooks + agent console) |
| Chat session creation (API call) | CSS | CS on CSS's behalf |
| Human agent console (claim + agent-ws) | CSS calls AI Platform directly | CSS calls CS; CS proxies to AI Platform |
| Webhook events receiver | CSS | stays with CSS (CS forwards our events) |
| Direct-human chat WS | CSS | stays with CSS (CSS-owned; CS not involved for `operator_flag=human`) |
| Outbound pstn dialing | us (Call Lead API) | CS dials, uses our answer URL — we handle audio |
| Pstn audio bridge | us | stays with us (CS is NOT in the audio path) |
| STT (Sarvam, Deepgram, Gemini) | us | stays with us |
| LLM (Gemini) | us | stays with us |
| TTS | us | stays with us |
| RAG / knowledge base | us | stays with us |
| CRM tool integration | us | stays with us |
| Escalation decision | us | stays with us |
| Media storage (S3) | us | stays with us |
| Chat transcript | DB + webhook | unchanged |

---

## Inbound Scenarios

**Entry point is always chat.** The customer opens the chat widget. The conversation can stay in chat, escalate to a human agent within chat, or transition to a voice call from within chat.

### 1. Customer → AI (chat only)

```
Customer opens chat widget
        │
        ▼
CS (chat relay) ──turns──► Us
                           STT/LLM/TTS for audio clips
                           Text responses back to CS
                           CS forwards to customer
        │
   session ends
        │
Full text transcript available in:
  - Our DB (GET /api/v1/chat/sessions/{id})
  - session_closed webhook payload (forwarded by CS to CSS)

No summarization call needed — transcript is already structured text.
```

### 2. Customer → AI → Human (chat handover)

```
Customer chats with AI via CS relay
AI decides to escalate
        │
We fire escalation_requested webhook ──► CS
CS does two things in parallel:
  1. Forwards the escalation WS frame to CRM Frontend over the relay
  2. Forwards escalation_requested webhook to CSS (with event_id + field rewrites)
CSS support agent claims session via CS (POST /chat/sessions/{cs_id}/claim)
CSS support agent connects to CS agent-ws (WS /chat/agent-ws/{cs_id})
CS proxies both claim and agent-ws to AI Platform
        │
Customer continues on the same CS relay connection (unchanged)
Support agent messages flow: AI Platform → CS relay → CRM Frontend
Customer messages flow: CRM Frontend → CS relay → AI Platform → CS agent-ws → agent console
        │
   session ends
        │
Full text transcript (AI portion + human agent portion) available in:
  - Our DB
  - session_closed webhook (CS forwards to CSS)

CSS never calls AI Platform directly — all claim and agent-ws traffic goes through CS.
No summarization call needed.
```

### 3. Customer → AI → Voice call

```
Customer chats with AI via CS
Customer requests a voice call  OR  AI sends call_offer
        │
AI Platform sends call_offer frame over the chat relay WS:
  {
    "type": "call_offer",
    "transport": "websocket | webrtc | pstn",
    "call_url": "wss://...",
    "ice_servers": [{"urls": "stun:..."}],   // webrtc only
    "to": "+919876543210"                    // pstn only
  }
        │
CS handles based on transport:

  transport = websocket:
    CS forwards call_offer as-is to CRM Frontend (widget)
    Widget POSTs to CSS's /api/chat/call to get an ephemeral voice URL
    CSS calls AI Platform server-side to obtain the URL
    Widget connects to that URL — CS is not in the audio path

  transport = webrtc:
    CS forwards call_offer as-is to CRM Frontend (widget)
    Widget connects directly to call_url as WebRTC signalling endpoint
    Widget uses ice_servers for ICE negotiation — CS is not in the audio path
        │
  transport = pstn:
    CS intercepts the frame (does NOT forward to CRM Frontend)
    CS sends CRM Frontend: {"type":"mode_change","mode":"voice_pending","message":"Calling you now…"}
    CS pre-registers the call:
      POST /api/v1/telephony/register-call  { provider, provider_call_sid, lead_id }
      → receives call_id; call.initiated fires → CS webhook
    CS dials the customer via the telephony provider, setting our answer URL:
      https://{host}/api/v1/telephony/{provider}/voice/{tenant_slug}
    Customer answers → our AI bridge takes over automatically
      call.answered fires → CS webhook
        │
   call ends:
        │
    call.completed (outcome + summary) fires → CS webhook → CSS
    CS sends CRM Frontend: {"type":"ended"}
```

CS is NOT in the audio path for pstn calls — we handle the media stream directly
once the provider fires our answer URL. CS just dials and waits for webhooks.

Chat transcript (before the voice call) is already in our DB.  
Voice recording — if any — is an optional concern for the CRM team.  
Summarization is CSS's choice, not mandatory.

### 4. Customer → Human (direct, operator_flag = human)

```
CSS receives POST /api/chat/start from CRM Frontend
CSS determines operator_flag = human
CSS returns ws_url pointing to its own WebSocket (wss://css.example.com/api/chat/ws/{ticket_id})
Customer connects to CSS's WS directly
CS is not involved at all.
```

CSS owns the direct-human chat path end-to-end. CS returns `{"handled_by": "crm"}` when `operator_flag=human` and takes no further action.

---

## Outbound Scenarios

### 1. AI voice call — transfer to human agent

When the AI decides to transfer a live voice call to a human, we pause the AI
and wait for CS to find one. **The telephony call stays active** — the caller
hears silence while CS looks.

```
AI is on a live voice call (S2S or cascade bridge)
AI decides to escalate → fires 'transfer' action
        │
We:
  1. Close the Gemini Live session (AI goes silent)
  2. POST call.transfer_requested to events_webhook_url:
       {
         "event": "call.transfer_requested",
         "call_sid": "CAxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
         "transfer_result_url": "https://{host}/api/v1/calls/{call_sid}/transfer-result"
       }
  3. Wait up to 30 s for CS to call back
        │
CS receives the webhook and tries to find an available agent.
        │
  ┌─── human found? ─────────────────────────────────────────────────────┐
  │                                                                      │
  Yes                                                                   No (or timeout)
  │                                                                      │
CS POSTs:                                                            CS POSTs:
  POST {transfer_result_url}                                           POST {transfer_result_url}
  Authorization: Bearer <tenant-token>                                 Authorization: Bearer <tenant-token>
  {"status": "success"}                                                {"status": "failure"}
        │                                                                      │
We drop the Twilio WebSocket                                          We synthesize a TTS apology
(call ends on our side).                                              and play it to the caller,
CS connects the caller to the                                         then drop the WebSocket.
human agent through its own                                           (call.completed fires with
telephony infrastructure.                                              outcome = "escalated")
        │                                                                      │
        └──────────────────────────────┬───────────────────────────────────────┘
                                       │
                          call.completed fires → CS webhook
```

**CS does not need to be in the audio path.** CS only needs to:
1. Receive the `call.transfer_requested` webhook.
2. POST `{"status": "success" | "failure"}` to `transfer_result_url` within **30 s**.
3. On success: connect the caller to a human via CS's own telephony infrastructure
   (SIP transfer, warm transfer, etc.) once the call ends on our side.

> If CS does not call back within 30 s, the platform treats it as `"failure"` —
> plays the apology and ends the call automatically.

See `tenant-call-events.md` → `call.transfer_requested` for the full event payload.

---

### 3. AI → Customer (AI-initiated voice via chat session)

```
AI is in an active chat session via CS relay
AI decides to call the customer back
        │
AI Platform sends call_offer with transport=pstn over the relay WS
CS intercepts; pre-registers the call:
  POST /api/v1/telephony/register-call  { provider, provider_call_sid, lead_id }
  → call.initiated fires → CS webhook
CS dials the customer via the telephony provider, setting our answer URL:
  https://{host}/api/v1/telephony/{provider}/voice/{tenant_slug}
Customer answers → our AI bridge takes over automatically
  call.answered fires → CS webhook
        │
   call ends
        │
call.completed (outcome + summary) fires → CS webhook → CSS
```

CS is NOT in the audio path — we handle media streaming directly.
We generate the transcript because we ran the full conversation — CSS receives
the outcome and summary via the `call.completed` webhook.

### 4. CS triggers an outbound AI voicebot call (campaign dial)

The simplest outbound pattern — CS requests the call and we handle dialing,
audio, and outcome. CS does not need to integrate with the telephony provider
at all.

```
CS decides to call a lead (campaign dial, follow-up, etc.)
        │
CS calls our campaign call endpoint:

  POST /api/v1/campaigns/{campaign_id}/calls
  Authorization: Bearer <tenant-token>
  Content-Type: application/json

  {
    "to_number": "+91XXXXXXXXXX",
    "voice": "Leda",           // S2S / TTS voice; agent gender auto-derived
    "caller_name": "Priya",    // {agent_name} token in the script
    "lead_name": "Rahul",      // optional — passed to LLM as lead context
    "lead_gender": "male"      // optional: "male" | "female" | ""
  }

Response 202:
  {
    "call_id": "call_4a3f2b1c8d9e0f1a",
    "status": "in_progress",
    "provider_call_sid": "CAxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  }
        │
We dial the lead via the tenant's telephony provider.
Lead answers → our AI bridge starts automatically.
  call.answered fires → CS webhook
        │
   call ends
        │
  call.completed (outcome + summary) fires → CS webhook → CSS
```

**CS does not need a telephony integration for this pattern.** We own the
dial, the audio bridge, and the outcome. CS only needs to make one API call
and receive one webhook.

**campaign_id** maps to the script, slots, knowledge base, and LLM behaviour
configured for that campaign in our system.

**voice + caller_name** override the campaign's default agent persona for this
specific call. Gender is auto-derived from the voice ID and drives grammatical
agreement in the script (e.g. "बात कर रही हूं" vs "बात कर रहा हूं").

**lead_name / lead_gender** are optional context passed to the LLM. If
lead_gender is omitted or unknown, the agent uses gender-neutral address.
The LLM also infers gender on the fly from the caller's speech (verb forms)
and adjusts mid-call.

---

### 5. Human → Customer (support agent-initiated call)

```
Support agent clicks "Call customer" in the agent console
        │
CSS sends call_offer{transport: webrtc | websocket} over the direct-human WS
  (/api/chat/ws/{ticket_id} — CSS's own WS)
Widget receives call_offer and initiates the voice connection directly
CS is NOT in this path — direct-human sessions bypass CS entirely
We are not involved during the call
        │
   call ends
        │
Optionally: CSS calls our summarization API with the recording
```

> `pstn` transport is not available for direct-human sessions — CS's voice channel adapter is not in the path. For pstn support in direct-human sessions, CSS would need its own voice channel integration (deferred).

---

## What CS Must Provide Us

For the CS integration to work, CS must implement the following towards AI Platform:

### Chat WebSocket relay

CS must implement our existing chat WS protocol — the same protocol currently implemented by CRM Backend (documented in `chat-widget-backend-integration.md`). CS calls our session creation API, relays frames bidirectionally between CRM Frontend and our WS, and rewrites media URLs before forwarding to CRM Frontend.

CS must also:
- Call `POST /api/v1/chat/sessions` with a tenant Bearer token to create sessions.
- Forward our lifecycle webhook events (`session_started`, `escalation_requested`, `session_closed`) to CSS's configured webhook endpoint. Before forwarding, CS adds `event_id` (UUID) and rewrites `session_id` to the `cs_session_id` form, and rewrites `claim_url` / `agent_ws_url` from AI Platform paths to CS paths (see CS PRD §1.4).
- Proxy the human agent console path: `POST /chat/sessions/{cs_id}/claim` → proxied to `POST /api/v1/chat/sessions/{platform_id}/claim`; `WS /chat/agent-ws/{cs_id}` → proxied to `WS /api/v1/chat/sessions/{platform_id}/agent-ws`. CSS calls these CS-form paths; CS does the translation. CSS never reaches AI Platform directly.

### Voice handoff (pstn transport)

Applies to Twilio and Exotel. Stringee is turn-based IVR and does not support
live call redirect.

When AI Platform sends a `call_offer` frame with `transport=pstn` over the chat relay:

```json
{
  "type": "call_offer",
  "transport": "pstn",
  "to": "+919876543210"
}
```

**CS is not in the audio path.** CS dials the customer and hands the media stream
to us via the answer URL. Choose one of the two patterns below.

---

#### Pattern A — Answer URL (recommended)

CS places the call and our answer URL fires when the customer answers.

1. **Intercept** the `call_offer` frame — do not forward to CRM Frontend.
2. Send CRM Frontend: `{"type":"mode_change","mode":"voice_pending","message":"Calling you now…"}`
3. **Dial** via the telephony provider, setting our slug-scoped answer URL as the webhook:
   ```
   https://{host}/api/v1/telephony/{provider}/voice/{tenant_slug}
   ```
   The provider returns the call SID synchronously in its response.
4. **Register** the SID with us so we have a conversation row before the webhook fires:
   ```
   POST /api/v1/telephony/register-call
   Authorization: Bearer <tenant-token>
   Content-Type: application/json

   {
     "provider": "twilio",
     "provider_call_sid": "<sid-from-step-3>",
     "lead_id": "..."
   }
   ```
   Response `201`: `{ "call_id": "call_...", "status": "in_progress" }`  
   Fires `call.initiated` (with `"source": "crm_register"`) to the CS webhook immediately.

5. Customer answers → provider fires our answer URL → AI bridge starts → `call.answered` fires.
6. When `call.completed` arrives on the CS webhook, send CRM Frontend: `{"type":"ended"}`

---

#### Pattern B — Mid-call patch-in

Use when CS needs to hold the customer (play a disclosure, route through its own
IVR) before engaging the AI.

1. CS dials the customer; holds them on IVR/hold music.
2. When ready, call the handoff endpoint with the **live** call SID:
   ```
   POST /api/v1/telephony/handoff
   Authorization: Bearer <tenant-token>
   Content-Type: application/json

   {
     "provider": "twilio",
     "call_sid": "CAxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
     "lead_id": "...",
     "crm_account_sid": null,   // omit if using the tenant's stored credentials
     "crm_auth_token": null
   }
   ```
   Response `200`: `{ "call_id": "call_...", "status": "answered", "provider_call_sid": "..." }`

   We call the provider's live-call update API to redirect the active call's media
   stream to our AI bridge WebSocket. Both `call.initiated` and `call.answered` fire
   immediately. `call.completed` fires at teardown.

3. Send CRM Frontend: `{"type":"ended"}` when `call.completed` arrives.

**Credential ownership for Pattern B:** by default we use the tenant's stored
telephony credentials. If CS holds the call on a **separate** Twilio/Exotel
account, pass `crm_account_sid` + `crm_auth_token` in the request, or redirect the
call to our answer URL using CS's own Twilio client and use Pattern A instead.

---

#### Pattern B error responses

| Status | Meaning |
|---|---|
| 400 | Provider unsupported (e.g. `stringee`) or credentials unresolvable |
| 409 | SID already registered — return the existing `call_id` |
| 502 | Provider rejected redirect (call already ended, SID invalid) |
| 503 | AI bridge not ready — retry in a few seconds |

---

For `websocket` and `webrtc` transports, CS forwards the `call_offer` frame as-is
— CS is not in the audio path for those either.

---

## Summarization API

Voice-only. For softphone calls where the human agent handled the conversation
and a recording is the only record of what was said (the AI was not on the call).

```
POST /api/v1/calls/{call_id}/summarize-outcome
Authorization: Bearer <tenant-token>
Content-Type: multipart/form-data

Fields:
  audio      — call recording file (required)
  audio_mime — MIME type, e.g. audio/mpeg, audio/wav (optional; inferred from file if omitted)
```

`call_id` is the platform call identifier returned by `POST /telephony/register-call`
or present in `call.initiated` webhook events. The call row must already exist (it
is created by the normal softphone flow or register-call).

Response `200`:
```json
{
  "call_id": "call_4a3f2b1c8d9e0f1a",
  "outcome": "callback_requested",
  "summary": "Customer asked to be called back tomorrow morning.",
  "notes": "Mentioned competitor pricing; prefers Hindi.",
  "callback_at": "2026-06-20T04:30:00+00:00"
}
```

The response also persists the outcome to the call row and emits `call.completed`
to the tenant's `events_webhook_url`.

**When to call it:**

| Scenario | Call summarization? |
|---|---|
| Customer → AI (chat only) | No — text transcript already in DB |
| Customer → AI → Human (chat) | No — text transcript already in DB |
| Customer → AI → Voice (pstn) | No — AI transcript available via `GET /conversations` |
| AI → Customer outbound (pstn) | No — outcome in `call.completed` webhook |
| Human → Customer softphone | Yes — when `outcome` is `null` or `recording-unavailable` |

---

## Migration Path

**Phase 1 — Add CS as the channel layer (chat relay)**  
CS is configured on a test tenant. We wire up the CS adapter (session proxy + WS relay + webhook forwarding + agent console proxy). All existing tenants on direct CRM Backend → AI Platform integration are unaffected.

**Phase 2 — Test all scenarios**  
Validate every inbound and outbound scenario end-to-end through CS. Run in parallel with direct integration on production tenants if needed.

**Phase 3 — Cut over**  
Once CS is proven, switch production tenants to route through CS. Direct CRM Backend → AI Platform calls are retired.

**Phase 4 — Voice channel (pstn)**  
With chat relay stable, wire up pstn voice for CS. CS places calls via the tenant's
telephony provider and uses our slug-scoped answer URL — no audio bridge needed on
the CS side. CS only needs: (a) outbound dialing via the provider SDK, (b) a call
to `POST /api/v1/telephony/register-call`, and (c) reception of `call.completed`
webhooks. See the "Voice handoff" section above.

Each phase is independently deployable. Rollback at any phase = switch tenant config back to previous path.

---

## Reconciliation

If `events_webhook_url` is not configured, or webhook delivery fails, CS can
recover outcomes via polling or re-analysis.

**List recent calls** (poll for `outcome: null`):
```
GET /api/v1/conversations?limit=20&offset=0
Authorization: Bearer <tenant-token>
```

**Re-run outcome analysis** from the stored transcript (AI voicebot calls):
```
POST /api/v1/conversations/{call_id}/reanalyze
Authorization: Bearer <tenant-token>
```
Returns `{ call_id, outcome, summary, notes, analysis_source }` and updates the row.

**Upload recording for analysis** (softphone / human-agent calls):
```
POST /api/v1/calls/{call_id}/summarize-outcome
Authorization: Bearer <tenant-token>
Content-Type: multipart/form-data

Fields:
  audio      — recording file (required)
  audio_mime — MIME type, e.g. audio/mpeg (optional)
```
Returns `{ call_id, outcome, summary, notes, callback_at }` and also emits
`call.completed` to the tenant's `events_webhook_url`.
