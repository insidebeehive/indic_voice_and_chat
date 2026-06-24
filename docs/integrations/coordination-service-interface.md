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
│    Session create proxy (CRM Frontend → AI Platform)        │
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
│    Bridges provider audio ↔ AI Platform voice WS            │  │
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
| Chat WebSocket relay | CRM Backend | CS (CSS retains webhooks + agent console) |
| Chat session creation (API call) | CRM Backend | CS on CSS's behalf |
| Human agent console (claim + agent-ws) | CRM Backend calls AI Platform directly | CRM Backend calls CS; CS proxies to AI Platform |
| Webhook events receiver | CRM Backend | stays with CSS (CS forwards our events) |
| Direct-human chat WS | CRM Backend | stays with CSS (CSS-owned; CS not involved for `operator_flag=human`) |
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
    CS calls IVoiceChannel.initiate_call(to, from_, stream_url=call_url, callback_url)
    CS bridges provider audio ↔ AI Platform voice WS (call_url)
    CRM Frontend receives mode_change{mode:"voice_pending"} instead
        │
   call ends (pstn only):
        │
    Provider POSTs callback ──► CS
    CS calls: POST /api/v1/chat/sessions/{session_id}/end
    AI Platform emits session_closed webhook ──► CS ──► CSS
    CS sends CRM Frontend an ended frame
```

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

### 1. AI → Customer (AI-initiated voice via chat session)

```
AI is in an active chat session via CS relay
AI decides to call the customer back
        │
AI Platform sends call_offer with transport=pstn over the relay WS
CS intercepts; calls IVoiceChannel.initiate_call(to, ...)
CS dials the customer; bridges provider audio ↔ AI Platform voice WS (call_url)
        │
   call ends
        │
Provider POSTs callback ──► CS
CS calls: POST /api/v1/chat/sessions/{session_id}/end
AI Platform emits session_closed webhook (with transcript + summary) ──► CS ──► CSS
```

We generate the transcript on our side because we ran the full conversation — CSS receives it via the `session_closed` webhook.

### 2. Human → Customer (support agent-initiated call)

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

When AI Platform sends a `call_offer` frame with `transport=pstn` over the chat relay:

```json
{
  "type": "call_offer",
  "transport": "pstn",
  "call_url": "wss://ai.example.com/voice/ws/plat_7f3a9b2e",
  "to": "+919876543210"
}
```

CS must:
1. **Intercept** this frame — do not forward to CRM Frontend.
2. Call `IVoiceChannel.initiate_call(to=frame.to, from_=tenant.caller_id, stream_url=frame.call_url, callback_url=cs_callback_url)`.
3. Bridge provider audio ↔ `frame.call_url` (AI Platform voice WS).
4. Send CRM Frontend: `{"type":"mode_change","mode":"voice_pending","message":"Calling you now…"}`
5. When call ends: call `POST /api/v1/chat/sessions/{session_id}/end` on AI Platform.
6. Send CRM Frontend: `{"type":"ended"}`

For `websocket` and `webrtc` transports, CS forwards the `call_offer` frame as-is — CS is not in the audio path.

### Audio stream (pstn bridging)

For pstn calls, CS bridges bidirectionally between the voice provider (customer's phone) and our voice WS at `call_url`:

```
Customer phone ──► provider RTP ──► CS bridge ──► PCM-16 binary ──► call_url (AI Platform)
Customer phone ◄── provider RTP ◄── CS bridge ◄── PCM-16 binary ◄── call_url (AI Platform)
```

After STT → LLM → TTS (or S2S), AI Platform sends the output audio back as PCM-16 binary frames on the same `call_url` connection. CS receives them and forwards them to the provider, which plays the audio to the customer's phone.

For `websocket` and `webrtc` transports the same principle holds, but CRM Frontend is connected directly to `call_url` — CRM Frontend sends mic audio in and receives TTS/S2S output back, with no CS in the audio path.

- **Binary frames (bidirectional):** PCM-16, 16 kHz, mono, little-endian (~20 ms chunks)
- **Text frames (JSON):**

| Direction | Frame | When |
|---|---|---|
| CS → us | `{"type":"start"}` | Stream ready |
| CS → us | `{"type":"barge_in"}` | Customer started speaking (CS VAD detected) |
| CS → us | `{"type":"end"}` | CS terminating the call |
| Us → CS | `{"type":"escalation","reason":"...","summary":"..."}` | AI requests human handover |
| Us → CS | `{"type":"ended","summary":"..."}` | AI ended the call |

### Session end (pstn call completion)

When a pstn call ends, CS calls:

```
POST /api/v1/chat/sessions/{session_id}/end
Authorization: Bearer <tenant-token>
Content-Type: application/json

{
  "call_id":    "cs-call-xyz",
  "status":    "completed | failed | no_answer",
  "duration_s": 142
}
```

We respond by emitting the `session_closed` webhook (which CS then forwards to CSS).

---

## Summarization API

Voice-only. For call recordings where audio is the only record of what was said.

```
POST /api/v1/summarize/call
Authorization: Bearer <tenant-token>
Content-Type: multipart/form-data

Fields:
  audio      — call recording file
  audio_mime — MIME type (audio/mpeg, audio/wav, audio/webm, audio/ogg)
  metadata   — optional JSON: { "session_id", "participants": ["ai","human_agent","customer"] }
```

Response `200`:
```json
{
  "transcript": [
    { "role": "customer",     "text": "My withdrawal has been pending for 3 days.", "ts": 0.0  },
    { "role": "ai",           "text": "I can see your withdrawal of ₹5,000...",    "ts": 4.2  },
    { "role": "human_agent",  "text": "I'll escalate this to our finance team.",   "ts": 38.1 }
  ],
  "summary": "Customer raised a 3-day withdrawal delay. AI clarified status. Human agent escalated to finance team.",
  "outcome": "resolved | escalated | no_resolution",
  "action_items": [
    "Finance team to follow up on withdrawal within 24 hours"
  ]
}
```

**When to call it:**

| Scenario | Call summarization? |
|---|---|
| Customer → AI (chat only) | No — text transcript already in DB |
| Customer → AI → Human (chat) | No — text transcript already in DB |
| Customer → AI → Voice (pstn) | Optional — CRM's choice |
| AI → Customer outbound (pstn) | Not needed — transcript in session_closed webhook |
| Human → Customer (outbound) | Optional — CRM's choice |

---

## Migration Path

**Phase 1 — Add CS as the channel layer (chat relay)**  
CS is configured on a test tenant. We wire up the CS adapter (session proxy + WS relay + webhook forwarding + agent console proxy). All existing tenants on direct CRM Backend → AI Platform integration are unaffected.

**Phase 2 — Test all scenarios**  
Validate every inbound and outbound scenario end-to-end through CS. Run in parallel with direct integration on production tenants if needed.

**Phase 3 — Cut over**  
Once CS is proven, switch production tenants to route through CS. Direct CRM Backend → AI Platform calls are retired.

**Phase 4 — Voice channel adapters**  
With chat relay stable, implement real voice channel adapters (Twilio, SIP/DiDLogic, WebRTC) so `pstn`-transport calls go live. Each adapter is independent — add one, register it, test it. No architectural changes to the chat relay.

Each phase is independently deployable. Rollback at any phase = switch tenant config back to previous path.
