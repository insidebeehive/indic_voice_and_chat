# Coordination Service — Integration Interface

**Audience:** CRM Architecture team + AI Platform team  
**Status:** Proposed — CS is conceptual; this document defines the interface so both sides can build independently  

---

## Approach

We add the Coordination Service (CS) as **one more telephony option** alongside our existing providers (Twilio, Stringee, Exotel). Nothing gets removed. Once CS is integrated, tested, and proven across all scenarios, we retire the other providers one by one.

This means:
- No disruption to current production integrations
- CS can be validated in staging before any cutover
- Rollback is trivial — just switch the tenant config back to Twilio/Stringee

---

## Roles

```
Customer (any channel)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│             Coordination Service  (CRM team)                │
│                                                             │
│  Channel adapters:                                          │
│    Chat WebSocket relay                                     │
│    Telephony (Twilio, Stringee, SIP, VoIP)                  │
│    WhatsApp, SMS (future)                                   │
│                                                             │
│  Session router:                                            │
│    operator flag = AI      ───────────────────────────────► │──┐
│    operator flag = human   ──► BO agent softphone           │  │
│    operator flag = hybrid  ──► AI first, escalate to human  │  │
│                                                             │  │
│  Call recording (owns the audio archive)                    │  │
└────────────────────────────────────────────────────────────┘  │
                                                                  │
                                              ┌───────────────────┘
                                              ▼
                              ┌─────────────────────────────┐
                              │       AI Platform (us)       │
                              │                              │
                              │  STT → LLM → TTS             │
                              │  RAG / knowledge base        │
                              │  CRM tool integration        │
                              │  Escalation decision         │
                              │  Media storage (S3)          │
                              │  Summarization API           │
                              └─────────────────────────────┘
```

**CS owns:** channels, routing, telephony, call recording  
**We own:** AI intelligence, speech processing, media storage, summarization

---

## What Changes, What Stays

| Concern | Today | With CS |
|---|---|---|
| Twilio / Stringee / Exotel adapters | active | stay active until CS is proven |
| SIP / DiDLogic trunk | active | stays until CS is proven |
| STT (Sarvam, Deepgram, Gemini) | us | stays with us |
| LLM (Gemini) | us | stays with us |
| TTS | us | stays with us |
| RAG / knowledge base | us | stays with us |
| CRM tool integration | us | stays with us |
| Escalation decision | us | stays with us |
| Media storage (S3) | us | stays with us |
| Chat transcript | DB + webhook | unchanged |
| Call recording | not our concern | CS owns it |

---

## Inbound Scenarios

**Entry point is always chat.** The customer opens the chat widget. The conversation can stay in chat or escalate to a voice call from within chat.

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
  - session_closed webhook payload

No summarization call needed — transcript is already structured text.
```

### 2. Customer → AI → Human (chat handover)

```
Customer chats with AI via CS
AI decides to escalate
        │
CS switches relay to human agent's console
Human agent types replies, customer types back
        │
   session ends
        │
Full text transcript (AI portion + human agent portion) available in:
  - Our DB
  - session_closed webhook

No summarization call needed.
```

### 3. Customer → AI → Voice call

```
Customer chats with AI via CS
Customer requests a voice call  OR  AI sends call_offer
        │
CS initiates a voice call via its telephony layer
CS streams call audio to us (PCM-16, 16 kHz, mono)
We continue as the AI voice agent (STT → LLM → TTS)
        │
   call ends
        │
CS saves the voice recording  ◄── CS responsibility
        │
Optionally: CRM calls our summarization API with the recording
            to get a structured transcript + summary
```

Chat transcript (before the voice call) is already in our DB.  
Voice recording is owned and archived by CS.  
Summarization is CRM's choice, not mandatory.

### 4. Customer → Human (direct, operator flag = human)

```
CS routes chat directly to human agent's console.
We are not involved at all.
```

---

## Outbound Scenarios

### 1. AI → Customer

```
Us ──POST /cs/calls──► CS  (initiate outbound call)
CS dials the customer
CS ──PCM-16 audio stream──► Us
STT → LLM → TTS — unchanged from today
        │
   call ends
        │
We push summary to CS callback URL  ◄── we generate this automatically
(we have the full STT transcript; CS doesn't need to send us audio)

CS also saves its own recording for archive/compliance.
```

We generate the summary ourselves because we ran the full conversation — CS gets it pushed at call end.

### 2. Human → Customer

```
BO agent initiates call via CS softphone
CS connects BO agent ↔ customer
We are not involved during the call
        │
   call ends
        │
CS saves the recording  ◄── CS responsibility
        │
Optionally: CRM calls our summarization API with the recording
```

---

## What CS Must Provide Us

For the CS telephony option to work, CS must expose the following to us:

### Inbound call delivery

When a customer call arrives (voice, from within chat), CS notifies us:

```
POST <our-inbound-webhook-url>
Content-Type: application/json

{
  "call_id":    "cs-call-xyz",
  "session_id": "cs-session-abc",   // CS session ID
  "stream_url": "wss://cs.example.com/streams/cs-call-xyz",
  "direction":  "inbound",
  "from":       "+919876543210",
  "context": {
    "customer_id":    "player-42",
    "customer_name":  "Rahul",
    "language":       "hi",
    "chat_session_id": "cs-session-abc"
  }
}
```

We respond `200` to accept the call, then connect to `stream_url`.

### Audio stream

Bidirectional WebSocket at `stream_url`:

- **Binary frames:** PCM-16, 16 kHz, mono, little-endian (~20 ms chunks)
- **Text frames (JSON):**

| Direction | Frame | When |
|---|---|---|
| CS → us | `{"type":"start"}` | Stream ready |
| CS → us | `{"type":"barge_in"}` | Customer started speaking (CS VAD detected) |
| CS → us | `{"type":"end"}` | CS terminating the call |
| Us → CS | `{"type":"escalation","reason":"...","summary":"..."}` | AI requests human handover |
| Us → CS | `{"type":"ended","summary":"..."}` | AI ended the call |

### Outbound call API

```
POST /cs/calls
Authorization: Bearer <cs-issued-service-token>
Content-Type: application/json

{
  "to":         "+919876543210",
  "from":       "+918204268005",
  "stream_url": "wss://us.example.com/voice/stream/cs-call-xyz",
  "callback_url": "https://us.example.com/api/v1/voice/cs-callback"
}
```

CS dials `to`, connects the audio to our `stream_url`, POSTs call-end event to `callback_url`.

### Call-end event

```
POST <our-callback-url>
Content-Type: application/json

{
  "call_id":   "cs-call-xyz",
  "session_id": "cs-session-abc",
  "status":    "completed | failed | no_answer",
  "duration_s": 142
}
```

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
| Customer → AI → Voice | Optional — CS owns the recording; CRM's choice |
| AI → Customer (outbound) | Not needed — we push summary automatically at call end |
| Human → Customer (outbound) | Optional — CS owns the recording; CRM's choice |

---

## Migration Path

**Phase 1 — Add CS as a telephony option**  
CS is configured on a test tenant alongside our existing providers. We wire up the CS adapter (inbound webhook + audio stream + outbound call API). All existing tenants on Twilio/Stringee are unaffected.

**Phase 2 — Test all scenarios**  
Validate every inbound and outbound scenario end-to-end through CS. Run in parallel with existing providers on production tenants if needed.

**Phase 3 — Cut over and retire**  
Once CS is proven, switch production tenants to CS. Retire the Twilio, Stringee, and Exotel adapters from our codebase (`src/api/telephony/`, `src/api/stringee/`).

Each phase is independently deployable. Rollback at any phase = switch tenant config back to previous provider.
