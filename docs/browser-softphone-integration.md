# Browser softphone — CRM integration

Let a CRM's **human agents** call leads from the browser (a softphone). Our AI
is **not** in the conversation — a human talks to the lead. Each call is
recorded, logged, billed, and marked as a **manual** call, and it hands back the
**same structured outcome an AI call does** (outcome / summary / notes /
callback), produced by transcribing the recording and running the same analyzer.

## How it works

```
CRM backend ──(tenant bearer)──▶ POST /api/v1/softphone/token ──▶ short-lived browser token
                                                                        │
CRM browser ◀───────────────────────────────────────────────────────────┘
   │  loads the provider JS SDK (Twilio Voice / Stringee Web), registers with the token,
   │  and dials the lead
   ▼
Telephony provider ──▶ TwiML App Voice URL ──▶ POST /api/v1/telephony/twilio/softphone-twiml/<slug>
   │  (we log a manual conversation row + return <Dial> with dual-channel recording)
   ▼
call happens (human ↔ lead), recorded
   ▼
provider ──▶ POST /api/v1/telephony/twilio/softphone-recording/<slug>
   transcribe (per channel) → analyze_call → record_outcome   ← SAME pipeline as AI calls
```

The CRM's **backend** mints the token (server-to-server, with its long-lived
tenant bearer). Only the **short-lived** provider token reaches the browser —
our tenant token never does.

## 1. Mint a browser token (CRM backend → us)

```http
POST /api/v1/softphone/token
Authorization: Bearer <tenant-token>
Content-Type: application/json

{ "agent_identity": "agent-42", "ttl_seconds": 3600 }
```

```json
{ "provider": "twilio", "token": "<jwt>", "identity": "agent-42",
  "ttl_seconds": 3600, "params": {} }
```

- Routes on the tenant's configured telephony provider.
- `400` if the provider has no browser SDK (see **Coverage**) or the tenant is
  missing softphone credentials.

## 2. Dial from the browser (CRM's app)

The CRM embeds the provider's JS SDK and registers with the token:

- **Twilio** — Voice JavaScript SDK: `new Device(token)`, then
  `device.connect({ params: { To: "<lead-number>" } })`. The `To` param is
  passed through to our TwiML App Voice URL.
- **Stringee** — Web SDK: register the client with the token (the JWT carries
  the agent `userId`) and place the call.

The SDK embedding and dialer UI are the CRM's work; we provide token-mint, dial
routing, recording, and outcome.

## 3. Outcome (identical to AI calls)

When the recording is ready the provider calls our recording webhook. We:

1. fetch the **dual-channel** recording (agent + lead on separate tracks),
2. batch-transcribe each channel with the tenant's STT
   (`ISTTProvider.transcribe`),
3. run the **same** `analyze_call` AI calls use, and
4. persist via the **same** `record_outcome`.

So `GET /api/v1/calls/{id}` and the backoffice return an identical
`outcome / summary / notes / callback_at` shape for human and AI calls. Manual
calls are marked `agent_type="human"`, `channel="softphone"` and broken out in
`GET /api/v1/tenants/{id}/analytics` under `by_agent_type`.

## Per-tenant setup

Register the tenant with `telephony.provider = "twilio"` and these keys (encrypted
at rest):

| key | purpose |
|---|---|
| `account_sid` / `auth_token` | server dialing + fetching the recording |
| `api_key_sid` / `api_key_secret` | sign the browser AccessToken |
| `twiml_app_sid` | the TwiML App whose Voice URL is our dial endpoint |

Create a Twilio **TwiML App** with its Voice URL set to:

```
https://<our-host>/api/v1/telephony/twilio/softphone-twiml/<tenant-slug>
```

(Stringee needs only its `account_sid` / `auth_token` — the client JWT reuses
them.)

## Coverage / limits

- **Twilio** ✅ and **Stringee** ✅ have browser WebRTC SDKs and are supported.
- **Exotel, DiDLogic / raw SIP** — no browser SDK; need a self-hosted
  WebRTC↔SIP gateway. **Deferred.**
- v1 transcribes the recording **post-call** (enough for outcome parity); live
  transcription during the call is a follow-up.
- Billing: a manual call's platform cost is the transcription (STT) + analysis
  (LLM); telephony stays the tenant's own trunk (shown tentative). The Stringee
  recording → outcome webhook is wired the same way once Stringee live calls are
  unblocked.
