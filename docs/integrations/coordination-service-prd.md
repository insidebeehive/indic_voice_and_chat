# Coordination Service — Product Requirements Document

**Audience:** CRM Architecture team  
**Authors:** AI Platform team  
**Status:** Proposed — for review and discussion

---

## Overview

The Coordination Service (CS) is a standalone microservice that owns the customer-facing channel layer. It sits between CRM Frontend (customers) and the AI Platform, handling session creation, WebSocket relay, webhook forwarding, and — eventually — telephony. CRM Backend retains its existing responsibilities: webhook receiver, human agent console, ticket system, and analytics.

CS makes the AI Platform pluggable: CRM Backend talks to CS, not directly to the AI Platform. Swapping the AI provider or adding a new channel adapter becomes a CS change, not a CRM Backend change.

---

## Goals

- **Chat relay (Phase 1 deliverable):** CS handles all chat traffic end-to-end — session creation, WebSocket relay, media proxying, webhook forwarding, and human agent handover.
- **Telephony framework (Phase 2 skeleton):** Define the voice channel interface and provider adapter stubs so telephony (Twilio, Stringee, SIP) can be implemented in a follow-up sprint without changing CS's core architecture.
- **Separation of concerns:** CRM Backend talks to CS. CRM Frontend talks to CS. Neither talks to the AI Platform directly.
- **Multi-tenancy from day one:** Every CS component is tenant-aware.

## Non-Goals

- Actual telephony implementation (Twilio / Stringee / SIP dialing) — skeleton only in this PRD.
- Call recording.
- WhatsApp, SMS, or any other channel adapter.
- Billing, usage tracking, or rate limiting.
- CS admin UI or dashboard.

---

## Topology

```
Customer browser
        │  WebSocket: wss://cs.example.com/chat/ws/{session_id}
        ▼
┌────────────────────────────────────────────────────────────────┐
│                  Coordination Service                           │
│                                                                │
│  Chat relay ──────────────────────────────────────────────────►│──┐
│  Session router (operator flag: ai / human / hybrid)           │  │
│  Media proxy                                                   │  │
│  Webhook forwarder                                             │  │
│  Voice channel registry (IVoiceChannel → adapter)             │  │
└────────────────────────────────────────────────────────────────┘  │
        │                    │                     ┌───────────────┘
        │ webhooks            │ telephony           │  calls AI Platform APIs
        │ (lifecycle events)  │ (skeleton — 501)    │
        ▼                    ▼                     ▼
┌──────────────┐   ┌──────────────────────────┐   ┌─────────────────────────────┐
│  CRM Backend  │   │   Telephony Providers     │   │       AI Platform            │
│               │   │                          │   │                              │
│  Webhook recv │   │  ┌──────────────────┐    │   │  POST /chat/sessions         │
│  Agent console│   │  │ SIP / DiDLogic   │    │   │  WS   /chat/ws/{id}          │
│  Ticket system│   │  │ (SIP UAC, RTP)   │    │   │  GET  /chat/sessions/{id}    │
│  Analytics    │   │  ├──────────────────┤    │   │  POST /sessions/{id}/claim   │
│  Business lgc │   │  │ Twilio           │    │   │  WS   /sessions/{id}/agent-ws│
└──────────────┘   │  │ (REST + TwiML)   │    │   │  GET  /chat/media/{id}        │
                   │  ├──────────────────┤    │   └─────────────────────────────┘
                   │  │ Stringee         │    │
                   │  │ (REST + SDK)     │    │
                   │  └──────────────────┘    │
                   └──────────────────────────┘
```

**Traffic rules:**
- CRM Frontend never calls AI Platform or CRM Backend directly.
- CRM Backend never calls AI Platform directly (CS is the bridge).
- The one exception: voice WS (`call_url` in `call_offer`) connects CRM Frontend to AI Platform directly — PCM-16 binary streams are too expensive to relay through an extra hop. This is documented explicitly and cannot be expanded.

---

## Component Map

| Component | Responsibility |
|---|---|
| **Tenant Registry** | Load and validate per-tenant config; resolve tenant from Bearer token or session |
| **Session Manager** | Map `session_id` → `{platform_ws_url, tenant, operator_flag}`; backed by Redis |
| **Chat Relay** | Bidirectional WS proxy between CRM Frontend and AI Platform; rewrite media URLs |
| **Webhook Forwarder** | Receive AI Platform lifecycle webhooks; verify HMAC; forward to CRM Backend |
| **Media Proxy** | Serve `GET /chat/media/{id}`; authenticate via session_id; proxy to AI Platform |
| **Agent Console Proxy** | Forward `POST /claim` and `WS /agent-ws` to AI Platform on behalf of CRM Backend |
| **Voice Channel Registry** | `IVoiceChannel` interface + provider adapter stubs; skeleton for telephony |

---

## 1. Chat Requirements (Full)

### 1.1 Session Creation Proxy

**Endpoint:** `POST /chat/sessions`  
**Auth:** Bearer token (CS-issued, per-tenant)

**Request from CRM Backend:**
```json
{
  "user_id": "player-42",
  "customer_name": "Rahul",
  "language": "hi",
  "operator_flag": "ai",
  "metadata": { "crm_ticket_id": "TKT-9001" }
}
```

**Operator flag routing:**

| `operator_flag` | CS action |
|---|---|
| `"human"` | Return `{"handled_by": "crm"}` immediately. Do NOT call AI Platform. |
| `"ai"` | Call `POST {ai_platform_base}/api/v1/chat/sessions` → get `{session_id, ws_url, greeting}`. Store in Session Manager. Return CS ws_url. |
| `"hybrid"` | Same as `"ai"`. Escalation transition is handled by AI Platform; CS relay is unchanged throughout. |

If `operator_flag` is absent, use the tenant's `operator.default_flag` from config.

**Response (201) for ai/hybrid:**
```json
{
  "session_id": "cs_a1b2c3d4",
  "ws_url": "wss://cs.example.com/chat/ws/cs_a1b2c3d4",
  "greeting": "Hello Rahul, how can I help?"
}
```

CS stores in Redis (`cs:session:{session_id}`, TTL 24 h):
```json
{
  "platform_session_id": "cs_a1b2c3d4",
  "platform_ws_url": "wss://ai.example.com/api/v1/chat/ws/cs_a1b2c3d4",
  "tenant_slug": "acme",
  "operator_flag": "ai"
}
```

### 1.2 Session Query Proxy

**GET /chat/sessions** and **GET /chat/sessions/{session_id}** are proxied to AI Platform with the tenant's Bearer token. CS adds no logic beyond auth and routing.

### 1.3 WebSocket Relay

**Endpoint:** `WS /chat/ws/{session_id}`  
**Auth:** `session_id` in URL path serves as the capability token (CS validates it exists in Redis).

**Connection lifecycle:**

1. CRM Frontend connects to CS WS.
2. CS looks up `platform_ws_url` from Redis.
3. CS connects to AI Platform WS.
4. AI Platform sends `history` frame immediately — CS forwards to CRM Frontend.
5. Bidirectional relay begins.

**CRM Frontend → CS → AI Platform:** All frames forwarded unchanged.

| Frame type | Action |
|---|---|
| `message` | Forward as-is |
| `image` | Forward as-is |
| `video` | Forward as-is |
| `audio` | Forward as-is |
| `end` | Forward; close both connections after delivery |

**AI Platform → CS → CRM Frontend:** Forward all frames; rewrite URL fields.

| Frame type | Action |
|---|---|
| `history` | Rewrite every `media_url` to CS proxy URL |
| `typing` | Forward as-is |
| `message` | Forward as-is |
| `audio_ack` | Rewrite `media_url` to CS proxy URL |
| `escalation` | Forward as-is |
| `mode_change` | Forward as-is |
| `call_offer` | Forward `call_url` as-is (voice WS exception — see §1.6) |
| `ended` | Forward; close both connections after delivery |
| `error` | Forward as-is |

**Media URL rewriting:**
```
AI Platform sends:  "media_url": "/api/v1/chat/media/103"
CS forwards:        "media_url": "https://cs.example.com/chat/media/103?session_id=cs_a1b2c3d4"
```

**Reconnection:**
- If AI Platform WS drops: CS waits 1 s, reconnects, discards the re-sent `history` frame (customer already has it), resumes relay.
- If CRM Frontend WS drops: CS keeps AI Platform WS open. Session stays active. Customer can reconnect; CS re-forwards the `history` frame from AI Platform on reconnect.

### 1.4 Webhook Forwarder

**Inbound endpoint:** `POST /internal/platform-webhook?tenant={slug}`

CS receives AI Platform lifecycle events:
- `session_started`
- `escalation_requested`
- `session_closed`

**Verification:** CS verifies the `X-Signature: sha256=<hmac>` header from AI Platform using the per-tenant `ai_platform_webhook_secret`.

**Forwarding:** CS POSTs the body unchanged to `crm_backend.webhook_url`, signing with `crm_backend.webhook_secret`:
```
X-CS-Signature: sha256=<hmac>
```

CS returns `200` to AI Platform as soon as the forward is accepted (fire-and-forget with a short retry). If CRM Backend is down, CS retries up to 3 times with exponential backoff before dropping.

### 1.5 Media Proxy

**Endpoint:** `GET /chat/media/{message_id}?session_id={session_id}`

1. CS validates `session_id` exists in Redis.
2. CS calls `GET {ai_platform_base}/api/v1/chat/media/{message_id}` with tenant Bearer token.
3. CS streams the 302 redirect (or the bytes directly) back to caller.

If `session_id` is absent or invalid: `401 Unauthorized`.  
If AI Platform returns 404: CS returns `404 Not Found`.

### 1.6 Human Agent Console Proxy

**Claim:**
```
POST /chat/sessions/{session_id}/claim
```
CS validates session exists, looks up tenant, proxies to:
```
POST {ai_platform_base}/api/v1/chat/sessions/{session_id}/claim
```
Returns AI Platform's response unchanged.

**Agent WebSocket:**
```
WS /chat/agent-ws/{session_id}?token={bearer_token}
```
CS validates the Bearer token identifies a valid CRM Backend tenant, then proxies the WebSocket connection to:
```
WS {ai_platform_base}/api/v1/chat/sessions/{session_id}/agent-ws?token={ai_platform_token}
```
All frames forwarded unchanged in both directions.

### 1.7 Voice Handoff (call_offer)

When AI Platform sends a `call_offer` frame, the `call_url` points to the AI Platform voice WS (or future CS voice WS). CS forwards this frame as-is.

CRM Frontend connects directly to `call_url`. This is the one case where CRM Frontend bypasses CS — binary PCM-16 audio streams are too expensive to relay through an extra hop.

CS must document this exception clearly and ensure it does not expand to other frame types.

---

## 2. Telephony Skeleton Requirements

These components must exist and be wired into CS configuration, but all actual implementations raise `NotImplementedError` or return HTTP 501. The goal is that adding a real provider in a follow-up sprint requires only implementing the interface — no architectural changes.

### 2.1 IVoiceChannel Interface

```python
from abc import ABC, abstractmethod

class IVoiceChannel(ABC):

    @abstractmethod
    async def initiate_call(
        self,
        to: str,
        from_: str,
        stream_url: str,
        callback_url: str,
    ) -> str:
        """Dial `to` from `from_`. Connect audio to `stream_url`.
        POST call-end event to `callback_url`. Return call_id."""

    @abstractmethod
    async def end_call(self, call_id: str) -> None:
        """Hang up an active call."""

    @abstractmethod
    async def get_call_status(self, call_id: str) -> dict:
        """Return {call_id, status, duration_s}."""
```

### 2.2 Provider Stubs

```python
class TwilioAdapter(IVoiceChannel):
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid  = account_sid
        self.auth_token   = auth_token
        self.from_number  = from_number

    async def initiate_call(self, to, from_, stream_url, callback_url) -> str:
        raise NotImplementedError
    async def end_call(self, call_id) -> None:
        raise NotImplementedError
    async def get_call_status(self, call_id) -> dict:
        raise NotImplementedError


class StringeeAdapter(IVoiceChannel):
    def __init__(self, api_key_sid: str, api_key_secret: str):
        self.api_key_sid    = api_key_sid
        self.api_key_secret = api_key_secret

    async def initiate_call(self, to, from_, stream_url, callback_url) -> str:
        raise NotImplementedError
    async def end_call(self, call_id) -> None:
        raise NotImplementedError
    async def get_call_status(self, call_id) -> dict:
        raise NotImplementedError


class SIPAdapter(IVoiceChannel):
    """
    SIP User Agent Client (UAC) adapter for SIP trunk providers such as DiDLogic.

    Outbound call flow:
      1. Send SIP INVITE to `server` with Digest Authentication.
      2. Receive 180 Ringing → 200 OK → send ACK.
      3. Negotiate RTP session via SDP in the INVITE / 200 OK exchange.
      4. Bridge RTP audio ↔ AI Platform PCM-16 WebSocket (`stream_url`).
      5. On hangup: send BYE; close RTP and WS connections.
    """

    def __init__(
        self,
        server: str,          # SIP proxy hostname  (e.g. sip.didlogic.net)
        port: int,            # 5060 for UDP/TCP; 5061 for TLS
        username: str,        # SIP account username / DID extension
        password: str,        # SIP account password (Digest auth)
        caller_id: str,       # E.164 number sent as SIP From: header
        transport: str,       # "udp" | "tcp" | "tls"
        realm: str = "",      # Digest auth realm; defaults to `server` if blank
    ):
        self.server    = server
        self.port      = port
        self.username  = username
        self.password  = password
        self.caller_id = caller_id
        self.transport = transport
        self.realm     = realm or server

    async def initiate_call(self, to, from_, stream_url, callback_url) -> str:
        """Send SIP INVITE; bridge RTP↔WebSocket; return call_id."""
        raise NotImplementedError

    async def end_call(self, call_id) -> None:
        """Send SIP BYE to tear down the call."""
        raise NotImplementedError

    async def get_call_status(self, call_id) -> dict:
        """Derive status from SIP response codes (200 OK, 486 Busy, 408 Timeout, etc.)."""
        raise NotImplementedError
```

### 2.3 Provider Registry

```python
def get_voice_channel(provider: str, config: dict) -> IVoiceChannel:
    if provider == "twilio":   return TwilioAdapter(**config)
    if provider == "stringee": return StringeeAdapter(**config)
    if provider == "sip":      return SIPAdapter(**config)
    raise ValueError(f"Unknown provider: {provider}")
```

### 2.4 SIP Implementation Notes

SIP differs from Twilio and Stringee in a fundamental way: **there is no REST API**. The SIP adapter must implement the SIP signalling protocol directly. Key points for the team that will implement this:

**Signalling library** — Use a Python SIP library (`aioSIP`, `pySIP`, or build on top of `twisted`) rather than raw socket programming. CS acts as a SIP UAC (User Agent Client): it sends INVITE, handles 1xx provisional responses, sends ACK on 200 OK, and sends BYE to hang up.

**Media bridging (hardest part)** — SIP calls carry audio over RTP, not WebSocket. The SIPAdapter must bridge between:
- The RTP session negotiated via SDP in the INVITE/200 OK exchange (G.711 µ-law or PCM, 8 kHz typically)
- The AI Platform's PCM-16 WebSocket stream (16 kHz mono little-endian)

This bridging requires a codec transcoder (8 kHz ↔ 16 kHz resampling). Options:
- A lightweight in-process bridge using `audioop` (stdlib) for resampling
- An external media gateway (FreeSWITCH, Asterisk) that CS connects to via AMI/ARI — more operationally complex but proven

**DiDLogic specifics** — DiDLogic provides SIP trunk credentials (username, password, SIP server, DID number). The expected call flow:
1. CS registers with DiDLogic's SIP proxy (REGISTER with Digest auth), or sends authenticated INVITE directly (trunk mode, no registration needed — DiDLogic supports both).
2. CS sends `INVITE sip:{to}@sip.didlogic.net` with SDP offering an RTP endpoint.
3. DiDLogic dials the destination PSTN number and replies 200 OK with SDP answer.
4. CS opens RTP socket, bridges audio to AI Platform WS.
5. On session end, CS sends `BYE`.

**Call status** — Unlike Twilio (REST poll) or Stringee (REST poll), SIP call status is derived from SIP response codes: 180/183 = Ringing, 200 = Connected, 486 = Busy, 408 = No Answer, 487 = Cancelled. `get_call_status` must track these in memory per `call_id`.

**This implementation is out of scope for the skeleton sprint.** The stub raises `NotImplementedError`. The notes above are captured here so the implementing team has the design context when the SIP sprint begins.

### 2.6 API Stubs

**Outbound call:**
```
POST /voice/calls
→ 501 Not Implemented
   {"detail": "telephony not implemented — configure a provider"}
```

**Audio stream:**
```
WS /voice/streams/{call_id}
→ Close with code 1001 (Going Away) + reason "not implemented"
```

---

## 3. Full API Contract

### CS Exposes

| Method | Path | Auth | Status |
|---|---|---|---|
| `POST` | `/chat/sessions` | Bearer (CS token) | Full |
| `GET` | `/chat/sessions` | Bearer (CS token) | Full (proxy) |
| `GET` | `/chat/sessions/{session_id}` | Bearer (CS token) | Full (proxy) |
| `POST` | `/chat/sessions/{session_id}/claim` | Bearer (CS token) | Full (proxy) |
| `WS` | `/chat/ws/{session_id}` | session_id in path | Full |
| `WS` | `/chat/agent-ws/{session_id}` | Bearer in ?token= | Full (proxy) |
| `GET` | `/chat/media/{message_id}` | ?session_id= | Full |
| `POST` | `/internal/platform-webhook` | HMAC verify | Full |
| `POST` | `/voice/calls` | Bearer (CS token) | Skeleton (501) |
| `WS` | `/voice/streams/{call_id}` | Bearer in ?token= | Skeleton (close 1001) |
| `GET` | `/health` | None | Always |

### CS Calls Outbound

| Destination | Call | When |
|---|---|---|
| AI Platform | `POST /api/v1/chat/sessions` | Session create (ai/hybrid) |
| AI Platform | `WS {platform_ws_url}` | Chat relay |
| AI Platform | `GET /api/v1/chat/sessions/{id}` | Session detail proxy |
| AI Platform | `GET /api/v1/chat/sessions` | Session list proxy |
| AI Platform | `GET /api/v1/chat/media/{id}` | Media proxy |
| AI Platform | `POST /api/v1/chat/sessions/{id}/claim` | Agent claim proxy |
| AI Platform | `WS /api/v1/chat/sessions/{id}/agent-ws` | Agent WS proxy |
| CRM Backend | `POST {webhook_url}` | Lifecycle event forward |

---

## 4. Multi-Tenancy and Configuration

### Config Schema

```yaml
redis_url: "redis://localhost:6379/0"   # omit for in-memory dev mode

tenants:
  acme:
    # Token CRM Backend uses to call CS
    cs_token: "cs_vox_abcdef..."

    ai_platform:
      base_url: "https://ai.example.com"
      token: "vox_..."             # AI Platform Bearer token for this tenant

    crm_backend:
      webhook_url: "https://crm.acme.com/webhooks/ai"
      webhook_secret: "..."        # CS signs outbound webhook with this

    ai_platform_webhook_secret: "..."  # CS verifies inbound AI Platform webhook with this

    operator:
      default_flag: "ai"           # ai | human | hybrid

    telephony:
      provider: "none"             # none | twilio | stringee | sip
      twilio:
        account_sid: ""
        auth_token: ""
        from_number: ""
      stringee:
        api_key_sid: ""
        api_key_secret: ""
      sip:
        server: ""           # SIP proxy hostname (e.g. sip.didlogic.net)
        port: 5060           # 5060 = UDP/TCP, 5061 = TLS
        username: ""         # SIP account username / DID extension
        password: ""         # Digest auth password
        caller_id: ""        # E.164 number shown to callee (From: header)
        transport: "udp"     # udp | tcp | tls
        realm: ""            # Digest auth realm; leave blank to default to server

  # Additional tenants follow the same shape
  betcorp:
    cs_token: "cs_vox_xyz..."
    ai_platform:
      base_url: "https://ai.example.com"
      token: "vox_..."
    ...
```

Sensitive fields (`cs_token`, `ai_platform.token`, secrets) should be injected via environment variable overrides rather than committed to the YAML file.

### Tenant Resolution

- **From CRM Backend HTTP calls:** Bearer token in `Authorization: Bearer cs_vox_...` header. CS looks up which tenant owns this token.
- **From WS connections (CRM Frontend):** `session_id` in URL path. CS resolves tenant from Redis session record.
- **From AI Platform webhooks:** `?tenant={slug}` query param in webhook URL (CS registers a per-tenant webhook URL with AI Platform, e.g. `https://cs.example.com/internal/platform-webhook?tenant=acme`).

---

## 5. Auth Design

| Caller | CS endpoint | Auth mechanism |
|---|---|---|
| CRM Backend | HTTP endpoints | `Authorization: Bearer {cs_token}` (opaque, per-tenant) |
| CRM Frontend / Customer | `WS /chat/ws/{session_id}` | `session_id` in path (capability token) |
| BO Agent | `WS /chat/agent-ws/{session_id}` | `?token={cs_token}` (same Bearer as CRM Backend) |
| AI Platform | `POST /internal/platform-webhook` | `X-Signature: sha256=<hmac>` |

CS calls AI Platform using the per-tenant `ai_platform.token`. CS signs outbound webhooks to CRM Backend using `crm_backend.webhook_secret`.

Tokens are never forwarded across boundaries — CS holds separate credentials for each relationship.

---

## 6. Session State

```
Redis key: cs:session:{session_id}
TTL:       86400 s (24 h)

Value (JSON):
{
  "platform_session_id": "cs_a1b2c3d4",
  "platform_ws_url": "wss://ai.example.com/api/v1/chat/ws/cs_a1b2c3d4",
  "tenant_slug": "acme",
  "operator_flag": "ai",
  "created_at": "2026-06-23T10:00:00Z"
}
```

Active WS connections are tracked in process memory (map from `session_id` to connection handles). On process restart, new customers must create new sessions; existing live sessions reconnect transparently (AI Platform re-sends history).

If Redis is unavailable, CS falls back to an in-process dict. This is acceptable for single-node deployments but not for multi-instance.

---

## 7. Tech Stack Recommendation

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | Matches AI Platform; shared team knowledge; no context switch |
| Framework | FastAPI | Same as AI Platform; WS support via Starlette; async-native |
| HTTP client | httpx (async) | Async, connection pooling, redirect following |
| WS client | websockets library | Mature; low-level control for relay logic |
| Session store | Redis (aioredis) | Multi-instance safe; TTL; fits existing infra |
| Config | PyYAML + Pydantic | Same pattern as AI Platform config.py |
| Deployment | Docker → Northflank | Same pipeline as AI Platform |
| Testing | pytest + pytest-asyncio | Same as AI Platform |

Alternative considered: **Node.js** is more natural for event-driven WS relay at scale, and would have slightly better raw throughput. Rejected in favour of Python for team consistency. If CS becomes a performance bottleneck at scale, a Node.js rewrite is a contained swap.

---

## 8. Implementation Phases

### Phase 1 — Foundation (Week 1)

**Deliverable:** Deployable CS shell that passes health checks.

- FastAPI app scaffold: `src/main.py`, `src/config.py`, `src/routers/`
- Config loader: parse YAML → Pydantic models; env-var overrides for secrets
- Tenant Registry: token → tenant lookup; slug → tenant lookup
- Auth middleware: validate CS Bearer tokens on inbound requests
- Redis client: connect on startup; in-memory fallback if `REDIS_URL` unset
- `GET /health` → `{"status": "ok", "tenants": ["acme", ...]}`

### Phase 2 — Session Proxy (Week 1–2)

**Deliverable:** CRM Backend can create, list, and query sessions through CS.

- `POST /chat/sessions`: operator flag routing; call AI Platform; store in Redis; return CS ws_url
- `GET /chat/sessions` and `GET /chat/sessions/{id}`: proxy to AI Platform
- Unit tests: operator flag routing, tenant resolution, Redis storage

### Phase 3 — WebSocket Relay (Week 2)

**Deliverable:** Full chat conversation works end-to-end through CS.

- `WS /chat/ws/{session_id}`: accept customer connection; look up platform_ws_url; connect to AI Platform; bidirectional relay
- Media URL rewriting in relay loop
- `history` frame handling on initial connect and on customer reconnect
- AI Platform WS reconnection (1 s wait, re-connect, discard re-sent history)
- Integration test: full message round-trip through relay

### Phase 4 — Webhook Forwarder (Week 2)

**Deliverable:** Lifecycle events flow from AI Platform → CS → CRM Backend.

- `POST /internal/platform-webhook`: HMAC verify (AI Platform signature); forward to CRM Backend; sign with CRM Backend secret
- Retry logic: up to 3 attempts with exponential backoff
- Unit tests: HMAC verify, HMAC sign, retry behaviour

### Phase 5 — Media Proxy (Week 3)

**Deliverable:** Images, audio, and video render correctly in CRM Frontend and BO console.

- `GET /chat/media/{message_id}?session_id={sid}`: validate session; call AI Platform with Bearer; stream response
- 401 on bad/missing session_id; 404 propagated from AI Platform
- Integration test: media fetch through CS

### Phase 6 — Agent Console Proxy (Week 3)

**Deliverable:** Human agents can claim sessions and chat through CS.

- `POST /chat/sessions/{session_id}/claim`: validate session; proxy to AI Platform
- `WS /chat/agent-ws/{session_id}?token={token}`: validate token; proxy WS to AI Platform agent-ws
- Test: claim → connect → send reply → receive customer message

### Phase 7 — Telephony Skeleton (Week 4)

**Deliverable:** Telephony framework wired in; all real calls return 501.

- `src/voice/interface.py`: `IVoiceChannel` ABC
- `src/voice/adapters/twilio.py`, `stringee.py`, `sip.py`: stubs
- `src/voice/registry.py`: `get_voice_channel(provider, config)` factory
- Config: telephony block added to Pydantic model
- `POST /voice/calls` → HTTP 501 with descriptive error
- `WS /voice/streams/{call_id}` → close with code 1001
- Unit test: factory resolves correct adapter class; 501 returned on call attempt

---

## 9. Open Questions for the Architecture Team

The following decisions affect the design of CS but are outside the AI Platform team's purview. Please resolve before development begins:

1. **Session persistence scale** — Is single-instance CS acceptable for the initial rollout, or is Redis required from day one? (Affects Phase 1 scope.)

2. **CS deployment topology** — Same Northflank project as AI Platform, or a separate project with its own scaling policy?

3. **CS token issuance** — Should CS issue tokens statically (pre-configured in YAML) or dynamically (a `POST /tokens` endpoint)? Static is simpler; dynamic is needed if tenants onboard frequently.

4. **Concurrency target** — Approximately how many simultaneous chat sessions in year 1? WS connection pool sizing and Redis keyspace design depend on this.

5. **Tenant config storage** — Static YAML file checked into the CS repo (simple; requires redeploy for new tenants) or DB-backed (dynamic onboarding without redeploy)?

6. **Voice WS exception** — Is the team comfortable with CRM Frontend connecting directly to AI Platform's voice WS (the `call_url` exception)? If all traffic must go through CS, voice relay adds significant complexity and latency.

7. **Hybrid flag ownership** — When `operator_flag = "hybrid"`, the AI escalates and the session transitions to a human agent. Should CS detect this transition and do anything (e.g. notify CRM Backend separately), or is the existing `escalation_requested` webhook sufficient?

8. **Webhook delivery guarantee** — CS retries up to 3 times on CRM Backend webhook failures. Is that sufficient, or does the architecture require a durable queue (Redis Streams, RabbitMQ) for guaranteed delivery?
