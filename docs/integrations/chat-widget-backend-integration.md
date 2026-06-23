# Chat Widget — Backend Integration Guide

**Audience:** CRM Backend team  
**Scope:** Everything server-side — creating AI chat sessions, proxying the WebSocket to your frontend, receiving lifecycle webhooks, handling all media types, and building the human agent console for escalations.

> **Share with your frontend team:** Once you have this integration working, share `chat-widget-frontend-integration.md` with the CRM Frontend team. It documents the WebSocket message protocol (including all text, image, video, and audio message types) so they can build the customer UI. You are responsible for implementing and relaying that full protocol — including all media types — between the AI platform and your frontend.

---

## Architecture Overview

```
CRM Frontend
     │  WebSocket (your WS, not ours)
     ▼
CRM Backend  ◄──────────────────────────────────────────► AI Platform (us)
     │                                                          │
     ├── POST /chat/sessions (Bearer token) ──────────────────►│ create session
     │◄── {session_id, ws_url} ──────────────────────────────── │
     │                                                          │
     ├── WebSocket relay (proxy ws_url) ◄────────────────────► │ AI conversation
     │   relay all frames: text, image, video, audio            │
     │                                                          │
     │◄── webhook: session_started ──────────────────────────── │
     │◄── webhook: escalation_requested ─────────────────────── │ AI escalates
     │◄── webhook: session_closed ────────────────────────────── │
     │                                                          │
     ├── POST /sessions/{id}/claim ──────────────────────────► │ agent claims
     └── WS   /sessions/{id}/agent-ws ◄────────────────────── │ agent relay

                      flag = human
CRM Backend  handles entirely in your own system — do not call us at all
```

**You are the only team that talks to us.** CRM Frontend never calls our APIs directly. You proxy the WebSocket and handle all media upload, download, and relay on their behalf.

> **When the Coordination Service is deployed:** CS becomes the channel adapter — it owns the WebSocket relay, serves the chat widget, and calls our session creation API on your behalf. Your integration shifts: register your webhook endpoint with CS instead of directly with us, and CS will forward our lifecycle events to you. The human agent console (`claim` + `agent-ws`), media download, and all business logic (ticket system, analytics) remain with you.

---

## Auth

Tenant API tokens are issued once during tenant registration (`vox_...` format, 32 bytes URL-safe). Store them securely — they're shown only once.

```
Authorization: Bearer vox_<token>
```

Send this header on every call to our management APIs. Keep it server-side — it never goes to the browser.

---

## 1. Session Creation

When your operator flag is set to AI, create a session with us before serving the chat to the customer. (When the Coordination Service is deployed, CS does this on your behalf — you will configure your session parameters with CS instead.)

```
POST /api/v1/chat/sessions
Authorization: Bearer <your-token>
Content-Type: application/json
```

**Request:**

```json
{
  "user_id": "player-42",       // your CRM player ID — stored with the session for correlation
  "customer_name": "Rahul",      // personalises the AI greeting
  "language": "hi",              // default "hi"; pass "en" for English
  "metadata": {                  // optional — any key/value you want stored
    "crm_ticket_id": "TKT-9001"
  }
}
```

**Response (201):**

```json
{
  "session_id": "cs_a1b2c3d4e5f6g7h8",
  "greeting":   "Hello Rahul, how can I help?",
  "ws_url":     "wss://platform.example.com/api/v1/chat/ws/cs_a1b2c3d4e5f6g7h8"
}
```

**Store the mapping:** `session_id` ↔ your `crm_ticket_id`. You'll need it when webhooks arrive.

Pass `session_id`, `ws_url`, and `greeting` to CRM Frontend. Do not pass the Bearer token.

---

## 2. Session Management

### List sessions

```
GET /api/v1/chat/sessions
Authorization: Bearer <your-token>
```

Query params (all optional):

| Param | Values | Description |
|---|---|---|
| `status` | `active`, `ended`, `escalated` | Filter by session status |
| `mode` | `ai`, `awaiting_human`, `human`, `closed` | Filter by current mode |
| `customer_id` | any string | Filter by your `user_id` |

Response:
```json
{
  "sessions": [
    {
      "session_id": "cs_...",
      "status": "escalated",
      "mode": "awaiting_human",
      "customer_id": "player-42",
      "customer_name": "Rahul",
      "language": "hi",
      "message_count": 7,
      "started_at": "2026-06-22T10:00:00",
      "ended_at": null
    }
  ]
}
```

Use `mode=awaiting_human` to poll for sessions waiting for a human agent.

### Get session detail

```
GET /api/v1/chat/sessions/{session_id}
Authorization: Bearer <your-token>
```

Returns the session summary plus the full message history:

```json
{
  "session_id": "cs_...",
  "status": "escalated",
  "mode": "awaiting_human",
  "customer_id": "player-42",
  "customer_name": "Rahul",
  "language": "hi",
  "message_count": 7,
  "started_at": "2026-06-22T10:00:00",
  "ended_at": null,
  "summary": null,
  "messages": [
    { "role": "customer",  "type": "text",  "content": "Hello",         "timestamp": "2026-06-22T10:00:00" },
    { "role": "agent",     "type": "text",  "content": "Hello Rahul…",  "timestamp": "2026-06-22T10:00:01" },
    { "role": "customer",  "type": "audio", "content": "[audio]",       "timestamp": "2026-06-22T10:01:00" }
  ]
}
```

---

## 3. WebSocket Relay

You open **two WebSocket connections** — one inward-facing (your frontend talks to you) and one outward-facing (you talk to us). Your job is to pipe frames between them.

```
CRM Frontend ──────► your WS server ──────► our WS (ws_url)
             ◄──────              ◄──────
```

### Connection lifecycle

1. CRM Frontend connects to your WS endpoint.
2. You connect to our `ws_url` (from `POST /sessions`).
3. We immediately send a `history` frame on our WS — forward it to CRM Frontend.
4. Keep both connections alive for the duration of the session.
5. If CRM Frontend disconnects, keep our WS open — the session stays active and the customer can reconnect.
6. If our WS drops, reconnect using the same `ws_url` — we will re-send the `history` frame on reconnect.
7. When the session ends (`ended` frame from us, or `end` frame from CRM Frontend), close both connections.

### Frame passthrough — CRM Frontend → Us

Forward every frame from CRM Frontend to our WS unchanged:

| type | Forward as-is |
|---|---|
| `message` | ✓ |
| `image` | ✓ (base64 data + mime) |
| `video` | ✓ (base64 data + mime) |
| `audio` | ✓ (base64 data + mime) |
| `end` | ✓ |

### Frame passthrough — Us → CRM Frontend

Forward every frame from our WS to CRM Frontend, **except rewrite any URL fields** (see Media URLs below):

| type | Forward | Notes |
|---|---|---|
| `history` | ✓ | Rewrite `media_url` fields |
| `typing` | ✓ | |
| `message` | ✓ | |
| `audio_ack` | ✓ | Rewrite `media_url` |
| `escalation` | ✓ | |
| `mode_change` | ✓ | |
| `call_offer` | ✓ | Forward as-is (see Voice Handoff below — `call_url` is never rewritten) |
| `ended` | ✓ | Close both connections after forwarding |
| `error` | ✓ | |

### Media URLs

Every `media_url` we send points to our platform (`/api/v1/chat/media/{id}`) and requires auth. Since CRM Frontend must never call our APIs directly, **rewrite these URLs to your own proxy endpoint** before forwarding to CRM Frontend.

```
We send:    "media_url": "/api/v1/chat/media/103"
You forward: "media_url": "https://your-backend.com/chat/media/103"
```

Your proxy endpoint then fetches from us with the Bearer token and streams the response to CRM Frontend:

```python
# Your proxy route: GET /chat/media/{message_id}
async def proxy_media(message_id: int, request: Request) -> StreamingResponse:
    # verify your own session auth here
    url = f"{PLATFORM_BASE_URL}/api/v1/chat/media/{message_id}"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {PLATFORM_TOKEN}"},
                             follow_redirects=True)
    return StreamingResponse(r.aiter_bytes(), media_type=r.headers.get("content-type"))
```

### Connection management — reference implementation

```python
import asyncio, json, websockets

async def relay_session(frontend_ws, ws_url: str, base_url: str, session_id: str):
    """Relay frames between CRM Frontend WS and the AI platform WS."""
    async with websockets.connect(ws_url) as platform_ws:

        async def frontend_to_platform():
            async for raw in frontend_ws:
                await platform_ws.send(raw)  # forward unchanged

        async def platform_to_frontend():
            async for raw in platform_ws:
                msg = json.loads(raw)
                # Rewrite media URLs before forwarding
                if msg.get("media_url"):
                    msg["media_url"] = rewrite_url(msg["media_url"], base_url)
                if msg.get("type") == "history":
                    for m in msg.get("messages", []):
                        if m.get("media_url"):
                            m["media_url"] = rewrite_url(m["media_url"], base_url)
                await frontend_ws.send(json.dumps(msg))
                if msg.get("type") == "ended":
                    return  # session over

        await asyncio.gather(frontend_to_platform(), platform_to_frontend())

def rewrite_url(platform_url: str, base_url: str) -> str:
    # "/api/v1/chat/media/103" → "https://your-backend.com/chat/media/103"
    return base_url + platform_url.replace("/api/v1/chat", "/chat")
```

### Voice handoff (`call_offer`)

`call_offer` frames now carry a `transport` field. Your relay behaviour differs per transport:

| `transport` | What you receive | What to do |
|---|---|---|
| `websocket` | `call_url` (WS endpoint) | Forward as-is. CRM Frontend connects to `call_url` directly. Binary PCM-16 is expensive to proxy. |
| `webrtc` | `call_url` (signalling endpoint) + `ice_servers` | Forward as-is. CRM Frontend does WebRTC ICE negotiation directly with `call_url`. |
| `pstn` | Neither — CS intercepts this frame entirely | **This frame never reaches you.** When CS is deployed, it swallows `transport=pstn` frames, dials the customer's phone via the voice channel adapter, and sends CRM Frontend a `mode_change` frame instead. Without CS, this transport is not supported. |

For `websocket` and `webrtc`, forwarding the frame as-is is the right call — CRM Frontend handles the rest.

---

## 4. Webhook Events

Configure your `events_webhook_url` on your tenant. We POST all lifecycle events there.

### Request format

```
POST <your-events_webhook_url>
Content-Type: application/json
X-Signature: sha256=<hmac-hex>   (only if events_webhook_secret_env is configured)
```

Body always contains `"event"` plus event-specific fields.

### Verifying signatures

If you configured a webhook secret, verify every request:

```python
import hmac, hashlib

def verify(secret: str, raw_body: bytes, header: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)
```

### `session_started`

Sent when a session is created.

```json
{
  "event": "session_started",
  "session_id": "cs_a1b2c3d4e5f6g7h8",
  "customer": {
    "name": "Rahul",
    "id": "player-42"
  }
}
```

Use this to create a corresponding support ticket in your system and store the `session_id` ↔ ticket mapping.

### `escalation_requested`

Sent when the AI decides it cannot handle the conversation and needs a human.

```json
{
  "event": "escalation_requested",
  "session_id": "cs_a1b2c3d4e5f6g7h8",
  "reason": "Customer requested human support",
  "summary": "Customer is asking about a withdrawal that has been pending for 3 days.",
  "customer": {
    "name": "Rahul",
    "id": "player-42"
  },
  "claim_url": "/api/v1/chat/sessions/cs_a1b2c3d4e5f6g7h8/claim",
  "agent_ws_url": "/api/v1/chat/sessions/cs_a1b2c3d4e5f6g7h8/agent-ws",
  "bo_available": true
}
```

| Field | What to do |
|---|---|
| `reason` | Display to your human agent as context |
| `summary` | AI-generated summary of the conversation so far |
| `claim_url` | Relative URL — prepend our base URL; your agent calls this to claim the session |
| `agent_ws_url` | Relative URL — prepend our base URL; your agent console connects here |
| `bo_available` | `false` means outside support hours — customer is already informed and waiting |

### `session_closed`

Sent when a session ends (by customer, agent, or AI).

```json
{
  "event": "session_closed",
  "session_id": "cs_a1b2c3d4e5f6g7h8",
  "mode_at_close": "ai",
  "summary": "Customer asked about withdrawal delay; resolved by AI.",
  "transcript": [
    { "role": "customer",  "text": "Hello",        "ts": "2026-06-22T10:00:00" },
    { "role": "agent",     "text": "Hello Rahul…", "ts": "2026-06-22T10:00:01" },
    { "role": "customer",  "text": "My withdrawal…","ts": "2026-06-22T10:00:30" },
    { "role": "agent",     "text": "I can see…",   "ts": "2026-06-22T10:00:32" }
  ]
}
```

`mode_at_close` values: `ai` (resolved by AI), `human` (closed while agent was live).

Use `transcript` to populate your ticket history. For media messages (`type=audio/image/video`), download via the media endpoint below.

---

## 5. Human Agent Console

When you receive `escalation_requested`, route it to a human agent in your system. Your agent console needs to:

### Step 1 — Claim the session

```
POST /api/v1/chat/sessions/{session_id}/claim
Authorization: Bearer <your-token>
Content-Type: application/json
```

```json
{
  "agent_id":   "agent-priya",
  "agent_name": "Priya"
}
```

Responses:
- `200` — claimed; session mode is now `human`; customer WS receives a `mode_change` frame with the agent name
- `409` — already claimed by another agent
- `400` — session is not in `awaiting_human` mode

### Step 2 — Connect the agent WebSocket

```
WS /api/v1/chat/sessions/{session_id}/agent-ws?token=<bearer-token>
```

Auth via `?token=` query param (same Bearer token as your API calls).

**On connect** the server immediately sends a `history` frame with the full conversation:

```json
{
  "type": "history",
  "messages": [
    { "id": 101, "role": "customer",  "text": "Hello",            "media_url": null,                     "media_mime": null, "ts": "2026-06-22T10:00:00" },
    { "id": 102, "role": "agent",     "text": "Hello Rahul…",     "media_url": null,                     "media_mime": null, "ts": "2026-06-22T10:00:01" },
    { "id": 103, "role": "customer",  "text": "[voice message]",  "media_url": "/api/v1/chat/media/103", "media_mime": "audio/webm;codecs=opus", "ts": "2026-06-22T10:01:00" }
  ]
}
```

For media, fetch via `GET /api/v1/chat/media/{id}` with the Bearer token (see §5).

**Agent → server frames:**

| Frame | Description |
|---|---|
| `{"type":"reply","text":"..."}` | Send a message to the customer; persisted as `human_agent` role |
| `{"type":"end"}` | End the session; triggers `session_closed` webhook |

**Server → agent frames:**

| Frame | Description |
|---|---|
| `{"type":"customer_message","text":"...","session_id":"..."}` | Customer sent a message |
| `{"type":"system","text":"Customer disconnected — session still open"}` | Customer WS dropped (session still open) |

**Close codes:**
- `4004` — session not found or not in handover mode
- `1008` — invalid/missing token

---

## 6. Media Download

Download media attachments (voice, images, videos) from sessions:

```
GET /api/v1/chat/media/{message_id}
Authorization: Bearer <your-token>
```

Returns **302 redirect** to a time-limited signed URL (default TTL: 1 hour). Follow the redirect to download.

```python
import httpx

async def download_media(message_id: int, token: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(
            f"https://platform.example.com/api/v1/chat/media/{message_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        return r.content
```

`message_id` comes from the `id` field in history frames or the `transcript` in `session_closed` webhooks.

---

## 7. Quick-Start Checklist

**Session flow:**
- [ ] When operator flag = AI: call `POST /chat/sessions` server-side with `user_id`, `customer_name`, `language`, and your `crm_ticket_id` in `metadata`
- [ ] Store the `session_id` ↔ `crm_ticket_id` mapping
- [ ] Pass `session_id`, `ws_url`, `greeting` to CRM Frontend — never the Bearer token
- [ ] When operator flag = human: don't call us; handle entirely in your own system

**WebSocket relay:**
- [ ] Open your own WS endpoint that CRM Frontend connects to
- [ ] On CRM Frontend connect: open our `ws_url`; forward the `history` frame immediately
- [ ] Forward all frames CRM Frontend → us unchanged (text, image, video, audio, end)
- [ ] Forward all frames us → CRM Frontend; rewrite `media_url` fields to your proxy endpoint
- [ ] Expose a media proxy route (`GET /chat/media/{id}`) that fetches from us with Bearer token
- [ ] If our WS drops: reconnect and re-forward the `history` frame to CRM Frontend
- [ ] On `ended` frame: close both connections

**Webhooks (implement a receiver at `events_webhook_url`):**
- [ ] `session_started` → create support ticket in your system, store `session_id` mapping
- [ ] `escalation_requested` → assign to available agent; pass `reason`, `summary`, `claim_url`, `agent_ws_url` to their console
- [ ] `session_closed` → update ticket status, store transcript
- [ ] Verify `X-Signature` header if you configured a webhook secret

**Human agent console:**
- [ ] `POST /sessions/{id}/claim` with `agent_id` and `agent_name` before connecting the WS
- [ ] `WS /sessions/{id}/agent-ws?token=` — connect; read `history` frame on open
- [ ] Render media in history via `GET /media/{id}` with Bearer token
- [ ] Send `{"type":"reply","text":"..."}` for agent messages
- [ ] Send `{"type":"end"}` to close the session
- [ ] Handle `customer_message` frames to show incoming customer messages
- [ ] Handle `system` frames to show connection status notices
