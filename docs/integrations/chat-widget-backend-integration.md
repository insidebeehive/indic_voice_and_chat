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

---

## Auth

Tenant API tokens are issued once during tenant registration (`vox_...` format, 32 bytes URL-safe). Store them securely — they're shown only once.

```
Authorization: Bearer vox_<token>
```

Send this header on every call to our management APIs. Keep it server-side — it never goes to the browser.

---

## 1. Session Creation

When your operator flag is set to AI, create a session with us before serving the chat to the customer:

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

## 3. Webhook Events

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

## 4. Human Agent Console

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

## 5. Media Download

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

## 6. Quick-Start Checklist

**Session flow:**
- [ ] When operator flag = AI: call `POST /chat/sessions` server-side with `user_id`, `customer_name`, `language`, and your `crm_ticket_id` in `metadata`
- [ ] Store the `session_id` ↔ `crm_ticket_id` mapping
- [ ] Pass `session_id`, `ws_url`, `greeting` to CRM Frontend — never the Bearer token
- [ ] When operator flag = human: don't call us; handle entirely in your own system

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
