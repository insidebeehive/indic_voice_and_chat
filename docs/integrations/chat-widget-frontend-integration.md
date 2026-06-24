# Chat Widget — WebSocket Protocol Reference

**Audience:** CSS (Chat Support System) engineering team — widget implementation  
**Scope:** WebSocket message protocol implemented internally by the CSS-owned chat widget.

> **CRM Frontend teams:** You do not need this document. The chat widget is a CSS-owned, hosted JS bundle — you embed it with one `<script>` tag. See **`chat-widget-embed-guide.md`** for the embed API (config).

> **CSS widget engineers:** This document describes the full WS message protocol your widget must implement. CSS / CS is the other end of every frame listed here.

---

## How It Fits Together

```
Customer browser
  └── CSS Widget (this code)
            │
            │  WebSocket
            ▼
         CSS / CS  (CSS backend, via CS relay)
            │  session creation, webhooks, media, human agent console
            ▼
       AI Platform
```

CSS (via CS) decides whether the conversation goes to AI or directly to a human agent. After calling `POST /api/chat/start`, the widget receives:

```
session_id  — e.g. "cs_a1b2c3d4"
ws_url      — the WebSocket URL to connect to
```

The widget connects to `ws_url` and implements the message protocol below. No credentials are needed — the `session_id` embedded in the URL is the capability token.

> **Important:** CSS / CS rewrites all platform URLs before forwarding frames to the widget. Use media URLs and call URLs exactly as provided — never construct platform paths or append auth parameters.

### Direct-human sessions

When CSS decides `operator_flag = "human"`, the conversation goes directly to a human agent — the AI Platform and CS are not involved. In this case:

- `session_id` in the `/api/chat/start` response is `null`
- `ws_url` points to CSS directly: `wss://css.example.com/api/chat/ws/{ticket_id}`
- The same message protocol applies: `message`, `typing`, `history`, `mode_change`, `ended` frames all work the same way
- `call_offer` frames are **not** sent in direct-human sessions (no voice handoff path)
- `escalation` frames are **not** sent (there is no AI to escalate)

The widget code does not need to branch on session type — the same handlers work for both paths.

---

## Starting a Session

Call this when the customer opens the chat drawer (lazy — not at page load):

```
POST /api/chat/start
Content-Type: application/json

{
  "operator_id": "acme",
  "user_id":     "player-42",
  "user_name":   "Rahul",
  "language":    "hi",
  "metadata":    { "page": "/withdraw", "account_tier": "vip" }
}
```

> These map from `window.SupportChat`: `operatorId` → `operator_id`, `user.id` → `user_id`, `user.name` → `user_name`, `user.language` → `language`, `user.metadata` → `metadata`. CSS resolves the tenant token from `operator_id` internally — the widget never handles a tenant token. If `user_name` is omitted, the AI greets the customer generically.

Response:
```json
{
  "session_id": "cs_a1b2c3d4",          // null when operator_flag = "human"
  "ws_url":     "wss://cs.example.com/chat/ws/cs_a1b2c3d4",
  "greeting":   "Hello Rahul, how can I help?"
}
```

For `operator_flag = "human"`: `session_id` is `null`; `ws_url` points to CSS's own WS (`wss://css.example.com/api/chat/ws/{ticket_id}`). The same message protocol applies for both paths.

---

## Connecting

```js
const ws = new WebSocket(wsUrl);
ws.onopen    = () => { /* show chat UI; render the greeting from the /api/chat/start response */ };
ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
ws.onclose   = (e) => { /* handle disconnect */ };
```

On every connect (including reconnects) the server immediately sends a `history` frame with all prior messages — use it to restore the chat log (see §History on Reconnect).

---

## Sending Messages

All outgoing frames are JSON text. Send with `ws.send(JSON.stringify(frame))`.

### Text

```json
{ "type": "message", "text": "What is my balance?" }
```

### Image

```json
{ "type": "image", "data": "<base64>", "mime": "image/jpeg", "text": "optional caption" }
```

Accepted: `image/jpeg`, `image/png`, `image/gif`, `image/webp` — **max 200 KB**

### Video

```json
{ "type": "video", "data": "<base64>", "mime": "video/mp4", "text": "optional caption" }
```

Accepted: `video/mp4`, `video/webm`, `video/quicktime` — **max 5 MB**

### Voice message

```json
{ "type": "audio", "data": "<base64>", "mime": "audio/webm;codecs=opus" }
```

Server transcribes the audio and feeds it to the AI. **Max 60 seconds** — enforce client-side. See §Voice Recording for the full flow.

### End session

```json
{ "type": "end" }
```

Cleanly closes the session. Disable the composer after sending.

---

## Receiving Messages

All incoming frames are JSON. Route by `type`:

```js
function handleMessage(msg) {
  switch (msg.type) {
    case "typing":      showTypingIndicator(); break;
    case "message":     renderAgentMessage(msg); break;
    case "audio_ack":   swapAudioUrl(msg.media_url); break;
    case "escalation":  showEscalationStatus(msg); break;
    case "mode_change": showModeChange(msg); break;
    case "call_offer":  showCallOffer(msg); break;
    case "ended":       closeChat(msg); break;
    case "error":       showInlineError(msg.message); break;
    case "history":     renderHistory(msg.messages); break;
  }
}
```

### `typing`
```json
{ "type": "typing" }
```
Show a typing indicator. Remove it when the next `message` arrives.

### `message`
```json
{
  "type": "message",
  "text": "Your balance is ₹1,200.",
  "sources": [],
  "suggestions": ["Show recent bets", "Deposit funds"],
  "action": "none"
}
```

| Field | What to do |
|---|---|
| `text` | Render as chat bubble |
| `suggestions` | Show as quick-reply chips below the bubble |
| `sources` | RAG references — omit from UI if unused |
| `action` | Behaviour hint from the AI. `"none"` = no special UI change. Other values are reserved for future use — safe to ignore unknown values. |

### `audio_ack`
```json
{ "type": "audio_ack", "media_url": "https://css.example.com/proxy/media/103" }
```
Server has uploaded the voice message. Swap the local blob URL on the `<audio>` element with the proxied URL CSS/CS already rewrote:
```js
audioEl.src = msg.media_url; // already proxied — use as-is
```

### `escalation`
```json
{
  "type": "escalation",
  "reason": "Customer requested human support",
  "summary": "Customer asked about a pending withdrawal..."
}
```
AI has escalated. Show "Connecting you to an agent…" The session mode is now `awaiting_human`.

> **Note:** This WS frame uses `summary`; the `escalation_requested` webhook CS sends to CSS also uses `summary`. Same value, two delivery channels.

### `mode_change`
```json
{ "type": "mode_change", "mode": "human", "agent_name": "Priya" }
```

| `mode` | Show |
|---|---|
| `awaiting_human` | "Waiting for an agent…" |
| `human` | "You're now chatting with Priya" |
| `voice_pending` | "Calling your number… pick up when your phone rings." |

For `human`: messages from the human agent arrive as normal `message` frames — no special handling needed.
For `voice_pending`: no audio in the browser; see pstn transport in §Voice Handoff.

### `call_offer`
```text
{
  "type":      "call_offer",
  "reason":    "Better handled on a call",
  "transport": "websocket | webrtc | pstn",
  "call_url":  "wss://...",        // webrtc: use directly; websocket: present but not used directly; pstn: absent
  "ice_servers": [{ "urls": "stun:stun.example.com" }]  // webrtc only; absent for websocket and pstn
}
```

`transport` tells you what to do. Field presence by transport:
- `call_url`: present for `webrtc` (use it directly as signalling endpoint) and `websocket` (present but not used directly — POST `/api/chat/call` instead to get the actual URL); absent for `pstn`
- `ice_servers`: present for `webrtc` only; absent for `websocket` and `pstn`

See §Voice Handoff for per-transport handling.

### `ended`
```json
{ "type": "ended", "summary": "Customer asked about withdrawal; resolved." }
```
Session closed. Disable the composer; show an end-of-chat message.

### `error`
```json
{ "type": "error", "message": "Could not transcribe voice message — please type instead." }
```
Non-fatal. The socket stays open. Show as an inline notice.

---

## Voice Recording

```js
// 1. Detect best supported MIME — Safari requires audio/mp4
const MIMES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
const mime  = MIMES.find(m => MediaRecorder.isTypeSupported(m)) || "audio/webm";

// 2. Start recording
const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
const recorder = new MediaRecorder(stream, { mimeType: mime });
const chunks   = [];

recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

recorder.onstop = () => {
  const blob = new Blob(chunks, { type: mime });

  // 3. Show local preview immediately (no server round-trip needed yet)
  const audioEl = document.createElement("audio");
  audioEl.controls = true;
  audioEl.src = URL.createObjectURL(blob);
  chatLog.appendChild(audioEl);
  pendingAudio = audioEl; // save ref for URL swap on audio_ack

  // 4. Base64-encode and send
  const reader = new FileReader();
  reader.onloadend = () => {
    ws.send(JSON.stringify({ type: "audio", data: reader.result.split(",")[1], mime }));
  };
  reader.readAsDataURL(blob);
};

// 5. Auto-stop at 60 s
recorder.start();
const autoStop = setTimeout(() => recorder.stop(), 60_000);

// Stop button: clearTimeout(autoStop); recorder.stop();

// 6. On audio_ack: pendingAudio.src = msg.media_url; // already proxied by CSS/CS — use as-is
```

---

## Media Playback

CSS/CS rewrites every `media_url` in platform frames to a CSS proxy URL before forwarding to the widget. Use the URL exactly as provided — no auth params, no AI Platform paths to construct.

```js
// Images
img.src = msg.media_url;

// Audio
audio.controls = true;
audio.src = msg.media_url;

// Video
video.controls = true;
video.src = msg.media_url;
```

---

## History on Reconnect

On every WebSocket connect the server sends one `history` frame before any other messages:

```json
{
  "type": "history",
  "messages": [
    { "id": 101, "role": "customer",    "text": "Hello",          "media_url": null,                     "media_mime": null,                    "ts": "2026-06-22T10:00:00" },
    { "id": 102, "role": "agent",       "text": "Hello Rahul…",   "media_url": null,                     "media_mime": null,                    "ts": "2026-06-22T10:00:01" },
    { "id": 103, "role": "customer",    "text": "[voice message]", "media_url": "https://css.example.com/proxy/media/103", "media_mime": "audio/webm;codecs=opus", "ts": "2026-06-22T10:01:00" },
    { "id": 104, "role": "human_agent", "text": "Let me check…",  "media_url": null,                     "media_mime": null,                    "ts": "2026-06-22T10:05:00" }
  ]
}
```

Render each entry based on `media_mime`. Media URLs in the history frame have already been rewritten by CSS/CS — use them as-is:

```js
function renderHistoryMessage(msg) {
  if (msg.media_url) {
    if (msg.media_mime?.startsWith("image/")) return `<img src="${msg.media_url}">`;
    if (msg.media_mime?.startsWith("video/")) return `<video controls src="${msg.media_url}"></video>`;
    if (msg.media_mime?.startsWith("audio/")) return `<audio controls src="${msg.media_url}"></audio>`;
  }
  return `<p>${escapeHtml(msg.text)}</p>`;
}
```

`role` values: `customer` | `agent` | `human_agent` | `system`  
`system` messages have no `media_url`; render as a muted notice (e.g. "Session transferred to agent").

---

## Chat → Voice Handoff

Handle based on `msg.transport`:

### transport: `websocket`

`call_url` is present in the frame but is not used directly — the widget must POST to CSS to get an authenticated, ephemeral call URL:

**1. Request a call URL from CSS's voice session endpoint:**
```
POST /api/chat/call
Content-Type: application/json

{ "session_id": "cs_a1b2c3d4" }
```
```json
{ "call_url": "wss://...", "call_id": "abc123" }
```
CSS calls AI Platform server-side and returns the `call_url`. The token expires in **10 minutes** — connect promptly.

**2. Connect and exchange PCM-16 audio:**
```js
const voiceWs = new WebSocket(callUrl);
voiceWs.binaryType = "arraybuffer";

// Send audio: raw PCM-16 binary frames — 16 kHz, mono, little-endian, ~20 ms chunks
// Receive binary: same PCM-16 format — pipe to AudioContext for playback
// Receive JSON text:
//   {"type":"hello"}                              — server ready
//   {"type":"transcript","role":"agent","text":…} — live captions
//   {"type":"state","state":"ended"}              — call over
```
The chat history is automatically pre-loaded into the voice agent.

### transport: `webrtc`

Use `msg.call_url` as the WebRTC signalling endpoint and `msg.ice_servers` for STUN/TURN. No additional request needed — the `call_url` and credentials come directly in the frame.

```js
const pc = new RTCPeerConnection({ iceServers: msg.ice_servers });
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
stream.getTracks().forEach(t => pc.addTrack(t, stream));

const signalingWs = new WebSocket(msg.call_url);
signalingWs.onmessage = async (e) => {
  const data = JSON.parse(e.data);
  if (data.type === "offer") {
    await pc.setRemoteDescription(data);
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    signalingWs.send(JSON.stringify(answer));
  }
  if (data.type === "candidate") {
    await pc.addIceCandidate(data.candidate);
  }
};
pc.onicecandidate = (e) => {
  if (e.candidate) signalingWs.send(JSON.stringify({ type: "candidate", candidate: e.candidate }));
};
```

### transport: `pstn`

CS is dialing the customer's phone via the voice channel adapter. Do not show a "connect" button. Show a waiting state instead:

```js
showUI("Calling your number… pick up when your phone rings.");
```

CS sends a `mode_change` frame when the call connects and an `ended` frame when it finishes. No audio is handled in the browser for this transport.

---

## Large File Upload (Alternative to Base64)

For images or videos the widget can POST multipart instead of base64-encoding over the WebSocket. Send to CSS's upload endpoint — CSS proxies to AI Platform:

```
POST /api/chat/upload
Content-Type: multipart/form-data
```

Form fields: `file` (image/* or video/*), `text` (optional caption). CSS proxies the upload and returns the AI's reply.

---

## Implementation Checklist

- [ ] Call `POST /api/chat/start` on CSS; receive `session_id` + `ws_url` + `greeting`; render `greeting` in the chat log immediately (it comes from the HTTP response, not the WS)
- [ ] `new WebSocket(wsUrl)` — server pushes `history` as the first frame on every connect
- [ ] `history` frame → restore prior messages; render media by `media_mime`
- [ ] `typing` → show indicator; next `message` → hide it + render text + chips
- [ ] Text → `{"type":"message","text":"..."}` on submit
- [ ] Image/video attach → base64 WS frame or multipart POST to CSS's `/api/chat/upload` endpoint
- [ ] Mic button → record → `audio` frame → show local blob `<audio>` immediately
- [ ] `audio_ack` → swap the pending `<audio>` element's `src` with `msg.media_url` (already proxied — use as-is)
- [ ] Media URLs: use exactly as provided by CSS/CS — never construct platform paths or append `?session_id=`
- [ ] `escalation` → show "Connecting to agent…"
- [ ] `mode_change` mode=`awaiting_human` → "Waiting for an agent…"; mode=`human` → show agent name; mode=`voice_pending` → "Calling your number…"
- [ ] `call_offer` → check `transport`: websocket = POST `/api/chat/call` → connect WS; webrtc = WebRTC peer conn using `ice_servers` + `call_url`; pstn = show "Calling your number…" and wait for `mode_change` (AI sessions only — `call_offer` is never sent in direct-human sessions)
- [ ] `ended` → disable composer
- [ ] `error` → inline notice, socket stays open
