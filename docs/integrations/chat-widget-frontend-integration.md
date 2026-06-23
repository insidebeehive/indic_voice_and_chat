# Chat Widget — Frontend Integration Guide

**Audience:** CRM Frontend team  
**Shared by:** CRM Backend team  
**Scope:** WebSocket message protocol for the AI-powered chat widget.

> **Note for CRM Frontend:** You do not interact with the AI platform directly. Your backend is the bridge — it talks to the AI platform, proxies the WebSocket connection, and handles all media. This document describes the message protocol that flows over that connection so you can build the customer-facing UI correctly.

---

## How It Fits Together

```
Customer browser (you)
        │
        │  WebSocket
        ▼
Your backend  (CRM Backend today, Coordination Service in future)
        │  proxies all platform communication
        │  handles: session creation, webhooks, media, human agent console
        ▼
AI Platform (us)
```

Your backend decides whether a conversation goes to AI or directly to a human agent. When it goes to AI, your backend creates a session with the platform and gives you two things:

```
session_id  — e.g. "cs_a1b2c3d4e5f6g7h8"
ws_url      — the WebSocket URL to connect to
```

Your job: connect to `ws_url` and implement the message protocol below. No credentials needed — the `session_id` embedded in the URL is the capability token.

> **Important:** Your backend rewrites all platform URLs before forwarding frames to you. Use media URLs and call URLs exactly as provided — never construct platform paths or append auth parameters yourself.

---

## Connecting

```js
const ws = new WebSocket(wsUrl);
ws.onopen    = () => { /* show chat UI; render the greeting your backend also gave you */ };
ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
ws.onclose   = (e) => { /* handle disconnect */ };
```

On every connect (including reconnects) the server immediately sends a `history` frame with all prior messages — use it to restore the chat log (see §History).

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

### `audio_ack`
```json
{ "type": "audio_ack", "media_url": "https://crm.example.com/proxy/media/103" }
```
Server has uploaded the voice message. Swap the local blob URL on the `<audio>` element with the proxied URL your backend already rewrote:
```js
audioEl.src = msg.media_url; // already proxied — use as-is
```

### `escalation`
```json
{
  "type": "escalation",
  "reason": "Customer requested human support",
  "context_summary": "Customer asked about a pending withdrawal..."
}
```
AI has escalated. Show "Connecting you to an agent…" The session mode is now `awaiting_human`.

### `mode_change`
```json
{ "type": "mode_change", "mode": "human", "agent_name": "Priya" }
```

| `mode` | Show |
|---|---|
| `awaiting_human` | "Waiting for an agent…" |
| `human` | "You're now chatting with Priya" |

From this point, messages from the human agent arrive as normal `message` frames — no special handling needed.

### `call_offer`
```json
{ "type": "call_offer", "reason": "Better handled on a call", "call_url": "wss://..." }
```
Show a "Switch to voice call" button. If accepted, connect to `call_url` (see §Voice Handoff).

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

// 6. On audio_ack: pendingAudio.src = msg.media_url; // already proxied by your backend
```

---

## Media Playback

Your backend rewrites every `media_url` in platform frames to its own proxy URL before forwarding to you. Use the URL exactly as provided — no auth params, no platform paths to construct.

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
    { "id": 103, "role": "customer",    "text": "[voice message]", "media_url": "/api/v1/chat/media/103", "media_mime": "audio/webm;codecs=opus", "ts": "2026-06-22T10:01:00" },
    { "id": 104, "role": "human_agent", "text": "Let me check…",  "media_url": null,                     "media_mime": null,                    "ts": "2026-06-22T10:05:00" }
  ]
}
```

Render each entry based on `media_mime`. Media URLs in the history frame have already been rewritten by your backend — use them as-is:

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

---

## Chat → Voice Handoff

When `call_offer` arrives and the customer accepts:

**1. Request a voice call URL from your backend:**

```
POST /chat/call          (your backend's endpoint — not ours)
```
```json
{ "call_url": "wss://...", "call_id": "abc123" }
```

Your backend calls our platform server-side and returns the `call_url`. The `call_url` token expires in **10 minutes** — connect promptly. The URL may point directly to us or to the Coordination Service voice endpoint depending on how your backend is configured; either way, your job is the same.

**2. Connect and exchange audio:**

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

---

## Large File Upload (Alternative to Base64)

For images or videos you can POST multipart instead of base64-encoding over the WebSocket. Send to your backend's upload endpoint — it proxies to our platform:

```
POST /chat/upload          (your backend's endpoint — not ours)
Content-Type: multipart/form-data
```

Form fields: `file` (image/* or video/*), `text` (optional caption). Your backend proxies the upload and returns the AI's reply.

---

## Quick-Start Checklist

- [ ] Receive `session_id` + `ws_url` + `greeting` from your backend — never request them from us directly
- [ ] `new WebSocket(wsUrl)` on page load; render `greeting` immediately
- [ ] `history` frame on connect → restore prior messages; render media by `media_mime`
- [ ] `typing` → show indicator; next `message` → hide it + render text + chips
- [ ] Text → `{"type":"message","text":"..."}` on submit
- [ ] Image/video attach → base64 WS frame or multipart POST to your backend's `/chat/upload`
- [ ] Mic button → record → `audio` frame → show blob `<audio>` → swap src on `audio_ack` (use `media_url` as-is)
- [ ] Media URLs: use as provided by your backend — never construct platform paths or append `?session_id=`
- [ ] `escalation` → show "Connecting to agent…"
- [ ] `mode_change` mode=`human` → show agent name
- [ ] `call_offer` → show "Switch to call" button → on accept POST to your backend's `/chat/call` → connect to returned `call_url`
- [ ] `ended` → disable composer
- [ ] `error` → inline notice, socket stays open
