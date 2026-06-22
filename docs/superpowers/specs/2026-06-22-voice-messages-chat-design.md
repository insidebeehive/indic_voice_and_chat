# Voice Messages in Chat — Design Spec

**Date:** 2026-06-22  
**Status:** Approved for implementation

---

## Overview

Add voice message support to the customer chat widget. Customers tap to record a voice clip, tap to send. The server uploads the clip to S3-compatible object storage, transcribes it via Gemini's batch audio API, feeds the transcript to the AI, and stores a permanent media URL. The same infrastructure also fixes image and video messages for the BO handover flow, where media previously appeared as `[image]` / `[video]` placeholders because blobs were never persisted.

---

## Scope

| Feature | Included |
|---|---|
| Customer records and sends voice messages | ✅ |
| Server transcribes via Gemini batch audio API | ✅ |
| Audio playable in customer widget, BO console, and CRM | ✅ |
| Image persistence fix (playable in BO handover history) | ✅ |
| Video persistence fix (playable in BO handover history) | ✅ |
| Video object storage architecture | ✅ (same S3 path as audio/image) |
| Live streaming transcription (Gemini Live) | ❌ out of scope |
| Object storage provider choice | Config-driven, any S3-compatible store |

---

## Architecture

### Media storage

All media (audio, image, video) is stored in an S3-compatible object store. The database stores only the object key; the server generates time-limited signed URLs on demand.

**Config (`media_storage` section):**
```yaml
media_storage:
  endpoint_url: "https://..."   # omit for AWS S3; set for R2, B2, MinIO, GCS, etc.
  access_key: "..."
  secret_key: "..."
  bucket: "chat-media"
  region: "auto"
  signed_url_ttl_seconds: 3600
```

**Object key format:**
```
chat/{tenant_id}/{session_id}/{uuid}.{ext}
```

A UUID is generated upfront (before the DB row exists) so the key is known before the upload. The key is stored in `media_url` when the `ChatMessage` row is inserted.

A new `IMediaStorage` interface (`src/providers/media/base.py`) with a single `aiobotocore`-backed `S3MediaStorage` implementation (`src/providers/media/s3.py`). Wired in `bootstrap.py`; injectable in tests via the interface.

### WebSocket protocol (no breaking changes)

The existing `image` and `video` message types are unchanged on the wire. A new `audio` type is added:

**Customer → server:**
```json
{ "type": "audio", "data": "<base64>", "mime": "audio/webm;codecs=opus" }
```

**Server → customer (acknowledgement):**
```json
{ "type": "audio_ack", "media_url": "/api/v1/chat/media/{message_id}" }
```

### Server-side handler (per media type)

When `chat_websocket` receives an `audio`, `image`, or `video` frame:

1. Decode base64 → bytes
2. Upload bytes to S3 → object key
3. Insert `ChatMessage` row (`role="customer"`, `type="{audio|image|video}"`, `content=transcript_or_caption`, `media_url=object_key`, `media_mime=mime`)
4. For `audio`: call `gemini.transcribe_audio(bytes, mime)` → transcript; pass transcript to `agent.handle_message(transcript)`
5. For `image`/`video`: existing `agent.handle_image(data, mime, caption)` call is unchanged; caption stored as `content`
6. Send `audio_ack` (audio only) with the permanent media URL

### Media endpoint

```
GET /api/v1/chat/media/{message_id}
```

- Reads `media_url` (object key) and `media_mime` from `chat_messages`
- Generates a signed S3 URL (TTL from config, default 1 hour)
- Returns `302 Location: <signed_url>` — client fetches directly from bucket

**Authentication** (either form accepted):
- `Authorization: Bearer <tenant_token>` — CRM and programmatic access
- `?session_id=<sid>` — customer widget and BO console HTML elements (`<img>`, `<audio>`, `<video>` cannot send Authorization headers; server verifies the message belongs to that session)

### DB changes

No new columns required. The existing `media_url String(500)` column on `chat_messages` is populated for all media messages (previously left NULL for image/video; new for audio).

The `type` column (`String(10)`) already fits `"audio"` (5 chars).

---

## Widget changes (`static/chat_widget.html`)

- Add 🎤 mic button next to the 📎 attach button
- Tapping starts `MediaRecorder` (captures `audio/webm;codecs=opus`)
- Button becomes red ⏹ stop button with a live elapsed timer while recording
- Auto-stops at **60 seconds**
- On stop: encode blob to base64, send `{type:"audio", data, mime}` over WS
- Immediately render `<audio controls>` bubble using `URL.createObjectURL(blob)` — customer hears playback without waiting for the server
- On `audio_ack`: swap blob URL for the permanent `/api/v1/chat/media/{id}` URL
- On history reconnect: render `<img>` / `<video controls>` / `<audio controls>` based on `media_mime` when `media_url` is present

---

## BO console changes (`static/bo_agent.html`)

The history frame already serialises all `chat_messages`. Audio/image/video messages now carry a populated `media_url`. The history renderer checks `media_mime`:

- `image/*` → `<img src="/api/v1/chat/media/{id}?session_id={sid}">`
- `video/*` → `<video controls src="/api/v1/chat/media/{id}?session_id={sid}">`
- `audio/*` → `<audio controls src="/api/v1/chat/media/{id}?session_id={sid}">`

The BO console has the current `session_id` available from the WS connection context.

No other BO console changes needed.

---

## Webhook changes

The `escalation_requested` and `session_closed` webhook payloads already include a full message transcript. Media message entries in the transcript now carry a populated `media_url`. The CRM fetches the clip using `Authorization: Bearer <api_key>`. No payload schema changes.

---

## Error handling

| Failure | Response |
|---|---|
| S3 upload fails | WS `{type:"error", message:"Could not save voice message — please try again."}`. No DB row created. |
| Transcription fails (Gemini error) | Audio is still uploaded and persisted. WS `{type:"error", message:"Could not transcribe voice message — please type your message instead."}`. BO agent can still play the clip. |
| Unsupported audio MIME | WS `{type:"error", message:"Unsupported audio format."}`. No upload. |
| Recording >60s | Enforced client-side with auto-stop. No server-side rejection needed. |
| Media endpoint — message not found | `404` |
| Media endpoint — auth fails | `401` |

---

## Testing

- **Unit:** `S3MediaStorage` with a mocked `aiobotocore` client — upload, signed URL generation
- **Unit:** `gemini.transcribe_audio()` with a mock LLM — transcript returned, error path
- **Integration:** WS `audio` frame → S3 upload → `ChatMessage` persisted with `media_url` and transcript in `content`
- **Integration:** `GET /chat/media/{id}` with bearer token → 302 to signed URL
- **Integration:** `GET /chat/media/{id}` with `?session_id=` → 302 to signed URL; wrong session_id → 401
- **Integration:** Image and video WS frames → `media_url` populated (regression: was previously NULL)
- **E2E (manual):** Record voice in customer widget → AI responds → claim as BO agent → audio playable in BO console history
