# CRM Integration — Chat Media Contract

The definitive contract for sending customer messages and attachments
(image / video / audio / documents) from the CRM relay to the AI Platform
chat. Share this with the CRM team; it supersedes any earlier assumptions.

**Status:** text, image, video, audio are LIVE. `document` (PDF/Word/Excel/
CSV/txt) is designed and scheduled — the frame shape below is final, but the
platform rejects it until the feature ships. Everything else applies today.

## 1. Connect

1. `POST /api/v1/chat/sessions` (tenant bearer token) →
   `{session_id, greeting, ws_url}`.
2. Connect to `WS /api/v1/chat/ws/{session_id}`. The `session_id` is the
   capability — no credentials go over the socket.

### Language preference

`POST /api/v1/chat/sessions` takes an optional `language` field:

```json
{"user_id": "...", "customer_name": "...", "language": "hi", "metadata": {}}
```

- **Format:** a short ISO-639 base code, lowercase, 2-3 letters — `hi`, `mr`,
  `ta`, `en`, etc. **Not** a full name: `"Hindi"` fails validation silently
  and falls back to the tenant default, it does NOT error. A region suffix
  is also fine (`"hi-IN"` — the region is stripped before use).
- **Omit the field** to use the tenant's configured default language.
- This only *seeds* the session — the AI still shifts language per turn to
  match what the customer actually types/says.
- The `greeting` returned by `POST /chat/sessions` is rendered in this
  language, so the customer's very first line is already in their preference —
  no English-then-switch. Languages outside the table below fall back to an
  English greeting (the session language itself is still honored by the AI
  from the first reply onward).

Text chat itself accepts any 2-3 letter code — there's no restricted list.
The restriction only bites if/when the session escalates to a voice call,
because each speech provider supports a different language subset:

| Language | Code | Sarvam TTS | IndicF5 TTS | S2S (Gemini Live) | ElevenLabs TTS |
|---|---|---|---|---|---|
| Hindi | `hi` | ✓ | ✓ | auto | auto |
| English | `en` | ✓ | – | auto | auto |
| Bengali | `bn` | ✓ | ✓ | auto | auto |
| Gujarati | `gu` | ✓ | ✓ | auto | auto |
| Kannada | `kn` | ✓ | ✓ | auto | auto |
| Malayalam | `ml` | ✓ | ✓ | auto | auto |
| Marathi | `mr` | ✓ | ✓ | auto | auto |
| Odia | `od` | ✓ | ✓ | auto | auto |
| Punjabi | `pa` | ✓ | ✓ | auto | auto |
| Tamil | `ta` | ✓ | ✓ | auto | auto |
| Telugu | `te` | ✓ | ✓ | auto | auto |
| Assamese | `as` | – | ✓ | auto | auto |

- **Sarvam / IndicF5** are the only providers with a real, enforced language
  list (11 codes each, shown above). They take the code as `xx-IN` internally
  (e.g. `hi-IN`); IndicF5 converts it to the bare 2-letter form for its own
  wire call. **Note the non-standard code: Odia is `od`, not the ISO-639-1
  `or`** — sending `or` passes shape validation but won't match either
  provider's voice catalog.
- **S2S (Gemini Live)** doesn't take an explicit language code on the current
  native-audio model — it auto-detects and switches language from the
  conversation itself; `language` only steers the initial system prompt.
- **ElevenLabs** has no language parameter at all — its multilingual models
  infer language from the text being spoken.

## 2. Client → server frames

All frames are JSON text frames.

### Text message

```json
{"type": "message", "text": "customer's words"}
```

- `text` must be non-empty.
- **Never put `media_url` on a `type:"message"` frame — it is silently
  ignored.** Attachments MUST use the typed media frames below (this was the
  root cause of the "missing 'text'" / dropped-attachment bug).

### Media frames (image / video / audio / document)

```json
{
  "type": "image" | "video" | "audio" | "document",
  "data": "<base64>",            // EITHER inline bytes...
  "media_url": "https://...",    // ...OR a URL the platform fetches
  "mime": "image/jpeg",          // see mime rules below
  "filename": "bill.pdf",        // document frames only
  "text": "optional caption"     // image/video/document only
}
```

Rules:

- **`data` or `media_url` — exactly one is required.**
- **`media_url` requirements:** `https://` only; publicly reachable host (no
  private/internal IPs); redirects are NOT followed; response body max
  **1 MB**; the response `Content-Type` must match the frame type's family
  (see table). Presigned S3/R2 URLs work well — make sure they haven't
  expired and are single-hop (no redirect).
- **`mime`:** required when sending `data`. Optional with `media_url` — the
  platform uses the fetch response's Content-Type when omitted.
- **`text` caption:** optional on image/video/document; ignored on audio
  (the voice recording itself is transcribed). An attachment with no caption
  is fine — do NOT send an empty `type:"message"` frame alongside it.
- **Size limit: 1 MB per file** (both inline base64 after decode and URL
  fetch). Enforce this CRM-side too for a better customer error.

Accepted content types per frame type:

| Frame type | Accepted mimes | Notes |
|---|---|---|
| `image` | `image/*` (jpeg, png, webp, ...) | AI describes/answers about the image |
| `video` | `video/*` | key frames analyzed |
| `audio` | `audio/*` (webm/opus, ogg, mp3, wav, ...) | transcribed, then answered like text; server sends an `audio_ack` frame with a playback URL |
| `document` *(upcoming)* | `application/pdf`, `.docx` (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`), `.xlsx` (`...spreadsheetml.sheet`), `text/csv`, `text/plain`, `text/markdown` | AI answers questions about the document in-session. Legacy `.doc`, `.rtf`, archives → rejected |

### End session

```json
{"type": "end"}
```

## 3. Server → client frames

| Frame | Shape | Meaning |
|---|---|---|
| `typing` | `{"type":"typing"}` | turn accepted, reply coming |
| `message` | `{"type":"message","session_id":...,"text":...,"sources":[...],"suggestions":[...],"action":...}` | the AI reply |
| `audio_ack` | `{"type":"audio_ack","media_url":"/api/v1/chat/media/<id>"}` | voice note stored; URL serves the recording for transcript UIs |
| `escalation` | `{"type":"escalation","reason":...,"context_summary":...}` | conversation escalated to a human |
| `call_offer` | `{"type":"call_offer","reason":...,"call_url":...}` | AI offered a voice call; `call_url` is the WS the browser dials |
| `ended` | `{"type":"ended","summary":...,"reason":"customer_ended"\|"idle_timeout"}` | session closed |
| `error` | `{"type":"error","message":...,"reason":...}` | that turn failed; **the socket stays open** — show the message, let the customer retry |

An `error` frame never closes the socket. Treat it as per-message failure,
not a connection failure. `reason` is machine-readable, for relays that want
to act on the failure kind rather than just display `message`:

| `reason` | Meaning |
|---|---|
| `llm_billing` | provider's monthly spending cap was hit — won't clear on its own, needs a human to raise the cap |
| `llm_quota` | ordinary rate/quota exhaustion — transient, likely to clear shortly |
| `timeout` | the turn exceeded the processing time budget |
| `internal` | anything else |

Relays SHOULD surface `error` frames to the customer — dropping them is
what makes the bot look unresponsive.

## 4. REST alternative for attachments

If a WS frame is awkward for large-ish files, use multipart REST instead —
same processing, synchronous JSON reply (the AI's answer):

```
POST /api/v1/chat/{session_id}/upload
Content-Type: multipart/form-data
  file: <the attachment>        (required)
  text: <optional caption>      (optional)
```

No auth header needed — the `session_id` is the capability. Same 1 MB and
content-type rules as the WS frames.

On a provider failure (billing cap, quota, timeout), this endpoint and
`POST /chat/message` return HTTP 503 (`llm_billing`/`llm_quota`) or 504
(`timeout`) with body `{"detail":{"message":...,"reason":...}}` — the same
reason codes as the WS `error` frames above, never a bare 500.

## 5. Quick reference — what to send when

| Customer action | Send |
|---|---|
| Types a message | `{"type":"message","text":...}` |
| Sends a photo, with or without caption | `{"type":"image", media_url or data, "text": caption or ""}` |
| Sends a video | `{"type":"video", ...}` same shape |
| Sends a voice note | `{"type":"audio", media_url or data}` (+ `mime` if inline) |
| Sends a PDF/Word/Excel/CSV/txt *(once shipped)* | `{"type":"document", media_url or data, "filename":..., "text": caption or ""}` |
| Leaves / closes chat | `{"type":"end"}` |

Common mistakes to avoid:
- Sending attachments on `type:"message"` (ignored) or with empty `text`
  (rejected).
- HTTP (non-https) or private-network `media_url`s (rejected).
- Presigned URLs that redirect (rejected — the fetch does not follow them).
- Files over 1 MB (rejected).
