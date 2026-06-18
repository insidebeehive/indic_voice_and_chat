# Tenant Call-Event Webhooks (CRM integration)

When a call starts, is answered, and ends, the platform POSTs a signed JSON
envelope to the tenant's configured **`events_webhook_url`** so the CRM receives
call lifecycle events and the final AI outcome. The same envelope is used for
both the AI **voicebot** and the human-agent **softphone** paths.

## Delivery semantics

- **Method:** `POST`
- **Content-Type:** `application/json`
- **Body:** compact JSON (no spaces), UTF-8.
- **Retries:** up to **3 attempts** with exponential backoff (~0.3s, 0.6s), 5s
  timeout each. Delivery is fire-and-forget — a slow/failing CRM endpoint never
  blocks or breaks call handling.
- **Expected response:** any **2xx**. Non-2xx (or no response) is retried, then
  dropped after the budget.
- **Ordering / idempotency:** events are independent and may arrive out of order
  or be retried. De-duplicate on **`event_id`** (unique per event).

## Signing (optional but recommended)

If a signing secret is configured, each request carries:

```
X-Signature: sha256=<hex>
```

`<hex>` = `HMAC-SHA256(secret, raw_request_body)` over the **exact bytes** posted.
Verify by recomputing the HMAC over the raw body and constant-time comparing. If
no secret is configured, the header is absent and the body is sent unsigned.

```python
import hmac, hashlib
def verify(secret: str, raw_body: bytes, header: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header or "")
```

## Common envelope

Every event has this shape:

```json
{
  "event_type": "call.initiated",
  "event_id": "9f8c1e2a4b5d4f6a8c0e1d2f3a4b5c6d",
  "call_id": "CA_or_internal_call_id",
  "tenant_id": "t_338c325735ed44dc",
  "channel": "voicebot",
  "occurred_at": "2026-06-18T08:30:12.345678+00:00",
  "data": { }
}
```

| Field | Type | Notes |
|---|---|---|
| `event_type` | string | `call.initiated` \| `call.answered` \| `call.completed` |
| `event_id` | string | Unique per event (uuid hex) — use for de-dup |
| `call_id` | string | Our call identifier (stable across the call's events) |
| `tenant_id` | string | Platform tenant id |
| `channel` | string | `voicebot` (AI) or `softphone` (human agent) |
| `occurred_at` | string | ISO-8601 timestamp, UTC |
| `data` | object | Event-specific, see below |

## Events

### `call.initiated` — call row created / dialing
```json
"data": {
  "provider_call_sid": "<telephony provider call id>",
  "mode": "layered",
  "campaign_id": "c_123",
  "lead_id": "l_456"
}
```
- `mode`: `layered` (STT→LLM→TTS cascade) or `s2s` (speech-to-speech).
- `campaign_id` / `lead_id`: `null` for ad-hoc/dev-console calls.

### `call.answered` — callee picked up
```json
"data": {
  "provider_call_sid": "<telephony provider call id>"
}
```
> Not guaranteed on every provider path; treat `call.completed` as the
> authoritative terminal signal.

### `call.completed` — terminal; carries the AI outcome
```json
"data": {
  "outcome": "callback_requested",
  "summary": "Lead asked to be called back tomorrow evening.",
  "notes": "Mentioned competitor pricing; prefers Hindi.",
  "callback_datetime": "2026-06-19T13:30:00+00:00",
  "status": "ended",
  "duration_ms": 64000,
  "provider_call_sid": "<telephony provider call id>"
}
```

| Field | Type | Notes |
|---|---|---|
| `outcome` | string \| null | Classified outcome — see enum below |
| `summary` | string \| null | One-line AI summary of the call |
| `notes` | string \| null | Free-text AI notes / salient details |
| `callback_datetime` | string \| null | ISO-8601 if the lead requested a callback, else `null` |
| `status` | string | Terminal call status (typically `ended`) |
| `duration_ms` | integer \| null | Call duration in **milliseconds** |
| `provider_call_sid` | string | Telephony provider call id |

#### `outcome` values
```
interested · callback_requested · not_interested · refused ·
escalated · angry_hostile · no_answer · voicemail · busy · call_failed
```
The first six are AI-classified from the conversation; the last four are derived
from telephony status when the call never connected.

## Notes
- Match events to a call via `call_id` (stable) or `data.provider_call_sid`.
- `call.completed` is the end-call signal — it's emitted for both voicebot and
  softphone calls once outcome analysis finishes.
