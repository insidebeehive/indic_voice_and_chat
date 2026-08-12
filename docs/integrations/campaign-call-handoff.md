# Campaign Call Handoff — CRM Integration

**Audience:** CRM/telephony engineering team
**Purpose:** Your system owns the phone call — you dial the customer through Twilio or Exotel (see [Provider support](#provider-support) for other trunks). Once the customer is on the line, our AI bridge takes over automatically, runs the conversation, delivers the outcome, and fires the same webhooks a platform-placed call gets.

This is a standalone API. It has no dependency on chat, chat sessions, or any other AI Platform feature — you never need to touch anything outside the endpoint below.

---

## Auth

The endpoint uses standard tenant Bearer auth:

```
Authorization: Bearer <tenant-token>
```

---

## Flow

1. **Register a campaign with us first**, if you haven't already:

   ```
   POST /api/v1/campaigns
   Authorization: Bearer <tenant-token>
   Content-Type: application/json

   { "name": "Q3 outreach" }
   ```

   Response `201` includes an `id` (e.g. `"camp_4a3f2b1c8d9e"`) — auto-generated unless you supply your own via `"id"` in the request. Reuse that `id` as `campaign_id` in every `register-call` request below. This is a one-time step per campaign, not per call.

2. **Register the call before dialing**, so a conversation row exists before our answer webhook fires:

   ```
   POST /api/v1/telephony/register-call
   Authorization: Bearer <tenant-token>
   Content-Type: application/json

   {
     "provider": "twilio",
     "provider_call_sid": "<sid-you-are-about-to-use>",
     "campaign_id": "camp_4a3f2b1c8d9e"
   }
   ```

   Response `201`:
   ```json
   { "call_id": "call_4a3f2b1c8d9e0f1a", "status": "in_progress" }
   ```
   Returns `200` with the same shape if the SID was already registered — safe to call idempotently, no error on retry.

   `campaign_id` must reference a campaign already registered with us via `POST /api/v1/campaigns` (step 1) — an unknown/stale `campaign_id` gets a `400`, not a silent accept.

   This step is not optional. If you skip it and go straight to placing the call, there's no conversation row for our answer webhook to attach to — you'll get no `call.initiated`, no `call.completed`, and no outcome/summary at all for that call. It fails silent, not loud.

3. **Place the call**, setting our slug-scoped answer URL as the webhook:
   ```
   https://{host}/api/v1/telephony/{provider}/voice/{tenant_slug}
   ```
   `{host}` is our public app hostname (same for every tenant — there's no per-tenant hostname). `{tenant_slug}` is your tenant's slug, given to you at onboarding.
4. Customer answers → the provider fires our answer URL → our AI bridge takes over automatically. No further API call needed.

   Note: for Exotel, the "answer" step is the provider fetching this URL to get the call-handling XML — same shape as Twilio, just Exotel's own webhook mechanics under the hood.

---

## Provider support

**Twilio and Exotel only.** Our answer-webhook + streaming-media integration exists for these two providers: a slug-scoped answer URL (`/{provider}/voice/{tenant_slug}`) that returns call-handling markup pointing at a matching media-stream WebSocket (`/{provider}/stream/{tenant_slug}`), which our AI bridge connects to. **Stringee is not supported** — it's turn-based IVR with no equivalent streaming-media handoff; its answer webhook works differently and isn't wired to the bridge the same way. If your underlying trunk is neither Twilio nor Exotel, there's no supported path today; routing through Twilio Elastic SIP Trunking (or Exotel's equivalent) so the call has a real Twilio/Exotel `call_sid` is the practical way to bridge that gap.

---

## ⚠️ `campaign_id` — accepted, but not yet functional

The request accepts `campaign_id`, which must reference a campaign already registered with us (see step 1 of [Flow](#flow)). It's stored on the call record and echoed back in the `call.initiated` webhook's `data` object — but it doesn't yet drive any behavior on the call itself:

- **`campaign_id`** does not select which campaign's script/persona/knowledge base the AI uses. Every registered call runs on the tenant's single active campaign. If a tenant has more than one active campaign, we use whichever was **created most recently** — not an error, just a silent "newest wins."

If a tenant only has one active campaign, this is invisible — everything works as expected. If you need per-call campaign selection, flag it to us before depending on it — a known gap we can prioritize fixing, not intended behavior.

---

## Errors

| Status | Meaning |
|---|---|
| 400 | Provider not `twilio`/`exotel` (includes Stringee); or `campaign_id` doesn't reference a campaign registered via `POST /api/v1/campaigns` |
| 401 | Missing or invalid `Authorization: Bearer` token |
| 403 | Token is valid but the tenant is suspended |
| 422 | Request body missing a required field (`provider`, `provider_call_sid`) |

**Stale-call safety net:** if a registered call never reaches `call.completed` (e.g. the bridge crashed or the WS dropped silently), a background job closes any call still `in_progress`/`answered` after 30 minutes, firing `call.completed` with an auto-generated note. It checks every 10 minutes, so the actual delay before you see the event is 30–40 minutes, not exactly 30.

---

## Webhooks and outcome

See `tenant-call-events.md` for the complete event/payload reference. Summary for this integration specifically:

- `call.initiated` fires immediately on registration (`data.source: "crm_register"`). **`call.answered` does not fire** — the answer-URL handler that takes over the call today doesn't emit it (this is a gap in the platform-placed-call path too, not specific to CRM registration). `call.completed` fires at teardown with the AI-classified outcome, summary, and notes.
- **No transcript in the webhook payload, and no transcript retrieval API today.** `call.completed`'s `data` carries `outcome`, `summary`, `notes`, `callback_datetime`, `status`, `duration_ms`, and `provider_call_sid` — not the turn-by-turn transcript. If you need the transcript, tell us; it's stored on our side but there's no endpoint exposing it yet.
- The AI outcome/summary pipeline is the same for every call, regardless of whether the platform placed it directly (campaign calls) or you registered it via `/register-call` — it operates on the conversation itself, not on which telephony path delivered the audio.

---

## Other things worth knowing up front

- **No duration cap.** Nothing about this mechanism limits how long a call can run — it runs until the conversation naturally ends.
- **No concurrency cap enforced on `/register-call`, but registered calls do count against the tenant's cap elsewhere.** `/register-call` doesn't reject a call for being over `max_concurrent_calls`. Platform-placed campaign calls (`POST /campaigns/{id}/calls`) DO enforce that cap and count all active calls tenant-wide — including ones you registered — so a burst of CRM registrations can cause campaign-call placement to start returning `429` even though `/register-call` itself never blocks you. If you run both campaign calls and CRM registrations on the same tenant, keep that interaction in mind, especially combined with the 30–40 minute reaper window on any registered call that never reaches `call.completed`.
- **Billing is channel-blind.** Platform cost is computed purely from mode/provider/duration — how the call reached us doesn't affect the cost calculation.
