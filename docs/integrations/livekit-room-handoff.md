# LiveKit Room Handoff — CRM Integration

**Audience:** CRM/telephony infra engineering
**Purpose:** You already run your own SIP/PSTN telephony infrastructure and front it with **LiveKit SIP**, so an inbound or outbound PSTN call becomes a **LiveKit room** with a SIP participant in it. You tell us, via a webhook, when that SIP leg is live. We join that room as a participant and run the AI conversation — listening to the caller's audio track and publishing our agent's speech back as an audio track.

This is a different mechanism from [`campaign-call-handoff.md`](campaign-call-handoff.md), which covers dialing directly through a Twilio or Exotel account (our answer-URL + media-stream integration). Both docs describe real, live integrations — use this one if LiveKit SIP is how your calls reach us; use the other if you dial straight through Twilio/Exotel.

---

## Auth

The one optional REST call this doc covers (`POST /register-call`, see [Flow](#flow)) uses standard tenant Bearer auth:

```
Authorization: Bearer <tenant-token>
```

The LiveKit webhook we receive from *you* is authenticated differently — see [What we need from you](#what-we-need-from-you) and [Errors](#errors).

---

## What we need from you

**This is a one-time, CRM-level registration — not something you configure per tenant.** A LiveKit project belongs to you (the CRM partner) as a whole, not to any individual tenant/operator registered under you: you give us one LiveKit URL/key/secret once, during onboarding, and every tenant we register against your CRM automatically uses it. You don't self-service this via any API call — `/crms` (where it's stored) is admin-only on our side — you hand it to us and we set it up.

1. **LiveKit server URL** (`wss://...`) — your LiveKit project, self-hosted or LiveKit Cloud.
2. **API Key**
3. **API Secret**

Items 2 and 3 are a **separate credential pair from any business-API key/secret** you may already exchange with us for CRM tool-calling (player lookups, etc.) — those are unrelated systems. The LiveKit API key/secret is used only to mint room-join tokens and to verify the authenticity of the webhook described below; it has nothing to do with your CRM tool credentials.

(In the rare case a specific tenant genuinely needs its own separate LiveKit project — e.g. testing against a sandbox project ahead of the shared one being ready — that's supported as a per-tenant override on top of the CRM-level default, but it is the exception, not the norm, and is set up on our side the same way.)

4. **The webhook URL to configure on your LiveKit project:**

   ```
   https://{host}/api/v1/telephony/livekit/webhook/{tenant_slug}
   ```

   `{host}` is our public app hostname (same for every tenant — there's no per-tenant hostname). `{tenant_slug}` is your tenant's slug, given to you at onboarding. Configure your LiveKit project to send webhooks (at minimum `participant_joined`) to this URL.

5. **Confirmation you can set room and/or participant metadata** (or attributes) on `CreateSIPParticipant` or your equivalent SIP-dispatch call — this is how you tell us which campaign, voice, and lead a given room/call is for. See [The `vox` metadata schema](#the-vox-metadata-schema).

6. **A sandbox LiveKit project** for initial testing, separate from your production project.

---

## The `vox` metadata schema

When you create the room and/or SIP participant, set a JSON object under a top-level `"vox"` key. **Room metadata is preferred; participant metadata is a fallback; flat `vox.`-prefixed participant attributes are the last-resort fallback** — this is the exact precedence our webhook route checks, in this order:

1. `room.metadata` — parsed as JSON, we read its top-level `"vox"` object.
2. `participant.metadata` — same, if room metadata had no usable `"vox"` object.
3. Flat participant **attributes** (a string-only map) prefixed `vox.` — e.g. `vox.campaign_id`, `vox.voice`, `vox.lead_name`, `vox.lead_gender`, `vox.call_ref`. Because attributes can't carry a nested object, `lead` is flattened here into `lead_name`/`lead_gender` and reassembled into `lead: {name, gender}` on our side.

Shape (room or participant metadata JSON):

```json
{
  "vox": {
    "campaign_id": "camp_4a3f2b1c8d9e",
    "voice": "hi-IN-female-1",
    "lead": { "name": "Rohit Sharma", "gender": "male" },
    "call_ref": "your-own-call-reference-id"
  }
}
```

All fields are optional:

- **`campaign_id`** — selects the campaign's script/persona/knowledge base for this call. Omit it to use the tenant's default campaign resolution.
- **`voice`** — a per-call voice override (validated against the tenant's allowed voices; falls back to the tenant's configured voice if not allowed).
- **`lead.name` / `lead.gender`** — personalizes the agent's opening/dialogue for this specific lead.
- **`call_ref`** — not consumed by the agent; it exists purely so you can correlate this call with your own records. It is *not* currently echoed back into our outbound webhooks — see [Webhooks and outcome](#webhooks-and-outcome).

**Pipeline mode and STT/LLM/TTS provider selection are tenant-level, not per-call — do not send a `mode` field, it will be ignored.** If a `mode` key is present, the bridge factory logs a warning and drops it; there is no per-call mode override for the LiveKit path (unlike our dev console for Twilio/Exotel). LiveKit room-join only works at all when the tenant is configured for `pipeline.mode == "s2s"` — see [Errors](#errors).

---

## Flow

1. **Optionally register the campaign with us**, if you want per-call campaign selection and haven't already:

   ```
   POST /api/v1/campaigns
   Authorization: Bearer <tenant-token>
   Content-Type: application/json

   { "name": "Q3 outreach" }
   ```

   Response `201` includes an `id` (e.g. `"camp_4a3f2b1c8d9e"`). Use that as `vox.campaign_id` in room/participant metadata (step 3). This is a one-time step per campaign, not per call — same as the Twilio/Exotel doc.

2. **Optionally pre-register the call**, if you want `call.initiated` to fire at dial time instead of at room-join time:

   ```
   POST /api/v1/telephony/register-call
   Authorization: Bearer <tenant-token>
   Content-Type: application/json

   {
     "provider": "livekit",
     "provider_call_sid": "<the exact LiveKit room name you are about to create>",
     "campaign_id": "camp_4a3f2b1c8d9e"
   }
   ```

   This step is genuinely optional, unlike the Twilio/Exotel flow — if you skip it, the room-join webhook auto-creates the conversation row for you with zero pre-registration needed (`call.initiated` firing late, at room-join time instead of dial time). Pre-registering buys you two things the auto-create path can't: `call.initiated` at dial time, and validation of an unknown `campaign_id` up front (`400`) instead of a silent fallback to the tenant's default campaign. `provider_call_sid` **must equal the exact room name** you create in step 3 — that's the join key between this row and the room.

   **Important:** the `campaign_id` you send here only tags the pre-created row for bookkeeping — it does **not** by itself select which campaign the agent actually runs. The campaign actually used for the conversation is whatever `campaign_id` you put in the room/participant's `vox` metadata in step 3. If you pre-register, make sure the two `campaign_id` values match — otherwise the row's recorded campaign (used in reporting) can diverge from the campaign that actually ran the call. If you skip pre-registration entirely, this can't happen: the auto-created row is stamped with the same `vox.campaign_id` that drove the call.

3. **Bring up your SIP call and create the LiveKit room + SIP participant**, setting the `vox` metadata (room metadata preferred) as described above.

4. **Our webhook fires** when the SIP participant joins the room. We verify the delivery, join the room ourselves, subscribe to the caller's audio track, publish our agent's audio track, and run the conversation.

5. **The call ends** — either side can hang up. If we end it, we delete the LiveKit room server-side (confirmed, live-tested) so your SIP leg actually terminates rather than being left connected to silence. `call.completed` fires with the AI-classified outcome, summary, and notes.

---

## Errors

| Condition | Behavior |
|---|---|
| Unknown tenant slug in the webhook URL | `404` |
| Missing/invalid webhook signature, missing auth token, non-UTF-8 body, or no LiveKit credentials configured for the tenant | `401` with an identical generic body in every case — the specific reason is only in our logs, never leaked to the sender |
| Tenant not configured for `pipeline.mode == "s2s"` | Webhook still returns `200` (we always ack fast); no agent joins the room, but a conversation row is created for the room-join attempt and `call.completed` fires with `outcome: "call_failed"` and a notes field explaining why — as long as the room connection itself succeeded (see the next row for the one remaining silent case). |
| Tenant already at `max_concurrent_calls` | `200`, but no agent joins. This can't be a rejection status — by the time the webhook fires, the SIP leg is already up and the caller is already in the room; a non-2xx here only makes LiveKit retry the same over-cap delivery. We decline to join and the room sits agent-less until the caller hangs up. No conversation row is created for this case (we never attempted the room). |
| No SIP participant ever joins the room | We wait up to **60 seconds** (`_JOIN_TIMEOUT_S`) for the caller's audio track to be subscribed; if it never is, we give up and never build an agent for that room. A conversation row is created and `call.completed` fires with `outcome: "no_answer"` and an explanatory note — this is no longer a silent failure. Plan your dispatch flow so the SIP participant reliably joins within that window regardless. |
| We fail to connect to the LiveKit room itself (bad `livekit_url`/credentials, network failure) | The one remaining case with **zero CRM-visible signal** — there is no conversation row yet to attach a failure to at this point in the lifecycle. This is logged loudly on our side. Flag this to us if you need it made visible before depending on it. |

---

## Webhooks and outcome

Same event set and payload shape as `campaign-call-handoff.md` / `tenant-call-events.md` — see that doc for the full payload reference. Notes specific to this path:

- **`call.initiated`** fires either at `/register-call` time (if you pre-registered) or at room-join time (if you didn't) — see [Flow](#flow) step 2.
- **`call.answered`** fires for **every** call whose caller audio track actually gets subscribed — regardless of whether you pre-registered. This differs from the Twilio/Exotel path, where `call.answered` currently never fires at all.
- **`call.completed`** fires at teardown with the AI-classified outcome, summary, and notes — same as every other provider.
- **No transcript in the webhook payload, and no transcript retrieval API today.** `call.completed`'s `data` carries `outcome`, `summary`, `notes`, `callback_datetime`, `status`, `duration_ms`, and `provider_call_sid` — not the turn-by-turn transcript.

---

## Other things worth knowing

- **No per-call mode/provider override.** Pipeline mode (s2s vs. layered) and STT/LLM/TTS provider selection are tenant-level configuration only — see [The `vox` metadata schema](#the-vox-metadata-schema).
- **Concurrency cap is shared with your other calls on the tenant.** `max_concurrent_calls` counts every active call tenant-wide (`in_progress`/`answered` status), regardless of whether it arrived via LiveKit room-join, a campaign call, or a CRM-registered Twilio/Exotel call — a burst on one channel can cause another channel to hit the cap.
- **Billing is channel-blind.** Platform cost is computed purely from mode/provider/duration — how the call reached us doesn't affect the cost calculation. Your own SIP trunk cost isn't billed by us; the LiveKit telephony leg is currently seeded as a tentative `$0.00/min` placeholder in our internal cost catalog (same principle as every other telephony provider: your trunk, your cost, tracked only for our own reference).
- **Transfer-to-human is not currently supported on this transport.** When the AI's dialogue decides the call should be transferred to a human agent, on Twilio/Exotel we hold the call and notify a coordination service to find a human. On LiveKit room-join, that hold/notify step is not implemented yet — a transfer decision simply ends the call (the room is deleted) rather than being held for a human agent to pick up. This is a known, deliberate gap (not a silent bug) and may be addressed in a future round if you need it.

---

## Summary

This document is the direct answer to "would you consider native LiveKit-room ingestion" — yes, and it's built and live-verified: no browser-widget-mechanism workaround is needed. You bring up the room and SIP participant on your side; we join it directly.
