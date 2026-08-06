# LiveKit SIP bridge (CRM outbound integration) — design doc

> Status: **implemented and live-verified against a real LiveKit server.** The bridge
> (`src/api/livekit_bridge.py`), the SDK-wiring runner (`src/api/livekit_runner.py`), the
> webhook route (`src/api/livekit_routes.py`), and the per-call factory
> (`src/bootstrap.py`'s `make_livekit_bridge_factory`) all exist and are wired in
> `src/main.py`. The CRM-facing artifact for this integration is
> [`docs/integrations/livekit-room-handoff.md`](integrations/livekit-room-handoff.md) — use
> that doc for onboarding a CRM partner; this doc remains as the original design rationale,
> with the sections below updated to reflect what was actually built where it diverged from
> the original recommendation.

## Context

A CRM customer's infra team is standing up their own **outbound SIP gateway**, and has
confirmed they will front it with **LiveKit SIP** — LiveKit's SIP service acts as their SIP
client (UAC), converting SIP/RTP PSTN calls into a **LiveKit room** (a WebRTC audio/video
room). Scope is **outbound only**: the CRM places the call to the end customer through their
own SIP trunk; our job starts once that call exists as a LiveKit room.

This is a meaningfully different shape from the DiDLogic plan
(`docs/sip-didlogic-integration-plan.md`), where *we* would have needed to front a raw SIP
trunk with a self-hosted media gateway (Jambonz) because nothing on our side speaks SIP. Here,
the CRM's infra team owns the SIP↔realtime-media conversion end-to-end via LiveKit SIP — a
mature, purpose-built service for exactly this job. **We never touch SIP/RTP and do not need
to host any telecom infrastructure.** Our job is simply to join a WebRTC room as a
participant, listen to the caller's audio track, and publish our AI agent's speech back as an
audio track — the same shape of work as our existing Twilio/Exotel WebSocket audio bridge,
just over LiveKit's RTC SDK instead of a raw WebSocket.

## Recommended approach: a LiveKit RTC bridge, reusing the existing dialogue core

Add a new transport that follows the same pattern as the existing Twilio/Exotel bridge:

- **`_BaseLiveBridge`** (`src/api/live_bridge_base.py`) is the transport-agnostic dialogue
  driver — Gemini Live session lifecycle, barge-in, turn commit, outcome/cost persistence.
  Existing subclasses (`TelephonyLiveBridge` in `src/api/telephony_live_bridge.py`) supply
  only the transport hooks: `_inbound_loop`, `_send_audio_out`, `_send_interrupt`, `_on_start`,
  `_on_teardown`, `_emit_status/_emit_transcript/_deliver_outcome`.
- A new **`LiveKitBridge`** subclass supplies those same hooks, backed by the LiveKit RTC SDK:
  join the room → subscribe to the caller's published audio track → feed frames into the
  dialogue core exactly like inbound telephony audio today → publish synthesized TTS audio
  back as an outgoing track.
- Everything downstream of the transport hooks — the dialogue engine, barge-in, outcome
  analysis, cost tracking — is reused untouched.

### Key nuance vs. existing telephony providers

Every existing `ITelephonyProvider` adapter (`src/interfaces/telephony.py`,
`src/providers/telephony/*.py`) is built around *us* originating or controlling the call:
`initiate_call`, `hangup`, `transfer`. Here, the CRM's LiveKit SIP trunk originates the
outbound PSTN leg — we never call `initiate_call`. Our role is **"join a room when told to,"**
not "dial out." This is a thinner integration than a REST-adapter provider like
`telnyx.py`/a hypothetical `jambonz.py` would have been — there's no REST call-control surface
on our side at all, just a room-join.

### Audio format

WebRTC rooms default to **48kHz Opus**, versus the 8kHz μ-law/PCM telephony audio the pipeline
handles today for Twilio/Exotel. This needs an extra resampling step on both the inbound and
outbound path, using the existing resampling utility (`src/pipeline/audio_utils.py`) already
shared across providers with different native sample rates.

## What we need from the CRM's infra team

**Connection & access**
1. LiveKit server URL (`wss://...`) — self-hosted or LiveKit Cloud.
2. API key + secret, or a mechanism for minting/receiving room-join tokens for our agent.
3. Confirmation of the LiveKit SDK/version they're standardizing on.

**How we know when to join**
4. **Decided and built: a webhook, not Agent Dispatch.** This doc originally preferred
   LiveKit's native **Agent Dispatch** feature (LiveKit's built-in mechanism for assigning a
   specific agent to a specific room on demand) over a custom webhook, on the reasoning that
   Agent Dispatch would avoid standing up and securing our own webhook receiver. The actual
   implementation reverses that: the CRM configures a tenant-scoped LiveKit webhook
   (`POST /api/v1/telephony/livekit/webhook/{tenant_slug}`, `src/api/livekit_routes.py`) that
   fires on `participant_joined` for the SIP participant, and that route spawns
   `livekit_runner.run_call` in the background. The deciding factor: this codebase runs as a
   single-process uvicorn deployment, which doesn't fit Agent Dispatch's worker/subprocess
   model cleanly, whereas a webhook reuses infrastructure this codebase already runs safely
   for four other telephony providers (Twilio, Exotel, Stringee, and the CRM-registration
   path). See `docs/integrations/livekit-room-handoff.md` for the CRM-facing webhook contract
   (signature verification, event filtering, concurrency cap, dedupe).
5. Room/participant naming or metadata convention — how we identify which tenant, which call,
   and which caller/callee number a given room corresponds to. Built as a `vox`-namespaced
   metadata schema (room metadata preferred, participant metadata/attributes as fallback) —
   see `docs/integrations/livekit-room-handoff.md`'s metadata schema section.

**Audio**
6. Confirmation of codec/sample rate in the room (Opus @48kHz is the WebRTC default).

**Call lifecycle**
7. How we learn a call ended — a room-finished/participant-disconnected event, either via
   their webhook or by us subscribing to LiveKit's own webhooks directly.

**Operational**
8. A test/sandbox LiveKit project, plus a way to place a test call end-to-end.
9. Expected concurrent call volume (capacity planning).
10. Token scoping/security — our agent's access token must only be able to join calls meant
    for it, not arbitrary rooms.

**Business / compliance**
11. Caller ID / DID presented to the destination number.
12. Call-recording consent — who owns the disclosure responsibility (relevant given India
    TRAI-adjacent regulatory considerations for some tenants).
13. CDRs (call detail records) for billing reconciliation — telephony cost is currently
    tracked per-tenant as a **tentative** figure in the backoffice (the tenant's own trunk,
    not billed by us); the same model would apply here.
14. Point of contact for their infra team, and an SLA/uptime expectation once this is live for
    real traffic.

## Open questions to confirm with the CRM before implementation starts

- Agent Dispatch vs. a custom webhook for join notification (item 4 above) — this materially
  changes what we build on our side (a LiveKit worker registered for dispatch, vs. a webhook
  receiver + manual room join).
- Self-hosted LiveKit vs. LiveKit Cloud on their side — doesn't change our integration shape,
  but affects latency/reliability expectations worth knowing up front.
- Final confirmation on codec/sample rate (item 6) before committing to a specific resampling
  path.

## Effort / risk (once the above is confirmed)

- **App code**: moderate — one new `_BaseLiveBridge` subclass (`LiveKitBridge`) plus a new
  LiveKit SDK dependency, tenant config for LiveKit connection details (fits the existing
  generic per-tenant `telephony.keys` → encrypted `tenant_secrets` mechanism in
  `src/config_tenant.py` / `src/models/tenant.py`, no schema change expected), a dispatch/
  join-trigger handler, and a dev-console option to test it. Most of the dialogue/barge-in/
  outcome code is reused unchanged.
- **Infra/ops**: minimal on our side — no self-hosted telecom component, unlike the DiDLogic/
  Jambonz path. The CRM's infra team owns the SIP↔WebRTC conversion.
- **Risk**: primarily audio-quality/resampling correctness (48kHz Opus ↔ our model's expected
  rate) and getting the join-trigger mechanism (Agent Dispatch vs. webhook) right — both are
  contained, well-scoped risks compared to operating our own SIP/RTP stack.

## Verification (when implemented)

- Manual: join a LiveKit test room with a non-AI participant (e.g. the LiveKit sample web
  client) and confirm the bridge can subscribe to and publish audio tracks correctly.
- Adapter/bridge unit test: feed synthetic LiveKit audio frames; assert audio reaches the
  agent and outbound frames are published.
- E2E (staging): CRM places a real test call through their SIP trunk into a LiveKit room;
  confirm our agent joins, converses, and a `conversations` row is recorded with outcome and
  cost data as with existing telephony providers.
