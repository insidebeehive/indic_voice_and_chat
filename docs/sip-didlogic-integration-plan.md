# SIP trunk (DiDLogic) outbound integration

> Status: **app-side implemented (pure-Python in-app transport), pending live
> validation.** Scope: **outbound only**, provider **DiDLogic**.
>
> Implemented (this PR): a SIP transport seam (`src/providers/telephony/sip/transport.py`),
> a pyVoIP-backed transport (`.../sip/pyvoip_call.py`, **UNVERIFIED — needs live
> DiDLogic creds**), a `SipMediaBridge` (`src/api/sip_media_bridge.py`) running the
> S2S agent over RTP, a `make_sip_bridge_factory` (bootstrap) wired in the lifespan,
> and a Call Lead path for telephony provider `didlogic` (places the INVITE, runs
> the agent as a background task, persists outcome + cost). `telephony.sip_server`
> added to the tenant config; SIP user/pass reuse the encrypted
> `account_sid`/`auth_token` secrets. Cost catalog + Admin UI include `didlogic`.
> pyVoIP is an optional extra (`pip install -e .[sip]`).
>
> Remaining: install pyVoIP in the deploy image, register a DiDLogic tenant with
> SIP creds, and live-test an outbound call (validate the pyVoIP API/RTP against
> the installed version). The gateway alternative below stays the production-grade
> option if the in-app pure-Python path proves too fragile.

---
## Original plan (gateway approach — kept for reference)

## Context
We want to place **outbound** AI-agent calls through **DiDLogic**, a wholesale **SIP trunk +
DID** provider. DiDLogic gives SIP credentials (user/pass/host or IP-auth) and numbers, and
carries calls as **SIP signaling + RTP media**. It has **no programmable-voice API, no answer
webhook, and no Media-Streams WebSocket**.

Our platform's telephony model is the opposite: **CPaaS REST control-plane + media-over-
WebSocket** (Twilio Media Streams shape). Adapters in `src/providers/telephony/*.py` only do
REST `initiate_call`/`hangup`/`transfer`; real audio flows through route-layer bridges
(`src/api/telephony_live_bridge.py` over `src/api/live_bridge_base.py`) as base64 μ-law/PCM
@ 8 kHz inside JSON over a WebSocket. A repo-wide search confirms **zero SIP/RTP/SDP code**.

**So DiDLogic is NOT equivalent to Twilio.** There is no REST dial, no TwiML, no WS media to
hook into. Something must terminate SIP/RTP and re-present the call the way our bridges expect.
The cleanest way — keeping ~all existing audio plumbing — is to **front DiDLogic with a SIP
media gateway** that exposes a webhook + a bidirectional WebSocket audio stream.

## Recommended approach: a SIP media gateway (Jambonz) in front of DiDLogic
Deploy **Jambonz** (open-source, self-hosted CPaaS built for bring-your-own SIP trunks + AI).
Configure DiDLogic as an outbound **carrier** (SIP creds). Jambonz re-presents DiDLogic as a
Twilio-like provider: place calls via its REST API, control them with webhook "verbs", and fork
call audio bidirectionally to our WebSocket. Jambonz owns the hard parts — SIP INVITE/SDP/BYE,
RTP, codec (PCMU 8 kHz), NAT/SBC, DTMF, jitter. The app-side then becomes close to
Twilio-equivalent (a new adapter + a media bridge, mostly reused).

## Two layers of work

### A. Infrastructure (the real new work — outside this repo)
- Stand up **Jambonz** (Docker/k8s). Size for expected concurrent calls; secure SIP + WS ingress
  (TLS/WSS, IP allow-listing, RTP port range, SBC/NAT).
- Add **DiDLogic as an outbound carrier** (SIP user/pass/host or IP-auth), codec **PCMU 8 kHz**.
  Verify a manual outbound test call with two-way audio before touching app code.
- This is the bulk of the effort and is genuinely new — nothing in the repo helps here.

### B. App changes (moderate; mirror existing patterns)
1. **Adapter** `src/providers/telephony/jambonz.py` implementing `ITelephonyProvider`
   (`src/interfaces/telephony.py`) — template: `src/providers/telephony/telnyx.py`.
   `initiate_call` → REST create-call to Jambonz (`from` = DiDLogic DID, `to`, `call_hook` = our
   answer webhook, routed over the DiDLogic carrier) → `CallSession(session_id, status="ringing")`.
   `hangup`/`transfer` via Jambonz REST. Register key `"didlogic"` in `TELEPHONY_PROVIDERS`
   (`src/providers/__init__.py`).
2. **Routes** in `src/api/telephony_hooks.py` — mirror `twilio/voice` + `twilio/stream/{slug}`:
   a `call_hook` returning Jambonz verb JSON with a `listen` verb (`bidirectionalAudio`) pointing
   at `wss://.../api/v1/telephony/didlogic/stream/{tenant_slug}`, plus the WS route that resolves
   the tenant by path slug and runs the bridge.
3. **Media bridge** — a new `_BaseLiveBridge` subclass (the one genuinely new piece) for Jambonz's
   `listen` frame format (initial JSON metadata, then binary L16 audio in; binary out). Reuse the
   dialogue core + the three transport hooks (`_inbound_loop`/`_send_audio_out`/`_send_interrupt`),
   resampling (`src/pipeline/audio_utils.py`), barge-in, and outcome/cost persistence from
   `live_bridge_base.py` / `telephony_live_bridge.py`. Wire the factory in `src/bootstrap.py` +
   `src/main.py` (alongside `set_stringee_bridge_factory`).
4. **Tenant config + secrets** — telephony `provider="didlogic"`; per-tenant SIP creds stored
   **encrypted in `tenant_secrets`** via the existing per-tenant-telephony-keys model. The Register
   endpoint (`src/api/tenants.py`) already accepts an arbitrary telephony `keys` dict and encrypts
   it, so SIP creds flow through with little/no schema change (optionally add explicit `sip_*`
   fields to `TenantTelephonyConfig` in `src/config_tenant.py`).
5. **Cost + billing** — add a `provider_costs` row `telephony/didlogic` (per-min). No billing-code
   change: telephony is already excluded from the platform total and shown as **tentative** (the
   tenant's own trunk) — exactly right for DiDLogic; it surfaces in the backoffice billing
   tentative figure automatically.
6. **Register UI** — add `didlogic` to the telephony dropdown in `static/admin_console.html` (the
   list is already data-driven from the cost catalog + a fallback set; one small addition).
7. **Call Lead** — `POST /campaigns/{id}/calls` (`src/api/calls.py`) already builds the adapter via
   `get_telephony_provider(...)`, inserts the conversation row, and snapshots cost. It works
   unchanged once the adapter exists. Outbound-only fits the current flow; no inbound DID routing.

## Why it's "extra" vs Twilio (the direct answer)
- **Twilio**: hosted — REST dial + TwiML + Media-Streams WS, all provided; adapter + bridge exist.
- **DiDLogic**: a bare SIP trunk — none of that exists. You must add a **SIP/RTP media gateway**
  (new infra) to terminate the trunk and bridge audio to a WebSocket. Only then does app-side work
  reduce to "a new adapter + a media bridge". The gateway + per-tenant SIP credentials are the
  genuinely extra pieces.

## Alternatives (noted for completeness)
- **FreeSWITCH + `mod_audio_fork`/`mod_audio_stream`**: registers/IP-auths to DiDLogic, originates
  the call, forks L16 audio to our WSS. Works, but no clean REST/webhook layer (ESL/dialplan).
- **LiveKit SIP**: dials out via the DiDLogic trunk into a LiveKit room; agent joins the room.
  Different audio model (WebRTC tracks) → larger refactor; only worth it if adopting LiveKit Agents.
- **Embedded pure-Python SIP (`pjsip`/`pjsua2`)**: INVITE + RTP in-process feeding `_BaseLiveBridge`.
  No extra infra, but fragile for production (NAT, jitter, codec, DTMF) — **POC only**.

## Verification (when implemented)
- Gateway first: manual Jambonz→DiDLogic outbound call with two-way audio.
- Adapter unit test: mock Jambonz REST create-call (respx); assert body + `CallSession`.
- Bridge unit test: feed synthetic Jambonz `listen` frames; assert audio reaches the agent and
  outbound frames are emitted.
- E2E (staging): register a `telephony=didlogic` tenant with SIP creds via `/admin`; Call Lead to a
  test number; confirm connect + agent audio + a `conversations` row with platform cost and a
  **tentative** DiDLogic telephony cost in the backoffice.

## Effort / risk
- **App code**: moderate — adapter (~`telnyx.py` size) + one new media bridge (the new work) +
  routes + factory + small config/cost/UI additions. ~80–90% of the bridge is reused.
- **Infra/ops**: the dominant cost — deploying/securing the gateway, the DiDLogic carrier config,
  concurrent-call capacity, NAT/SBC, codec/DTMF. New operational surface.
- **Risk**: live SIP/RTP voice quality (jitter, one-way audio, codec mismatch) + NAT — owned by the
  gateway, validated in the gateway-first step.
