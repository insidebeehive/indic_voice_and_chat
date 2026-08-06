"""LiveKit SDK glue — the real ``livekit.rtc``/``livekit.api`` wiring.

Deliberately isolated from ``livekit_bridge.py`` (which stays SDK-agnostic per
Phase 2's design — it only ever sees constructor-injected transport objects).
This module is where those objects actually get created: connecting to a
LiveKit room, subscribing to the caller's (SIP participant's) audio track,
publishing our own outbound audio track, and driving the whole per-call
lifecycle end to end.

``rtc``/``api`` are imported at module level (not lazily inside functions) so
tests can monkeypatch ``livekit_runner.rtc`` / ``livekit_runner.api`` wholesale
with fakes — see ``tests/unit/test_livekit_runner.py``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from livekit import api, rtc
from sqlalchemy import select

from src.api.call_store import insert_call, mark_answered, record_outcome
from src.campaign.models import LeadCallOutcome
from src.config_tenant import resolve_livekit_creds
from src.models.conversation import Conversation

log = logging.getLogger(__name__)

# Bounded wait for the caller's (SIP participant's) audio track to be
# subscribed after we join the room. If this expires the phone call never
# actually got answered on the other end — clean up and never build a bridge.
_JOIN_TIMEOUT_S = 60.0

# LiveKit's synthetic AudioSource output rate this runner publishes at. Must
# match the LiveKitBridge's ``output_rate`` default (24000 — Gemini Live's
# native output rate) so the bridge's resample-to-output-rate step is a no-op
# in the common case.
_OUTBOUND_SAMPLE_RATE = 24000
# Inbound AudioStream is requested at 16kHz mono — the model's expected input
# rate — so LiveKitBridge's inbound resample is a no-op in the common case too.
_INBOUND_SAMPLE_RATE = 16000


def _mint_token(api_key: str, api_secret: str, *, room_name: str, identity: str) -> str:
    """Mint a room-join access token for our agent identity.

    TTL is the SDK default (6h as of this SDK version) — deliberately not
    shortened: the token only gates the *initial* room join, not the ongoing
    session, and a call could legitimately run long.
    """
    grants = api.VideoGrants(
        room_join=True, room=room_name, can_publish=True, can_subscribe=True)
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(grants)
        .to_jwt()
    )


def _frame_factory(pcm16: bytes, sample_rate: int, num_channels: int):
    """callable(pcm16_bytes, sample_rate, num_channels) -> rtc.AudioFrame.

    Matches ``rtc.AudioFrame.__init__``'s actual signature (verified against
    ``livekit/rtc/audio_frame.py``): ``data``, ``sample_rate``, ``num_channels``,
    and a required ``samples_per_channel`` — computed here as
    ``len(pcm16) // 2 (bytes/int16 sample) // num_channels``.
    """
    return rtc.AudioFrame(
        data=pcm16, sample_rate=sample_rate, num_channels=num_channels,
        samples_per_channel=len(pcm16) // 2 // num_channels)


async def _ensure_conversation_row(sessionmaker, tenant, room_name: str, meta: dict) -> None:
    """Best-effort: insert a fresh ``in_progress`` row, or find the existing one —
    tenant-scoped, so a CRM reusing the same room name across two tenants can
    never match the wrong tenant's row (see module docstring / Fix 1 in the
    room-join review). Called immediately after the room connects, BEFORE the
    caller-track wait, so ``call.initiated`` fires as early as is honestly
    possible. A DB hiccup here must never prevent the call itself from running,
    so every failure is caught and logged, not raised — mirrors
    ``bridge_console.py``'s post-place-call insert."""
    try:
        async with sessionmaker() as db:
            existing = (await db.execute(
                select(Conversation).where(
                    Conversation.provider_call_sid == room_name,
                    Conversation.tenant_id == tenant.id,
                )
            )).scalar_one_or_none()
            if existing is None:
                call_id = f"call_{uuid.uuid4().hex[:16]}"
                await insert_call(
                    db, call_id=call_id, tenant=tenant, provider_call_sid=room_name,
                    campaign_id=meta.get("campaign_id"), voice=meta.get("voice"),
                    mode="s2s", channel="voice",
                    extra_event_data={"source": "livekit_room"})
            else:
                # Pre-registered row found. Bonus (Fix 3): if this call's own vox
                # metadata names a different campaign than the row was
                # pre-registered with, keep the row in sync with what's actually
                # driving the live call — otherwise reporting silently diverges.
                live_campaign_id = meta.get("campaign_id")
                if live_campaign_id and existing.campaign_id != live_campaign_id:
                    log.info(
                        "livekit run_call: pre-registered campaign_id %r differs from "
                        "live call metadata %r — updating row to match",
                        existing.campaign_id, live_campaign_id,
                        extra={"room_name": room_name, "tenant": tenant.slug})
                    existing.campaign_id = live_campaign_id
                    await db.commit()
    except Exception:  # noqa: BLE001 — best-effort, the call must run regardless
        log.exception(
            "livekit run_call: failed to record conversation row (best-effort)",
            extra={"room_name": room_name, "tenant": getattr(tenant, "slug", None)})


async def _mark_call_answered(sessionmaker, tenant, room_name: str) -> None:
    """Fires ``call.answered`` unconditionally for every call whose caller audio
    track actually got subscribed — not just pre-registered ones (Fix 3). Best-
    effort like every other DB touch in this module."""
    try:
        async with sessionmaker() as db:
            await mark_answered(db, room_name, tenant_id=tenant.id)
    except Exception:  # noqa: BLE001 — best-effort, the call must run regardless
        log.exception(
            "livekit run_call: failed to mark call answered (best-effort)",
            extra={"room_name": room_name, "tenant": getattr(tenant, "slug", None)})


async def _record_call_failure(sessionmaker, tenant, room_name: str, *, outcome: str, notes: str) -> None:
    """Best-effort ``call.completed`` signal for an early-exit failure that
    happened AFTER a conversation row was created (Fix 2) — so the CRM gets
    something actionable instead of silence. Never raises."""
    try:
        async with sessionmaker() as db:
            await record_outcome(
                db, room_name, tenant_id=tenant.id, status="ended",
                outcome=outcome, notes=notes)
    except Exception:  # noqa: BLE001 — best-effort, must not mask the original failure
        log.exception(
            "livekit run_call: failed to record call failure outcome (best-effort)",
            extra={"room_name": room_name, "tenant": getattr(tenant, "slug", None)})


async def _resolve_creds_for_call(tenant, sessionmaker) -> tuple[str, str, str] | None:
    """Wraps ``config_tenant.resolve_livekit_creds`` with the session this
    module actually has available: a ``sessionmaker`` (not an open session).

    When a ``sessionmaker`` is available, open one short-lived session for the
    lookup (tenant-level override, falling back to the CRM-level project).
    When it isn't (``sessionmaker=None`` — no DB configured), pass
    ``session=None`` straight through: ``resolve_livekit_creds``'s
    tenant-level branch never touches the DB, so that path still resolves;
    the CRM-level fallback is simply unavailable in that case. No logic is
    duplicated here — the resolver is the single source of truth for both
    branches.
    """
    if sessionmaker is not None:
        async with sessionmaker() as db:
            return await resolve_livekit_creds(db, tenant)
    return await resolve_livekit_creds(None, tenant)


async def _delete_room_best_effort(url: str, api_key: str, api_secret: str, room_name: str) -> None:
    """Drop the room server-side (and with it, the SIP leg) via LiveKit's
    RoomService ``DeleteRoom`` call, so a hangup we initiate actually ends the
    call instead of leaving the caller connected to silence.

    Best-effort: if this fails (network hiccup, room already gone, etc.) we
    still fall back to ``room.disconnect()`` in the caller — that at least
    drops *our* participant, even if the SIP leg itself lingers."""
    try:
        async with api.LiveKitAPI(url, api_key, api_secret) as lkapi:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
    except Exception:  # noqa: BLE001 — best-effort; room.disconnect() still runs
        log.exception("livekit run_call: delete_room failed (best-effort)",
                       extra={"room_name": room_name})


async def run_call(tenant, room_name: str, meta: dict, *, bridge_factory, sessionmaker=None) -> None:
    """Per-call lifecycle for one LiveKit room-join call. Invoked once per
    inbound call (the webhook trigger that calls this is Phase 5's scope).

    Lifecycle (restructured so early failures still produce a CRM-visible
    signal wherever a conversation row can honestly exist to attach one to):
      1. Resolve connection info (url + creds) — tenant-level override if
         fully configured, else the CRM-level project shared by every tenant
         under that CRM (``config_tenant.resolve_livekit_creds``).
      2. Mint a join token and connect to the room.
      3. Insert (or find + tenant-scope) the conversation row IMMEDIATELY —
         before waiting for the caller's track — so ``call.initiated`` fires
         as early as is honestly possible.
      4. Wait (bounded) for the caller's (SIP participant's) audio track to be
         subscribed.
      5a. On success: fire ``call.answered`` UNCONDITIONALLY (not just for
          pre-registered rows), build + publish our outbound audio track, then
          build and run the SDK-agnostic ``LiveKitBridge``.
      5b. On any failure from this point on (track-wait timeout, tenant not in
          s2s mode, campaign resolution failure, a crash building the bridge):
          write a ``call.completed`` outcome with a diagnosable reason instead
          of leaving the CRM with silence.
      6. Always disconnect the room on the way out.

    Failures BEFORE step 3 (missing config, the room connection itself
    failing) are the only category that can produce zero CRM signal — there is
    no conversation row to attach a failure to yet. That gap is logged loudly
    (not just ``log.exception``) so it's unmistakable in our own logs even
    though the CRM sees nothing.
    """
    meta = meta or {}

    # Credential resolution (tenant-level override, falling back to the
    # CRM-level shared project — src.config_tenant.resolve_livekit_creds) and
    # token minting happen before any conversation row can exist, so a
    # failure here gets a loud, unmistakable log line rather than silently
    # falling through to a generic "livekit run_call failed" a layer up in
    # livekit_routes.py. In practice the webhook route resolves these same
    # credentials and 401s first (see livekit_routes.py), so this path is
    # defensive rather than reachable in normal operation.
    try:
        resolved = await _resolve_creds_for_call(tenant, sessionmaker)
    except Exception:
        log.error(
            "livekit run_call: failed to resolve LiveKit credentials — "
            "NO CONVERSATION ROW COULD BE CREATED, CRM WILL SEE NOTHING for this attempt",
            extra={"room_name": room_name, "tenant": tenant.slug}, exc_info=True)
        raise

    if resolved is None:
        log.error(
            "livekit run_call: tenant %r has no usable LiveKit credentials (checked tenant-"
            "level override and CRM-level fallback) — "
            "NO CONVERSATION ROW COULD BE CREATED, CRM WILL SEE NOTHING for this attempt",
            tenant.slug, extra={"room_name": room_name, "tenant": tenant.slug})
        raise RuntimeError(
            f"tenant {tenant.slug!r} has no usable LiveKit credentials configured (tenant "
            "override or CRM-level fallback) — cannot join a LiveKit room")

    url, api_key, api_secret = resolved

    try:
        identity = f"vox-agent-{room_name}"
        token = _mint_token(api_key, api_secret, room_name=room_name, identity=identity)
    except Exception:
        log.error(
            "livekit run_call: failed to mint LiveKit join token — "
            "NO CONVERSATION ROW COULD BE CREATED, CRM WILL SEE NOTHING for this attempt",
            extra={"room_name": room_name, "tenant": tenant.slug}, exc_info=True)
        raise

    room = rtc.Room()

    # Populated by the track_subscribed handler once the caller's audio track
    # is actually available; the connect flow blocks on caller_track_ready
    # before building anything downstream (mirrors the design note: _on_start
    # should be able to assume the room is joined AND the caller's track is
    # already subscribed).
    caller_track_ready = asyncio.Event()
    state: dict = {"audio_stream": None}

    def _on_track_subscribed(track, publication, participant) -> None:
        if getattr(track, "kind", None) != rtc.TrackKind.KIND_AUDIO:
            return
        if state["audio_stream"] is not None:
            return  # already have the caller's track — ignore any further ones
        state["audio_stream"] = rtc.AudioStream(
            track, sample_rate=_INBOUND_SAMPLE_RATE, num_channels=1)
        caller_track_ready.set()
        log.info("livekit run_call: caller audio track subscribed",
                  extra={"room_name": room_name, "tenant": tenant.slug})

    def _on_room_closed(*_args) -> None:
        # Caller hung up or the room closed from the other side. The bridge
        # (once built) tears itself down when its inbound audio stream/session
        # ends; this handler exists so a hangup that happens *before* the
        # bridge is built (e.g. during the caller-track wait) is at least
        # logged rather than silently hanging until the join timeout.
        log.info("livekit run_call: room disconnected/participant left",
                  extra={"room_name": room_name, "tenant": tenant.slug})

    room.on("track_subscribed", _on_track_subscribed)
    room.on("participant_disconnected", _on_room_closed)
    room.on("disconnected", _on_room_closed)

    try:
        try:
            await room.connect(url, token, options=rtc.RoomOptions(auto_subscribe=True))
        except Exception:
            log.error(
                "livekit run_call: room.connect failed — "
                "NO CONVERSATION ROW COULD BE CREATED, CRM WILL SEE NOTHING for this attempt",
                extra={"room_name": room_name, "tenant": tenant.slug}, exc_info=True)
            raise
        log.info("livekit run_call: room joined", extra={"room_name": room_name, "tenant": tenant.slug})

        # Row exists (or is found) BEFORE the caller-track wait, so every
        # failure past this point has somewhere to attach a CRM-visible outcome.
        if sessionmaker is not None:
            await _ensure_conversation_row(sessionmaker, tenant, room_name, meta)

        try:
            await asyncio.wait_for(caller_track_ready.wait(), timeout=_JOIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            notes = f"caller audio track never subscribed within {_JOIN_TIMEOUT_S:.0f}s"
            log.warning(
                "livekit run_call: caller audio track never subscribed within %ss — "
                "call was never actually answered, no bridge built",
                _JOIN_TIMEOUT_S, extra={"room_name": room_name, "tenant": tenant.slug})
            if sessionmaker is not None:
                await _record_call_failure(
                    sessionmaker, tenant, room_name,
                    outcome=LeadCallOutcome.NO_ANSWER.value, notes=notes)
            return

        audio_stream = state["audio_stream"]

        audio_source = rtc.AudioSource(sample_rate=_OUTBOUND_SAMPLE_RATE, num_channels=1)
        local_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
        await room.local_participant.publish_track(
            local_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

        # call.answered fires for EVERY answered call now (Fix 3) — not only
        # calls that were pre-registered before the row-lookup existed.
        if sessionmaker is not None:
            await _mark_call_answered(sessionmaker, tenant, room_name)

        async def _hangup() -> None:
            await _delete_room_best_effort(url, api_key, api_secret, room_name)
            await room.disconnect()

        try:
            from src.bootstrap import LiveKitModeNotSupported
            build = await bridge_factory(tenant, room_name, meta)
            bridge = await build(
                audio_stream=audio_stream, audio_source=audio_source,
                frame_factory=_frame_factory, on_hangup=_hangup)
        except LiveKitModeNotSupported:
            if sessionmaker is not None:
                await _record_call_failure(
                    sessionmaker, tenant, room_name,
                    outcome=LeadCallOutcome.CALL_FAILED.value, notes="tenant not in s2s mode")
            raise
        except Exception as exc:  # noqa: BLE001 - campaign resolution failure, factory
                                   # crash, or any other bridge-construction error
            if sessionmaker is not None:
                await _record_call_failure(
                    sessionmaker, tenant, room_name,
                    outcome=LeadCallOutcome.CALL_FAILED.value,
                    notes=f"call setup failed before the bridge could start: {exc}")
            raise

        await bridge.run()
    finally:
        # rtc.Room.disconnect() is already a no-op if not connected, so this
        # is safe to call unconditionally (including when connect() itself
        # raised, or the caller-track wait timed out).
        try:
            await room.disconnect()
        except Exception:  # noqa: BLE001 — teardown must never raise
            log.exception("livekit run_call: room.disconnect() failed on teardown",
                           extra={"room_name": room_name})
