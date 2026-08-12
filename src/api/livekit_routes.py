"""LiveKit webhook receiver — the join trigger for room-based calls.

LiveKit has no inbound HTTP leg of its own (unlike Twilio/Exotel's answer
webhook + media-stream WS): a SIP trunk drops the caller straight into a
room, and the only signal we get is this webhook firing ``participant_joined``
for the SIP participant. This route's entire job is: verify the delivery is
genuinely from LiveKit (for this tenant), recognize that one event, and spawn
``livekit_runner.run_call`` to actually join and bridge the call — everything
downstream (tenant/campaign config, the bridge itself) is Phases 1-4, already
built and live-verified; this module only triggers it.

The LiveKit Python SDK has no HTTP layer at all (``WebhookReceiver.receive``
takes the raw body + auth token as plain arguments) — this route does the
header extraction and body handling the SDK doesn't provide.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from livekit import api
from livekit.protocol.models import ParticipantInfo
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.call_store import count_active_calls
from src.api.deps import get_db_session
from src.api.livekit_runner import run_call
from src.auth.middleware import tenant_from_slug
from src.bootstrap import LiveKitModeNotSupported
from src.config_tenant import resolve_livekit_creds
from src.models.database import get_sessionmaker

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony-livekit"])


# Factory: async (tenant, room_name, meta) -> builder closure (bootstrap.
# make_livekit_bridge_factory's two-stage shape). Registered by main.py's
# lifespan and read LIVE at request time (module attribute, not a captured
# import) — a `from livekit_routes import _livekit_bridge_factory` at import
# time would freeze the pre-lifespan None forever.
_livekit_bridge_factory = None


def set_livekit_bridge_factory(factory) -> None:
    """Register the LiveKit room-join bridge factory."""
    global _livekit_bridge_factory
    _livekit_bridge_factory = factory


# Rooms we already have a runner for: room_name -> the runner's asyncio.Task.
# LiveKit retries webhook delivery, so the same participant_joined can arrive
# twice; without this guard two agents would join one room and talk over each
# other. Single-process, single-threaded (asyncio) like dev_call_control's
# stores, so no locking is needed: the claim (setting the dict entry) and the
# create_task() that fills it are adjacent with no `await` between them, so
# the event loop cannot interleave a second delivery in between.
# Holding the Task also keeps a strong reference to it — a bare
# asyncio.create_task() result with nothing referencing it can be
# garbage-collected mid-call.
_in_flight_rooms: dict[str, asyncio.Task | None] = {}

_EVENT_PARTICIPANT_JOINED = "participant_joined"
_VOX_ATTR_PREFIX = "vox."
# The only keys the bridge factory actually consumes (src/bootstrap.py's
# make_livekit_bridge_factory) plus call_ref (echoed by the webhook/runner
# layer, not agent config) and mode (logged + ignored downstream — LiveKit
# room-join is tenant-level s2s only, no per-call mode override).
_META_KEYS = ("campaign_id", "voice", "lead", "call_ref", "mode")


def _parse_vox_blob(raw: str | None) -> dict | None:
    """Parse a metadata JSON string and return its top-level "vox" object.

    Returns None for empty/non-JSON/non-dict payloads and for JSON that has no
    "vox" key — LiveKit SIP dispatch rules and other integrations may put their
    own unrelated JSON in room/participant metadata, so anything not
    namespaced under "vox" is not ours to read.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        log.debug("livekit webhook: metadata is not valid JSON", extra={"raw": raw[:200]})
        return None
    if not isinstance(data, dict):
        return None
    vox = data.get("vox")
    return vox if isinstance(vox, dict) else None


_STRING_META_KEYS = ("campaign_id", "voice", "call_ref")


def _validate_vox_meta(meta: dict) -> dict:
    """Trust-boundary validation for CRM-supplied ``vox`` metadata.

    The CRM's JSON is untyped as far as we're concerned — a malformed field
    (e.g. ``campaign_id`` sent as an object, ``voice`` sent as a number,
    ``lead`` sent as a string) must never reach ``bootstrap.py`` and blow up
    deep inside call setup (``voice.strip()`` on an int, ``lead.get("name")``
    on a string, etc.) into a silent/dead line for the caller. Anything
    malformed is dropped (with a warning naming the field + type received) so
    the call falls back to defaults instead of crashing.

    ``mode`` is intentionally left untouched here — it's already ignored
    downstream regardless of type (bootstrap.py only checks ``is not None``
    before logging-and-dropping it), so no validation is needed for it.
    """
    out = dict(meta)
    for key in _STRING_META_KEYS:
        if key in out and not isinstance(out[key], str):
            log.warning("livekit webhook: metadata field %r has non-string type %s — dropping",
                        key, type(out[key]).__name__)
            out.pop(key)
    if "lead" in out:
        lead = out["lead"]
        if not isinstance(lead, dict):
            log.warning("livekit webhook: metadata field 'lead' has non-dict type %s — dropping",
                        type(lead).__name__)
            out.pop("lead")
        else:
            clean_lead = dict(lead)
            for sub_key in ("name", "gender"):
                if sub_key in clean_lead and not isinstance(clean_lead[sub_key], str):
                    log.warning(
                        "livekit webhook: metadata field 'lead.%s' has non-string type %s — dropping",
                        sub_key, type(clean_lead[sub_key]).__name__)
                    clean_lead.pop(sub_key)
            out["lead"] = clean_lead
    return out


def _extract_vox_metadata(event) -> dict:
    """Per-call metadata for the bridge factory, from whichever place the CRM set it.

    Precedence: room.metadata -> participant.metadata -> flat "vox."-prefixed
    participant attributes. Returns {} when nothing is set (the factory then
    falls back entirely to tenant/campaign config). Every value is validated
    at this trust boundary (see ``_validate_vox_meta``) before being handed to
    the bridge factory — a malformed CRM-supplied field is dropped, not passed
    through untyped.
    """
    # Accessing event.room / event.participant on an event that doesn't carry
    # them is safe: proto3 returns a default instance (metadata == ""), never
    # raises — no need to guard on event.HasField(...) here.
    for raw in (event.room.metadata, event.participant.metadata):
        vox = _parse_vox_blob(raw)
        if vox:
            return _validate_vox_meta({k: vox[k] for k in _META_KEYS if k in vox})

    # Attributes fallback: ScalarMap[str, str], so values are strings only — a
    # nested `lead` object can't be expressed there, hence the lead_name/
    # lead_gender flattening below (bootstrap.py reads meta["lead"]["name"]/
    # ["gender"]).
    attrs = {
        k[len(_VOX_ATTR_PREFIX):]: v
        for k, v in event.participant.attributes.items()
        if k.startswith(_VOX_ATTR_PREFIX)
    }
    meta = {k: attrs[k] for k in _META_KEYS if k in attrs and k != "lead"}
    lead: dict = {}
    if attrs.get("lead_name"):
        lead["name"] = attrs["lead_name"]
    if attrs.get("lead_gender"):
        lead["gender"] = attrs["lead_gender"]
    if lead:
        meta["lead"] = lead
    # Attributes are a ScalarMap[str, str] (proto), so every value here is
    # already a str — _validate_vox_meta is still applied for uniformity /
    # future-proofing, but is a no-op on this path today.
    return _validate_vox_meta(meta)


async def _run_call_task(tenant, room_name: str, meta: dict, bridge_factory) -> None:
    """Own one call end to end, then release the room's in-flight claim.

    ``bridge_factory`` is passed by value (read live in the request handler,
    not re-read here), so a lifespan teardown mid-call can't null it out from
    under a call that was already accepted.
    """
    try:
        sessionmaker = get_sessionmaker()
    except Exception:  # noqa: BLE001 - no DB configured: the call still runs, unlogged
        log.exception("livekit webhook: no sessionmaker; running call without DB bookkeeping")
        sessionmaker = None
    try:
        await run_call(tenant, room_name, meta,
                        bridge_factory=bridge_factory, sessionmaker=sessionmaker)
    except LiveKitModeNotSupported:
        log.warning("livekit webhook: tenant is not in s2s mode — no agent joined",
                    extra={"tenant": tenant.slug, "room_name": room_name})
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a failed call must not surface as an
                        # unretrieved task exception at GC time
        log.exception("livekit run_call failed",
                       extra={"tenant": tenant.slug, "room_name": room_name})
    finally:
        _in_flight_rooms.pop(f"{tenant.id}:{room_name}", None)


@router.post("/livekit/webhook/{tenant_slug}", response_class=Response)
async def livekit_webhook(
    tenant_slug: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """LiveKit server webhook — the room-join trigger.

    Verifies the delivery (tenant-scoped LiveKit key/secret), then spawns
    ``run_call`` for exactly one event shape: a SIP participant joining a
    room. Every other event is acknowledged and ignored. Always returns fast
    (never awaits the call itself) — LiveKit expects a prompt 2xx.
    """
    # Tenant first, before any signature work — an unknown slug is a routing
    # problem (wrong webhook URL configured), not an auth problem, so it gets
    # its own 404 rather than being folded into the generic 401 below.
    tenant = await tenant_from_slug(tenant_slug)

    # WebhookReceiver re-encodes with body.encode() to check the hash, so the
    # decode here must round-trip byte-for-byte.
    raw = await request.body()
    try:
        body = raw.decode("utf-8")
    except UnicodeDecodeError:
        log.warning("livekit webhook: body is not valid utf-8", extra={"tenant": tenant.slug})
        raise HTTPException(status_code=401, detail="invalid webhook signature")

    # LiveKit sends the webhook JWT bare in Authorization (no scheme); tolerate
    # an optional "Bearer " prefix in case a proxy/relay adds one.
    token = (request.headers.get("authorization") or "").strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1].strip()
    if not token:
        log.warning("livekit webhook: no auth token on request",
                    extra={"tenant": tenant.slug, "headers": sorted(request.headers.keys())})
        raise HTTPException(status_code=401, detail="invalid webhook signature")

    # Same LiveKit credential resolution as livekit_runner.run_call (tenant-
    # level override, falling back to the CRM-level shared project), so a
    # webhook only verifies against the exact key/secret pair the runner will
    # use to actually join the room.
    try:
        resolved = await resolve_livekit_creds(session, tenant)
    except Exception:  # noqa: BLE001 - MissingEnvError, decrypt failure, etc.
        log.exception("livekit webhook: failed to resolve LiveKit credentials",
                       extra={"tenant": tenant.slug})
        raise HTTPException(status_code=401, detail="invalid webhook signature")
    if resolved is None:
        log.error("livekit webhook: tenant has no LiveKit credentials configured "
                   "(tenant override or CRM-level fallback)",
                  extra={"tenant": tenant.slug})
        raise HTTPException(status_code=401, detail="invalid webhook signature")
    _url, api_key, api_secret = resolved

    try:
        event = api.WebhookReceiver(api.TokenVerifier(api_key, api_secret)).receive(body, token)
    except Exception:  # noqa: BLE001 - bad signature, hash mismatch, malformed JSON
        log.warning("livekit webhook: signature verification failed",
                    extra={"tenant": tenant.slug}, exc_info=True)
        raise HTTPException(status_code=401, detail="invalid webhook signature")
    # All four failure modes above return the identical 401 body — the reason
    # is only ever in our logs, never leaked to the caller.

    room_name = event.room.name
    if event.event != _EVENT_PARTICIPANT_JOINED or event.participant.kind != ParticipantInfo.Kind.SIP:
        # Positive match only: event == participant_joined AND kind == SIP.
        # Everything else (room_started, a browser participant joining, etc.)
        # is acknowledged and ignored — a wrong event-name guess fails closed
        # (no agent spawned) rather than spawning wrongly, and this log line
        # is what confirms the real constant on the first live delivery.
        log.info("livekit webhook: ignored event", extra={
            "tenant": tenant.slug, "event": event.event, "room_name": room_name,
            "participant_kind": int(event.participant.kind),
            "participant_identity": event.participant.identity})
        return Response(status_code=200)
    if not room_name:
        log.warning("livekit webhook: participant_joined with no room name",
                    extra={"tenant": tenant.slug})
        return Response(status_code=200)

    # Concurrency cap: same policy as calls.py's outbound guard, but 200 not
    # 429 — rejecting this webhook cannot "un-ring" the phone, the SIP leg is
    # already up and the caller already connected to the room. A non-2xx here
    # only buys LiveKit retrying the same over-cap delivery. We decline to
    # join and let the room sit agent-less until the caller hangs up.
    cap = tenant.settings.max_concurrent_calls
    try:
        active = await count_active_calls(session, tenant.id)
    except Exception:  # noqa: BLE001 - a DB hiccup must not drop a live call
        log.exception("livekit webhook: concurrency check failed, allowing the call",
                      extra={"tenant": tenant.slug, "room_name": room_name})
        active = 0
    if active >= cap:
        log.warning("livekit webhook: tenant at max concurrent calls (%s/%s) — not joining room",
                    active, cap, extra={"tenant": tenant.slug, "room_name": room_name})
        return Response(status_code=200)

    factory = _livekit_bridge_factory   # read live, per request
    if factory is None:
        log.error("livekit webhook: no bridge factory registered — cannot join room",
                  extra={"tenant": tenant.slug, "room_name": room_name})
        return Response(status_code=503)

    # Keyed on tenant + room, not room alone — room names are chosen by the
    # CRM and aren't guaranteed unique across tenants, so an unscoped key
    # could let one tenant's room name silently swallow another tenant's
    # call as a "duplicate".
    #
    # NOTE (post CRM-level LiveKit credentials): this key was originally
    # written when every tenant had its own separate LiveKit project. Since
    # tenants under one CRM can now share a single LiveKit project
    # (config_tenant.resolve_livekit_creds's CRM-level fallback), a room name
    # IS guaranteed unique within that shared project already — the tenant
    # prefix here no longer buys uniqueness in that case, it only preserves
    # this guard's behavior for tenants that still have their own separate
    # project. It also means the webhook's signature no longer binds "this
    # request is genuinely for this tenant" when the CRM-level secret is
    # shared: any tenant registered under the same CRM verifies against the
    # same key/secret, so tenant identity is enforced solely by the
    # {tenant_slug} path segment, not by the signature itself. This is an
    # accepted consequence of moving LiveKit credentials to CRM-level scope,
    # not a bug — flagged here for anyone revisiting this guard.
    in_flight_key = f"{tenant.id}:{room_name}"
    if in_flight_key in _in_flight_rooms:
        log.info("livekit webhook: duplicate delivery for a room already being handled",
                 extra={"tenant": tenant.slug, "room_name": room_name})
        return Response(status_code=200)

    # Claim + spawn: no `await` from here to create_task(), so a second
    # concurrent delivery for the same room cannot slip in between the
    # membership check above and the claim below.
    meta = _extract_vox_metadata(event)
    _in_flight_rooms[in_flight_key] = None
    _in_flight_rooms[in_flight_key] = asyncio.create_task(
        _run_call_task(tenant, room_name, meta, factory))
    log.info("livekit webhook: spawning room runner", extra={
        "tenant": tenant.slug, "room_name": room_name,
        "campaign_id": meta.get("campaign_id"), "call_ref": meta.get("call_ref")})
    return Response(status_code=200)   # never await the call itself
