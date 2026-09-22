"""Telephony provider webhook endpoints (PRD §7.4).

Twilio voice webhook + Media Streams websocket, tenant-aware.

Inbound flow:
1. Twilio rings the called number and POSTs to ``/twilio/voice`` with the
   ``To`` form param.
2. The voice handler resolves the tenant from the ``To`` number, builds
   the TwiML response with a ``<Stream url="wss://.../stream?tenant=<slug>"/>``
   so the websocket leg can re-resolve the same tenant.
3. Twilio opens the WebSocket; the WS handler reads ``?tenant=`` from the
   query string and asks the registered bridge factory for a per-call
   bridge wired with the tenant's agent + provider stack.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.api.telephony_exotel import voicebot_xml
from src.api.telephony_stringee import reprompt_scco
from src.api.telephony_stringee_bridge import StringeeIvrBridge, registry
from src.api.telephony_twilio import softphone_dial_twiml, voice_twiml
from src.auth import TenantContext
from src.auth.audit import log_denied
from src.auth.middleware import tenant_from_slug, tenant_from_twilio_to_number
from src.auth.webhook_auth import (
    WebhookAuthError,
    signature_mode,
    verify_exotel_basic,
    verify_stringee,
    verify_twilio,
)
from src.utils import public_url
from src.utils.http_fetch import assert_safe_url, fetch_capped
from src.utils.logging import debug_event

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony"])

# Twilio recording URLs always live on api.twilio.com or a subdomain scoped
# api.<region>.twilio.com — anchored so a suffix trick like
# "api.twilio.com.attacker.com" can never match.
_TWILIO_RECORDING_HOST_PATTERN = r"^api\.twilio\.com$|^api\.[a-z0-9-]+\.twilio\.com$"
# Stringee recording URLs live on api.stringee.com or a regional subdomain
# (e.g. asia-2.api.stringee.com).
_STRINGEE_HOST_PATTERN = r"^([a-z0-9-]+\.)*stringee\.com$"

# Twilio softphone recordings are dual-channel (record-from-answer-dual, see
# telephony_twilio.py) 8kHz/16-bit WAV at ~32 KB/s — the original (pre-C3-fix)
# code had no cap at all. 50 MiB covers ~26 min of audio, well above any real
# call, while still bounding a malicious/huge response.
_TWILIO_RECORDING_MAX_BYTES = 50 * 1024 * 1024  # ~26 min of dual-channel 8kHz/16-bit WAV
# A single Stringee IVR turn recording, fetched inline while a live call is on
# the line — bounded much tighter than the Twilio manual-call recording above.
_STRINGEE_TURN_MAX_BYTES = 10 * 1024 * 1024  # a single IVR turn's recording


# --- Browser softphone (human agent ↔ lead) deps ---------------------------
# The recording webhook needs the per-tenant provider registry (to build the
# tenant's STT + LLM for transcription + outcome analysis). main.py wires it in
# the lifespan; unset (tests without the full app) → the webhook 503s.
_softphone_providers: object | None = None


def set_softphone_providers(providers: object | None) -> None:
    """Register the per-tenant provider registry used by the recording webhook."""
    global _softphone_providers
    _softphone_providers = providers


# Sessionmaker for the softphone answer's background logging task (it needs its
# own session — the request's is closed once the fast response is sent). Defaults
# to the process-wide sessionmaker; tests inject an in-memory one.
_softphone_sessionmaker: object | None = None


def set_softphone_sessionmaker(sessionmaker: object | None) -> None:
    """Register the sessionmaker used by the answer webhook's background logger."""
    global _softphone_sessionmaker
    _softphone_sessionmaker = sessionmaker


# Factory: takes (websocket, tenant_context) -> bridge instance with .run()
BridgeFactory = Callable[[WebSocket, TenantContext], object]
_bridge_factory: BridgeFactory | None = None
_exotel_bridge_factory: BridgeFactory | None = None


def set_bridge_factory(factory: BridgeFactory | None) -> None:
    """Register the Twilio bridge factory."""
    global _bridge_factory
    _bridge_factory = factory


def set_exotel_bridge_factory(factory: BridgeFactory | None) -> None:
    """Register the Exotel bridge factory."""
    global _exotel_bridge_factory
    _exotel_bridge_factory = factory


_stringee_bridge_factory: Callable[..., StringeeIvrBridge] | None = None


def set_stringee_bridge_factory(factory) -> None:
    """Register the Stringee IVR bridge factory."""
    global _stringee_bridge_factory
    _stringee_bridge_factory = factory


async def prewarm_stringee_call(call_id: str, tenant) -> None:
    """Pre-synthesize the opening audio during the ringing phase so it's instant on answer.

    Called as a background task right after a Stringee outbound call is placed.
    The ringing window (~8-15s) is enough to synthesize the greeting TTS; the
    bridge is stored in the registry so _stringee_answer reuses it.
    """
    if _stringee_bridge_factory is None:
        debug_event(log, "stringee prewarm_call skipped", call_id=call_id, reason="no_bridge_factory")
        return
    from src.config_tenant import platform_webhook_base_url as _plat_wb
    raw_base = _plat_wb() or None
    if not raw_base:
        log.warning("stringee prewarm: no platform webhook_base_url set, skipping")
        return
    stringee_base = raw_base.rstrip("/") + "/stringee"
    try:
        bridge = _stringee_bridge_factory(
            call_id=call_id, tenant=tenant, base_url=stringee_base,
            fetch=functools.partial(_download, tenant=tenant))
        if inspect.isawaitable(bridge):
            bridge = await bridge
        from src.api.telephony_stringee_bridge import registry
        registry.put(bridge)
        await bridge.prewarm()
        debug_event(
            log, "stringee prewarm_call succeeded",
            call_id=call_id, tenant=getattr(tenant, "slug", None),
        )
    except Exception:
        log.exception("stringee prewarm failed", extra={"call_id": call_id})


def _ws_stream_url(request: Request, path: str) -> str:
    """Build the media-stream WSS URL for ``/api/v1/telephony/{path}``, honoring
    the reverse-proxy forwarded host/proto (Northflank terminates TLS
    upstream). Purely header-derived — never reads ``platform_webhook_base_url()``
    (see ``src/utils/public_url.py``'s ``origin_from_headers`` and
    ``platform_webhook_base_url()``'s docstring in ``src/config_tenant.py``:
    inbound telephony URLs must always resolve to the host that actually
    received the request).
    """
    origin = public_url.origin_from_headers(request)
    parsed = urlsplit(origin)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{ws_scheme}://{parsed.netloc}/api/v1/telephony/{path}"


# --- Inbound webhook authentication (Twilio / Exotel / Stringee) -----------
#
# Wires src/auth/webhook_auth.py's verify_* helpers into the routes that
# actually establish a call (the "voice"/"answer" webhooks) -- the ones an
# attacker who merely knows the URL could otherwise use to drive a call flow.
# Follows the src/api/external_chat.py Chatwoot-webhook precedent exactly:
#
#   - No secret configured for the tenant -> verification is skipped
#     entirely (non-breaking for a tenant that hasn't set one up).
#   - signature_mode() decides enforce (reject) vs log_only (warn + allow).
#   - Every provider's *_env / secret name below is an established
#     resolution path already used elsewhere in this tree -- see each
#     helper's docstring -- never a newly-invented config key.
#
# NOT wired here (same gap, deliberately left unaddressed -- see the
# deliverable notes): the per-turn `/stringee/event/{slug}` and lifecycle
# `/stringee/status/{slug}` webhooks (neither currently resolves a
# TenantContext at all; adding one purely to check a secret is a bigger,
# separate change), the Stringee/Twilio browser-softphone answer+recording
# routes, and Twilio's recording-status callback.


def _tenant_secret_optional(tenant: object, name: str | None) -> str | None:
    """``tenant.secret_optional(name)``, tolerant of a tenant object that
    doesn't implement it.

    Every real caller passes a ``TenantContext`` (which always has this
    method), but several existing telephony-route tests pass a bare
    ``SimpleNamespace(slug=..., id=...)`` stand-in for speed. Treating a
    missing method the same as "secret not configured" keeps this a pure
    addition for those tests -- skipped verification, not a crash -- rather
    than requiring every test double in the tree to grow a full
    ``TenantContext`` surface for an unrelated change.
    """
    if name is None:
        return None
    fn = getattr(tenant, "secret_optional", None)
    return fn(name) if fn is not None else None


def _reject_webhook_auth(message: str, *, reason: str, route: str, tenant: str | None) -> HTTPException:
    """Log a webhook auth rejection and return the uniform 401 to raise.

    Same log+return shape as external_chat.py's ``_reject_unauthorized``
    (``raise _reject_webhook_auth(...) from None`` at the call site), but
    logs via ``log_denied`` -- the auth-rejection primitive most of
    src/api/ already uses (calls.py, chat.py, knowledge.py, campaigns.py) --
    so these land in the same suppressed/rate-limited rejection stream
    instead of a parallel one. Never pass the signature/secret in ``extra``.
    """
    log_denied(
        logging.WARNING, message,
        event="auth_rejected", reason=reason, route=route, tenant=tenant,
    )
    return HTTPException(status_code=401, detail="invalid webhook signature")


def _signature_url_candidates(request: Request) -> list[str]:
    """URL(s) Twilio may have computed its ``X-Twilio-Signature`` over.

    The primary candidate is built the same proxy-aware way as
    ``_ws_stream_url``/``_forwarded_base`` (``origin_from_headers``) --
    Twilio signs the URL it actually POSTed to, which is the public,
    proxy-fronted one Northflank terminates TLS in front of, not necessarily
    what Starlette's own ``request.url`` reconstructs from the ASGI scope.
    That raw reconstruction is included as a second candidate so a
    header/proxy mismatch doesn't turn into a false reject.
    """
    path_and_query = request.url.path
    if request.url.query:
        path_and_query += f"?{request.url.query}"
    forwarded = f"{public_url.origin_from_headers(request)}{path_and_query}"
    raw = str(request.url)
    return [forwarded, raw] if forwarded != raw else [forwarded]


async def _verify_twilio_signature(request: Request, tenant: object, *, route: str) -> None:
    """Verify the inbound Twilio webhook's ``X-Twilio-Signature`` against the
    tenant's Twilio Auth Token.

    Twilio signs webhooks with the SAME Auth Token used to authenticate our
    outbound API calls (``twilio.request_validator.RequestValidator`` -- there
    is no separate webhook-signing secret to provision), so this resolves
    ``pipeline.telephony.creds_for("twilio").auth_token_env`` -- the existing
    credential slot src/api/dev_console.py's ``_cred(pcreds.auth_token_env)``
    already reads for the SAME provider's outbound calls.

    Skipped entirely when the tenant has no Twilio auth token configured --
    the non-breaking case for a tenant that hasn't set one up.
    """
    settings = getattr(tenant, "settings", None)
    telephony = getattr(getattr(settings, "pipeline", None), "telephony", None)
    auth_token_env = telephony.creds_for("twilio").auth_token_env if telephony is not None else None
    auth_token = _tenant_secret_optional(tenant, auth_token_env)
    tenant_slug = getattr(tenant, "slug", None)
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "telephony twilio_signature resolved",
            tenant=tenant_slug, route=route, mode=signature_mode(), configured=bool(auth_token),
        )
    if not auth_token:
        return
    try:
        verify_twilio(
            _signature_url_candidates(request), dict(await request.form()),
            request.headers.get("X-Twilio-Signature"), auth_token,
        )
    except WebhookAuthError as e:
        if signature_mode() == "enforce":
            raise _reject_webhook_auth(
                "twilio webhook: signature verification failed",
                reason=e.reason, route=route, tenant=tenant_slug,
            ) from None
        log.warning(
            "twilio webhook: signature check would have rejected (log_only mode)",
            extra={"reason": e.reason, "tenant": tenant_slug, "route": route, "mode": "log_only"},
        )
    else:
        if log.isEnabledFor(logging.DEBUG):
            debug_event(log, "telephony twilio_signature passed", tenant=tenant_slug, route=route)


async def _verify_exotel_basic_auth(request: Request, tenant: object, *, route: str) -> None:
    """Verify the inbound Exotel webhook's HTTP Basic ``Authorization`` header
    against the tenant's ``webhook:exotel_basic_user``/``webhook:exotel_basic_password``
    pair.

    That pair is minted by ``POST /tenants/{id}/webhook-credentials/rotate``
    for the tenant to configure as Basic Auth on Exotel's own
    Passthru/Voicebot applet -- the same pair src/api/answer_paths.py's
    ``answer_url_for`` already injects into the OUTBOUND-call answer URL's
    netloc for this exact purpose.

    Skipped entirely unless BOTH halves are configured (an incomplete pair
    is treated as unconfigured, matching ``answer_url_for``'s
    ``both_configured`` check) -- the non-breaking case for a tenant that
    hasn't rotated these credentials.
    """
    user = _tenant_secret_optional(tenant, "webhook:exotel_basic_user")
    password = _tenant_secret_optional(tenant, "webhook:exotel_basic_password")
    configured = bool(user and password)
    tenant_slug = getattr(tenant, "slug", None)
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "telephony exotel_basic_auth resolved",
            tenant=tenant_slug, route=route, mode=signature_mode(), configured=configured,
        )
    if not configured:
        return
    try:
        verify_exotel_basic(request.headers.get("Authorization"), username=user, password=password)
    except WebhookAuthError as e:
        if signature_mode() == "enforce":
            raise _reject_webhook_auth(
                "exotel webhook: basic auth verification failed",
                reason=e.reason, route=route, tenant=tenant_slug,
            ) from None
        log.warning(
            "exotel webhook: basic auth check would have rejected (log_only mode)",
            extra={"reason": e.reason, "tenant": tenant_slug, "route": route, "mode": "log_only"},
        )
    else:
        if log.isEnabledFor(logging.DEBUG):
            debug_event(log, "telephony exotel_basic_auth passed", tenant=tenant_slug, route=route)


async def _verify_stringee_signature(request: Request, tenant: object, *, route: str) -> None:
    """Verify the inbound Stringee webhook's ``X-STRINGEE-SIGNATURE`` header
    against the tenant's ``webhook:stringee_signing_secret`` (the tenant's
    Stringee Project Signing secret key -- set via the generic
    ``PATCH /tenants/{id}`` secret-update path; never auto-minted, see
    src/api/tenants.py's ``rotate_webhook_credentials`` docstring).

    Per Stringee's own docs (developer.stringee.com/docs/validating-requests-
    are-coming-from-stringee): base64(HMAC-SHA1(secret, data)) in the
    ``X-STRINGEE-SIGNATURE`` header, where ``data`` is the raw POST body for
    an event_url POST, or the Request-URI (path+query, leading ``/``) for an
    answer_url GET -- exactly what ``verify_stringee`` already implements.

    Skipped entirely when the tenant has no signing secret configured -- the
    non-breaking case for a tenant that hasn't set one up.
    """
    secret = _tenant_secret_optional(tenant, "webhook:stringee_signing_secret")
    tenant_slug = getattr(tenant, "slug", None)
    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "telephony stringee_signature resolved",
            tenant=tenant_slug, route=route, mode=signature_mode(), configured=bool(secret),
        )
    if not secret:
        return
    is_post = request.method == "POST"
    raw_body = await request.body() if is_post else None
    path_and_query = request.url.path
    if request.url.query:
        path_and_query += f"?{request.url.query}"
    try:
        verify_stringee(
            raw_body=raw_body, url_path_and_query=path_and_query,
            signature=request.headers.get("X-STRINGEE-SIGNATURE"), signing_secret=secret,
        )
    except WebhookAuthError as e:
        if signature_mode() == "enforce":
            raise _reject_webhook_auth(
                "stringee webhook: signature verification failed",
                reason=e.reason, route=route, tenant=tenant_slug,
            ) from None
        log.warning(
            "stringee webhook: signature check would have rejected (log_only mode)",
            extra={"reason": e.reason, "tenant": tenant_slug, "route": route, "mode": "log_only"},
        )
    else:
        if log.isEnabledFor(logging.DEBUG):
            debug_event(log, "telephony stringee_signature passed", tenant=tenant_slug, route=route)


@router.post("/twilio/voice", response_class=Response)
async def twilio_voice(
    request: Request,
    To: str = Form(...),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Twilio voice webhook → returns TwiML opening a tenant-aware media stream.

    Tenant resolution is direction-aware:
    - **inbound** (a customer dials our Twilio number): ``To`` is the
      owned number; look it up in ``tenant_phone_numbers``.
    - **outbound-api / outbound-dial** (we initiated the call via the
      orchestrator or place_test_call.py): ``To`` is the dialed destination
      (an end-user number we don't own); the tenant owns ``From`` instead.

    The resolved slug is embedded in the WS URL so the stream handler can
    re-resolve the same tenant on connect.
    """
    if log.isEnabledFor(logging.DEBUG):
        # Only the fields FastAPI declared above (To/From/CallSid/Direction)
        # are captured anywhere at any level -- Form(...) silently drops
        # every other field in Twilio's POST body (CallStatus, AccountSid,
        # ApiVersion, Caller, Called, FromCity, ...). A shape Twilio has
        # already changed is otherwise invisible until this is raised.
        debug_event(
            log, "twilio voice webhook_received",
            form=dict((await request.form()).multi_items()),
        )
    is_outbound = (Direction or "").startswith("outbound")
    lookup_number = From if (is_outbound and From) else To
    # Decision with a directly user-visible outcome: the wrong lookup_number
    # resolves the wrong tenant, which answers with the wrong agent/prompt.
    debug_event(
        log, "twilio voice tenant_lookup_decision",
        direction=Direction, is_outbound=is_outbound, to=To, from_=From,
        lookup_number=lookup_number,
    )
    tenant = await tenant_from_twilio_to_number(lookup_number)
    await _verify_twilio_signature(request, tenant, route="twilio_voice")

    # Tenant slug goes in the URL **path** — Twilio strips query strings
    # from <Stream url=...> attributes when opening the WSS connection.
    stream_url = _ws_stream_url(request, f"twilio/stream/{tenant.slug}")
    body = voice_twiml(stream_url)
    log.info(
        "twilio voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.post("/twilio/voice/{tenant_slug}", response_class=Response)
async def twilio_voice_for_tenant(
    request: Request,
    tenant_slug: str,
    To: str | None = Form(None),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Slug-scoped answer URL for **outbound** calls we place ourselves (dev
    console / orchestrator). The placing tenant is already known, so we resolve
    by slug and skip the caller-ID lookup — the outbound caller-ID need NOT be
    registered in ``tenant_phone_numbers`` (mirrors ``/stringee/answer/{slug}``).
    Inbound calls keep using the bare ``/twilio/voice`` (resolve by number)."""
    from src.auth.middleware import tenant_from_slug

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "twilio voice webhook_received (slug-scoped)",
            tenant_slug=tenant_slug, form=dict((await request.form()).multi_items()),
        )
    tenant = await tenant_from_slug(tenant_slug)
    await _verify_twilio_signature(request, tenant, route="twilio_voice_for_tenant")
    # Embed CallSid in the stream path so the bridge factory can look up
    # per-call overrides (voice/caller_name/lead_name/lead_gender). Twilio
    # strips query strings from <Stream url=...> so the SID must be in the path.
    stream_path = (f"twilio/stream/{tenant.slug}/{CallSid}"
                   if CallSid else f"twilio/stream/{tenant.slug}")
    stream_url = _ws_stream_url(request, stream_path)
    log.info(
        "twilio voice webhook (slug-scoped)",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=voice_twiml(stream_url), media_type="application/xml")


@router.websocket("/twilio/stream/{tenant_slug}")
async def twilio_stream(websocket: WebSocket, tenant_slug: str) -> None:
    """Twilio Media Streams websocket → bridges audio to the tenant's agent.

    Tenant slug arrives as a path segment because Twilio strips query
    strings from ``<Stream url=...>`` attributes when establishing the
    WSS connection.
    """
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("twilio stream tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
        return

    if _bridge_factory is None:
        log.warning("twilio stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="bridge factory unset")
        return

    bridge = _bridge_factory(websocket, tenant)
    if inspect.isawaitable(bridge):
        bridge = await bridge
    debug_event(log, "twilio stream bridged", tenant=tenant.slug)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("twilio stream client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("twilio stream bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


@router.websocket("/twilio/stream/{tenant_slug}/{call_sid}")
async def twilio_stream_with_sid(
    websocket: WebSocket, tenant_slug: str, call_sid: str
) -> None:
    """Outbound-call variant: call_sid in path lets the factory look up
    per-call overrides (voice / caller_name / lead params) keyed by SID."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("twilio stream (sid) tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
        return

    if _bridge_factory is None:
        log.warning("twilio stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="bridge factory unset")
        return

    bridge = _bridge_factory(websocket, tenant)
    if inspect.isawaitable(bridge):
        bridge = await bridge
    debug_event(log, "twilio stream bridged", tenant=tenant.slug, call_sid=call_sid)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("twilio stream client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("twilio stream bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# --- Browser softphone (human agent ↔ lead) ---------------------------------


def _forwarded_base(request: Request) -> str:
    """Public base URL for our telephony routes, honoring reverse-proxy
    headers. Purely header-derived — never reads ``platform_webhook_base_url()``
    (see ``src/utils/public_url.py``'s ``origin_from_headers``)."""
    return f"{public_url.origin_from_headers(request)}/api/v1/telephony"


async def _provider_caller_id(
    session: AsyncSession, tenant: TenantContext, provider: str
) -> str | None:
    """Outbound caller-ID for a provider's softphone leg.

    Prefers the tenant's number registered for ``provider`` in
    ``tenant_phone_numbers`` — a number that provider actually owns / dials out
    on (providers reject a caller-ID they don't own) — then falls back to the
    telephony config's ``outbound_from[provider]`` and finally ``from_number``.
    """
    from sqlalchemy import select

    from src.models.tenant import TenantPhoneNumber

    owned = (await session.execute(
        select(TenantPhoneNumber.phone_number).where(
            TenantPhoneNumber.tenant_id == tenant.id,
            TenantPhoneNumber.provider == provider,
        ).limit(1)
    )).scalars().first()
    tel = tenant.settings.pipeline.telephony
    outbound_from = (tel.outbound_from or {}).get(provider)
    caller_id = owned or outbound_from or tel.from_number
    debug_event(
        log, "telephony caller_id resolved",
        tenant=tenant.slug, provider=provider, caller_id=caller_id,
        source=("owned_number" if owned else
                "outbound_from_config" if outbound_from else
                "from_number_fallback" if tel.from_number else "none"),
    )
    return caller_id


@router.post("/twilio/softphone-twiml/{tenant_slug}", response_class=Response)
async def twilio_softphone_twiml(
    tenant_slug: str,
    request: Request,
    To: str = Form(...),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """TwiML App Voice URL for a tenant's browser softphone.

    Hit when a human agent's browser (Twilio Voice JS SDK) places a call: ``To``
    is the lead number the SDK passed; ``From`` is ``client:<identity>``. We log
    the call as a **manual** conversation row (so the recording webhook can
    finalize its outcome by CallSid) and return TwiML that dials the lead from
    the tenant's caller-ID with dual-channel recording.
    """
    import uuid

    from sqlalchemy import select

    from src.api.call_store import insert_call
    from src.models.conversation import Conversation

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "twilio softphone_twiml webhook_received",
            tenant_slug=tenant_slug, form=dict((await request.form()).multi_items()),
        )
    tenant = await tenant_from_slug(tenant_slug)
    caller_id = await _provider_caller_id(session, tenant, "twilio")
    if not caller_id:
        # Raised as an HTTPException (visible to the browser SDK as a 400),
        # but nothing today records WHY the tenant had no caller-ID -- an
        # operator sees only "call failed" from the human agent's side.
        debug_event(
            log, "twilio softphone_twiml rejected",
            tenant=tenant.slug, reason="no_caller_id_configured",
        )
        raise HTTPException(status_code=400, detail="tenant has no Twilio caller-ID configured")

    if CallSid:
        existing = (await session.execute(
            select(Conversation).where(Conversation.provider_call_sid == CallSid)
        )).scalar_one_or_none()
        if existing is None:
            row = await insert_call(
                session, call_id=f"call_{uuid.uuid4().hex[:16]}", tenant=tenant,
                provider_call_sid=CallSid, channel="softphone", agent_type="human")
            row.notes = f"manual softphone call → {To}"
            await session.commit()
            debug_event(log, "twilio softphone_twiml call_row created", tenant=tenant.slug, sid=CallSid)
        else:
            debug_event(log, "twilio softphone_twiml call_row exists", tenant=tenant.slug, sid=CallSid)

    base = _forwarded_base(request)
    cb = f"{base}/twilio/softphone-recording/{tenant_slug}"
    body = softphone_dial_twiml(to_number=To, caller_id=caller_id, recording_callback_url=cb)
    log.info("twilio softphone twiml", extra={
        "tenant": tenant.slug, "to": To, "from": From, "sid": CallSid})
    return Response(content=body, media_type="application/xml")


async def _download_twilio_recording(url: str, account_sid: str | None, auth_token: str | None) -> bytes:
    """Fetch a Twilio recording as a WAV. Twilio recordings require account auth.

    ``url`` is caller-supplied (the webhook's ``RecordingUrl`` form field) —
    an unauthenticated attacker could point it anywhere, and since Basic Auth
    carries the tenant's real Twilio Account SID + Auth Token, blindly
    fetching it would leak those credentials to an arbitrary host (SSRF +
    credential leak). Reject before making any network call unless the host
    is Twilio's own and (when we have a SID to protect) the URL path is
    scoped to that same account.
    """
    wav_url = url if url.endswith(".wav") else f"{url}.wav"
    try:
        parsed = assert_safe_url(
            wav_url, allowed_host_pattern=_TWILIO_RECORDING_HOST_PATTERN, require_https=True)
        if ".." in parsed.path.split("/"):
            # httpx normalizes dot-segments before sending, so a path like
            # /Accounts/AC_REAL/../AC_OTHER/... would pass the account-binding
            # regex below on the raw path but actually reach AC_OTHER on the
            # wire — reject before that check runs.
            raise ValueError("recording URL path contains a dot-segment")
        if account_sid and not re.search(rf"/Accounts/{re.escape(account_sid)}/", parsed.path):
            raise ValueError("recording URL is not scoped to the tenant's Twilio account")
    except ValueError as e:
        log.warning("twilio recording download rejected: %s", e)
        raise

    auth = (account_sid, auth_token) if (account_sid and auth_token) else None
    # `wav_url` is the RecordingUrl Twilio's own webhook sent us (a media URL,
    # the category the standard names explicitly) -- never account_sid/
    # auth_token, which are the actual credentials and never appear here.
    debug_event(log, "twilio recording_download request", url=wav_url, authenticated=auth is not None)
    content, _ct = await fetch_capped(
        wav_url, auth=auth, allowed_host_pattern=_TWILIO_RECORDING_HOST_PATTERN,
        max_bytes=_TWILIO_RECORDING_MAX_BYTES,
        timeout=httpx.Timeout(30.0, connect=10.0))
    debug_event(log, "twilio recording_download response", url=wav_url, bytes=len(content))
    return content


@router.post("/twilio/softphone-recording/{tenant_slug}", response_class=Response)
async def twilio_softphone_recording(
    tenant_slug: str,
    request: Request,
    CallSid: str | None = Form(None),
    RecordingUrl: str | None = Form(None),
    RecordingDuration: str | None = Form(None),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    """Twilio recording-status callback → transcribe → analyze → outcome.

    Produces the **same** structured outcome an AI call does: the dual-channel
    recording is split (agent track + lead track), each transcribed with the
    tenant's STT, then the same ``analyze_call`` runs and the result is persisted
    via the same ``record_outcome`` keyed by CallSid.
    """
    from datetime import datetime, timezone

    from src.analysis.manual_call import finalize_manual_call
    from src.interfaces.stt import STTConfig
    from src.pipeline.audio_utils import pcm16_to_wav, wav_split_stereo

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "twilio softphone_recording webhook_received",
            tenant_slug=tenant_slug, form=dict((await request.form()).multi_items()),
        )
    tenant = await tenant_from_slug(tenant_slug)
    if not (CallSid and RecordingUrl):
        log.warning("twilio softphone recording missing CallSid/RecordingUrl")
        return Response(status_code=200)
    if _softphone_providers is None:
        log.warning("softphone recording webhook hit but provider registry unset")
        return Response(status_code=503)

    from src.api.call_store import record_outcome
    from src.campaign.models import LeadCallOutcome

    c = tenant.settings.pipeline.telephony.active_creds()
    try:
        wav = await _download_twilio_recording(
            RecordingUrl, tenant.secret(c.account_sid_env), tenant.secret(c.auth_token_env))
        left, right, sr = wav_split_stereo(wav)
        debug_event(
            log, "twilio softphone_recording split",
            sid=CallSid, sample_rate=sr, left_bytes=len(left), right_bytes=len(right),
        )
    except Exception:  # noqa: BLE001 — a fetch/parse failure must not 500 Twilio
        log.exception("twilio softphone recording fetch/split failed", extra={"sid": CallSid})
        await record_outcome(
            session, CallSid,
            status="ended",
            outcome=LeadCallOutcome.RECORDING_UNAVAILABLE.value,
            summary="Call recording could not be retrieved for analysis.",
            duration_ms=int(float(RecordingDuration) * 1000) if RecordingDuration else None,
        )
        return Response(status_code=200)

    # Twilio dual-channel <Dial>: channel 1 = the parent (human agent) leg,
    # channel 2 = the dialed lead. Map agent→assistant, lead→user.
    channels = [("assistant", pcm16_to_wav(left, sr)), ("user", pcm16_to_wav(right, sr))]
    stt = _softphone_providers.get_stt(tenant)
    llm = _softphone_providers.get_llm(tenant)
    stt_config = STTConfig(language=tenant.settings.pipeline.stt.language, sample_rate=sr)
    dur_ms = int(float(RecordingDuration) * 1000) if RecordingDuration else None

    row = await finalize_manual_call(
        session, provider_call_sid=CallSid, channels=channels,
        stt=stt, stt_config=stt_config, llm=llm,
        tenant_timezone=tenant.settings.timezone,
        now=datetime.now(timezone.utc), duration_ms=dur_ms, recording_url=RecordingUrl)
    log.info("twilio softphone recording finalized", extra={
        "tenant": tenant.slug, "sid": CallSid, "found": row is not None,
        "outcome": getattr(row, "outcome", None)})
    return Response(status_code=200)


# --- Stringee browser softphone --------------------------------------------
# A tenant points two project URLs at us (slug in the path → tenant routing):
#   Answer URL → /telephony/stringee/softphone-answer/<slug>
#   Event URL  → /telephony/stringee/softphone-recording/<slug>
# The agent's Web SDK call (client → PSTN) hits the Answer URL; we connect it to
# the lead with dual-channel recording, and the Event URL delivers the recording
# URL → transcribe → analyze → same outcome as AI calls.


def _stringee_recording_url(data: dict) -> str | None:
    for key in ("recordUrl", "recording_url", "recordingUrl", "url", "link",
                "fileUrl", "file_url"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    return None


async def _log_softphone_call(tenant: TenantContext, provider_call_sid: str, to_number: str) -> None:
    """Insert the manual-call row for a Stringee softphone call (background).

    Runs AFTER the answer response is sent so the DB write never adds latency to
    Stringee's answer-URL fetch (a slow answer response is dropped as
    REQUEST_ANSWER_URL_ERROR). Opens its own session — the request's is closed.
    """
    import uuid

    from sqlalchemy import select

    from src.api.call_store import insert_call, mark_answered
    from src.models.conversation import Conversation
    from src.models.database import get_sessionmaker

    sessionmaker = _softphone_sessionmaker or get_sessionmaker()
    try:
        async with sessionmaker() as session:
            existing = (await session.execute(
                select(Conversation).where(Conversation.provider_call_sid == provider_call_sid)
            )).scalar_one_or_none()
            if existing is None:
                row = await insert_call(
                    session, call_id=f"call_{uuid.uuid4().hex[:16]}", tenant=tenant,
                    provider_call_sid=provider_call_sid, channel="softphone", agent_type="human")
                row.notes = f"manual softphone call → {to_number}"
                await session.commit()
            # The softphone answer IS the connect → mark answered (call.answered).
            await mark_answered(session, provider_call_sid)
            debug_event(
                log, "stringee softphone log_call succeeded",
                tenant=tenant.slug, call_sid=provider_call_sid, to_number=to_number,
                row_created=existing is None,
            )
    except Exception:  # noqa: BLE001 — logging must never break; the call already connected
        log.exception("stringee softphone: manual-call logging failed", extra={
            "call_sid": provider_call_sid})


@router.api_route("/stringee/softphone-answer/{tenant_slug}", methods=["GET", "POST"])
async def stringee_softphone_answer(
    tenant_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
):
    """Stringee answer webhook for a browser-softphone call → connect SCCO.

    Returns the SCCO **immediately** (Stringee's answer-URL fetch has a tight
    timeout; a slow response is dropped). The manual-call row is logged in a
    background task so the DB write doesn't block the response.
    """
    from src.api.telephony_stringee import softphone_connect_scco

    data = await _stringee_params(request)
    log.info("stringee softphone answer", extra={
        "tenant": tenant_slug, "method": request.method, "data": data})
    tenant = await tenant_from_slug(tenant_slug)
    # The connect `from` is the caller-ID NUMBER (type "internal"), bare. The
    # browser places the call FROM this number, so it arrives as the request's
    # `from`; use it (keeps the makeCall `from` and the SCCO `from` identical, as
    # in Stringee's working call). Fall back to the tenant's configured caller-ID.
    req_from = str(data.get("from") or "").lstrip("+")
    caller_id = req_from if req_from.isdigit() else \
        (await _provider_caller_id(session, tenant, "stringee") or "").lstrip("+")
    debug_event(
        log, "stringee softphone_answer caller_id_decision",
        tenant=tenant.slug, req_from=req_from, used_req_from=req_from.isdigit(), caller_id=caller_id,
    )
    if not caller_id:
        log.warning("stringee softphone: no caller-ID for tenant %s", tenant.slug)
        return Response(status_code=400)
    to_number = _stringee_number(data.get("to")) or _stringee_number(data.get("toNumber")) \
        or _stringee_number(data.get("called")) or _stringee_custom_destination(data)
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    if not to_number:
        log.warning("stringee softphone answer: no destination; keys=%s", sorted(data.keys()))
        return Response(status_code=404)

    # Log the manual call AFTER responding — never block Stringee's answer fetch.
    if call_id:
        background_tasks.add_task(_log_softphone_call, tenant, call_id, to_number)

    event_url = f"{_stringee_base(request)}/softphone-recording/{tenant_slug}"
    scco = softphone_connect_scco(
        caller_id=caller_id, to_number=to_number, event_url=event_url)
    debug_event(
        log, "stringee softphone_answer scco_built",
        tenant=tenant.slug, call_id=call_id, to_number=to_number, event_url=event_url, scco=scco,
    )
    return JSONResponse(scco)


@router.api_route("/stringee/softphone-recording/{tenant_slug}", methods=["GET", "POST"])
async def stringee_softphone_recording(
    tenant_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Stringee event webhook → on a recording URL, transcribe → analyze → outcome.

    The fetch + transcribe + analyze runs in a BACKGROUND task: recordings lag the
    event (the download retries on 404) and STT/LLM take time, so Stringee's
    webhook gets a fast 200 instead of being held open. Tolerant of Stringee's
    field names; non-recording events are acknowledged 200.
    """
    data = await _stringee_params(request)
    log.info("stringee softphone recording", extra={
        "tenant": tenant_slug, "method": request.method, "data": data})
    tenant = await tenant_from_slug(tenant_slug)
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    rec_url = _stringee_recording_url(data)
    if not (call_id and rec_url):
        # Silent ack today -- an operator watching for a missing outcome sees
        # nothing distinguishing "not a recording event" (expected, e.g. a
        # status ping) from "was a recording event but we couldn't find the
        # URL field" (a shape change worth chasing).
        debug_event(
            log, "stringee softphone_recording not_a_recording_event",
            tenant=tenant_slug, has_call_id=bool(call_id), has_rec_url=bool(rec_url),
            keys=sorted(data.keys()),
        )
        return Response(status_code=200)   # not a recording event — ack and move on
    if _softphone_providers is None:
        log.warning("stringee softphone recording hit but provider registry unset")
        return Response(status_code=503)
    dur_raw = data.get("duration") or data.get("callDuration") or data.get("recordDuration")
    dur_ms = int(float(dur_raw) * 1000) if dur_raw else None
    debug_event(
        log, "stringee softphone_recording finalize_dispatched",
        tenant=tenant_slug, call_id=call_id, rec_url=rec_url, duration_ms=dur_ms,
    )
    background_tasks.add_task(_finalize_softphone_recording, tenant, call_id, rec_url, dur_ms)
    return Response(status_code=200)


def _recording_mime(audio: bytes) -> str:
    """Sniff the recording's content type for the multimodal transcriber."""
    if audio[:4] == b"RIFF":
        return "audio/wav"
    return "audio/mpeg"   # Stringee records mono mp3


def _audio_transcriber(tenant_llm):
    """An LLM that can transcribe audio. Use the tenant's analysis LLM if it can
    (Gemini tenants); otherwise (e.g. Groq) fall back to a platform Gemini
    transcriber from GEMINI_API_KEY — Gemini handles Indian languages + the long
    mono mp3. Returns None if no audio-capable transcriber is available."""
    if getattr(tenant_llm, "transcribe_audio", None):
        debug_event(log, "softphone_recording transcriber_decision", source="tenant_llm")
        return tenant_llm
    from src.providers import get_llm_provider
    try:
        provider = get_llm_provider({"provider": "gemini"})  # GEMINI_API_KEY from env
        debug_event(log, "softphone_recording transcriber_decision", source="platform_gemini_fallback")
        return provider
    except Exception:  # noqa: BLE001
        log.warning("softphone recording: no Gemini transcriber — set the platform GEMINI_API_KEY")
        return None


async def _mark_recording_unavailable(call_id: str, dur_ms: int | None) -> None:
    """Persist recording-unavailable outcome when the recording can't be fetched or
    transcribed. Opens its own session — used from background tasks."""
    from src.api.call_store import record_outcome
    from src.campaign.models import LeadCallOutcome
    from src.models.database import get_sessionmaker

    sm = _softphone_sessionmaker or get_sessionmaker()
    try:
        async with sm() as session:
            await record_outcome(
                session, call_id,
                status="ended",
                outcome=LeadCallOutcome.RECORDING_UNAVAILABLE.value,
                summary="Call recording could not be retrieved for analysis.",
                duration_ms=dur_ms,
            )
        debug_event(log, "softphone_recording marked_unavailable", call_id=call_id, duration_ms=dur_ms)
    except Exception:  # noqa: BLE001
        log.exception("failed to mark recording-unavailable", extra={"call_id": call_id})


async def _finalize_softphone_recording(
    tenant: TenantContext, call_id: str, rec_url: str, dur_ms: int | None,
) -> None:
    """Fetch the recording (retry-on-404), transcribe it with the tenant's
    multimodal LLM (Gemini — strong on Indian languages, handles the long mono mp3
    the turn-STT can't), analyze, and persist the outcome. Background task → own
    DB session."""
    from datetime import datetime, timezone

    from src.analysis.manual_call import finalize_manual_call
    from src.interfaces.llm import LLMMessage
    from src.models.database import get_sessionmaker

    try:
        audio = await _download_stringee_recording(rec_url, tenant)
    except Exception:  # noqa: BLE001 — a fetch failure must not break anything
        log.exception("stringee softphone recording fetch failed", extra={"call_id": call_id})
        await _mark_recording_unavailable(call_id, dur_ms)
        return

    llm = _softphone_providers.get_llm(tenant)
    transcriber = _audio_transcriber(llm)
    if transcriber is None:
        log.warning("softphone recording: no audio transcriber; marking recording-unavailable",
                    extra={"call_id": call_id})
        await _mark_recording_unavailable(call_id, dur_ms)
        return
    try:
        text = await transcriber.transcribe_audio(audio, _recording_mime(audio))
    except Exception:  # noqa: BLE001
        log.exception("stringee softphone recording transcription failed", extra={"call_id": call_id})
        await _mark_recording_unavailable(call_id, dur_ms)
        return
    if not (text or "").strip():
        # STT produced nothing -- finalize proceeds with an empty transcript
        # rather than failing, so nothing else marks this. The existing INFO
        # line's transcript_chars=0 hints at it after the fact; this names
        # the branch at the point it's decided, for `event=~"telephony .*"`
        # queries that don't already know to look for a zero.
        debug_event(log, "stringee softphone_recording transcript_empty", call_id=call_id)
    transcript = [LLMMessage(role="user", content=text)] if (text or "").strip() else []

    sm = _softphone_sessionmaker or get_sessionmaker()
    try:
        async with sm() as session:
            row = await finalize_manual_call(
                session, provider_call_sid=call_id, transcript=transcript, llm=llm,
                tenant_timezone=tenant.settings.timezone,
                now=datetime.now(timezone.utc), duration_ms=dur_ms, recording_url=rec_url)
        log.info("stringee softphone recording finalized", extra={
            "tenant": tenant.slug, "call_id": call_id, "found": row is not None,
            "outcome": getattr(row, "outcome", None), "transcript_chars": len(text or "")})
    except Exception:  # noqa: BLE001 — never raise from a background task
        log.exception("stringee softphone recording finalize failed", extra={"call_id": call_id})


async def _download_stringee_recording(
    url: str, tenant: TenantContext, *, attempts: int = 6, _sleep=asyncio.sleep,
) -> bytes:
    """Fetch a Stringee recording, authenticating with a server token when the
    tenant's Stringee keys are available (signed/public URLs work without it).

    The recording lags the event — Stringee briefly returns 404 right after the
    call ends — so retry on 404 with backoff until it's ready."""
    from src.providers.telephony.stringee import mint_server_token

    tel = tenant.settings.pipeline.telephony
    creds = tel.active_creds()
    sid = tenant.secret(creds.account_sid_env) if creds.account_sid_env else None
    secret = tenant.secret(creds.auth_token_env) if creds.auth_token_env else None
    # Force https unconditionally (Stringee 301-redirects http→https, but we no
    # longer rely on following that redirect — see follow_redirects=False below),
    # and rewrite the host to the tenant's regional Stringee REST base when set —
    # the URL arrives pointing at api.stringee.com, but a regional project's
    # keySid is only valid on its regional host (e.g. asia-2.api.stringee.com),
    # else r:5 "keySid invalid" (see scripts/stringee_recording_probe.py). Auth
    # is the same X-STRINGEE-AUTH server token the callout uses.
    extra_host: str | None = None
    if tel.stringee_base_url:
        base = tel.stringee_base_url.rstrip("/")
        base_parts = urlsplit(base if "://" in base else "https://" + base)
        host = base_parts.netloc
        # extra_allowed_hosts is compared against assert_safe_url's
        # parts.hostname (port-stripped), so it must be built from .hostname
        # here too, not .netloc — a base like "myvendor.example:8443" would
        # otherwise never match its own allowlist entry. `host` (netloc) is
        # still used below to build the actual request URL, where the port
        # must be kept.
        extra_host = (base_parts.hostname or "").lower() or None
    else:
        url_parts = urlsplit(url if "://" in url else "https://" + url)
        host = url_parts.netloc
    parts = urlsplit(url if "://" in url else "https://" + url)
    url = f"https://{host}{parts.path}" + (f"?{parts.query}" if parts.query else "")

    assert_safe_url(
        url,
        allowed_host_pattern=_STRINGEE_HOST_PATTERN,
        extra_allowed_hosts={extra_host} if extra_host else None,
        require_https=True,
    )

    headers = {}
    if sid and secret:
        headers["X-STRINGEE-AUTH"] = mint_server_token(sid, secret)
    # `authenticated` only -- never sid/secret/the minted server token itself.
    debug_event(
        log, "stringee recording_download request",
        url=url, authenticated=bool(headers), attempts=attempts,
    )
    delay = 3.0
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=False) as client:
        for attempt in range(attempts):
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404 and attempt < attempts - 1:
                # Bounded to `attempts` (default 6) -- not a hot loop, safe
                # to log every retry rather than only the edge.
                debug_event(
                    log, "stringee recording_download retry",
                    url=url, attempt=attempt, next_delay_s=delay,
                )
                await _sleep(delay)            # recording not ready yet — wait + retry
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
            debug_event(
                log, "stringee recording_download response",
                url=url, status=resp.status_code, bytes=len(resp.content), attempt=attempt,
            )
            return resp.content
    resp.raise_for_status()   # exhausted retries on 404
    return resp.content


# --- Exotel webhook + WS ----------------------------------------------------


@router.post("/exotel/voice", response_class=Response)
async def exotel_voice(
    request: Request,
    To: str = Form(...),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Exotel Passthru / Voicebot webhook → returns ExotelML opening a stream.

    Exotel's form params mirror Twilio's (``To``, ``From``, ``CallSid``,
    ``Direction``) so the resolution logic is identical: for outbound calls
    the tenant owns ``From``; for inbound it owns ``To``.
    """
    if log.isEnabledFor(logging.DEBUG):
        # Same gap as the Twilio webhook: Form(...) only captures the fields
        # declared above, dropping every other Exotel field (CallType,
        # DialCallStatus, digits, ...).
        debug_event(
            log, "exotel voice webhook_received",
            form=dict((await request.form()).multi_items()),
        )
    is_outbound = (Direction or "").startswith("outbound")
    lookup_number = From if (is_outbound and From) else To
    debug_event(
        log, "exotel voice tenant_lookup_decision",
        direction=Direction, is_outbound=is_outbound, to=To, from_=From,
        lookup_number=lookup_number,
    )
    tenant = await tenant_from_twilio_to_number(lookup_number)
    await _verify_exotel_basic_auth(request, tenant, route="exotel_voice")

    stream_url = _ws_stream_url(request, f"exotel/stream/{tenant.slug}")
    body = voicebot_xml(stream_url)
    log.info(
        "exotel voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.post("/exotel/voice/{tenant_slug}", response_class=Response)
async def exotel_voice_for_tenant(
    request: Request,
    tenant_slug: str,
    To: str | None = Form(None),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
) -> Response:
    """Slug-scoped answer URL for **outbound** calls we place (dev console /
    orchestrator) — resolves the placing tenant by slug, so the outbound
    caller-ID need not be registered in ``tenant_phone_numbers``. Symmetric to
    the Twilio slug route; inbound keeps the bare ``/exotel/voice``."""
    from src.auth.middleware import tenant_from_slug

    if log.isEnabledFor(logging.DEBUG):
        debug_event(
            log, "exotel voice webhook_received (slug-scoped)",
            tenant_slug=tenant_slug, form=dict((await request.form()).multi_items()),
        )
    tenant = await tenant_from_slug(tenant_slug)
    await _verify_exotel_basic_auth(request, tenant, route="exotel_voice_for_tenant")
    stream_url = _ws_stream_url(request, f"exotel/stream/{tenant.slug}")
    log.info(
        "exotel voice webhook (slug-scoped)",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=voicebot_xml(stream_url), media_type="application/xml")


@router.websocket("/exotel/stream/{tenant_slug}")
async def exotel_stream(websocket: WebSocket, tenant_slug: str) -> None:
    """Exotel Voicebot Streaming websocket → bridges audio to the agent."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("exotel stream tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
        return

    if _exotel_bridge_factory is None:
        log.warning("exotel stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="exotel bridge factory unset")
        return

    bridge = _exotel_bridge_factory(websocket, tenant)
    if inspect.isawaitable(bridge):
        bridge = await bridge
    debug_event(log, "exotel stream bridged", tenant=tenant.slug)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("exotel stream client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("exotel stream bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# --- Stringee IVR webhook routes --------------------------------------------


def _stringee_base(request: Request) -> str:
    """Public base URL for Stringee IVR webhook callbacks, honoring
    reverse-proxy headers. Purely header-derived — never reads
    ``platform_webhook_base_url()`` (see ``src/utils/public_url.py``'s
    ``origin_from_headers``)."""
    return f"{public_url.origin_from_headers(request)}/api/v1/telephony/stringee"


async def _download(url: str, tenant: TenantContext) -> bytes:
    """Fetch a Stringee-hosted recording URL for the IVR turn bridge.

    ``url`` arrives via the ``event_url``/``recording_url`` webhook fields —
    caller-controlled input from Stringee's side, so it's restricted to
    Stringee's own hosts (plus the tenant's configured regional/whitelabel
    Stringee host) before any request goes out. This blocks a live call turn,
    so it keeps the original tight 8s/5s timeout and a cap sized for a single
    IVR turn's recording rather than fetch_capped's shared defaults.
    """
    tel = tenant.settings.pipeline.telephony
    extra_host = None
    if tel.stringee_base_url:
        base = tel.stringee_base_url.rstrip("/")
        base_parts = urlsplit(base if "://" in base else "https://" + base)
        # See the matching comment in _download_stringee_recording: must be
        # .hostname (port-stripped), not .netloc, to match what
        # assert_safe_url compares extra_allowed_hosts against.
        extra_host = (base_parts.hostname or "").lower() or None
    # Force https unconditionally, same as _download_stringee_recording and for
    # the same reason: Stringee sends/redirects http→https, or a configured
    # base URL might carry any scheme, and fetch_capped requires https up
    # front (require_https=True) — an unmodified http:// URL here would be
    # rejected outright instead of upgraded, and since this runs inline on a
    # live call turn, that failure is swallowed by the bridge's broad
    # exception handling and silently reprompts the caller forever.
    parts = urlsplit(url if "://" in url else "https://" + url)
    url = f"https://{parts.netloc}{parts.path}" + (f"?{parts.query}" if parts.query else "")
    debug_event(log, "stringee ivr_turn_recording request", url=url)
    content, _ct = await fetch_capped(
        url, allowed_host_pattern=_STRINGEE_HOST_PATTERN,
        extra_allowed_hosts={extra_host} if extra_host else None,
        max_bytes=_STRINGEE_TURN_MAX_BYTES,
        timeout=httpx.Timeout(8.0, connect=5.0))
    debug_event(log, "stringee ivr_turn_recording response", url=url, bytes=len(content))
    return content


def _stringee_number(value: object) -> str | None:
    """Pull a phone number out of a webhook field that may be a bare string,
    an object (``{"number": ...}`` — the shape our callout uses), or a list of
    those. The exact answer-webhook shape is confirmed by the first live call.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        return value.get("number") or value.get("e164") or value.get("alias")
    return value if isinstance(value, str) else None


def _stringee_custom_destination(data: dict) -> str | None:
    """Pull the dialed number out of Stringee's ``customData``.

    For an app-to-phone softphone call Stringee does **not** forward the SDK's
    ``to`` to the answer webhook — it arrives empty with ``fromInternal=true``
    (confirmed by the live call debugger). The browser instead carries the lead
    number in ``customData``, delivered here as the ``custom`` param: usually a
    JSON string like ``{"to": "+91…"}``, occasionally a bare number.
    """
    raw = data.get("custom") or data.get("customData") \
        or data.get("customDataFromYourServer")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return _stringee_number(raw)  # a bare number string, not JSON
    if isinstance(raw, dict):
        return _stringee_number(
            raw.get("to") or raw.get("toNumber") or raw.get("number"))
    return None


async def _stringee_params(request: Request) -> dict:
    """Merge a Stringee webhook's data regardless of method. Stringee fetches
    the answer_url via GET (call info in the query string); other hooks POST
    JSON. We read both so the routes work either way."""
    data = dict(request.query_params)
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001 - tolerate empty/non-JSON bodies
            # Swallowed on purpose (Stringee sometimes POSTs an empty body),
            # but a genuinely malformed JSON body is silently indistinguishable
            # from that today -- the caller only ever sees query_params, with
            # every POSTed field missing, which looks exactly like a routing
            # miss rather than a parse failure.
            debug_event(
                log, "stringee params body_parse_failed",
                method=request.method, error_type=type(e).__name__,
            )
            body = None
        if isinstance(body, dict):
            data = {**data, **body}
    return data


async def _resolve_stringee_tenant(data: dict):
    """Resolve the tenant by trying every number field (both from and to). The
    caller-id is the registered number for outbound, the called number for
    inbound — so trying both is robust to whether Stringee sends a direction."""
    for key in ("from", "fromNumber", "caller", "to", "toNumber", "called"):
        num = _stringee_number(data.get(key))
        if not num:
            continue
        try:
            tenant = await tenant_from_twilio_to_number(num)
            debug_event(log, "stringee tenant_resolve matched", field=key, number=num, tenant=tenant.slug)
            return tenant
        except Exception as e:  # noqa: BLE001 - not this number; try the next
            debug_event(
                log, "stringee tenant_resolve candidate_failed",
                field=key, number=num, error_type=type(e).__name__,
            )
            continue
    return None


async def _stringee_answer(request: Request, tenant: "TenantContext | None"):
    """Build the call's bridge for an already-resolved tenant and return the
    opening SCCO. Shared by the slug route (outbound, tenant from the URL) and the
    number route (inbound fallback, tenant from the dialed/caller number)."""
    data = await _stringee_params(request)
    log.info("stringee answer", extra={"method": request.method, "data": data})
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    if not call_id:
        log.warning("stringee answer missing call_id; keys=%s", sorted(data.keys()))
    if tenant is None:
        log.warning("stringee answer: no tenant; keys=%s", sorted(data.keys()))
        return Response(status_code=404)
    await _verify_stringee_signature(request, tenant, route="stringee_answer")
    if _stringee_bridge_factory is None:
        debug_event(log, "stringee answer rejected", call_id=call_id, reason="no_bridge_factory")
        return Response(status_code=503)
    # Derive the base URL for audio/event webhook links. Prefer the platform
    # webhook_base_url (always the public-internet address) over request headers,
    # which can resolve to internal/container addresses under Northflank's proxy.
    from src.config_tenant import platform_webhook_base_url as _plat_wb
    _raw_base = _plat_wb() or None
    stringee_base = (_raw_base.rstrip("/") + "/stringee") if _raw_base else _stringee_base(request)
    log.info("stringee answer base", extra={"base": stringee_base, "tenant": tenant.slug})
    # Reuse a pre-warmed bridge (synthesized during ringing) if available;
    # otherwise build a fresh one and synthesize on-demand.
    bridge = registry.get(call_id) if call_id else None
    if bridge is not None:
        log.info("stringee answer: reusing prewarmed bridge", extra={"call_id": call_id})
    else:
        debug_event(log, "stringee answer building_fresh_bridge", call_id=call_id, tenant=tenant.slug)
        bridge = _stringee_bridge_factory(
            call_id=call_id, tenant=tenant,
            base_url=stringee_base, fetch=functools.partial(_download, tenant=tenant),
        )
        if inspect.isawaitable(bridge):
            bridge = await bridge
        registry.put(bridge)
    scco = await bridge.start_call()
    log.info("stringee answer SCCO", extra={"call_id": call_id, "scco": scco})
    if call_id:
        from src.api import dev_call_control
        dev_call_control.monitor.set_status(call_id, "answered")
        debug_event(log, "stringee answer dev_call_control_updated", call_id=call_id, status="answered")
    log.info("stringee answer registered", extra={"tenant": tenant.slug, "call_id": call_id})
    return JSONResponse(scco)


@router.api_route("/stringee/answer/{tenant_slug}", methods=["GET", "POST"])
async def stringee_answer_for_tenant(tenant_slug: str, request: Request):
    """Outbound answer URL: the placing tenant is the slug in the path (set when we
    built the callout's answer_url), so attribution + the live bridge's config
    follow who placed the call — not the shared caller/dialed number."""
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except Exception as e:  # noqa: BLE001 - unknown slug → 404 below
        debug_event(
            log, "stringee answer_for_tenant slug_resolution_failed",
            tenant_slug=tenant_slug, error_type=type(e).__name__,
        )
        tenant = None
    return await _stringee_answer(request, tenant)


@router.api_route("/stringee/answer", methods=["GET", "POST"])
async def stringee_answer(request: Request):
    """Inbound fallback: resolve the tenant from the dialed/caller number when the
    answer URL carries no slug. Stringee fetches the answer_url via GET (call info
    in the query string); we accept POST/JSON too.
    """
    tenant = await _resolve_stringee_tenant(await _stringee_params(request))
    return await _stringee_answer(request, tenant)


@router.api_route("/stringee/event/{tenant_slug}", methods=["GET", "POST"])
async def stringee_event(tenant_slug: str, request: Request, call_id: str | None = None):
    """Per-turn recordMessage webhook -> run a turn -> return the next SCCO.

    Note: tenant_slug is URL namespacing only; the bridge is looked up by
    call_id — tenant_slug is not validated here.
    call_id is optional so that a missing ?call_id= query param returns a
    graceful reprompt (200) instead of a FastAPI 422.
    """
    data = await _stringee_params(request)
    log.info("stringee event", extra={"tenant": tenant_slug, "call_id": call_id,
                                       "method": request.method, "data": data})
    rec_url = (
        data.get("recording_url")
        or data.get("url")
        or data.get("link")
        or data.get("fileUrl")
        or data.get("file_url")
        or data.get("recordingUrl")
    )
    bridge = registry.get(call_id) if call_id else None
    if bridge is None or not rec_url:
        # A silent reprompt today, with the two distinct causes (registry
        # miss -- the bridge expired/was never registered -- vs. Stringee
        # simply not sending a recording URL for this event) indistinguishable
        # from outside. A caller stuck reprompting forever is exactly the
        # motivating shape for this category.
        debug_event(
            log, "stringee event reprompt",
            tenant=tenant_slug, call_id=call_id, bridge_found=bridge is not None,
            has_rec_url=bool(rec_url),
        )
        base = _stringee_base(request)
        return JSONResponse(reprompt_scco(
            text="Maaf kijiye, dobara boliye?",
            event_url=f"{base}/event/{tenant_slug}?call_id={call_id or ''}",
        ))
    scco = await bridge.handle_turn(recording_url=rec_url)
    debug_event(log, "stringee event turn_handled", tenant=tenant_slug, call_id=call_id, scco=scco)
    return JSONResponse(scco)


@router.get("/stringee/audio/{token}")
async def stringee_audio(token: str, call_id: str | None = None):
    """Serve a hosted reply/opening WAV for Stringee's `play` to fetch."""
    wav = None
    if call_id:
        bridge = registry.get(call_id)
        if bridge is not None:
            wav = bridge.audio.get(token)
    if wav is None:
        # Stringee may fetch without our call_id query: scan live calls.
        for b in registry.iter_bridges():
            wav = b.audio.get(token)
            if wav is not None:
                break
    log.info("stringee audio fetch", extra={"token": token, "call_id": call_id,
             "found": wav is not None, "bytes": len(wav) if wav else 0})
    if wav is not None:
        return Response(content=wav, media_type="audio/wav")
    return Response(status_code=404)


@router.api_route("/stringee/status/{tenant_slug}", methods=["GET", "POST"])
async def stringee_status(tenant_slug: str, request: Request):
    """Lifecycle webhook: on call end, record the outcome and clean up."""
    data = await _stringee_params(request)
    call_id = str(data.get("call_id") or data.get("callId") or data.get("call_sid") or "")
    status = (data.get("status") or data.get("event") or data.get("call_status") or "").upper()
    log.info("stringee status", extra={"call_id": call_id, "status": status,
                                        "method": request.method, "data": data})
    is_terminal = status in ("ENDED", "FAILED", "NO_ANSWER", "BUSY")
    if is_terminal:
        await registry.end(call_id)
        if call_id:
            from src.api import dev_call_control
            dev_call_control.monitor.set_status(call_id, "ended")
    else:
        # A status Stringee sends that isn't in the terminal set skips
        # registry cleanup entirely and silently leaves the bridge (and its
        # audio cache) registered -- worth naming explicitly since the INFO
        # line above already shows `status` but not whether it was RECOGNIZED
        # as terminal.
        debug_event(log, "stringee status not_terminal", call_id=call_id, status=status)
    return Response(status_code=200)
