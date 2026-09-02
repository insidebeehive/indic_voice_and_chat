"""Generic external chat integration + thin OpenAI-compatible adapter.

Any external chat platform (Chatwoot Captain, tawk.to, Freshdesk, …) can
route messages here and receive bot replies using our RAG + CRM tools.

Two endpoints:

    POST /integrations/message
        Our own simple format — the recommended integration point.
        Callers supply a ``conversation_id`` they own; we handle session
        continuity transparently via Redis.

    POST /integrations/ai/v1/chat/completions
        Thin OpenAI-format adapter — for platforms like Chatwoot Captain
        that speak the OpenAI Chat Completions protocol.
        Delegates entirely to the generic endpoint above; adds no AI logic.

Auth: ``Authorization: Bearer <tenant-token>`` on both endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.auth import middleware as auth_middleware
from src.auth.audit import token_fingerprint
from src.auth.webhook_auth import WebhookAuthError, signature_mode, verify_chatwoot
from src.models.chat import ChatSession
from src.models.database import get_sessionmaker

log = logging.getLogger(__name__)
router = APIRouter(prefix="/integrations", tags=["integrations"])

_redis: Any = None
_EXT_SESSION_TTL = 86_400 * 7  # 7 days


def set_ext_redis(redis_client: Any) -> None:
    global _redis
    _redis = redis_client


def _ext_key(tenant_id: str, conversation_id: str) -> str:
    return f"ext_sess:{tenant_id}:{conversation_id}"


# ---------------------------------------------------------------------------
# Generic endpoint
# ---------------------------------------------------------------------------


class ExternalMessageRequest(BaseModel):
    conversation_id: str = Field(min_length=1, description="Platform's own conversation identifier")
    text: str = Field(min_length=1)
    user_id: Optional[str] = None
    customer_name: Optional[str] = None


class ExternalMessageResponse(BaseModel):
    text: str
    suggestions: list[str] = []
    session_id: str


@router.post("/message", response_model=ExternalMessageResponse)
async def external_message(
    req: ExternalMessageRequest,
    tenant: TenantContext = Depends(current_tenant),
    db: AsyncSession = Depends(get_db_session),
) -> ExternalMessageResponse:
    """Send a message and receive a bot reply.

    On the first call for a ``conversation_id`` a session is created and its
    ID is stored in Redis so subsequent calls reuse the same agent context
    (full conversation memory).
    """
    from src.api.chat import new_session_id, process_message

    # --- Session lookup / creation ---
    session_id: Optional[str] = None
    redis_key = _ext_key(tenant.id, req.conversation_id)

    if _redis is not None:
        raw = await _redis.get(redis_key)
        if raw:
            session_id = raw.decode() if isinstance(raw, bytes) else str(raw)

    if session_id is None:
        session_id = new_session_id()
        language = getattr(tenant.settings, "default_language", None) or "hi"
        db.add(ChatSession(
            id=session_id,
            tenant_id=tenant.id,
            customer_id=req.user_id,
            customer_name=req.customer_name,
            language=language,
            status="active",
            extra_data={"source": "external", "conversation_id": req.conversation_id},
        ))
        await db.commit()
        if _redis is not None:
            await _redis.set(redis_key, session_id, ex=_EXT_SESSION_TTL)
        log.info("external chat session created", extra={
            "tenant": tenant.slug, "conversation_id": req.conversation_id,
            "session_id": session_id,
            "user_fp": token_fingerprint(req.user_id) if req.user_id else None,
        })

    # --- Run agent turn ---
    result = await process_message(tenant, session_id, req.text)
    return ExternalMessageResponse(
        text=result.response_text,
        suggestions=result.suggested_followups,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Thin OpenAI adapter
# ---------------------------------------------------------------------------


class _OAIMessage(BaseModel):
    role: str
    content: str = ""


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[_OAIMessage]
    user: Optional[str] = Field(
        None,
        description=(
            "End-user identifier — used as conversation_id for session continuity "
            "and as user_id for player-specific CRM tools. Should be set to the "
            "platform's conversation or contact identifier."
        ),
    )


class OpenAIChatResponse(BaseModel):
    id: str = "chatcmpl-ext"
    object: str = "chat.completion"
    choices: list[dict]


@router.post("/ai/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(
    req: OpenAIChatRequest,
    tenant: TenantContext = Depends(current_tenant),
    db: AsyncSession = Depends(get_db_session),
) -> OpenAIChatResponse:
    """OpenAI Chat Completions-compatible endpoint.

    Translates the OpenAI ``messages`` array into a single-turn call to the
    generic ``/integrations/message`` endpoint and returns an OpenAI-shaped
    response.  All AI logic lives in the generic endpoint.

    Set the ``user`` field to your platform's conversation or contact ID for
    full session continuity and player-specific CRM tool support.
    """
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=422, detail="messages must contain at least one user message")

    text = user_msgs[-1].content
    if not text.strip():
        raise HTTPException(status_code=422, detail="latest user message is empty")

    if not req.user:
        raise HTTPException(
            status_code=422,
            detail=(
                "user field is required — set it to your platform's conversation or "
                "contact ID to enable session continuity and CRM tool access"
            ),
        )

    ext_result = await external_message(
        ExternalMessageRequest(
            conversation_id=req.user,
            text=text,
            user_id=req.user,
        ),
        tenant=tenant,
        db=db,
    )

    return OpenAIChatResponse(choices=[{
        "index": 0,
        "message": {"role": "assistant", "content": ext_result.text},
        "finish_reason": "stop",
    }])


# ---------------------------------------------------------------------------
# Thin Chatwoot webhook adapter
# ---------------------------------------------------------------------------
# Chatwoot fires a webhook for EVERY event (message_created, conversation_created,
# agent_assigned, label_added, …).  This endpoint absorbs them all and acts only
# on incoming customer messages (message_type == 0, sender.type == "contact").
# All other events receive a 200 with {"ignored": true} and no further work.
#
# Chatwoot webhook payload shape (message_created):
#   {
#     "event": "message_created",
#     "message_type": 0,          // 0 incoming, 1 outgoing, 2 activity
#     "content": "Hello",
#     "conversation": { "id": 456 },
#     "sender": { "name": "Ravi", "identifier": "<external_id>", "type": "contact" }
#   }
#
# Auth (H3 remediation): tenant identity comes from an opaque, unguessable
# ``webhook_id`` capability token in the URL path — never from anything in
# the request body. Optionally layered with per-tenant HMAC-SHA256 signature
# verification (``chatwoot:webhook_hmac_secret``) and an inbox_id
# consistency cross-check (``chatwoot:inbox_id``). See
# ``POST /integrations/chatwoot/webhook/{webhook_id}`` below. A legacy,
# tokenless route (``POST /integrations/chatwoot/webhook``, no path param)
# is kept only for backward compatibility during migration — see its own
# docstring for exactly how it behaves under each signature mode.
# ---------------------------------------------------------------------------


def _reject_unauthorized(message: str, **extra: Any) -> HTTPException:
    """Log a webhook auth failure and return the uniform 401 to raise.

    Every distinct auth-failure reason (unknown webhook_id, HMAC failure in
    enforce mode, inbox_id mismatch, legacy route in enforce mode) gets the
    IDENTICAL status+detail on the wire — differentiated only in our own
    logs via ``extra`` — matching the uniform-rejection convention used by
    the LiveKit webhook (see src/api/livekit_routes.py). Callers use
    ``raise _reject_unauthorized(...)``. Never pass the full body or secrets
    in ``extra``.
    """
    log.warning(message, extra=extra)
    return HTTPException(status_code=401, detail="invalid webhook signature")


@router.post("/chatwoot/webhook/{webhook_id}")
async def chatwoot_webhook(
    webhook_id: str,
    request: Request,
) -> dict:
    """Chatwoot Agent Bot webhook — tenant identity comes from the
    unguessable ``webhook_id`` capability token in the URL path, never from
    the request body (see H3 remediation).

    Configure ``chatwoot:webhook_id`` for each tenant via the backoffice
    Chat tab or PATCH /tenants/{id}. Optionally also configure
    ``chatwoot:webhook_hmac_secret`` (HMAC-SHA256 signature verification)
    and/or ``chatwoot:inbox_id`` (defense-in-depth cross-check against the
    payload's own inbox id).
    """
    # Resolve tenant from the path segment BEFORE reading the request body at
    # all — an unknown webhook_id is rejected without ever looking at the
    # payload.
    resolver = getattr(request.app.state, "tenant_resolver", None) or auth_middleware._resolver
    tenant: TenantContext | None = None
    if resolver is not None and hasattr(resolver, "resolve_by_chatwoot_webhook_id"):
        tenant = await resolver.resolve_by_chatwoot_webhook_id(webhook_id)

    if tenant is None:
        raise _reject_unauthorized(
            "chatwoot webhook: unknown webhook_id",
            reason="unknown_webhook_id", webhook_id=webhook_id,
        )

    # Read raw bytes ourselves (not FastAPI's auto-parsed `payload: dict`) so
    # we have both the exact bytes for HMAC verification AND the parsed dict,
    # without risking a re-serialization mismatch between the two.
    raw_body = await request.body()
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise _reject_unauthorized(
            "chatwoot webhook: invalid JSON body",
            reason="invalid_json", tenant=tenant.slug,
        ) from None

    # json.loads accepts any JSON value, not just objects — a list, string,
    # number, null, or bool would all pass the try/except above and then blow
    # up on the first `.get()` call below with an unhandled AttributeError
    # (bare 500 on an unauthenticated endpoint, and a second oracle: 500 =
    # valid webhook_id + non-dict body vs 401 = unknown webhook_id). Reject
    # through the same uniform path as every other auth failure here.
    if not isinstance(payload, dict):
        raise _reject_unauthorized(
            "chatwoot webhook: non-dict JSON body",
            reason="non_dict_json_body", tenant=tenant.slug,
        )

    # HMAC layer — opt-in per tenant. Absence of a configured secret is the
    # normal/expected case (the webhook_id capability token is the primary
    # control), so it's silently skipped with no log line.
    hmac_secret = tenant.secret_optional("chatwoot:webhook_hmac_secret")
    if hmac_secret:
        try:
            verify_chatwoot(
                raw_body=raw_body,
                secret=hmac_secret,
                signature_header=request.headers.get("X-Chatwoot-Signature"),
                timestamp_header=request.headers.get("X-Chatwoot-Timestamp"),
            )
        except WebhookAuthError as e:
            if signature_mode() == "enforce":
                raise _reject_unauthorized(
                    "chatwoot webhook: HMAC verification failed",
                    reason=e.reason, tenant=tenant.slug,
                ) from None
            log.warning(
                "chatwoot webhook: HMAC check would have rejected (log_only mode)",
                extra={"reason": e.reason, "tenant": tenant.slug, "mode": "log_only"},
            )

    # inbox_id consistency cross-check (defense-in-depth, not the primary
    # auth boundary): the webhook_id path segment above is already
    # sufficient proof of authorization (capability token). But a legitimate
    # Chatwoot integration should never send a payload claiming a DIFFERENT
    # inbox than the one configured for this tenant, so a mismatch here is a
    # real signal of misuse — a valid capability token used with a payload
    # claiming a different inbox — and is treated as a hard reject rather
    # than a soft warn-and-proceed. Skipped entirely when no inbox_id is
    # configured for the tenant (nothing to compare against).
    configured_inbox_id = tenant.secret_optional("chatwoot:inbox_id")
    if configured_inbox_id:
        conversation = payload.get("conversation") or {}
        raw_inbox_id = (
            str((payload.get("inbox") or {}).get("id", ""))
            or str(conversation.get("inbox_id", ""))
        )
        if raw_inbox_id and str(configured_inbox_id) != raw_inbox_id:
            raise _reject_unauthorized(
                "chatwoot webhook: inbox_id mismatch",
                reason="inbox_id_mismatch", tenant=tenant.slug,
            )

    return await _handle_chatwoot_event(tenant, payload)


@router.post("/chatwoot/webhook")
async def chatwoot_webhook_legacy(
    request: Request,
) -> dict:
    """DEPRECATED legacy tokenless Chatwoot webhook route — superseded by
    ``POST /integrations/chatwoot/webhook/{webhook_id}`` (see H3 remediation).

    This route has no ``webhook_id`` path segment to resolve a tenant from at
    all — that absence is exactly the vulnerability being fixed (tenant
    identity was resolvable from a small, enumerable ``inbox_id`` embedded in
    the attacker-controlled request body). So in enforce mode (the default)
    it is hard-rejected without reading or parsing the body at all: this
    handler takes ``request: Request`` (not a ``payload: dict`` parameter),
    so FastAPI does not parse the body before routing here, and the enforce
    check below returns before ``await request.body()`` is ever called.

    In log_only mode it can't "process anyway" the way a signature-only
    check could — there's no tenant-independent secret to check against
    before a tenant is even resolved — so the only thing that changes in
    log_only mode is that every hit is now logged as a deprecated-path
    warning. The actual processing behavior below (resolve tenant by
    ``chatwoot:inbox_id`` extracted from the body, or fail with the existing
    "not mapped" response) is UNCHANGED from pre-remediation behavior,
    preserved deliberately so log_only mode doesn't produce a hard behavior
    change on top of a logging change. A malformed or non-dict JSON body in
    log_only mode is treated the same as "no inbox_id extracted" — it falls
    into the existing "not mapped" 200 response rather than crashing or
    inventing a new response shape.
    """
    if signature_mode() == "enforce":
        raise _reject_unauthorized(
            "chatwoot webhook: legacy tokenless route disabled",
            reason="legacy_route_disabled",
        )

    log.warning(
        "chatwoot webhook: legacy tokenless route hit (deprecated — will be "
        "hard-rejected once enforce mode is on; migrate to "
        "/integrations/chatwoot/webhook/{webhook_id})",
        extra={"reason": "legacy_route_deprecated", "mode": "log_only"},
    )

    # --- Unchanged pre-remediation behavior below: resolve tenant from
    # inbox_id embedded in the body (no bearer token, no webhook_id). Only
    # reached in log_only mode, so we parse the body ourselves here (rather
    # than via a `payload: dict` parameter) and treat malformed/non-dict JSON
    # as "no inbox_id extracted" — same as any other unmapped payload. ---
    raw_body = await request.body()
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        log.warning(
            "chatwoot webhook: legacy route received malformed/non-dict JSON body",
            extra={"reason": "invalid_json_body", "mode": "log_only"},
        )
        return {"ignored": True, "reason": "inbox_id not mapped to any tenant"}

    conversation = payload.get("conversation") or {}
    raw_inbox_id = (
        str((payload.get("inbox") or {}).get("id", ""))
        or str(conversation.get("inbox_id", ""))
    )
    log.info("chatwoot inbox_id extracted", extra={"inbox_id": raw_inbox_id})

    tenant: TenantContext | None = None
    if raw_inbox_id:
        resolver = getattr(request.app.state, "tenant_resolver", None) or auth_middleware._resolver
        if resolver is not None and hasattr(resolver, "resolve_by_chatwoot_inbox"):
            tenant = await resolver.resolve_by_chatwoot_inbox(raw_inbox_id)

    if tenant is None:
        log.warning(
            "chatwoot webhook: no tenant mapped to inbox_id — "
            "set chatwoot:inbox_id via backoffice Chat tab",
            extra={"inbox_id": raw_inbox_id},
        )
        # Return 200 so Chatwoot doesn't retry; we just can't process it yet.
        return {"ignored": True, "reason": "inbox_id not mapped to any tenant"}

    return await _handle_chatwoot_event(tenant, payload)


async def _handle_chatwoot_event(tenant: TenantContext, payload: dict) -> dict:
    """Given an already-authenticated tenant + parsed payload, filter to real
    incoming customer messages and kick off the background agent turn.

    Shared by both the ``webhook_id``-based route and the legacy route once
    each has resolved a tenant — auth (or lack thereof) is entirely the
    caller's concern; this helper only decides whether an event is one we
    act on.
    """
    event = payload.get("event", "")
    sender = payload.get("sender") or {}
    log.info("chatwoot webhook received", extra={
        "event": event,
        "message_type": payload.get("message_type"),
        "sender_type": sender.get("type"),
        "private": payload.get("private"),
        "payload_keys": list(payload.keys()),
    })

    # Only act on incoming customer messages.
    if event != "message_created":
        return {"ignored": True, "reason": f"event={event}"}

    # Chatwoot serializes message_type as int (0=incoming) in most versions,
    # but some versions/serializers use the string "incoming".
    message_type = payload.get("message_type")
    if message_type not in (0, "incoming"):
        return {"ignored": True, "reason": f"message_type={message_type}"}

    # Skip outgoing messages from agents/bots (e.g. our own replies triggering a webhook).
    # Chatwoot Agent Bot webhooks often omit sender.type for customer messages, so we
    # don't gate on sender.type == "contact" — message_type="incoming" is sufficient.
    if sender.get("type") in ("agent", "agent_bot"):
        return {"ignored": True, "reason": f"sender_type={sender.get('type')}"}

    text = (payload.get("content") or "").strip()
    if not text:
        return {"ignored": True, "reason": "empty content"}

    conversation = payload.get("conversation") or {}
    conversation_id = str(conversation.get("id", ""))
    if not conversation_id:
        log.warning("chatwoot webhook missing conversation.id")
        return {"ignored": True, "reason": "missing conversation.id"}

    # Log secrets_resolved keys (not values) to help diagnose missing credentials.
    log.info(
        "chatwoot tenant resolved",
        extra={
            "tenant": tenant.slug,
            "secrets_keys": list(tenant.secrets_resolved.keys()),
        },
    )

    # sender.identifier = contact's external_id (player UUID set by BetStudio)
    user_id = sender.get("identifier") or None
    customer_name = sender.get("name") or None

    log.info("chatwoot message received", extra={
        "tenant": tenant.slug, "conversation_id": conversation_id,
        "user_fp": token_fingerprint(user_id) if user_id else None,
        "has_user_id": user_id is not None, "text_len": len(text),
    })

    # Return 200 to Chatwoot immediately — Chatwoot has a ~10 s webhook timeout,
    # but LLM + CRM tool call + second LLM round easily exceeds that.
    # Processing runs in a background task with its own DB session.
    asyncio.create_task(
        _process_chatwoot_turn(tenant, conversation_id, text, user_id, customer_name)
    )
    return {"accepted": True}


async def _process_chatwoot_turn(
    tenant: TenantContext,
    conversation_id: str,
    text: str,
    user_id: Optional[str],
    customer_name: Optional[str],
) -> None:
    """Background task: run the agent turn and deliver the reply to Chatwoot.

    Uses its own DB session so the webhook handler can return 200 immediately
    without waiting for LLM + tool-call round-trips (which can exceed Chatwoot's
    10 s webhook timeout).
    """
    try:
        sm = get_sessionmaker()
        async with sm() as db:
            result = await external_message(
                ExternalMessageRequest(
                    conversation_id=f"chatwoot:{conversation_id}",
                    text=text,
                    user_id=user_id,
                    customer_name=customer_name,
                ),
                tenant=tenant,
                db=db,
            )
        log.info("chatwoot turn complete", extra={
            "tenant": tenant.slug, "conversation_id": conversation_id,
            "response_len": len(result.text),
        })
        await _chatwoot_send(tenant, conversation_id, result.text)
    except Exception:
        log.exception("chatwoot background turn failed", extra={
            "tenant": tenant.slug, "conversation_id": conversation_id,
        })


async def _chatwoot_send(tenant: TenantContext, conversation_id: str, text: str) -> None:
    """POST the bot reply to Chatwoot's messages API. Fails silently — a delivery
    error must not break the webhook acknowledgment Chatwoot is waiting for."""
    sr = tenant.secrets_resolved
    api_token = sr.get("chatwoot:api_token")
    account_id = sr.get("chatwoot:account_id")
    log.info(
        "chatwoot_send: credential check",
        extra={
            "tenant": tenant.slug,
            "has_api_token": bool(api_token),
            "has_account_id": bool(account_id),
            "all_secret_keys": list(sr.keys()),
        },
    )
    if not api_token or not account_id:
        log.warning("chatwoot credentials not configured — reply not delivered",
                    extra={"tenant": tenant.slug})
        return
    api_url = (sr.get("chatwoot:api_url") or "https://app.chatwoot.com").rstrip("/")
    url = f"{api_url}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    log.info("chatwoot_send: calling API", extra={"tenant": tenant.slug, "url": url})
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={
                "content": text,
                "message_type": "outgoing",
                "private": False,
            }, headers={"api_access_token": api_token})
        if resp.status_code >= 300:
            log.error("chatwoot delivery failed", extra={
                "tenant": tenant.slug, "status": resp.status_code,
                "conversation_id": conversation_id, "body": resp.text[:500],
            })
        else:
            log.info("chatwoot delivery ok", extra={
                "tenant": tenant.slug, "status": resp.status_code,
                "conversation_id": conversation_id,
            })
    except Exception:
        log.exception("chatwoot delivery error", extra={
            "tenant": tenant.slug, "conversation_id": conversation_id,
        })
