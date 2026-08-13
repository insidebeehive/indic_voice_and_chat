"""ChatBot endpoints.

Surfaces, all backed by the same ``ChatBotAgent``:

- ``POST /chat/sessions``            create a session (authed) → session_id + ws_url
- ``GET  /chat/sessions``            list this tenant's sessions
- ``GET  /chat/sessions/{id}``       session detail + message history
- ``WS   /chat/ws/{session_id}``     real-time conversation; the **session_id is
  the capability** — the WS resolves the owning tenant from the chat_sessions row,
  so the browser client needs no tenant credentials over the socket.
- ``POST /chat/message``             single-turn HTTP (WhatsApp / async channels),
  authenticated by tenant bearer token / ``X-Tenant-Slug`` header.
- ``GET  /chat/history/{session_id}`` retrieve the persisted Redis history.

The per-session agent is built by ``set_chatbot_factory(factory)``; the DB
sessionmaker (for the WS tenant lookup + message persistence) is injected via
``set_chat_sessionmaker`` (defaults to the app's global sessionmaker).
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import socket
import uuid
from typing import Awaitable, Callable, NoReturn, Optional
from urllib.parse import urlsplit

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.chatbot import ChatBotAgent, ChatTurnResult, MAX_HISTORY_TURNS
from src.api.chat_cost import compute_chat_turn_cost
from src.interfaces.media_storage import IMediaStorage
from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.dialogue.language import normalize_lang
from src.interfaces.llm import LLMMessage
from src.models.chat import ChatMessage, ChatSession

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# --- DI -----------------------------------------------------------------


# Factories may additionally accept an optional keyword-only ``customer_id``:
# the WS connect path passes it (from the ChatSession row it already fetched)
# so the factory can skip re-querying the same row on another pool checkout.
# Callers that don't have the row omit it and the factory falls back to a
# DB lookup.
ChatBotFactory = Callable[[TenantContext, str], Awaitable[ChatBotAgent]]
_factory: Optional[ChatBotFactory] = None
_sessionmaker: object | None = None
_handoff_store: object | None = None  # SessionStore: carries chat→voice handoff context

# Per-session queues bridging customer WS ↔ BO agent WS in handover mode.
# _bo_queues:       customer WS puts here → BO agent WS reads
# _customer_queues: BO agent WS puts here → customer WS reads
_bo_queues: dict = {}
_customer_queues: dict = {}
_media_store: Optional[IMediaStorage] = None

# Active voice-call WebSockets keyed by chat_session_id.
# Populated by chat_voice_ws; cleared by _end_session so ending the chat
# also terminates any live voice call for the same session.
_active_voice_ws: dict[str, WebSocket] = {}


def set_chat_handoff_store(store: object | None) -> None:
    """Inject the Redis-backed store the chat→voice handoff uses to pass the
    chat summary into the voice agent."""
    global _handoff_store
    _handoff_store = store


def set_media_store(store: Optional[IMediaStorage]) -> None:
    """Inject (or clear) the S3-compatible media store used to persist chat media."""
    global _media_store
    _media_store = store


def _mime_ext(mime: str) -> str:
    mapping = {
        "audio/webm": "webm", "audio/ogg": "ogg",
        "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/wav": "wav",
        "image/jpeg": "jpg", "image/png": "png",
        "image/gif": "gif", "image/webp": "webp",
        "video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov",
    }
    base = mime.split(";")[0].strip().lower()
    return mapping.get(base) or base.split("/")[-1]


def _media_key(tenant_id: str, session_id: str, mime: str) -> str:
    ext = _mime_ext(mime)
    return f"chat/{tenant_id}/{session_id}/{uuid.uuid4().hex}.{ext}"


# Ceiling on processing one chat turn (LLM + tools + RAG). Without it a hung
# provider call under load piles up pending tasks indefinitely — the only other
# timeout in the WS flow guards waiting for INPUT, not processing. Timeouts
# surface through the existing per-turn error handler (socket stays open).
# Sized above the LLM adapter's worst-case escalating 429 schedule (~35s of
# backoff + several request RTTs across a multi-round tool turn): at 60s, a
# 200-parallel load test still timed out 10/200 turns that would have
# succeeded moments later; at 90s all 200 complete.
_TURN_TIMEOUT_S = 90.0


async def _run_turn(coro: Awaitable):
    return await asyncio.wait_for(coro, timeout=_TURN_TIMEOUT_S)


# --- Long-turn keepalive -------------------------------------------------

# A turn that makes CRM tool calls can block for tens of seconds (the tools run
# sequentially across up to _max_tool_rounds rounds — src/agents/chatbot.py),
# during which the socket carried no application traffic at all: one `typing`
# frame at the start, then nothing until the reply. The CRM's downstream relay
# reads that silence as a dead connection (abnormal close 1006), guesses
# `reason: idle_timeout` and closes the ticket while the bot is still working.
# Transport-level pings do not help: uvicorn already sends WS pings every 20s
# by default (ws_ping_interval) and the relay closed anyway, and Starlette
# exposes no ping API to us regardless. So repeat the frame the CRM contract
# already documents (docs/crm-chat-media-contract.md) — no new frame type, and
# consumers already treat `typing` as idempotent.
_KEEPALIVE_INTERVAL_S = 8.0
# Grace for an in-flight keepalive send to finish once the turn is done, before
# cancelling it outright. Bounded so a wedged socket can never delay a reply.
_KEEPALIVE_DRAIN_S = 1.0


async def _typing_keepalive(websocket: WebSocket, stop: asyncio.Event) -> None:
    """Emit ``{"type":"typing"}`` every _KEEPALIVE_INTERVAL_S until ``stop``.

    Parks in ``stop.wait()`` between frames rather than sleeping blindly, so
    the common exit path is a clean return with no frame in flight — never a
    cancellation landing in the middle of a send. Never raises out: a send
    failure (client already gone) must not mask the real turn's result, and
    the turn path detects a dead socket on its own.
    """
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=_KEEPALIVE_INTERVAL_S)
            return  # turn finished — must not emit another frame
        except asyncio.TimeoutError:
            pass
        if stop.is_set():
            return
        try:
            await websocket.send_text(json.dumps({"type": "typing"}))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — keepalive is strictly best-effort
            log.debug("typing keepalive send failed; stopping", exc_info=True)
            return


async def _stop_keepalive(ka: asyncio.Task, stop: asyncio.Event) -> None:
    """Cancellation-safe teardown for a keepalive task started via
    ``asyncio.ensure_future(_typing_keepalive(websocket, stop))``.

    Uses ``asyncio.wait`` (not a bare ``await ka``) for BOTH the drain and the
    final reap: ``asyncio.wait`` never propagates the awaited task's own
    result/exception into the caller, it only reports back which tasks
    finished — so any `CancelledError` raised out of an `asyncio.wait` call
    here is unambiguously the ENCLOSING task's own cancellation, never `ka`'s,
    and must always propagate. (A bare `await ka` does not have this
    property: if the enclosing task is cancelled while suspended there,
    CPython delivers the cancellation by cancelling `ka` itself since it's
    the task's current `_fut_waiter`, making `ka.cancelled()` true and
    indistinguishable from "ka finished via its own unrelated cancellation"
    — that ambiguity was a real bug in an earlier version of this function.)

    Idempotent: safe to call more than once with the same ``ka``/``stop`` —
    once ``ka`` is done, every step below becomes a no-op.
    """
    stop.set()
    if not ka.done():
        # Normally resolves on the next tick (the loop is parked in
        # stop.wait()). The bounded wait matters only if it is mid-send,
        # where a cancel could truncate a frame on the wire.
        try:
            await asyncio.wait({ka}, timeout=_KEEPALIVE_DRAIN_S)
        except asyncio.CancelledError:
            ka.cancel()
            raise  # this is OUR cancellation — must propagate, not be absorbed
        if not ka.done():
            ka.cancel()
    if not ka.done():
        # ka.cancel() above hasn't necessarily taken effect yet — wait for it
        # to actually finish. Still via asyncio.wait, same reasoning as above.
        await asyncio.wait({ka})
    if not ka.cancelled():
        # Retrieve ka's own exception (if any) so asyncio doesn't log it as
        # "exception was never retrieved" — but never let it propagate here,
        # a keepalive-internal failure must never mask the turn's result.
        exc = ka.exception()
        if exc is not None:
            log.debug("keepalive task ended with an exception", exc_info=exc)


async def _run_turn_with_keepalive(websocket: WebSocket, coro):
    """``_run_turn`` plus periodic ``typing`` frames while the turn is in flight.

    The keepalive is fully torn down before this returns *or* raises, so no
    `typing` frame can land after the reply — or after the `error` frame the
    per-turn handler sends on failure. Nothing in the teardown can replace the
    turn's own result or exception. A cancellation of the enclosing task that
    lands while teardown is itself in flight (see ``_stop_keepalive``) is
    re-raised rather than absorbed, so a real cancellation of this whole
    function is never mistaken for a clean return.
    """
    stop = asyncio.Event()
    # Turn task first: if task creation somehow failed after the keepalive
    # existed we would leak an un-awaited coroutine.
    turn = asyncio.ensure_future(_run_turn(coro))
    ka = asyncio.ensure_future(_typing_keepalive(websocket, stop))
    try:
        return await turn
    finally:
        await _stop_keepalive(ka, stop)


def _classify_turn_error(exc: Exception) -> tuple[str, str]:
    """Map a failed chat turn's exception to (reason, customer_message).

    The billing-cap case is checked before the generic 429 branch: both are
    ``google.genai.errors.ClientError`` with ``.code == 429``, but a spending
    cap doesn't clear on its own the way a per-minute/per-day quota does, so
    telling the customer to "try again in a little while" is actively
    misleading and the ops signal needs to be distinct (see the extra log
    line at the call site).
    """
    if getattr(exc, "code", None) == 429:
        if "spending cap" in str(exc).lower():
            return "llm_billing", (
                "The AI assistant is temporarily unavailable due to a "
                "service limit on our side. Our team has been notified — "
                "please try again later."
            )
        return "llm_quota", ("We're experiencing very high demand right now — "
                              "please try again in a little while.")
    # asyncio.TimeoutError is TimeoutError on 3.11+, but _run_turn wraps
    # every provider call so either spelling can surface depending on where
    # the underlying client raises.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout", "That took longer than expected — please try again."
    return "internal", "Sorry, something went wrong — please try again."


def _raise_turn_http_error(exc: Exception, session_id: str) -> NoReturn:
    """Map a failed REST chat turn to an honest HTTPException (never a bare 500).

    Mirrors the WS per-turn error handling in ``chat_websocket`` — same
    (reason, message) classification and the same ops log line for a billing
    cap, just surfaced as an HTTP status instead of an ``error`` WS frame."""
    reason, message = _classify_turn_error(exc)
    log.exception("chat REST turn failed", extra={"session_id": session_id, "reason": reason})
    if reason == "llm_billing":
        # Distinct from ordinary quota noise: this needs a human to raise the
        # cap, it won't clear on its own.
        log.error(
            "gemini monthly spending cap exceeded — raise it at "
            "https://ai.studio/spend to restore chat",
            extra={"session_id": session_id},
        )
    status = {"llm_billing": 503, "llm_quota": 503, "timeout": 504}.get(reason, 500)
    raise HTTPException(status_code=status, detail={"message": message, "reason": reason})


# Background webhook deliveries. Tasks must be referenced until done or the
# event loop may GC them mid-flight.
_webhook_tasks: set[asyncio.Task] = set()


def _spawn_webhook_task(coro: Awaitable) -> None:
    task = asyncio.ensure_future(coro)
    _webhook_tasks.add(task)
    task.add_done_callback(_webhook_tasks.discard)


_MAX_MEDIA_FETCH_BYTES = 1 * 1024 * 1024  # cap for media pulled via media_url
_MEDIA_FETCH_TIMEOUT = httpx.Timeout(20.0, connect=10.0)


def _is_public_host(hostname: str) -> bool:
    """Reject hostnames resolving to private/loopback/link-local addresses —
    media_url is untrusted client input (unlike vendor recording URLs
    elsewhere in this codebase), so this guards against using it to probe
    internal network services."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = info[4][0]
        ip = ipaddress.ip_address(addr)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


async def _fetch_media_url(url: str) -> tuple[bytes, str]:
    """Fetch a caller-supplied media URL (e.g. a presigned R2/S3 URL) server-side.

    Streams the body with a byte-size cap and requires the response
    Content-Type to be image/*, video/* or audio/* before accepting it.
    Redirects are not followed (the default httpx behavior) to avoid an SSRF
    bypass via a redirect chain."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("media_url must be an https URL")
    if not _is_public_host(parts.hostname):
        raise ValueError("media_url resolves to a non-public address")

    async with httpx.AsyncClient(timeout=_MEDIA_FETCH_TIMEOUT) as client:
        async with client.stream("GET", url) as resp:
            if resp.is_redirect:
                raise ValueError("media_url redirects are not followed")
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if not (content_type.startswith("image/") or content_type.startswith("video/")
                    or content_type.startswith("audio/")):
                raise ValueError(f"unsupported content-type: {content_type or 'unknown'}")
            chunks = bytearray()
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                chunks.extend(chunk)
                if len(chunks) > _MAX_MEDIA_FETCH_BYTES:
                    raise ValueError("media_url content exceeds size limit")
            return bytes(chunks), content_type


def set_chatbot_factory(factory: Optional[ChatBotFactory]) -> None:
    """Register / unregister the per-session agent factory."""
    global _factory
    _factory = factory


def set_chat_sessionmaker(sessionmaker: object | None) -> None:
    """Inject the async sessionmaker the WS uses to resolve/persist sessions
    (tests pass an in-memory one; prod falls back to the app's global)."""
    global _sessionmaker
    _sessionmaker = sessionmaker


def _sm():
    from src.models.database import get_sessionmaker
    return _sessionmaker or get_sessionmaker()


async def _get_agent(tenant: TenantContext, session_id: str) -> ChatBotAgent:
    if _factory is None:
        raise HTTPException(
            status_code=503,
            detail="chatbot factory not initialized; set_chatbot_factory() not called",
        )
    return await _factory(tenant, session_id)


def _scoped_session(tenant: TenantContext, session_id: str) -> str:
    """Namespace the Redis session id by tenant so two tenants can use the same id."""
    return f"{tenant.id}:{session_id}"


# Roles in chat_messages that carry conversational content worth replaying.
_HYDRATE_ROLES = {"customer", "agent", "human_agent"}
# How many persisted messages to replay on connect. ChatBotAgent._compose only
# ever sends the last MAX_HISTORY_TURNS exchanges to the LLM
# (src/agents/chatbot.py), so loading more than that window is dead weight —
# this keeps the query O(1) regardless of how long-lived the session is.
_HYDRATE_LIMIT = 2 * MAX_HISTORY_TURNS


async def _hydrate_agent_history(agent: ChatBotAgent, session_id: str, tenant_id: str) -> int:
    """Replay this session's persisted transcript into a freshly-built agent.

    The factory always hands back an AgentSession with turns=[] (see
    make_chatbot_factory in src/bootstrap.py), so an agent built for an
    *existing* session_id starts with no memory of it. Without this, a browser
    reconnect — network drop, backgrounded tab, page reload, all routine on
    mobile — makes the bot answer the next message as if the conversation had
    just begun (a full re-greeting to a bare "Ok sir"), even though the whole
    transcript is sitting in chat_messages.

    ``tenant_id`` scopes the query to messages belonging to a session owned by
    that tenant (joined through ``ChatSession``). This is required even though
    most call sites already derive ``session_id`` from a tenant-validated
    ``ChatSession`` row — the REST ``/chat/message`` endpoint takes
    ``session_id`` straight from client input, and without this filter a
    caller could replay another tenant's persisted transcript into their own
    LLM context by guessing/reusing a foreign session_id. A session_id that
    doesn't belong to ``tenant_id`` degrades to the same "no history" result
    as a nonexistent session_id, not an error.

    Best-effort: a hydration failure degrades to the old memory-less behaviour
    rather than failing the connect. Returns the number of turns replayed.
    """
    session = getattr(agent, "session", None)
    if session is None:
        # Not a real ChatBotAgent — e.g. a minimal test double standing in for
        # one (some chat_websocket tests swap in a bare stub exposing only
        # handle_message/summarize_session). Nothing to hydrate into.
        return 0
    if session.turns:  # already populated (e.g. request_call's own load)
        return 0
    try:
        async with _sm()() as db:
            rows = (await db.execute(
                select(ChatMessage)
                .join(ChatSession, ChatMessage.session_id == ChatSession.id)
                .where(
                    ChatMessage.session_id == session_id,
                    ChatSession.tenant_id == tenant_id,
                )
                .order_by(ChatMessage.id.desc())
                .limit(_HYDRATE_LIMIT)
            )).scalars().all()
    except Exception:  # noqa: BLE001 — memory is a nicety, the socket is not
        log.exception("chat history hydration failed", extra={"session_id": session_id})
        return 0

    turns: list[LLMMessage] = []
    for m in reversed(rows):  # id DESC + LIMIT gives the newest window; flip it
        if m.role not in _HYDRATE_ROLES:
            continue
        content = (m.content or "").strip()
        if not content:
            continue
        # human_agent -> assistant: a human took over mid-chat, but from the
        # model's point of view those lines are still "what my side already
        # said", so replaying them as assistant turns stops the bot repeating
        # help a human already gave.
        role = "user" if m.role == "customer" else "assistant"
        if not turns and role != "user":
            # Never open the replayed window on an assistant turn: the LIMIT
            # can cut mid-pair, and Anthropic rejects a messages[] whose first
            # entry is an assistant message (see providers/llm/anthropic_claude.py,
            # which passes roles through verbatim).
            continue
        if turns and turns[-1].role == role:
            # Consecutive same-role rows (an unanswered voice note, a burst of
            # human-agent lines) — merge so the replay stays strictly
            # alternating, which every provider accepts.
            turns[-1] = LLMMessage(role=role, content=f"{turns[-1].content}\n{content}")
            continue
        turns.append(LLMMessage(role=role, content=content))
    if turns and turns[-1].role == "user":
        # _compose appends the incoming user message after this window; leaving
        # a trailing unanswered customer line would put two user turns
        # back to back. The dropped line is at most one unanswered message.
        turns.pop()

    session.turns.extend(turns)
    if turns:
        log.info("chat history hydrated", extra={
            "session_id": session_id, "turns": len(turns)})
    return len(turns)


def new_session_id() -> str:
    return f"cs_{uuid.uuid4().hex[:16]}"


# Keep private alias so existing callers don't break.
_new_session_id = new_session_id


def _ws_base(conn) -> str:
    """Build the WebSocket API base from any HTTPConnection (Request or WebSocket)."""
    base = conn.headers.get("x-forwarded-host") or conn.url.netloc
    proto = conn.headers.get("x-forwarded-proto")
    scheme = "wss" if (proto == "https" or conn.url.scheme == "https") else "ws"
    return f"{scheme}://{base}/api/v1"


def _ws_url(request: Request, session_id: str) -> str:
    return f"{_ws_base(request)}/chat/ws/{session_id}"


def _voice_call_url(conn, tenant_slug: str, token: str) -> str:
    return f"{_ws_base(conn)}/chat/voice?tenant={tenant_slug}&handoff={token}"


# First line the customer sees on session creation. Keys are the base language
# codes from the CRM contract (docs/crm-chat-media-contract.md) -- note "od" for
# Odia, this platform's non-standard code, matching the Sarvam/IndicF5 catalogs.
# "{who}" expands to " <name>" or "" -- it always sits directly after the
# greeting word in every language here, so the no-name form reads naturally.
_GREETINGS: dict[str, str] = {
    "en": "Hello{who}, how can I help?",
    "hi": "नमस्ते{who}, मैं आपकी कैसे मदद करूँ?",
    "bn": "নমস্কার{who}, আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
    "gu": "નમસ્તે{who}, હું તમને કેવી રીતે મદદ કરી શકું?",
    "kn": "ನಮಸ್ಕಾರ{who}, ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "ml": "നമസ്കാരം{who}, ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",
    "mr": "नमस्कार{who}, मी तुमची कशी मदत करू?",
    "od": "ନମସ୍କାର{who}, ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ{who}, ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰਾਂ?",
    "ta": "வணக்கம்{who}, நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "te": "నమస్కారం{who}, నేను మీకు ఎలా సహాయం చేయగలను?",
    "as": "নমস্কাৰ{who}, মই আপোনাক কেনেকৈ সহায় কৰিব পাৰোঁ?",
}


def _greeting(company: str, customer_name: Optional[str], language: str) -> str:
    """Opening line, in the session's language.

    ``language`` is already tenant-default-resolved by the caller, so an
    unmapped code falls back to English rather than to the tenant default --
    there is no better mapped candidate at this point.
    """
    name = (customer_name or "").strip()
    who = f" {name}" if name else ""
    template = _GREETINGS.get(normalize_lang(language)) or _GREETINGS["en"]
    return template.format(who=who)


# --- Schemas ------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None
    customer_name: Optional[str] = None
    language: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    session_id: str
    greeting: str
    ws_url: str


class SessionSummary(BaseModel):
    session_id: str
    status: str
    mode: str
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    language: str
    message_count: int
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionMessage(BaseModel):
    role: str
    type: str
    content: str
    sources: Optional[list] = None
    timestamp: Optional[str] = None


class SessionDetailResponse(SessionSummary):
    summary: Optional[str] = None
    messages: list[SessionMessage] = []


class ChatMessageRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1)


class ChatMessageResponse(BaseModel):
    session_id: str
    response_text: str
    language: str
    confidence: str
    sources_used: list[str]
    action: str
    suggested_followups: list[str] = []


class HistoryEntry(BaseModel):
    role: str
    content: str
    metadata: Optional[dict] = None


class HistoryResponse(BaseModel):
    session_id: str
    history: list[HistoryEntry]


# --- Session lifecycle (authed) ----------------------------------------


@router.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    req: CreateSessionRequest,
    request: Request,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> CreateSessionResponse:
    session_id = _new_session_id()
    language = (
        normalize_lang(req.language)
        or getattr(tenant.settings, "default_language", None)
        or "hi"
    )
    session.add(ChatSession(
        id=session_id, tenant_id=tenant.id,
        customer_id=req.user_id, customer_name=req.customer_name,
        language=language, status="active", extra_data=req.metadata or {},
    ))
    await session.commit()
    from src.api.chat_webhooks import send_bo_webhook
    # Fire-and-forget: the webhook is a notification, not a dependency of
    # session creation. Awaiting it inline stalls every create for up to
    # ~16s (3 attempts × 5s) when a tenant's CRM endpoint is slow/down —
    # under a session burst that alone wedges the whole flow. Delivery
    # failures are logged inside send_bo_webhook.
    _spawn_webhook_task(send_bo_webhook(tenant, "session_started", {
        "session_id": session_id,
        "customer": {"name": req.customer_name, "id": req.user_id},
    }))
    return CreateSessionResponse(
        session_id=session_id,
        greeting=_greeting(tenant.name, req.customer_name, language),
        ws_url=_ws_url(request, session_id),
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    status: Optional[str] = None,
    mode: Optional[str] = None,
    customer_id: Optional[str] = None,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> SessionListResponse:
    q = select(ChatSession).where(ChatSession.tenant_id == tenant.id)
    if status:
        q = q.where(ChatSession.status == status)
    if mode:
        q = q.where(ChatSession.mode == mode)
    if customer_id:
        q = q.where(ChatSession.customer_id == customer_id)
    q = q.order_by(ChatSession.started_at.desc())
    rows = (await session.execute(q)).scalars().all()
    return SessionListResponse(sessions=[_summary(r) for r in rows])


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str = Path(min_length=1),
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> SessionDetailResponse:
    row = await session.get(ChatSession, session_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="chat session not found")
    msgs = (await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )).scalars().all()
    base = _summary(row).model_dump()
    return SessionDetailResponse(
        **base, summary=row.summary,
        messages=[SessionMessage(
            role=m.role, type=m.type, content=m.content,
            sources=list(m.sources) if m.sources else None,
            timestamp=m.created_at.isoformat() if m.created_at else None,
        ) for m in msgs],
    )


def _summary(row: ChatSession) -> SessionSummary:
    return SessionSummary(
        session_id=row.id, status=row.status, mode=row.mode,
        customer_id=row.customer_id, customer_name=row.customer_name,
        language=row.language, message_count=row.message_count,
        started_at=row.started_at.isoformat() if row.started_at else None,
        ended_at=row.ended_at.isoformat() if row.ended_at else None,
    )


# --- HTTP single-turn (header-authed; WhatsApp / async) ----------------


@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(
    req: ChatMessageRequest, tenant: TenantContext = Depends(current_tenant),
) -> ChatMessageResponse:
    session_id = req.session_id or _new_session_id()
    agent = await _get_agent(tenant, _scoped_session(tenant, session_id))
    if req.session_id:
        await _hydrate_agent_history(agent, session_id, tenant.id)
    try:
        result = await agent.handle_message(req.message)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — map provider failures to honest HTTP errors
        _raise_turn_http_error(exc, session_id)
    await _persist_turn(session_id, req.message, result)
    await _emit_escalation(tenant.id, session_id, result)
    return _to_message_response(session_id, result)


@router.post("/{session_id}/upload", response_model=ChatMessageResponse)
async def upload_media(
    session_id: str,
    file: UploadFile = File(...),
    text: str = Form(""),
    session: AsyncSession = Depends(get_db_session),
) -> ChatMessageResponse:
    """Multipart image/video upload (PRD §4.3). Capability model: the tenant is
    resolved from the session row (same as the WS), so no auth header is needed.
    Processes synchronously and returns the agent's reply."""
    from src.auth.middleware import tenant_from_id

    row = await session.get(ChatSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    tenant = await tenant_from_id(row.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=503, detail="tenant unavailable")
    if _factory is None:
        raise HTTPException(status_code=503, detail="chatbot factory not initialized")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    mime = file.content_type or "application/octet-stream"
    agent = await _factory(tenant, _scoped_session(tenant, session_id))
    await _hydrate_agent_history(agent, session_id, tenant.id)
    try:
        result = await agent.handle_image(data, mime, text)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — map provider failures to honest HTTP errors
        _raise_turn_http_error(exc, session_id)
    await _persist_turn(
        session_id, text or f"[{mime}]", result,
        user_type=("image" if mime.startswith("image/") else "video"), media_mime=mime)
    await _emit_escalation(tenant.id, session_id, result)
    return _to_message_response(session_id, result)


class CallHandoffResponse(BaseModel):
    call_url: str
    call_id: str


@router.post("/{session_id}/call", response_model=CallHandoffResponse)
async def request_call(
    session_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> CallHandoffResponse:
    """Chat→voice handoff (PRD §4.4): summarize the chat, stash the context under
    a short-lived token, and return a browser-voice call URL. Capability model:
    the tenant is resolved from the session row (no auth header)."""
    from src.auth.middleware import tenant_from_id

    row = await session.get(ChatSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    tenant = await tenant_from_id(row.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=503, detail="tenant unavailable")
    if _factory is None or _handoff_store is None:
        raise HTTPException(status_code=503, detail="chat voice handoff not initialized")

    # Summarize the conversation so far (from persisted messages) for the voicebot.
    agent = await _factory(tenant, _scoped_session(tenant, session_id))
    msgs = (await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
    )).scalars().all()
    for m in msgs:
        role = "user" if m.role == "customer" else ("assistant" if m.role == "agent" else m.role)
        agent.session.turns.append(LLMMessage(role=role, content=m.content))
    summary = await agent.summarize_session()

    token = uuid.uuid4().hex
    context = {
        "chat_session_id": session_id,
        "customer_name": row.customer_name,
        "customer_id": row.customer_id,
        "language": row.language,
        "chat_summary": summary,
    }
    await _handoff_store.redis.set(
        f"chat_handoff:{token}", json.dumps(context), ex=600)  # 10-min TTL
    return CallHandoffResponse(
        call_url=_voice_call_url(request, tenant.slug, token), call_id=token)


@router.websocket("/voice")
async def chat_voice_ws(websocket: WebSocket) -> None:
    """Always-on browser-voice WS for the chat→voice handoff. Reuses the dev
    console's browser bridge (un-gated). Tenant from ?tenant; handoff context
    from ?handoff (resolved by the bridge factory). The dev console's own
    /dev/voice stays behind VOX_DEV_CONSOLE."""
    from src.api.dev_console import run_browser_voice
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(websocket.query_params.get("tenant", ""))
    except Exception:  # noqa: BLE001
        await websocket.close(code=1008, reason="unknown tenant")
        return

    # Resolve chat_session_id from the handoff token so we can register this WS.
    # If the chat session ends while the call is live, _end_session will close us.
    chat_session_id: Optional[str] = None
    handoff_token = (websocket.query_params.get("handoff") or "").strip()
    if handoff_token and _handoff_store is not None:
        try:
            raw = await _handoff_store.redis.get(f"chat_handoff:{handoff_token}")
            if raw:
                chat_session_id = json.loads(raw).get("chat_session_id")
        except Exception:  # noqa: BLE001
            pass
    if chat_session_id:
        _active_voice_ws[chat_session_id] = websocket
    try:
        await run_browser_voice(websocket, tenant)
    finally:
        _active_voice_ws.pop(chat_session_id, None)


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def chat_history(
    session_id: str = Path(min_length=1),
    tenant: TenantContext = Depends(current_tenant),
) -> HistoryResponse:
    agent = await _get_agent(tenant, _scoped_session(tenant, session_id))
    raw = await agent.get_history()
    return HistoryResponse(
        session_id=session_id,
        history=[HistoryEntry(**h) for h in raw],
    )


@router.get("/media/{message_id}")
async def get_media(
    message_id: int,
    session: AsyncSession = Depends(get_db_session),
    authorization: Optional[str] = Header(default=None),
    session_id: Optional[str] = Query(default=None),
) -> RedirectResponse:
    """Serve a signed URL for a chat media message.

    Accepts either:
      Authorization: Bearer <token>   (CRM / programmatic)
      ?session_id=<sid>               (widget / BO console HTML elements)
    """
    from src.auth.middleware import tenant_from_bearer_token

    if _media_store is None:
        raise HTTPException(status_code=503, detail="media storage not configured")

    msg = await session.get(ChatMessage, message_id)
    if msg is None or not msg.media_url:
        raise HTTPException(status_code=404, detail="media not found")

    authed = False
    if authorization and authorization.startswith("Bearer "):
        try:
            tenant = await tenant_from_bearer_token(authorization[len("Bearer "):])
        except Exception:  # noqa: BLE001
            tenant = None
        if tenant is not None:
            chat_row = await session.get(ChatSession, msg.session_id)
            if chat_row and chat_row.tenant_id == tenant.id:
                authed = True

    if not authed and session_id and msg.session_id == session_id:
        authed = True

    if not authed:
        raise HTTPException(status_code=401, detail="unauthorized")

    ttl = 3600
    try:
        from src.config import get_settings
        cfg = get_settings().media_storage
        if cfg is not None:
            ttl = cfg.signed_url_ttl_seconds
    except Exception:  # noqa: BLE001
        pass

    url = await _media_store.signed_url(msg.media_url, ttl_seconds=ttl)
    return RedirectResponse(url=url, status_code=302)


@router.get("/local-media/{key:path}", include_in_schema=False)
async def serve_local_media(key: str = Path(...)):
    """Serve a media blob from in-memory storage (used when S3 is not configured)."""
    from fastapi.responses import Response
    from src.providers.media.local import LocalMediaStorage

    if not isinstance(_media_store, LocalMediaStorage):
        raise HTTPException(status_code=404, detail="local media not available")

    entry = _media_store.get(key)
    if entry is None:
        raise HTTPException(status_code=404, detail="media not found")

    data, content_type = entry
    return Response(content=data, media_type=content_type)


# --- BO handover: claim + agent WebSocket --------------------------------


class ClaimRequest(BaseModel):
    agent_id: str
    agent_name: str


class ClaimResponse(BaseModel):
    status: str
    agent_id: str


@router.post("/sessions/{session_id}/claim", response_model=ClaimResponse)
async def claim_session(
    session_id: str,
    req: ClaimRequest,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> ClaimResponse:
    """BO agent claims an awaiting_human session; 409 if already claimed; 400 if wrong mode."""
    row = await session.get(ChatSession, session_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="chat session not found")
    if row.mode == "human":
        raise HTTPException(
            status_code=409,
            detail=f"session already claimed by {row.claimed_by or 'another agent'}",
        )
    if row.mode != "awaiting_human":
        raise HTTPException(
            status_code=400,
            detail=f"session is not awaiting handover (mode: {row.mode})",
        )
    from datetime import datetime, timezone
    row.mode = "human"
    row.claimed_by = req.agent_id
    row.claimed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await session.commit()
    cq = _customer_queues.get(session_id)
    if cq:
        await cq.put(json.dumps({
            "type": "mode_change", "mode": "human",
            "agent_name": req.agent_name,
        }))
    log.info("session claimed", extra={"session_id": session_id, "agent": req.agent_id})
    return ClaimResponse(status="claimed", agent_id=req.agent_id)


class DeclineResponse(BaseModel):
    status: str


@router.post("/sessions/{session_id}/decline", response_model=DeclineResponse)
async def decline_session(
    session_id: str,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> DeclineResponse:
    """CRM calls this when it cannot assign a human agent to an awaiting_human session.
    Reverts the session to bot mode and signals the customer WS to resume bot conversation."""
    row = await session.get(ChatSession, session_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="chat session not found")
    if row.mode != "awaiting_human":
        raise HTTPException(
            status_code=400,
            detail=f"session is not awaiting handover (mode: {row.mode})",
        )
    row.mode = "bot"
    await session.commit()
    cq = _customer_queues.get(session_id)
    if cq:
        await cq.put(json.dumps({"type": "declined"}))
    log.info("session declined by CRM", extra={"session_id": session_id})
    return DeclineResponse(status="declined")


@router.websocket("/sessions/{session_id}/agent-ws")
async def agent_websocket(websocket: WebSocket, session_id: str) -> None:
    """BO agent WebSocket. Auth via ?token= query param (tenant bearer token).
    On connect sends full history; then forwards customer messages to BO and
    BO replies to the customer WS. BO sends {type:'reply', text:'...'} frames."""
    await websocket.accept()

    token = (websocket.query_params.get("token") or "").strip()
    if not token:
        await websocket.close(code=1008, reason="missing token")
        return
    from src.auth.middleware import tenant_from_bearer_token
    try:
        tenant = await tenant_from_bearer_token(token)
    except Exception:  # noqa: BLE001
        await websocket.close(code=1008, reason="invalid token")
        return
    if tenant is None:
        await websocket.close(code=1008, reason="invalid token")
        return

    async with _sm()() as db:
        row = await db.get(ChatSession, session_id)
    if row is None or row.tenant_id != tenant.id:
        await websocket.close(code=4004, reason="session not found")
        return
    if row.mode not in ("awaiting_human", "human"):
        await websocket.close(code=4004, reason="session not in handover mode")
        return

    # Send full history on connect
    async with _sm()() as db:
        msgs = (await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        )).scalars().all()
    await websocket.send_text(json.dumps({
        "type": "history",
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "text": m.content,
                "media_url": (f"/api/v1/chat/media/{m.id}" if m.media_url else None),
                "media_mime": m.media_mime,
                "ts": m.created_at.isoformat() if m.created_at else None,
            }
            for m in msgs
        ],
    }))

    bq: asyncio.Queue = _bo_queues.setdefault(session_id, asyncio.Queue())
    cq: asyncio.Queue = _customer_queues.setdefault(session_id, asyncio.Queue())

    try:
        while True:
            ws_task = asyncio.ensure_future(websocket.receive_text())
            bq_task = asyncio.ensure_future(bq.get())
            done, pending = await asyncio.wait(
                {ws_task, bq_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

            if bq_task in done:
                item = bq_task.result()
                if item is None:  # customer WS dropped — notify agent but keep loop alive
                    await websocket.send_text(json.dumps({
                        "type": "system", "text": "Customer disconnected — session still open"}))
                else:
                    await websocket.send_text(item)

            if ws_task in done:
                try:
                    raw = ws_task.result()
                except WebSocketDisconnect:
                    break
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                mtype = msg.get("type", "reply")
                if mtype == "reply":
                    text = (msg.get("text") or "").strip()
                    if not text:
                        continue
                    async with _sm()() as db:
                        r2 = await db.get(ChatSession, session_id)
                        if r2:
                            db.add(ChatMessage(
                                session_id=session_id, role="human_agent",
                                type="text", content=text,
                            ))
                            r2.message_count = (r2.message_count or 0) + 1
                            await db.commit()
                    await cq.put(json.dumps({"type": "message", "text": text, "from": "human_agent"}))
                elif mtype == "end":
                    await cq.put(json.dumps({"type": "ended"}))
                    await _end_session(session_id, "Ended by support agent")
                    break
    except WebSocketDisconnect:
        log.info("agent ws disconnected", extra={"session_id": session_id})
    except Exception as _exc:  # noqa: BLE001
        log.error(
            "agent ws crashed: %s: %s", type(_exc).__name__, _exc,
            extra={"session_id": session_id},
        )
        log.exception("agent ws crashed (full tb)", extra={"session_id": session_id})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# --- WebSocket (session_id is the capability) --------------------------


@router.websocket("/ws/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    if _factory is None:
        await websocket.close(code=1011, reason="chatbot factory unset")
        return

    # Resolve the owning tenant from the session row — no creds over the socket.
    from src.auth.middleware import tenant_from_id

    async with _sm()() as db:
        row = await db.get(ChatSession, session_id)
    if row is None:
        await websocket.close(code=4004, reason="chat session not found")
        return
    tenant = await tenant_from_id(row.tenant_id)
    if tenant is None:
        await websocket.close(code=1011, reason="tenant unavailable")
        return

    try:
        # Pass customer_id from the row already fetched above so the factory
        # doesn't re-query the same ChatSession row on another pool checkout.
        agent = await _factory(tenant, _scoped_session(tenant, session_id),
                               customer_id=row.customer_id)
    except Exception:
        # A failure here (e.g. a DB connection pool exhausted under a burst of
        # concurrent new sessions) must not leave the client waiting forever
        # with no response — close with a clear code instead of hanging.
        log.exception("chatbot factory failed", extra={"session_id": session_id})
        await websocket.close(code=1011, reason="chatbot unavailable — please retry")
        return

    await _hydrate_agent_history(agent, session_id, tenant.id)

    try:
        # Handle reconnect: session may already be in human/awaiting_human mode
        # (reuse the row fetched at connect — no second lookup for the same row).
        if row.mode in ("awaiting_human", "human"):
            if await _run_human_mode(websocket, session_id, tenant):
                return  # session fully ended
            # CRM declined — fall through to bot loop below

        idle_timeout = getattr(tenant.settings.chat_support, "chat_idle_timeout_seconds", 300) or None

        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                # Customer silent for idle_timeout seconds — close gracefully and
                # fire session_closed so the CRM can auto-close the ticket.
                farewell = (
                    "It looks like you've stepped away. I'll close this chat now — "
                    "feel free to start a new conversation anytime!"
                )
                await websocket.send_text(json.dumps({
                    "type": "message", "session_id": session_id,
                    "text": farewell, "sources": [], "suggestions": [], "action": "end",
                }))
                try:
                    summary = await agent.summarize_session()
                except Exception:  # noqa: BLE001 — closing must never fail the customer
                    log.exception("summarize on idle timeout failed", extra={"session_id": session_id})
                    summary = ""
                await _end_session(session_id, summary)
                await _send_close_webhook(tenant, session_id, "ai", summary)
                await websocket.send_text(json.dumps({
                    "type": "ended", "summary": summary, "reason": "idle_timeout",
                }))
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "invalid json"}))
                continue

            mtype = msg.get("type", "message")
            if mtype == "end":
                try:
                    summary = await agent.summarize_session()
                except Exception:  # noqa: BLE001 — closing must never fail the customer
                    log.exception("summarize on end failed", extra={"session_id": session_id})
                    summary = ""
                await _end_session(session_id, summary)
                await _send_close_webhook(tenant, session_id, "ai", summary)
                await websocket.send_text(json.dumps({"type": "ended", "summary": summary, "reason": "customer_ended"}))
                break

            # A single turn's failure must not black-hole the conversation: send an
            # error frame and keep the socket open (a real disconnect re-raises).
            try:
                if mtype == "audio":
                    raw_data = msg.get("data")
                    audio_media_url = (msg.get("media_url") or "").strip()
                    mime = (msg.get("mime") or "").strip()
                    if not raw_data and not audio_media_url:
                        await websocket.send_text(json.dumps(
                            {"type": "error", "message": "audio needs 'data' (base64) or 'media_url'"}))
                        continue
                    if raw_data and (not mime or not mime.startswith("audio/")):
                        await websocket.send_text(json.dumps(
                            {"type": "error", "message": "audio needs 'data' (base64) + 'mime' (audio/*)"}))
                        continue
                    if _media_store is None:
                        await websocket.send_text(json.dumps(
                            {"type": "error", "message": "voice messages not available — media storage not configured"}))
                        continue
                    audio_bytes: bytes | None = None
                    if raw_data:
                        try:
                            audio_bytes = base64.b64decode(raw_data)
                        except Exception:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "message": "invalid base64 in audio data"}))
                            continue

                    await websocket.send_text(json.dumps({"type": "typing"}))

                    # Keepalive spans the whole pre-turn media window (fetch,
                    # upload, transcription) plus the turn itself — a slow
                    # voice note can stall here just as long as a slow CRM
                    # tool call, so coverage can't start only at the turn.
                    _ka_stop = asyncio.Event()
                    _ka = asyncio.ensure_future(_typing_keepalive(websocket, _ka_stop))
                    try:
                        if audio_bytes is None:
                            try:
                                audio_bytes, fetched_mime = await _fetch_media_url(audio_media_url)
                            except Exception:
                                log.exception("media_url fetch failed", extra={"session_id": session_id})
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": "Could not fetch media_url — check it's reachable and points at an audio file.",
                                }))
                                continue
                            mime = mime or fetched_mime
                            if not mime.startswith("audio/"):
                                await websocket.send_text(json.dumps(
                                    {"type": "error", "message": "audio media_url must serve an audio/* content type"}))
                                continue

                        object_key = _media_key(tenant.id, session_id, mime)
                        # Upload to S3 and transcribe in parallel
                        transcript = ""
                        try:
                            upload_coro = _media_store.upload(audio_bytes, object_key, mime.split(";")[0])
                            _transcriber = getattr(agent, "_llm", None) or getattr(agent, "llm", None)
                            if _transcriber and hasattr(_transcriber, "transcribe_audio"):
                                transcript, _ = await asyncio.gather(
                                    _transcriber.transcribe_audio(audio_bytes, mime.split(";")[0]),
                                    upload_coro,
                                )
                            else:
                                await upload_coro
                        except Exception:
                            log.exception("audio upload/transcription failed", extra={"session_id": session_id})
                            await websocket.send_text(json.dumps(
                                {"type": "error", "message": "Could not save voice message — please try again."}))
                            continue

                        # If transcription succeeded, get AI response; else inform customer
                        if transcript:
                            result = await _run_turn(agent.handle_message(transcript))
                            msg_id = await _persist_turn(
                                session_id, transcript, result,
                                user_type="audio", media_mime=mime, media_url=object_key,
                            )
                            if msg_id is not None:
                                await websocket.send_text(json.dumps({
                                    "type": "audio_ack",
                                    "media_url": f"/api/v1/chat/media/{msg_id}",
                                }))
                            await _send_reply(websocket, session_id, result, tenant.id)
                            # Stop the keepalive now, before any human-handoff
                            # escalation: `_run_human_mode` below is an
                            # unbounded loop for the rest of the human
                            # conversation, and the branch-level `finally`
                            # only fires once that loop exits — leaving the
                            # keepalive running would keep emitting `typing`
                            # frames for the entire human conversation. Safe
                            # to call again in the `finally` below (idempotent).
                            await _stop_keepalive(_ka, _ka_stop)
                            if result.escalation:
                                if await _handle_escalation(websocket, session_id, tenant, row, result):
                                    if await _run_human_mode(websocket, session_id, tenant):
                                        break
                        else:
                            # Persist audio without agent reply
                            async with _sm()() as db:
                                r = await db.get(ChatSession, session_id)
                                if r:
                                    audio_msg = ChatMessage(
                                        session_id=session_id, role="customer", type="audio",
                                        content="[audio]", media_mime=mime, media_url=object_key,
                                    )
                                    db.add(audio_msg)
                                    r.message_count = (r.message_count or 0) + 1
                                    await db.flush()
                                    msg_id = audio_msg.id
                                    await db.commit()
                                else:
                                    msg_id = None
                            if msg_id is not None:
                                await websocket.send_text(json.dumps({
                                    "type": "audio_ack",
                                    "media_url": f"/api/v1/chat/media/{msg_id}",
                                }))
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Could not transcribe voice message — please type your message instead.",
                            }))
                    finally:
                        await _stop_keepalive(_ka, _ka_stop)
                    continue

                if mtype in ("image", "video"):
                    data = msg.get("data")
                    media_url = (msg.get("media_url") or "").strip()
                    mime = msg.get("mime") or ""
                    if not data and not media_url:
                        await websocket.send_text(json.dumps(
                            {"type": "error", "message": "image/video needs 'data' (base64) or 'media_url'"}))
                        continue
                    caption = (msg.get("text") or "").strip()
                    await websocket.send_text(json.dumps({"type": "typing"}))

                    # Keepalive spans the whole pre-turn media window (fetch,
                    # upload) plus the turn itself — a slow media_url fetch or
                    # large upload can stall here just as long as a slow CRM
                    # tool call, so coverage can't start only at the turn.
                    _ka_stop = asyncio.Event()
                    _ka = asyncio.ensure_future(_typing_keepalive(websocket, _ka_stop))
                    try:
                        fetched_bytes: Optional[bytes] = None
                        if media_url:
                            try:
                                fetched_bytes, fetched_mime = await _fetch_media_url(media_url)
                                mime = mime or fetched_mime
                            except Exception:
                                log.exception("media_url fetch failed", extra={"session_id": session_id})
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": "Could not fetch media_url — check it's reachable and points at an image/video.",
                                }))
                                continue
                        if not mime:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "message": "image/video needs 'mime'"}))
                            continue

                        # Upload to S3 if storage is configured
                        object_key: Optional[str] = None
                        if _media_store is not None:
                            try:
                                raw_bytes = fetched_bytes if fetched_bytes is not None else base64.b64decode(data)
                                object_key = _media_key(tenant.id, session_id, mime)
                                await _media_store.upload(raw_bytes, object_key, mime.split(";")[0])
                            except Exception:
                                log.exception("media upload failed", extra={"session_id": session_id})
                                object_key = None

                        result = await _run_turn(
                            agent.handle_image(fetched_bytes if fetched_bytes is not None else data, mime, caption))
                        await _persist_turn(session_id, caption or f"[{mtype}]", result,
                                            user_type=mtype, media_mime=mime, media_url=object_key)
                        await _send_reply(websocket, session_id, result, tenant.id)
                        # Stop the keepalive now, before any human-handoff
                        # escalation: `_run_human_mode` below is an unbounded
                        # loop for the rest of the human conversation, and the
                        # branch-level `finally` only fires once that loop
                        # exits — leaving the keepalive running would keep
                        # emitting `typing` frames for the entire human
                        # conversation. Safe to call again in the `finally`
                        # below (idempotent).
                        await _stop_keepalive(_ka, _ka_stop)
                        if result.escalation:
                            if await _handle_escalation(websocket, session_id, tenant, row, result):
                                if await _run_human_mode(websocket, session_id, tenant):
                                    break
                    finally:
                        await _stop_keepalive(_ka, _ka_stop)
                    continue

                user_text = (msg.get("text") or msg.get("message") or "").strip()
                if not user_text:
                    await websocket.send_text(json.dumps({"type": "error", "message": "missing 'text'"}))
                    continue

                await websocket.send_text(json.dumps({"type": "typing"}))
                result = await _run_turn_with_keepalive(websocket, agent.handle_message(user_text))
                await _persist_turn(session_id, user_text, result)
                call_url: Optional[str] = None
                if result.call_offer and _handoff_store is not None:
                    summary = await agent.summarize_session()
                    token = uuid.uuid4().hex
                    context = {
                        "chat_session_id": session_id,
                        "customer_name": row.customer_name,
                        "customer_id": row.customer_id,
                        "language": row.language,
                        "chat_summary": summary,
                    }
                    await _handoff_store.redis.set(
                        f"chat_handoff:{token}", json.dumps(context), ex=600)
                    call_url = _voice_call_url(websocket, tenant.slug, token)
                await _send_reply(websocket, session_id, result, tenant.id, call_url=call_url)
                if result.response.action == "resolved":
                    # Agent confirmed the user has no more questions — close immediately
                    # without waiting for the idle timeout.
                    summary = await agent.summarize_session()
                    await _end_session(session_id, summary)
                    await _send_close_webhook(tenant, session_id, "ai", summary)
                    await websocket.send_text(json.dumps({
                        "type": "ended", "summary": summary, "reason": "resolved",
                    }))
                    break
                if result.escalation:
                    if await _handle_escalation(websocket, session_id, tenant, row, result):
                        if await _run_human_mode(websocket, session_id, tenant):
                            break
            except WebSocketDisconnect:
                raise
            except Exception as turn_exc:  # noqa: BLE001 — one bad turn must not drop the chat
                log.exception("chat turn failed", extra={"session_id": session_id})
                reason, message = _classify_turn_error(turn_exc)
                if reason == "llm_billing":
                    # Distinct from ordinary quota noise: this needs a human to
                    # raise the cap, it won't clear on its own.
                    log.error(
                        "gemini monthly spending cap exceeded — raise it at "
                        "https://ai.studio/spend to restore chat",
                        extra={"session_id": session_id},
                    )
                await websocket.send_text(json.dumps({
                    "type": "error", "message": message, "reason": reason,
                }))
    except WebSocketDisconnect:
        log.info("chat ws client disconnected", extra={"session_id": session_id})
    except Exception:  # noqa: BLE001 — never let the websocket task escape
        log.exception("chat websocket crashed")
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# --- Persistence helpers ------------------------------------------------


async def _send_reply(
    websocket: WebSocket, session_id: str, result: ChatTurnResult, tenant_id: str,
    *, call_url: Optional[str] = None,
) -> None:
    """Send the agent's reply frame, plus an escalation/call_offer frame if the
    agent's tools fired one, and emit a signed chat.escalated tenant event."""
    await websocket.send_text(json.dumps({
        "type": "message",
        "session_id": session_id,
        "text": result.response.response_text,
        "sources": result.response.sources_used,
        "suggestions": result.response.suggested_followups,
        "action": result.response.action,
    }))
    if result.escalation:
        await websocket.send_text(json.dumps({
            "type": "escalation",
            "reason": result.escalation.get("reason", ""),
            "context_summary": result.escalation.get("summary", ""),
        }))
    if result.call_offer:
        await websocket.send_text(json.dumps({
            "type": "call_offer",
            "reason": result.call_offer.get("reason", ""),
            "call_url": call_url,  # WebSocket URL the browser connects to directly
        }))


async def _emit_escalation(tenant_id: str, session_id: str, result: ChatTurnResult) -> None:
    """Emit a signed ``chat.escalated`` tenant event when the agent escalated.
    Delivered via the same notifier the call events use (no-op if unwired)."""
    if not result.escalation:
        return
    from src.api.call_store import emit_tenant_event
    from src.integration.tenant_events import build_envelope
    await emit_tenant_event(build_envelope(
        event_type="chat.escalated",
        call_id=session_id,
        tenant_id=tenant_id,
        channel="chat",
        data={
            "reason": result.escalation.get("reason", ""),
            "summary": result.escalation.get("summary", ""),
        },
    ))


async def _persist_turn(
    session_id: str, user_text: str, result: ChatTurnResult,
    *, user_type: str = "text", media_mime: Optional[str] = None,
    media_url: Optional[str] = None,
) -> Optional[int]:
    """Append the customer + agent messages to chat_messages and bump the count.
    Returns the customer ChatMessage.id on success, None on error or missing session."""
    try:
        async with _sm()() as db:
            row = await db.get(ChatSession, session_id)
            if row is None:
                return None
            customer_msg = ChatMessage(
                session_id=session_id, role="customer", type=user_type,
                content=user_text, media_mime=media_mime, media_url=media_url,
            )
            db.add(customer_msg)
            agent_msg = ChatMessage(
                session_id=session_id, role="agent", type="text",
                content=result.response.response_text,
                sources=result.response.sources_used or None,
            )
            db.add(agent_msg)
            row.message_count = (row.message_count or 0) + 2
            # Chat cost tracking (Pass 1). The ProviderCost rate lookup runs on
            # its own independent DB session (not `db`) so that a failure there
            # (e.g. a pre-migration deploy where the new columns don't exist
            # yet, or a transient DB error) can never poison the `db`
            # transaction that the customer/agent message rows above are
            # staged on. On Postgres a failed statement aborts the *whole*
            # enclosing transaction, so running this on `db` would silently
            # drop the message rows too when the cost lookup fails — isolating
            # it in its own session/connection avoids that entirely.
            if result.llm_provider and (result.input_tokens or result.output_tokens):
                try:
                    async with _sm()() as cost_db:
                        turn_cost = await compute_chat_turn_cost(
                            cost_db, provider=result.llm_provider, model=result.llm_model,
                            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                        )
                    agent_msg.input_tokens = result.input_tokens
                    agent_msg.output_tokens = result.output_tokens
                    agent_msg.cost = turn_cost
                    agent_msg.llm_provider = result.llm_provider
                    agent_msg.llm_model = result.llm_model
                    row.cost = (row.cost or 0.0) + turn_cost
                    row.input_tokens = (row.input_tokens or 0) + result.input_tokens
                    row.output_tokens = (row.output_tokens or 0) + result.output_tokens
                except Exception:  # noqa: BLE001 — never let cost bookkeeping break persistence
                    log.exception("chat turn cost computation failed", extra={"session_id": session_id})
            await db.flush()
            msg_id = customer_msg.id
            await db.commit()
            return msg_id
    except Exception:
        log.exception("chat message persistence failed", extra={"session_id": session_id})
        return None


async def _end_session(session_id: str, summary: str = "") -> None:
    from datetime import datetime, timezone
    try:
        async with _sm()() as db:
            row = await db.get(ChatSession, session_id)
            if row is None:
                return
            row.status = "ended"
            row.mode = "closed"
            row.ended_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if summary:
                row.summary = summary
            await db.commit()
    except Exception:  # noqa: BLE001
        log.exception("chat session end failed", extra={"session_id": session_id})
    # Close any live voice call that was started from this chat session.
    voice_ws = _active_voice_ws.pop(session_id, None)
    if voice_ws:
        try:
            await voice_ws.close(code=1000, reason="chat session ended")
        except Exception:  # noqa: BLE001
            pass


async def _handle_escalation(
    websocket: WebSocket,
    session_id: str,
    tenant: TenantContext,
    row: ChatSession,
    result: ChatTurnResult,
) -> bool:
    """Flip session to awaiting_human, notify BO, inform customer of queue status.
    Returns True if the CRM acknowledged and the session entered human-agent mode.
    Returns False if the webhook failed — caller should keep the bot running."""
    from src.api.chat_webhooks import send_bo_webhook
    from src.chatbot.support_hours import is_bo_available

    reason = result.escalation.get("reason", "")
    summary = result.escalation.get("summary", "")
    available, next_slot = is_bo_available(tenant.settings.chat_support)

    async with _sm()() as db:
        r = await db.get(ChatSession, session_id)
        if r:
            r.mode = "awaiting_human"
            await db.commit()

    _bo_queues.setdefault(session_id, asyncio.Queue())
    _customer_queues.setdefault(session_id, asyncio.Queue())

    ok = await send_bo_webhook(tenant, "escalation_requested", {
        "session_id": session_id,
        "reason": reason,
        "summary": summary,
        "customer": {"name": row.customer_name, "id": row.customer_id},
        "claim_url": f"/api/v1/chat/sessions/{session_id}/claim",
        "agent_ws_url": f"/api/v1/chat/sessions/{session_id}/agent-ws",
        "bo_available": available,
    })

    if not ok:
        # CRM couldn't accept the transfer — revert to bot mode and let the
        # bot continue so the customer isn't left with no one to talk to.
        log.warning("escalation webhook failed; reverting session to bot mode",
                    extra={"session_id": session_id})
        async with _sm()() as db:
            r = await db.get(ChatSession, session_id)
            if r:
                r.mode = "bot"
                await db.commit()
        await websocket.send_text(json.dumps({
            "type": "message",
            "text": ("I'm sorry, I wasn't able to connect you to a human agent right now. "
                     "Let me continue helping you — what would you like to know?"),
            "sources": [], "suggestions": [], "action": "continue",
        }))
        return False

    await websocket.send_text(json.dumps({"type": "mode_change", "mode": "awaiting_human"}))

    if not available and next_slot:
        await websocket.send_text(json.dumps({
            "type": "message",
            "text": (
                f"Our support team is currently unavailable. "
                f"They will be available {next_slot} and will assist you then. "
                "Please stay in this chat and we will connect you as soon as they come online."
            ),
            "sources": [], "suggestions": [], "action": "wait",
        }))

    await _emit_escalation(tenant.id, session_id, result)
    return True


_AWAIT_HUMAN_TIMEOUT_S = 300.0  # revert to bot after 5 min with no claim/decline
CHAT_IDLE_TIMEOUT_S = 300.0  # close AI-mode chat if customer silent for 5 min


async def _revert_to_bot(websocket: WebSocket, session_id: str) -> None:
    """Revert session to bot mode and notify the customer. Used on decline and timeout."""
    async with _sm()() as db:
        r = await db.get(ChatSession, session_id)
        if r:
            r.mode = "bot"
            await db.commit()
    await websocket.send_text(json.dumps({"type": "mode_change", "mode": "bot"}))
    await websocket.send_text(json.dumps({
        "type": "message",
        "text": "Looks like no one is available at this time but am happy to help you again.",
        "sources": [], "suggestions": [], "action": "continue",
    }))


async def _run_human_mode(
    websocket: WebSocket,
    session_id: str,
    tenant: TenantContext,
) -> bool:
    """Handle the customer WS while mode is awaiting_human or human.
    Returns True when the session ended normally (agent closed it / customer left).
    Returns False when the CRM declined or timed out — caller should resume bot loop.

    A deadline of _AWAIT_HUMAN_TIMEOUT_S applies while awaiting a claim/decline.
    The countdown stops as soon as an agent claims (mode_change → human)."""
    bq: asyncio.Queue = _bo_queues.setdefault(session_id, asyncio.Queue())
    cq: asyncio.Queue = _customer_queues.setdefault(session_id, asyncio.Queue())
    loop = asyncio.get_event_loop()
    deadline = loop.time() + _AWAIT_HUMAN_TIMEOUT_S
    claimed = False  # True once mode_change → human received; disables deadline
    try:
        while True:
            ws_task = asyncio.ensure_future(websocket.receive_text())
            cq_task = asyncio.ensure_future(cq.get())
            timeout = None if claimed else max(0.0, deadline - loop.time())
            done, pending = await asyncio.wait(
                {ws_task, cq_task}, return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

            if not done:
                # Timeout — CRM never called claim or decline
                log.warning("await_human timed out; reverting to bot",
                            extra={"session_id": session_id})
                await _revert_to_bot(websocket, session_id)
                return False

            if cq_task in done:
                item = cq_task.result()
                frame = json.loads(item)
                if frame.get("type") == "declined":
                    await _revert_to_bot(websocket, session_id)
                    return False
                if frame.get("type") == "mode_change" and frame.get("mode") == "human":
                    claimed = True  # agent joined — stop the countdown
                await websocket.send_text(item)
                if frame.get("type") == "ended":
                    break

            if ws_task in done:
                try:
                    raw = ws_task.result()
                except WebSocketDisconnect:
                    raise  # outer except handles bq.put(None)
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                mtype = msg.get("type", "message")
                if mtype == "end":
                    from src.api.chat_webhooks import send_bo_webhook
                    summary = "Ended by customer"
                    await _end_session(session_id, summary)
                    await send_bo_webhook(tenant, "session_closed", {
                        "session_id": session_id,
                        "mode_at_close": "human",
                        "summary": summary,
                    })
                    await bq.put(None)
                    await websocket.send_text(json.dumps({"type": "ended"}))
                    break
                user_text = (msg.get("text") or "").strip()
                if user_text:
                    async with _sm()() as db:
                        r = await db.get(ChatSession, session_id)
                        if r:
                            db.add(ChatMessage(
                                session_id=session_id, role="customer",
                                type="text", content=user_text,
                            ))
                            r.message_count = (r.message_count or 0) + 1
                            await db.commit()
                    await bq.put(json.dumps({
                        "type": "customer_message",
                        "text": user_text,
                        "session_id": session_id,
                    }))
    except WebSocketDisconnect:
        await bq.put(None)
        raise
    return True


async def _send_close_webhook(
    tenant: TenantContext, session_id: str, mode_at_close: str, summary: str,
) -> None:
    """Send session_closed webhook with full transcript. Best-effort."""
    from src.api.chat_webhooks import send_bo_webhook
    try:
        async with _sm()() as db:
            msgs = (await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id)
            )).scalars().all()
        transcript = [
            {"role": m.role, "text": m.content,
             "ts": m.created_at.isoformat() if m.created_at else None}
            for m in msgs
        ]
        await send_bo_webhook(tenant, "session_closed", {
            "session_id": session_id,
            "mode_at_close": mode_at_close,
            "summary": summary,
            "transcript": transcript,
        })
    except Exception:  # noqa: BLE001
        log.exception("session_closed webhook failed", extra={"session_id": session_id})


async def process_message(
    tenant: TenantContext, session_id: str, text: str,
) -> ChatMessageResponse:
    """Single-turn chat: get agent, handle message, persist, emit events.
    Shared by the HTTP /message endpoint and the external integrations adapter."""
    agent = await _get_agent(tenant, _scoped_session(tenant, session_id))
    await _hydrate_agent_history(agent, session_id, tenant.id)
    result = await agent.handle_message(text)
    await _persist_turn(session_id, text, result)
    await _emit_escalation(tenant.id, session_id, result)
    return _to_message_response(session_id, result)


def _to_message_response(session_id: str, result: ChatTurnResult) -> ChatMessageResponse:
    r = result.response
    return ChatMessageResponse(
        session_id=session_id,
        response_text=r.response_text,
        language=r.language,
        confidence=r.confidence,
        sources_used=r.sources_used,
        action=r.action,
        suggested_followups=r.suggested_followups,
    )
