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
import json
import logging
import uuid
from typing import Awaitable, Callable, Optional

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

from src.agents.chatbot import ChatBotAgent, ChatTurnResult
from src.interfaces.media_storage import IMediaStorage
from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.interfaces.llm import LLMMessage
from src.models.chat import ChatMessage, ChatSession

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# --- DI -----------------------------------------------------------------


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


def _greeting(company: str, customer_name: Optional[str], language: str) -> str:
    name = (customer_name or "").strip()
    who = name if name else "there"
    return f"Hello {who}, how can I help?"


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
    language = req.language or getattr(tenant.settings, "default_language", None) or "hi"
    session.add(ChatSession(
        id=session_id, tenant_id=tenant.id,
        customer_id=req.user_id, customer_name=req.customer_name,
        language=language, status="active", extra_data=req.metadata or {},
    ))
    await session.commit()
    from src.api.chat_webhooks import send_bo_webhook
    await send_bo_webhook(tenant, "session_started", {
        "session_id": session_id,
        "customer": {"name": req.customer_name, "id": req.user_id},
    })
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
    result = await agent.handle_message(req.message)
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
    result = await agent.handle_image(data, mime, text)
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
    await run_browser_voice(websocket, tenant)


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

    agent = await _factory(tenant, _scoped_session(tenant, session_id))
    try:
        # Handle reconnect: session may already be in human/awaiting_human mode.
        async with _sm()() as db:
            cur_row = await db.get(ChatSession, session_id)
        if cur_row and cur_row.mode in ("awaiting_human", "human"):
            await _run_human_mode(websocket, session_id, tenant)
        else:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"type": "error", "message": "invalid json"}))
                    continue

                mtype = msg.get("type", "message")
                if mtype == "end":
                    summary = await agent.summarize_session()
                    await _end_session(session_id, summary)
                    await _send_close_webhook(tenant, session_id, "ai", summary)
                    await websocket.send_text(json.dumps({"type": "ended", "summary": summary}))
                    break

                # A single turn's failure must not black-hole the conversation: send an
                # error frame and keep the socket open (a real disconnect re-raises).
                try:
                    if mtype == "audio":
                        raw_data = msg.get("data")
                        mime = (msg.get("mime") or "").strip()
                        if not raw_data or not mime or not mime.startswith("audio/"):
                            await websocket.send_text(json.dumps(
                                {"type": "error", "message": "audio needs 'data' (base64) + 'mime' (audio/*)"}))
                            continue
                        if _media_store is None:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "message": "voice messages not available — media storage not configured"}))
                            continue
                        try:
                            audio_bytes = base64.b64decode(raw_data)
                        except Exception:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "message": "invalid base64 in audio data"}))
                            continue

                        await websocket.send_text(json.dumps({"type": "typing"}))

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
                            result = await agent.handle_message(transcript)
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
                            if result.escalation:
                                await _handle_escalation(websocket, session_id, tenant, row, result)
                                await _run_human_mode(websocket, session_id, tenant)
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
                        continue

                    if mtype in ("image", "video"):
                        data = msg.get("data")
                        mime = msg.get("mime") or ""
                        if not data or not mime:
                            await websocket.send_text(json.dumps(
                                {"type": "error", "message": "image/video needs 'data' + 'mime'"}))
                            continue
                        caption = (msg.get("text") or "").strip()
                        await websocket.send_text(json.dumps({"type": "typing"}))

                        # Upload to S3 if storage is configured
                        object_key: Optional[str] = None
                        if _media_store is not None:
                            try:
                                raw_bytes = base64.b64decode(data)
                                object_key = _media_key(tenant.id, session_id, mime)
                                await _media_store.upload(raw_bytes, object_key, mime.split(";")[0])
                            except Exception:
                                log.exception("media upload failed", extra={"session_id": session_id})
                                object_key = None

                        result = await agent.handle_image(data, mime, caption)
                        await _persist_turn(session_id, caption or f"[{mtype}]", result,
                                            user_type=mtype, media_mime=mime, media_url=object_key)
                        await _send_reply(websocket, session_id, result, tenant.id)
                        if result.escalation:
                            await _handle_escalation(websocket, session_id, tenant, row, result)
                            await _run_human_mode(websocket, session_id, tenant)
                            break
                        continue

                    user_text = (msg.get("text") or msg.get("message") or "").strip()
                    if not user_text:
                        await websocket.send_text(json.dumps({"type": "error", "message": "missing 'text'"}))
                        continue

                    await websocket.send_text(json.dumps({"type": "typing"}))
                    result = await agent.handle_message(user_text)
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
                    if result.escalation:
                        await _handle_escalation(websocket, session_id, tenant, row, result)
                        await _run_human_mode(websocket, session_id, tenant)
                        break
                except WebSocketDisconnect:
                    raise
                except Exception:  # noqa: BLE001 — one bad turn must not drop the chat
                    log.exception("chat turn failed", extra={"session_id": session_id})
                    await websocket.send_text(json.dumps(
                        {"type": "error", "message": "Sorry, something went wrong — please try again."}))
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
            db.add(ChatMessage(
                session_id=session_id, role="agent", type="text",
                content=result.response.response_text,
                sources=result.response.sources_used or None,
            ))
            row.message_count = (row.message_count or 0) + 2
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


async def _handle_escalation(
    websocket: WebSocket,
    session_id: str,
    tenant: TenantContext,
    row: ChatSession,
    result: ChatTurnResult,
) -> None:
    """Flip session to awaiting_human, notify BO, inform customer of queue status.
    Checks support hours: outside hours the customer is queued and told when BO opens."""
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

    await send_bo_webhook(tenant, "escalation_requested", {
        "session_id": session_id,
        "reason": reason,
        "summary": summary,
        "customer": {"name": row.customer_name, "id": row.customer_id},
        "claim_url": f"/api/v1/chat/sessions/{session_id}/claim",
        "agent_ws_url": f"/api/v1/chat/sessions/{session_id}/agent-ws",
        "bo_available": available,
    })

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


async def _run_human_mode(
    websocket: WebSocket,
    session_id: str,
    tenant: TenantContext,
) -> None:
    """Handle the customer WS while mode is awaiting_human or human.
    Races between messages from the customer and replies from the BO agent WS."""
    bq: asyncio.Queue = _bo_queues.setdefault(session_id, asyncio.Queue())
    cq: asyncio.Queue = _customer_queues.setdefault(session_id, asyncio.Queue())
    try:
        while True:
            ws_task = asyncio.ensure_future(websocket.receive_text())
            cq_task = asyncio.ensure_future(cq.get())
            done, pending = await asyncio.wait(
                {ws_task, cq_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

            if cq_task in done:
                item = cq_task.result()
                await websocket.send_text(item)
                if json.loads(item).get("type") == "ended":
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
