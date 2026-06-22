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

import json
import logging
import uuid
from typing import Awaitable, Callable, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.chatbot import ChatBotAgent, ChatTurnResult
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


def set_chat_handoff_store(store: object | None) -> None:
    """Inject the Redis-backed store the chat→voice handoff uses to pass the
    chat summary into the voice agent."""
    global _handoff_store
    _handoff_store = store


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


def _new_session_id() -> str:
    return f"cs_{uuid.uuid4().hex[:16]}"


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
    if language.startswith("hi"):
        who = f"{name} जी" if name else "आप"
        return f"नमस्ते {who}! मैं {company} की सहायक हूँ। मैं आपकी कैसे मदद कर सकती हूँ?"
    who = name or "there"
    return f"Hi {who}! I'm the {company} assistant. How can I help you today?"


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
    return CreateSessionResponse(
        session_id=session_id,
        greeting=_greeting(tenant.name, req.customer_name, language),
        ws_url=_ws_url(request, session_id),
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    status: Optional[str] = None,
    customer_id: Optional[str] = None,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> SessionListResponse:
    q = select(ChatSession).where(ChatSession.tenant_id == tenant.id)
    if status:
        q = q.where(ChatSession.status == status)
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
        session_id=row.id, status=row.status,
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
                await websocket.send_text(json.dumps({"type": "ended", "summary": summary}))
                break

            # A single turn's failure must not black-hole the conversation: send an
            # error frame and keep the socket open (a real disconnect re-raises).
            try:
                if mtype in ("image", "video"):
                    data = msg.get("data")
                    mime = msg.get("mime") or ""
                    if not data or not mime:
                        await websocket.send_text(json.dumps(
                            {"type": "error", "message": "image/video needs 'data' + 'mime'"}))
                        continue
                    caption = (msg.get("text") or "").strip()
                    await websocket.send_text(json.dumps({"type": "typing"}))
                    result = await agent.handle_image(data, mime, caption)
                    await _persist_turn(session_id, caption or f"[{mtype}]", result,
                                        user_type=mtype, media_mime=mime)
                    await _send_reply(websocket, session_id, result, tenant.id)
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
    await _emit_escalation(tenant_id, session_id, result)


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
) -> None:
    """Append the customer + agent messages to chat_messages and bump the count.
    Best-effort: a missing session row (e.g. /message with an ad-hoc id) is a
    no-op so the conversational reply is never blocked on persistence."""
    try:
        async with _sm()() as db:
            row = await db.get(ChatSession, session_id)
            if row is None:
                return
            db.add(ChatMessage(session_id=session_id, role="customer", type=user_type,
                               content=user_text, media_mime=media_mime))
            db.add(ChatMessage(
                session_id=session_id, role="agent", type="text",
                content=result.response.response_text,
                sources=result.response.sources_used or None,
            ))
            row.message_count = (row.message_count or 0) + 2
            await db.commit()
    except Exception:  # noqa: BLE001 — persistence must not break the conversation
        log.exception("chat message persistence failed", extra={"session_id": session_id})


async def _end_session(session_id: str, summary: str = "") -> None:
    from datetime import datetime, timezone
    try:
        async with _sm()() as db:
            row = await db.get(ChatSession, session_id)
            if row is None:
                return
            row.status = "ended"
            row.ended_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if summary:
                row.summary = summary
            await db.commit()
    except Exception:  # noqa: BLE001
        log.exception("chat session end failed", extra={"session_id": session_id})


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
