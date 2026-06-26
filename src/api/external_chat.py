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

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.models.chat import ChatSession

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
    (full conversation memory, CRM tool results carry forward).
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
            "session_id": session_id, "user_id": req.user_id,
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
