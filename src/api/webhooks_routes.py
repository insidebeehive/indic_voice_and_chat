"""Webhook registration endpoints (PRD §7.8).

Thin wrapper around ``WebhookManager`` — the manager owns the in-memory
registry, the routes just expose register/list/delete.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.integration.webhooks import WebhookManager
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# --- DI -----------------------------------------------------------------


_manager: Optional[WebhookManager] = None


def set_webhook_manager(manager: Optional[WebhookManager]) -> None:
    global _manager
    _manager = manager


def _require_manager() -> WebhookManager:
    if _manager is None:
        debug_event(log, "webhooks manager unavailable", reason="not_initialized")
        raise HTTPException(status_code=503, detail="webhook manager not initialized")
    return _manager


# --- Schemas ------------------------------------------------------------


class RegisterWebhookRequest(BaseModel):
    url: str = Field(min_length=1)
    event_filters: list[str] = Field(default_factory=lambda: ["*"])
    secret: Optional[str] = None


class WebhookResponse(BaseModel):
    id: str
    url: str
    event_filters: list[str]
    active: bool


class WebhooksListResponse(BaseModel):
    webhooks: list[WebhookResponse]
    total: int


# --- Routes -------------------------------------------------------------


@router.post("", response_model=WebhookResponse)
async def register_webhook(req: RegisterWebhookRequest) -> WebhookResponse:
    m = _require_manager()
    reg = m.register(url=req.url, event_filters=req.event_filters, secret=req.secret)
    # req.secret is a webhook-signing credential -- presence only, never the value.
    debug_event(log, "webhooks register response", webhook_id=reg.id, url=reg.url,
                event_filters=reg.event_filters, active=reg.active, has_secret=bool(req.secret))
    return WebhookResponse(id=reg.id, url=reg.url, event_filters=reg.event_filters, active=reg.active)


@router.get("", response_model=WebhooksListResponse)
async def list_webhooks() -> WebhooksListResponse:
    m = _require_manager()
    items = [
        WebhookResponse(id=r.id, url=r.url, event_filters=r.event_filters, active=r.active)
        for r in m.list()
    ]
    debug_event(log, "webhooks list response", webhook_count=len(items),
                webhooks=[i.model_dump() for i in items])
    return WebhooksListResponse(webhooks=items, total=len(items))


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str) -> dict:
    m = _require_manager()
    deleted = m.unregister(webhook_id)
    debug_event(log, "webhooks delete response", webhook_id=webhook_id, deleted=deleted)
    if not deleted:
        raise HTTPException(status_code=404, detail="webhook not found")
    return {"deleted": webhook_id}
