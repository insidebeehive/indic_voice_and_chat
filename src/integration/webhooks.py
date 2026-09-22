"""Webhook delivery manager.

Subscribes to the EventBus on ``"*"`` and fans events out to registered
HTTP endpoints. Each registration filters by event-type prefix (``"call.*"``
matches ``call.initiated`` and ``call.completed`` but not ``lead.qualified``).

Delivery is best-effort with bounded retry (default 3 attempts, exponential
backoff). Failures are logged but never raised — webhook delivery must not
block the conversation pipeline.

Constructor accepts an injectable ``http_post`` callable so tests can
substitute a fake without touching httpx.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

import httpx

from src.integration.event_bus import Event, EventBus
from src.utils.logging import debug_event
from src.utils.redact import redact_url

log = logging.getLogger(__name__)


# ``http_post(url, json, timeout) -> status_code``. ``-1`` means no response.
HTTPPoster = Callable[[str, dict, float], Awaitable[int]]


async def _default_http_post(url: str, json: dict, timeout: float) -> int:
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=json)
            return resp.status_code
        except httpx.HTTPError as e:
            log.warning("webhook delivery failed", extra={"url": url, "error": str(e)})
            return -1


@dataclass
class WebhookRegistration:
    id: str
    url: str
    event_filters: list[str] = field(default_factory=lambda: ["*"])
    secret: Optional[str] = None       # reserved for HMAC signing (Phase 6+)
    active: bool = True

    def matches(self, event_type: str) -> bool:
        if not self.active:
            return False
        for pat in self.event_filters:
            if pat == "*" or pat == event_type:
                return True
            if pat.endswith(".*") and event_type.startswith(pat[:-2] + "."):
                return True
        return False


@dataclass
class WebhookConfig:
    timeout_s: float = 5.0
    max_attempts: int = 3
    backoff_base_s: float = 0.2  # 0.2, 0.4, 0.8 ...


class WebhookManager:
    def __init__(
        self,
        bus: Optional[EventBus] = None,
        http_post: Optional[HTTPPoster] = None,
        config: Optional[WebhookConfig] = None,
    ) -> None:
        self._bus = bus
        self._post = http_post or _default_http_post
        self._cfg = config or WebhookConfig()
        self._registry: dict[str, WebhookRegistration] = {}
        self._delivered: list[tuple[str, str, int]] = []  # (webhook_id, event_type, status)
        if bus is not None:
            bus.subscribe("*", self._on_event)

    # --- registration --------------------------------------------------

    def register(self, url: str, event_filters: Optional[Iterable[str]] = None, secret: Optional[str] = None) -> WebhookRegistration:
        reg = WebhookRegistration(
            id=f"wh_{uuid.uuid4().hex[:12]}",
            url=url,
            event_filters=list(event_filters) if event_filters else ["*"],
            secret=secret,
        )
        self._registry[reg.id] = reg
        return reg

    def unregister(self, webhook_id: str) -> bool:
        return self._registry.pop(webhook_id, None) is not None

    def list(self) -> list[WebhookRegistration]:
        return list(self._registry.values())

    # --- delivery ------------------------------------------------------

    @property
    def delivered(self) -> list[tuple[str, str, int]]:
        return list(self._delivered)

    async def _on_event(self, event: Event) -> None:
        targets = [r for r in self._registry.values() if r.matches(event.type)]
        if not targets:
            debug_event(
                log, "integration webhooks dispatch_skipped",
                event_type=event.type, reason="no_matching_registrations",
                registered_count=len(self._registry),
            )
            return
        debug_event(
            log, "integration webhooks dispatch_request",
            event_type=event.type, target_count=len(targets),
        )
        await asyncio.gather(*(self._deliver(r, event) for r in targets))

    async def _deliver(self, reg: WebhookRegistration, event: Event) -> None:
        body = {
            "event_type": event.type,
            "occurred_at": event.occurred_at.isoformat(),
            "payload": event.payload,
            "source": event.source,
            "webhook_id": reg.id,
        }
        for attempt in range(self._cfg.max_attempts):
            status = await self._post(reg.url, body, self._cfg.timeout_s)
            # Success previously carried no log at any level -- only the
            # final-failure warning below existed, so a webhook that
            # succeeds on attempt 2 or 3 looked, from outside, identical to
            # one that succeeded first try.
            debug_event(
                log, "integration webhooks deliver_response",
                webhook_id=reg.id, url=redact_url(reg.url), event_type=event.type,
                attempt=attempt + 1, max_attempts=self._cfg.max_attempts, status=status,
            )
            if 200 <= status < 300:
                self._delivered.append((reg.id, event.type, status))
                return
            # Backoff between attempts (skip after last try).
            if attempt < self._cfg.max_attempts - 1:
                await asyncio.sleep(self._cfg.backoff_base_s * (2**attempt))
        # Final failure — record and move on.
        self._delivered.append((reg.id, event.type, -1))
        log.warning(
            "webhook delivery exhausted retries",
            extra={"webhook_id": reg.id, "url": reg.url, "event_type": event.type},
        )
