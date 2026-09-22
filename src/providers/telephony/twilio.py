"""Twilio telephony adapter.

Twilio's REST SDK is synchronous, so blocking calls are wrapped in
``asyncio.to_thread``. Bidirectional audio streaming requires the Twilio
Media Streams websocket, which lives in ``src/api/telephony_hooks.py`` and is
Phase 3 work — ``stream_audio_in/out`` raise ``NotImplementedError`` here so
the interface stays honest.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator

from twilio.rest import Client as TwilioClient

from src.interfaces.telephony import (
    CallConfig,
    CallSession,
    ITelephonyProvider,
)
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


_STATUS_MAP = {
    "queued": "ringing",
    "ringing": "ringing",
    "in-progress": "answered",
    "completed": "answered",
    "busy": "busy",
    "no-answer": "no_answer",
    "failed": "failed",
    "canceled": "failed",
}


class TwilioAdapter(ITelephonyProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        account_sid = config.get("account_sid") or os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = config.get("auth_token") or os.environ.get("TWILIO_AUTH_TOKEN")
        if not (account_sid and auth_token):
            raise ValueError(
                "TwilioAdapter requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN"
            )
        self._client: TwilioClient = config.get("client") or TwilioClient(account_sid, auth_token)

    async def initiate_call(self, config: CallConfig) -> CallSession:
        def _create() -> Any:
            return self._client.calls.create(
                to=config.to_number,
                from_=config.from_number,
                url=config.webhook_url,
                timeout=config.timeout_seconds,
            )

        # No credential in this request body -- account_sid/auth_token were
        # already consumed building `self._client` and never appear per-call --
        # so the full request/response is safe to log without redaction.
        debug_event(log, "twilio initiate_call request", to=config.to_number,
                    from_=config.from_number, webhook_url=config.webhook_url,
                    timeout_seconds=config.timeout_seconds)
        try:
            call = await asyncio.to_thread(_create)
        except Exception as exc:  # noqa: BLE001 - re-raised unchanged; this adapter has no logging at all today
            debug_event(log, "twilio initiate_call failed", to=config.to_number,
                        from_=config.from_number, error=str(exc))
            raise
        debug_event(log, "twilio initiate_call response", session_id=call.sid,
                    status=call.status)
        return CallSession(
            session_id=call.sid,
            status=_STATUS_MAP.get(call.status, call.status or "ringing"),
            to_number=config.to_number,
            from_number=config.from_number,
        )

    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "Twilio Media Streams audio is wired in Phase 3 via the websocket "
            "route in src/api/telephony_hooks.py"
        )
        if False:  # pragma: no cover  (satisfies AsyncIterator return type)
            yield b""

    async def stream_audio_out(
        self,
        session_id: str,
        audio_stream: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError(
            "Twilio Media Streams audio is wired in Phase 3 via the websocket "
            "route in src/api/telephony_hooks.py"
        )

    async def hangup(self, session_id: str) -> None:
        # Call-state transition; none of hangup/transfer/redirect_to_stream
        # logged anything before this pass, so a hangup that silently no-ops
        # (wrong session_id, call already ended) looked identical to one that
        # worked.
        debug_event(log, "twilio hangup", session_id=session_id)

        def _hangup() -> None:
            self._client.calls(session_id).update(status="completed")

        await asyncio.to_thread(_hangup)

    async def transfer(self, session_id: str, to_number: str) -> None:
        twiml = f'<Response><Dial>{to_number}</Dial></Response>'
        debug_event(log, "twilio transfer", session_id=session_id, to_number=to_number)

        def _transfer() -> None:
            self._client.calls(session_id).update(twiml=twiml)

        await asyncio.to_thread(_transfer)

    async def redirect_to_stream(self, call_sid: str, stream_wss_url: str) -> None:
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Connect><Stream url="{stream_wss_url}"/></Connect>'
            "</Response>"
        )
        debug_event(log, "twilio redirect_to_stream", call_sid=call_sid,
                    stream_wss_url=stream_wss_url)

        def _redirect() -> None:
            self._client.calls(call_sid).update(twiml=twiml)

        await asyncio.to_thread(_redirect)
