"""Exotel telephony adapter.

Exotel is an India-native CPaaS with direct interconnects to Indian
carriers — useful when Twilio's US trial origins get filtered by TRAI /
mobile-operator anti-spam systems on Indian destinations.

REST surface implemented here:
- ``initiate_call``  POST /v1/Accounts/{sid}/Calls/connect  (outbound dial)
- ``hangup``         DELETE the active call resource
- ``transfer``       update CallType to redirect

Media Streams (bidirectional audio) goes through Exotel's "Voicebot
Streaming" WebSocket — handled separately by ``ExotelMediaBridge``
(see ``src/bootstrap.py``). ``stream_audio_in/out`` here remain stubbed
because the canonical bridge lives at the FastAPI route layer (same
pattern as the Twilio adapter — the REST surface and the WS bridge are
two different concerns).

Endpoint reference: https://developer.exotel.com/api/
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator, Optional
from urllib.parse import urljoin

import httpx

from src.interfaces.telephony import (
    CallConfig,
    CallSession,
    ITelephonyProvider,
)


# Region-aware base URLs. Exotel's Indian region (``api.exotel.com``) is the
# default; Singapore region uses ``api.sg.exotel.com``.
EXOTEL_BASE_URL = "https://api.exotel.com"

# Map Exotel call statuses to our internal vocabulary so the rest of the
# framework can stay provider-agnostic.
_STATUS_MAP = {
    "queued":      "ringing",
    "in-progress": "answered",
    "completed":   "answered",
    "busy":        "busy",
    "no-answer":   "no_answer",
    "failed":      "failed",
    "canceled":    "failed",
}


class ExotelAdapter(ITelephonyProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        # Per-tenant credentials get injected via the tenant-aware factory.
        api_key = config.get("api_key") or os.environ.get("EXOTEL_API_KEY")
        api_token = (
            config.get("api_token")
            or config.get("auth_token")
            or os.environ.get("EXOTEL_API_TOKEN")
        )
        # Exotel's account identifier is variously called ``account_sid``,
        # ``sid``, ``subdomain``, or ``account_sid_exotel`` across their docs.
        # Be lenient about the config key but DO NOT fall through to api_key —
        # treating the API key as an account SID hits the wrong URL path.
        self._account_sid = (
            config.get("account_sid")
            or config.get("account_sid_exotel")
            or config.get("sid")
            or config.get("subdomain")
            or os.environ.get("EXOTEL_ACCOUNT_SID")
        )
        if not (api_key and api_token and self._account_sid):
            raise ValueError(
                "ExotelAdapter requires api_key + api_token + account_sid "
                "(or EXOTEL_API_KEY / EXOTEL_API_TOKEN / EXOTEL_ACCOUNT_SID env vars)"
            )
        self._auth = (api_key, api_token)
        self._base_url = config.get("base_url", EXOTEL_BASE_URL).rstrip("/")
        self._timeout = config.get("timeout", 30.0)

    def _account_url(self, *path: str) -> str:
        parts = "/".join(p.strip("/") for p in path)
        return f"{self._base_url}/v1/Accounts/{self._account_sid}/{parts}"

    async def initiate_call(self, config: CallConfig) -> CallSession:
        """Place an outbound call.

        Exotel's outbound API connects two legs: ``From`` (the agent /
        ExoPhone) and ``To`` (the destination). When the destination
        answers, Exotel POSTs to ``Url`` with ``CallSid`` and call params,
        and the body's ExotelML controls what happens next.
        """
        data = {
            "From": config.from_number,
            "To": config.to_number,
            "CallerId": config.from_number,
            "Url": config.webhook_url,
            "CallType": "trans",  # full duplex
            "TimeLimit": str(config.timeout_seconds) if config.timeout_seconds else "30",
        }
        async with httpx.AsyncClient(auth=self._auth, timeout=self._timeout) as client:
            resp = await client.post(self._account_url("Calls", "connect"), data=data)
            resp.raise_for_status()
            payload = resp.json()
        # Response shape: {"Call": {"Sid": "...", "Status": "queued", ...}}
        call = payload.get("Call") or payload.get("call") or {}
        sid = call.get("Sid") or call.get("CallSid") or ""
        status = (call.get("Status") or "queued").lower()
        return CallSession(
            session_id=sid,
            status=_STATUS_MAP.get(status, status),
            to_number=config.to_number,
            from_number=config.from_number,
        )

    async def hangup(self, session_id: str) -> None:
        """Terminate an in-progress call by call SID."""
        async with httpx.AsyncClient(auth=self._auth, timeout=self._timeout) as client:
            resp = await client.delete(self._account_url("Calls", session_id))
            # Exotel returns 200 even for already-ended calls; treat as best-effort.
            if resp.status_code >= 500:
                resp.raise_for_status()

    async def transfer(self, session_id: str, to_number: str) -> None:
        """Redirect the call to ``to_number`` via an updated CallType."""
        data = {"To": to_number, "CallType": "trans"}
        async with httpx.AsyncClient(auth=self._auth, timeout=self._timeout) as client:
            resp = await client.post(self._account_url("Calls", session_id), data=data)
            resp.raise_for_status()

    async def redirect_to_stream(self, call_sid: str, stream_wss_url: str) -> None:
        """Redirect a live call's media to our Voicebot Streaming WebSocket.

        Exotel's live-call update endpoint accepts an ``Xml`` body parameter
        containing ExotelML. We pass the same ``<Connect><Stream>`` XML that
        the answer webhook returns, which causes Exotel to transition the
        active call onto the streaming WS.
        """
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Connect><Stream url="{stream_wss_url}"/></Connect>'
            "</Response>"
        )
        data = {"Xml": xml}
        async with httpx.AsyncClient(auth=self._auth, timeout=self._timeout) as client:
            resp = await client.post(self._account_url("Calls", call_sid), data=data)
            resp.raise_for_status()

    # --- Media Streams stubs --------------------------------------------
    # The canonical bridge lives in ``src/bootstrap.py``; see
    # ``ExotelMediaBridge``. These remain unimplemented at the adapter
    # layer for the same reason as ``TwilioAdapter`` — REST surface and
    # WS bridge are two separate concerns.

    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "Exotel media streaming is wired through the Voicebot Streaming "
            "WebSocket route; see ExotelMediaBridge"
        )
        if False:  # pragma: no cover (AsyncIterator return-type appease)
            yield b""

    async def stream_audio_out(
        self,
        session_id: str,
        audio_stream: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError(
            "Exotel media streaming is wired through the Voicebot Streaming "
            "WebSocket route; see ExotelMediaBridge"
        )
