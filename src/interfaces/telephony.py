"""Telephony provider interface (PRD §4.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class CallConfig:
    to_number: str
    from_number: str
    webhook_url: str
    audio_format: str = "pcm"
    sample_rate: int = 8000
    timeout_seconds: int = 60


@dataclass
class CallSession:
    session_id: str
    status: str  # "ringing" | "answered" | "busy" | "no_answer" | "failed"
    to_number: str
    from_number: str


class ITelephonyProvider(ABC):
    @abstractmethod
    async def initiate_call(self, config: CallConfig) -> CallSession:
        """Initiate an outbound call."""

    @abstractmethod
    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        """Receive audio from the call (caller's speech)."""

    @abstractmethod
    async def stream_audio_out(
        self,
        session_id: str,
        audio_stream: AsyncIterator[bytes],
    ) -> None:
        """Send audio to the call (agent's speech)."""

    @abstractmethod
    async def hangup(self, session_id: str) -> None:
        """End the call."""

    @abstractmethod
    async def transfer(self, session_id: str, to_number: str) -> None:
        """Transfer the call to another number (warm transfer)."""

    @abstractmethod
    async def redirect_to_stream(self, call_sid: str, stream_wss_url: str) -> None:
        """Redirect a live in-progress call to stream audio to ``stream_wss_url``.

        Used by the CRM patch-in handoff: the CRM holds a customer on a live
        call, then calls our handoff endpoint which calls this method to
        transition the media stream to our AI bridge WebSocket.
        """
