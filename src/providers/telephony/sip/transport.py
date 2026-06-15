"""SIP transport interface — the seam between the agent bridge and the SIP/RTP
stack. Keeping this abstract means the bridge + Call Lead wiring are unit-testable
with a fake call, and the actual pyVoIP/RTP implementation is isolated.

Audio is **PCM16 mono @ 8 kHz** in both directions; the implementation owns the
codec (μ-law/PCMU on the wire) and RTP packetization/pacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Protocol, runtime_checkable


class SipError(RuntimeError):
    """SIP registration / INVITE / media failure."""


@dataclass
class SipCallParams:
    """Everything needed to place one outbound call over a SIP trunk."""
    to_number: str          # destination (E.164 or trunk-accepted format)
    from_number: str        # caller-ID / DID owned on the trunk
    sip_user: str           # trunk auth username
    sip_password: str       # trunk auth password
    sip_server: str         # trunk host, e.g. "sip.didlogic.com"
    sip_port: int = 5060
    codec: str = "PCMU"     # μ-law 8 kHz — the telephony default
    answer_timeout_s: float = 45.0


@runtime_checkable
class ISipCall(Protocol):
    """An in-progress outbound SIP call. PCM16 @ 8 kHz both ways."""

    @property
    def call_id(self) -> str: ...

    async def wait_answered(self) -> bool:
        """Block until the callee answers (True) or the call fails/times out (False)."""
        ...

    def audio_in(self) -> AsyncIterator[bytes]:
        """Yield inbound caller audio as PCM16 @ 8 kHz frames until the call ends."""
        ...

    async def send_audio(self, pcm16_8k: bytes) -> None:
        """Queue outbound PCM16 @ 8 kHz audio to the callee (impl paces RTP)."""
        ...

    async def flush(self) -> None:
        """Drop any queued/playing outbound audio (barge-in). Optional / best-effort."""
        ...

    async def hangup(self) -> None:
        """End the call and release the SIP/RTP resources. Idempotent."""
        ...


# Places the INVITE and returns a live ISipCall (or raises SipError).
SipCallFactory = Callable[[SipCallParams], Awaitable[ISipCall]]
