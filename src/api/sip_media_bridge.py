"""SIP/RTP media bridge — runs the live agent over a raw SIP trunk call.

Same dialogue core as the WebSocket telephony bridge (``_BaseLiveBridge``); the
only difference is the transport: instead of a Media-Streams WebSocket, audio
flows over an in-process ``ISipCall`` (PCM16 @ 8 kHz), resampled to/from the
model's 16 kHz. Used for outbound calls on SIP-trunk providers (e.g. DiDLogic).
"""

from __future__ import annotations

import logging

from src.api.live_bridge_base import _BaseLiveBridge
from src.pipeline.audio_utils import resample_pcm16
from src.providers.telephony.sip.transport import ISipCall

log = logging.getLogger(__name__)

_TEL_RATE = 8000      # telephony / RTP is 8 kHz mono
_MODEL_RATE = 16000   # the realtime model speaks 16 kHz PCM16


class SipMediaBridge(_BaseLiveBridge):
    """One bridge per outbound SIP call. Drive with ``run()``.

    ``sip_call`` is a live ISipCall (the INVITE has already been sent by the
    factory). The bridge waits for answer, then pumps caller RTP -> model and
    model audio -> caller RTP until the call ends.
    """

    def __init__(self, *, sip_call: ISipCall, agent, config, connect_session,
                 llm=None, tenant_timezone: str = "Asia/Kolkata") -> None:
        super().__init__(agent=agent, config=config, connect_session=connect_session,
                         llm=llm, tenant_timezone=tenant_timezone)
        self._sip = sip_call
        self._up_state = None       # resampler state: caller 8k -> 16k
        self._down_state = None     # resampler state: model rate -> 8k

    async def run(self) -> None:
        await self._drive()

    # --- transport hooks ---
    async def _on_start(self) -> None:
        # Don't start streaming until the callee actually answers.
        if not await self._sip.wait_answered():
            log.info("sip call not answered", extra={"call_id": self._sip.call_id})
            self._stopped = True   # -> inbound loop exits immediately -> teardown

    async def _inbound_loop(self) -> None:
        async for pcm8k in self._sip.audio_in():
            if self._stopped:
                break
            if not pcm8k:
                continue
            pcm16k, self._up_state = resample_pcm16(pcm8k, _TEL_RATE, _MODEL_RATE, self._up_state)
            if self._session is not None:
                await self._session.send_audio(pcm16k)

    async def _send_audio_out(self, pcm16: bytes, rate: int) -> None:
        if not pcm16:
            return
        if not self._speaking:
            self._speaking = True
            await self._emit_status("speaking")
        pcm8k, self._down_state = resample_pcm16(pcm16, rate, _TEL_RATE, self._down_state)
        await self._sip.send_audio(pcm8k)

    async def _send_interrupt(self) -> None:
        await self._sip.flush()   # barge-in: drop queued agent audio

    async def _on_teardown(self) -> None:
        await self._sip.hangup()
