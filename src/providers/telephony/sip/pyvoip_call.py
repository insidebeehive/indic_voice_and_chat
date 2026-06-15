"""pyVoIP-backed ISipCall — the concrete SIP/RTP transport for raw trunks.

pyVoIP is a pure-Python SIP/RTP stack (``pip install pyVoIP``). It is threaded
and blocking, so this wraps it for asyncio: blocking SIP/RTP calls run in worker
threads and inbound audio is pumped into an asyncio queue. The wire codec is PCMU
(μ-law) @ 8 kHz; pyVoIP decodes RTP to PCM16 @ 8 kHz for ``read_audio`` and
accepts PCM16 @ 8 kHz for ``write_audio``.

!! UNVERIFIED: this follows pyVoIP's documented API but has NOT been exercised
   against live DiDLogic yet (no creds at build time). The pyVoIP version on the
   host may need small API adjustments (constructor args / read_audio signature).
   Everything *above* this transport (bridge, Call Lead, billing) is tested with
   a fake ISipCall, so only this glue needs live validation.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from typing import AsyncIterator, Optional

from src.providers.telephony.sip.transport import ISipCall, SipCallParams, SipError

log = logging.getLogger(__name__)

_FRAME_SAMPLES = 160      # 20 ms @ 8 kHz
_FRAME_BYTES = _FRAME_SAMPLES * 2   # PCM16


def _local_ip() -> str:
    """Best-effort local IP for the SIP/RTP contact address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


async def place_pyvoip_call(params: SipCallParams) -> ISipCall:
    """Register to the trunk and send the outbound INVITE. Returns a live call."""
    call = _PyVoipCall(params)
    try:
        await asyncio.to_thread(call._connect)   # blocking: register + INVITE
    except SipError:
        raise
    except Exception as e:  # noqa: BLE001
        raise SipError(f"pyVoIP call setup failed: {e}") from e
    return call


class _PyVoipCall:
    def __init__(self, params: SipCallParams) -> None:
        self._p = params
        self._phone = None
        self._call = None
        self._loop = asyncio.get_event_loop()
        self._in_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=200)
        self._ended = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._call_id = f"sip_{id(self):x}"

    @property
    def call_id(self) -> str:
        return self._call_id

    # --- blocking setup (runs in a thread) ------------------------------
    def _connect(self) -> None:
        try:
            from pyVoIP.VoIP import VoIPPhone  # lazy: pyVoIP is optional
        except ImportError as e:
            raise SipError("pyVoIP not installed — `pip install pyVoIP`") from e
        self._phone = VoIPPhone(
            self._p.sip_server, self._p.sip_port, self._p.sip_user, self._p.sip_password,
            myIP=_local_ip())
        self._phone.start()
        self._call = self._phone.call(self._p.to_number)   # outbound INVITE
        self._call_id = getattr(self._call, "call_id", self._call_id)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        """Pump decoded inbound PCM16 frames onto the asyncio queue (thread)."""
        from pyVoIP.VoIP import CallState
        try:
            while not self._ended.is_set():
                state = getattr(self._call, "state", None)
                if state is not None and state == CallState.ENDED:
                    break
                try:
                    audio = self._call.read_audio(_FRAME_BYTES, blocking=True)
                except Exception:  # noqa: BLE001 — call ended / not ready
                    break
                if audio:
                    self._loop.call_soon_threadsafe(self._in_q.put_nowait, audio)
        finally:
            self._loop.call_soon_threadsafe(self._in_q.put_nowait, None)   # sentinel: EOF

    # --- ISipCall (async) ----------------------------------------------
    async def wait_answered(self) -> bool:
        from pyVoIP.VoIP import CallState

        deadline = self._loop.time() + self._p.answer_timeout_s
        while self._loop.time() < deadline:
            state = getattr(self._call, "state", None)
            if state == CallState.ANSWERED:
                return True
            if state in (getattr(CallState, "ENDED", None), getattr(CallState, "DECLINED", None)):
                return False
            await asyncio.sleep(0.1)
        return False

    async def audio_in(self) -> AsyncIterator[bytes]:
        while True:
            frame = await self._in_q.get()
            if frame is None:      # EOF sentinel
                break
            yield frame

    async def send_audio(self, pcm16_8k: bytes) -> None:
        if not pcm16_8k or self._call is None:
            return
        await asyncio.to_thread(self._call.write_audio, pcm16_8k)

    async def flush(self) -> None:
        # pyVoIP has no documented outbound flush; best-effort no-op.
        return

    async def hangup(self) -> None:
        self._ended.set()
        try:
            if self._call is not None:
                await asyncio.to_thread(self._call.hangup)
        except Exception:  # noqa: BLE001 — already ended
            pass
        try:
            if self._phone is not None:
                await asyncio.to_thread(self._phone.stop)
        except Exception:  # noqa: BLE001
            pass
