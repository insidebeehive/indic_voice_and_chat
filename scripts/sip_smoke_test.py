"""Standalone SIP transport smoke test for a raw SIP trunk (DiDLogic).

Validates the pyVoIP/RTP layer (src/providers/telephony/sip/pyvoip_call.py) in
ISOLATION — no Gemini agent — so we can confirm SIP registration, INVITE, answer,
and two-way RTP before wiring the full Call Lead flow.

Needs real creds in the environment:
  DIDLOGIC_SIP_USER, DIDLOGIC_SIP_PASSWORD, DIDLOGIC_SIP_SERVER, DIDLOGIC_DID

Usage:
  .venv/bin/python scripts/sip_smoke_test.py <destination_number> [seconds]

It places the call, waits for answer, then for N seconds streams silence outbound
and counts inbound RTP frames (proving two-way media), and hangs up.
"""

from __future__ import annotations

import asyncio
import os
import sys

from src.providers.telephony.sip.pyvoip_call import place_pyvoip_call
from src.providers.telephony.sip.transport import SipCallParams

_FRAME = b"\x00" * 320   # 20 ms of PCM16 silence @ 8 kHz


async def _send_silence(call, seconds: float) -> None:
    for _ in range(int(seconds / 0.02)):
        await call.send_audio(_FRAME)
        await asyncio.sleep(0.02)


async def _count_inbound(call, stop: asyncio.Event) -> int:
    n = 0
    async for frame in call.audio_in():
        if frame:
            n += 1
        if stop.is_set():
            break
    return n


async def main(dest: str, seconds: float) -> None:
    for k in ("DIDLOGIC_SIP_USER", "DIDLOGIC_SIP_PASSWORD", "DIDLOGIC_SIP_SERVER"):
        if not os.environ.get(k):
            print(f"missing env var: {k}")
            sys.exit(2)
    params = SipCallParams(
        to_number=dest,
        from_number=os.environ.get("DIDLOGIC_DID", ""),
        sip_user=os.environ["DIDLOGIC_SIP_USER"],
        sip_password=os.environ["DIDLOGIC_SIP_PASSWORD"],
        sip_server=os.environ["DIDLOGIC_SIP_SERVER"],
    )
    print(f"placing call to {dest} via {params.sip_server} as {params.sip_user} ...")
    call = await place_pyvoip_call(params)
    print("INVITE sent; waiting for answer (up to 45s)...")
    answered = await call.wait_answered()
    print("answered:", answered)
    inbound = 0
    if answered:
        stop = asyncio.Event()
        reader = asyncio.create_task(_count_inbound(call, stop))
        try:
            await _send_silence(call, seconds)
        finally:
            stop.set()
            try:
                inbound = await asyncio.wait_for(reader, timeout=2.0)
            except asyncio.TimeoutError:
                reader.cancel()
        print(f"streamed {seconds}s silence; inbound RTP frames received: {inbound}")
        print("two-way audio:", "OK" if inbound > 0 else "NONE (one-way / NAT issue?)")
    await call.hangup()
    print("hung up.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/sip_smoke_test.py <destination_number> [seconds]")
        sys.exit(1)
    dest_num = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    asyncio.run(main(dest_num, secs))
