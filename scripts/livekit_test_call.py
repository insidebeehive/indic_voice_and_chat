"""Standalone LiveKit connectivity + audio-pacing spike.

Zero dependency on this repo's tenant/DB/agent machinery — this only proves
two things against a REAL LiveKit server:

  1. The SDK can actually connect, mint a token, join a room, and publish a
     track (basic connectivity).
  2. Whether ``AudioSource.capture_frame()`` provides real-time pacing via FFI
     backpressure (the open question from Phase 1 that ``LiveKitBridge``'s
     sender loop — src/api/livekit_bridge.py — relies on instead of a
     hand-rolled sleep loop). If ``capture_frame`` blocks until there's queue
     room, pushing N seconds of audio as fast as Python can go should take
     close to N seconds wall-clock. If it returns instantly regardless, the
     loop finishes near-instantly — meaning the bridge needs a manual pacer.

Credentials are read from the environment ONLY — never hardcode them here:
    TENANT_DEV_LIVEKIT_URL
    TENANT_DEV_LIVEKIT_API_KEY
    TENANT_DEV_LIVEKIT_API_SECRET

Usage:
    export $(grep '^TENANT_DEV_LIVEKIT' .env | xargs)   # or: source .env (set -a)
    .venv/bin/python scripts/livekit_test_call.py
    .venv/bin/python scripts/livekit_test_call.py --duration 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import struct
import sys
import time
import uuid

log = logging.getLogger("livekit_test_call")

_FRAME_MS = 20  # outbound chunk size, matches LiveKitBridge._FRAME_S
_SAMPLE_RATE = 24000  # matches the bridge's outbound AudioSource rate
_TONE_HZ = 440.0  # A4 — arbitrary, just needs to be valid PCM16


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LiveKit connectivity + audio-pacing spike")
    p.add_argument("--duration", type=float, default=4.0,
                    help="seconds of synthetic audio to push through capture_frame (default: 4.0)")
    p.add_argument("--join-timeout", type=float, default=15.0,
                    help="seconds to wait for room.connect() (default: 15.0)")
    return p


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"error: {name} is not set. Load it from .env first, e.g.:\n"
              f"  export $(grep '^TENANT_DEV_LIVEKIT' .env | xargs)\n"
              f"  .venv/bin/python scripts/livekit_test_call.py",
              file=sys.stderr)
        raise SystemExit(2)
    return value


def _sine_wave_pcm16(duration_s: float, sample_rate: int, freq_hz: float) -> bytes:
    """Generate a mono 16-bit PCM sine wave — doesn't need to be real speech,
    just valid PCM16 frames at a declared sample rate."""
    n_samples = int(duration_s * sample_rate)
    amplitude = 8000  # comfortably under int16 range, avoids clipping
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq_hz * (i / sample_rate)))
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *samples)


async def _run(args: argparse.Namespace) -> int:
    url = _require_env("TENANT_DEV_LIVEKIT_URL")
    api_key = _require_env("TENANT_DEV_LIVEKIT_API_KEY")
    api_secret = _require_env("TENANT_DEV_LIVEKIT_API_SECRET")

    from livekit import api, rtc

    room_name = f"test-{uuid.uuid4().hex[:8]}"
    identity = f"spike-{uuid.uuid4().hex[:8]}"
    print(f"room: {room_name}")

    grants = api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True)
    token = api.AccessToken(api_key, api_secret).with_identity(identity).with_grants(grants).to_jwt()

    room = rtc.Room()
    connected = False
    room_deleted = False
    try:
        print(f"connecting to {url} ...")
        try:
            await asyncio.wait_for(
                room.connect(url, token, options=rtc.RoomOptions(auto_subscribe=False)),
                timeout=args.join_timeout,
            )
        except asyncio.TimeoutError:
            print(f"FAIL: room.connect() did not complete within {args.join_timeout}s "
                  "(network blocked, or server unreachable)")
            return 1
        connected = True
        print("connected OK.")

        audio_source = rtc.AudioSource(sample_rate=_SAMPLE_RATE, num_channels=1)
        local_track = rtc.LocalAudioTrack.create_audio_track("spike-audio", audio_source)
        await room.local_participant.publish_track(
            local_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        print("published a synthetic audio track OK.")

        pcm = _sine_wave_pcm16(args.duration, _SAMPLE_RATE, _TONE_HZ)
        chunk_bytes = int(_SAMPLE_RATE * (_FRAME_MS / 1000.0)) * 2  # 16-bit mono
        frames = [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]
        audio_duration_s = len(pcm) / 2 / _SAMPLE_RATE
        print(f"pushing {len(frames)} frames ({audio_duration_s:.2f}s of audio) through "
              f"capture_frame() back-to-back, no manual sleep ...")

        start = time.monotonic()
        for piece in frames:
            if not piece:
                continue
            frame = rtc.AudioFrame(
                data=piece, sample_rate=_SAMPLE_RATE, num_channels=1,
                samples_per_channel=len(piece) // 2)
            await audio_source.capture_frame(frame)
        wall_clock_s = time.monotonic() - start

        ratio = wall_clock_s / audio_duration_s if audio_duration_s else 0.0
        print()
        print("=== RESULT ===")
        print(f"audio duration:  {audio_duration_s:.3f}s")
        print(f"wall-clock time: {wall_clock_s:.3f}s")
        print(f"ratio (wall/audio): {ratio:.3f}")
        if ratio >= 0.7:
            print("PASS: capture_frame() provides real-time pacing via FFI backpressure — "
                  "wall-clock tracked audio duration. LiveKitBridge's sender loop can rely on "
                  "capture_frame() alone; no manual pacer needed.")
        else:
            print("FAIL: capture_frame() did NOT block for pacing — it returned almost "
                  "instantly regardless of audio duration. LiveKitBridge._sender_loop would "
                  "need a manual real-time pacer added back in (like TelephonyLiveBridge's).")

        try:
            async with api.LiveKitAPI(url, api_key, api_secret) as lkapi:
                await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
            room_deleted = True
            print("cleanup: room deleted via RoomService.DeleteRoom OK.")
        except Exception as e:  # noqa: BLE001 — cleanup best-effort, don't mask the result above
            print(f"cleanup: delete_room failed (non-fatal): {e}")

        return 0
    finally:
        if connected:
            await room.disconnect()
            print("cleanup: room.disconnect() done." + ("" if room_deleted else
                  " (room delete not confirmed — it may still show as active server-side)"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
