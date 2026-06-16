"""Audio format conversion utilities.

Supports the formats the voice pipeline actually uses:
- 16-bit signed little-endian PCM (the canonical internal format)
- 8-bit μ-law (G.711, what Twilio Media Streams sends/expects, 8 kHz)
- WAV with PCM payload (for STT providers that want a file)

All functions are pure — no I/O, no global state — so they're trivial to test
and safe to call from multiple coroutines.
"""

from __future__ import annotations

import audioop
import io
import wave
from typing import Optional


# --- μ-law <-> 16-bit PCM (Twilio uses μ-law @ 8 kHz) --------------------


def mulaw_to_pcm16(mulaw: bytes) -> bytes:
    """Decode 8-bit μ-law samples to 16-bit signed little-endian PCM."""
    return audioop.ulaw2lin(mulaw, 2)


def pcm16_to_mulaw(pcm16: bytes) -> bytes:
    """Encode 16-bit signed little-endian PCM to 8-bit μ-law."""
    return audioop.lin2ulaw(pcm16, 2)


# --- Sample rate conversion ---------------------------------------------


def resample_pcm16(
    pcm16: bytes,
    src_rate: int,
    dst_rate: int,
    state: Optional[tuple] = None,
) -> tuple[bytes, tuple]:
    """Resample 16-bit mono PCM. Returns ``(resampled_bytes, new_state)``.

    Pass the returned ``state`` back on the next call when streaming so block
    boundaries don't introduce clicks.
    """
    if src_rate == dst_rate:
        return pcm16, state or (None,)
    out, new_state = audioop.ratecv(pcm16, 2, 1, src_rate, dst_rate, state)
    return out, new_state


# --- WAV envelope -------------------------------------------------------


def pcm16_to_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw PCM in a WAV container (mono, 16-bit). For STT uploads."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw PCM + sample rate from a WAV blob.

    Mono is required; stereo is downmixed by averaging channels.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchannels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM supported, got {sample_width * 8}-bit")
        frames = wf.readframes(wf.getnframes())
    if nchannels == 2:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
    elif nchannels != 1:
        raise ValueError(f"Unsupported channel count: {nchannels}")
    return frames, sample_rate


def wav_split_stereo(wav_bytes: bytes) -> tuple[bytes, bytes, int]:
    """Split a 16-bit WAV into its two mono channels: ``(left, right, sample_rate)``.

    Used for dual-channel call recordings where each party is on its own track.
    A mono WAV returns the same PCM for both channels (so callers can treat it
    uniformly). Raises on non-16-bit or >2-channel audio.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchannels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM supported, got {sample_width * 8}-bit")
        frames = wf.readframes(wf.getnframes())
    if nchannels == 1:
        return frames, frames, sample_rate
    if nchannels != 2:
        raise ValueError(f"Unsupported channel count: {nchannels}")
    left = audioop.tomono(frames, 2, 1.0, 0.0)
    right = audioop.tomono(frames, 2, 0.0, 1.0)
    return left, right, sample_rate


# --- Energy / loudness --------------------------------------------------


def rms_energy_pcm16(pcm16: bytes) -> float:
    """Root-mean-square energy of a PCM16 frame. 0.0 for empty/silent."""
    if not pcm16:
        return 0.0
    return float(audioop.rms(pcm16, 2))


# --- Padding ------------------------------------------------------------


def pcm16_silence_ms(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate ``duration_ms`` of silent 16-bit mono PCM."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * n_samples
