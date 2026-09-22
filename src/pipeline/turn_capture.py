"""Shared per-turn audio capture core for media bridges.

Each transport bridge (Twilio, Exotel, browser) feeds inbound PCM16 frames
through the same VAD + endpoint-detection loop. This helper is that loop, so
the logic lives in one place rather than being copied per transport.
"""

from __future__ import annotations

import logging

from src.pipeline.vad import EndpointDetector, VADDetector, VADFrame
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


def accumulate_and_detect(
    pcm16: bytes,
    vad: VADDetector,
    endpoint: EndpointDetector,
    capture_buffer: bytearray,
    *,
    frame: VADFrame | None = None,
) -> bool:
    """Append a PCM16 frame to the capture buffer and run endpointing.

    Mutates ``capture_buffer`` in place (appends ``pcm16`` bytes) — this is
    the only side effect.

    If the caller already computed a ``VADFrame`` for this chunk (e.g. to
    drive an idle-silence timer), pass it via ``frame=`` so that
    ``vad.detect`` is not called a second time.  This matters for stateful
    VADs such as ``SileroVAD``, whose internal LSTM state advances on every
    ``detect`` call; calling it twice per chunk would corrupt the model state.
    When ``frame`` is ``None`` (the default) the helper calls ``vad.detect``
    itself, which is the normal path for callers that do not pre-compute it.

    Returns True once the endpoint detector reports end-of-utterance, i.e.
    the caller should dispatch the buffered audio as a completed turn.
    """
    # Called once per inbound audio frame (~50/s per open call) by
    # browser_bridge.py and telephony_twilio.py. telephony_exotel.py does
    # NOT call this helper -- it inlines the same
    # capture/detect/feed/reset steps itself (src/api/telephony_exotel.py,
    # `_capture_buffer.extend` / `_vad.detect` / `_endpoint.feed` /
    # `_endpoint.reset`), so the "turn_capture accumulate_and_detect fired"
    # event below never fires for an Exotel call. `endpoint.feed` already
    # logs its own state transitions (vad.py), and those DO fire for Exotel
    # since it calls `endpoint.feed` directly -- so an operator comparing a
    # Twilio trace against an Exotel trace sees "endpoint utterance_complete
    # fired" on both but the turn-dispatch event only on Twilio. This
    # function logs only the outcome it adds on top of that: the
    # accumulated capture buffer at the moment a turn is considered
    # complete. Nothing is logged on the per-frame path.
    #
    # `endpoint.feed`'s `complete` return value is NOT one-shot: once
    # trailing silence crosses the threshold it stays True on every further
    # silent frame until `reset()` runs, which normally happens in the
    # caller right after seeing True -- but nothing here guarantees that.
    # Snapshotting the endpoint's own dedup latch before calling feed() and
    # only logging on the frame that flips it is what stops this event from
    # flooding at frame rate if a caller is ever slow to (or fails to)
    # reset, mirroring the guard `endpoint.feed` already applies to its own
    # "utterance_complete fired" event.
    already_reported = endpoint.utterance_reported
    capture_buffer.extend(pcm16)
    if frame is None:
        frame = vad.detect(pcm16)
    complete = endpoint.feed(frame)
    if complete and not already_reported:
        debug_event(
            log, "turn_capture accumulate_and_detect fired",
            captured_bytes=len(capture_buffer), last_frame_bytes=len(pcm16),
        )
    return complete
