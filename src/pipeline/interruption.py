"""Interruption (barge-in) handler.

While the agent is speaking (``RESPONDING`` state), incoming audio from the
caller is fed through ``InterruptionWatcher``. Once speech is detected for
at least ``min_speech_ms`` of contiguous frames, the watcher fires its
callback — the pipeline engine then cancels the in-flight TTS, drops any
pending audio in the playback buffer, and transitions back to LISTENING.

Kept as a small focused class so it's easy to test deterministically with
synthetic frame streams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from src.pipeline.vad import VADFrame
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


@dataclass
class InterruptionConfig:
    min_speech_ms: int = 200       # need this much before declaring barge-in
    detection_interval_ms: int = 20  # nominal frame rate


InterruptCallback = Callable[[], Awaitable[None]]


class InterruptionWatcher:
    def __init__(
        self,
        cfg: InterruptionConfig,
        frame_ms: int,
        on_interrupt: Optional[InterruptCallback] = None,
    ) -> None:
        self._cfg = cfg
        self._frame_ms = frame_ms
        self._on_interrupt = on_interrupt
        self._enabled = False
        self._speech_ms = 0
        self._fired = False

    def enable(self) -> None:
        """Start watching — call when entering RESPONDING state."""
        # Not per-frame -- called once per agent turn. Logs the PRIOR state
        # before clearing it: if a previous watcher was left `fired` or
        # mid-count here, that's the stale-flag bug class
        # (docs/SESSION-HANDOFF-barge-in.md's Deepgram `_endpointed` bug)
        # and this is the only place it would be visible.
        debug_event(
            log, "interruption armed",
            prior_speech_ms=self._speech_ms, prior_fired=self._fired,
            min_speech_ms=self._cfg.min_speech_ms,
        )
        self._enabled = True
        self._speech_ms = 0
        self._fired = False

    def disable(self) -> None:
        """Stop watching — call when leaving RESPONDING state."""
        # Not per-frame -- called once per agent turn. Carries the final
        # accumulated speech_ms/fired state so "why didn't it fire during
        # this RESPONDING turn" is answerable from this one line instead of
        # a per-frame trail.
        debug_event(
            log, "interruption disabled",
            speech_ms=self._speech_ms, fired=self._fired,
            min_speech_ms=self._cfg.min_speech_ms,
        )
        self._enabled = False
        self._speech_ms = 0
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    async def feed(self, frame: VADFrame) -> bool:
        """Feed one VAD frame. Returns True if barge-in was detected this call."""
        # Per-frame (~50/s) for the whole duration of every agent RESPONDING
        # turn. Only the one-time flip to `fired` logs -- once per turn at
        # most, using values (speech_ms, frame.energy/probability) already
        # computed for the decision itself, nothing built for the log.
        if not self._enabled or self._fired:
            return False
        if frame.is_speech:
            self._speech_ms += self._frame_ms
            if self._speech_ms >= self._cfg.min_speech_ms:
                self._fired = True
                debug_event(
                    log, "interruption fired",
                    speech_ms=self._speech_ms, min_speech_ms=self._cfg.min_speech_ms,
                    energy=frame.energy, probability=frame.probability,
                )
                if self._on_interrupt is not None:
                    await self._on_interrupt()
                return True
        else:
            # Brief silence inside speech shouldn't fully reset; allow up to
            # one frame of jitter before resetting the counter.
            self._speech_ms = max(0, self._speech_ms - self._frame_ms)
        return False
