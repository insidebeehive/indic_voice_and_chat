"""Voice Activity Detection.

Two implementations behind a common ``VADDetector`` Protocol:

1. ``EnergyVAD`` — a 30-line RMS-threshold detector. No model dependencies,
   useful for tests and as a fallback. Quality is rough but predictable.

2. ``SileroVAD`` — wraps the ``snakers4/silero-vad`` ONNX model via the
   ``silero-vad`` PyPI package. Imports lazily so users who don't need it
   (i.e. anyone running unit tests) don't pay the install cost.

Both consume 16-bit mono PCM and report per-frame ``is_speech``. The
pipeline engine uses the detector to find utterance boundaries (endpointing)
and to drive the interruption handler (barge-in).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, Protocol

from src.pipeline.audio_utils import rms_energy_pcm16
from src.utils.logging import debug_event

log = logging.getLogger(__name__)


@dataclass
class VADFrame:
    is_speech: bool
    energy: float
    probability: float = 0.0  # filled by SileroVAD; EnergyVAD leaves at 0.0


class VADDetector(Protocol):
    sample_rate: int
    frame_ms: int

    def detect(self, pcm16: bytes) -> VADFrame: ...

    def reset(self) -> None: ...


# --- EnergyVAD ----------------------------------------------------------


class EnergyVAD:
    """Trivial RMS-threshold VAD. Noisy environments will need Silero."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        rms_threshold: float = 300.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._threshold = rms_threshold
        # Tracks only the previous frame's verdict, so `detect` (per-frame,
        # ~50/s) can tell a flip from a repeat without any per-frame log —
        # see the transition check below.
        self._last_is_speech: bool | None = None

    @property
    def frame_bytes(self) -> int:
        # 16-bit mono => 2 bytes per sample
        return int(self.sample_rate * self.frame_ms / 1000) * 2

    def detect(self, pcm16: bytes) -> VADFrame:
        energy = rms_energy_pcm16(pcm16)
        is_speech = energy >= self._threshold
        # Per-frame function (~50 calls/s while a call is open). Only the
        # flip fires a log line -- a handful of times per utterance, not
        # per frame -- and the comparison itself is a cheap float/bool
        # check paid whether or not DEBUG is on.
        if is_speech != self._last_is_speech:
            debug_event(
                log, "vad energy_threshold transition",
                is_speech=is_speech, energy=energy, threshold=self._threshold,
            )
            self._last_is_speech = is_speech
        return VADFrame(is_speech=is_speech, energy=energy)

    def reset(self) -> None:
        self._last_is_speech = None


# --- SileroVAD ----------------------------------------------------------


_SILERO_SESSION = None  # shared, stateless onnxruntime session (state is per-instance)


def _silero_session():
    """Load and cache the bundled silero ONNX model via onnxruntime.

    Uses the model file shipped with the ``silero-vad`` package WITHOUT
    importing the package (which pulls in torch). The session is stateless —
    LSTM state is passed in/out per call — so it is safe to share across VADs.
    """
    global _SILERO_SESSION
    if _SILERO_SESSION is not None:
        return _SILERO_SESSION
    import importlib.util
    import os

    import onnxruntime

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "SileroVAD requires the 'silero-vad' package (for its bundled ONNX "
            "model) and 'onnxruntime'. Install with: pip install silero-vad onnxruntime"
        )
    model_path = os.path.join(os.path.dirname(spec.origin), "data", "silero_vad.onnx")
    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    _SILERO_SESSION = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"], sess_options=opts
    )
    return _SILERO_SESSION


class SileroVAD:
    """Silero VAD via onnxruntime (no torch dependency).

    Runs the bundled silero ONNX model directly. It distinguishes speech from
    background noise far better than ``EnergyVAD``, which keeps utterance
    endpointing from running on through room noise / speaker bleed.

    The model requires fixed frame sizes: 512 samples at 16 kHz (``frame_ms=32``)
    or 256 samples at 8 kHz. Feed exactly that many samples per ``detect`` call.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        threshold: float = 0.5,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports only 8000 or 16000 Hz")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._threshold = threshold
        self._context_size = 64 if sample_rate == 16000 else 32
        self._session = None  # lazy
        self._state = None
        self._context = None

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        # Boundary, and a slow one -- loads/caches the ONNX session on first
        # use of this VAD instance. Not per-frame: guarded by the None check
        # above, so this body runs once per SileroVAD instance.
        started = time.monotonic()
        self._session = _silero_session()
        self.reset()
        debug_event(
            log, "vad silero_model load",
            sample_rate=self.sample_rate, frame_ms=self.frame_ms,
            threshold=self._threshold, load_ms=round((time.monotonic() - started) * 1000, 1),
        )

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000) * 2

    def detect(self, pcm16: bytes) -> VADFrame:
        import numpy as np

        self._ensure_model()
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return VADFrame(is_speech=False, energy=0.0, probability=0.0)
        x = samples.reshape(1, -1)
        # silero v5 expects the saved context (last 64 samples) prepended.
        inp = np.concatenate([self._context, x], axis=1).astype(np.float32)
        out, new_state = self._session.run(
            None,
            {
                "input": inp,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        self._state = new_state
        self._context = x[:, -self._context_size:]
        prob = float(out[0][0])
        return VADFrame(
            is_speech=prob >= self._threshold,
            energy=rms_energy_pcm16(pcm16),
            probability=prob,
        )

    def reset(self) -> None:
        import numpy as np

        # Not per-frame -- called once per turn/call by owners of this VAD.
        # Clears the LSTM state carried between `detect` calls; a stale
        # state here is the same bug class as the stale Deepgram
        # `_endpointed` flag (docs/SESSION-HANDOFF-barge-in.md), so log
        # whether there was anything to clear.
        debug_event(
            log, "vad silero_state reset",
            had_state=self._state is not None,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)


# --- Endpoint detection -------------------------------------------------


@dataclass
class EndpointConfig:
    min_speech_ms: int = 250
    min_silence_ms: int = 600


class EndpointDetector:
    """Tracks speech/silence runs to detect end-of-utterance.

    Feed VAD frames in order; ``utterance_complete()`` returns True once
    we've seen at least ``min_speech_ms`` of speech followed by
    ``min_silence_ms`` of contiguous silence.
    """

    def __init__(self, frame_ms: int, cfg: EndpointConfig) -> None:
        self._frame_ms = frame_ms
        self._cfg = cfg
        self._speech_ms = 0
        self._trailing_silence_ms = 0
        self._saw_enough_speech = False
        # Logging-only bookkeeping (does not affect feed()'s return value):
        # _in_speech lets a per-frame call detect a flip without remembering
        # the previous frame's is_speech anywhere else; _fired_logged stops
        # the "fired" event from repeating on every subsequent silent frame
        # if a caller is slow to (or never does) call reset() after a
        # completed utterance -- exactly the stale-flag shape this package
        # is watching for, applied to our own log line so a forgotten
        # reset() degrades to a missing log rather than a flood.
        self._in_speech = False
        self._fired_logged = False

    @property
    def utterance_reported(self) -> bool:
        """Whether this utterance's completion has already been logged.

        Public because ``turn_capture.accumulate_and_detect`` needs the same
        dedup latch for its own event: ``feed`` returns True on EVERY silent
        frame once the threshold is crossed, not just the first, so a caller
        logging on the return value alone floods at frame rate if ``reset()``
        is ever late. Exposed as a property rather than read off the private
        attribute, because that read sits in a 50/s loop -- renaming the field
        would turn a logging detail into an AttributeError raised 50 times a
        second on live audio.
        """
        return self._fired_logged

    def feed(self, frame: VADFrame) -> bool:
        # Called once per audio frame (~50/s per open call) by every
        # transport bridge. No log fires on a steady-state frame -- only on
        # the state flips below, which happen a handful of times per
        # utterance.
        if frame.is_speech:
            if not self._in_speech:
                debug_event(
                    log, "endpoint speech_run started",
                    energy=frame.energy, probability=frame.probability,
                )
                self._in_speech = True
            self._speech_ms += self._frame_ms
            self._trailing_silence_ms = 0
            if not self._saw_enough_speech and self._speech_ms >= self._cfg.min_speech_ms:
                self._saw_enough_speech = True
                debug_event(
                    log, "endpoint min_speech reached",
                    speech_ms=self._speech_ms, min_speech_ms=self._cfg.min_speech_ms,
                )
            return False
        if self._in_speech:
            debug_event(
                log, "endpoint speech_run ended",
                speech_ms=self._speech_ms, saw_enough_speech=self._saw_enough_speech,
            )
            self._in_speech = False
        if not self._saw_enough_speech:
            return False
        self._trailing_silence_ms += self._frame_ms
        complete = self._trailing_silence_ms >= self._cfg.min_silence_ms
        if complete and not self._fired_logged:
            debug_event(
                log, "endpoint utterance_complete fired",
                speech_ms=self._speech_ms,
                trailing_silence_ms=self._trailing_silence_ms,
                min_silence_ms=self._cfg.min_silence_ms,
                min_speech_ms=self._cfg.min_speech_ms,
            )
            self._fired_logged = True
        return complete

    def feed_many(self, frames: Iterable[VADFrame]) -> bool:
        complete = False
        for f in frames:
            if self.feed(f):
                complete = True
                break
        return complete

    def reset(self) -> None:
        # Not per-frame -- called once per turn by every caller of this
        # detector: browser_bridge.py and telephony_twilio.py via
        # accumulate_and_detect (turn_capture.py), and telephony_exotel.py
        # directly -- it does not go through accumulate_and_detect, it
        # inlines the same capture/detect/feed loop and calls
        # `_endpoint.reset()` itself (src/api/telephony_exotel.py). Logged
        # unconditionally
        # (with the state being cleared) so a reset that runs with nothing
        # accumulated -- or, via its absence in the logs, a reset that never
        # runs between turns -- is visible without a code read.
        debug_event(
            log, "endpoint state reset",
            prior_speech_ms=self._speech_ms,
            prior_trailing_silence_ms=self._trailing_silence_ms,
            prior_saw_enough_speech=self._saw_enough_speech,
        )
        self._speech_ms = 0
        self._trailing_silence_ms = 0
        self._saw_enough_speech = False
        self._in_speech = False
        self._fired_logged = False
