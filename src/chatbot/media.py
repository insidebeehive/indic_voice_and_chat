"""Multimodal input prep for Gemini (PRD §5.4).

Turns a customer's image/video (+ optional text) into ``list[ContentPart]`` for
``LLMMessage.content``. Images pass through as ``inline_data`` (Gemini handles
them natively). Video is best-effort: we extract a few key frames as images if a
decoder is available, otherwise we degrade to a text note (image-first per the
plan — video never blocks a reply).
"""

from __future__ import annotations

import base64
import logging

from src.interfaces.llm import ContentPart

log = logging.getLogger(__name__)

_MAX_VIDEO_FRAMES = 5


def _as_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    return base64.b64decode(data)


def prepare_multimodal_content(text: str, media: object, mime: str) -> list[ContentPart]:
    """Build multimodal parts. ``media`` is base64 str or raw bytes."""
    parts: list[ContentPart] = []
    if text:
        parts.append(ContentPart(type="text", text=text))

    if mime.startswith("image/"):
        parts.append(ContentPart(type="image", inline_data={"mime_type": mime, "data": _as_bytes(media)}))
    elif mime.startswith("video/"):
        frames = extract_key_frames(_as_bytes(media), max_frames=_MAX_VIDEO_FRAMES)
        if frames:
            parts.append(ContentPart(type="text", text="[Customer sent a video. Key frames:]"))
            for f in frames:
                parts.append(ContentPart(type="image", inline_data={"mime_type": "image/jpeg", "data": f}))
        else:
            parts.append(ContentPart(
                type="text",
                text="[Customer sent a video that couldn't be decoded — ask them to describe the issue.]"))
    else:
        # Unknown media type — keep any text; ignore the blob.
        log.warning("unsupported chat media mime", extra={"mime": mime})
    return parts


def extract_key_frames(video: bytes, max_frames: int = _MAX_VIDEO_FRAMES) -> list[bytes]:
    """Best-effort: evenly-spaced JPEG frames via PyAV if installed, else []."""
    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
        log.info("PyAV not installed; video frame extraction disabled")
        return []
    import io

    try:
        frames: list[bytes] = []
        with av.open(io.BytesIO(video)) as container:
            stream = container.streams.video[0]
            total = stream.frames or 0
            step = max(1, total // max_frames) if total else 1
            for i, frame in enumerate(container.decode(stream)):
                if i % step:
                    continue
                buf = io.BytesIO()
                frame.to_image().save(buf, format="JPEG")
                frames.append(buf.getvalue())
                if len(frames) >= max_frames:
                    break
        return frames
    except Exception:  # noqa: BLE001 — never let a bad video break the turn
        log.exception("video frame extraction failed")
        return []
