"""In-memory media storage — temporary fallback when S3 is not configured.

Data is lost on process restart. Suitable for dev/stage testing only.
"""

from __future__ import annotations

import logging

from src.interfaces.media_storage import IMediaStorage
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

_SERVE_PREFIX = "/api/v1/chat/local-media"


class LocalMediaStorage(IMediaStorage):
    """Stores media blobs in a process-local dict and serves them via the app."""

    def __init__(self, serve_prefix: str = _SERVE_PREFIX) -> None:
        self._store: dict[str, tuple[bytes, str]] = {}  # key → (data, content_type)
        self._serve_prefix = serve_prefix.rstrip("/")

    async def upload(self, data: bytes, key: str, content_type: str) -> None:
        # Process-local only -- this is the dev/stage fallback (see module
        # docstring) -- so a restart silently drops everything ever uploaded
        # here. Data itself is never logged (binary), just its shape.
        debug_event(log, "local media upload", key=key, content_type=content_type,
                    bytes=len(data))
        self._store[key] = (data, content_type)

    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        clean_key = key.lstrip("/")
        return f"{self._serve_prefix}/{clean_key}"

    async def download(self, key: str) -> tuple[bytes, str]:
        entry = self._store.get(key)
        if entry is None:
            # A miss here means either the key was never uploaded in THIS
            # process, or (more likely in practice) a different worker
            # process handled the upload -- this store has no cross-process
            # visibility at all, unlike S3. That distinction is exactly what
            # a bare `FileNotFoundError` loses.
            debug_event(log, "local media download not found", key=key,
                        known_keys=len(self._store))
            raise FileNotFoundError(key)
        return entry

    def get(self, key: str) -> tuple[bytes, str] | None:
        return self._store.get(key)
