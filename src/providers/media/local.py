"""In-memory media storage — temporary fallback when S3 is not configured.

Data is lost on process restart. Suitable for dev/stage testing only.
"""

from __future__ import annotations

from src.interfaces.media_storage import IMediaStorage

_SERVE_PREFIX = "/api/v1/chat/local-media"


class LocalMediaStorage(IMediaStorage):
    """Stores media blobs in a process-local dict and serves them via the app."""

    def __init__(self, serve_prefix: str = _SERVE_PREFIX) -> None:
        self._store: dict[str, tuple[bytes, str]] = {}  # key → (data, content_type)
        self._serve_prefix = serve_prefix.rstrip("/")

    async def upload(self, data: bytes, key: str, content_type: str) -> None:
        self._store[key] = (data, content_type)

    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        clean_key = key.lstrip("/")
        return f"{self._serve_prefix}/{clean_key}"

    def get(self, key: str) -> tuple[bytes, str] | None:
        return self._store.get(key)
