"""Local-filesystem media storage — temporary fallback when S3 is not configured."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from src.interfaces.media_storage import IMediaStorage

_DEFAULT_BASE = "/tmp/chat_media"


class LocalMediaStorage(IMediaStorage):
    """Stores media files on the local filesystem and serves them via the app."""

    def __init__(self, base_dir: str = _DEFAULT_BASE, serve_prefix: str = "/api/v1/chat/local-media") -> None:
        self._base = Path(base_dir)
        self._serve_prefix = serve_prefix.rstrip("/")

    def _safe_path(self, key: str) -> Path:
        # Resolve to prevent directory traversal
        target = (self._base / key).resolve()
        if not str(target).startswith(str(self._base.resolve())):
            raise ValueError(f"invalid key: {key!r}")
        return target

    async def upload(self, data: bytes, key: str, content_type: str) -> None:
        path = self._safe_path(key)
        await asyncio.to_thread(self._write, path, data)

    def _write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        # Return an app-relative URL; the app's /local-media endpoint serves it
        clean_key = key.lstrip("/")
        return f"{self._serve_prefix}/{clean_key}"

    async def read(self, key: str) -> bytes:
        path = self._safe_path(key)
        return await asyncio.to_thread(path.read_bytes)

    def base_dir(self) -> Path:
        return self._base
