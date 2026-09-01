"""Object-storage abstraction for chat media (audio, image, video)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class IMediaStorage(ABC):
    @abstractmethod
    async def upload(self, data: bytes, key: str, content_type: str) -> None:
        """Upload bytes to the store at the given object key."""

    @abstractmethod
    async def signed_url(self, key: str, ttl_seconds: int) -> str:
        """Return a time-limited URL for reading the object at key."""

    @abstractmethod
    async def download(self, key: str) -> tuple[bytes, str]:
        """Return (data, content_type) for the object at key.

        Raises FileNotFoundError if key does not exist.
        """
