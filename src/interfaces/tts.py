"""Text-to-Speech provider interface (PRD §4.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class TTSConfig:
    language: str = "hi-IN"
    voice_id: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0
    output_format: str = "pcm"  # "pcm" | "wav" | "mp3"
    sample_rate: int = 16000
    extra_pronunciations: Optional[dict[str, str]] = None


@dataclass
class TTSResult:
    audio: bytes
    duration_ms: float
    sample_rate: int


class ITTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        """Synthesize complete text to audio."""

    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        config: TTSConfig,
    ) -> AsyncIterator[bytes]:
        """Stream audio as text segments arrive."""

    @abstractmethod
    def get_available_voices(self, language: str) -> list[dict]:
        """Return available voices for a language."""
