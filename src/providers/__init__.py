"""Provider factories.

Maps a config slice (e.g. ``{"provider": "sarvam", ...}``) to the right adapter
instance for each interface. Phase 1 / 2 ships exactly one adapter per layer
(critical-path stack from PRD §12.4); adding more providers is a one-line
change to the corresponding registry dict.
"""

from __future__ import annotations

from typing import Any

from src.interfaces.llm import ILLMProvider
from src.interfaces.stt import ISTTProvider, IStreamingSTTProvider
from src.interfaces.telephony import ITelephonyProvider
from src.interfaces.tts import ITTSProvider
from src.interfaces.vector_store import IVectorStore
from src.providers.llm.anthropic_claude import AnthropicClaudeAdapter
from src.providers.llm.gemini import GeminiLLMAdapter
from src.providers.llm.groq import GroqLLMAdapter
from src.providers.stt.deepgram import DeepgramSTTAdapter
from src.providers.stt.gemini import GeminiSTTAdapter
from src.providers.stt.groq_whisper import GroqSTTAdapter
from src.providers.stt.sarvam import SarvamSTTAdapter
from src.providers.telephony.exotel import ExotelAdapter
from src.providers.telephony.infobip import InfobipAdapter
from src.providers.telephony.stringee import StringeeAdapter
from src.providers.telephony.telnyx import TelnyxAdapter
from src.providers.telephony.twilio import TwilioAdapter
from src.providers.tts.azure import AzureTTSAdapter
from src.providers.tts.elevenlabs import ElevenLabsTTSAdapter
from src.providers.tts.gemini import GeminiTTSAdapter
from src.providers.tts.google import GoogleTTSAdapter
from src.providers.tts.indicf5 import IndicF5TTSAdapter
from src.providers.tts.sarvam import SarvamTTSAdapter
from src.providers.vector_store.faiss_store import FAISSAdapter
from src.providers.vector_store.pgvector_store import PGVectorAdapter

STT_PROVIDERS: dict[str, type[ISTTProvider]] = {
    "sarvam": SarvamSTTAdapter,
    "groq": GroqSTTAdapter,
    "gemini": GeminiSTTAdapter,
}

STREAMING_STT_PROVIDERS: dict[str, type[IStreamingSTTProvider]] = {
    "deepgram": DeepgramSTTAdapter,
}

LLM_PROVIDERS: dict[str, type[ILLMProvider]] = {
    "groq": GroqLLMAdapter,
    "gemini": GeminiLLMAdapter,
    "anthropic": AnthropicClaudeAdapter,
    "claude": AnthropicClaudeAdapter,
}

TTS_PROVIDERS: dict[str, type[ITTSProvider]] = {
    "sarvam": SarvamTTSAdapter,
    "gemini": GeminiTTSAdapter,
    "google": GoogleTTSAdapter,
    "azure": AzureTTSAdapter,
    "elevenlabs": ElevenLabsTTSAdapter,
    "indicf5": IndicF5TTSAdapter,
}

TELEPHONY_PROVIDERS: dict[str, type[ITelephonyProvider]] = {
    "twilio": TwilioAdapter,
    "exotel": ExotelAdapter,
    "stringee": StringeeAdapter,
    "infobip": InfobipAdapter,
    "telnyx": TelnyxAdapter,
}

VECTOR_STORE_PROVIDERS: dict[str, type[IVectorStore]] = {
    "faiss": FAISSAdapter,
    "pgvector": PGVectorAdapter,
}


class UnknownProviderError(ValueError):
    """Raised when a configured provider name is not registered."""


def _lookup(registry: dict[str, type], name: str, kind: str):
    try:
        return registry[name]
    except KeyError as e:
        raise UnknownProviderError(
            f"Unknown {kind} provider '{name}'. Registered: {sorted(registry)}"
        ) from e


def get_stt_provider(config: dict[str, Any]) -> ISTTProvider:
    cls = _lookup(STT_PROVIDERS, config["provider"], "STT")
    return cls(config)


def get_streaming_stt_provider(config: dict[str, Any]) -> IStreamingSTTProvider:
    cls = _lookup(STREAMING_STT_PROVIDERS, config["provider"], "streaming STT")
    return cls(config)


def get_llm_provider(config: dict[str, Any]) -> ILLMProvider:
    cls = _lookup(LLM_PROVIDERS, config["provider"], "LLM")
    return cls(config)


def get_tts_provider(config: dict[str, Any]) -> ITTSProvider:
    cls = _lookup(TTS_PROVIDERS, config["provider"], "TTS")
    return cls(config)


def get_telephony_provider(config: dict[str, Any]) -> ITelephonyProvider:
    cls = _lookup(TELEPHONY_PROVIDERS, config["provider"], "telephony")
    return cls(config)


def get_vector_store(config: dict[str, Any]) -> IVectorStore:
    cls = _lookup(VECTOR_STORE_PROVIDERS, config["provider"], "vector store")
    return cls(config)


__all__ = [
    "STT_PROVIDERS",
    "STREAMING_STT_PROVIDERS",
    "LLM_PROVIDERS",
    "TTS_PROVIDERS",
    "TELEPHONY_PROVIDERS",
    "VECTOR_STORE_PROVIDERS",
    "UnknownProviderError",
    "get_stt_provider",
    "get_streaming_stt_provider",
    "get_llm_provider",
    "get_tts_provider",
    "get_telephony_provider",
    "get_vector_store",
]
