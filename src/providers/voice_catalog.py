"""Voice rosters per provider — a static catalog (no adapter/key needed).

Powers ``GET /api/v1/voices?provider=&language=``. For TTS providers the roster
is the provider's available speakers (with gender); for the S2S realtime
provider it's the available voices.
"""

from __future__ import annotations

from src.providers.tts.sarvam import LANGUAGE_VOICES as _SARVAM_VOICES

# Gemini Live (S2S) realtime voices — full set of 30.
# Source: https://ai.google.dev/gemini-api/docs/speech-generation
# Gender inferred from mythological/astronomical origin; Google does not publish official labels.
_GEMINI_LIVE_VOICES = [
    # Original launch voices
    {"voice_id": "Aoede",           "gender": "female", "style": "Breezy"},
    {"voice_id": "Kore",            "gender": "female", "style": "Firm"},
    {"voice_id": "Leda",            "gender": "female", "style": "Youthful"},
    {"voice_id": "Puck",            "gender": "male",   "style": "Upbeat"},
    {"voice_id": "Charon",          "gender": "male",   "style": "Informative"},
    {"voice_id": "Fenrir",          "gender": "male",   "style": "Excitable"},
    {"voice_id": "Orus",            "gender": "male",   "style": "Firm"},
    {"voice_id": "Zephyr",          "gender": "female", "style": "Bright"},
    # Extended voices
    {"voice_id": "Achernar",        "gender": "female", "style": "Soft"},
    {"voice_id": "Achird",          "gender": "male",   "style": "Friendly"},
    {"voice_id": "Algenib",         "gender": "male",   "style": "Gravelly"},
    {"voice_id": "Algieba",         "gender": "male",   "style": "Smooth"},
    {"voice_id": "Alnilam",         "gender": "female", "style": "Firm"},
    {"voice_id": "Autonoe",         "gender": "female", "style": "Bright"},
    {"voice_id": "Callirrhoe",      "gender": "female", "style": "Easy-going"},
    {"voice_id": "Despina",         "gender": "female", "style": "Smooth"},
    {"voice_id": "Enceladus",       "gender": "male",   "style": "Breathy"},
    {"voice_id": "Erinome",         "gender": "female", "style": "Clear"},
    {"voice_id": "Gacrux",          "gender": "male",   "style": "Mature"},
    {"voice_id": "Iapetus",         "gender": "male",   "style": "Clear"},
    {"voice_id": "Laomedeia",       "gender": "female", "style": "Upbeat"},
    {"voice_id": "Pulcherrima",     "gender": "female", "style": "Forward"},
    {"voice_id": "Rasalgethi",      "gender": "male",   "style": "Informative"},
    {"voice_id": "Sadachbia",       "gender": "male",   "style": "Lively"},
    {"voice_id": "Sadaltager",      "gender": "male",   "style": "Knowledgeable"},
    {"voice_id": "Schedar",         "gender": "female", "style": "Even"},
    {"voice_id": "Sulafat",         "gender": "female", "style": "Warm"},
    {"voice_id": "Umbriel",         "gender": "male",   "style": "Easy-going"},
    {"voice_id": "Vindemiatrix",    "gender": "female", "style": "Gentle"},
    {"voice_id": "Zubenelgenubi",   "gender": "male",   "style": "Casual"},
]


def list_voices(provider: str, language: str = "hi-IN") -> list[dict]:
    """Return ``[{voice_id, gender}, ...]`` for a provider (+ language for TTS).

    Empty list for an unknown provider/language. Sarvam ``bulbul:v2`` roster comes
    straight from the adapter's ``LANGUAGE_VOICES`` (no key needed).
    """
    p = (provider or "").lower()
    if p == "sarvam":
        return list(_SARVAM_VOICES.get(language, []))
    if p in ("gemini_live", "gemini"):
        return list(_GEMINI_LIVE_VOICES)
    return []


def supported_providers() -> list[str]:
    return ["sarvam", "gemini_live"]
