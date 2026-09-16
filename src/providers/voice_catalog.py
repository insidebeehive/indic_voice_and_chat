"""Voice rosters per provider — a static catalog (no adapter/key needed).

Powers ``GET /api/v1/voices?provider=&language=``. For TTS providers the roster
is the provider's available speakers (with gender); for the S2S realtime
provider it's the available voices.

Covers every registered TTS provider (``src.providers.TTS_PROVIDERS``: sarvam,
gemini, google, azure, elevenlabs, indicf5) plus the separate ``gemini_live``
S2S realtime catalog, which has no provider registry of its own (see
``tests/unit/test_model_catalog_drift.py``'s note on ``s2s``). Each TTS
provider's roster is imported straight from that adapter module's own
module-level constant (``LANGUAGE_VOICES`` / ``_VOICES`` / ``_PRESET_VOICES``)
rather than hand-copied here — the same failure mode already bit the model
catalog (``src/providers/model_catalog.py``, fixed in e773c4b) and the CRM
tool registry (cb60ccd). Those two were fixed differently, by guarding a
hand-maintained list with a drift test; importing the source of truth
directly, as here, is only possible because the adapters already expose one.
Building adapters is deliberately avoided: several
raise in ``__init__`` without an API key (e.g. Sarvam), and ``/voices`` is
reference data an operator may need before any key is configured.

ElevenLabs is the one provider whose *real* roster (a tenant's cloned/custom
voices) requires a live API call to fetch — see
``ElevenLabsTTSAdapter.get_available_voices``. This catalog exposes only its
static ``_PRESET_VOICES`` fallback (8 built-in voices), the same list the
adapter itself falls back to when keyless. Cloned/custom voices are
intentionally NOT visible here.

Language handling: Azure, Google, Sarvam, and IndicF5 rosters are
per-language; Gemini and ElevenLabs are language-independent (one flat list
regardless of ``language``). For a per-language provider, a ``language`` it
does not support returns an empty list here — deliberately *not* replicating
the Azure/Google adapters' own synthesis-time fallback to ``hi-IN`` (see
``AzureTTSAdapter.get_available_voices`` / ``GoogleTTSAdapter.get_available_voices``).
That fallback exists so a synthesis call doesn't fail outright; this catalog
is reference data for a UI dropdown, where silently substituting a different
language's roster would misrepresent what the provider actually supports for
the language asked about.
"""

from __future__ import annotations

from src.providers.tts.azure import _VOICES as _AZURE_VOICES
from src.providers.tts.elevenlabs import _PRESET_VOICES as _ELEVENLABS_VOICES
from src.providers.tts.gemini import _VOICES as _GEMINI_TTS_VOICES
from src.providers.tts.google import _VOICES as _GOOGLE_VOICES
from src.providers.tts.indicf5 import LANGUAGE_VOICES as _INDICF5_VOICES
from src.providers.tts.sarvam import LANGUAGE_VOICES as _SARVAM_VOICES

# ``_VOICES`` (azure/gemini/google) and ``_PRESET_VOICES`` (elevenlabs) are
# private-by-convention module constants in their respective adapters, not
# part of a public API. They are imported directly here rather than through
# an adapter instance because that is exactly the data ``get_available_voices``
# itself serves (see each adapter), and building a third, hand-maintained copy
# is the same drift trap this file already fell into once. Adapters are
# out of scope for this change (see module docstring above), so the rename to
# a public name (e.g. ``AVAILABLE_VOICES``) that would make this import
# convention-clean is left as a follow-up, not made here. ``LANGUAGE_VOICES``
# (sarvam, indicf5) is already public and imported the same way.

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
    {"voice_id": "Fenrir",          "gender": "female", "style": "Excitable"},
    {"voice_id": "Orus",            "gender": "male",   "style": "Firm"},
    {"voice_id": "Zephyr",          "gender": "female", "style": "Bright"},
    # Extended voices
    {"voice_id": "Achernar",        "gender": "female", "style": "Soft"},
    {"voice_id": "Achird",          "gender": "male",   "style": "Friendly"},
    {"voice_id": "Algenib",         "gender": "male",   "style": "Gravelly"},
    {"voice_id": "Algieba",         "gender": "male",   "style": "Smooth"},
    {"voice_id": "Alnilam",         "gender": "male",   "style": "Firm"},
    {"voice_id": "Autonoe",         "gender": "female", "style": "Bright"},
    {"voice_id": "Callirrhoe",      "gender": "female", "style": "Easy-going"},
    {"voice_id": "Despina",         "gender": "female", "style": "Smooth"},
    {"voice_id": "Enceladus",       "gender": "male",   "style": "Breathy"},
    {"voice_id": "Erinome",         "gender": "female", "style": "Clear"},
    {"voice_id": "Gacrux",          "gender": "male",   "style": "Mature"},
    {"voice_id": "Iapetus",         "gender": "male",   "style": "Clear"},
    {"voice_id": "Laomedeia",       "gender": "female", "style": "Upbeat"},
    {"voice_id": "Pulcherrima",     "gender": "male",   "style": "Forward"},
    {"voice_id": "Rasalgethi",      "gender": "male",   "style": "Informative"},
    {"voice_id": "Sadachbia",       "gender": "male",   "style": "Lively"},
    {"voice_id": "Sadaltager",      "gender": "male",   "style": "Knowledgeable"},
    {"voice_id": "Schedar",         "gender": "male",   "style": "Even"},
    {"voice_id": "Sulafat",         "gender": "female", "style": "Warm"},
    {"voice_id": "Umbriel",         "gender": "male",   "style": "Easy-going"},
    {"voice_id": "Vindemiatrix",    "gender": "female", "style": "Gentle"},
    {"voice_id": "Zubenelgenubi",   "gender": "male",   "style": "Casual"},
]


def _normalize(entries: list[dict]) -> list[dict]:
    """Project a roster entry down to the shape ``VoiceItem`` (src/api/catalog.py)
    declares: ``{"voice_id": str, "gender": str}``. Some adapters' rosters carry
    extra fields (elevenlabs' preset list has ``name``; the gemini_live catalog
    has ``style``) — those extra fields are meaningless per-provider noise to a
    UI dropdown built generically off this catalog, and ``VoiceItem`` would drop
    them silently anyway. Normalizing here keeps ``list_voices``'s own return
    value uniform for any caller, not just the ones going through the API route.
    """
    return [{"voice_id": v["voice_id"], "gender": v.get("gender", "")} for v in entries]


# Per-language TTS rosters: provider -> {language: [voices]}.
_TTS_LANGUAGE_ROSTERS: dict[str, dict[str, list[dict]]] = {
    "sarvam": _SARVAM_VOICES,
    "azure": _AZURE_VOICES,
    "google": _GOOGLE_VOICES,
    "indicf5": _INDICF5_VOICES,
}

# Language-independent TTS rosters: provider -> [voices] (same list for every
# language; Gemini TTS and ElevenLabs don't key their roster by language).
_TTS_FLAT_ROSTERS: dict[str, list[dict]] = {
    "gemini": _GEMINI_TTS_VOICES,
    "elevenlabs": _ELEVENLABS_VOICES,
}


def list_voices(provider: str, language: str = "hi-IN") -> list[dict]:
    """Return ``[{voice_id, gender}, ...]`` for a provider (+ language for TTS).

    Empty list for an unknown provider, or for a language a per-language
    provider does not support (no hi-IN fallback here — see module docstring).
    ``gemini_live`` is the S2S realtime catalog (language-independent, richer
    metadata) and is returned as-is, unnormalized, since dev_console's UI
    reads its ``style`` field directly.
    """
    p = (provider or "").lower()
    if p == "gemini_live":
        return list(_GEMINI_LIVE_VOICES)
    if p in _TTS_FLAT_ROSTERS:
        return _normalize(_TTS_FLAT_ROSTERS[p])
    if p in _TTS_LANGUAGE_ROSTERS:
        return _normalize(_TTS_LANGUAGE_ROSTERS[p].get(language, []))
    return []


def supported_providers() -> list[str]:
    return [*sorted(_TTS_LANGUAGE_ROSTERS), *sorted(_TTS_FLAT_ROSTERS), "gemini_live"]


def gender_from_voice_id(voice_id: str) -> str:
    """Return 'male', 'female', or '' for a voice_id across all known TTS catalogs."""
    vid = (voice_id or "").strip().lower()
    if not vid:
        return ""
    all_lists: list[list[dict]] = (
        list(_SARVAM_VOICES.values())
        + list(_GOOGLE_VOICES.values())
        + list(_AZURE_VOICES.values())
        + list(_INDICF5_VOICES.values())
        + [_GEMINI_TTS_VOICES, _GEMINI_LIVE_VOICES, _ELEVENLABS_VOICES]
    )
    for voices in all_lists:
        for v in voices:
            if (v.get("voice_id") or "").lower() == vid:
                return v.get("gender", "")
    return ""
