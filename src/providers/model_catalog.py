"""Selectable model variants per provider, by kind.

Powers ``GET /api/v1/models`` — the source the Register Tenant UI and the
backoffice Pipeline tab use to populate their provider + model dropdowns (so
an operator picks e.g. Gemini *flash* vs *flash-lite* vs *pro* rather than
typing a model id). The first entry in each list is the recommended default,
and it must match the adapter's own default so the UI never recommends
something the adapter would not pick. Grounded in the adapters' DEFAULT_MODEL
constants + each provider's current public model line-up; maintained here by
hand as vendors add/retire models.

This is a hand-maintained mirror of the code-maintained provider registries
in ``src/providers/__init__.py`` (``STT_PROVIDERS``, ``LLM_PROVIDERS``,
``TTS_PROVIDERS``), so it can drift from them silently — a provider gets
registered/retired in code but nobody remembers to update this dict, and an
operator can't select (or is offered a dead) provider in the UI.
``tests/unit/test_model_catalog_drift.py`` guards against exactly that: it
fails if a registered provider is missing from a kind's list here, or a
listed provider isn't registered.

Some providers genuinely have no model dimension (Azure/Google TTS — voice is
the only selectable knob, chosen per-language elsewhere via
``TTSConfig.voice_id`` and each adapter's ``get_available_voices``); those are
listed with an empty model list rather than a fabricated model string.
"""

from __future__ import annotations

MODELS: dict[str, dict[str, list[str]]] = {
    "stt": {
        "sarvam": ["saaras:v3"],
        "groq": ["whisper-large-v3", "whisper-large-v3-turbo"],
        "gemini": ["gemini-3.5-flash"],
        # deepgram lives in the separate STREAMING_STT_PROVIDERS registry
        # (src/providers/__init__.py), not STT_PROVIDERS — it's a genuinely
        # distinct streaming-only interface, not a drifted duplicate. Still a
        # real, selectable "stt"-kind option for this catalog response.
        "deepgram": ["nova-2", "nova-3"],
    },
    "llm": {
        # Self-hosted vLLM on the IndicF5 RunPod pod (OpenAI-compatible).
        "vllm": ["Qwen/Qwen2.5-14B-Instruct-AWQ"],
        "gemini": [
            "gemini-3.5-flash",
            # 2.x models still work on older Gemini projects but 404 on
            # projects created after mid-2026 ("no longer available to new
            # users") — kept for tenants pinned to an old key.
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ],
        "groq": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        "anthropic": [
            "claude-haiku-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-8",
        ],
    },
    "tts": {
        # "bulbul:v2" was deprecated by Sarvam in 2026-09 (the API 400s on it
        # now — "has been deprecated. Please use 'bulbul:v3' instead.") and is
        # dropped here so the registration UI stops offering a model the API
        # rejects. "bulbul:v3" is the recommended default (see
        # src.providers.tts.sarvam.DEFAULT_MODEL).
        "sarvam": ["bulbul:v3", "bulbul:v3-beta"],
        "gemini": [
            "gemini-2.5-flash-preview-tts",
            # The adapter's own automatic retry target when the primary 400s
            # or 404s (src.providers.tts.gemini._FALLBACK_MODEL). Listed
            # (not just used silently) so an operator can pin to it
            # deliberately too — it's a real, callable model id, just not
            # the recommended one.
            "gemini-2.0-flash-exp",
        ],
        # Azure and Google TTS have no model concept at all — voice is the
        # only selectable knob (chosen per-language via TTSConfig.voice_id /
        # get_available_voices), so there is no model id to recommend as a
        # default. An empty list here is intentional, not missing data: the
        # backoffice Pipeline tab's provider dropdown is built from this
        # dict's *keys* (static/backoffice.html's pipeLayerHtml), so the
        # provider still shows up and is selectable; its model dropdown just
        # offers nothing beyond "leave unchanged", which is correct since
        # there is nothing to pick.
        "azure": [],
        "google": [],
        # Recommended models per the adapter's own docstring
        # (src/providers/tts/elevenlabs.py); eleven_flash_v2_5 is also
        # DEFAULT_MODEL there. ElevenLabs is the fastest TTS in this
        # project's 462-turn latency benchmark (1442ms avg).
        "elevenlabs": [
            "eleven_flash_v2_5",
            "eleven_turbo_v2_5",
            "eleven_multilingual_v2",
        ],
        # Self-hosted fine-tuned IndicF5 voice server (one model = one voice).
        "indicf5": ["indicf5-finetuned"],
    },
    # "s2s" has no counterpart registry in src/providers/__init__.py at all —
    # src/bootstrap.py (lines ~637, 673) imports and instantiates
    # GeminiLiveSession directly rather than dispatching through a provider
    # registry, so pipeline.realtime.provider is validated as present but
    # never looked up in a dict. Nothing to drift-check here; kept hand-listed.
    "s2s": {
        "gemini_live": [
            "gemini-3.1-flash-live-preview",
            "gemini-2.5-flash-live-preview",
        ],
    },
}


def list_models() -> dict[str, dict[str, list[str]]]:
    """Deep copy of the catalog (so callers can't mutate the source)."""
    return {kind: {prov: list(models) for prov, models in provs.items()}
            for kind, provs in MODELS.items()}


def models_for(kind: str, provider: str) -> list[str]:
    return list(MODELS.get(kind, {}).get(provider, []))
