"""Selectable model variants per provider, by kind.

Powers ``GET /api/v1/models`` — the source the Register Tenant UI uses to
populate its provider + model dropdowns (so an operator picks e.g. Gemini
*flash* vs *flash-lite* vs *pro* rather than typing a model id). The first entry
in each list is the recommended default. Grounded in the adapters' DEFAULT_MODEL
constants + each provider's current public model line-up; maintained here as
vendors add/retire models.
"""

from __future__ import annotations

MODELS: dict[str, dict[str, list[str]]] = {
    "stt": {
        "sarvam": ["saaras:v3"],
        "groq": ["whisper-large-v3", "whisper-large-v3-turbo"],
        "deepgram": ["nova-2", "nova-3"],
    },
    "llm": {
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
        "sarvam": ["bulbul:v2", "bulbul:v3", "bulbul:v3-beta"],
    },
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
