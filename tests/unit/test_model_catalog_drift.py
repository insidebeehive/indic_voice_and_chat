"""tests/unit/test_model_catalog_drift.py

Tests that ``src/providers/model_catalog.py``'s hand-maintained ``MODELS``
dict stays in sync with the code-maintained provider registries in
``src/providers/__init__.py``. Modeled on
``tests/unit/test_crm_catalog_seeding.py``, which solves the identical
problem for CRM tools (a hand-maintained list beside a code-maintained one).

``MODELS`` drives ``GET /api/v1/models``, which populates BOTH the Register
Tenant page's and the backoffice Pipeline tab's provider/model dropdowns. A
provider that is registered (a working adapter exists) but missing from
``MODELS`` is unselectable in either UI. A provider listed in ``MODELS`` but
not registered is offered in the dropdown and then fails at runtime
(``UnknownProviderError``) the moment someone selects it. The fix differs by
direction, so failures below name both the provider and the direction.
"""
from __future__ import annotations

import pytest

from src.providers import (
    LLM_PROVIDERS,
    STREAMING_STT_PROVIDERS,
    STT_PROVIDERS,
    TTS_PROVIDERS,
)
from src.providers.model_catalog import MODELS

# --- Known, deliberate exceptions -------------------------------------------
#
# Each of these looks like drift at first glance. They are not bugs — this
# test must not flag them — so each is spelled out with the reason, rather
# than the test just happening to pass.

# 1. "deepgram" is registered under STREAMING_STT_PROVIDERS, a separate
#    registry for the streaming-only STT interface (src/providers/__init__.py
#    lines 44-46) — not STT_PROVIDERS. It is still a real, selectable
#    "stt"-kind provider for this catalog response, so MODELS["stt"]
#    correctly lists it even though it will never appear in STT_PROVIDERS
#    itself. Merge the two registries for the stt comparison so deepgram
#    counts as registered instead of flagging as "offered but not registered".
_STT_REGISTRY: dict[str, type] = dict(STT_PROVIDERS)
_STT_REGISTRY.update(STREAMING_STT_PROVIDERS)

# 2. "claude" and "anthropic" are two registry keys that map to the exact
#    same AnthropicClaudeAdapter class (src/providers/__init__.py) — an
#    alias, not two distinct providers. One catalog entry ("anthropic") is
#    enough to make the adapter selectable, so "claude" showing up as
#    "registered but not listed" is expected, not drift.
_LLM_ALIASES_COVERED_BY_ANOTHER_KEY = {"claude"}

# 3. "s2s" / "gemini_live" has no provider registry at all: src/bootstrap.py
#    (~line 637, ~673) imports and instantiates GeminiLiveSession directly
#    rather than dispatching through a registry dict, so
#    pipeline.realtime.provider is validated as present but never looked up
#    anywhere. There is nothing to diff MODELS["s2s"] against — it is
#    excluded from the registry comparison below entirely (not compared in
#    either direction), not silently skipped.
_KINDS_WITH_NO_REGISTRY = {"s2s"}

_REGISTRIES: dict[str, dict[str, type]] = {
    "stt": _STT_REGISTRY,
    "llm": LLM_PROVIDERS,
    "tts": TTS_PROVIDERS,
}


@pytest.mark.parametrize("kind", sorted(_REGISTRIES))
def test_every_registered_provider_is_selectable(kind: str) -> None:
    """A provider with a working adapter must be choosable in the catalog —
    otherwise an operator can never pick it in the Register Tenant page or
    the backoffice Pipeline tab dropdowns (this is exactly how ElevenLabs,
    Azure, Google TTS, and Gemini STT went dark)."""
    registered = set(_REGISTRIES[kind])
    if kind == "llm":
        registered -= _LLM_ALIASES_COVERED_BY_ANOTHER_KEY
    listed = set(MODELS.get(kind, {}))
    missing = registered - listed
    assert not missing, (
        f"{kind}: provider(s) {sorted(missing)} are registered but not "
        f"selectable — registered in src/providers/__init__.py but absent "
        f"from MODELS[{kind!r}] in src/providers/model_catalog.py. "
        f"Fix: add {sorted(missing)} to MODELS[{kind!r}]."
    )


@pytest.mark.parametrize("kind", sorted(_REGISTRIES))
def test_every_listed_provider_is_registered(kind: str) -> None:
    """A provider offered in the catalog must actually work when picked —
    otherwise selecting it in either UI raises UnknownProviderError at
    runtime instead of failing during code review."""
    registered = set(_REGISTRIES[kind])
    listed = set(MODELS.get(kind, {}))
    offered_but_dead = listed - registered
    assert not offered_but_dead, (
        f"{kind}: provider(s) {sorted(offered_but_dead)} are offered but "
        f"not registered — listed in MODELS[{kind!r}] "
        f"(src/providers/model_catalog.py) but absent from "
        f"src/providers/__init__.py's registry. "
        f"Fix: remove {sorted(offered_but_dead)} from MODELS[{kind!r}], or "
        f"register the adapter if it's meant to exist."
    )


def test_s2s_kind_is_a_documented_exception_not_a_silent_skip() -> None:
    """s2s has no provider registry to diff against (see the module-level
    comment above) — assert that fact explicitly so a reader sees it was a
    deliberate exclusion, and so this test breaks loudly if someone later
    adds an s2s registry without also wiring it into _REGISTRIES here."""
    assert _KINDS_WITH_NO_REGISTRY == {"s2s"}
    assert "s2s" not in _REGISTRIES
    assert "s2s" in MODELS  # still hand-maintained; just not drift-checked


# --- Recommended-default convention -----------------------------------------


def test_recommended_default_matches_adapter_default() -> None:
    """The first model in each non-empty list is documented as the
    recommended default (model_catalog.py's module docstring). Check it
    against the adapter's own default constant where one is exported, so the
    UI never recommends a model the adapter itself would not pick."""
    from src.providers.stt.gemini import _DEFAULT_MODEL as gemini_stt_default
    from src.providers.stt.sarvam import DEFAULT_MODEL as sarvam_stt_default
    from src.providers.tts.elevenlabs import DEFAULT_MODEL as elevenlabs_tts_default
    from src.providers.tts.gemini import _DEFAULT_MODEL as gemini_tts_default
    from src.providers.tts.sarvam import DEFAULT_MODEL as sarvam_tts_default

    checks = {
        ("stt", "sarvam"): sarvam_stt_default,
        ("stt", "gemini"): gemini_stt_default,
        ("tts", "sarvam"): sarvam_tts_default,
        ("tts", "gemini"): gemini_tts_default,
        ("tts", "elevenlabs"): elevenlabs_tts_default,
    }
    for (kind, provider), expected_default in checks.items():
        first = MODELS[kind][provider][0]
        assert first == expected_default, (
            f"MODELS[{kind!r}][{provider!r}][0] is {first!r}, but the "
            f"adapter's own default is {expected_default!r} — the catalog "
            f"would recommend a model the adapter would not itself pick."
        )


def test_azure_and_google_tts_are_intentionally_empty_not_missing() -> None:
    """Azure and Google TTS have no model dimension — voice is the only
    selectable knob, chosen per-language elsewhere (TTSConfig.voice_id /
    each adapter's get_available_voices). Pin the empty-list representation
    down explicitly so a future edit doesn't "fix" it into a fabricated
    model id, and so the two providers stay covered by the registered/listed
    checks above (present as keys, just with nothing under them)."""
    assert MODELS["tts"]["azure"] == []
    assert MODELS["tts"]["google"] == []
