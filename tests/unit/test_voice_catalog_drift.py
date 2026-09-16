"""tests/unit/test_voice_catalog_drift.py

Tests that ``src/providers/voice_catalog.py``'s hand-maintained voice rosters
stay in sync with the code-maintained TTS provider registry in
``src/providers/__init__.py``. Modeled on
``tests/unit/test_model_catalog_drift.py``, which solves the identical
problem for the model catalog (itself modeled on ``test_crm_catalog_seeding.py``
for CRM tools).

``voice_catalog.py`` had exactly this drift bug: it covered only 2 of the 6
registered TTS providers (sarvam, gemini) until azure/google/elevenlabs/
indicf5 were added. ``list_voices`` drives ``GET /api/v1/voices``, which
populates a tenant's TTS voice-selection dropdown -- a provider that is
registered (a working adapter exists in ``TTS_PROVIDERS``) but returns no
voices here is unselectable in that dropdown, the same pattern that recently
hid four unregistered CRM tools (commit cb60ccd -- catalog.py listed them,
but no tenant's crm_tools rows did), four unselectable TTS providers, and a
deprecated default model until production broke.
"""
from __future__ import annotations

import pytest

from src.api.catalog import VoiceItem
from src.providers import TTS_PROVIDERS
from src.providers.voice_catalog import list_voices

# Candidate BCP-47 languages to probe for "at least one language a provider
# supports" -- covers both per-language rosters (sarvam, azure, google,
# indicf5, all of which include hi-IN) and language-independent ones (gemini,
# elevenlabs, which return the same roster regardless of language).
_CANDIDATE_LANGUAGES = ["hi-IN", "en-IN", "en-US"]


def _roster_for_any_language(provider: str) -> list[dict]:
    for lang in _CANDIDATE_LANGUAGES:
        voices = list_voices(provider, lang)
        if voices:
            return voices
    return []


@pytest.mark.parametrize("provider", sorted(TTS_PROVIDERS))
def test_every_registered_tts_provider_has_a_voice_roster(provider: str) -> None:
    """A TTS provider with a working adapter (registered in ``TTS_PROVIDERS``)
    must have a non-empty voice roster in the catalog for at least one
    language it claims to support -- otherwise it is unselectable in the
    /voices-driven dropdown even though the adapter itself works fine."""
    voices = _roster_for_any_language(provider)
    assert voices, (
        f"TTS provider {provider!r} is registered in "
        f"src/providers/__init__.py's TTS_PROVIDERS but "
        f"src.providers.voice_catalog.list_voices({provider!r}, ...) returns "
        f"an empty roster for every candidate language "
        f"({_CANDIDATE_LANGUAGES!r}). Fix: add {provider!r}'s roster to "
        f"src/providers/voice_catalog.py."
    )


@pytest.mark.parametrize("provider", sorted(TTS_PROVIDERS))
def test_every_voice_entry_satisfies_voice_item(provider: str) -> None:
    """Every entry the catalog returns for a registered provider must satisfy
    ``VoiceItem`` (src/api/catalog.py) -- the Pydantic model the ``/voices``
    route serializes into. A provider returning a differently-shaped dict
    would 500 the route the moment that provider's voices are requested.

    Two things this test would otherwise miss silently, both asserted
    explicitly rather than left to the loop below:

    - An empty roster (e.g. a provider dropped from
      ``_TTS_LANGUAGE_ROSTERS``/``_TTS_FLAT_ROSTERS`` in voice_catalog.py)
      makes the ``for v in voices`` loop body never execute, so the test
      passes vacuously -- the very drift this file exists to catch.
    - ``VoiceItem.gender`` is ``Optional`` (src/api/catalog.py), so a roster
      entry missing ``gender`` entirely still satisfies ``VoiceItem`` and the
      loop below would say nothing -- but the voicebot prompt selects
      gendered Hindi/Marathi forms off this field (see
      static/backoffice.html's voicePickerHtml), so a silently-blank gender
      is a content bug, not a schema violation.
    """
    voices = _roster_for_any_language(provider)
    assert voices, (
        f"TTS provider {provider!r} has an empty roster for every candidate "
        f"language ({_CANDIDATE_LANGUAGES!r}) -- test_every_voice_entry_"
        f"satisfies_voice_item would pass vacuously on this provider. Fix: "
        f"add {provider!r}'s roster to src/providers/voice_catalog.py."
    )
    for v in voices:
        try:
            VoiceItem(**v)
        except Exception as e:  # noqa: BLE001
            pytest.fail(
                f"TTS provider {provider!r} returned a voice entry that does "
                f"not satisfy VoiceItem: {v!r} ({e})"
            )
        assert v.get("gender"), (
            f"TTS provider {provider!r} returned a voice entry with no "
            f"gender populated: {v!r}. VoiceItem.gender is Optional so this "
            f"would not otherwise be caught, but the voicebot prompt selects "
            f"gendered Hindi/Marathi forms off this field."
        )


def test_gemini_live_s2s_catalog_is_a_documented_exception() -> None:
    """``gemini_live`` (S2S realtime) has no entry in ``TTS_PROVIDERS`` -- it's
    dispatched directly in src/bootstrap.py rather than through a provider
    registry (see test_model_catalog_drift.py's identical note for MODELS).
    It still has a real, non-empty catalog; just not drift-checked against a
    registry that doesn't exist for it."""
    assert "gemini_live" not in TTS_PROVIDERS
    assert list_voices("gemini_live")


def test_gemini_live_is_returned_unnormalized_with_style_intact() -> None:
    """``list_voices("gemini_live")`` must return entries as-is, NOT run
    through ``_normalize`` -- unlike every other provider, this catalog is
    deliberately unnormalized (see the ``list_voices`` docstring) because
    dev_console.html:782 reads each voice's ``style`` field directly (fed by
    src/api/dev_console.py:186). ``_normalize`` projects entries down to
    ``{voice_id, gender}`` and would silently drop ``style``, which
    dev_console would then render as "undefined" for every voice -- and
    every one of this file's other tests, plus the wider suite, passes
    either way, since none of them read ``style``. This test exists because
    that specific mutation (swapping the gemini_live branch's ``return
    list(_GEMINI_LIVE_VOICES)`` for ``return _normalize(_GEMINI_LIVE_VOICES)``)
    was verified to leave all other voice-catalog tests green."""
    voices = list_voices("gemini_live")
    assert voices, "gemini_live catalog must be non-empty"
    for v in voices:
        assert "style" in v, (
            f"gemini_live voice entry is missing 'style': {v!r} -- "
            f"list_voices('gemini_live') must return raw entries, not "
            f"_normalize(...)'d ones, since dev_console.html reads .style "
            f"directly."
        )
