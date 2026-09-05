from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from src.pipeline.text_normalize import (
    DEFAULT_PRONUNCIATIONS,
    apply_pronunciations,
    normalize_for_tts,
)


def _load_migration_betting_pronunciations() -> dict[str, str]:
    """Load ``_BETTING_PRONUNCIATIONS`` straight from the migration module
    (alembic/versions/0018_crm_pronunciation_overrides.py) rather than
    duplicating the dict here -- a future transcription error in the
    migration's actual backfill data must fail this test, not go unnoticed
    behind a second, independent copy. ``alembic/versions`` has no
    ``__init__.py`` (not a regular importable package, and the installed
    ``alembic`` pip package shadows it), so load the file directly by path.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic" / "versions" / "0018_crm_pronunciation_overrides.py"
    )
    spec = importlib.util.spec_from_file_location("_migration_0018_crm_pronunciation_overrides", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._BETTING_PRONUNCIATIONS


def test_rewrites_known_terms_to_devanagari() -> None:
    out = apply_pronunciations("WhatsApp par link bhejun? Bonus bhi hai.")
    # Mispronounced English terms are gone, replaced by Devanagari.
    assert "WhatsApp" not in out and "Bonus" not in out and "link" not in out
    assert DEFAULT_PRONUNCIATIONS["WhatsApp"] in out
    assert DEFAULT_PRONUNCIATIONS["bonus"] in out
    assert DEFAULT_PRONUNCIATIONS["link"] in out
    # Surrounding text is preserved.
    assert "bhejun" in out and "bhi hai" in out


def test_case_insensitive_whole_word_only() -> None:
    out = apply_pronunciations("whatsapp WHATSAPP Whatsapp")
    assert out.count(DEFAULT_PRONUNCIATIONS["WhatsApp"]) == 3
    # 'app' must not match inside another word, only standalone.
    assert apply_pronunciations("happy apple") == "happy apple"
    assert DEFAULT_PRONUNCIATIONS["app"] in apply_pronunciations("open the app")


def test_empty_and_no_match_passthrough() -> None:
    assert apply_pronunciations("") == ""
    assert apply_pronunciations("sirf hindi text hai") == "sirf hindi text hai"


def test_extra_overrides_merge_over_defaults() -> None:
    out = apply_pronunciations("Cash lo aur ZyxBrand", extra={"ZyxBrand": "ज़िक्सब्रांड"})
    assert "ज़िक्सब्रांड" in out
    assert DEFAULT_PRONUNCIATIONS["cash"] in out


def test_rewrites_generic_gaming_adjacent_terms_by_default() -> None:
    # "join"/"explore"/"risk" are generic English loanwords (not betting-vertical)
    # and stay in the shared default regardless of which CRM/vertical is active.
    out = apply_pronunciations(
        "Aap join karke explore kar sakte hain, koi risk nahi."
    )
    assert "join" not in out and DEFAULT_PRONUNCIATIONS["join"] in out
    assert "explore" not in out and DEFAULT_PRONUNCIATIONS["explore"] in out
    assert "risk" not in out and DEFAULT_PRONUNCIATIONS["risk"] in out
    # Surrounding Hindi/Hinglish text is preserved untouched.
    assert "koi" in out and "nahi" in out


def test_betting_vertical_terms_not_in_shared_default() -> None:
    # Casino/Cricket/Football/Matka/Sports/Tennis/Basketball/market/match/
    # matches/IPL/Aviator/betting are betting-vertical vocabulary, moved to
    # per-CRM Crm.pronunciation_overrides (see src/models/crm.py) instead of
    # the shared, industry-neutral DEFAULT_PRONUNCIATIONS -- a CRM with no
    # overrides must NOT get these substitutions.
    betting_words = [
        "Casino", "Aviator", "betting", "Cricket", "Football", "Matka",
        "Sports", "Tennis", "Basketball", "market", "match", "matches", "IPL",
    ]
    for word in betting_words:
        assert word not in DEFAULT_PRONUNCIATIONS
    out = apply_pronunciations("Casino mein Cricket aur Matka ka market hai.")
    # No overrides supplied -> passes through completely unchanged (whole-word
    # betting terms have no default entry to match against).
    assert out == "Casino mein Cricket aur Matka ka market hai."


def test_betting_crm_override_reproduces_prior_behavior() -> None:
    # A CRM whose Crm.pronunciation_overrides was backfilled with the removed
    # betting entries (see alembic/versions/0018_crm_pronunciation_overrides.py)
    # gets byte-for-byte the same substitution as before this change.
    betting_overrides = _load_migration_betting_pronunciations()
    # Pin the migration's actual values independently of the loaded dict —
    # without this, every assertion below compares betting_overrides against
    # itself and a mistyped Devanagari string in the backfill would still
    # pass. This is the literal set alembic/versions/0018_crm_pronunciation_overrides.py
    # ships to the production CRM's pronunciation_overrides.
    assert betting_overrides == {
        "Casino": "कसीनो",
        "Aviator": "एविएटर",
        "betting": "बेटिंग",
        "Cricket": "क्रिकेट",
        "Football": "फुटबॉल",
        "Matka": "मटका",
        "Sports": "स्पोर्ट्स",
        "Tennis": "टेनिस",
        "Basketball": "बास्केटबॉल",
        "market": "मार्केट",
        "match": "मैच",
        "matches": "मैचेस",
        "IPL": "आईपीएल",
    }
    out = apply_pronunciations(
        "Cricket aur Football dono hai, aap join karke explore kar sakte hain, "
        "koi risk nahi. Matka bhi khel sakte hain.",
        extra=betting_overrides,
    )
    assert "Cricket" not in out and betting_overrides["Cricket"] in out
    assert "Football" not in out and betting_overrides["Football"] in out
    assert "join" not in out and DEFAULT_PRONUNCIATIONS["join"] in out
    assert "explore" not in out and DEFAULT_PRONUNCIATIONS["explore"] in out
    assert "risk" not in out and DEFAULT_PRONUNCIATIONS["risk"] in out
    assert "Matka" not in out and betting_overrides["Matka"] in out
    # Surrounding Hindi/Hinglish text is preserved untouched.
    assert "aur" in out and "dono hai" in out and "koi" in out and "nahi" in out


def test_normalize_for_tts_warns_on_devanagari_language_gap(caplog) -> None:
    # Devanagari-dominant text with one genuine gap word -- "ZyxUnknownBrand"
    # is guaranteed not to be in DEFAULT_PRONUNCIATIONS. Script-dominant text
    # (>=60% Devanagari) with a residual Latin word is exactly the case this
    # mechanism must catch.
    with caplog.at_level(logging.WARNING):
        normalize_for_tts(
            "व्हाट्सऐप पर लिंक भेजता हूं ZyxUnknownBrand", language="hi-IN"
        )
    assert any(
        r.levelno == logging.WARNING and "ZyxUnknownBrand" in str(r.__dict__.get("words"))
        for r in caplog.records
    )


def test_normalize_for_tts_no_warning_when_fully_covered(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        normalize_for_tts("व्हाट्सऐप पर लिंक भेजता हूं", language="hi-IN")
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_normalize_for_tts_no_warning_for_romanized_hinglish(caplog) -> None:
    # Romanized Hinglish ("par", "bhejun", "aur", "khelo" are ordinary Hindi
    # words correctly left in Latin script) must NOT be flagged -- this is
    # expected input shape, not a pronunciation-dictionary gap. This is the
    # exact false positive an earlier version of this mechanism produced.
    with caplog.at_level(logging.WARNING):
        normalize_for_tts("WhatsApp par link bhejun aur Cricket khelo", language="hi-IN")
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


def test_normalize_for_tts_warns_for_script_dominant_unsupported_language(caplog) -> None:
    # Telugu ("te") has zero normalization today. This is real Telugu-script
    # text (generated via indic_transliteration from a known-correct
    # Devanagari sentence, so the Telugu itself is linguistically valid) with
    # one residual English word -- exactly the "no normalization exists yet
    # for this language" gap this task must surface.
    with caplog.at_level(logging.WARNING):
        result = normalize_for_tts(
            "WhatsApp మైం ఆపకో లింక భేజతా హూం", language="te-IN"
        )
    assert result == "WhatsApp మైం ఆపకో లింక భేజతా హూం"  # unchanged, as before this task
    assert any(
        r.levelno == logging.WARNING and "WhatsApp" in str(r.__dict__.get("words"))
        for r in caplog.records
    )


def test_normalize_for_tts_empty_text_no_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = normalize_for_tts("", language="hi-IN")
    assert result == ""
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
