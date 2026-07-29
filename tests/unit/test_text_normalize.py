from __future__ import annotations

from src.pipeline.text_normalize import DEFAULT_PRONUNCIATIONS, apply_pronunciations


def test_rewrites_known_terms_to_devanagari() -> None:
    out = apply_pronunciations("WhatsApp par link bhejun? Casino bhi hai.")
    # Mispronounced English terms are gone, replaced by Devanagari.
    assert "WhatsApp" not in out and "Casino" not in out and "link" not in out
    assert DEFAULT_PRONUNCIATIONS["WhatsApp"] in out
    assert DEFAULT_PRONUNCIATIONS["Casino"] in out
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
    out = apply_pronunciations("Khelo Aviator aur ZyxBrand", extra={"ZyxBrand": "ज़िक्सब्रांड"})
    assert "ज़िक्सब्रांड" in out
    assert DEFAULT_PRONUNCIATIONS["Aviator"] in out


def test_rewrites_newly_added_sports_and_gaming_terms() -> None:
    # These specific words were observed un-transliterated in real Stage test
    # transcripts (still in Latin script when reaching TTS) before this
    # expansion — this test guards against a future accidental removal.
    out = apply_pronunciations(
        "Cricket aur Football dono hai, aap join karke explore kar sakte hain, "
        "koi risk nahi. Matka bhi khel sakte hain."
    )
    assert "Cricket" not in out and DEFAULT_PRONUNCIATIONS["Cricket"] in out
    assert "Football" not in out and DEFAULT_PRONUNCIATIONS["Football"] in out
    assert "join" not in out and DEFAULT_PRONUNCIATIONS["join"] in out
    assert "explore" not in out and DEFAULT_PRONUNCIATIONS["explore"] in out
    assert "risk" not in out and DEFAULT_PRONUNCIATIONS["risk"] in out
    assert "Matka" not in out and DEFAULT_PRONUNCIATIONS["Matka"] in out
    # Surrounding Hindi/Hinglish text is preserved untouched.
    assert "aur" in out and "dono hai" in out and "koi" in out and "nahi" in out
