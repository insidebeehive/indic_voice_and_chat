# TTS Un-transliterated Word Gap-Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log a warning whenever a genuine English/Latin-script word survives `normalize_for_tts` and would reach TTS un-transliterated, so pronunciation-dictionary gaps show up automatically in logs instead of requiring a human to notice a mispronounced word by ear.

**Architecture:** A new module-level logger + a script-dominance-aware scanning helper called from every return path of `normalize_for_tts`. Applies across all 11 non-English languages Sarvam/IndicF5 support, not just the 2 currently in `DEVANAGARI_LANGS` (hi, mr) — for the other 9, it applies the same logic once the text is confirmed to actually be written in that language's native script. Pure observability; never changes what reaches TTS.

**Revision note (read this before implementing):** an earlier version of this plan used a naive heuristic — flag any remaining Latin-script word, with no further check — and dispatching it to an implementer surfaced a real false positive: for Devanagari-language text, `apply_pronunciations` is routinely run against **romanized Hinglish** (e.g. "WhatsApp par link bhejun aur Cricket khelo" — where "par"/"bhejun"/"aur"/"khelo" are ordinary Hindi words legitimately left in Latin script; only WhatsApp/link/Cricket are English loanwords meant to be rewritten). The naive heuristic couldn't tell "a genuine un-transliterated loanword" from "ordinary Hindi vocabulary correctly staying in Latin script" and flagged both, which is wrong — that isn't a pronunciation gap, it's expected input shape. This revision fixes that by only treating a residual Latin word as a genuine gap when the surrounding text is **script-dominant** in the active language (≥60% of alphabetic characters are in that language's native Unicode script range, mirroring the exact threshold already used by `_detect_script` in `src/agents/chatbot.py`) — i.e. "this looks like real native-script text with isolated Latin words stuck in it," not "this is romanized Hinglish/English written wholesale in Latin script" (a different, out-of-scope problem — see the note in `src/agents/chatbot.py`'s `_detect_script` docstring for why that distinction matters generally).

**Tech Stack:** Python 3, stdlib `re`/`logging`, pytest (`caplog`).

## Global Constraints

- Branch is `stage` — re-verify with `git rev-parse --abbrev-ref HEAD` immediately before committing (this session had repeated incidents where a commit landed on `main` because the working tree had silently drifted off `stage`). Do not create a new branch.
- Run `.venv/bin/python -m pytest tests/unit -q` after the task. Baseline immediately before this task: 2 failed (both known pre-existing, unrelated — `test_chat_routes.py::test_claim_session_and_agent_ws`, `test_prompts.py::test_chatbot_prompt_has_scope_guardrails`), 1167 passed, 1 skipped, 0 errors. Do not touch either of those two known failures; this task must add exactly 4 net new tests and zero new failures.
- No Alembic migration. **No new dependencies** — this is pure `re` + stdlib `logging`. Do NOT install or reference `indic-transliteration`, `pronouncing`, or `cmudict` — those are for a deferred, separate general-fallback engine, entirely out of scope here.
- Do not change `apply_pronunciations`, `normalize_currency`, `DEFAULT_PRONUNCIATIONS`, or `DEVANAGARI_LANGS` themselves — only add the new logger/helper(s) and change `normalize_for_tts`'s body.
- If, while implementing, a test fails in a way this plan did not anticipate and the fix isn't a mechanical one obviously implied by this plan's own text, STOP and report back rather than improvising a design change — exactly as the previous attempt at this same plan correctly did.

---

### Task 1: Log un-transliterated words on every `normalize_for_tts` return path

**Files:**
- Modify: `src/pipeline/text_normalize.py`
- Test: `tests/unit/test_text_normalize.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the only task).
- Produces: `normalize_for_tts`'s signature and return behavior are completely unchanged (same inputs produce the same output text) — this task only adds a side-effect (logging), so `SarvamTTSAdapter`/`IndicF5TTSAdapter` (both already calling `normalize_for_tts`) need no changes.

- [ ] **Step 1: Confirm you're on the right branch**

```bash
git rev-parse --abbrev-ref HEAD
```
Expected: `stage`. If it prints anything else, stop and report back — do not proceed or switch branches yourself.

- [ ] **Step 2: Write the failing tests**

In `tests/unit/test_text_normalize.py`, add `import logging` at the top of the file (alongside the existing `from __future__ import annotations`), and change the existing import line:
```python
from src.pipeline.text_normalize import DEFAULT_PRONUNCIATIONS, apply_pronunciations
```
to:
```python
from src.pipeline.text_normalize import (
    DEFAULT_PRONUNCIATIONS,
    apply_pronunciations,
    normalize_for_tts,
)
```

Then add these 5 tests at the end of the file:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_text_normalize.py -v -k "normalize_for_tts"`
Expected: `test_normalize_for_tts_warns_on_devanagari_language_gap` and `test_normalize_for_tts_warns_for_script_dominant_unsupported_language` FAIL (no logging exists yet, so `any(...)` is `False`). The other 3 tests (`test_normalize_for_tts_no_warning_when_fully_covered`, `test_normalize_for_tts_no_warning_for_romanized_hinglish`, `test_normalize_for_tts_empty_text_no_warning`) already trivially PASS at this point, since no warnings exist yet at all — that's expected, not a problem; confirm this split matches what you observe.

- [ ] **Step 4: Implement the gap-logging**

In `src/pipeline/text_normalize.py`, add `import logging` right after `import re`:
```python
from __future__ import annotations

import logging
import re
```

Add a module-level logger right after the imports (before `DEFAULT_PRONUNCIATIONS`):
```python
log = logging.getLogger(__name__)
```

Find `DEVANAGARI_LANGS` (currently):
```python
# Indian languages written in Devanagari. The pronunciation + currency rewrites
# above are Devanagari, so they're correct for these and wrong (wrong script) for
# others (Telugu, Malayalam, Tamil, …).
DEVANAGARI_LANGS = frozenset({"hi", "mr", "ne", "sa", "kok", "mai", "bho", "doi"})
```

Directly after it, add the new helpers:
```python
# Any Latin-script word run. Used only for gap visibility (see
# _warn_if_untransliterated below) -- never to decide what to rewrite.
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z']*\b")

# Unicode block start/end for each script, keyed by base language code. Used
# only to judge whether TTS-bound text is actually written in the active
# language's native script before treating a residual Latin word as a
# genuine gap -- romanized Hinglish/English is expected to stay in Latin
# script for languages that don't have their own normalization yet, and must
# not be flagged as a mispronunciation risk (see the "Revision note" above).
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "hi": (0x0900, 0x097F),  # Devanagari -- also covers mr, ne, sa, kok, mai, bho, doi
    "bn": (0x0980, 0x09FF),  # Bengali -- also used as a stand-in for Assamese ("as")
    "as": (0x0980, 0x09FF),
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "od": (0x0B00, 0x0B7F),  # Odia
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
}


def _is_script_dominant(text: str, language: str) -> bool:
    """True when at least 60% of the text's alphabetic characters are in the
    active language's native script -- i.e. this looks like real native-script
    text with isolated Latin words in it, not romanized Hinglish/English
    written wholesale in Latin script (a different, out-of-scope problem).
    """
    lang_key = "hi" if language in DEVANAGARI_LANGS else language
    script_range = _SCRIPT_RANGES.get(lang_key)
    if script_range is None:
        return False
    lo, hi = script_range
    native_count = latin_count = 0
    for ch in text:
        if ch.isascii() and ch.isalpha():
            latin_count += 1
        elif lo <= ord(ch) <= hi:
            native_count += 1
    total = native_count + latin_count
    return total > 0 and (native_count / total) >= 0.6


def _warn_if_untransliterated(text: str, language: str) -> None:
    """Log a warning for every Latin-script word still present in
    script-dominant TTS-bound text. For a DEVANAGARI_LANGS language this
    means a DEFAULT_PRONUNCIATIONS gap (a real word TTS will likely
    mispronounce); for any other language it means every word in the call,
    since no normalization runs for that language yet. Text that isn't
    script-dominant (e.g. romanized Hinglish) is skipped entirely -- that's
    expected input shape, not a gap. Purely observational -- never raises,
    never changes the text that reaches TTS.
    """
    if not _is_script_dominant(text, language):
        return
    words = _LATIN_WORD_RE.findall(text)
    if words:
        log.warning(
            "tts text has un-transliterated Latin-script word(s); TTS will "
            "likely mispronounce them",
            extra={"language": language, "words": words, "text_sample": text[:160]},
        )
```

Find the current `normalize_for_tts`:
```python
def normalize_for_tts(
    text: str, language: str | None = None, extra: dict[str, str] | None = None
) -> str:
    """Language-aware TTS text normalization.

    Applies the Devanagari currency + pronunciation rewrites only for
    Devanagari-script languages (Hindi, Marathi, …). For other scripts (Telugu,
    Malayalam) injecting Devanagari would render the wrong script, so the text is
    returned unchanged until per-language maps exist. An unknown/empty language
    keeps the legacy behaviour (apply — assumes Hindi).
    """
    if not text:
        return text
    base = (language or "").strip().lower().split("-")[0]
    if base and base not in DEVANAGARI_LANGS:
        return text
    return apply_pronunciations(normalize_currency(text), extra=extra)
```

Replace it with:
```python
def normalize_for_tts(
    text: str, language: str | None = None, extra: dict[str, str] | None = None
) -> str:
    """Language-aware TTS text normalization.

    Applies the Devanagari currency + pronunciation rewrites only for
    Devanagari-script languages (Hindi, Marathi, …). For other scripts (Telugu,
    Malayalam) injecting Devanagari would render the wrong script, so the text is
    returned unchanged until per-language maps exist. An unknown/empty language
    keeps the legacy behaviour (apply — assumes Hindi).

    Either way, any Latin-script word still present in genuinely
    script-dominant text this function returns is logged as a warning (see
    ``_warn_if_untransliterated``) — a DEFAULT_PRONUNCIATIONS gap for a
    Devanagari-script language, or simply "no normalization exists yet for
    this language" for any other one. Romanized Hinglish/English is not
    flagged, since that's expected input shape, not a gap. This makes
    coverage gaps visible in logs instead of requiring someone to notice a
    mispronounced word by ear.
    """
    if not text:
        return text
    base = (language or "").strip().lower().split("-")[0]
    if base and base not in DEVANAGARI_LANGS:
        _warn_if_untransliterated(text, base or "unknown")
        return text
    result = apply_pronunciations(normalize_currency(text), extra=extra)
    _warn_if_untransliterated(result, base or "hi")
    return result
```

Do not change `apply_pronunciations`, `normalize_currency`, `DEFAULT_PRONUNCIATIONS`, or `DEVANAGARI_LANGS`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_text_normalize.py -v`
Expected: all 10 tests in this file pass (the 5 pre-existing ones plus the 5 new ones).

- [ ] **Step 6: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 2 failed (the same two known pre-existing, unrelated failures), 1 skipped, 0 errors, passed count up by 5 from baseline (1172).

- [ ] **Step 7: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # must print "stage" — stop if it doesn't
git add src/pipeline/text_normalize.py tests/unit/test_text_normalize.py
git status --short
```

Confirm the status output shows exactly these 2 modified files, no unrelated changes. Then commit:

```bash
git commit -m "$(cat <<'EOF'
feat(tts): log un-transliterated words as a warning, all 11 languages

DEFAULT_PRONUNCIATIONS only ever gets fixed reactively, after someone
notices a mispronounced word by ear -- and it currently covers only 2 of
the 11 languages Sarvam/IndicF5 support (hi, mr); the other 9 (Bengali,
Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese) get
zero normalization with zero visibility into what's being missed.

normalize_for_tts now logs a warning for every Latin-script word still
present in script-dominant text -- a real DEFAULT_PRONUNCIATIONS gap for a
Devanagari language, or every word for any of the other 9 languages, since
nothing is normalized for them yet. Script-dominance is checked (>=60% of
alphabetic characters in the active language's native Unicode range)
specifically so romanized Hinglish text (ordinary Hindi words legitimately
left in Latin script, e.g. "par"/"aur"/"bhejun") isn't flagged as a false
gap -- an earlier version of this without that check produced exactly
that false positive. Pure observability: the text that reaches TTS is
completely unchanged, only a log line is added.

A general fallback transliteration engine (so unknown words get
auto-transliterated instead of just logged) is deliberately deferred --
a CMUdict/ARPAbet-based prototype was tested and found to produce
systematically wrong results for Indian-English loanword pronunciation
(American phonetics reduce vowels to schwa and lack the "tu"->"choo"
palatalization Indian English applies), so that approach needs more work
before it's worth shipping. This just gets us visibility in the meantime.
EOF
)"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — 2 failed (both known pre-existing), 0 errors, passed count up by 5 from baseline.
- `.venv/bin/python -m pytest tests/unit/test_text_normalize.py -v` — all 10 pass.
- `git diff --stat` (before committing) touches only `src/pipeline/text_normalize.py` and `tests/unit/test_text_normalize.py`.
- Manually confirm `apply_pronunciations`, `normalize_currency`, `DEFAULT_PRONUNCIATIONS`, and `DEVANAGARI_LANGS` are byte-for-byte unchanged in the diff.
- Manually confirm `test_normalize_for_tts_no_warning_for_romanized_hinglish` genuinely passes against the real `apply_pronunciations` (not just because the test itself has a bug) — this is the exact scenario that broke the previous version of this plan.
