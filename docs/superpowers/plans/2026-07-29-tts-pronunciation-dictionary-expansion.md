# TTS Pronunciation Dictionary Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `DEFAULT_PRONUNCIATIONS` in `src/pipeline/text_normalize.py` with English/brand/sports/gaming loanwords that real voicebot test transcripts on Stage showed reaching TTS un-transliterated (still in Latin script), causing mispronunciation on Sarvam and IndicF5 — the two TTS adapters that actually call this normalization.

**Architecture:** Pure data-table expansion. `apply_pronunciations`/`normalize_currency`/`normalize_for_tts` already work correctly and are already unit-tested — no logic changes anywhere. This task only adds dictionary entries and matching test coverage.

**Tech Stack:** Python 3, pytest.

## Global Constraints

- Branch is `stage` — re-verify with `git rev-parse --abbrev-ref HEAD` immediately before committing (this session had repeated incidents where a commit landed on `main` because the working tree had silently drifted off `stage`). Do not create a new branch.
- Run `.venv/bin/python -m pytest tests/unit -q` after the task. Baseline immediately before this task: 2 failed (both known pre-existing, unrelated — `test_chat_routes.py::test_claim_session_and_agent_ws`, `test_prompts.py::test_chatbot_prompt_has_scope_guardrails`), 1166 passed, 1 skipped, 0 errors. Do not touch either of those two known failures; this task must not add any new failures and should add exactly 1 net new test.
- No Alembic migration. No logic changes anywhere — only the `DEFAULT_PRONUNCIATIONS` dict literal and one new test.
- Do NOT add "XYZ" to the dictionary — it's a dev-console test placeholder company name, not real production content. A real campaign's own company name belongs in that campaign's `extra_pronunciations` override, not the global default table.

---

### Task 1: Expand the pronunciation dictionary with observed missing words

**Files:**
- Modify: `src/pipeline/text_normalize.py`
- Test: `tests/unit/test_text_normalize.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the only task). Uses the existing `apply_pronunciations`/`DEFAULT_PRONUNCIATIONS` (unchanged signatures).
- Produces: `DEFAULT_PRONUNCIATIONS` gains 24 new keys; no function signature changes, so `SarvamTTSAdapter`/`IndicF5TTSAdapter` (both already calling `normalize_for_tts`) need no changes.

- [ ] **Step 1: Confirm you're on the right branch**

```bash
git rev-parse --abbrev-ref HEAD
```
Expected: `stage`. If it prints anything else, stop and report back — do not proceed or switch branches yourself.

- [ ] **Step 2: Write the failing test**

In `tests/unit/test_text_normalize.py`, add this test after the existing `test_extra_overrides_merge_over_defaults` (at the end of the file):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_text_normalize.py::test_rewrites_newly_added_sports_and_gaming_terms -v`
Expected: FAIL — `DEFAULT_PRONUNCIATIONS["Cricket"]` raises `KeyError` (none of these words exist in the dict yet).

- [ ] **Step 3: Expand the dictionary**

In `src/pipeline/text_normalize.py`, find the current `DEFAULT_PRONUNCIATIONS` dict:

```python
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "WhatsApp": "व्हाट्सऐप",
    "Casino": "कसीनो",
    "Aviator": "एविएटर",
    "app": "ऐप",
    "link": "लिंक",
    "bonus": "बोनस",
    "cash": "कैश",
    "commission": "कमीशन",
    "registration": "रजिस्ट्रेशन",
    "deposit": "डिपॉज़िट",
    "withdrawal": "विड्रॉल",
    "instant": "इंस्टंट",
    "account": "अकाउंट",
    "support": "सपोर्ट",
    "update": "अपडेट",
    "trusted": "ट्रस्टेड",
    "safe": "सेफ",
    "official": "ऑफिशियल",
    "free": "फ्री",
    "automatic": "ऑटोमैटिक",
    "minimum": "मिनिमम",
}
```

Replace with (24 new entries added at the end, existing 21 entries unchanged):

```python
DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "WhatsApp": "व्हाट्सऐप",
    "Casino": "कसीनो",
    "Aviator": "एविएटर",
    "app": "ऐप",
    "link": "लिंक",
    "bonus": "बोनस",
    "cash": "कैश",
    "commission": "कमीशन",
    "registration": "रजिस्ट्रेशन",
    "deposit": "डिपॉज़िट",
    "withdrawal": "विड्रॉल",
    "instant": "इंस्टंट",
    "account": "अकाउंट",
    "support": "सपोर्ट",
    "update": "अपडेट",
    "trusted": "ट्रस्टेड",
    "safe": "सेफ",
    "official": "ऑफिशियल",
    "free": "फ्री",
    "automatic": "ऑटोमैटिक",
    "minimum": "मिनिमम",
    "join": "जॉइन",
    "betting": "बेटिंग",
    "Cricket": "क्रिकेट",
    "Football": "फुटबॉल",
    "Matka": "मटका",
    "download": "डाउनलोड",
    "platform": "प्लेटफॉर्म",
    "Sports": "स्पोर्ट्स",
    "Tennis": "टेनिस",
    "Basketball": "बास्केटबॉल",
    "live": "लाइव",
    "guide": "गाइड",
    "explore": "एक्सप्लोर",
    "opportunity": "अपॉर्चुनिटी",
    "risk": "रिस्क",
    "perfect": "परफेक्ट",
    "Sales": "सेल्स",
    "team": "टीम",
    "refer": "रेफर",
    "market": "मार्केट",
    "match": "मैच",
    "matches": "मैचेस",
    "international": "इंटरनेशनल",
    "fan": "फैन",
    "IPL": "आईपीएल",
}
```

Do not change the module docstring, `apply_pronunciations`, `normalize_currency`, `normalize_for_tts`, or `DEVANAGARI_LANGS` — only the dict literal.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_text_normalize.py -v`
Expected: all 5 tests in this file pass (the 4 pre-existing ones plus the new one).

- [ ] **Step 5: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 2 failed (the same two known pre-existing, unrelated failures), 1 skipped, 0 errors, passed count up by 1 from baseline (1167).

- [ ] **Step 6: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # must print "stage" — stop if it doesn't
git add src/pipeline/text_normalize.py tests/unit/test_text_normalize.py
git status --short
```

Confirm the status output shows exactly these 2 modified files, no unrelated changes. Then commit:

```bash
git commit -m "$(cat <<'EOF'
fix(tts): expand the pronunciation dictionary with observed missing words

Real Stage test transcripts (layered/cascade mode, Sarvam and IndicF5 TTS)
showed sports/gaming/generic English loanwords reaching TTS still in
Latin script -- cricket, football, join, explore, risk, matka, and
others -- none of which were in DEFAULT_PRONUNCIATIONS yet, so they got
mispronounced the same way WhatsApp/Casino/Aviator did before those were
added. Pure data-table expansion, no logic changes; only affects Sarvam
and IndicF5 (the two adapters that call normalize_for_tts). Deliberately
excludes the placeholder test company name "XYZ" -- a real campaign's own
brand name belongs in that campaign's extra_pronunciations override.
EOF
)"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — 2 failed (both known pre-existing), 0 errors, passed count up by 1 from baseline.
- `.venv/bin/python -m pytest tests/unit/test_text_normalize.py -v` — all 5 pass.
- `git diff --stat` (before committing) touches only `src/pipeline/text_normalize.py` and `tests/unit/test_text_normalize.py`.
- Manually confirm no logic in `apply_pronunciations`/`normalize_currency`/`normalize_for_tts` was touched — only the dict literal.

## Known follow-up (not part of this task)

Several real Stage transcripts showed entire turns rendered fully in Roman-script Hinglish (not just isolated English loanwords mixed into otherwise-correct Devanagari text) — e.g. "Bilkul, koi baat nahi! Par ek baat kehna chahta hoon...". A word-level pronunciation dictionary cannot fix this case, since it would require transliterating ordinary Hindi words (nahi, hai, kar, chahta, hoon, etc.), not just English/brand loanwords — this is a separate, more severe language-compliance issue than what this task addresses, and should be raised with the user as a distinct follow-up item once this task ships.
