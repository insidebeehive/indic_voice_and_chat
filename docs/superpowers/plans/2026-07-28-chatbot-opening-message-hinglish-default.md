# ChatBot Opening-Message Hinglish Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a chat session's very first user message carries no clear language signal (e.g. a bare word like "games"), default the reply to Roman-script Hinglish instead of the configured default language (often Devanagari Hindi) — since there's no established conversation language yet to protect, and Hinglish is readable by both English-only and Hindi/Hinglish speakers. Scoped strictly to the opening message; every later turn's behavior is unchanged.

**Architecture:** One new conditional branch in `ChatBotAgent._compose` (`src/agents/chatbot.py`), inserted between the existing language-signal branches and the final "no signal, follow the conversation" fallback. Detects "opening message" by checking whether any prior turn in `self.session.turns` has `role == "user"` — `_compose` runs before the current message is appended to `session.turns`, so this check is reliable. No changes to `_detect_script`, `_latin_language_hint`, or `src/dialogue/prompts.py`.

**Tech Stack:** Python 3, pytest, pytest-asyncio.

## Global Constraints

- Branch is `stage` — re-verify with `git rev-parse --abbrev-ref HEAD` immediately before committing (this session had an incident where a commit landed on `main` because the working tree had silently drifted off `stage`). Do not create a new branch.
- Run `.venv/bin/python -m pytest tests/unit -q` after the task. Baseline immediately before this task (freshly measured): 2 failed (both known pre-existing, unrelated — `test_chat_routes.py::test_claim_session_and_agent_ws`, `test_prompts.py::test_chatbot_prompt_has_scope_guardrails`), 1164 passed, 1 skipped, 0 errors. Do not touch either of those two known failures. After this task: same 2 known failures, passed count up by however many new tests are added, zero new failures.
- No Alembic migration — in-memory prompt-composition logic only.
- Do NOT touch `src/dialogue/prompts.py`, `_detect_script`, or `_latin_language_hint` — every change lives in `ChatBotAgent._compose` and its test file.
- This code area has caused three real regressions before (documented in the code's own comment block) — the fix must be scoped so it can NEVER fire once a user turn already exists in the session, to avoid reopening regression #3 (Hinglish momentum overriding an established English conversation).

---

### Task 1: Default the opening message's no-signal case to Hinglish

**Files:**
- Modify: `src/agents/chatbot.py:405-433` (`ChatBotAgent._compose`)
- Test: `tests/unit/test_chatbot_agent.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the only task). Reads `self.session.turns` (a `list[LLMMessage]`, already an existing field on `AgentSession` — no new attribute needed) and the existing `_detect_script`/`_latin_language_hint` functions, both unchanged.
- Produces: no new public interface — this changes `_compose`'s internal `extra` directive list construction only. Nothing outside this method depends on the new branch.

- [ ] **Step 1: Confirm you're on the right branch**

```bash
git rev-parse --abbrev-ref HEAD
```
Expected: `stage`. If it prints anything else, stop and report back — do not proceed or switch branches yourself.

- [ ] **Step 2: Write the failing tests**

In `tests/unit/test_chatbot_agent.py`, replace the existing `test_short_ack_gets_no_language_directive` test (currently):

```python
@pytest.mark.asyncio
async def test_short_ack_gets_no_language_directive(retriever) -> None:
    # A bare "ok" carries no language signal — forcing English would flip a
    # Hinglish conversation mid-stream. No directive: history governs.
    llm = FakeLLM({
        "response_text": "Theek hai!",
        "language": "hi",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("ok")
    system_prompt = llm.calls[0][0].content
    assert "MUST be in" not in system_prompt
    assert "romanized Hindi (Hinglish)" not in system_prompt
```

with these two tests (the first is the corrected mid-conversation version — it now seeds a real prior turn via an actual `handle_message` call before sending "ok", so it genuinely exercises the mid-conversation path; the second is new coverage for the opening-message fix):

```python
@pytest.mark.asyncio
async def test_short_ack_gets_no_language_directive(retriever) -> None:
    # A bare "ok" mid-conversation carries no language signal — forcing a
    # language would flip an established conversation mid-stream. No
    # directive: history governs. Seed a real prior turn first (via an
    # actual handle_message call, the same way session.turns gets populated
    # in production) so this genuinely exercises the MID-conversation path,
    # not the opening-message one (which now gets a Hinglish-opener
    # directive — see test_opening_message_with_no_signal_gets_hinglish_directive).
    llm = FakeLLM({
        "response_text": "Your balance is 100.",
        "language": "en",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("tell me about this site")  # establishes English
    await agent.handle_message("ok")
    system_prompt = llm.calls[-1][0].content
    assert "MUST be in" not in system_prompt
    assert "romanized Hindi (Hinglish)" not in system_prompt
    assert "very first message" not in system_prompt


@pytest.mark.asyncio
async def test_opening_message_with_no_signal_gets_hinglish_directive(retriever) -> None:
    # A bare, ambiguous single word ("games") as the FIRST message of a fresh
    # session has no established conversation language to fall back on, and
    # the configured default (often Devanagari Hindi) risks alienating an
    # English-only user on their very first message. Default to Roman
    # Hinglish instead — readable by both English and Hindi/Hinglish
    # speakers — rather than the configured default language.
    llm = FakeLLM({
        "response_text": "Yaha kai games available hain!",
        "language": "hi",
        "sources_used": [],
        "confidence": "high",
        "action": "none",
    })
    agent = _make_agent(llm, retriever)
    await agent.handle_message("games")
    system_prompt = llm.calls[0][0].content
    assert "very first message" in system_prompt
    assert "Roman-script Hinglish" in system_prompt
    assert "MUST be in" not in system_prompt  # not the named-language branch
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_chatbot_agent.py -v -k "short_ack or opening_message_with_no_signal"`
Expected: `test_short_ack_gets_no_language_directive` PASSES already (unchanged behavior for the mid-conversation case — the fix hasn't been added yet, so `extra` stays `None` for both calls, same as today). `test_opening_message_with_no_signal_gets_hinglish_directive` FAILS — `assert "very first message" in system_prompt` fails because the current code has no such branch yet.

- [ ] **Step 4: Implement the fix**

In `src/agents/chatbot.py`, find `_compose` (currently lines 405-433):

```python
    def _compose(
        self, rag_text: str, user_msg: LLMMessage, query_text: str = "",
    ) -> list[LLMMessage]:
        # Per-turn language directive. History of this logic (three real bugs):
        # 1. All-Latin text was labeled "English" → romanized Hindi got forced
        #    into English replies.
        # 2. Then Latin text got NO signal → the "Default language: hi"
        #    fallback answered plain English in Devanagari.
        # 3. Then an advisory "pick English or Hinglish yourself" directive →
        #    Hinglish history momentum kept answering plain English in
        #    Hinglish. Hence the deterministic marker-based classification:
        #    the directive must NAME the language, firmly, each turn.
        lang = _detect_script(query_text)
        if lang is None:
            lang = _latin_language_hint(query_text)
        if lang == "Hinglish":
            extra = [
                "The user's current message is romanized Hindi (Hinglish). Reply in Roman-"
                "script Hinglish — NEVER Devanagari, regardless of the conversation's "
                "default language or the language of earlier turns."
            ]
        elif lang:
            extra = [
                f"The user's current message is in {lang}. Your response_text MUST be in "
                f"{lang} — regardless of the conversation's default language or the "
                "language of earlier turns."
            ]
        else:
            extra = None  # no signal (empty/short/ambiguous) — follow the conversation
```

Replace the `if lang == "Hinglish": ... elif lang: ... else: extra = None` block with:

```python
        if lang == "Hinglish":
            extra = [
                "The user's current message is romanized Hindi (Hinglish). Reply in Roman-"
                "script Hinglish — NEVER Devanagari, regardless of the conversation's "
                "default language or the language of earlier turns."
            ]
        elif lang:
            extra = [
                f"The user's current message is in {lang}. Your response_text MUST be in "
                f"{lang} — regardless of the conversation's default language or the "
                "language of earlier turns."
            ]
        elif not any(m.role == "user" for m in self.session.turns):
            # Opening message, no language signal at all (e.g. a bare "games").
            # There's no established conversation language yet to fall back on,
            # and the configured default (often Devanagari Hindi) risks
            # alienating an English-only user on their very first message.
            # Roman Hinglish is readable by both English and Hindi/Hinglish
            # speakers, so it's the safer opener; a later message with a real
            # signal switches language deterministically from there, same as
            # any other turn. Scoped to the opening message ONLY — a
            # mid-conversation short ack ("ok") must keep following the
            # established conversation's language (see bug #3 above), so this
            # branch must never fire once a user turn already exists.
            extra = [
                "This is the very first message of the conversation and it carries no "
                "clear language signal (e.g. a bare word like 'games'). Reply in "
                "Roman-script Hinglish for this opening turn — readable by English and "
                "Hindi/Hinglish speakers alike — rather than the configured default "
                "language."
            ]
        else:
            extra = None  # no signal mid-conversation — follow the conversation
```

Do not change anything else in `_compose` — the rest of the method (the `build_chatbot_system_prompt(...)` call and everything after it) is untouched.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_chatbot_agent.py -v`
Expected: all tests in this file pass, including both `test_short_ack_gets_no_language_directive` and `test_opening_message_with_no_signal_gets_hinglish_directive`.

- [ ] **Step 6: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 2 failed (the same two known pre-existing, unrelated failures), 1 skipped, 0 errors, passed count up by 1 from baseline (one net new test — `test_short_ack_gets_no_language_directive` was rewritten, not added, but `test_opening_message_with_no_signal_gets_hinglish_directive` is new).

- [ ] **Step 7: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # must print "stage" — stop if it doesn't
git add src/agents/chatbot.py tests/unit/test_chatbot_agent.py
git status --short
```

Confirm the status output shows exactly these 2 modified files, no unrelated changes. Then commit:

```bash
git commit -m "$(cat <<'EOF'
fix(chatbot): default an ambiguous opening message to Hinglish

A single ambiguous word as the FIRST message of a fresh chat (e.g. "games")
fell through to the configured default language (often Devanagari Hindi),
which can alienate an English-only user on their very first message. There
is no established conversation language yet to protect at that point, so
default the opening turn to Roman-script Hinglish instead — readable by
both English and Hindi/Hinglish speakers. Scoped strictly to the opening
message: this code area has caused three real regressions before (see the
comment block in _compose), and a mid-conversation short ack ("ok") must
keep following the established conversation's language unchanged.
EOF
)"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — 2 failed (both known pre-existing), 0 errors, passed count up by 1 from baseline.
- `.venv/bin/python -m pytest tests/unit/test_chatbot_agent.py -v` — all pass.
- `git diff --stat` (before committing) touches only `src/agents/chatbot.py` and `tests/unit/test_chatbot_agent.py` — nothing under `src/dialogue/prompts.py`.
- Manually re-read the diff's new branch placement: it must sit strictly between `elif lang:` and the final `else:`, never able to fire when `lang` is truthy, and never able to fire when `self.session.turns` already contains a `role == "user"` entry.
