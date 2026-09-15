"""Tests for the voicebot's pre-TTS unverified-currency-figure guard.

Two layers:
- `_voice_sentence_guard` directly — the pure policy function (reuses
  context_builder's currency/number helpers; must never re-implement them).
- `VoiceBotAgent._make_sentence_guard` / wiring into handle_turn_text — that
  the agent actually threads a working guard into the engine call, grounded
  in the call's KB context + the caller's own transcribed speech.
"""

from __future__ import annotations

import logging

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent, _voice_sentence_guard
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import LLMMessage
from src.pipeline.engine import TurnMetrics, TurnResult
from src.rag.context_builder import GuardConfig


# --- _voice_sentence_guard (pure policy function) -------------------------


def test_sentence_with_no_currency_figure_is_never_touched():
    assert _voice_sentence_guard(
        "Your KYC is already complete.",
        grounded_text="",
        customer_text="",
        is_hindi=False,
    ) is None


def test_grounded_figure_is_spoken_unchanged():
    replacement = _voice_sentence_guard(
        "Your withdrawal of ₹8,100 has been processed.",
        grounded_text="tool_result: amount=8100",
        customer_text="",
        is_hindi=False,
    )
    assert replacement is None


def test_ungrounded_figure_trips_english_fallback():
    cfg = GuardConfig()
    replacement = _voice_sentence_guard(
        "Your balance is ₹19,600.",
        grounded_text="(no relevant sources found)",
        customer_text="",
        is_hindi=False,
    )
    assert replacement == cfg.unverified_data_fallback_en


def test_ungrounded_figure_trips_hindi_fallback():
    cfg = GuardConfig()
    replacement = _voice_sentence_guard(
        "Aapka balance ₹19,600 hai.",
        grounded_text="",
        customer_text="",
        is_hindi=True,
    )
    assert replacement == cfg.unverified_data_fallback_hi


def test_ungrounded_figure_names_the_callers_disputed_figure():
    """When the caller themselves stated a figure, the _with_figure variant
    names it — offering to connect them, per the escalation-rule phrasing
    (an offer pending confirmation, not a declared handoff)."""
    cfg = GuardConfig()
    replacement = _voice_sentence_guard(
        "I can confirm your refund of ₹19,600 is complete.",
        grounded_text="",
        customer_text="I was told I'd get ₹19,600 back",
        is_hindi=False,
    )
    assert replacement == cfg.unverified_data_fallback_en_with_figure.format(figure="₹19,600")


def test_callers_own_speech_grounds_the_figure():
    """A figure the caller stated themselves ('I deposited 500 rupees') must
    not trip the guard just because it isn't in the KB dump — the caller's
    own transcribed speech is grounding material too."""
    replacement = _voice_sentence_guard(
        "So you deposited Rs. 500 and it hasn't reflected yet, correct?",
        grounded_text="I deposited 500 rupees yesterday",
        customer_text="I deposited 500 rupees yesterday",
        is_hindi=False,
    )
    assert replacement is None


def test_reuses_shared_numeric_normalization_helpers():
    """₹4,250.75 in the reply must match a grounded '4250.75' (comma +
    decimal normalization) — exercises the exact helpers
    apply_unverified_data_guard uses, not a second implementation."""
    replacement = _voice_sentence_guard(
        "Your balance is ₹4,250.75.",
        grounded_text="tool_result: {\"amount\": 4250.75}",
        customer_text="",
        is_hindi=False,
    )
    assert replacement is None


# --- Voice-specific "number THEN currency word" detector -------------------
#
# _CURRENCY_FIGURE_PATTERN (chat's) requires the currency token BEFORE the
# number ("Rs 8,100"). A voicebot states amounts in the OPPOSITE, speakable
# order — number first, currency word after — because
# src/pipeline/text_normalize.py's normalize_currency actively rewrites
# "₹8,100" into "8100 रुपये" before synthesis. These tests cover the gap:
# every form that shape produces or that the LLM/STT might emit directly.


@pytest.mark.parametrize(
    "sentence",
    [
        "Aapka balance 8,100 rupees hai.",
        "Aapka balance 8100 rupaye hai.",
        "Aapka balance 8100 rupaya hai.",
        "Aapka balance 8100 rupee hai.",
        "Aapka balance 8100 rs hai.",
        "Aapka balance 8100 INR hai.",
        "आपका बैलेंस 8,100 रुपये है।",
        "आपका बैलेंस 8100 रुपया है।",
        "आपका बैलेंस 8100 रुपए है।",
        # Devanagari digits (STT/LLM may emit either digit set for hi/mr).
        "आपका बैलेंस ८,१०० रुपये है।",
        "Aapka balance ८१०० rupees hai.",
    ],
)
def test_number_then_currency_word_trips_when_ungrounded(sentence):
    assert _voice_sentence_guard(
        sentence, grounded_text="", customer_text="", is_hindi=True,
    ) is not None


@pytest.mark.parametrize(
    "sentence,grounded_text",
    [
        # Same script on both sides.
        ("Aapka balance 8,100 rupees hai.", "tool_result: amount=8100"),
        ("Aapka balance 8100 rupaye hai.", "the caller said 8100 earlier"),
        ("आपका बैलेंस 8100 रुपये है।", "tool_result: amount=8100"),
        # Cross-script: caller/tool speaks one digit set, reply the other.
        ("Aapka balance ५०० rupees hai.", "caller said 500 rupees"),
        ("Your balance is 500 rupees.", "caller said ५०० rupees"),
    ],
)
def test_number_then_currency_word_passes_when_grounded(sentence, grounded_text):
    assert _voice_sentence_guard(
        sentence, grounded_text=grounded_text, customer_text=grounded_text, is_hindi=True,
    ) is None


@pytest.mark.parametrize(
    "sentence",
    [
        "Your balance is Rs 19,600 right now.",
        "Your balance is ₹19,600 right now.",
        "Your balance is INR 19600 right now.",
    ],
)
def test_prefix_forms_still_trip_when_ungrounded(sentence):
    """No regression on the forms the guard already caught before this
    voice-specific detector was added."""
    assert _voice_sentence_guard(
        sentence, grounded_text="", customer_text="", is_hindi=False,
    ) is not None


@pytest.mark.parametrize(
    "sentence",
    [
        "within 24 hours",
        "press 1 for support",
        "your OTP is 6 digits",
        "call back after 50 hrs",
        "aapka OTP 4 digits ka hai",
    ],
)
def test_bare_numbers_with_no_currency_word_never_trip(sentence):
    """Deliberate design choice (see _VOICE_CURRENCY_WORD_PATTERN's comment):
    a bare number with no currency word is NOT treated as a currency figure
    — the currency word is the only reliable signal distinguishing a stated
    amount from a date/duration/menu-option/OTP-length, and dropping it
    would trade a narrow gap for a broad false-positive surface."""
    assert _voice_sentence_guard(
        sentence, grounded_text="", customer_text="", is_hindi=False,
    ) is None


def test_spelled_out_amount_is_a_documented_gap_not_caught():
    """Current, known behaviour: spelled-out numbers ("eight thousand one
    hundred rupees") are not parsed by either currency pattern, same
    documented limitation as apply_unverified_data_guard's Hinglish-
    multiplier gap. This test pins the behaviour so the limitation stays
    visible rather than silently assumed fixed."""
    assert _voice_sentence_guard(
        "Aapka balance eight thousand one hundred rupees hai.",
        grounded_text="",
        customer_text="",
        is_hindi=True,
    ) is None


# --- VoiceBotAgent wiring ---------------------------------------------------


def _agent(engine, kb_context: str = "") -> VoiceBotAgent:
    return VoiceBotAgent(
        session=AgentSession(session_id="sess-1", campaign_id="camp-1", lead_data={}),
        state_machine=AgentStateMachine(),
        slot_schema=SlotSchema(),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=engine,
        store=None,
        kb_context=kb_context,
    )


class _CaptureEngine:
    """Records the sentence_guard callback handle_turn_text hands it, and
    returns a canned TurnResult without actually calling it (the engine's
    OWN use of the callback is covered by test_engine_sentence_guard.py)."""

    def __init__(self, agent_text: str) -> None:
        self._agent_text = agent_text
        self.captured_guard = None

    async def run_turn_text(self, user_text, history, audio_sink, cancel_event=None, **kw):
        self.captured_guard = kw.get("sentence_guard")
        return TurnResult(
            user_text=user_text, user_language="hi", user_confidence=1.0,
            agent_text=self._agent_text, audio_bytes_sent=1, metrics=TurnMetrics(),
        )


@pytest.mark.asyncio
async def test_handle_turn_text_passes_a_working_sentence_guard():
    engine = _CaptureEngine('{"response_text": "ok", "action": "continue"}')
    agent = _agent(engine, kb_context="[06-casino-games.md]\nMinimum deposit is ₹100.")
    await agent.start()

    async def sink(a):
        pass

    await agent.handle_turn_text("hello", sink)
    guard = engine.captured_guard
    assert guard is not None

    # Grounded in the KB dump injected at call start -> unchanged.
    assert guard("Minimum deposit is ₹100.") is None
    # Not grounded anywhere -> replaced.
    assert guard("You've won ₹50,000 in the jackpot.") is not None


@pytest.mark.asyncio
async def test_sentence_guard_grounds_current_turns_transcript_on_text_path():
    """handle_turn_text already has this turn's transcript before calling the
    engine, so the guard it builds must ground against it too, not just
    prior turns."""
    engine = _CaptureEngine('{"response_text": "ok", "action": "continue"}')
    agent = _agent(engine, kb_context="")
    await agent.start()

    async def sink(a):
        pass

    await agent.handle_turn_text("I deposited 500 rupees yesterday", sink)
    guard = engine.captured_guard
    assert guard("So that Rs. 500 deposit hasn't shown up yet?") is None


@pytest.mark.asyncio
async def test_sentence_guard_grounds_prior_turns_speech():
    engine = _CaptureEngine('{"response_text": "ok", "action": "continue"}')
    agent = _agent(engine, kb_context="")
    await agent.start()
    agent.session.turns.append(LLMMessage(role="user", content="I deposited 500 rupees"))

    async def sink(a):
        pass

    await agent.handle_turn_text("did it go through", sink)
    guard = engine.captured_guard
    assert guard("Your Rs. 500 deposit is confirmed.") is None


@pytest.mark.asyncio
async def test_guard_trip_is_logged_at_warning_with_session_id(caplog):
    engine = _CaptureEngine('{"response_text": "ok", "action": "continue"}')
    agent = _agent(engine, kb_context="")
    await agent.start()

    async def sink(a):
        pass

    with caplog.at_level(logging.WARNING, logger="src.agents.voicebot"):
        await agent.handle_turn_text("hello", sink)
        guard = engine.captured_guard
        guard("You've won ₹50,000 in the jackpot.")

    records = [r for r in caplog.records if "pre-TTS guard tripped" in r.message]
    assert records
    assert records[0].session_id == "sess-1"
    assert "₹50,000" in records[0].figures
