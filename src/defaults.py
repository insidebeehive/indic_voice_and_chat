"""Shared default constants with no ``src.bootstrap`` / ``src.api`` dependency.

This module exists for the same reason as ``src.exceptions``: to give values
that both ``src.bootstrap`` and ``src.api.*`` modules need a home that neither
of those packages owns, so nobody is tempted to reach for a top-level
``from src.bootstrap import ...`` (which drags in ``src.bootstrap``'s own
``src.api.telephony_*`` imports and, transitively, ``src.api/__init__.py`` --
the exact cycle ``src.exceptions`` was carved out to avoid; see that module's
docstring). Anything defined here may depend on leaf modules like
``src.dialogue.prompts``, but must never import from ``src.bootstrap`` or
``src.api`` -- that would reintroduce the hazard this module exists to close.
"""

from __future__ import annotations

from src.dialogue.prompts import VoiceBotScript

DEFAULT_DEMO_SCRIPT = VoiceBotScript(
    agent_name="Priya",
    agent_role="Customer Engagement Specialist",
    company_name="Vox Demo",
    language_default="hi-IN",
    opening=(
        "Namaste! Main Priya bol rahi hoon Vox Demo se. "
        "Aapse ek choti si baat karni thi — kya aapke paas do minute hain?"
    ),
    talking_points=[
        "Vox Demo ek end-to-end AI voice agent platform hai.",
    ],
    qualifying_questions=["Aap abhi kya use kar rahe hain customer calls ke liye?"],
    objection_responses={
        "is_ai": "Haan, main ek AI assistant hoon — Vox Demo ki taraf se.",
        "busy": "Bilkul, samajh sakti hoon. Kya main baad mein call karun?",
    },
    closing={
        "positive": "Bahut accha! Dhanyavaad aapke time ke liye.",
        "negative": "Koi baat nahi. Aapka din shubh ho!",
    },
)
