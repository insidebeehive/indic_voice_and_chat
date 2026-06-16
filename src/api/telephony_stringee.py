"""Stringee Call Control Object (SCCO) builders for the IVR voicebot.

Pure functions that return SCCO JSON (a list of action dicts). Stringee
fetches/returns SCCO at call answer and after each recorded turn; see
docs/superpowers/specs/2026-06-09-stringee-ivr-design.md.
"""

from __future__ import annotations

from typing import Any

# Silence (ms) after the caller stops speaking before Stringee ends the
# recording and POSTs us the utterance. Tuned down from a typical 4000ms to
# keep per-turn latency tolerable (see spec, latency section).
SILENCE_TIMEOUT_MS = 1500


def _record(event_url: str) -> dict[str, Any]:
    return {
        "action": "recordMessage",
        "eventUrl": event_url,
        "format": "wav",
        "silenceTimeout": SILENCE_TIMEOUT_MS,
        "beepStart": False,
    }


def answer_scco(*, audio_url: str, event_url: str) -> list[dict[str, Any]]:
    """Opening turn: play the greeting (interruptible), then record the reply."""
    return [
        {"action": "play", "url": audio_url, "bargeIn": True},
        _record(event_url),
    ]


def reply_scco(*, audio_url: str, event_url: str) -> list[dict[str, Any]]:
    """A normal turn: play the agent's reply, then record the next utterance."""
    return [
        {"action": "play", "url": audio_url, "bargeIn": True},
        _record(event_url),
    ]


def reprompt_scco(*, text: str, event_url: str) -> list[dict[str, Any]]:
    """Empty/failed capture: speak a short re-prompt and record again."""
    return [
        {"action": "talk", "text": text, "bargeIn": True},
        _record(event_url),
    ]


def closing_scco(*, audio_url: str) -> list[dict[str, Any]]:
    """Terminal turn: play the closing line and hang up (no further record)."""
    return [
        {"action": "play", "url": audio_url},
        {"action": "hangup"},
    ]


def softphone_connect_scco(
    *, caller_id: str, to_number: str, event_url: str, record: bool = True,
) -> list[dict[str, Any]]:
    """Bridge a human agent's browser call to the lead (PSTN).

    Matches a **known-working** Stringee app-to-phone SCCO (captured from a live
    successful call → ``SCCO_PARSER_RESULT: OK``):
    - a ``record`` action BEFORE ``connect`` — ``format: mp3``,
      ``recordStereo: false``, ``record_type: 1`` (a ``record`` *field* inside
      connect, or wav/``channel`` fields, are rejected → "Unknown").
    - ``connect`` ``from`` is the **caller-ID number** (a Stringee-owned number,
      ``type: internal``), ``to`` is the lead (``type: external``),
      ``customData: "{}"``, ``peerToPeerCall: false``.

    ``event_url`` receives the recording event (incl. the recording URL).
    """
    actions: list[dict[str, Any]] = []
    if record:
        actions.append({
            "action": "record",
            "eventUrl": event_url,
            "format": "mp3",
            "recordStereo": False,
            "record_type": 1,
        })
    actions.append({
        "action": "connect",
        "from": {"number": caller_id, "alias": caller_id, "type": "internal"},
        "customData": "{}",
        "to": {"number": to_number, "alias": to_number, "type": "external"},
        "peerToPeerCall": False,
    })
    return actions
