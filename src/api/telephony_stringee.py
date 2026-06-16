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
    *, agent_user: str, to_number: str, event_url: str, record: bool = True,
) -> list[dict[str, Any]]:
    """Bridge a human agent's browser (Stringee app user) call to the lead (PSTN).

    Per Stringee's app-to-phone ``connect`` SCCO:
    - ``from`` is the agent's **app user** (``type: internal`` — the existing
      client call leg to bridge), NOT a phone number. The PSTN caller-ID shown to
      the lead is the Stringee project's configured outbound number.
    - ``to`` is the lead (``type: external``).
    - Recording is a **separate** ``record`` action placed BEFORE ``connect`` (a
      ``record`` *field* inside connect is invalid → Stringee logs "Unknown" /
      REQUEST_ANSWER_URL_ERROR), and ``peerToPeerCall`` must be ``false`` or the
      call can't be recorded.

    ``event_url`` receives call + recording events (incl. the recording URL).
    """
    actions: list[dict[str, Any]] = []
    if record:
        # Dual-channel wav (agent on ch1, lead on ch2) so the recording webhook
        # can split tracks for clean role attribution without transcoding.
        actions.append({
            "action": "record",
            "eventUrl": event_url,
            "format": "wav",
            "channel": "two",
        })
    actions.append({
        "action": "connect",
        "from": {"type": "internal", "number": agent_user, "alias": agent_user},
        "to": {"type": "external", "number": to_number, "alias": to_number},
        "eventUrl": event_url,
        "timeout": 45,
        "maxConnectTime": -1,
        "peerToPeerCall": False,
    })
    return actions
