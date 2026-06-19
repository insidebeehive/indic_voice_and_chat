"""Outcome analysis for a manual (human agent ↔ lead) softphone call.

A human softphone call has no live AI transcript, so we reach the **same**
structured outcome an AI call produces by:

  1. **Transcribing** the call recording — one batch ``ISTTProvider.transcribe``
     pass per channel (dual-channel recording keeps the agent and lead on
     separate tracks, so each transcript is cleanly role-attributable).
  2. **Building** a role-attributed ``list[LLMMessage]`` (agent channel →
     ``assistant``, lead channel → ``user``) — interleaved by word timestamps
     when the STT provider returns them, else one message per channel.
  3. Running the **same** ``analyze_call`` the AI path uses, then persisting via
     the same ``record_outcome`` — so ``GET /calls/{id}`` returns an identical
     ``CallAnalysis`` shape (outcome / summary / notes / callback) for human and
     AI calls alike.

This module owns only the transcribe→analyze→persist orchestration; fetching the
recording bytes (provider-specific) is the caller's job.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.analysis.call_outcome import analyze_call
from src.api.call_store import record_outcome
from src.interfaces.llm import ILLMProvider, LLMMessage
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.models.conversation import Conversation

log = logging.getLogger(__name__)

# A channel of the recording: (role, raw audio bytes). ``role`` is the LLMMessage
# role — "assistant" for the human agent's track, "user" for the lead's track.
RecordingChannel = tuple[str, bytes]

# word-timestamp dict keys vary by provider; accept the common spellings.
_START_KEYS = ("start", "start_time", "startTime", "start_offset")
_WORD_KEYS = ("punctuated_word", "word", "text", "value")


def _word_start(w: dict) -> Optional[float]:
    for k in _START_KEYS:
        v = w.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _word_text(w: dict) -> str:
    for k in _WORD_KEYS:
        v = w.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def build_transcript(channels: list[tuple[str, STTResult]]) -> list[LLMMessage]:
    """Build a role-attributed transcript from per-channel STT results.

    If every channel exposes per-word start times, the words are interleaved
    across channels and grouped into consecutive same-role utterances (so the
    conversational back-and-forth is preserved). Otherwise we fall back to one
    message per channel in the order given.
    """
    # Try timestamp interleaving — only if every non-empty channel has timed words.
    timed: list[tuple[float, str, str]] = []  # (start, role, word)
    can_interleave = True
    for role, res in channels:
        words = res.word_timestamps or []
        if not words and (res.text or "").strip():
            can_interleave = False
            break
        for w in words:
            start = _word_start(w)
            text = _word_text(w)
            if start is None or not text:
                can_interleave = False
                break
            timed.append((start, role, text))
        if not can_interleave:
            break

    if can_interleave and timed:
        timed.sort(key=lambda t: t[0])
        messages: list[LLMMessage] = []
        cur_role: Optional[str] = None
        cur_words: list[str] = []
        for _start, role, word in timed:
            if role != cur_role and cur_words:
                messages.append(LLMMessage(role=cur_role, content=" ".join(cur_words)))
                cur_words = []
            cur_role = role
            cur_words.append(word)
        if cur_words and cur_role is not None:
            messages.append(LLMMessage(role=cur_role, content=" ".join(cur_words)))
        return messages

    # Coarse fallback: one message per channel (whole-channel transcript).
    return [
        LLMMessage(role=role, content=(res.text or "").strip())
        for role, res in channels
        if (res.text or "").strip()
    ]


async def transcribe_channels(
    channels: list[RecordingChannel],
    *,
    stt: ISTTProvider,
    stt_config: STTConfig,
) -> list[tuple[str, STTResult]]:
    """Batch-transcribe each recording channel. A channel that fails to
    transcribe becomes an empty result (so one bad track can't lose the call)."""
    out: list[tuple[str, STTResult]] = []
    for role, audio in channels:
        try:
            res = await stt.transcribe(audio, stt_config)
        except Exception:  # noqa: BLE001 — one bad channel must not lose the call
            log.exception("manual-call channel transcription failed", extra={"role": role})
            res = STTResult(text="", confidence=0.0)
        out.append((role, res))
    return out


async def finalize_manual_call(
    session: AsyncSession,
    *,
    provider_call_sid: str,
    llm: ILLMProvider,
    tenant_timezone: str,
    now: datetime,
    channels: Optional[list[RecordingChannel]] = None,
    stt: Optional[ISTTProvider] = None,
    stt_config: Optional[STTConfig] = None,
    transcript: Optional[list[LLMMessage]] = None,
    telephony_status: Optional[str] = None,
    final_action: Optional[str] = None,
    duration_ms: Optional[int] = None,
    recording_url: Optional[str] = None,
) -> Optional[Conversation]:
    """Transcribe → analyze → persist a manual call's outcome.

    Either pass per-channel audio (``channels`` + ``stt``/``stt_config``, the
    dual-channel STT path) OR a pre-built ``transcript`` (e.g. from a multimodal
    LLM that transcribed the whole recording). Produces the identical
    ``CallAnalysis`` an AI call would (same ``analyze_call``) and writes it through
    the same ``record_outcome``. Returns the row, or ``None`` if no SID matches.
    """
    if transcript is None:
        channel_results = await transcribe_channels(
            channels or [], stt=stt, stt_config=stt_config or STTConfig())
        transcript = build_transcript(channel_results)
    analysis = await analyze_call(
        transcript=transcript,
        slots={},
        telephony_status=telephony_status,
        final_action=final_action,
        tenant_timezone=tenant_timezone,
        now=now,
        llm=llm,
    )
    # We deliberately do NOT echo recording_url back to the CRM — the call_id
    # (provider_call_sid) is already in the event, and the CRM can fetch the
    # recording itself if it needs it.
    return await record_outcome(
        session,
        provider_call_sid,
        status="ended",
        outcome=analysis.outcome.value,
        summary=analysis.summary,
        notes=analysis.notes or "",
        callback_at=analysis.callback_datetime,
        duration_ms=duration_ms,
    )
