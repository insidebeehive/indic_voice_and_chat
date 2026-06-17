"""Gemini Live (speech-to-speech) session, implementing IRealtimeSession.

Wraps ``google-genai``'s Live API (``client.aio.live.connect``) using the exact
shapes validated in ``spikes/gemini_live_spike.py``: 16kHz PCM in, 24kHz PCM out,
prebuilt voice + language_code, in/out transcription, and one function tool. The
``connect`` factory retries transient open failures (the preview models
intermittently close with 1008/1011).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from src.interfaces.realtime import (
    IRealtimeSession,
    RealtimeConfig,
    RealtimeEvent,
    RealtimeTool,
)

log = logging.getLogger(__name__)

_OPEN_RETRIES = 3
_OPEN_BACKOFF_S = 0.4


def _schema(types, d: dict[str, Any] | None):
    """Translate a JSON-Schema-ish dict into a google.genai types.Schema."""
    if not d:
        return None
    kwargs: dict[str, Any] = {}
    if "type" in d:
        kwargs["type"] = d["type"]
    if "enum" in d:
        kwargs["enum"] = d["enum"]
    if "properties" in d:
        kwargs["properties"] = {k: _schema(types, v) for k, v in d["properties"].items()}
    if "required" in d:
        kwargs["required"] = d["required"]
    if "items" in d:
        kwargs["items"] = _schema(types, d["items"])
    return types.Schema(**kwargs)


def _to_tool(types, tool: RealtimeTool):
    return types.Tool(function_declarations=[types.FunctionDeclaration(
        name=tool.name, description=tool.description, parameters=_schema(types, tool.parameters))])


def _build_activity_detection(types):
    """Caller-speech (VAD) detection tuned so a short opening utterance is caught
    quickly. With the server defaults, a brief first "hello" was slow to register
    — callers had to repeat it before the model engaged. HIGH start sensitivity
    detects speech onset readily; a little prefix padding keeps the first phoneme
    from being clipped. End-of-speech is left at the server default so natural
    mid-sentence pauses don't endpoint early (the cascade hit that with too-eager
    endpointing — see stt_streaming.endpointing in the tenant config)."""
    return types.AutomaticActivityDetection(
        start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
        prefix_padding_ms=200,
    )


def _build_speech_config(types, language_code, voice):
    """Speech config for the Live session. ``language_code`` is set ONLY when
    provided: native-audio Live models (e.g. gemini-3.1-flash-live) auto-detect
    and switch languages naturally and don't accept an explicit code — leave it
    unset (``pipeline.realtime.language_code: null``) and steer language via the
    system instruction. A code is still honored for half-cascade models."""
    speech = types.SpeechConfig()
    if language_code:
        speech.language_code = language_code
    if voice:
        speech.voice_config = types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
    return speech


class GeminiLiveSession(IRealtimeSession):
    def __init__(self, cm, session) -> None:
        self._cm = cm          # the live.connect async context manager
        self._session = session

    @classmethod
    async def connect(cls, config: RealtimeConfig, *, api_key: str) -> "GeminiLiveSession":
        import os

        # Fail clearly when NO key resolves: the tenant's realtime.api_key_env is
        # unset AND no platform GEMINI_API_KEY/GOOGLE_API_KEY env (the by-design
        # fallback). Otherwise the SDK fails mid-connect with an opaque error.
        if not api_key and not (
                os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError(
                "Gemini Live needs an API key: set the tenant's pipeline.realtime."
                "api_key_env or the platform GEMINI_API_KEY / GOOGLE_API_KEY")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        speech = _build_speech_config(types, config.language_code, config.voice)
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=config.system_instruction or None,
            speech_config=speech,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            tools=[_to_tool(types, t) for t in config.tools] or None,
            max_output_tokens=config.max_output_tokens,
            # Explicit realtime-input/VAD so multi-turn works: auto speech detection,
            # caller speech interrupts the agent (native barge-in), and a turn is
            # just the detected speech (not the surrounding silence).
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=_build_activity_detection(types),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
        )
        last_err: Exception | None = None
        for attempt in range(_OPEN_RETRIES):
            cm = client.aio.live.connect(model=config.model, config=live_config)
            try:
                session = await cm.__aenter__()
                return cls(cm, session)
            except Exception as e:  # noqa: BLE001 - transient open failures retry
                last_err = e
                log.warning("gemini live connect failed (attempt %d): %s", attempt + 1, e)
                await asyncio.sleep(_OPEN_BACKOFF_S * (attempt + 1))
        raise RuntimeError(f"gemini live connect failed after {_OPEN_RETRIES} attempts: {last_err}")

    async def send_audio(self, pcm16: bytes) -> None:
        from google.genai import types
        await self._session.send_realtime_input(
            audio=types.Blob(data=pcm16, mime_type="audio/pcm;rate=16000"))

    async def send_text(self, text: str) -> None:
        # Turn-based content (NOT realtime input): mixing send_realtime_input(text)
        # with the audio stream disrupts automatic VAD so subsequent caller audio
        # never endpoints. send_client_content keeps the realtime audio path clean.
        from google.genai import types
        await self._session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True)

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        # session.receive() is a PER-TURN generator (it ends at turn_complete),
        # so loop it to read every turn. When the session closes, receive()
        # yields nothing and we stop.
        while True:
            produced = False
            async for ev in self._events_one_turn():
                produced = True
                yield ev
            if not produced:
                return

    async def _events_one_turn(self) -> AsyncIterator[RealtimeEvent]:
        async for msg in self._session.receive():
            sc = getattr(msg, "server_content", None)
            if sc is not None:
                it = getattr(sc, "input_transcription", None)
                if it and it.text:
                    yield RealtimeEvent(type="input_transcript", text=it.text)
                ot = getattr(sc, "output_transcription", None)
                if ot and ot.text:
                    yield RealtimeEvent(type="output_transcript", text=ot.text)
                mt = getattr(sc, "model_turn", None)
                if mt is not None:
                    for part in mt.parts or []:
                        d = getattr(part, "inline_data", None)
                        if d and d.data:
                            yield RealtimeEvent(type="audio", audio=d.data, audio_rate=24000)
                if getattr(sc, "interrupted", False):
                    yield RealtimeEvent(type="interrupted")
                if getattr(sc, "turn_complete", False):
                    yield RealtimeEvent(type="turn_complete")
            tc = getattr(msg, "tool_call", None)
            if tc is not None:
                for fc in tc.function_calls or []:
                    yield RealtimeEvent(type="tool_call", tool_name=fc.name,
                                        tool_args=dict(fc.args or {}), tool_id=fc.id or "")

    async def send_tool_response(self, *, tool_id: str, name: str, response: dict[str, Any]) -> None:
        from google.genai import types
        await self._session.send_tool_response(function_responses=[
            types.FunctionResponse(id=tool_id, name=name, response=response)])

    async def aclose(self) -> None:
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001 - teardown best-effort
            pass
