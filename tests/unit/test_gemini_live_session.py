from __future__ import annotations

import pytest

from src.interfaces.realtime import RealtimeTool
from src.providers.realtime.gemini_live import GeminiLiveSession, _to_tool


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeSession:
    def __init__(self, msgs):
        self._msgs = msgs
        self.tool_responses = []
        self._consumed = False

    async def receive(self):
        # Per-turn generator: yields the turn's messages once, then nothing
        # (session closed) — matching the real SDK so events()'s loop terminates.
        if self._consumed:
            return
        self._consumed = True
        for m in self._msgs:
            yield m

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)


@pytest.mark.asyncio
async def test_events_translate_live_messages():
    msgs = [
        _Obj(server_content=_Obj(input_transcription=_Obj(text="yeh app safe hai"),
                                 output_transcription=None, model_turn=None,
                                 interrupted=False, turn_complete=False),
             tool_call=None),
        _Obj(server_content=_Obj(input_transcription=None,
                                 output_transcription=_Obj(text="bilkul safe hai"),
                                 model_turn=_Obj(parts=[_Obj(inline_data=_Obj(data=b"PCM24"))]),
                                 interrupted=False, turn_complete=True),
             tool_call=None),
        _Obj(server_content=None,
             tool_call=_Obj(function_calls=[
                 _Obj(name="record_turn_signal", args={"action": "send_info"}, id="t1")])),
    ]
    sess = GeminiLiveSession(cm=None, session=_FakeSession(msgs))
    events = [e async for e in sess.events()]
    by = {e.type for e in events}
    assert {"input_transcript", "output_transcript", "audio", "turn_complete", "tool_call"} <= by
    audio = [e for e in events if e.type == "audio"][0]
    assert audio.audio == b"PCM24" and audio.audio_rate == 24000
    tc = [e for e in events if e.type == "tool_call"][0]
    assert tc.tool_name == "record_turn_signal"
    assert tc.tool_args == {"action": "send_info"} and tc.tool_id == "t1"


@pytest.mark.asyncio
async def test_interrupted_event():
    msgs = [_Obj(server_content=_Obj(input_transcription=None, output_transcription=None,
                                     model_turn=None, interrupted=True, turn_complete=False),
                 tool_call=None)]
    sess = GeminiLiveSession(cm=None, session=_FakeSession(msgs))
    events = [e async for e in sess.events()]
    assert [e.type for e in events] == ["interrupted"]


def test_speech_config_omits_language_code_when_unset():
    # Native-audio Live models auto-switch language and reject an explicit code,
    # so an unset language_code must NOT be pinned on the speech config.
    from google.genai import types

    from src.providers.realtime.gemini_live import _build_speech_config

    sc = _build_speech_config(types, None, "Aoede")
    assert not getattr(sc, "language_code", None)          # left unset
    assert sc.voice_config is not None                     # voice still applied

    sc2 = _build_speech_config(types, "hi-IN", None)
    assert sc2.language_code == "hi-IN"                     # honored when given


def test_activity_detection_tuned_for_quick_first_utterance():
    # The default VAD was sluggish to detect a short opening "hello" (callers had
    # to repeat it). Start-of-speech sensitivity must be HIGH so onset is caught
    # promptly, with a little prefix padding so the first phoneme isn't clipped.
    from google.genai import types

    from src.providers.realtime.gemini_live import _build_activity_detection

    ad = _build_activity_detection(types)
    assert ad.start_of_speech_sensitivity == types.StartSensitivity.START_SENSITIVITY_HIGH
    assert ad.prefix_padding_ms and ad.prefix_padding_ms > 0


def test_to_tool_builds_function_declaration():
    from google.genai import types
    tool = _to_tool(types, RealtimeTool(
        name="record_turn_signal", description="record action+slots",
        parameters={"type": "OBJECT",
                    "properties": {"action": {"type": "STRING", "enum": ["continue", "send_info"]},
                                   "updated_slots": {"type": "OBJECT"}},
                    "required": ["action"]}))
    fd = tool.function_declarations[0]
    assert fd.name == "record_turn_signal"
    assert "action" in fd.parameters.properties
