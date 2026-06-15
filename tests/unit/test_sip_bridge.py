"""SipMediaBridge: caller RTP -> model, model audio -> caller RTP, hangup.

Uses a fake ISipCall + fake realtime session so the whole SIP bridge is tested
without any real SIP/RTP stack (pyVoIP). Transport hooks are exercised directly
(deterministic), the way the Gemini live-bridge tests do.
"""

from __future__ import annotations

import pytest

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api.sip_media_bridge import SipMediaBridge
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.realtime import RealtimeConfig, RealtimeEvent


class _FakeSipCall:
    def __init__(self, frames=(), answered=True):
        self._frames = list(frames)
        self._answered = answered
        self.sent_audio = []
        self.hung_up = False
        self.flushed = False

    @property
    def call_id(self):
        return "sip_test"

    async def wait_answered(self):
        return self._answered

    async def audio_in(self):
        for f in self._frames:
            yield f

    async def send_audio(self, pcm16_8k):
        self.sent_audio.append(pcm16_8k)

    async def flush(self):
        self.flushed = True

    async def hangup(self):
        self.hung_up = True


class _FakeSession:
    def __init__(self):
        self.audio_in = []

    async def send_audio(self, p):
        self.audio_in.append(p)

    async def send_tool_response(self, *, tool_id, name, response):
        pass

    async def aclose(self):
        pass


def _bridge(sip):
    agent = VoiceBotAgent(
        session=AgentSession(session_id="sip1", lead_data={}),
        state_machine=AgentStateMachine(), slot_schema=SlotSchema(),
        script=VoiceBotScript(agent_name="Anaaya", agent_role="sales", company_name="X"),
        engine=object(), store=None)

    async def connect(cfg):
        return _FakeSession()

    b = SipMediaBridge(sip_call=sip, agent=agent, config=RealtimeConfig(model="m"),
                       connect_session=connect)
    b._session = _FakeSession()
    return b


@pytest.mark.asyncio
async def test_inbound_forwards_caller_audio_to_model():
    sip = _FakeSipCall(frames=[b"\x00\x01" * 160, b"\x02\x03" * 160])
    b = _bridge(sip)
    await b._inbound_loop()
    assert len(b._session.audio_in) == 2          # both 20ms frames forwarded (8k->16k)


@pytest.mark.asyncio
async def test_model_audio_goes_out_over_sip():
    sip = _FakeSipCall()
    b = _bridge(sip)
    await b._send_audio_out(b"\x05\x06" * 160, 16000)
    assert len(sip.sent_audio) == 1               # resampled 16k->8k and sent to RTP


@pytest.mark.asyncio
async def test_interrupt_flushes_outbound():
    sip = _FakeSipCall()
    b = _bridge(sip)
    await b._send_interrupt()
    assert sip.flushed is True


@pytest.mark.asyncio
async def test_teardown_hangs_up():
    sip = _FakeSipCall()
    b = _bridge(sip)
    await b._on_teardown()
    assert sip.hung_up is True


@pytest.mark.asyncio
async def test_on_start_stops_when_not_answered():
    sip = _FakeSipCall(answered=False)
    b = _bridge(sip)
    await b._on_start()
    assert b._stopped is True
