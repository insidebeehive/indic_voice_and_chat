import pytest

from src.api.telephony_stringee_bridge import (
    StringeeIvrBridge,
    pcm16_to_wav,
    registry,
)


class _FakeResponse:
    def __init__(self, response_text="जी हाँ", action="continue"):
        self.response = type("R", (), {"response_text": response_text, "action": action})()


class _FakeAgent:
    """Records calls; play_opening + handle_turn push PCM into the sink."""
    def __init__(self):
        self.started = False
        self.turns = []
        self.hung_up = False
        self.state = type("S", (), {"is_terminal": False})()
        self._next = _FakeResponse()

    async def start(self):
        self.started = True

    async def play_opening(self, sink):
        await sink(b"\x10\x00" * 8)  # 16 bytes of "opening" PCM

    async def handle_turn(self, captured, sink):
        self.turns.append(captured)
        await sink(b"\x20\x00" * 8)  # "reply" PCM
        return type("O", (), {"response": self._next.response})()

    async def handle_hangup(self):
        self.hung_up = True


async def _fetch_ok(url):  # injected downloader -> returns WAV of silence
    return pcm16_to_wav(b"\x00\x00" * 80, sample_rate=8000)


def _bridge(agent):
    return StringeeIvrBridge(
        call_id="call-1", agent=agent, llm=None,
        tenant_timezone="Asia/Kolkata", tts_sample_rate=16000,
        base_url="https://host/api/v1/telephony/stringee", tenant_slug="dev",
        fetch=_fetch_ok,
    )


@pytest.mark.asyncio
async def test_start_call_returns_answer_scco_with_hosted_opening():
    agent = _FakeAgent()
    bridge = _bridge(agent)
    scco = await bridge.start_call()
    assert scco[0]["action"] == "play"
    assert scco[0]["url"].startswith("https://host/api/v1/telephony/stringee/audio/")
    assert scco[1]["action"] == "recordMessage"
    token = scco[0]["url"].rsplit("/", 1)[1]
    assert bridge.audio.get(token) is not None


@pytest.mark.asyncio
async def test_handle_turn_runs_agent_and_returns_reply_scco():
    agent = _FakeAgent()
    bridge = _bridge(agent)
    scco = await bridge.handle_turn(recording_url="https://rec/1.wav")
    assert len(agent.turns) == 1
    assert agent.turns[0]
    assert scco[0]["action"] == "play"
    assert scco[1]["action"] == "recordMessage"


@pytest.mark.asyncio
async def test_handle_turn_terminal_action_returns_closing_scco():
    agent = _FakeAgent()
    agent._next = _FakeResponse(response_text="Dhanyavaad", action="close_positive")
    bridge = _bridge(agent)
    scco = await bridge.handle_turn(recording_url="https://rec/1.wav")
    assert [a["action"] for a in scco] == ["play", "hangup"]


@pytest.mark.asyncio
async def test_handle_turn_empty_reply_reprompts():
    agent = _FakeAgent()
    agent._next = _FakeResponse(response_text="", action="continue")
    bridge = _bridge(agent)
    scco = await bridge.handle_turn(recording_url="https://rec/1.wav")
    assert scco[0]["action"] == "talk"
    assert scco[1]["action"] == "recordMessage"


@pytest.mark.asyncio
async def test_registry_create_lookup_end():
    agent = _FakeAgent()
    bridge = _bridge(agent)
    registry.put(bridge)
    assert registry.get("call-1") is bridge
    await registry.end("call-1")
    assert agent.hung_up is True
    assert registry.get("call-1") is None


@pytest.mark.asyncio
async def test_start_call_starts_the_agent_before_opening():
    agent = _FakeAgent()
    bridge = _bridge(agent)
    await bridge.start_call()
    assert agent.started is True


# --- Fix 1: agent failures must NOT 500 / drop the call ---


class _RaisingAgent(_FakeAgent):
    """Agent whose handle_turn raises to simulate a transient LLM/STT/TTS error."""

    async def handle_turn(self, captured, sink):
        raise RuntimeError("simulated provider failure")


@pytest.mark.asyncio
async def test_handle_turn_agent_raises_returns_reprompt_scco():
    """When agent.handle_turn raises, handle_turn must return a reprompt (talk) SCCO."""
    agent = _RaisingAgent()
    bridge = _bridge(agent)
    scco = await bridge.handle_turn(recording_url="https://rec/1.wav")
    assert scco[0]["action"] == "talk", f"expected 'talk' reprompt, got {scco!r}"
    assert scco[1]["action"] == "recordMessage"


async def test_make_stringee_bridge_factory_builds_a_bridge():
    from types import SimpleNamespace

    from src.bootstrap import make_stringee_bridge_factory
    from src.dialogue.prompts import VoiceBotScript
    from src.dialogue.slots import SlotSchema

    class _Providers:
        def get_stt(self, t): return object()
        def get_llm(self, t): return None
        def get_tts(self, t): return object()

    tenant = SimpleNamespace(
        id="dev", slug="dev",
        settings=SimpleNamespace(pipeline=SimpleNamespace(
            stt=SimpleNamespace(language="hi-IN"),
            llm=SimpleNamespace(temperature=0.5, max_tokens=256, response_format="json"),
            tts=SimpleNamespace(language="hi-IN", voice_id=None),
        )),
    )
    script = VoiceBotScript.from_campaign_yaml(
        {"agent_name": "A", "agent_role": "R", "company_name": "C"}
    )
    factory = make_stringee_bridge_factory(
        providers=_Providers(), script=script, slots=SlotSchema(),
    )
    async def _fetch(url): return b""
    bridge = await factory(call_id="c-9", tenant=tenant,
                           base_url="https://h/api/v1/telephony/stringee", fetch=_fetch)
    assert bridge.call_id == "c-9"
