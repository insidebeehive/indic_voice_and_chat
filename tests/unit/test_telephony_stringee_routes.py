import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.telephony_hooks as hooks
from src.api.telephony_stringee_bridge import StringeeIvrBridge, pcm16_to_wav


class _Agent:
    def __init__(self):
        self.state = type("S", (), {"is_terminal": False})()
        self._action = "continue"
    async def start(self): pass
    async def play_opening(self, sink): await sink(b"\x01\x00" * 8)
    async def handle_turn(self, captured, sink):
        await sink(b"\x02\x00" * 8)
        return type("O", (), {"response": type("R", (), {"response_text": "ok", "action": self._action})()})()
    async def handle_hangup(self): pass


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(hooks.router, prefix="/api/v1")

    async def _fetch(url): return pcm16_to_wav(b"\x00\x00" * 80, 8000)

    def _factory(*, call_id, tenant, base_url, fetch):
        return StringeeIvrBridge(
            call_id=str(call_id), agent=_Agent(), llm=None,
            tenant_timezone="Asia/Kolkata", tts_sample_rate=16000,
            base_url=base_url, tenant_slug="dev", fetch=_fetch,
        )
    hooks.set_stringee_bridge_factory(_factory)

    from types import SimpleNamespace
    async def _resolve(num): return SimpleNamespace(slug="dev", id="dev")
    monkeypatch.setattr(hooks, "tenant_from_twilio_to_number", _resolve)
    yield TestClient(app)
    hooks.set_stringee_bridge_factory(None)


def test_answer_returns_opening_scco_and_audio_served(client):
    r = client.post("/api/v1/telephony/stringee/answer",
                    json={"call_id": "c1", "to": "+1", "from": "+2", "direction": "outbound"})
    assert r.status_code == 200
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"
    token = scco[0]["url"].rsplit("/", 1)[1]
    a = client.get(f"/api/v1/telephony/stringee/audio/{token}")
    assert a.status_code == 200 and a.content[:4] == b"RIFF"


def test_answer_by_slug_resolves_tenant_from_path(client, monkeypatch):
    """The slug route resolves the tenant from the URL path (the placing tenant),
    NOT from the shared caller/dialed number."""
    from types import SimpleNamespace
    seen = {}

    async def _by_slug(slug):
        seen["slug"] = slug
        return SimpleNamespace(slug=slug, id=slug)
    monkeypatch.setattr(hooks, "tenant_from_slug", _by_slug)

    r = client.get("/api/v1/telephony/stringee/answer/stage"
                   "?call_id=s1&from=918204268005&to=918618795697")
    assert r.status_code == 200
    assert seen["slug"] == "stage"          # resolved by slug, not number
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"


def test_answer_accepts_get_with_query_params(client):
    # Stringee fetches answer_url via GET with call info in the query string.
    r = client.get("/api/v1/telephony/stringee/answer"
                   "?call_id=g1&from=918204268005&to=918618795697")
    assert r.status_code == 200
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"
    assert "call_id=g1" in scco[1]["eventUrl"]


def test_event_runs_a_turn_and_returns_reply_scco(client):
    client.post("/api/v1/telephony/stringee/answer",
                json={"call_id": "c1", "to": "+1", "from": "+2", "direction": "outbound"})
    r = client.post("/api/v1/telephony/stringee/event/dev?call_id=c1",
                    json={"recording_url": "https://rec/1.wav"})
    assert r.status_code == 200
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage"


def test_stringee_number_extracts_from_str_dict_list():
    from src.api.telephony_hooks import _stringee_number
    assert _stringee_number("+918204268005") == "+918204268005"
    assert _stringee_number({"number": "+918204268005"}) == "+918204268005"
    assert _stringee_number([{"number": "+918204268005"}]) == "+918204268005"
    assert _stringee_number({"alias": "918204268005"}) == "918204268005"
    assert _stringee_number(None) is None
    assert _stringee_number([]) is None


def test_event_unknown_call_returns_reprompt(client):
    r = client.post("/api/v1/telephony/stringee/event/dev?call_id=nope",
                    json={"recording_url": "https://rec/1.wav"})
    assert r.status_code == 200
    assert r.json()[0]["action"] == "talk"


# --- Fix 1 route test: agent handle_turn raises -> 200 reprompt, never 500 ---


class _RaisingAgent:
    def __init__(self):
        self.state = type("S", (), {"is_terminal": False})()

    async def start(self): pass

    async def play_opening(self, sink): await sink(b"\x01\x00" * 8)

    async def handle_turn(self, captured, sink):
        raise RuntimeError("simulated provider failure")

    async def handle_hangup(self): pass


@pytest.fixture
def raising_client(monkeypatch):
    """Client whose bridge uses a raising agent."""
    app = FastAPI()
    app.include_router(hooks.router, prefix="/api/v1")

    async def _fetch(url): return pcm16_to_wav(b"\x00\x00" * 80, 8000)

    def _factory(*, call_id, tenant, base_url, fetch):
        return StringeeIvrBridge(
            call_id=str(call_id), agent=_RaisingAgent(), llm=None,
            tenant_timezone="Asia/Kolkata", tts_sample_rate=16000,
            base_url=base_url, tenant_slug="dev", fetch=_fetch,
        )

    hooks.set_stringee_bridge_factory(_factory)
    from types import SimpleNamespace
    async def _resolve(num): return SimpleNamespace(slug="dev", id="dev")
    monkeypatch.setattr(hooks, "tenant_from_twilio_to_number", _resolve)
    yield TestClient(app)
    hooks.set_stringee_bridge_factory(None)


def test_event_agent_raises_returns_200_reprompt(raising_client):
    """When the agent's handle_turn raises, the event route must return 200 with a talk SCCO."""
    raising_client.post("/api/v1/telephony/stringee/answer",
                        json={"call_id": "c-raise", "to": "+1", "from": "+2", "direction": "outbound"})
    r = raising_client.post("/api/v1/telephony/stringee/event/dev?call_id=c-raise",
                            json={"recording_url": "https://rec/1.wav"})
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    scco = r.json()
    assert scco[0]["action"] == "talk", f"expected talk reprompt, got {scco!r}"


# --- Fix 2 route test: audio fallback scan without call_id query param ---


def test_audio_served_without_call_id_query_param(client):
    """Stringee may fetch audio without the call_id query param; the fallback scan must serve it."""
    r = client.post("/api/v1/telephony/stringee/answer",
                    json={"call_id": "c-audio-scan", "to": "+1", "from": "+2", "direction": "outbound"})
    assert r.status_code == 200
    scco = r.json()
    token = scco[0]["url"].rsplit("/", 1)[1]
    # Fetch WITHOUT the call_id param — forces the fallback scan path
    a = client.get(f"/api/v1/telephony/stringee/audio/{token}")
    assert a.status_code == 200, f"expected 200, got {a.status_code}"
    assert a.content[:4] == b"RIFF", "expected WAV RIFF header"


# --- Hardening: optional call_id query param + broadened field-name probes ---


def test_event_without_call_id_query_returns_reprompt_not_422(client):
    """POST to /event/{tenant} with NO ?call_id= must return 200 reprompt (talk SCCO), not 422."""
    r = client.post(
        "/api/v1/telephony/stringee/event/dev",
        json={"recording_url": "https://rec/1.wav"},
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    scco = r.json()
    assert scco[0]["action"] == "talk", f"expected talk reprompt, got {scco!r}"


def test_event_accepts_file_url_field_name(client):
    """The broadened recording-url probe must recognise 'fileUrl' as a valid field name."""
    client.post(
        "/api/v1/telephony/stringee/answer",
        json={"call_id": "c1", "to": "+1", "from": "+2", "direction": "outbound"},
    )
    r = client.post(
        "/api/v1/telephony/stringee/event/dev?call_id=c1",
        json={"fileUrl": "https://rec/1.wav"},
    )
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    scco = r.json()
    assert scco[0]["action"] == "play" and scco[1]["action"] == "recordMessage", (
        f"expected play+recordMessage reply SCCO, got {scco!r}"
    )
