# tests/unit/test_dev_console.py
from __future__ import annotations

from src.api.dev_console import dev_console_enabled


def test_dev_console_enabled_flag(monkeypatch):
    monkeypatch.delenv("VOX_DEV_CONSOLE", raising=False)
    assert dev_console_enabled() is False
    monkeypatch.setenv("VOX_DEV_CONSOLE", "1")
    assert dev_console_enabled() is True


# append to tests/unit/test_dev_console.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dev_console import dev_router


def test_dev_voice_page_served():
    app = FastAPI()
    app.include_router(dev_router)
    client = TestClient(app)
    resp = client.get("/dev/voice")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Voice Dev Console" in resp.text


# --- place-call + status endpoints --------------------------------------------

import src.api.dev_console as devmod
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import (
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
)
from src.interfaces.telephony import CallSession


def _register_dev_tenant(provider="stringee", outbound_from=None):
    register_tenant_for_test(TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider=provider, from_number="+918204268005",
            webhook_base_url="https://example.test/api/v1/telephony",
            outbound_from=outbound_from or {}))))


def _client():
    app = FastAPI()
    app.include_router(dev_router)
    return TestClient(app)


def test_place_call_uses_selected_provider_and_its_caller_id(monkeypatch):
    """The dropdown drives the provider: even though the tenant's default is
    stringee, selecting twilio builds the twilio adapter and dials from the
    twilio caller-ID."""
    captured = {}

    class _FakeAdapter:
        async def initiate_call(self, cfg):
            captured["cfg"] = cfg
            return CallSession(session_id="CA123", status="ringing",
                               to_number=cfg.to_number, from_number=cfg.from_number)

    def _fake_build(cfg):
        captured["provider"] = cfg["provider"]
        return _FakeAdapter()

    monkeypatch.setattr(devmod, "get_telephony_provider", _fake_build)
    _register_dev_tenant(provider="stringee", outbound_from={"twilio": "+15705255679"})
    try:
        client = _client()
        resp = client.post("/dev/place-call", json={
            "provider": "twilio", "to_number": "+919999999999",
            "mode": "s2s", "voice": "Kore", "lead_name": "Raju"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["call_sid"] == "CA123"
        assert captured["provider"] == "twilio"             # selected provider's adapter
        cfg = captured["cfg"]
        assert cfg.to_number == "+919999999999"
        assert cfg.from_number == "+15705255679"            # twilio caller-ID, not stringee's
        assert cfg.webhook_url == "https://example.test/api/v1/telephony/twilio/voice"

        from src.api import dev_call_control
        assert dev_call_control.monitor.get("CA123")["status"] == "calling"
        assert client.get("/dev/call-status/CA123").json()["status"] == "calling"
        assert client.get("/dev/call-status/NOPE").json()["status"] == "unknown"
        assert dev_call_control.pop_override("dev") == {
            "mode": "s2s", "voice": "Kore", "lead_name": "Raju"}
    finally:
        set_tenant_resolver(None)


def test_dev_voices_per_mode():
    """The Voice dropdown source is per-mode: layered = the configured TTS
    provider's roster (default = pipeline.tts.voice_id); s2s = the realtime
    allowed_voices (default = pipeline.realtime.voice)."""
    from src.config_tenant import (
        TenantPipelineConfig,
        TenantRealtimeConfig,
        TenantSettings,
        TenantTTSConfig,
    )

    register_tenant_for_test(TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(
            tts=TenantTTSConfig(language="hi-IN", voice_id="anushka"),
            realtime=TenantRealtimeConfig(voice="Aoede", allowed_voices=["Aoede", "Kore", "Leda"]),
        )))

    class _FakeTTS:
        def get_available_voices(self, language):
            assert language == "hi-IN"
            return [{"voice_id": "anushka", "gender": "female"},
                    {"voice_id": "karun", "gender": "male"}]

    class _FakeProviders:
        def get_tts(self, tenant):
            return _FakeTTS()

    try:
        app = FastAPI()
        app.include_router(dev_router)
        app.state.providers = _FakeProviders()
        resp = TestClient(app).get("/dev/voices?tenant=dev")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["layered"]["voices"] == ["anushka", "karun"]
        assert data["layered"]["default"] == "anushka"
        assert data["s2s"]["voices"] == ["Aoede", "Kore", "Leda"]
        assert data["s2s"]["default"] == "Aoede"
    finally:
        set_tenant_resolver(None)


def test_place_call_no_caller_id_for_provider(monkeypatch):
    def _no_build(cfg):
        raise AssertionError("adapter should not be built without a caller-ID")

    monkeypatch.setattr(devmod, "get_telephony_provider", _no_build)
    _register_dev_tenant(provider="stringee", outbound_from={"twilio": "+15705255679"})
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "exotel", "to_number": "+919999999999"})  # no exotel caller-ID
        assert resp.status_code == 400
        assert "exotel" in resp.json()["detail"]
    finally:
        set_tenant_resolver(None)


def test_place_call_rejects_unsupported_provider():
    _register_dev_tenant()
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "telnyx", "to_number": "+919999999999"})
        assert resp.status_code == 400
        assert "telnyx" in resp.json()["detail"]
    finally:
        set_tenant_resolver(None)


def test_place_call_stringee_uses_answer_webhook_no_override(monkeypatch):
    """Stringee is accepted, dials from its configured from_number via the IVR
    /stringee/answer webhook, and does NOT set a Mode/Voice override (IVR-only)."""
    captured = {}

    class _FakeAdapter:
        async def initiate_call(self, cfg):
            captured["cfg"] = cfg
            return CallSession(session_id="STR1", status="starting",
                               to_number=cfg.to_number, from_number=cfg.from_number)

    monkeypatch.setattr(devmod, "get_telephony_provider", lambda cfg: _FakeAdapter())
    _register_dev_tenant(provider="stringee")          # from_number +918204268005
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "stringee", "to_number": "+918618795697", "mode": "s2s", "voice": "Kore"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["call_sid"] == "STR1"
        cfg = captured["cfg"]
        assert cfg.from_number == "+918204268005"
        assert cfg.webhook_url == "https://example.test/api/v1/telephony/stringee/answer"
        from src.api import dev_call_control
        assert dev_call_control.monitor.get("STR1")["status"] == "calling"
        assert dev_call_control.pop_override("dev") is None     # IVR ignores Mode/Voice
    finally:
        set_tenant_resolver(None)


def test_place_call_failure_clears_override(monkeypatch):
    class _BoomAdapter:
        async def initiate_call(self, cfg):
            raise RuntimeError("twilio rejected")

    monkeypatch.setattr(devmod, "get_telephony_provider", lambda cfg: _BoomAdapter())
    _register_dev_tenant(provider="twilio", outbound_from={"twilio": "+15705255679"})
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "twilio", "to_number": "+919999999999", "mode": "s2s"})
        assert resp.status_code == 502
        from src.api import dev_call_control
        assert dev_call_control.pop_override("dev") is None   # not left stale
    finally:
        set_tenant_resolver(None)
