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

import pytest

import src.api.dev_console as devmod
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import (
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
)
from src.interfaces.telephony import CallSession


@pytest.fixture(autouse=True)
def _no_db_recording(monkeypatch):
    """Place-call records a conversation row via insert_call; unit tests have no
    DB, so stub it to a no-op by default. The dedicated recording test overrides
    this to assert the wiring."""
    async def _noop(db, **kw):
        return None
    monkeypatch.setattr(devmod, "insert_call", _noop, raising=False)


def _register_dev_tenant(provider="stringee", outbound_from=None):
    register_tenant_for_test(TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider=provider, from_number="+918204268005",
            outbound_from=outbound_from or {}))))


def _client():
    app = FastAPI()
    app.include_router(dev_router)
    return TestClient(app)


def _pin_platform_webhook(monkeypatch, base="https://example.test/api/v1/telephony"):
    """The webhook base is platform-level (WEBHOOK_BASE_URL), not per-tenant —
    pin it so the outbound Answer URL is deterministic."""
    from types import SimpleNamespace

    import src.config as _cfg
    monkeypatch.setattr(_cfg, "get_settings", lambda: SimpleNamespace(
        pipeline=SimpleNamespace(telephony=SimpleNamespace(webhook_base_url=base))))


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
    _pin_platform_webhook(monkeypatch)
    _register_dev_tenant(provider="stringee", outbound_from={"twilio": "+15705255679"})
    try:
        client = _client()
        resp = client.post("/dev/place-call", json={
            "provider": "twilio", "to_number": "+919999999999",
            "mode": "s2s", "voice": "Kore", "lead_name": "Raju"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["provider_call_sid"] == "CA123"
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


def test_place_call_passes_tenant_creds_to_adapter(monkeypatch):
    """The dev place-call must build the adapter with the SELECTED provider's
    per-tenant creds (no platform-env fallback) — Stringee gets them mapped onto
    api_key_sid/api_key_secret."""
    monkeypatch.setenv("DEV_STR_SID", "SK.dev-sid")
    monkeypatch.setenv("DEV_STR_SECRET", "dev-secret")
    captured = {}

    class _FakeAdapter:
        async def initiate_call(self, cfg):
            return CallSession(session_id="CA1", status="ringing",
                               to_number=cfg.to_number, from_number=cfg.from_number)

    monkeypatch.setattr(devmod, "get_telephony_provider",
                        lambda cfg: captured.update(cfg) or _FakeAdapter())
    register_tenant_for_test(TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider="stringee", from_number="+918204268005",
            account_sid_env="DEV_STR_SID", auth_token_env="DEV_STR_SECRET"))))
    resp = _client().post("/dev/place-call", json={
        "provider": "stringee", "to_number": "+918618795697", "mode": "s2s", "voice": "Kore"})
    assert resp.status_code == 200, resp.text
    assert captured["account_sid"] == "SK.dev-sid"
    assert captured["api_key_sid"] == "SK.dev-sid"        # mapped for the Stringee adapter
    assert captured["api_key_secret"] == "dev-secret"


def test_place_call_passes_stringee_user_id_to_adapter(monkeypatch):
    """Stringee's callout needs a non-null userId, or it degrades to a
    phone->phone external call and the Answer URL/SCCO never runs (the bot stays
    silent). The dev place-call must resolve the tenant's per-provider
    ``user_id_env`` and pass it to the adapter as ``user_id``."""
    monkeypatch.setenv("DEV_STR_SID", "SK.dev-sid")
    monkeypatch.setenv("DEV_STR_SECRET", "dev-secret")
    monkeypatch.setenv("DEV_STR_USER", "dev")
    captured = {}

    class _FakeAdapter:
        async def initiate_call(self, cfg):
            return CallSession(session_id="CA1", status="ringing",
                               to_number=cfg.to_number, from_number=cfg.from_number)

    monkeypatch.setattr(devmod, "get_telephony_provider",
                        lambda cfg: captured.update(cfg) or _FakeAdapter())
    register_tenant_for_test(TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider="stringee", from_number="+918204268005",
            account_sid_env="DEV_STR_SID", auth_token_env="DEV_STR_SECRET",
            user_id_env="DEV_STR_USER"))))
    resp = _client().post("/dev/place-call", json={
        "provider": "stringee", "to_number": "+918618795697", "mode": "s2s", "voice": "Kore"})
    assert resp.status_code == 200, resp.text
    assert captured["user_id"] == "dev"


def test_place_call_records_conversation_for_requesting_tenant(monkeypatch):
    """An outbound dev place-call registers a conversation row for the tenant that
    PLACED it (not derived from the number), as a voice/voicebot call, and returns
    the same shape as the campaign path (call_id + provider_call_sid)."""
    _pin_platform_webhook(monkeypatch)
    captured = {}

    class _FakeAdapter:
        async def initiate_call(self, cfg):
            return CallSession(session_id="STR1", status="starting",
                               to_number=cfg.to_number, from_number=cfg.from_number)
    monkeypatch.setattr(devmod, "get_telephony_provider", lambda c: _FakeAdapter())

    async def _fake_insert(db, **kw):
        captured.update(kw)
    monkeypatch.setattr(devmod, "insert_call", _fake_insert)

    class _Ctx:
        async def __aenter__(self): return "db"
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(devmod, "get_sessionmaker", lambda: (lambda: _Ctx()))

    register_tenant_for_test(TenantSettings(
        id="t_stage", slug="stage", name="Stage",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider="stringee", from_number="+918204268005"))))
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "stringee", "to_number": "+918618795697",
            "mode": "s2s", "voice": "Kore", "tenant": "stage"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["provider_call_sid"] == "STR1"
        assert body["call_id"]
        # row attributed to the PLACING tenant, as a voice/voicebot call
        assert captured["tenant"].id == "t_stage"
        assert captured["channel"] == "voice"
        assert captured["provider_call_sid"] == "STR1"
    finally:
        set_tenant_resolver(None)


def test_place_call_falls_back_to_platform_webhook_base_url(monkeypatch):
    """A tenant with no webhook_base_url uses the platform-level WEBHOOK_BASE_URL
    to build the outbound Answer URL (the inbound callback is always our app)."""
    from types import SimpleNamespace

    import src.config as cfg
    fake = SimpleNamespace(pipeline=SimpleNamespace(telephony=SimpleNamespace(
        webhook_base_url="https://platform.example/api/v1/telephony")))
    monkeypatch.setattr(cfg, "get_settings", lambda: fake)

    captured = {}

    class _FakeAdapter:
        async def initiate_call(self, c):
            captured["webhook_url"] = c.webhook_url
            return CallSession(session_id="CA1", status="ringing",
                               to_number=c.to_number, from_number=c.from_number)

    monkeypatch.setattr(devmod, "get_telephony_provider", lambda cfg_: _FakeAdapter())
    register_tenant_for_test(TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider="stringee", from_number="+918204268005"))))  # no webhook_base_url
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "stringee", "to_number": "+918618795697", "mode": "s2s", "voice": "Kore"})
        assert resp.status_code == 200, resp.text
        assert captured["webhook_url"] == "https://platform.example/api/v1/telephony/stringee/answer/dev"
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
    _pin_platform_webhook(monkeypatch)
    _register_dev_tenant(provider="stringee")          # from_number +918204268005
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "stringee", "to_number": "+918618795697", "mode": "s2s", "voice": "Kore"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["provider_call_sid"] == "STR1"
        cfg = captured["cfg"]
        assert cfg.from_number == "+918204268005"
        assert cfg.webhook_url == "https://example.test/api/v1/telephony/stringee/answer/dev"
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
