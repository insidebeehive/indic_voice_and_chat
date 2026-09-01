"""Route-level tests for the Twilio telephony hooks."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import telephony_hooks
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import TenantSettings


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(telephony_hooks.router)
    return app


def _register_dev_tenant_with_phone(phone: str = "+18888888888") -> None:
    register_tenant_for_test(
        TenantSettings(
            id="t_dev", slug="dev", name="Dev",
            phone_numbers=[phone],
        ),
    )


def test_twilio_voice_returns_twiml_with_tenant_scoped_stream_url() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice", data={"To": "+18888888888"})
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "<Response>" in body
        # Tenant slug embedded as path segment (Twilio strips query params).
        assert "/api/v1/telephony/twilio/stream/dev" in body
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_stream_url_ignores_configured_webhook_base_url(monkeypatch) -> None:
    """Regression test: the stream URL embedded in TwiML must be derived from
    the request that actually arrived, never from platform_webhook_base_url()
    — even when that config value IS set to a different host. (An earlier
    implementation attempt wired the inbound URL builders through the
    config-first public_origin(), which would have silently retargeted every
    inbound Twilio stream URL whenever WEBHOOK_BASE_URL pointed elsewhere.)
    """
    from src.utils import public_url

    monkeypatch.setattr(
        public_url, "platform_webhook_base_url",
        lambda: "https://configured-elsewhere.example/api/v1/telephony",
    )
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice", data={"To": "+18888888888"})
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "/api/v1/telephony/twilio/stream/dev" in body
        assert "configured-elsewhere.example" not in body
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_unknown_number_returns_404() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice", data={"To": "+919999999999"})
        assert resp.status_code == 404
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_outbound_resolves_tenant_by_from() -> None:
    """For ``Direction=outbound-api``, Twilio sets ``To`` to the end-user
    destination (we don't own it) and ``From`` to our Twilio number.
    The webhook must look up the tenant by ``From``, not ``To``."""
    _register_dev_tenant_with_phone("+18888888888")
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/telephony/twilio/voice",
            data={
                "To": "+14086605438",
                "From": "+18888888888",
                "Direction": "outbound-api",
                "CallSid": "CAtest",
            },
        )
        assert resp.status_code == 200, resp.text
        assert "/twilio/stream/dev" in resp.text
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_outbound_unknown_from_returns_404() -> None:
    _register_dev_tenant_with_phone("+18888888888")
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/telephony/twilio/voice",
            data={
                "To": "+14086605438",
                "From": "+19999999999",
                "Direction": "outbound-api",
            },
        )
        assert resp.status_code == 404
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_slug_route_resolves_by_slug_without_phone_registration() -> None:
    """Outbound calls WE place use a slug-scoped answer URL, so the tenant is
    resolved by slug — the caller-ID need NOT be in ``tenant_phone_numbers``
    (mirrors the Stringee ``/stringee/answer/{slug}`` design)."""
    # Tenant owns an UNRELATED number; the outbound From caller-ID is NOT registered.
    _register_dev_tenant_with_phone("+18888888888")
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/telephony/twilio/voice/dev",
            data={
                "To": "+918618795697",
                "From": "+15705255679",   # not in phone_numbers — must still work
                "Direction": "outbound-api",
                "CallSid": "CAslug",
            },
        )
        assert resp.status_code == 200, resp.text
        assert "/api/v1/telephony/twilio/stream/dev" in resp.text
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_slug_route_unknown_slug_returns_404() -> None:
    _register_dev_tenant_with_phone("+18888888888")
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice/ghost", data={"To": "+918618795697"})
        assert resp.status_code == 404
    finally:
        set_tenant_resolver(None)


def test_twilio_voice_missing_to_param_returns_422() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/twilio/voice")  # no form data
        assert resp.status_code == 422
    finally:
        set_tenant_resolver(None)


def test_websocket_without_factory_closes() -> None:
    _register_dev_tenant_with_phone()
    telephony_hooks.set_bridge_factory(None)
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/twilio/stream/dev") as ws:
            from starlette.websockets import WebSocketDisconnect

            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    finally:
        set_tenant_resolver(None)


def test_websocket_unknown_tenant_slug_closes() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/twilio/stream/ghost") as ws:
            from starlette.websockets import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    finally:
        set_tenant_resolver(None)


def test_websocket_drives_registered_bridge_with_tenant() -> None:
    """Factory receives (websocket, tenant) and runs the bridge.

    Tenant slug arrives as a URL path segment because Twilio strips
    query strings from <Stream url=...> attributes.
    """
    _register_dev_tenant_with_phone()
    received: list[tuple[str, str]] = []

    class MiniBridge:
        def __init__(self, ws, tenant):
            self._ws = ws
            self._tenant = tenant

        async def run(self):
            msg = await self._ws.receive_text()
            received.append((self._tenant.slug, msg))
            await self._ws.send_text(f"ack:{self._tenant.slug}")

    telephony_hooks.set_bridge_factory(lambda ws, tenant: MiniBridge(ws, tenant))
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/twilio/stream/dev") as ws:
            ws.send_text("hello")
            assert ws.receive_text() == "ack:dev"
    finally:
        telephony_hooks.set_bridge_factory(None)
        set_tenant_resolver(None)

    assert received == [("dev", "hello")]


# --- Exotel routes -----------------------------------------------------


def test_exotel_voice_returns_xml_with_tenant_scoped_stream_url() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/exotel/voice", data={"To": "+18888888888"})
        assert resp.status_code == 200, resp.text
        body = resp.text
        assert "<Response>" in body
        assert "/api/v1/telephony/exotel/stream/dev" in body
    finally:
        set_tenant_resolver(None)


def test_exotel_voice_unknown_number_returns_404() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/exotel/voice", data={"To": "+919999999999"})
        assert resp.status_code == 404
    finally:
        set_tenant_resolver(None)


def test_exotel_voice_outbound_resolves_tenant_by_from() -> None:
    _register_dev_tenant_with_phone("+18888888888")
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/telephony/exotel/voice",
            data={
                "To": "+919999999999",
                "From": "+18888888888",
                "Direction": "outbound-api",
                "CallSid": "EXtest",
            },
        )
        assert resp.status_code == 200, resp.text
        assert "/exotel/stream/dev" in resp.text
    finally:
        set_tenant_resolver(None)


def test_exotel_voice_slug_route_resolves_by_slug_without_phone_registration() -> None:
    _register_dev_tenant_with_phone("+18888888888")
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/telephony/exotel/voice/dev",
            data={
                "To": "+918618795697",
                "From": "+15705255679",   # not in phone_numbers — must still work
                "Direction": "outbound-api",
                "CallSid": "EXslug",
            },
        )
        assert resp.status_code == 200, resp.text
        assert "/api/v1/telephony/exotel/stream/dev" in resp.text
    finally:
        set_tenant_resolver(None)


def test_exotel_voice_slug_route_unknown_slug_returns_404() -> None:
    _register_dev_tenant_with_phone("+18888888888")
    try:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/telephony/exotel/voice/ghost", data={"To": "+918618795697"})
        assert resp.status_code == 404
    finally:
        set_tenant_resolver(None)


def test_exotel_websocket_without_factory_closes() -> None:
    _register_dev_tenant_with_phone()
    telephony_hooks.set_exotel_bridge_factory(None)
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/exotel/stream/dev") as ws:
            from starlette.websockets import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    finally:
        set_tenant_resolver(None)


def test_exotel_websocket_unknown_tenant_slug_closes() -> None:
    _register_dev_tenant_with_phone()
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/exotel/stream/ghost") as ws:
            from starlette.websockets import WebSocketDisconnect
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    finally:
        set_tenant_resolver(None)


def test_exotel_websocket_drives_registered_bridge_with_tenant() -> None:
    _register_dev_tenant_with_phone()
    received: list[tuple[str, str]] = []

    class MiniBridge:
        def __init__(self, ws, tenant):
            self._ws = ws
            self._tenant = tenant

        async def run(self):
            msg = await self._ws.receive_text()
            received.append((self._tenant.slug, msg))
            await self._ws.send_text(f"ack:{self._tenant.slug}")

    telephony_hooks.set_exotel_bridge_factory(lambda ws, tenant: MiniBridge(ws, tenant))
    try:
        app = _make_app()
        client = TestClient(app)
        with client.websocket_connect("/telephony/exotel/stream/dev") as ws:
            ws.send_text("hello")
            assert ws.receive_text() == "ack:dev"
    finally:
        telephony_hooks.set_exotel_bridge_factory(None)
        set_tenant_resolver(None)

    assert received == [("dev", "hello")]
