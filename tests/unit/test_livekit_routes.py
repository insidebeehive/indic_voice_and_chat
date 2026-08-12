"""Route-level tests for the LiveKit webhook (Phase 5): tenant resolution,
real-JWT signature verification, event filtering, dedupe/concurrency, and
metadata extraction. Zero network access — no LiveKit env vars needed."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from livekit import api
from livekit.protocol.models import ParticipantInfo, Room
from livekit.protocol.webhook import WebhookEvent
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import livekit_routes
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth import secrets as crypto
from src.auth.middleware import set_tenant_resolver
from src.config_tenant import (
    TelephonyCreds,
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
)
from src.models.crm import Crm, CrmSecret
from src.models.database import Base

_LK_KEY = "devkey"
_LK_SECRET = "devsecret-devsecret-devsecret-32"   # >=32 bytes: avoids PyJWT's InsecureKeyLengthWarning
_URL = "/telephony/livekit/webhook/dev"


def _register_tenant(*, max_concurrent_calls: int = 5) -> None:
    settings = TenantSettings(
        id="t_dev", slug="dev", name="Dev",
        max_concurrent_calls=max_concurrent_calls,
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            livekit_url="wss://fake.example/rtc",
            creds_by_provider={
                "livekit": TelephonyCreds(
                    account_sid_env="TEST_LK_KEY", auth_token_env="TEST_LK_SECRET"),
            },
        )),
    )
    ctx = register_tenant_for_test(settings)
    # secrets_resolved is populated by the DB resolver in prod; the in-memory
    # test resolver only takes plaintext_tokens, so set it directly via the
    # frozen dataclass's already-registered instance is not possible — instead
    # monkeypatch TenantContext.secret indirectly by injecting env vars.
    import os
    os.environ["TEST_LK_KEY"] = _LK_KEY
    os.environ["TEST_LK_SECRET"] = _LK_SECRET
    return ctx


def _body(event="participant_joined", *, room="room-abc", kind="SIP",
          room_meta="", part_meta="", attrs=None) -> str:
    return json.dumps({
        "event": event,
        "room": {"sid": "RM_1", "name": room, "metadata": room_meta},
        "participant": {"sid": "PA_1", "identity": "sip_+919999999999",
                         "kind": kind, "metadata": part_meta, "attributes": attrs or {}},
        "id": "EV_1", "createdAt": 1700000000,
    })


def _sign(body: str, secret: str = _LK_SECRET) -> str:
    digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
    return api.AccessToken(_LK_KEY, secret).with_sha256(digest).to_jwt()


@pytest.fixture
async def client(monkeypatch):
    _register_tenant()

    async def _session_override():
        yield object()

    async def _no_active(session, tenant_id):
        return 0

    monkeypatch.setattr(livekit_routes, "count_active_calls", _no_active)
    livekit_routes._in_flight_rooms.clear()
    sentinel = object()
    livekit_routes.set_livekit_bridge_factory(sentinel)

    app = FastAPI()
    app.include_router(livekit_routes.router)
    app.dependency_overrides[get_db_session] = _session_override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c._sentinel_factory = sentinel  # stash for tests that assert identity
        yield c

    livekit_routes.set_livekit_bridge_factory(None)
    livekit_routes._in_flight_rooms.clear()
    set_tenant_resolver(None)
    import os
    os.environ.pop("TEST_LK_KEY", None)
    os.environ.pop("TEST_LK_SECRET", None)


def _spy_run_call(recorded: list, *, event: "asyncio.Event | None" = None,
                   release: "asyncio.Event | None" = None, raise_error: bool = False):
    async def fake(tenant, room_name, meta, *, bridge_factory, sessionmaker=None):
        recorded.append((tenant, room_name, meta, bridge_factory))
        if event is not None:
            event.set()
        if release is not None:
            await release.wait()
        if raise_error:
            raise RuntimeError("boom")
    return fake


async def _post(client, body: str, token: str | None) -> "object":
    headers = {"Content-Type": "application/webhook+json"}
    if token is not None:
        headers["Authorization"] = token
    return await client.post(_URL, content=body, headers=headers)


# --- signature / tenant resolution --------------------------------------


async def test_unknown_tenant_slug_404_before_signature_check(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    body = _body()
    resp = await client.post(
        "/telephony/livekit/webhook/nope", content=body,
        headers={"Authorization": "garbage-not-a-jwt",
                 "Content-Type": "application/webhook+json"})
    assert resp.status_code == 404
    assert recorded == []


async def test_missing_auth_header_401(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    resp = await _post(client, _body(), None)
    assert resp.status_code == 401
    assert recorded == []


async def test_bad_signature_401(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    body = _body()
    token = _sign(body, secret="a-totally-different-secret-32-bytes")
    resp = await _post(client, body, token)
    assert resp.status_code == 401
    assert recorded == []


async def test_tampered_body_401(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    body_a = _body(room="room-a")
    body_b = _body(room="room-b")
    token = _sign(body_a)
    resp = await _post(client, body_b, token)
    assert resp.status_code == 401
    assert recorded == []


async def test_401_body_does_not_leak_reason(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    body = _body()
    r_missing = await _post(client, body, None)
    r_bad_sig = await _post(client, body, _sign(body, secret="wrong-secret-at-least-32-bytes!!"))
    r_tampered = await _post(client, _body(room="other"), _sign(body))
    assert r_missing.status_code == r_bad_sig.status_code == r_tampered.status_code == 401
    assert r_missing.json() == r_bad_sig.json() == r_tampered.json()


# --- event filtering ------------------------------------------------------


async def test_room_started_ignored(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    body = _body(event="room_started")
    resp = await _post(client, body, _sign(body))
    assert resp.status_code == 200
    assert recorded == []


async def test_participant_joined_non_sip_ignored(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    body = _body(kind="STANDARD")
    resp = await _post(client, body, _sign(body))
    assert resp.status_code == 200
    assert recorded == []


# --- spawn / dedupe / concurrency -----------------------------------------


async def test_participant_joined_sip_spawns_run_call(client, monkeypatch):
    recorded: list = []
    started = asyncio.Event()
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded, event=started))
    body = _body(room="room-abc")
    resp = await _post(client, body, _sign(body))
    assert resp.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=2.0)
    assert len(recorded) == 1
    tenant, room_name, meta, bridge_factory = recorded[0]
    assert room_name == "room-abc"
    assert tenant.slug == "dev"
    assert meta == {}
    assert bridge_factory is client._sentinel_factory
    # drain in-flight task so the fixture teardown doesn't warn — production
    # code keys _in_flight_rooms as f"{tenant.id}:{room_name}", not a bare
    # room name (Fix 5).
    task = livekit_routes._in_flight_rooms.get(f"{tenant.id}:room-abc")
    if task is not None:
        await task


async def test_duplicate_delivery_spawns_one_runner(client, monkeypatch):
    recorded: list = []
    started = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        livekit_routes, "run_call", _spy_run_call(recorded, event=started, release=release))
    body = _body(room="room-dup")
    tok = _sign(body)

    resp1 = await _post(client, body, tok)
    assert resp1.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=2.0)

    resp2 = await _post(client, body, tok)
    assert resp2.status_code == 200
    assert len(recorded) == 1

    tenant = recorded[0][0]
    key = f"{tenant.id}:room-dup"   # composite key (Fix 5) — bare "room-dup" is never in the dict
    release.set()
    task = livekit_routes._in_flight_rooms.get(key)
    if task is not None:
        await task
    await asyncio.sleep(0)   # let the finally: block run
    assert key not in livekit_routes._in_flight_rooms


async def test_room_is_reusable_after_the_runner_finishes(client, monkeypatch):
    recorded: list = []
    started = asyncio.Event()
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded, event=started))
    body = _body(room="room-reuse")
    tok = _sign(body)

    await _post(client, body, tok)
    await asyncio.wait_for(started.wait(), timeout=2.0)
    tenant = recorded[0][0]
    key = f"{tenant.id}:room-reuse"   # composite key (Fix 5)
    task = livekit_routes._in_flight_rooms.get(key)
    if task is not None:
        await task

    started.clear()
    resp3 = await _post(client, body, tok)
    assert resp3.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=2.0)
    assert len(recorded) == 2
    task = livekit_routes._in_flight_rooms.get(key)
    if task is not None:
        await task


async def test_over_concurrency_cap_returns_200_without_spawning(client, monkeypatch, caplog):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))

    async def _one_active(session, tenant_id):
        return 5   # matches the fixture's max_concurrent_calls=5

    monkeypatch.setattr(livekit_routes, "count_active_calls", _one_active)
    body = _body(room="room-cap")
    with caplog.at_level("WARNING", logger="src.api.livekit_routes"):
        resp = await _post(client, body, _sign(body))
    assert resp.status_code == 200
    assert recorded == []
    assert any("max concurrent calls" in m for m in caplog.messages)


async def test_no_bridge_factory_returns_503(client, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    livekit_routes.set_livekit_bridge_factory(None)
    body = _body(room="room-no-factory")
    resp = await _post(client, body, _sign(body))
    assert resp.status_code == 503
    assert recorded == []


async def test_run_call_failure_releases_the_room(client, monkeypatch):
    recorded: list = []
    started = asyncio.Event()
    monkeypatch.setattr(
        livekit_routes, "run_call", _spy_run_call(recorded, event=started, raise_error=True))
    body = _body(room="room-fails")
    resp = await _post(client, body, _sign(body))
    assert resp.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=2.0)
    tenant = recorded[0][0]
    key = f"{tenant.id}:room-fails"   # composite key (Fix 5)
    task = livekit_routes._in_flight_rooms.get(key)
    if task is not None:
        await task   # must not raise out of the task
    await asyncio.sleep(0)
    assert key not in livekit_routes._in_flight_rooms


# --- _extract_vox_metadata --------------------------------------------------


def _event(*, room_meta="", part_meta="", attrs=None) -> WebhookEvent:
    return WebhookEvent(
        event="participant_joined",
        room=Room(name="r1", metadata=room_meta),
        participant=ParticipantInfo(
            identity="sip_x", kind=ParticipantInfo.Kind.SIP,
            metadata=part_meta, attributes=attrs or {}),
    )


def test_extract_metadata_from_room_metadata():
    ev = _event(
        room_meta=json.dumps({"vox": {"campaign_id": "c1", "voice": "Kore"}}),
        part_meta=json.dumps({"vox": {"campaign_id": "ignored"}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert meta == {"campaign_id": "c1", "voice": "Kore"}


def test_extract_metadata_from_participant_metadata():
    ev = _event(part_meta=json.dumps({"vox": {"campaign_id": "c2"}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert meta == {"campaign_id": "c2"}


def test_extract_metadata_from_flat_attributes():
    ev = _event(attrs={
        "vox.campaign_id": "c1", "vox.voice": "Kore",
        "vox.lead_name": "Asha", "vox.lead_gender": "female",
    })
    meta = livekit_routes._extract_vox_metadata(ev)
    assert meta == {
        "campaign_id": "c1", "voice": "Kore",
        "lead": {"name": "Asha", "gender": "female"},
    }


# --- Fix 4: metadata type validation (trust boundary) -----------------------
#
# Reproduced crash: {"campaign_id": {"a": 1}, "voice": 123, "lead": "Rohit"}
# threw AttributeErrors deep inside bootstrap.py (voice.strip() on an int,
# lead.get("name") on a string), caught by a generic handler, caller ends up
# on a silent/dead line. These fields must be dropped at the trust boundary
# (_extract_vox_metadata), not passed through untyped.


def test_extract_metadata_drops_non_string_campaign_id():
    ev = _event(room_meta=json.dumps({"vox": {"campaign_id": {"a": 1}, "voice": "Kore"}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert "campaign_id" not in meta
    assert meta == {"voice": "Kore"}


def test_extract_metadata_drops_non_string_voice():
    ev = _event(room_meta=json.dumps({"vox": {"campaign_id": "c1", "voice": 123}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert "voice" not in meta
    assert meta == {"campaign_id": "c1"}


def test_extract_metadata_drops_non_string_call_ref():
    ev = _event(room_meta=json.dumps({"vox": {"call_ref": ["not", "a", "string"]}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert meta == {}


def test_extract_metadata_drops_non_dict_lead():
    ev = _event(room_meta=json.dumps({"vox": {"campaign_id": "c1", "lead": "Rohit"}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert "lead" not in meta
    assert meta == {"campaign_id": "c1"}


def test_extract_metadata_drops_malformed_nested_lead_fields():
    ev = _event(room_meta=json.dumps(
        {"vox": {"lead": {"name": 42, "gender": "female"}}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert meta == {"lead": {"gender": "female"}}   # name dropped, gender kept


def test_extract_metadata_drops_malformed_lead_gender():
    ev = _event(room_meta=json.dumps(
        {"vox": {"lead": {"name": "Asha", "gender": {"nested": "object"}}}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert meta == {"lead": {"name": "Asha"}}   # gender dropped, name kept


def test_extract_metadata_all_malformed_fields_dropped_together():
    """The exact reproduction from the bug report: every field malformed at
    once must still return a clean, usable dict — not crash and not leave any
    malformed value in place."""
    ev = _event(room_meta=json.dumps(
        {"vox": {"campaign_id": {"a": 1}, "voice": 123, "lead": "Rohit"}}))
    meta = livekit_routes._extract_vox_metadata(ev)
    assert meta == {}


async def test_webhook_with_malformed_metadata_spawns_call_with_sanitized_meta(client, monkeypatch):
    """End-to-end: a call with malformed vox metadata must still run (falling
    back to defaults) instead of crashing into the silent-failure path."""
    recorded: list = []
    started = asyncio.Event()
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded, event=started))
    body = _body(room="room-malformed", room_meta=json.dumps(
        {"vox": {"campaign_id": {"a": 1}, "voice": 123, "lead": "Rohit"}}))
    resp = await _post(client, body, _sign(body))
    assert resp.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=2.0)
    assert len(recorded) == 1
    _, room_name, meta, _ = recorded[0]
    assert room_name == "room-malformed"
    assert meta == {}   # every malformed field dropped — no crash, sane fallback
    tenant = recorded[0][0]
    task = livekit_routes._in_flight_rooms.get(f"{tenant.id}:room-malformed")
    if task is not None:
        await task


def test_extract_metadata_empty_returns_empty_dict():
    ev = _event()
    assert livekit_routes._extract_vox_metadata(ev) == {}


def test_extract_metadata_ignores_malformed_and_unnamespaced_json():
    ev = _event(room_meta="not json")
    assert livekit_routes._extract_vox_metadata(ev) == {}
    ev2 = _event(room_meta=json.dumps({"sip": {}}))
    assert livekit_routes._extract_vox_metadata(ev2) == {}


# --- routing ----------------------------------------------------------------


def test_webhook_route_is_reachable_under_api_v1():
    from src.api import api_router
    app = FastAPI()
    app.include_router(api_router)
    paths = {r.path for r in app.routes}
    assert "/api/v1/telephony/livekit/webhook/{tenant_slug}" in paths


# --- CRM-level credential fallback (config_tenant.resolve_livekit_creds) ----
#
# The default `client` fixture's tenant has a fully-configured tenant-level
# LiveKit override, so it never exercises the CRM-level DB fallback (and its
# get_db_session override yields a bare `object()`, which would blow up if it
# ever tried). These tests build their own app + real sqlite DB so the CRM
# fallback branch is actually exercised end to end.

_CRM_LK_KEY = "crmdevkey"
_CRM_LK_SECRET = "crmdevsecret-crmdevsecret-secret-32"  # >=32 bytes


@pytest.fixture
async def crm_only_client(monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x",
                    livekit_url="wss://crm.example/rtc"))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_key",
                          value_encrypted=crypto.encrypt(_CRM_LK_KEY)))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_secret",
                          value_encrypted=crypto.encrypt(_CRM_LK_SECRET)))
        await db.commit()

    # No tenant-level livekit config at all — crm_id is the only link.
    settings = TenantSettings(
        id="t_crmdev", slug="crmdev", name="CrmDev", max_concurrent_calls=5,
        crm_id="betstudio",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig()),
    )
    register_tenant_for_test(settings)

    async def _session_override():
        async with sm() as session:
            yield session

    async def _no_active(session, tenant_id):
        return 0

    livekit_routes._in_flight_rooms.clear()
    sentinel = object()
    livekit_routes.set_livekit_bridge_factory(sentinel)

    app = FastAPI()
    app.include_router(livekit_routes.router)
    app.dependency_overrides[get_db_session] = _session_override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c._sentinel_factory = sentinel
        c._no_active = _no_active
        yield c

    livekit_routes.set_livekit_bridge_factory(None)
    livekit_routes._in_flight_rooms.clear()
    set_tenant_resolver(None)
    await engine.dispose()


async def test_webhook_crm_level_fallback_verifies_and_spawns(crm_only_client, monkeypatch):
    """No tenant-level LiveKit config; tenant.crm_id points at a Crm whose
    livekit_url + encrypted CrmSecret key/secret are the ONLY credentials
    configured anywhere — the webhook must resolve + verify against those via
    resolve_livekit_creds's CRM-level fallback."""
    recorded: list = []
    started = asyncio.Event()
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded, event=started))
    monkeypatch.setattr(livekit_routes, "count_active_calls", crm_only_client._no_active)

    body = _body(room="room-crmdev")
    digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
    token = api.AccessToken(_CRM_LK_KEY, _CRM_LK_SECRET).with_sha256(digest).to_jwt()

    resp = await crm_only_client.post(
        "/telephony/livekit/webhook/crmdev", content=body,
        headers={"Authorization": token, "Content-Type": "application/webhook+json"})
    assert resp.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=2.0)
    assert len(recorded) == 1
    tenant = recorded[0][0]
    assert tenant.slug == "crmdev"
    assert recorded[0][3] is crm_only_client._sentinel_factory

    task = livekit_routes._in_flight_rooms.get(f"{tenant.id}:room-crmdev")
    if task is not None:
        await task


async def test_webhook_crm_level_wrong_secret_401(crm_only_client, monkeypatch):
    """Signing with a key/secret pair that doesn't match the CRM's stored
    secrets must still 401 — the fallback resolves real values, it doesn't
    bypass verification."""
    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    monkeypatch.setattr(livekit_routes, "count_active_calls", crm_only_client._no_active)

    body = _body(room="room-crmdev-bad")
    digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
    token = api.AccessToken(
        _CRM_LK_KEY, "a-totally-different-secret-32-bytes").with_sha256(digest).to_jwt()

    resp = await crm_only_client.post(
        "/telephony/livekit/webhook/crmdev", content=body,
        headers={"Authorization": token, "Content-Type": "application/webhook+json"})
    assert resp.status_code == 401
    assert recorded == []


async def test_webhook_no_tenant_and_no_crm_config_401(monkeypatch):
    """crm_id unset and no tenant-level LiveKit config — resolve_livekit_creds
    returns None, and the webhook must 401 (never spawn a run_call) rather
    than crash."""
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    settings = TenantSettings(
        id="t_nocfg", slug="nocfg", name="NoCfg",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig()),
    )
    register_tenant_for_test(settings)

    async def _session_override():
        yield object()

    recorded: list = []
    monkeypatch.setattr(livekit_routes, "run_call", _spy_run_call(recorded))
    livekit_routes._in_flight_rooms.clear()
    app = FastAPI()
    app.include_router(livekit_routes.router)
    app.dependency_overrides[get_db_session] = _session_override
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            body = _body(room="room-nocfg")
            token = _sign(body)
            resp = await c.post(
                "/telephony/livekit/webhook/nocfg", content=body,
                headers={"Authorization": token, "Content-Type": "application/webhook+json"})
        assert resp.status_code == 401
        assert recorded == []
    finally:
        livekit_routes._in_flight_rooms.clear()
        set_tenant_resolver(None)
