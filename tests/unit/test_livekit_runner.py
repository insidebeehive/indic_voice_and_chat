"""Unit tests for src/api/livekit_runner.py — the real LiveKit SDK glue.

Zero network access: livekit.rtc / livekit.api are fully replaced (monkeypatch
on the module-level names ``livekit_runner.rtc`` / ``livekit_runner.api``) with
plain fakes, matching this repo's SDK-boundary mocking convention (see
tests/unit/test_dev_console.py's module-level monkeypatch.setattr style).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import livekit_runner
from src.auth import secrets as crypto
from src.auth.context import TenantContext
from src.config_tenant import (
    TelephonyCreds,
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
)
from src.models.crm import Crm, CrmSecret
from src.models.database import Base


# --- fakes: livekit.rtc ------------------------------------------------


class _FakeTrackKind:
    KIND_AUDIO = 1
    KIND_VIDEO = 2


class _FakeTrackSource:
    SOURCE_MICROPHONE = 2


class _FakeTrack:
    def __init__(self, kind=_FakeTrackKind.KIND_AUDIO):
        self.kind = kind


class _FakeAudioStream:
    def __init__(self, track, sample_rate=16000, num_channels=1):
        self.track = track
        self.sample_rate = sample_rate
        self.num_channels = num_channels


class _FakeAudioSource:
    def __init__(self, sample_rate=24000, num_channels=1):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.captured = []

    async def capture_frame(self, frame):
        self.captured.append(frame)

    def clear_queue(self):
        pass


class _FakeLocalAudioTrack:
    def __init__(self, name, source):
        self.name = name
        self.source = source

    @staticmethod
    def create_audio_track(name, source):
        return _FakeLocalAudioTrack(name, source)


class _FakeTrackPublishOptions:
    def __init__(self, source=None):
        self.source = source


class _FakeRoomOptions:
    def __init__(self, auto_subscribe=True):
        self.auto_subscribe = auto_subscribe


class _FakeAudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class _FakeLocalParticipant:
    def __init__(self):
        self.published = []

    async def publish_track(self, track, options):
        self.published.append((track, options))


class _FakeRoom:
    """Configurable fake ``rtc.Room``.

    ``fire_track_subscribed`` controls whether ``connect()`` synchronously
    invokes the registered ``track_subscribed`` handler (simulating the SIP
    participant's audio track being subscribed) — False reproduces the
    "caller never answered" timeout path.
    """

    def __init__(self, *, fire_track_subscribed=True):
        self._handlers: dict = {}
        self.local_participant = _FakeLocalParticipant()
        self.disconnect_calls = 0
        self.connect_calls = []
        self._fire_track_subscribed = fire_track_subscribed

    def on(self, event, callback):
        self._handlers[event] = callback
        return callback

    async def connect(self, url, token, options=None):
        self.connect_calls.append((url, token, options))
        if self._fire_track_subscribed:
            cb = self._handlers.get("track_subscribed")
            if cb is not None:
                cb(_FakeTrack(), object(), object())

    async def disconnect(self):
        self.disconnect_calls += 1


def _fake_rtc_module(*, fire_track_subscribed=True):
    room_instance = _FakeRoom(fire_track_subscribed=fire_track_subscribed)
    return SimpleNamespace(
        Room=lambda: room_instance,
        AudioStream=_FakeAudioStream,
        AudioSource=_FakeAudioSource,
        LocalAudioTrack=_FakeLocalAudioTrack,
        TrackPublishOptions=_FakeTrackPublishOptions,
        TrackSource=_FakeTrackSource,
        TrackKind=_FakeTrackKind,
        RoomOptions=_FakeRoomOptions,
        AudioFrame=_FakeAudioFrame,
    ), room_instance


# --- fakes: livekit.api --------------------------------------------------


class _FakeVideoGrants:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAccessToken:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.identity = None
        self.grants = None

    def with_identity(self, identity):
        self.identity = identity
        return self

    def with_grants(self, grants):
        self.grants = grants
        return self

    def to_jwt(self):
        return f"fake-jwt:{self.identity}:{self.grants.kwargs.get('room')}"


def _fake_api_module():
    return SimpleNamespace(AccessToken=_FakeAccessToken, VideoGrants=_FakeVideoGrants)


# --- fakes: tenant / bridge / db -----------------------------------------


def _tenant(livekit_url="wss://fake.example/rtc"):
    """Tenant-level LiveKit override fully configured — this is the override
    path (config_tenant.resolve_livekit_creds's first branch), which must
    keep working unchanged now that a CRM-level fallback also exists."""
    settings = TenantSettings(
        id="t1", slug="dev", name="Dev",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            livekit_url=livekit_url,
            creds_by_provider={
                "livekit": TelephonyCreds(
                    account_sid_env="TEST_LK_KEY", auth_token_env="TEST_LK_SECRET"),
            },
        )),
    )
    return TenantContext(
        settings=settings,
        secrets_resolved={"TEST_LK_KEY": "key123", "TEST_LK_SECRET": "secret456"})


def _crm_only_tenant(crm_id="betstudio"):
    """No tenant-level LiveKit config at all — resolution must fall through
    entirely to the CRM-level project via resolve_livekit_creds."""
    settings = TenantSettings(
        id="t1", slug="dev", name="Dev", crm_id=crm_id,
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig()),
    )
    return TenantContext(settings=settings)


class _FakeBridge:
    def __init__(self):
        self.ran = False

    async def run(self):
        self.ran = True


def _bridge_factory(holder: dict):
    async def factory(tenant, room_name, meta):
        holder["factory_args"] = (tenant, room_name, meta)

        async def build(*, audio_stream, audio_source, frame_factory, on_hangup=None):
            bridge = _FakeBridge()
            holder["bridge"] = bridge
            holder["build_kwargs"] = dict(
                audio_stream=audio_stream, audio_source=audio_source,
                frame_factory=frame_factory, on_hangup=on_hangup)
            return bridge

        return build

    return factory


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeDB:
    def __init__(self, existing_row=None, raise_on_execute=False):
        self.existing_row = existing_row
        self.raise_on_execute = raise_on_execute
        self.last_stmt = None
        self.commits = 0

    async def execute(self, stmt):
        self.last_stmt = stmt
        if self.raise_on_execute:
            raise RuntimeError("db exploded")
        return _FakeResult(self.existing_row)

    async def commit(self):
        self.commits += 1


class _FakeSessionCtx:
    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *exc):
        return False


def _sessionmaker_for(db):
    return lambda: _FakeSessionCtx(db)


# --- tests: _mint_token ----------------------------------------------------


def test_mint_token_builds_jwt_with_room_and_identity(monkeypatch):
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())
    token = livekit_runner._mint_token(
        "key123", "secret456", room_name="room-abc", identity="vox-agent-room-abc")
    assert isinstance(token, str)
    assert token == "fake-jwt:vox-agent-room-abc:room-abc"


def test_mint_token_grants_room_join_and_publish_subscribe(monkeypatch):
    captured = {}

    class _CapturingAccessToken(_FakeAccessToken):
        def with_grants(self, grants):
            captured["grants"] = grants.kwargs
            return super().with_grants(grants)

    monkeypatch.setattr(livekit_runner, "api",
                         SimpleNamespace(AccessToken=_CapturingAccessToken, VideoGrants=_FakeVideoGrants))
    livekit_runner._mint_token("k", "s", room_name="r1", identity="id1")
    assert captured["grants"] == {
        "room_join": True, "room": "r1", "can_publish": True, "can_subscribe": True}


# --- tests: run_call happy path -------------------------------------------


@pytest.mark.asyncio
async def test_run_call_happy_path_builds_and_runs_bridge(monkeypatch):
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    holder: dict = {}
    tenant = _tenant()

    await livekit_runner.run_call(
        tenant, "room-1", {}, bridge_factory=_bridge_factory(holder))

    assert holder["bridge"].ran is True
    # audio_stream/audio_source/frame_factory were all real fake objects, not None
    assert isinstance(holder["build_kwargs"]["audio_stream"], _FakeAudioStream)
    assert isinstance(holder["build_kwargs"]["audio_source"], _FakeAudioSource)
    assert callable(holder["build_kwargs"]["frame_factory"])
    # room published our outbound track and was disconnected in the finally block
    assert len(room.local_participant.published) == 1
    assert room.disconnect_calls == 1


@pytest.mark.asyncio
async def test_run_call_frame_factory_matches_audio_frame_shape(monkeypatch):
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    holder: dict = {}
    await livekit_runner.run_call(
        _tenant(), "room-1", {}, bridge_factory=_bridge_factory(holder))

    frame_factory = holder["build_kwargs"]["frame_factory"]
    pcm = b"\x00\x01" * 480  # 480 int16 samples, mono
    frame = frame_factory(pcm, 24000, 1)
    assert isinstance(frame, _FakeAudioFrame)
    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert frame.samples_per_channel == 480


# --- tests: run_call timeout path -----------------------------------------


@pytest.mark.asyncio
async def test_run_call_timeout_builds_no_bridge_and_disconnects(monkeypatch):
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=False)  # caller never subscribes
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())
    monkeypatch.setattr(livekit_runner, "_JOIN_TIMEOUT_S", 0.05)

    holder: dict = {}
    factory_invocations = []

    async def factory(tenant, room_name, meta):
        factory_invocations.append(True)
        raise AssertionError("bridge_factory must not be called on timeout")

    # Should return cleanly, no exception, no bridge built.
    await livekit_runner.run_call(_tenant(), "room-timeout", {}, bridge_factory=factory)

    assert factory_invocations == []
    assert room.disconnect_calls == 1
    assert len(room.local_participant.published) == 0


# --- tests: conversation row bookkeeping ----------------------------------


@pytest.mark.asyncio
async def test_run_call_inserts_conversation_row_when_absent(monkeypatch):
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    inserted = {}
    marked = {}

    async def _fake_insert_call(db, **kwargs):
        inserted.update(kwargs)

    async def _fake_mark_answered(db, provider_call_sid, *, tenant_id=None):
        # Fix 3: call.answered now fires unconditionally for every answered
        # call, including the auto-create (never pre-registered) path — this
        # used to be a hard AssertionError-if-called here before that fix.
        marked["provider_call_sid"] = provider_call_sid
        marked["tenant_id"] = tenant_id

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)
    monkeypatch.setattr(livekit_runner, "mark_answered", _fake_mark_answered)

    db = _FakeDB(existing_row=None)
    holder: dict = {}
    await livekit_runner.run_call(
        _tenant(), "room-new", {"campaign_id": "camp1", "voice": "Kore"},
        bridge_factory=_bridge_factory(holder), sessionmaker=_sessionmaker_for(db))

    assert inserted["provider_call_sid"] == "room-new"
    assert inserted["campaign_id"] == "camp1"
    assert inserted["voice"] == "Kore"
    assert inserted["mode"] == "s2s"
    assert inserted["channel"] == "voice"
    assert inserted["extra_event_data"] == {"source": "livekit_room"}
    assert marked["provider_call_sid"] == "room-new"
    assert marked["tenant_id"] == "t1"
    assert holder["bridge"].ran is True


@pytest.mark.asyncio
async def test_run_call_marks_answered_when_row_exists(monkeypatch):
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    marked = {}

    async def _fake_insert_call(db, **kwargs):
        raise AssertionError("insert_call must not be called when a row already exists")

    async def _fake_mark_answered(db, provider_call_sid, *, tenant_id=None):
        marked["provider_call_sid"] = provider_call_sid
        marked["tenant_id"] = tenant_id

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)
    monkeypatch.setattr(livekit_runner, "mark_answered", _fake_mark_answered)

    db = _FakeDB(existing_row=SimpleNamespace(campaign_id=None))
    holder: dict = {}
    await livekit_runner.run_call(
        _tenant(), "room-existing", {}, bridge_factory=_bridge_factory(holder),
        sessionmaker=_sessionmaker_for(db))

    assert marked["provider_call_sid"] == "room-existing"
    # Fix 1: mark_answered is now tenant-scoped so a CRM-reused room name can't
    # collide across tenants.
    assert marked["tenant_id"] == "t1"
    assert holder["bridge"].ran is True


@pytest.mark.asyncio
async def test_run_call_db_failure_does_not_prevent_bridge_run(monkeypatch):
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_insert_call(db, **kwargs):
        raise AssertionError("must not reach insert_call — db.execute already raised")

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)

    db = _FakeDB(raise_on_execute=True)
    holder: dict = {}
    # Must not raise — best-effort, matches bridge_console.py's pattern.
    await livekit_runner.run_call(
        _tenant(), "room-db-down", {}, bridge_factory=_bridge_factory(holder),
        sessionmaker=_sessionmaker_for(db))

    assert holder["bridge"].ran is True
    assert room.disconnect_calls == 1


@pytest.mark.asyncio
async def test_run_call_no_sessionmaker_skips_db_entirely(monkeypatch):
    """sessionmaker=None (the default) — run_call must not touch the DB at all,
    and still build + run the bridge normally."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_insert_call(db, **kwargs):
        raise AssertionError("insert_call must not be called without a sessionmaker")

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)

    holder: dict = {}
    await livekit_runner.run_call(_tenant(), "room-no-db", {}, bridge_factory=_bridge_factory(holder))
    assert holder["bridge"].ran is True


# --- tests: Fix 1 — the row-lookup query itself is tenant-scoped -----------


@pytest.mark.asyncio
async def test_ensure_conversation_row_query_is_tenant_scoped(monkeypatch):
    """The existence-check query in ``_ensure_conversation_row`` must filter on
    tenant_id, not just provider_call_sid — a bare room-name lookup is exactly
    the cross-tenant collision bug (Fix 1). Real end-to-end tenant-scoping
    (row A untouched while row B is correctly matched) is proven against a
    real DB in tests/unit/test_call_store.py; this test pins that the SQL
    actually sent by the LiveKit runner carries the filter."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_insert_call(db, **kwargs):
        pass

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)

    db = _FakeDB(existing_row=None)
    holder: dict = {}
    await livekit_runner.run_call(
        _tenant(), "room-scoped", {}, bridge_factory=_bridge_factory(holder),
        sessionmaker=_sessionmaker_for(db))

    assert db.last_stmt is not None
    compiled = str(db.last_stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "tenant_id" in compiled
    assert "'t1'" in compiled or '"t1"' in compiled


@pytest.mark.asyncio
async def test_run_call_updates_diverged_campaign_id_on_pre_registered_row(monkeypatch):
    """Fix 3 bonus: a pre-registered row's campaign_id must be kept in sync
    with the campaign actually driving the live call (this call's own vox
    metadata), not silently left to diverge."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_mark_answered(db, provider_call_sid, *, tenant_id=None):
        pass

    monkeypatch.setattr(livekit_runner, "mark_answered", _fake_mark_answered)

    existing = SimpleNamespace(campaign_id="camp_OLD")
    db = _FakeDB(existing_row=existing)
    holder: dict = {}
    await livekit_runner.run_call(
        _tenant(), "room-diverge", {"campaign_id": "camp_NEW"},
        bridge_factory=_bridge_factory(holder), sessionmaker=_sessionmaker_for(db))

    assert existing.campaign_id == "camp_NEW"
    assert db.commits == 1


@pytest.mark.asyncio
async def test_run_call_campaign_id_untouched_when_not_diverged(monkeypatch):
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_mark_answered(db, provider_call_sid, *, tenant_id=None):
        pass

    monkeypatch.setattr(livekit_runner, "mark_answered", _fake_mark_answered)

    existing = SimpleNamespace(campaign_id="camp_SAME")
    db = _FakeDB(existing_row=existing)
    holder: dict = {}
    await livekit_runner.run_call(
        _tenant(), "room-no-diverge", {"campaign_id": "camp_SAME"},
        bridge_factory=_bridge_factory(holder), sessionmaker=_sessionmaker_for(db))

    assert existing.campaign_id == "camp_SAME"
    assert db.commits == 0


# --- tests: Fix 2 — structural error signaling on every early-exit path ----


@pytest.mark.asyncio
async def test_run_call_timeout_creates_row_before_wait_and_records_failure_outcome(monkeypatch):
    """Before Fix 2, the caller-track timeout produced ZERO CRM signal: the
    conversation row was only inserted AFTER the (successful) track wait, so a
    call that was never answered had no row to attach anything to. Now the row
    is created BEFORE the wait, and the timeout writes a diagnosable
    call.completed outcome instead of leaving the CRM with silence."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=False)  # caller never answers
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())
    monkeypatch.setattr(livekit_runner, "_JOIN_TIMEOUT_S", 0.05)

    inserted = {}
    failures = []

    async def _fake_insert_call(db, **kwargs):
        inserted.update(kwargs)

    async def _fake_record_outcome(db, provider_call_sid, **kwargs):
        failures.append((provider_call_sid, kwargs))

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)
    monkeypatch.setattr(livekit_runner, "record_outcome", _fake_record_outcome)

    async def factory(tenant, room_name, meta):
        raise AssertionError("bridge_factory must not be called on timeout")

    db = _FakeDB(existing_row=None)
    await livekit_runner.run_call(
        _tenant(), "room-timeout-signal", {}, bridge_factory=factory,
        sessionmaker=_sessionmaker_for(db))

    # Row created BEFORE the (failed) track wait — not skipped.
    assert inserted["provider_call_sid"] == "room-timeout-signal"
    # A diagnosable call.completed outcome was written instead of silence.
    assert len(failures) == 1
    sid, kwargs = failures[0]
    assert sid == "room-timeout-signal"
    assert kwargs["tenant_id"] == "t1"
    assert kwargs["status"] == "ended"
    assert kwargs["outcome"] == "no_answer"
    assert "never subscribed" in kwargs["notes"]


@pytest.mark.asyncio
async def test_run_call_mode_not_supported_records_failure_outcome(monkeypatch):
    """LiveKitModeNotSupported (tenant not in s2s mode) used to produce zero
    CRM signal — the row didn't exist yet when this fired. Now it does, and a
    diagnosable outcome is recorded before the exception is re-raised."""
    from src.bootstrap import LiveKitModeNotSupported

    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_insert_call(db, **kwargs):
        pass

    async def _fake_mark_answered(db, provider_call_sid, *, tenant_id=None):
        pass

    failures = []

    async def _fake_record_outcome(db, provider_call_sid, **kwargs):
        failures.append((provider_call_sid, kwargs))

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)
    monkeypatch.setattr(livekit_runner, "mark_answered", _fake_mark_answered)
    monkeypatch.setattr(livekit_runner, "record_outcome", _fake_record_outcome)

    async def factory(tenant, room_name, meta):
        raise LiveKitModeNotSupported("tenant not s2s")

    db = _FakeDB(existing_row=None)
    with pytest.raises(LiveKitModeNotSupported):
        await livekit_runner.run_call(
            _tenant(), "room-not-s2s", {}, bridge_factory=factory,
            sessionmaker=_sessionmaker_for(db))

    assert len(failures) == 1
    sid, kwargs = failures[0]
    assert sid == "room-not-s2s"
    assert kwargs["tenant_id"] == "t1"
    assert kwargs["outcome"] == "call_failed"
    assert "s2s" in kwargs["notes"]


@pytest.mark.asyncio
async def test_run_call_bridge_factory_crash_records_failure_outcome(monkeypatch):
    """Any other crash while resolving campaign/config or building the bridge
    (e.g. an unknown campaign_id) also used to vanish silently. Now it writes
    a diagnosable outcome (with the actual exception message) before
    re-raising."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_insert_call(db, **kwargs):
        pass

    async def _fake_mark_answered(db, provider_call_sid, *, tenant_id=None):
        pass

    failures = []

    async def _fake_record_outcome(db, provider_call_sid, **kwargs):
        failures.append((provider_call_sid, kwargs))

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)
    monkeypatch.setattr(livekit_runner, "mark_answered", _fake_mark_answered)
    monkeypatch.setattr(livekit_runner, "record_outcome", _fake_record_outcome)

    async def factory(tenant, room_name, meta):
        raise RuntimeError("unknown campaign_id and no active campaign fallback")

    db = _FakeDB(existing_row=None)
    with pytest.raises(RuntimeError, match="unknown campaign_id"):
        await livekit_runner.run_call(
            _tenant(), "room-crash", {}, bridge_factory=factory,
            sessionmaker=_sessionmaker_for(db))

    assert len(failures) == 1
    sid, kwargs = failures[0]
    assert sid == "room-crash"
    assert kwargs["tenant_id"] == "t1"
    assert kwargs["outcome"] == "call_failed"
    assert "unknown campaign_id" in kwargs["notes"]


@pytest.mark.asyncio
async def test_run_call_connect_failure_logs_loudly_and_never_touches_db(monkeypatch, caplog):
    """The one category of failure that genuinely can't produce a CRM signal
    (room.connect() itself fails, before any row can exist) must still be
    logged loudly and unmistakably, and must not touch the DB at all."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)

    async def _boom_connect(url, token, options=None):
        raise RuntimeError("network unreachable")

    room.connect = _boom_connect
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    async def _fake_insert_call(db, **kwargs):
        raise AssertionError("insert_call must not run — room never connected")

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)

    holder: dict = {}
    db = _FakeDB(existing_row=None)
    with caplog.at_level("ERROR", logger="src.api.livekit_runner"):
        with pytest.raises(RuntimeError, match="network unreachable"):
            await livekit_runner.run_call(
                _tenant(), "room-connect-fail", {}, bridge_factory=_bridge_factory(holder),
                sessionmaker=_sessionmaker_for(db))

    assert any("NO CONVERSATION ROW COULD BE CREATED" in m for m in caplog.messages)


# --- tests: CRM-level credential fallback (config_tenant.resolve_livekit_creds) --


@pytest.mark.asyncio
async def test_run_call_crm_level_fallback_resolves_and_connects(monkeypatch):
    """No tenant-level LiveKit config at all — resolution must fall through to
    the CRM-level project shared by every tenant under that CRM, and the
    runner must actually connect/mint the join token using those values."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x",
                    livekit_url="wss://crm.example/rtc"))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_key",
                          value_encrypted=crypto.encrypt("crm_key")))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_secret",
                          value_encrypted=crypto.encrypt("crm_secret")))
        await db.commit()

    async def _fake_insert_call(db, **kwargs):
        pass

    async def _fake_mark_answered(db, provider_call_sid, *, tenant_id=None):
        pass

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)
    monkeypatch.setattr(livekit_runner, "mark_answered", _fake_mark_answered)

    holder: dict = {}
    try:
        await livekit_runner.run_call(
            _crm_only_tenant(), "room-crm", {},
            bridge_factory=_bridge_factory(holder), sessionmaker=sm)

        assert holder["bridge"].ran is True
        assert room.connect_calls[0][0] == "wss://crm.example/rtc"
        assert room.connect_calls[0][1] == "fake-jwt:vox-agent-room-crm:room-crm"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_run_call_raises_when_neither_tenant_nor_crm_configured(monkeypatch):
    """No tenant-level LiveKit config and no crm_id — resolve_livekit_creds
    returns None, and run_call must fail loudly before any conversation row
    can be created (same honest-limitation category as a missing url used to
    be, pre-CRM-fallback)."""
    fake_rtc, room = _fake_rtc_module(fire_track_subscribed=True)
    monkeypatch.setattr(livekit_runner, "rtc", fake_rtc)
    monkeypatch.setattr(livekit_runner, "api", _fake_api_module())

    settings = TenantSettings(id="t1", slug="dev", name="Dev")  # no crm_id, no telephony config
    tenant = TenantContext(settings=settings)

    async def _fake_insert_call(db, **kwargs):
        raise AssertionError("must not create a row — credentials never resolved")

    monkeypatch.setattr(livekit_runner, "insert_call", _fake_insert_call)

    with pytest.raises(RuntimeError, match="no usable LiveKit credentials"):
        await livekit_runner.run_call(
            tenant, "room-no-creds", {}, bridge_factory=_bridge_factory({}))

    assert room.connect_calls == []
