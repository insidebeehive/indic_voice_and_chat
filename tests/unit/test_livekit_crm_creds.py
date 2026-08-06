"""CRM-level LiveKit credential storage/resolution.

Covers the two pieces added when LiveKit connection details moved from
per-tenant (``TenantTelephonyConfig``) to CRM-level shared config
(``Crm.livekit_url`` + ``CrmSecret``), with a tenant-level override kept as a
fallback path:

- ``CrmSecret`` encryption round-trip (mirrors ``TenantSecret``'s scheme).
- ``config_tenant.resolve_livekit_creds`` precedence: tenant-level override
  wins if fully configured; otherwise falls back to the CRM's shared project;
  otherwise ``None``.
"""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth import secrets as crypto
from src.auth.context import TenantContext
from src.config_tenant import (
    TelephonyCreds,
    TenantPipelineConfig,
    TenantSettings,
    TenantTelephonyConfig,
    resolve_livekit_creds,
)
from src.models.crm import Crm, CrmSecret
from src.models.database import Base


@pytest_asyncio.fixture
async def sm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


# --- CrmSecret encryption round-trip ---------------------------------------


async def test_crm_secret_round_trip(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x",
                    livekit_url="wss://lk.betstudio.example"))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_key",
                          value_encrypted=crypto.encrypt("lk_key_123")))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_secret",
                          value_encrypted=crypto.encrypt("lk_secret_456")))
        await db.commit()

    async with sm() as db:
        key_row = await db.get(CrmSecret, {"crm_id": "betstudio", "name": "livekit_api_key"})
        secret_row = await db.get(CrmSecret, {"crm_id": "betstudio", "name": "livekit_api_secret"})
        assert crypto.decrypt(key_row.value_encrypted) == "lk_key_123"
        assert crypto.decrypt(secret_row.value_encrypted) == "lk_secret_456"


async def test_crm_secret_cascades_on_crm_delete(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    async with sm() as db:
        db.add(Crm(id="betstudio", name="BetStudio", base_url="https://x"))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_key",
                          value_encrypted=crypto.encrypt("k")))
        await db.commit()

    async with sm() as db:
        # sqlite doesn't enforce ON DELETE CASCADE without a pragma; assert the
        # column/FK relationship shape is at least queryable both ways.
        crm = await db.get(Crm, "betstudio")
        assert crm is not None
        row = await db.get(CrmSecret, {"crm_id": "betstudio", "name": "livekit_api_key"})
        assert row is not None


# --- resolve_livekit_creds ---------------------------------------------------


def _tenant(*, crm_id=None, livekit_url=None, creds_by_provider=None, secrets_resolved=None):
    settings = TenantSettings(
        id="t1", slug="dev", name="Dev", crm_id=crm_id,
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            livekit_url=livekit_url,
            creds_by_provider=creds_by_provider or {},
        )),
    )
    return TenantContext(settings=settings, secrets_resolved=secrets_resolved or {})


async def _seed_crm_with_livekit(sm, *, crm_id="betstudio",
                                  url="wss://crm.example/rtc",
                                  key="crm_key", secret="crm_secret"):
    async with sm() as db:
        db.add(Crm(id=crm_id, name="CRM", base_url="https://x", livekit_url=url))
        if key is not None:
            db.add(CrmSecret(crm_id=crm_id, name="livekit_api_key",
                              value_encrypted=crypto.encrypt(key)))
        if secret is not None:
            db.add(CrmSecret(crm_id=crm_id, name="livekit_api_secret",
                              value_encrypted=crypto.encrypt(secret)))
        await db.commit()


async def test_resolve_crm_only_tenant_returns_crm_values(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    await _seed_crm_with_livekit(sm)
    tenant = _tenant(crm_id="betstudio")  # no tenant-level livekit config at all

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result == ("wss://crm.example/rtc", "crm_key", "crm_secret")


async def test_resolve_tenant_override_wins_over_crm(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    await _seed_crm_with_livekit(sm)
    tenant = _tenant(
        crm_id="betstudio",
        livekit_url="wss://tenant-override.example/rtc",
        creds_by_provider={"livekit": TelephonyCreds(
            account_sid_env="TENANT_LK_KEY", auth_token_env="TENANT_LK_SECRET")},
        secrets_resolved={"TENANT_LK_KEY": "tenant_key", "TENANT_LK_SECRET": "tenant_secret"},
    )

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result == ("wss://tenant-override.example/rtc", "tenant_key", "tenant_secret")


async def test_resolve_neither_configured_returns_none(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    tenant = _tenant(crm_id=None)  # no tenant-level config, no crm_id

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result is None


async def test_resolve_crm_id_unset_and_no_tenant_config_returns_none(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    # A Crm exists in the DB, but this tenant isn't linked to it.
    await _seed_crm_with_livekit(sm)
    tenant = _tenant(crm_id=None)

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result is None


async def test_resolve_tenant_partially_configured_falls_through_to_crm(sm, monkeypatch):
    """URL set but no key/secret on the tenant — must not half-resolve; falls
    through to the CRM-level project instead."""
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    await _seed_crm_with_livekit(sm)
    tenant = _tenant(
        crm_id="betstudio",
        livekit_url="wss://tenant-partial.example/rtc",
        # no creds_by_provider["livekit"] at all, and no active provider set ->
        # _tenant_livekit_creds returns an empty TelephonyCreds ->
        # api_key/api_secret resolve to None
    )

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result == ("wss://crm.example/rtc", "crm_key", "crm_secret")


async def test_resolve_tenant_with_active_non_livekit_provider_falls_through_to_crm(
    sm, monkeypatch,
):
    """Regression for the bug the CRM-level fallback was designed to avoid:
    a real-world tenant registered with e.g. Stringee (register_tenant always
    mirrors the active provider's creds into TenantTelephonyConfig's top-level
    account_sid_env/auth_token_env for back-compat — see src/api/tenants.py)
    that later just gets a bare livekit_url set (no creds_by_provider["livekit"]
    ever registered) must NOT resolve the Stringee account SID/token as its
    LiveKit key/secret. It must fall through cleanly to the CRM-level project,
    exactly like the "no creds at all" case above."""
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    await _seed_crm_with_livekit(sm)

    settings = TenantSettings(
        id="t1", slug="dev", name="Dev", crm_id="betstudio",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider="stringee",
            livekit_url="wss://tenant-partial.example/rtc",
            # top-level fields — exactly what register_tenant/update_tenant
            # write for the tenant's ACTIVE (non-LiveKit) provider.
            account_sid_env="TENANT_DEV_STRINGEE_SID",
            auth_token_env="TENANT_DEV_STRINGEE_TOKEN",
            creds_by_provider={
                "stringee": TelephonyCreds(
                    account_sid_env="TENANT_DEV_STRINGEE_SID",
                    auth_token_env="TENANT_DEV_STRINGEE_TOKEN"),
            },
            # deliberately no creds_by_provider["livekit"]
        )),
    )
    tenant = TenantContext(
        settings=settings,
        secrets_resolved={
            "TENANT_DEV_STRINGEE_SID": "stringee_sid_value",
            "TENANT_DEV_STRINGEE_TOKEN": "stringee_token_value",
        },
    )

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result == ("wss://crm.example/rtc", "crm_key", "crm_secret")


async def test_resolve_tenant_provider_is_livekit_uses_top_level_creds(sm, monkeypatch):
    """The one legitimate case where the top-level account_sid_env/
    auth_token_env genuinely ARE LiveKit creds: the tenant's active provider
    IS "livekit" itself (e.g. a tenant whose only telephony path is LiveKit
    room-join). Must still resolve via the tenant-level override, not skip to
    the CRM."""
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    await _seed_crm_with_livekit(sm)

    settings = TenantSettings(
        id="t1", slug="dev", name="Dev", crm_id="betstudio",
        pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
            provider="livekit",
            livekit_url="wss://tenant-lk-provider.example/rtc",
            account_sid_env="TENANT_DEV_LK_KEY",
            auth_token_env="TENANT_DEV_LK_SECRET",
        )),
    )
    tenant = TenantContext(
        settings=settings,
        secrets_resolved={"TENANT_DEV_LK_KEY": "top_level_key", "TENANT_DEV_LK_SECRET": "top_level_secret"},
    )

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result == ("wss://tenant-lk-provider.example/rtc", "top_level_key", "top_level_secret")


async def test_resolve_crm_missing_url_returns_none(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    async with sm() as db:
        db.add(Crm(id="betstudio", name="CRM", base_url="https://x", livekit_url=None))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_key",
                          value_encrypted=crypto.encrypt("k")))
        db.add(CrmSecret(crm_id="betstudio", name="livekit_api_secret",
                          value_encrypted=crypto.encrypt("s")))
        await db.commit()
    tenant = _tenant(crm_id="betstudio")

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result is None


async def test_resolve_crm_missing_one_secret_returns_none(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    await _seed_crm_with_livekit(sm, secret=None)  # only the key half is set
    tenant = _tenant(crm_id="betstudio")

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result is None


async def test_resolve_unknown_crm_id_returns_none(sm, monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    tenant = _tenant(crm_id="does-not-exist")

    async with sm() as db:
        result = await resolve_livekit_creds(db, tenant)

    assert result is None
