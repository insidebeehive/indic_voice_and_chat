from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.db_resolver import DbTenantResolver
from src.models import Base
from src.models.crm import Crm
from src.models.tenant import Tenant, TenantApiKey, TenantPhoneNumber, TenantSecret


@pytest_asyncio.fixture
async def sm(monkeypatch):
    monkeypatch.setenv(crypto.VOX_SECRET_KEY_ENV, crypto.generate_key())
    crypto.reset_cache_for_tests()
    eng = create_async_engine("sqlite+aiosqlite://")
    async with eng.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(eng, expire_on_commit=False)
    await eng.dispose()
    crypto.reset_cache_for_tests()


async def _seed_one(sm):
    async with sm() as s:
        s.add(Tenant(
            id="t_acme", slug="acme", name="Acme", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=3,
            pipeline_config={
                "mode": "layered",
                "tts": {"provider": "sarvam", "voice_id": "anushka", "api_key_env": "SARVAM_API_KEY"},
                "telephony": {"provider": "twilio", "from_number": "+1555",
                              "account_sid_env": "twilio_sid", "auth_token_env": "twilio_token"},
            }))
        s.add(TenantPhoneNumber(phone_number="+1555", tenant_id="t_acme", provider="twilio"))
        s.add(TenantApiKey(token_hash=hash_api_token("tok-abc"), tenant_id="t_acme", label="x"))
        s.add(TenantSecret(tenant_id="t_acme", name="twilio_sid",
                           value_encrypted=crypto.encrypt("AC-real-sid")))
        await s.commit()


@pytest.mark.asyncio
async def test_resolver_rebuilds_settings_and_splits_secrets(sm, monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "master-sarvam")
    await _seed_one(sm)

    r = DbTenantResolver(sm)
    assert await r.reload() == 1

    ctx = await r.resolve_by_slug("acme")
    assert ctx is not None
    assert ctx.id == "t_acme" and ctx.settings.timezone == "Asia/Kolkata"
    assert ctx.settings.max_concurrent_calls == 3
    assert ctx.settings.pipeline.telephony.provider == "twilio"
    assert ctx.settings.pipeline.tts.voice_id == "anushka"

    # telephony key resolves from the DB (decrypted); master keys fall back to env
    assert ctx.secret("twilio_sid") == "AC-real-sid"
    assert ctx.secret("SARVAM_API_KEY") == "master-sarvam"

    assert (await r.resolve_by_token(hash_api_token("tok-abc"))).slug == "acme"
    assert (await r.resolve_by_phone_number("+1555")).slug == "acme"
    assert await r.resolve_by_slug("nope") is None


@pytest.mark.asyncio
async def test_resolver_denormalizes_crm_prompt_pack(sm):
    """crm_id=tenant.crm_id has a companion denormalization onto
    settings.prompt_pack: a tenant linked to a Crm with prompt_pack='betting'
    gets 'betting'; a tenant with no linked CRM at all falls back to
    'generic' rather than raising or leaving it unset."""
    async with sm() as s:
        s.add(Crm(id="betstudio", name="BetStudio", base_url="https://x", prompt_pack="betting"))
        s.add(Tenant(
            id="t_linked", slug="linked", name="Linked", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, crm_id="betstudio", pipeline_config={}))
        s.add(Tenant(
            id="t_unlinked", slug="unlinked", name="Unlinked", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, pipeline_config={}))
        await s.commit()

    r = DbTenantResolver(sm)
    assert await r.reload() == 2

    linked = await r.resolve_by_slug("linked")
    assert linked.settings.prompt_pack == "betting"

    unlinked = await r.resolve_by_slug("unlinked")
    assert unlinked.settings.prompt_pack == "generic"


@pytest.mark.asyncio
async def test_resolver_denormalizes_crm_pronunciation_overrides(sm):
    """crm_id=tenant.crm_id has a companion denormalization onto
    settings.pronunciation_overrides (mirrors prompt_pack): a tenant linked to
    a Crm with pronunciation_overrides set gets that dict; a tenant with no
    linked CRM, or linked to a Crm with no overrides set, gets None (that
    CRM's TTS then uses only the generic DEFAULT_PRONUNCIATIONS default)."""
    async with sm() as s:
        s.add(Crm(
            id="betstudio", name="BetStudio", base_url="https://x",
            pronunciation_overrides={"Casino": "कसीनो"},
        ))
        s.add(Crm(id="plain_crm", name="Plain", base_url="https://y"))
        s.add(Tenant(
            id="t_linked", slug="linked", name="Linked", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, crm_id="betstudio", pipeline_config={}))
        s.add(Tenant(
            id="t_linked_no_overrides", slug="linked-no-overrides", name="LinkedNoOverrides",
            status="active", timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, crm_id="plain_crm", pipeline_config={}))
        s.add(Tenant(
            id="t_unlinked", slug="unlinked", name="Unlinked", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, pipeline_config={}))
        await s.commit()

    r = DbTenantResolver(sm)
    assert await r.reload() == 3

    linked = await r.resolve_by_slug("linked")
    assert linked.settings.pronunciation_overrides == {"Casino": "कसीनो"}

    linked_no_overrides = await r.resolve_by_slug("linked-no-overrides")
    assert linked_no_overrides.settings.pronunciation_overrides is None

    unlinked = await r.resolve_by_slug("unlinked")
    assert unlinked.settings.pronunciation_overrides is None


def test_secret_optional_tenant_then_env_then_none(monkeypatch):
    """Optional secrets (e.g. webhook signing) resolve from the decrypted per-tenant
    secrets first, then process env, and return None (NOT raise) when unset."""
    from src.auth.context import TenantContext
    from src.config_tenant import TenantSettings

    ctx = TenantContext(
        settings=TenantSettings(id="t1", slug="t1", name="T1"),
        secrets_resolved={"CRM_SIGNING_SECRET": "tenant-secret"})
    assert ctx.secret_optional("CRM_SIGNING_SECRET") == "tenant-secret"  # per-tenant wins
    monkeypatch.setenv("ENV_ONLY_SECRET", "from-env")
    assert ctx.secret_optional("ENV_ONLY_SECRET") == "from-env"          # env fallback
    assert ctx.secret_optional("MISSING_SECRET") is None                 # no raise (unlike secret())
    assert ctx.secret_optional(None) is None
