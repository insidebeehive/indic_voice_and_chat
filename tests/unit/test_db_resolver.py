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


@pytest.mark.asyncio
async def test_undecryptable_secret_is_skipped_without_killing_the_reload(sm):
    """Regression for the `extra={"name": ...}` LogRecord collision: a tenant
    secret that fails to decrypt (e.g. VOX_SECRET_KEY rotated/absent) used to
    blow up the except-handler itself (`KeyError: "Attempt to overwrite
    'name' in LogRecord"`), aborting reload() for every tenant, not just the
    broken one. One good secret + one corrupt secret on the same tenant must
    still let reload() return cleanly, keeping the good secret and skipping
    the bad one."""
    async with sm() as s:
        s.add(Tenant(
            id="t_acme", slug="acme", name="Acme", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, pipeline_config={}))
        s.add(TenantSecret(tenant_id="t_acme", name="twilio_sid",
                            value_encrypted=crypto.encrypt("AC-real-sid")))
        s.add(TenantSecret(tenant_id="t_acme", name="twilio_token",
                            value_encrypted="not-a-valid-fernet-token"))
        await s.commit()

    r = DbTenantResolver(sm)
    # Without the fix, this raises KeyError from inside the logging call
    # instead of returning cleanly.
    assert await r.reload() == 1

    ctx = await r.resolve_by_slug("acme")
    assert ctx is not None
    assert ctx.secrets_resolved["twilio_sid"] == "AC-real-sid"
    assert "twilio_token" not in ctx.secrets_resolved


@pytest.mark.asyncio
async def test_reload_collision_on_secret_backed_key_logs_error_with_both_tenant_ids(sm, caplog):
    """Two tenants sharing a Chatwoot inbox id is a cross-tenant routing
    fault, not a diagnostic curiosity: whichever tenant loads second silently
    wins that key with no trace at any level below ERROR (DEBUG is off in
    normal running). This pins that reload() (a) still returns both tenants
    and resolves the key to the last-loaded one -- last-write-wins is kept,
    this fix is only about making the collision visible -- and (b) logs an
    ERROR naming the colliding key, its kind, and BOTH tenant ids so an
    operator can act without a database query."""
    async with sm() as s:
        s.add(Tenant(
            id="t_first", slug="first", name="First", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, pipeline_config={}))
        s.add(Tenant(
            id="t_second", slug="second", name="Second", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, pipeline_config={}))
        # chatwoot:inbox_id is a per-tenant secret with no DB-level uniqueness
        # constraint, unlike tenant_phone_numbers.phone_number (that one's
        # collision path is covered separately below, since the PK on that
        # table makes duplicating it via real rows impossible).
        s.add(TenantSecret(tenant_id="t_first", name="chatwoot:inbox_id",
                            value_encrypted=crypto.encrypt("42")))
        s.add(TenantSecret(tenant_id="t_second", name="chatwoot:inbox_id",
                            value_encrypted=crypto.encrypt("42")))
        await s.commit()

    r = DbTenantResolver(sm)
    with caplog.at_level("ERROR", logger="src.auth.db_resolver"):
        assert await r.reload() == 2

    errors = [rec for rec in caplog.records if rec.levelname == "ERROR"]
    inbox_errors = [rec for rec in errors if getattr(rec, "key_kind", None) == "chatwoot_inbox_id"]
    assert len(inbox_errors) == 1, "expected exactly one ERROR for the chatwoot_inbox_id collision"

    rec = inbox_errors[0]
    # Both tenant ids must be present so an operator can act without a
    # database query -- names alone (slug) are not the requirement, ids are.
    assert rec.existing_tenant_id == "t_first"
    assert rec.new_tenant_id == "t_second"
    assert rec.existing_tenant_slug == "first"
    assert rec.new_tenant_slug == "second"
    assert rec.key_repr == "42"

    # Last-write-wins behaviour is unchanged: both tenants are still loaded,
    # and the colliding key resolves to whichever tenant loaded last (t_second).
    assert await r.resolve_by_slug("first") is not None
    assert await r.resolve_by_slug("second") is not None
    resolved_inbox = await r.resolve_by_chatwoot_inbox("42")
    assert resolved_inbox.id == "t_second"


class _FakeScalarsResult:
    """Stand-in for the `Result` of `select(Tenant)...` -- `.scalars().all()`
    returns pre-built rows instead of querying the DB."""

    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _PhoneCollisionSession:
    """Wraps a real (empty) AsyncSession, substituting canned Tenant rows for
    the `select(Tenant)...` query only -- the two other reload() queries
    (Crm.id/prompt_pack, Crm.id/pronunciation_overrides) still hit the real,
    empty DB via delegation, exactly as they would for any tenant with no
    linked CRM.

    Why fake this one query at all: `tenant_phone_numbers.phone_number` is
    the table's PRIMARY KEY (see alembic/versions/0001_initial.py), so two
    different tenants can never actually share a phone number in the DB --
    inserting the second row raises IntegrityError before reload() runs. The
    collision branch in reload() for phone numbers is real defensive code
    (worth keeping, e.g. a future schema change or bulk-import bug), but it
    cannot be exercised through real seeded rows. This constructs two
    transient (never persisted, never added to any session) Tenant ORM
    objects whose in-memory `.phone_numbers` collections happen to share a
    number, and hands them back as the query result -- exercising the exact
    same reload() code path a real collision would hit.
    """

    def __init__(self, real_cm, fake_tenants):
        self._real_cm = real_cm
        self._fake_tenants = fake_tenants
        self._real_session = None

    async def __aenter__(self):
        self._real_session = await self._real_cm.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._real_cm.__aexit__(*exc)

    async def execute(self, stmt):
        try:
            is_tenant_query = stmt.column_descriptions[0]["type"] is Tenant
        except Exception:
            is_tenant_query = False
        if is_tenant_query:
            return _FakeScalarsResult(self._fake_tenants)
        return await self._real_session.execute(stmt)


@pytest.mark.asyncio
async def test_reload_collision_on_phone_number_logs_error_with_both_tenant_ids(sm, caplog):
    """Same guarantee as the secret-backed test above, for the phone_number
    key kind specifically -- see `_PhoneCollisionSession` for why this can't
    be seeded through two real DB rows."""
    fake_tenants = [
        Tenant(
            id="t_first", slug="first", name="First", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, pipeline_config={}, crm_id=None,
            api_keys=[], secrets=[],
            phone_numbers=[TenantPhoneNumber(phone_number="+1999", tenant_id="t_first", provider="twilio")],
        ),
        Tenant(
            id="t_second", slug="second", name="Second", status="active",
            timezone="Asia/Kolkata", default_language="hi", mode="layered",
            max_concurrent_calls=1, pipeline_config={}, crm_id=None,
            api_keys=[], secrets=[],
            phone_numbers=[TenantPhoneNumber(phone_number="+1999", tenant_id="t_second", provider="twilio")],
        ),
    ]

    def fake_sm():
        return _PhoneCollisionSession(sm(), fake_tenants)

    r = DbTenantResolver(fake_sm)
    with caplog.at_level("ERROR", logger="src.auth.db_resolver"):
        assert await r.reload() == 2

    errors = [rec for rec in caplog.records if rec.levelname == "ERROR"]
    phone_errors = [rec for rec in errors if getattr(rec, "key_kind", None) == "phone_number"]
    assert len(phone_errors) == 1, "expected exactly one ERROR for the phone_number collision"

    rec = phone_errors[0]
    assert rec.existing_tenant_id == "t_first"
    assert rec.new_tenant_id == "t_second"
    assert rec.existing_tenant_slug == "first"
    assert rec.new_tenant_slug == "second"
    # Phone numbers are not a credential -- safe to log in full (see
    # docs/debug-logging.md), so key_repr is the raw number, not a fingerprint.
    assert rec.key_repr == "+1999"

    # Last-write-wins behaviour is unchanged.
    resolved = await r.resolve_by_phone_number("+1999")
    assert resolved.id == "t_second"
    assert await r.resolve_by_slug("first") is not None
    assert await r.resolve_by_slug("second") is not None


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
