"""Route tests for the provider cost catalog + voice list endpoints."""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import catalog
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_admin_tokens, set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.database import Base
from src.models.tenant import ProviderCost
from src.providers.tts.elevenlabs import _PRESET_VOICES, ElevenLabsTTSAdapter

TENANT_HEADERS = {"Authorization": "Bearer tenant-token"}
ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(ProviderCost(kind="tts", provider="sarvam", cost_per_min=0.0))
        s.add(ProviderCost(kind="telephony", provider="twilio", cost_per_min=0.014))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(
        TenantSettings(id="t1", slug="t1", name="T1"), plaintext_tokens=["tenant-token"]
    )
    set_admin_tokens(["admin-token"])

    app = FastAPI()
    app.include_router(catalog.router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    set_tenant_resolver(None)
    set_admin_tokens([])
    await engine.dispose()


@pytest.fixture(autouse=True)
def _reset_elevenlabs_live_cache():
    # catalog._elevenlabs_live_cache is a single module-level dict (one
    # platform-wide ElevenLabs account, see its own comment) -- without a
    # reset, whichever test runs first to populate it would leak its roster
    # (real or mocked) into every test that runs after it within the TTL.
    catalog._elevenlabs_live_cache["voices"] = None
    catalog._elevenlabs_live_cache["fetched_at"] = 0.0
    yield
    catalog._elevenlabs_live_cache["voices"] = None
    catalog._elevenlabs_live_cache["fetched_at"] = 0.0


async def test_list_providers_returns_catalog(client: AsyncClient) -> None:
    resp = await client.get("/providers", headers=TENANT_HEADERS)
    assert resp.status_code == 200
    items = {(p["kind"], p["provider"]): p["cost_per_min"] for p in resp.json()["providers"]}
    assert items[("telephony", "twilio")] == 0.014
    assert items[("tts", "sarvam")] == 0.0


async def test_list_providers_requires_tenant(client: AsyncClient) -> None:
    assert (await client.get("/providers")).status_code == 401


async def test_update_provider_cost_admin_updates_live(client: AsyncClient) -> None:
    resp = await client.put(
        "/providers/telephony/twilio", json={"cost_per_min": 0.02}, headers=ADMIN_HEADERS
    )
    assert resp.status_code == 200
    assert resp.json()["cost_per_min"] == 0.02
    listed = {(p["kind"], p["provider"]): p["cost_per_min"]
              for p in (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]}
    assert listed[("telephony", "twilio")] == 0.02


async def test_update_provider_cost_inserts_missing(client: AsyncClient) -> None:
    resp = await client.put(
        "/providers/stt/deepgram", json={"cost_per_min": 0.0043}, headers=ADMIN_HEADERS
    )
    assert resp.status_code == 200
    listed = {(p["kind"], p["provider"]) for p in
              (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]}
    assert ("stt", "deepgram") in listed


async def test_update_provider_cost_model_level(client: AsyncClient) -> None:
    # A per-model rate lands as its own (kind, provider, model) row.
    resp = await client.put(
        "/providers/llm/gemini",
        json={"cost_per_min": 0.012, "model": "gemini-2.5-pro"}, headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["model"] == "gemini-2.5-pro"
    rows = (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]
    pro = [r for r in rows if r["kind"] == "llm" and r["provider"] == "gemini"
           and r["model"] == "gemini-2.5-pro"]
    assert pro and pro[0]["cost_per_min"] == 0.012


async def test_update_provider_cost_partial_update_preserves_token_rates(client: AsyncClient) -> None:
    """The live admin UI only ever sends {cost_per_min, model} on a plain
    per-minute rate edit. That PUT must not reset an existing row's
    cost_per_1k_input_tokens/cost_per_1k_output_tokens to 0 -- they're
    Optional[float] = None on the request model precisely so "omitted" can be
    told apart from "explicitly zero"."""
    # Seed a row with existing (nonzero) token rates, as if set previously.
    put_resp = await client.put(
        "/providers/llm/gemini",
        json={
            "cost_per_min": 0.002, "model": "gemini-3.5-flash",
            "cost_per_1k_input_tokens": 0.0003, "cost_per_1k_output_tokens": 0.0025,
        },
        headers=ADMIN_HEADERS,
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["cost_per_1k_input_tokens"] == pytest.approx(0.0003)
    assert put_resp.json()["cost_per_1k_output_tokens"] == pytest.approx(0.0025)

    # Now the admin UI sends only cost_per_min/model (no token-rate fields).
    resp = await client.put(
        "/providers/llm/gemini",
        json={"cost_per_min": 0.0025, "model": "gemini-3.5-flash"},
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    # cost_per_min was updated...
    assert body["cost_per_min"] == pytest.approx(0.0025)
    # ...but the token rates set earlier must survive untouched, not reset to 0.
    assert body["cost_per_1k_input_tokens"] == pytest.approx(0.0003)
    assert body["cost_per_1k_output_tokens"] == pytest.approx(0.0025)

    # The response reflects the actual stored row, not just an echo of the
    # request body (which had no token-rate fields at all).
    listed = (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]
    row = [r for r in listed if r["kind"] == "llm" and r["provider"] == "gemini"
           and r["model"] == "gemini-3.5-flash"][0]
    assert row["cost_per_1k_input_tokens"] == pytest.approx(0.0003)
    assert row["cost_per_1k_output_tokens"] == pytest.approx(0.0025)


async def test_update_provider_cost_cached_rate_partial_update(client: AsyncClient) -> None:
    """Same partial-update contract as cost_per_1k_input_tokens/output_tokens
    above, extended to cost_per_1k_cached_tokens: (1) a brand-new row created
    from a cost_per_min-only PUT must come back with cached rate `None`
    (unconfigured), never 0.0 -- 0.0 is a distinct, real "this provider is
    free to cache" value, so defaulting a never-sent field to it would be a
    silent lie about what was configured. (2) once a cached rate IS set, a
    later PUT that omits the field must not reset it to null/0."""
    # New row, cost_per_min + model only -- cached rate never mentioned.
    created = await client.put(
        "/providers/llm/gemini",
        json={"cost_per_min": 0.002, "model": "gemini-9-ultra"},
        headers=ADMIN_HEADERS,
    )
    assert created.status_code == 200
    assert created.json()["cost_per_1k_cached_tokens"] is None

    # Explicitly set the cached rate.
    set_resp = await client.put(
        "/providers/llm/gemini",
        json={
            "cost_per_min": 0.002, "model": "gemini-9-ultra",
            "cost_per_1k_cached_tokens": 0.00015,
        },
        headers=ADMIN_HEADERS,
    )
    assert set_resp.status_code == 200
    assert set_resp.json()["cost_per_1k_cached_tokens"] == pytest.approx(0.00015)

    # A later per-minute-only edit (the live UI's actual call shape) must
    # leave the cached rate exactly as it was, not reset it to null/0.
    later = await client.put(
        "/providers/llm/gemini",
        json={"cost_per_min": 0.0021, "model": "gemini-9-ultra"},
        headers=ADMIN_HEADERS,
    )
    assert later.status_code == 200
    assert later.json()["cost_per_min"] == pytest.approx(0.0021)
    assert later.json()["cost_per_1k_cached_tokens"] == pytest.approx(0.00015)

    listed = (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]
    row = [r for r in listed if r["kind"] == "llm" and r["provider"] == "gemini"
           and r["model"] == "gemini-9-ultra"][0]
    assert row["cost_per_1k_cached_tokens"] == pytest.approx(0.00015)


async def test_update_provider_cost_can_explicitly_zero_token_rate(client: AsyncClient) -> None:
    """An explicit 0.0 must be distinguished from an omitted field: sending
    cost_per_1k_input_tokens=0.0 must actually set it to zero, not be treated
    as 'unset' and left alone."""
    await client.put(
        "/providers/llm/gemini",
        json={
            "cost_per_min": 0.002, "model": "gemini-3.5-flash",
            "cost_per_1k_input_tokens": 0.0003, "cost_per_1k_output_tokens": 0.0025,
        },
        headers=ADMIN_HEADERS,
    )
    resp = await client.put(
        "/providers/llm/gemini",
        json={
            "cost_per_min": 0.002, "model": "gemini-3.5-flash",
            "cost_per_1k_input_tokens": 0.0, "cost_per_1k_output_tokens": 0.0025,
        },
        headers=ADMIN_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cost_per_1k_input_tokens"] == 0.0
    assert body["cost_per_1k_output_tokens"] == pytest.approx(0.0025)

    listed = (await client.get("/providers", headers=TENANT_HEADERS)).json()["providers"]
    row = [r for r in listed if r["kind"] == "llm" and r["provider"] == "gemini"
           and r["model"] == "gemini-3.5-flash"][0]
    assert row["cost_per_1k_input_tokens"] == 0.0


async def test_update_provider_cost_requires_admin(client: AsyncClient) -> None:
    resp = await client.put(
        "/providers/tts/sarvam", json={"cost_per_min": 1.0}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 403


async def test_get_voices_sarvam(client: AsyncClient) -> None:
    resp = await client.get(
        "/voices", params={"provider": "sarvam", "language": "hi-IN"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    voices = resp.json()["voices"]
    voice_ids = {v["voice_id"] for v in voices}
    assert {"priya", "aditya"} <= voice_ids  # bulbul:v3 roster (v2's anushka/abhilash are gone)
    assert all(v["gender"] in ("male", "female") for v in voices)


async def test_get_voices_gemini_live(client: AsyncClient) -> None:
    resp = await client.get(
        "/voices", params={"provider": "gemini_live"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    assert "Aoede" in {v["voice_id"] for v in resp.json()["voices"]}


async def test_get_voices_elevenlabs_flat_roster_ignores_language(client: AsyncClient) -> None:
    # ElevenLabs' catalog roster is its static preset list, language-independent
    # — a different `language` must return the identical roster, not an empty
    # one (only its live-fetched cloned/custom voices are language-scoped, and
    # those aren't in this catalog at all — see src/providers/voice_catalog.py).
    resp_a = await client.get(
        "/voices", params={"provider": "elevenlabs", "language": "hi-IN"}, headers=TENANT_HEADERS
    )
    resp_b = await client.get(
        "/voices", params={"provider": "elevenlabs", "language": "mr-IN"}, headers=TENANT_HEADERS
    )
    assert resp_a.status_code == 200 and resp_b.status_code == 200
    voices_a, voices_b = resp_a.json()["voices"], resp_b.json()["voices"]
    assert voices_a and voices_a == voices_b
    assert all(v["gender"] in ("male", "female") for v in voices_a)


async def test_get_voices_unsupported_language_empty(client: AsyncClient) -> None:
    # Per-language providers (sarvam/azure/google/indicf5) return [] for a
    # language they don't publish a roster for — deliberately NOT falling
    # back to hi-IN, since this is reference data for a UI dropdown, not a
    # synthesis call (see the module docstring on src/providers/voice_catalog.py).
    resp = await client.get(
        "/voices", params={"provider": "sarvam", "language": "xx-XX"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    assert resp.json()["voices"] == []


async def test_get_voices_unknown_provider_empty(client: AsyncClient) -> None:
    resp = await client.get("/voices", params={"provider": "nope"}, headers=TENANT_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["voices"] == []


async def test_get_voices_public_no_auth(client: AsyncClient) -> None:
    # Voices are public reference data — the Register dropdowns need them
    # before any token exists.
    resp = await client.get("/voices", params={"provider": "sarvam"})
    assert resp.status_code == 200


async def test_list_providers_admin_token_allowed(client: AsyncClient) -> None:
    # The cost catalog is readable by tenant OR admin (both consoles list it).
    assert (await client.get("/providers", headers=ADMIN_HEADERS)).status_code == 200


def _mock_live_roster(monkeypatch, voices=None, raise_exc=None):
    """Monkeypatch the adapter's own live fetch (no real network) so
    ``_live_elevenlabs_voices`` -- which constructs its own
    ``ElevenLabsTTSAdapter`` instance internally -- picks it up regardless of
    instance."""
    live = voices if voices is not None else [
        {"voice_id": "cloned-abc123", "name": "Tenant Custom Voice",
         "gender": "female", "category": "cloned"},
        {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel",
         "gender": "female", "category": "premade"},
    ]

    def _get_available_voices(self, language):
        if raise_exc is not None:
            raise raise_exc
        return list(live)

    monkeypatch.setattr(ElevenLabsTTSAdapter, "get_available_voices", _get_available_voices)
    return live


async def test_get_voices_elevenlabs_presets_have_names(client: AsyncClient) -> None:
    # Static catalog (no admin/live fetch involved) -- VoiceItem.name must be
    # populated for ElevenLabs' presets, which is the whole point of carrying
    # `name` through _normalize.
    resp = await client.get(
        "/voices", params={"provider": "elevenlabs"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    voices = resp.json()["voices"]
    assert {"Rachel", "Domi", "Bella"} <= {v["name"] for v in voices}


async def test_get_voices_elevenlabs_anonymous_gets_presets_only(
    client: AsyncClient, monkeypatch
) -> None:
    # Anonymous caller must never see the live roster -- it's the only thing
    # standing between "public reference data" and leaking every tenant's
    # cloned voice names to anyone. A live roster IS available (mocked); the
    # anonymous caller still must not get its custom voice.
    _mock_live_roster(monkeypatch)
    resp = await client.get("/voices", params={"provider": "elevenlabs"})
    assert resp.status_code == 200
    voices = resp.json()["voices"]
    assert {v["voice_id"] for v in voices} == {v["voice_id"] for v in _PRESET_VOICES}
    assert "cloned-abc123" not in {v["voice_id"] for v in voices}


async def test_get_voices_elevenlabs_tenant_token_gets_presets_only(
    client: AsyncClient, monkeypatch
) -> None:
    # A non-admin tenant bearer token is not sufficient either -- only a
    # platform-admin token unlocks the live roster.
    _mock_live_roster(monkeypatch)
    resp = await client.get(
        "/voices", params={"provider": "elevenlabs"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    voice_ids = {v["voice_id"] for v in resp.json()["voices"]}
    assert voice_ids == {v["voice_id"] for v in _PRESET_VOICES}
    assert "cloned-abc123" not in voice_ids


async def test_get_voices_elevenlabs_admin_gets_live_roster(
    client: AsyncClient, monkeypatch
) -> None:
    live = _mock_live_roster(monkeypatch)
    resp = await client.get(
        "/voices", params={"provider": "elevenlabs"}, headers=ADMIN_HEADERS
    )
    assert resp.status_code == 200
    voices = resp.json()["voices"]
    assert {v["voice_id"] for v in voices} == {v["voice_id"] for v in live}
    custom = next(v for v in voices if v["voice_id"] == "cloned-abc123")
    assert custom["name"] == "Tenant Custom Voice"
    assert custom["category"] == "cloned"


async def test_get_voices_elevenlabs_live_fetch_failure_falls_back_to_presets(
    client: AsyncClient, monkeypatch
) -> None:
    _mock_live_roster(monkeypatch, raise_exc=RuntimeError("boom"))
    resp = await client.get(
        "/voices", params={"provider": "elevenlabs"}, headers=ADMIN_HEADERS
    )
    assert resp.status_code == 200
    voice_ids = {v["voice_id"] for v in resp.json()["voices"]}
    assert voice_ids == {v["voice_id"] for v in _PRESET_VOICES}


async def test_get_voices_elevenlabs_admin_live_roster_cached_within_ttl(
    client: AsyncClient, monkeypatch
) -> None:
    calls = {"n": 0}

    def _get_available_voices(self, language):
        calls["n"] += 1
        return [{"voice_id": "cloned-xyz", "name": "Fresh Clone",
                  "gender": "male", "category": "cloned"}]

    monkeypatch.setattr(ElevenLabsTTSAdapter, "get_available_voices", _get_available_voices)

    r1 = await client.get("/voices", params={"provider": "elevenlabs"}, headers=ADMIN_HEADERS)
    r2 = await client.get("/voices", params={"provider": "elevenlabs"}, headers=ADMIN_HEADERS)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["voices"] == r2.json()["voices"]
    # Second call within the TTL must be served from cache, not refetched.
    assert calls["n"] == 1


async def test_models_public(client: AsyncClient) -> None:
    resp = await client.get("/models")          # no auth — public reference data
    assert resp.status_code == 200
    models = resp.json()["models"]
    # gemini exposes multiple variants (flash / flash-lite / pro / …).
    assert "gemini" in models["llm"]
    assert len(models["llm"]["gemini"]) >= 2
    assert any("lite" in m for m in models["llm"]["gemini"])
    assert "sarvam" in models["tts"]
    assert "gemini_live" in models["s2s"]


async def test_get_voices_elevenlabs_cached_live_roster_never_reaches_anonymous(
    client: AsyncClient, monkeypatch
) -> None:
    """Fill the cache as admin, then ask anonymously: still presets only. The
    property most likely to break if the cache is ever consulted before the
    admin check."""
    _mock_live_roster(monkeypatch)
    admin = await client.get("/voices", params={"provider": "elevenlabs"}, headers=ADMIN_HEADERS)
    assert any(v["voice_id"] == "cloned-abc123" for v in admin.json()["voices"])
    assert admin.headers.get("cache-control") == "private, no-store"
    for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "bearer not-an-admin"}):
        anon = await client.get("/voices", params={"provider": "elevenlabs"}, headers=headers)
        ids = {v["voice_id"] for v in anon.json()["voices"]}
        assert ids == {v["voice_id"] for v in _PRESET_VOICES}, headers
        assert "cloned-abc123" not in ids


async def test_get_voices_elevenlabs_failed_fetch_is_not_cached(
    client: AsyncClient, monkeypatch
) -> None:
    """A failed fetch (the adapter returns presets, which carry no category)
    must not be cached, or one blip hides cloned voices for the whole TTL."""
    state = {"fail": True}

    def _get_available_voices(self, language):
        if state["fail"]:
            return list(_PRESET_VOICES)
        return [{"voice_id": "cloned-late", "name": "Late Clone", "gender": "female",
                 "category": "cloned"}]

    monkeypatch.setattr(ElevenLabsTTSAdapter, "get_available_voices", _get_available_voices)
    r1 = await client.get("/voices", params={"provider": "elevenlabs"}, headers=ADMIN_HEADERS)
    assert "cloned-late" not in {v["voice_id"] for v in r1.json()["voices"]}
    state["fail"] = False
    r2 = await client.get("/voices", params={"provider": "elevenlabs"}, headers=ADMIN_HEADERS)
    assert "cloned-late" in {v["voice_id"] for v in r2.json()["voices"]}
