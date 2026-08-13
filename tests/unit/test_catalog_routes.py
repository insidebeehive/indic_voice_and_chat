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
    assert {"anushka", "abhilash"} <= voice_ids
    assert all(v["gender"] in ("male", "female") for v in voices)


async def test_get_voices_gemini_live(client: AsyncClient) -> None:
    resp = await client.get(
        "/voices", params={"provider": "gemini_live"}, headers=TENANT_HEADERS
    )
    assert resp.status_code == 200
    assert "Aoede" in {v["voice_id"] for v in resp.json()["voices"]}


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
