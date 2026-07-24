from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.auth import (
    TenantContext,
    current_tenant,
    optional_tenant,
    require_admin,
    register_tenant_for_test,
)
from src.auth.context import hash_api_token
from src.auth.middleware import (
    InMemoryTenantResolver,
    set_admin_tokens,
    set_tenant_resolver,
    tenant_from_twilio_to_number,
    tenant_from_ws_query,
)
from src.config_tenant import TenantSettings


def _settings(slug: str, *, phones: list[str] = (), status: str = "active") -> TenantSettings:
    return TenantSettings(
        id=f"t_{slug}",
        slug=slug,
        name=slug.title(),
        status=status,
        phone_numbers=list(phones),
    )


@pytest.fixture
def resolver():
    r = InMemoryTenantResolver()
    set_tenant_resolver(r)
    yield r
    set_tenant_resolver(None)
    set_admin_tokens([])


def _app(route) -> FastAPI:
    app = FastAPI()
    app.add_api_route("/who", route, methods=["GET"])
    return app


# --- hash_api_token -----------------------------------------------------


def test_hash_token_deterministic() -> None:
    assert hash_api_token("hello") == hash_api_token("hello")
    assert hash_api_token("hello") != hash_api_token("world")


# --- current_tenant -----------------------------------------------------


def test_bearer_token_resolves_tenant(resolver) -> None:
    resolver.register(_settings("acme"), plaintext_tokens=["secret-token"])

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"slug": t.slug}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"Authorization": "Bearer secret-token"})
    assert resp.status_code == 200
    assert resp.json()["slug"] == "acme"


def test_x_tenant_slug_header_alone_returns_401(resolver) -> None:
    """X-Tenant-Slug with no Authorization header must not resolve a tenant —
    slugs are not secrets, so this must not be enough to impersonate a tenant."""
    resolver.register(_settings("acme"))

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"slug": t.slug}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"X-Tenant-Slug": "acme"})
    assert resp.status_code == 401


def test_x_tenant_slug_header_with_non_admin_token_returns_401(resolver) -> None:
    """A non-admin bearer token must not unlock the X-Tenant-Slug path either."""
    set_admin_tokens(["super-admin-token"])
    resolver.register(_settings("acme"), plaintext_tokens=["tenant-token"])

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"slug": t.slug}

    client = TestClient(_app(route))
    resp = client.get(
        "/who",
        headers={"X-Tenant-Slug": "acme", "Authorization": "Bearer some-random-token"},
    )
    assert resp.status_code == 401


def test_x_tenant_slug_header_with_valid_admin_token_resolves(resolver) -> None:
    """A genuine admin bearer token alongside X-Tenant-Slug is the only way
    to resolve a tenant via the slug header — the intended, secure behavior."""
    set_admin_tokens(["super-admin-token"])
    resolver.register(_settings("acme"))

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"slug": t.slug}

    client = TestClient(_app(route))
    resp = client.get(
        "/who",
        headers={"X-Tenant-Slug": "acme", "Authorization": "Bearer super-admin-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["slug"] == "acme"


def test_missing_auth_returns_401(resolver) -> None:
    async def route(t: TenantContext = Depends(current_tenant)):
        return {"ok": True}

    client = TestClient(_app(route))
    resp = client.get("/who")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_invalid_token_returns_401(resolver) -> None:
    resolver.register(_settings("acme"), plaintext_tokens=["good"])

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"ok": True}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_unknown_slug_returns_401(resolver) -> None:
    async def route(t: TenantContext = Depends(current_tenant)):
        return {"ok": True}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"X-Tenant-Slug": "ghost"})
    assert resp.status_code == 401


def test_suspended_tenant_returns_403(resolver) -> None:
    resolver.register(_settings("acme", status="suspended"), plaintext_tokens=["t"])

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"ok": True}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"Authorization": "Bearer t"})
    assert resp.status_code == 403


# --- optional_tenant ----------------------------------------------------


def test_optional_tenant_returns_none_without_auth(resolver) -> None:
    async def route(t=Depends(optional_tenant)):
        return {"present": t is not None}

    client = TestClient(_app(route))
    assert client.get("/who").json()["present"] is False


# --- require_admin ------------------------------------------------------


def test_require_admin_with_valid_token(resolver) -> None:
    set_admin_tokens(["super-admin-token"])

    async def route(_=Depends(require_admin)):
        return {"ok": True}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"Authorization": "Bearer super-admin-token"})
    assert resp.status_code == 200


def test_require_admin_with_tenant_token_returns_403(resolver) -> None:
    set_admin_tokens(["super-admin-token"])
    resolver.register(_settings("acme"), plaintext_tokens=["tenant-token"])

    async def route(_=Depends(require_admin)):
        return {"ok": True}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"Authorization": "Bearer tenant-token"})
    assert resp.status_code == 403


def test_require_admin_without_token_returns_401(resolver) -> None:
    async def route(_=Depends(require_admin)):
        return {"ok": True}

    client = TestClient(_app(route))
    resp = client.get("/who")
    assert resp.status_code == 401


# --- Twilio-style resolvers ---------------------------------------------


@pytest.mark.asyncio
async def test_tenant_from_twilio_to_number(resolver) -> None:
    resolver.register(_settings("acme", phones=["+918888888888"]))
    t = await tenant_from_twilio_to_number("+918888888888")
    assert t.slug == "acme"


@pytest.mark.asyncio
async def test_tenant_from_twilio_unknown_number(resolver) -> None:
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        await tenant_from_twilio_to_number("+919999")
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_tenant_from_ws_query_resolves(resolver) -> None:
    resolver.register(_settings("acme"))

    class _FakeWS:
        query_params = {"tenant": "acme"}

    t = await tenant_from_ws_query(_FakeWS())  # type: ignore[arg-type]
    assert t.slug == "acme"


@pytest.mark.asyncio
async def test_tenant_from_ws_missing_param(resolver) -> None:
    from fastapi import HTTPException

    class _FakeWS:
        query_params = {}

    with pytest.raises(HTTPException) as ei:
        await tenant_from_ws_query(_FakeWS())  # type: ignore[arg-type]
    assert ei.value.status_code == 400


# --- helper register_tenant_for_test -----------------------------------


def test_register_tenant_for_test_creates_resolver_if_needed() -> None:
    set_tenant_resolver(None)
    register_tenant_for_test(_settings("acme"), plaintext_tokens=["t"])

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"slug": t.slug}

    client = TestClient(_app(route))
    resp = client.get("/who", headers={"Authorization": "Bearer t"})
    assert resp.status_code == 200
    set_tenant_resolver(None)
