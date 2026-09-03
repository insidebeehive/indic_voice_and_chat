from __future__ import annotations

import logging

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocketException
from fastapi.testclient import TestClient

from src.auth import (
    TenantContext,
    current_tenant,
    optional_tenant,
    require_admin,
    register_tenant_for_test,
)
from src.auth.audit import current_admin_label, reset_suppression_state, token_fingerprint
from src.auth.context import hash_api_token
from src.auth.middleware import (
    InMemoryTenantResolver,
    admin_label_for_token,
    admin_token_labels,
    is_admin_token,
    require_admin_ws,
    set_admin_tokens,
    set_tenant_resolver,
    tenant_from_slug,
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
    reset_suppression_state()
    yield r
    set_tenant_resolver(None)
    set_admin_tokens([])
    reset_suppression_state()


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


def test_admin_slug_header_resolution_sets_admin_label(resolver) -> None:
    """The admin-bearer-token + X-Tenant-Slug path is the highest-privilege
    path in this module (an admin token acting as a tenant) — it must
    attribute the resulting request to the operator via the ambient
    admin_label ContextVar, not leave admin-impersonation activity
    anonymous. Read current_admin_label() from inside the route (same
    asyncio Task as _resolve's set_admin_label call) since the ContextVar
    doesn't propagate back out to the test's own context."""
    set_admin_tokens(["lbl=ops-bob:super-admin-token"])
    resolver.register(_settings("acme"))

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"slug": t.slug, "admin_label": current_admin_label()}

    client = TestClient(_app(route))
    resp = client.get(
        "/who",
        headers={"X-Tenant-Slug": "acme", "Authorization": "Bearer super-admin-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["slug"] == "acme"
    assert resp.json()["admin_label"] == "ops-bob"


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


# --- log_denied: one case per auth-rejection site -----------------------


def test_log_no_valid_tenant_credential(resolver, caplog) -> None:
    async def route(t: TenantContext = Depends(current_tenant)):
        return {"ok": True}

    client = TestClient(_app(route))
    with caplog.at_level(logging.INFO):
        resp = client.get("/who")
    assert resp.status_code == 401
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "no_valid_tenant_credential")
    assert rec.levelno == logging.WARNING
    assert rec.event == "auth_rejected"


def test_log_tenant_suspended(resolver, caplog) -> None:
    resolver.register(_settings("acme", status="suspended"), plaintext_tokens=["t"])

    async def route(t: TenantContext = Depends(current_tenant)):
        return {"ok": True}

    client = TestClient(_app(route))
    with caplog.at_level(logging.INFO):
        resp = client.get("/who", headers={"Authorization": "Bearer t"})
    assert resp.status_code == 403
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "tenant_suspended")
    assert rec.levelno == logging.WARNING
    assert rec.event == "auth_rejected"
    assert rec.token_fp == token_fingerprint("t")


def test_log_admin_no_bearer_header(resolver, caplog) -> None:
    async def route(_=Depends(require_admin)):
        return {"ok": True}

    client = TestClient(_app(route))
    with caplog.at_level(logging.INFO):
        resp = client.get("/who")
    assert resp.status_code == 401
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "admin_no_bearer_header")
    assert rec.levelno == logging.WARNING
    assert rec.event == "auth_rejected"


def test_log_admin_token_not_recognized(resolver, caplog) -> None:
    set_admin_tokens(["real-admin-token"])

    async def route(_=Depends(require_admin)):
        return {"ok": True}

    client = TestClient(_app(route))
    with caplog.at_level(logging.INFO):
        resp = client.get("/who", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 403
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "admin_token_not_recognized")
    assert rec.levelno == logging.WARNING
    assert rec.event == "auth_rejected"
    assert rec.token_fp == token_fingerprint("wrong-token")


@pytest.mark.asyncio
async def test_log_admin_ws_token_invalid(resolver, caplog) -> None:
    set_admin_tokens(["real-admin-token"])

    class _FakeWS:
        query_params = {"token": "wrong"}

    with caplog.at_level(logging.INFO):
        with pytest.raises(WebSocketException):
            await require_admin_ws(_FakeWS())  # type: ignore[arg-type]
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "admin_ws_token_invalid")
    assert rec.levelno == logging.WARNING
    assert rec.event == "auth_rejected"
    assert rec.token_fp == token_fingerprint("wrong")


@pytest.mark.asyncio
async def test_log_resolver_uninitialized_twilio(caplog) -> None:
    set_tenant_resolver(None)
    reset_suppression_state()
    with caplog.at_level(logging.INFO):
        with pytest.raises(HTTPException):
            await tenant_from_twilio_to_number("+911234567890")
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "resolver_uninitialized")
    assert rec.levelno == logging.ERROR
    assert rec.event == "auth_rejected"


@pytest.mark.asyncio
async def test_log_unknown_inbound_number(resolver, caplog) -> None:
    with caplog.at_level(logging.INFO):
        with pytest.raises(HTTPException):
            await tenant_from_twilio_to_number("+919999999999")
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "unknown_inbound_number")
    assert rec.levelno == logging.WARNING
    assert rec.event == "auth_rejected"
    assert rec.to_fp == token_fingerprint("+919999999999", domain="vox-logfp-tel-v1")


@pytest.mark.asyncio
async def test_log_resolver_uninitialized_ws_query(caplog) -> None:
    set_tenant_resolver(None)
    reset_suppression_state()

    class _FakeWS:
        query_params = {"tenant": "acme"}

    with caplog.at_level(logging.INFO):
        with pytest.raises(HTTPException):
            await tenant_from_ws_query(_FakeWS())  # type: ignore[arg-type]
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "resolver_uninitialized")
    assert rec.levelno == logging.ERROR
    assert rec.event == "auth_rejected"


@pytest.mark.asyncio
async def test_log_ws_missing_tenant_param(resolver, caplog) -> None:
    class _FakeWS:
        query_params = {}

    with caplog.at_level(logging.INFO):
        with pytest.raises(HTTPException):
            await tenant_from_ws_query(_FakeWS())  # type: ignore[arg-type]
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "ws_missing_tenant_param")
    assert rec.levelno == logging.INFO
    assert rec.event == "auth_rejected"


@pytest.mark.asyncio
async def test_log_resolver_uninitialized_slug(caplog) -> None:
    set_tenant_resolver(None)
    reset_suppression_state()
    with caplog.at_level(logging.INFO):
        with pytest.raises(HTTPException):
            await tenant_from_slug("acme")
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "resolver_uninitialized")
    assert rec.levelno == logging.ERROR
    assert rec.event == "auth_rejected"


@pytest.mark.asyncio
async def test_log_unknown_tenant_slug(resolver, caplog) -> None:
    with caplog.at_level(logging.INFO):
        with pytest.raises(HTTPException):
            await tenant_from_slug("ghost-slug")
    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "unknown_tenant_slug")
    assert rec.levelno == logging.WARNING
    assert rec.event == "auth_rejected"
    assert rec.tenant == "ghost-slug"


# --- No-raw-token regression ---------------------------------------------


def test_wrong_admin_token_never_appears_in_raw_logs(resolver, caplog) -> None:
    set_admin_tokens(["real-admin-token"])
    wrong = "attacker-guessed-token-98765"

    async def route(_=Depends(require_admin)):
        return {"ok": True}

    client = TestClient(_app(route))
    with caplog.at_level(logging.INFO):
        resp = client.get("/who", headers={"Authorization": f"Bearer {wrong}"})
    assert resp.status_code == 403

    for rec in caplog.records:
        assert wrong not in repr(rec.__dict__)
        assert wrong not in repr(rec)

    rec = next(r for r in caplog.records if getattr(r, "reason", None) == "admin_token_not_recognized")
    assert rec.token_fp == token_fingerprint(wrong)


# --- S9: tenant_from_ws_query must not double-log via tenant_from_slug ---


@pytest.mark.asyncio
async def test_ws_query_unknown_slug_logs_exactly_once(resolver, caplog) -> None:
    class _FakeWS:
        query_params = {"tenant": "unknown-slug"}

    with caplog.at_level(logging.INFO):
        with pytest.raises(HTTPException):
            await tenant_from_ws_query(_FakeWS())  # type: ignore[arg-type]

    matches = [r for r in caplog.records if getattr(r, "event", None) == "auth_rejected"]
    assert len(matches) == 1
    assert matches[0].reason == "unknown_tenant_slug"


# --- _resolve must never log itself (catches a regression in the shared --
# --- helper both current_tenant and optional_tenant call) ----------------


def test_resolve_never_logs_when_falling_through_to_admin(resolver, caplog) -> None:
    """Mirrors src/api/catalog.py's ``tenant_or_admin`` shape: optional_tenant
    (which delegates to _resolve) returns None for a legitimate admin caller
    with no tenant credential, and that must be silent. Only require_admin's
    own rejection branch is allowed to log — and here it doesn't reject at
    all, since the admin token is valid."""
    set_admin_tokens(["real-admin-token"])

    async def route(
        request: Request,
        tenant: TenantContext | None = Depends(optional_tenant),
    ):
        if tenant is None:
            await require_admin(request)
        return {"ok": True}

    client = TestClient(_app(route))
    with caplog.at_level(logging.INFO):
        resp = client.get("/who", headers={"Authorization": "Bearer real-admin-token"})
    assert resp.status_code == 200
    assert not any(getattr(r, "event", None) == "auth_rejected" for r in caplog.records)


# --- Labeled admin tokens -------------------------------------------------


def test_labeled_admin_token_registers_label(resolver) -> None:
    set_admin_tokens(["lbl=ops-alice:sometoken123"])
    assert admin_label_for_token("sometoken123") == "ops-alice"


def test_colon_in_token_without_lbl_prefix_stays_unlabeled(resolver) -> None:
    """A stray colon inside a plain token must NOT be parsed as a label
    prefix — only the explicit ``lbl=<name>:`` form opts in."""
    set_admin_tokens(["weird:token:value"])
    assert is_admin_token("weird:token:value") is True
    label = admin_label_for_token("weird:token:value")
    assert label is not None
    assert label.startswith("unlabeled-")


def test_plain_admin_tokens_old_call_shape_still_works(resolver) -> None:
    set_admin_tokens(["plaintoken1", "plaintoken2"])
    assert is_admin_token("plaintoken1") is True
    assert is_admin_token("plaintoken2") is True
    labels = admin_token_labels()
    assert len(labels) == 2
    assert labels == sorted(labels)
    assert all(l.startswith("unlabeled-") for l in labels)
