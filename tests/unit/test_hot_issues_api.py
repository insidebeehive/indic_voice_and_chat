"""tests/unit/test_hot_issues_api.py

Marked real_hot_issues (see tests/conftest.py's autouse fixture) -- these
tests exercise src/api/hot_issues.py's real routes end to end against a real
(sqlite) DB, not the stubbed loader every other test module gets.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import hot_issues
from src.api.deps import get_db_session
from src.auth.middleware import InMemoryTenantResolver, set_admin_tokens, set_tenant_resolver
from src.config_tenant import TenantSettings
from src.models.crm import Crm
from src.models.database import Base
from src.models.hot_issue import HotIssue
from src.models.tenant import Tenant

pytestmark = pytest.mark.real_hot_issues

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}
TENANT_A_HEADERS = {"Authorization": "Bearer tenant-a-token"}
TENANT_B_HEADERS = {"Authorization": "Bearer tenant-b-token"}
GHOST_TENANT_HEADERS = {"Authorization": "Bearer ghost-token"}


def _settings(tenant_id: str, slug: str) -> TenantSettings:
    return TenantSettings(id=tenant_id, slug=slug, name=slug.title(), status="active")


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="tenant-a", slug="tenant-a", name="Tenant A"))
        s.add(Tenant(id="tenant-b", slug="tenant-b", name="Tenant B"))
        s.add(Crm(id="crm1", name="CRM One", base_url="https://crm.example"))
        s.add(Crm(id="crm2", name="CRM Two", base_url="https://crm2.example"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    resolver = InMemoryTenantResolver()
    set_tenant_resolver(resolver)
    resolver.register(_settings("tenant-a", "tenant-a"), plaintext_tokens=["tenant-a-token"])
    resolver.register(_settings("tenant-b", "tenant-b"), plaintext_tokens=["tenant-b-token"])
    # Resolvable by token but with NO backing `tenants` row -- exercises the
    # replace transaction's 404 guard (the InMemoryTenantResolver case the
    # plan calls out explicitly).
    resolver.register(_settings("ghost", "ghost"), plaintext_tokens=["ghost-token"])
    set_admin_tokens(["admin-token"])

    app = FastAPI()
    app.include_router(hot_issues.router)
    app.include_router(hot_issues.crm_router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    set_tenant_resolver(None)
    set_admin_tokens([])
    await engine.dispose()


# --- Tenant-scope: replace + read round trip --------------------------------


async def test_put_then_get_round_trip(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "pg-delay", "title": "Deposits delayed",
                           "body": "Deposits may take up to 30 minutes."}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scope"] == "tenant"
    assert len(body["issues"]) == 1
    assert body["issues"][0]["key"] == "pg-delay"

    got = await client.get("/hot-issues", headers=TENANT_A_HEADERS)
    assert got.status_code == 200
    assert got.json()["issues"][0]["key"] == "pg-delay"


async def test_replace_is_atomic_and_keys_persist_across_replaces(client: AsyncClient) -> None:
    first = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [
            {"key": "pg-delay", "title": "Deposits delayed", "body": "v1"},
            {"key": "kyc-slow", "title": "KYC slow", "body": "v1"},
        ]},
    )
    assert first.status_code == 200

    second = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "pg-delay", "title": "Deposits delayed", "body": "v2"}]},
    )
    assert second.status_code == 200
    issues = second.json()["issues"]
    assert [i["key"] for i in issues] == ["pg-delay"]
    assert issues[0]["body"] == "v2"

    got = await client.get("/hot-issues", headers=TENANT_A_HEADERS)
    assert [i["key"] for i in got.json()["issues"]] == ["pg-delay"]


async def test_empty_list_clears_the_set(client: AsyncClient) -> None:
    await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "pg-delay", "title": "x", "body": "y"}]},
    )
    cleared = await client.put("/hot-issues", headers=TENANT_A_HEADERS, json={"issues": []})
    assert cleared.status_code == 200
    assert cleared.json()["issues"] == []

    got = await client.get("/hot-issues", headers=TENANT_A_HEADERS)
    assert got.json()["issues"] == []


# --- 422s --------------------------------------------------------------------


async def test_422_too_many_issues(client: AsyncClient) -> None:
    issues = [{"key": f"k{i}", "title": "t", "body": "b"} for i in range(6)]
    resp = await client.put("/hot-issues", headers=TENANT_A_HEADERS, json={"issues": issues})
    assert resp.status_code == 422


async def test_422_title_too_long(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "T" * 81, "body": "b"}]},
    )
    assert resp.status_code == 422


async def test_422_body_too_long(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "B" * 401}]},
    )
    assert resp.status_code == 422


async def test_422_per_scope_total_chars_exceeded(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [
            {"key": "k1", "title": "T" * 80, "body": "B" * 400},
            {"key": "k2", "title": "T" * 80, "body": "B" * 400},
        ]},
    )
    assert resp.status_code == 422


async def test_422_key_regex_rejects_invalid_key(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "Bad Key!", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 422


async def test_422_key_regex_rejects_trailing_newline(client: AsyncClient) -> None:
    """`match` (vs `fullmatch`) only anchors the START of the string -- the
    pattern's own trailing `$` matches just before a final "\\n", not
    end-of-string -- so "pg-delay\\n" used to slip through validation."""
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "pg-delay\n", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 422


async def test_422_key_regex_rejects_max_length_key_with_trailing_newline(
    client: AsyncClient,
) -> None:
    """A key at exactly the 64-char cap, PLUS a trailing newline, is the
    sharpest case of the `match`-vs-`fullmatch` gap: the first 64 characters
    alone fully satisfy the pattern, so only anchoring the end of the whole
    string (fullmatch) catches the extra "\\n"."""
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "a" * 64 + "\n", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 422


async def test_422_title_whitespace_only(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "   ", "body": "b"}]},
    )
    assert resp.status_code == 422


async def test_422_body_whitespace_only(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "\n\t  "}]},
    )
    assert resp.status_code == 422


async def test_422_duplicate_keys_in_one_request(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [
            {"key": "dup", "title": "t1", "body": "b1"},
            {"key": "dup", "title": "t2", "body": "b2"},
        ]},
    )
    assert resp.status_code == 422


async def test_422_expiry_in_the_past(client: AsyncClient) -> None:
    past = _iso(datetime.now(timezone.utc) - timedelta(hours=1))
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b", "expires_at": past}]},
    )
    assert resp.status_code == 422


async def test_422_expiry_too_far_in_the_future(client: AsyncClient) -> None:
    far = _iso(datetime.now(timezone.utc) + timedelta(days=8))
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b", "expires_at": far}]},
    )
    assert resp.status_code == 422


async def test_expiry_defaults_to_24h_and_resets_on_replace(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 200
    expires_at = datetime.fromisoformat(resp.json()["issues"][0]["expires_at"].rstrip("Z"))
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    delta_hours = (expires_at - now).total_seconds() / 3600
    assert 23.9 < delta_hours < 24.1


# --- 404s ----------------------------------------------------------------


async def test_404_missing_tenant_row(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=GHOST_TENANT_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 404


async def test_404_unknown_crm_on_put(client: AsyncClient) -> None:
    resp = await client.put(
        "/crms/does-not-exist/hot-issues", headers=ADMIN_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 404


async def test_404_unknown_crm_on_get(client: AsyncClient) -> None:
    resp = await client.get("/crms/does-not-exist/hot-issues", headers=ADMIN_HEADERS)
    assert resp.status_code == 404


# --- Auth --------------------------------------------------------------------


async def test_crm_route_requires_admin_401_without_token(client: AsyncClient) -> None:
    resp = await client.get("/crms/crm1/hot-issues")
    assert resp.status_code == 401


async def test_crm_route_requires_admin_403_with_unrecognized_token(client: AsyncClient) -> None:
    resp = await client.get(
        "/crms/crm1/hot-issues", headers={"Authorization": "Bearer not-an-admin-token"})
    assert resp.status_code == 403


async def test_crm_route_rejects_a_valid_tenant_token_as_403(client: AsyncClient) -> None:
    """A tenant's own (real, resolvable) bearer token must not work on the
    admin-only CRM routes either -- require_admin only recognizes admin
    tokens, so a genuine tenant token is "not an admin token" just like any
    other unrecognized bearer value, and gets 403 (a bearer header IS
    present -- 401 is only for no/malformed Authorization header at all)."""
    resp = await client.get("/crms/crm1/hot-issues", headers=TENANT_A_HEADERS)
    assert resp.status_code == 403


async def test_admin_token_plus_tenant_slug_header_works_on_tenant_routes(
    client: AsyncClient,
) -> None:
    """current_tenant (the tenant-scope routes' dependency) accepts an admin
    bearer token plus X-Tenant-Slug exactly like any other current_tenant
    route (see tests/unit/test_tenant_auth.py) -- not just the tenant's own
    token."""
    admin_plus_slug = {"Authorization": "Bearer admin-token", "X-Tenant-Slug": "tenant-a"}
    put_resp = await client.put(
        "/hot-issues", headers=admin_plus_slug,
        json={"issues": [{"key": "k1", "title": "t", "body": "b"}]},
    )
    assert put_resp.status_code == 200, put_resp.text
    assert put_resp.json()["scope"] == "tenant"

    got = await client.get("/hot-issues", headers=admin_plus_slug)
    assert [i["key"] for i in got.json()["issues"]] == ["k1"]


# --- Scoping isolation ---------------------------------------------------


async def test_tenant_a_never_sees_tenant_bs_issues(client: AsyncClient) -> None:
    await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "a-only", "title": "t", "body": "b"}]},
    )
    got_b = await client.get("/hot-issues", headers=TENANT_B_HEADERS)
    assert got_b.json()["issues"] == []


async def test_crm_issues_scoped_to_their_own_crm(client: AsyncClient) -> None:
    await client.put(
        "/crms/crm1/hot-issues", headers=ADMIN_HEADERS,
        json={"issues": [{"key": "crm1-only", "title": "t", "body": "b"}]},
    )
    got_crm2 = await client.get("/crms/crm2/hot-issues", headers=ADMIN_HEADERS)
    assert got_crm2.json()["issues"] == []
    got_crm1 = await client.get("/crms/crm1/hot-issues", headers=ADMIN_HEADERS)
    assert [i["key"] for i in got_crm1.json()["issues"]] == ["crm1-only"]


# --- Response schema -------------------------------------------------------


async def test_response_schema_and_z_timestamp(client: AsyncClient) -> None:
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"scope", "issues"}
    issue = body["issues"][0]
    assert set(issue.keys()) == {"key", "title", "body", "expires_at", "updated_at"}
    assert issue["expires_at"].endswith("Z")
    assert issue["updated_at"].endswith("Z")


async def test_expires_at_with_non_utc_offset_stored_and_returned_as_utc(
    client: AsyncClient,
) -> None:
    """An explicit, non-UTC offset (+05:30) must be converted to the
    equivalent UTC instant, not have its raw wall-clock numbers reinterpreted
    as if they were already UTC."""
    # Pick a moment 2 days out (inside the 7-day cap) and render it in
    # +05:30 explicitly -- zeroing microseconds keeps the isoformat()
    # comparison below exact.
    target_utc = (datetime.now(timezone.utc) + timedelta(days=2)).replace(microsecond=0)
    local_offset = timezone(timedelta(hours=5, minutes=30))
    local_iso = target_utc.astimezone(local_offset).isoformat()
    assert local_iso.endswith("+05:30")

    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b", "expires_at": local_iso}]},
    )
    assert resp.status_code == 200, resp.text

    expected = target_utc.replace(tzinfo=None).isoformat() + "Z"
    assert resp.json()["issues"][0]["expires_at"] == expected


# --- Expiry filtering on read -----------------------------------------------


async def test_get_omits_an_expired_row(client: AsyncClient, monkeypatch) -> None:
    """GET must never return a row whose expires_at has passed, even though
    the replace endpoint itself refuses to accept an already-past expires_at
    (so the only way to observe this is to advance the read side's own
    clock past a validity window that WAS in the future when written)."""
    future = (datetime.now(timezone.utc) + timedelta(seconds=30)).replace(microsecond=0)
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b",
                           "expires_at": future.isoformat()}]},
    )
    assert resp.status_code == 200

    past_the_window = future.replace(tzinfo=None) + timedelta(seconds=1)
    monkeypatch.setattr(hot_issues, "_utcnow", lambda: past_the_window)

    got = await client.get("/hot-issues", headers=TENANT_A_HEADERS)
    assert got.json()["issues"] == []


# --- IntegrityError -> 409 ---------------------------------------------------


async def test_integrity_error_on_commit_returns_409(client: AsyncClient, monkeypatch) -> None:
    """A leftover IntegrityError surfacing from the replace transaction's
    commit (e.g. a UNIQUE-constraint race between two concurrent replaces of
    the same scope) must come back as a 409, not a 500."""
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import AsyncSession

    async def _raise_integrity_error(self, *a, **kw):
        raise IntegrityError("INSERT", {}, Exception("forced for test"))

    monkeypatch.setattr(AsyncSession, "commit", _raise_integrity_error)

    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "b"}]},
    )
    assert resp.status_code == 409


# --- Cache invalidation across the API/loader boundary ----------------------


@pytest_asyncio.fixture
async def client_with_shared_sessionmaker():
    """Like `client` above, but also hands back the sessionmaker so a test
    can point src.chatbot.hot_issues's loader (which reads through
    src.models.database.get_sessionmaker()) at the SAME db this API test
    client writes through. The plain `client` fixture doesn't expose this --
    changing its signature would ripple into every other test in this file."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    async with sm() as s:
        s.add(Tenant(id="tenant-a", slug="tenant-a", name="Tenant A"))
        s.add(Crm(id="crm1", name="CRM One", base_url="https://crm.example"))
        await s.commit()

    async def _session_override():
        async with sm() as session:
            yield session

    resolver = InMemoryTenantResolver()
    set_tenant_resolver(resolver)
    resolver.register(_settings("tenant-a", "tenant-a"), plaintext_tokens=["tenant-a-token"])
    set_admin_tokens(["admin-token"])

    app = FastAPI()
    app.include_router(hot_issues.router)
    app.include_router(hot_issues.crm_router)
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, sm

    set_tenant_resolver(None)
    set_admin_tokens([])
    await engine.dispose()


async def test_api_replace_invalidates_cache_for_subsequent_tenant_loader_reads(
    client_with_shared_sessionmaker, monkeypatch,
) -> None:
    """The replace endpoint's own invalidate_scope call (NOT anything this
    test does directly -- it never calls invalidate_scope itself) must make
    the loader's very next read reflect the new set immediately, instead of
    continuing to serve the pre-replace snapshot for up to the cache's 30s
    TTL."""
    client, sm = client_with_shared_sessionmaker
    from src.chatbot import hot_issues as hi
    from src.models import database as database_mod

    monkeypatch.setattr(database_mod, "get_sessionmaker", lambda: sm)
    hi.clear_cache()

    await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "first", "title": "First", "body": "v1"}]},
    )
    warm = await hi.get_active_hot_issues("tenant-a")
    assert warm.keys == ("t:first",)

    await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "second", "title": "Second", "body": "v2"}]},
    )

    reloaded = await hi.get_active_hot_issues("tenant-a")
    assert reloaded.keys == ("t:second",)


async def test_api_replace_invalidates_cache_for_subsequent_crm_loader_reads(
    client_with_shared_sessionmaker, monkeypatch,
) -> None:
    client, sm = client_with_shared_sessionmaker
    from src.chatbot import hot_issues as hi
    from src.models import database as database_mod

    monkeypatch.setattr(database_mod, "get_sessionmaker", lambda: sm)
    hi.clear_cache()

    await client.put(
        "/crms/crm1/hot-issues", headers=ADMIN_HEADERS,
        json={"issues": [{"key": "first", "title": "First", "body": "v1"}]},
    )
    warm = await hi.get_active_hot_issues("tenant-a", "crm1")
    assert warm.keys == ("c:first",)

    await client.put(
        "/crms/crm1/hot-issues", headers=ADMIN_HEADERS,
        json={"issues": [{"key": "second", "title": "Second", "body": "v2"}]},
    )

    reloaded = await hi.get_active_hot_issues("tenant-a", "crm1")
    assert reloaded.keys == ("c:second",)


# --- Ordering is preserved end to end ----------------------------------------


async def test_put_response_get_and_loader_keys_preserve_request_order(
    client_with_shared_sessionmaker, monkeypatch,
) -> None:
    """Pins ordering end to end for a single replace with keys given in a
    deliberately non-alphabetical order ("zz", "aa", "mm"): the PUT response,
    a following GET, and the loader's own `keys` (src/chatbot/hot_issues.py)
    must all preserve the REQUEST's own order, not get silently resorted
    (e.g. alphabetically) by any layer.

    Also pins that created_at is STRICTLY increasing in request order:
    _replace_scope stamps each row a microsecond after the previous one
    specifically so "order by created_at" stays deterministic even on a DB
    (or test clock) with coarser resolution than a microsecond. Reverting
    that Python-side stamping would make this assertion fail (rows would
    compare equal, or sort only by luck) even though the ORDER BY clause
    itself is unchanged -- this is read via the session directly, not
    through the API response, which doesn't expose created_at at all.
    """
    client, sm = client_with_shared_sessionmaker
    from src.chatbot import hot_issues as hi
    from src.models import database as database_mod

    monkeypatch.setattr(database_mod, "get_sessionmaker", lambda: sm)
    hi.clear_cache()

    put_resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [
            {"key": "zz", "title": "t-zz", "body": "b-zz"},
            {"key": "aa", "title": "t-aa", "body": "b-aa"},
            {"key": "mm", "title": "t-mm", "body": "b-mm"},
        ]},
    )
    assert put_resp.status_code == 200, put_resp.text
    assert [i["key"] for i in put_resp.json()["issues"]] == ["zz", "aa", "mm"]

    got = await client.get("/hot-issues", headers=TENANT_A_HEADERS)
    assert [i["key"] for i in got.json()["issues"]] == ["zz", "aa", "mm"]

    snap = await hi.get_active_hot_issues("tenant-a")
    assert snap.keys == ("t:zz", "t:aa", "t:mm")

    async with sm() as session:
        rows = (await session.execute(
            select(HotIssue).where(HotIssue.tenant_id == "tenant-a").order_by(HotIssue.id)
        )).scalars().all()
    assert [r.key for r in rows] == ["zz", "aa", "mm"]
    created_ats = [r.created_at for r in rows]
    assert all(created_ats[i] < created_ats[i + 1] for i in range(len(created_ats) - 1)), (
        f"created_at must be strictly increasing in request order, got {created_ats}"
    )


async def test_crm_put_response_get_and_loader_keys_preserve_request_order(
    client_with_shared_sessionmaker, monkeypatch,
) -> None:
    """CRM-scope counterpart to the tenant-scope ordering test above -- same
    non-alphabetical key order, same three assertions (PUT response, GET,
    loader keys) plus the strictly-increasing created_at check."""
    client, sm = client_with_shared_sessionmaker
    from src.chatbot import hot_issues as hi
    from src.models import database as database_mod

    monkeypatch.setattr(database_mod, "get_sessionmaker", lambda: sm)
    hi.clear_cache()

    put_resp = await client.put(
        "/crms/crm1/hot-issues", headers=ADMIN_HEADERS,
        json={"issues": [
            {"key": "zz", "title": "t-zz", "body": "b-zz"},
            {"key": "aa", "title": "t-aa", "body": "b-aa"},
            {"key": "mm", "title": "t-mm", "body": "b-mm"},
        ]},
    )
    assert put_resp.status_code == 200, put_resp.text
    assert [i["key"] for i in put_resp.json()["issues"]] == ["zz", "aa", "mm"]

    got = await client.get("/crms/crm1/hot-issues", headers=ADMIN_HEADERS)
    assert [i["key"] for i in got.json()["issues"]] == ["zz", "aa", "mm"]

    snap = await hi.get_active_hot_issues("tenant-a", "crm1")
    assert snap.keys == ("c:zz", "c:aa", "c:mm")

    async with sm() as session:
        rows = (await session.execute(
            select(HotIssue).where(HotIssue.crm_id == "crm1").order_by(HotIssue.id)
        )).scalars().all()
    assert [r.key for r in rows] == ["zz", "aa", "mm"]
    created_ats = [r.created_at for r in rows]
    assert all(created_ats[i] < created_ats[i + 1] for i in range(len(created_ats) - 1)), (
        f"created_at must be strictly increasing in request order, got {created_ats}"
    )


# --- Invisible-character-only title/body is rejected as empty ---------------


async def test_422_title_is_only_a_zero_width_space(client: AsyncClient) -> None:
    """A title made up of a single zero-width space (U+200B, Unicode category
    "Cf") reads as non-empty to plain `str.strip()` since it isn't
    whitespace -- the validator must still treat it as empty."""
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "​", "body": "b"}]},
    )
    assert resp.status_code == 422


async def test_422_body_is_zero_width_space_and_bom_around_a_space(client: AsyncClient) -> None:
    """A body of a zero-width space, a literal space, and a BOM (U+FEFF,
    also category "Cf") has nothing visible once the format characters are
    dropped and the remaining whitespace is stripped -- must still 422."""
    resp = await client.put(
        "/hot-issues", headers=TENANT_A_HEADERS,
        json={"issues": [{"key": "k1", "title": "t", "body": "​ ﻿"}]},
    )
    assert resp.status_code == 422
