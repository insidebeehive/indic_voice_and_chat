"""Hot-issue notices: tenant- and CRM-scoped live incident notices.

- ``PUT /api/v1/hot-issues``                    replace a tenant's own set
- ``GET /api/v1/hot-issues``                    the tenant's active set
- ``PUT /api/v1/crms/{crm_id}/hot-issues``      replace a CRM-wide set (admin)
- ``GET /api/v1/crms/{crm_id}/hot-issues``      a CRM's active set (admin)

A replace is atomic (delete-then-insert, one commit) and REPLACES the whole
scope — ``{"issues": []}`` clears it. CRM-scope writes are admin-only; a
tenant writes its own set with its own token (or an admin token plus
``X-Tenant-Slug``, same as any other ``current_tenant`` route). See
``src/chatbot/hot_issues.py`` for how a chat/voice turn actually consumes
these rows (the union of a tenant's own + its linked CRM's, tenant first,
cached for up to 30s).

A second ``APIRouter(prefix="/crms")`` is used for the CRM-scope endpoints —
this does not clash with ``src/api/crm_kb.py``'s router, which is also
mounted at ``/crms`` but owns the disjoint ``/crms/{crm_id}/kb/...`` paths.
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.auth.audit import current_admin_label
from src.auth.middleware import require_admin
from src.chatbot.hot_issues import invalidate_scope
from src.models.crm import Crm
from src.models.hot_issue import (
    HOT_ISSUE_BODY_MAX_LEN,
    HOT_ISSUE_DEFAULT_TTL_HOURS,
    HOT_ISSUE_KEY_RE,
    HOT_ISSUE_MAX_PER_SCOPE,
    HOT_ISSUE_MAX_TTL_DAYS,
    HOT_ISSUE_TITLE_MAX_LEN,
    HOT_ISSUE_TOTAL_CHARS_MAX_PER_SCOPE,
    HotIssue,
)
from src.models.tenant import Tenant
from src.utils.logging import debug_event

log = logging.getLogger(__name__)

# Tenant-scope router: mounted with no extra prefix — the final paths are
# /api/v1/hot-issues (src/api/__init__.py adds the /api/v1 prefix).
router = APIRouter(tags=["hot-issues"])
# CRM-scope router: admin-only, /api/v1/crms/{crm_id}/hot-issues.
crm_router = APIRouter(prefix="/crms", tags=["hot-issues"])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _to_naive_utc(dt: datetime) -> datetime:
    """Aware input is converted to naive UTC; naive input is assumed already
    UTC and passed through unchanged, matching src/models/chat.py's
    convention for this codebase."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# --- Schemas --------------------------------------------------------------


class HotIssueIn(BaseModel):
    key: str
    title: str
    body: str
    # Defaults to HOT_ISSUE_DEFAULT_TTL_HOURS from now when omitted — applied
    # in _resolved_expiry below, not here, so the default is computed at
    # write time rather than at request-parse time.
    expires_at: Optional[datetime] = None


class ReplaceHotIssuesRequest(BaseModel):
    issues: list[HotIssueIn] = Field(default_factory=list)


class HotIssueOut(BaseModel):
    key: str
    title: str
    body: str
    expires_at: str
    updated_at: str


class HotIssuesResponse(BaseModel):
    scope: Literal["tenant", "crm"]
    issues: list[HotIssueOut]


# --- Validation -------------------------------------------------------------


def _visible(text: str) -> str:
    """``text`` with every Unicode "Cf" (format) character -- zero-width
    space/joiner/non-joiner, BOM, left-to-right/right-to-left marks, etc. --
    dropped, then whitespace-stripped. A title/body made up of nothing but
    such characters (e.g. a single U+200B zero-width space) reads as
    non-empty to plain ``str.strip()``, since those characters aren't
    whitespace; this is the emptiness check's own notion of "nothing visible
    remains", used only for validation below -- the stored title/body text
    itself is never altered."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf").strip()


def _resolved_expiry(item: HotIssueIn) -> datetime:
    if item.expires_at is None:
        return _utcnow() + timedelta(hours=HOT_ISSUE_DEFAULT_TTL_HOURS)
    return _to_naive_utc(item.expires_at)


def _validate_replace_request(issues: list[HotIssueIn]) -> None:
    """Raise a 422 for any of the caps/format rules the write side owns:
    count per scope, title/body length, the per-scope title+body total, the
    key regex, duplicate keys within one request, and the expiry window.
    Nothing here touches the DB — this is pure request validation, run
    before the replace transaction opens."""
    if len(issues) > HOT_ISSUE_MAX_PER_SCOPE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"at most {HOT_ISSUE_MAX_PER_SCOPE} hot issues per scope "
                f"({len(issues)} given)"
            ),
        )
    seen_keys: set[str] = set()
    total_chars = 0
    now = _utcnow()
    max_expiry = now + timedelta(days=HOT_ISSUE_MAX_TTL_DAYS)
    for item in issues:
        # fullmatch, not match: `match` only anchors the START of the string
        # (the pattern's own trailing `$` matches just before a final "\n",
        # not end-of-string), so e.g. "pg-delay\n" would otherwise pass.
        if not HOT_ISSUE_KEY_RE.fullmatch(item.key):
            raise HTTPException(
                status_code=422,
                detail=f"key {item.key!r} must match {HOT_ISSUE_KEY_RE.pattern!r}",
            )
        if item.key in seen_keys:
            raise HTTPException(status_code=422, detail=f"duplicate key {item.key!r}")
        seen_keys.add(item.key)
        if not _visible(item.title):
            raise HTTPException(
                status_code=422, detail=f"title must not be empty (key {item.key!r})",
            )
        if not _visible(item.body):
            raise HTTPException(
                status_code=422, detail=f"body must not be empty (key {item.key!r})",
            )
        if len(item.title) > HOT_ISSUE_TITLE_MAX_LEN:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"title exceeds {HOT_ISSUE_TITLE_MAX_LEN} characters (key {item.key!r})"
                ),
            )
        if len(item.body) > HOT_ISSUE_BODY_MAX_LEN:
            raise HTTPException(
                status_code=422,
                detail=f"body exceeds {HOT_ISSUE_BODY_MAX_LEN} characters (key {item.key!r})",
            )
        total_chars += len(item.title) + len(item.body)
        expires_at = _resolved_expiry(item)
        if expires_at <= now:
            raise HTTPException(
                status_code=422,
                detail=f"expires_at must be in the future (key {item.key!r})",
            )
        if expires_at > max_expiry:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"expires_at is more than {HOT_ISSUE_MAX_TTL_DAYS} days out "
                    f"(key {item.key!r})"
                ),
            )
    if total_chars > HOT_ISSUE_TOTAL_CHARS_MAX_PER_SCOPE:
        raise HTTPException(
            status_code=422,
            detail=(
                f"title+body across this scope's issues exceeds "
                f"{HOT_ISSUE_TOTAL_CHARS_MAX_PER_SCOPE} characters ({total_chars} given)"
            ),
        )


# --- Shared helpers ---------------------------------------------------------


def _iso_z(dt: datetime) -> str:
    return dt.isoformat() + "Z"


def _to_out(row: HotIssue) -> HotIssueOut:
    return HotIssueOut(
        key=row.key, title=row.title, body=row.body,
        expires_at=_iso_z(row.expires_at), updated_at=_iso_z(row.updated_at),
    )


async def _replace_scope(
    session: AsyncSession,
    *,
    scope: Literal["tenant", "crm"],
    scope_id: str,
    column,
    issues: list[HotIssueIn],
) -> list[HotIssue]:
    """Delete-then-insert the whole scope's row set, in one transaction.
    Caller has already locked the parent (tenant/crm) row and validated the
    request. Commits; a leftover IntegrityError is translated to a 409 by
    the caller."""
    existing = (await session.execute(
        select(HotIssue).where(column == scope_id)
    )).scalars().all()
    for row in existing:
        await session.delete(row)
    await session.flush()

    new_rows: list[HotIssue] = []
    # created_at/updated_at are set from Python here, not left to the
    # column's `server_default=func.now()` -- on Postgres, func.now() is the
    # TRANSACTION start time, so every row inserted in this one replace would
    # otherwise get the IDENTICAL created_at, making "order by created_at"
    # non-deterministic across rows from the same replace. Stamping each row
    # a microsecond after the previous one preserves the request's own order
    # (each later item's timestamp sorts after the one before it) regardless
    # of DB server timezone/clock resolution; `.order_by(HotIssue.created_at,
    # HotIssue.id)` (the two GETs above, and the loader's own query) still
    # adds `id` as a tiebreaker for the edge case of equal timestamps.
    stamp_base = _utcnow()
    for i, item in enumerate(issues):
        stamp = stamp_base + timedelta(microseconds=i)
        kwargs = {"key": item.key, "title": item.title, "body": item.body,
                  "expires_at": _resolved_expiry(item),
                  "created_at": stamp, "updated_at": stamp}
        if scope == "tenant":
            kwargs["tenant_id"] = scope_id
        else:
            kwargs["crm_id"] = scope_id
        row = HotIssue(**kwargs)
        session.add(row)
        new_rows.append(row)
    await session.commit()
    for row in new_rows:
        await session.refresh(row)
    return new_rows


def _actor_label(fallback: str) -> str:
    return current_admin_label() or fallback


# --- Tenant-scope routes -----------------------------------------------------


@router.put("/hot-issues", response_model=HotIssuesResponse)
async def replace_tenant_hot_issues(
    req: ReplaceHotIssuesRequest,
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> HotIssuesResponse:
    _validate_replace_request(req.issues)

    stmt = select(Tenant).where(Tenant.id == tenant.id)
    if session.get_bind().dialect.name != "sqlite":
        # FOR NO KEY UPDATE, not plain FOR UPDATE: this lock only needs to
        # serialize concurrent hot-issues replaces against this same tenant
        # row, not block every FK-referencing insert elsewhere (e.g. a chat
        # session row) that merely needs the tenant's key to still exist.
        stmt = stmt.with_for_update(key_share=True)
    tenant_row = (await session.execute(stmt)).scalar_one_or_none()
    if tenant_row is None:
        # Covers the InMemoryTenantResolver case (a tenant resolvable by
        # token/config with no backing `tenants` row) — SQLite doesn't
        # enforce foreign keys, so this explicit check is the actual guard
        # against writing an orphaned hot_issues row in the unit-test suite.
        raise HTTPException(status_code=404, detail=f"tenant {tenant.id!r} not found")

    try:
        rows = await _replace_scope(
            session, scope="tenant", scope_id=tenant.id,
            column=HotIssue.tenant_id, issues=req.issues,
        )
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="hot issues replace conflicted; retry")

    invalidate_scope(tenant_id=tenant.id)
    debug_event(
        log, "hot_issues replace response", scope="tenant", scope_id=tenant.id,
        keys=[r.key for r in rows], actor=_actor_label(tenant.id),
    )
    return HotIssuesResponse(scope="tenant", issues=[_to_out(r) for r in rows])


@router.get("/hot-issues", response_model=HotIssuesResponse)
async def get_tenant_hot_issues(
    tenant: TenantContext = Depends(current_tenant),
    session: AsyncSession = Depends(get_db_session),
) -> HotIssuesResponse:
    rows = (await session.execute(
        select(HotIssue)
        .where(HotIssue.tenant_id == tenant.id, HotIssue.expires_at > _utcnow())
        .order_by(HotIssue.created_at, HotIssue.id)
    )).scalars().all()
    return HotIssuesResponse(scope="tenant", issues=[_to_out(r) for r in rows])


# --- CRM-scope routes (admin-only) -------------------------------------------


async def _require_crm(session: AsyncSession, crm_id: str) -> None:
    if await session.get(Crm, crm_id) is None:
        raise HTTPException(status_code=404, detail=f"CRM {crm_id!r} not found")


@crm_router.put("/{crm_id}/hot-issues", response_model=HotIssuesResponse)
async def replace_crm_hot_issues(
    crm_id: str,
    req: ReplaceHotIssuesRequest,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> HotIssuesResponse:
    _validate_replace_request(req.issues)

    stmt = select(Crm).where(Crm.id == crm_id)
    if session.get_bind().dialect.name != "sqlite":
        # FOR NO KEY UPDATE -- see the tenant route's identical comment above.
        stmt = stmt.with_for_update(key_share=True)
    crm_row = (await session.execute(stmt)).scalar_one_or_none()
    if crm_row is None:
        raise HTTPException(status_code=404, detail=f"CRM {crm_id!r} not found")

    try:
        rows = await _replace_scope(
            session, scope="crm", scope_id=crm_id,
            column=HotIssue.crm_id, issues=req.issues,
        )
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="hot issues replace conflicted; retry")

    invalidate_scope(crm_id=crm_id)
    debug_event(
        log, "hot_issues replace response", scope="crm", scope_id=crm_id,
        keys=[r.key for r in rows], actor=_actor_label(crm_id),
    )
    return HotIssuesResponse(scope="crm", issues=[_to_out(r) for r in rows])


@crm_router.get("/{crm_id}/hot-issues", response_model=HotIssuesResponse)
async def get_crm_hot_issues(
    crm_id: str,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> HotIssuesResponse:
    await _require_crm(session, crm_id)
    rows = (await session.execute(
        select(HotIssue)
        .where(HotIssue.crm_id == crm_id, HotIssue.expires_at > _utcnow())
        .order_by(HotIssue.created_at, HotIssue.id)
    )).scalars().all()
    return HotIssuesResponse(scope="crm", issues=[_to_out(r) for r in rows])
