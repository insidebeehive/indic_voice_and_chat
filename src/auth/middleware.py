"""FastAPI dependencies for tenant resolution.

Resolution sources (in priority order):

1. ``Authorization: Bearer <token>``  — looked up by SHA-256 hash in the
   ``tenant_api_keys`` table (or the in-process registry under test).
2. ``X-Tenant-Slug: <slug>``  — admin-style header for trusted internal
   callers. Only honored when ``allow_header`` is True *and* the request
   also carries a valid ``Authorization: Bearer <admin-token>`` (checked
   against the same ``_admin_token_labels`` map ``require_admin`` uses) —
   a tenant slug alone (or a non-admin bearer token) is never sufficient.
3. Twilio voice webhook: ``To`` form param → ``tenant_phone_numbers`` row.
4. Twilio Media Streams WS: ``?tenant=<slug>`` query param the voice TwiML
   set on the stream URL.

The actual lookup is delegated to a ``TenantResolver`` so tests can inject
an in-process registry without spinning up Postgres. The default resolver
is set during application startup.
"""

from __future__ import annotations

import logging
import re
from typing import Awaitable, Callable, Optional, Protocol

from fastapi import Depends, HTTPException, Request, WebSocket, WebSocketException, status

from src.auth.audit import log_denied, set_admin_label, token_fingerprint
from src.auth.context import TenantContext, hash_api_token
from src.config_tenant import TenantSettings

log = logging.getLogger(__name__)


class TenantResolver(Protocol):
    async def resolve_by_token(self, token_hash: str) -> Optional[TenantContext]: ...

    async def resolve_by_slug(self, slug: str) -> Optional[TenantContext]: ...

    async def resolve_by_phone_number(self, phone_number: str) -> Optional[TenantContext]: ...

    async def resolve_by_id(self, tenant_id: str) -> Optional[TenantContext]: ...

    async def resolve_by_chatwoot_inbox(self, inbox_id: str) -> Optional[TenantContext]: ...

    async def resolve_by_stringee_webhook_token(self, token: str) -> Optional[TenantContext]: ...

    async def resolve_by_chatwoot_webhook_id(self, webhook_id: str) -> Optional[TenantContext]: ...


class InMemoryTenantResolver:
    """Test/bootstrap resolver: registers tenants by token, slug, and phone."""

    def __init__(self) -> None:
        self._by_token: dict[str, TenantContext] = {}
        self._by_slug: dict[str, TenantContext] = {}
        self._by_phone: dict[str, TenantContext] = {}
        self._by_id: dict[str, TenantContext] = {}
        self._by_chatwoot_inbox: dict[str, TenantContext] = {}
        self._by_stringee_webhook_token: dict[str, TenantContext] = {}
        self._by_chatwoot_webhook_id: dict[str, TenantContext] = {}

    def register(
        self,
        settings: TenantSettings,
        *,
        plaintext_tokens: Optional[list[str]] = None,
        secrets: Optional[dict[str, str]] = None,
    ) -> TenantContext:
        ctx = TenantContext(settings=settings, secrets_resolved=dict(secrets or {}))
        self._by_slug[settings.slug] = ctx
        self._by_id[settings.id] = ctx
        for token in plaintext_tokens or []:
            self._by_token[hash_api_token(token)] = ctx
        for phone in settings.phone_numbers:
            self._by_phone[phone] = ctx
        secrets = secrets or {}
        inbox_id = secrets.get("chatwoot:inbox_id")
        if inbox_id:
            self._by_chatwoot_inbox[str(inbox_id)] = ctx
        stringee_webhook_token = secrets.get("webhook:stringee_path_token")
        if stringee_webhook_token:
            self._by_stringee_webhook_token[str(stringee_webhook_token)] = ctx
        chatwoot_webhook_id = secrets.get("chatwoot:webhook_id")
        if chatwoot_webhook_id:
            self._by_chatwoot_webhook_id[str(chatwoot_webhook_id)] = ctx
        return ctx

    def clear(self) -> None:
        self._by_token.clear()
        self._by_slug.clear()
        self._by_phone.clear()
        self._by_id.clear()
        self._by_chatwoot_inbox.clear()
        self._by_stringee_webhook_token.clear()
        self._by_chatwoot_webhook_id.clear()

    async def resolve_by_token(self, token_hash: str) -> Optional[TenantContext]:
        return self._by_token.get(token_hash)

    async def resolve_by_slug(self, slug: str) -> Optional[TenantContext]:
        return self._by_slug.get(slug)

    async def resolve_by_phone_number(self, phone_number: str) -> Optional[TenantContext]:
        return self._by_phone.get(phone_number)

    async def resolve_by_id(self, tenant_id: str) -> Optional[TenantContext]:
        return self._by_id.get(tenant_id)

    async def resolve_by_chatwoot_inbox(self, inbox_id: str) -> Optional[TenantContext]:
        return self._by_chatwoot_inbox.get(str(inbox_id))

    async def resolve_by_stringee_webhook_token(self, token: str) -> Optional[TenantContext]:
        return self._by_stringee_webhook_token.get(str(token))

    async def resolve_by_chatwoot_webhook_id(self, webhook_id: str) -> Optional[TenantContext]:
        return self._by_chatwoot_webhook_id.get(str(webhook_id))


_resolver: Optional[TenantResolver] = None

_LABELED_TOKEN_RE = re.compile(r"^lbl=([A-Za-z0-9._-]{1,32}):(.+)$", re.DOTALL)

# token-hash -> operator label. Replaces the former `_admin_token_hashes` set;
# membership semantics are identical (`in` on a dict tests its keys).
_admin_token_labels: dict[str, str] = {}


def set_tenant_resolver(resolver: Optional[TenantResolver]) -> None:
    global _resolver
    _resolver = resolver


def _parse_admin_entry(entry: str) -> tuple[str, str]:
    """Split one VOX_ADMIN_TOKENS entry into (token, label).

    Labels are OPT-IN via an explicit ``lbl=<name>:`` prefix — deliberately
    not "any token containing a colon", because a colon is a perfectly legal
    character inside a real token and silently truncating one at its first
    colon would turn a valid admin credential into an unusable prefix.
    Anything without the prefix is treated as entirely unlabeled: the WHOLE
    string is the token, and it gets a synthetic label derived from its
    fingerprint so every admin action is still attributable to *a* credential.
    """
    m = _LABELED_TOKEN_RE.match(entry)
    if m:
        return m.group(2), m.group(1)
    return entry, f"unlabeled-{token_fingerprint(entry)[:6]}"


def set_admin_tokens(plaintext_tokens: list[str]) -> None:
    """Register tokens that grant platform-admin access (benchmarks etc.).

    Accepts the same plain ``list[str]`` it always has; an entry may
    additionally carry an operator label as ``lbl=<name>:<token>``.
    """
    global _admin_token_labels
    _admin_token_labels = {}
    for entry in plaintext_tokens:
        token, label = _parse_admin_entry(entry)
        _admin_token_labels[hash_api_token(token)] = label


def admin_label_for_token(token: str | None) -> str | None:
    """Operator label for a registered admin token, or None if unregistered."""
    if not token:
        return None
    return _admin_token_labels.get(hash_api_token(token))


def admin_token_labels() -> list[str]:
    """Sorted labels of every registered admin token (one entry per token;
    duplicates are possible and meaningful). Non-secret — safe to log."""
    return sorted(_admin_token_labels.values())


def register_tenant_for_test(
    settings: TenantSettings,
    *,
    plaintext_tokens: Optional[list[str]] = None,
    secrets: Optional[dict[str, str]] = None,
) -> TenantContext:
    """Convenience used by tests to seed a tenant on the in-memory resolver."""
    global _resolver
    if not isinstance(_resolver, InMemoryTenantResolver):
        _resolver = InMemoryTenantResolver()
    return _resolver.register(settings, plaintext_tokens=plaintext_tokens, secrets=secrets)


# --- FastAPI dependencies ----------------------------------------------


async def _resolve(request: Request, *, allow_slug_header: bool = False) -> Optional[TenantContext]:
    if _resolver is None:
        return None

    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    bearer_token: Optional[str] = None
    if auth and auth.lower().startswith("bearer "):
        bearer_token = auth.split(" ", 1)[1].strip()
        tctx = await _resolver.resolve_by_token(hash_api_token(bearer_token))
        if tctx is not None:
            return tctx

    if allow_slug_header and bearer_token:
        admin_label = _admin_token_labels.get(hash_api_token(bearer_token))
        if admin_label is not None:
            slug = request.headers.get("x-tenant-slug") or request.headers.get("X-Tenant-Slug")
            if slug:
                tctx = await _resolver.resolve_by_slug(slug)
                if tctx is not None:
                    # Highest-privilege path in this file: an admin token
                    # impersonating/acting-as a tenant via the slug header.
                    # Attribute it to the operator so admin-impersonation
                    # activity is never anonymous. Not a logging call — safe
                    # under the "_resolve never logs" invariant below.
                    set_admin_label(admin_label)
                return tctx

    return None


def _bearer_fp(request: Request) -> str | None:
    """Fingerprint of the presented bearer token, or None if none presented."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    return token_fingerprint(token) if token else None


async def current_tenant(request: Request) -> TenantContext:
    """Require a tenant — 401 if missing, 403 if invalid."""
    tctx = await _resolve(request, allow_slug_header=True)
    if tctx is None:
        log_denied(
            logging.WARNING, "tenant auth rejected",
            event="auth_rejected", reason="no_valid_tenant_credential",
            route=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid tenant credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if tctx.settings.status != "active":
        log_denied(
            logging.WARNING, "tenant auth rejected",
            event="auth_rejected", reason="tenant_suspended",
            route=request.url.path, tenant=tctx.slug, tenant_id=tctx.id,
            token_fp=_bearer_fp(request),
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="tenant suspended")
    return tctx


async def optional_tenant(request: Request) -> Optional[TenantContext]:
    """Return tenant if present, else None — for routes that scope by tenant
    but tolerate platform-admin too."""
    return await _resolve(request, allow_slug_header=True)


def _ws_route(websocket) -> str | None:
    """Path of a WebSocket, tolerating the duck-typed fakes used in tests."""
    return getattr(getattr(websocket, "url", None), "path", None)


async def require_admin(request: Request) -> None:
    """Gate platform-admin routes (benchmarks, tenant CRUD)."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        log_denied(
            logging.WARNING, "admin auth rejected",
            event="auth_rejected", reason="admin_no_bearer_header",
            route=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth.split(" ", 1)[1].strip()
    label = _admin_token_labels.get(hash_api_token(token))
    if label is None:
        log_denied(
            logging.WARNING, "admin auth rejected",
            event="auth_rejected", reason="admin_token_not_recognized",
            route=request.url.path,
            token_fp=token_fingerprint(token) if token else None,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin access denied")
    set_admin_label(label)


def is_admin_token(token: str | None) -> bool:
    """True iff ``token`` is a registered platform-admin token.

    Sync + None/empty-safe so WebSocket handlers (which get the token from a
    query param, not an Authorization header) can check it without duplicating
    the hashing rule ``require_admin`` uses.
    """
    if not token:
        return False
    return hash_api_token(token) in _admin_token_labels


async def require_admin_ws(websocket: WebSocket) -> None:
    """Gate an admin WebSocket route on ``?token=<admin-token>``.

    Used as a router-level dependency so the check runs BEFORE the route body
    calls ``websocket.accept()`` — an unauthenticated client never gets a live
    socket. Raising ``WebSocketException`` (rather than closing here) is
    required: FastAPI's own handler performs the close, and closing twice
    raises ``RuntimeError: Cannot call "send" once a close message has been
    sent``. Because the close happens pre-accept, a real ASGI server rejects
    the HTTP handshake with 403; the browser therefore sees a 1006 close, while
    an in-process TestClient sees the 1008 below.
    """
    token = (websocket.query_params.get("token") or "").strip()
    label = admin_label_for_token(token)
    if label is None:
        log_denied(
            logging.WARNING, "admin ws auth rejected",
            event="auth_rejected", reason="admin_ws_token_invalid",
            route=_ws_route(websocket),
            token_fp=token_fingerprint(token) if token else None,
        )
        raise WebSocketException(code=1008, reason="admin token required")
    set_admin_label(label)


async def tenant_from_twilio_to_number(to_number: str) -> TenantContext:
    """Resolve the tenant that owns a Twilio number (inbound voice webhook)."""
    if _resolver is None:
        log_denied(
            logging.ERROR, "tenant auth rejected",
            event="auth_rejected", reason="resolver_uninitialized",
        )
        raise HTTPException(status_code=503, detail="tenant resolver not initialized")
    tctx = await _resolver.resolve_by_phone_number(to_number)
    if tctx is None:
        log_denied(
            logging.WARNING, "tenant auth rejected",
            event="auth_rejected", reason="unknown_inbound_number",
            to_fp=token_fingerprint(to_number, domain="vox-logfp-tel-v1"),
        )
        raise HTTPException(status_code=404, detail=f"no tenant owns number {to_number}")
    return tctx


async def tenant_from_ws_query(websocket: WebSocket) -> TenantContext:
    """Resolve the tenant from a Twilio Media Streams ``?tenant=`` query param."""
    if _resolver is None:
        log_denied(
            logging.ERROR, "tenant auth rejected",
            event="auth_rejected", reason="resolver_uninitialized",
            route=_ws_route(websocket),
        )
        raise HTTPException(status_code=503, detail="tenant resolver not initialized")
    slug = websocket.query_params.get("tenant")
    if not slug:
        log_denied(
            logging.INFO, "tenant auth rejected",
            event="auth_rejected", reason="ws_missing_tenant_param",
            route=_ws_route(websocket),
        )
        raise HTTPException(status_code=400, detail="missing 'tenant' query param")
    # tenant_from_slug logs its own resolver_uninitialized/unknown_tenant_slug
    # rejections — do NOT wrap this call in a try/except that logs again here.
    return await tenant_from_slug(slug)


async def tenant_from_slug(slug: str) -> TenantContext:
    """Resolve a tenant by slug. Used by the Media Streams WS handler
    which receives the slug as a URL path segment.
    """
    if _resolver is None:
        log_denied(
            logging.ERROR, "tenant auth rejected",
            event="auth_rejected", reason="resolver_uninitialized",
        )
        raise HTTPException(status_code=503, detail="tenant resolver not initialized")
    tctx = await _resolver.resolve_by_slug(slug)
    if tctx is None:
        log_denied(
            logging.WARNING, "tenant auth rejected",
            event="auth_rejected", reason="unknown_tenant_slug",
            tenant=slug,
        )
        raise HTTPException(status_code=404, detail=f"unknown tenant slug {slug!r}")
    return tctx


async def tenant_from_id(tenant_id: str) -> Optional[TenantContext]:
    """Resolve a tenant by its id (e.g. from a chat_sessions row). Returns None
    if unknown — callers decide how to fail (a closed WS, not an HTTP error)."""
    if _resolver is None:
        return None
    return await _resolver.resolve_by_id(tenant_id)


async def tenant_from_bearer_token(token: str) -> Optional[TenantContext]:
    """Resolve a tenant from a raw bearer token (no 'Bearer ' prefix).
    Used for WebSocket endpoints where the token is passed as a query param."""
    if _resolver is None:
        return None
    return await _resolver.resolve_by_token(hash_api_token(token))
