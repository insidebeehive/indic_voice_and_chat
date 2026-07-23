# Platform-level CRM tool auth — design

**Date:** 2026-07-22
**Status:** Approved

## Purpose

CRM tools (the built-in catalog of 18 player/operator tools the ChatBot can
call) currently require each tenant to configure its own auth token
(`crm:api_token` in `tenant_secrets`) before the catalog fallback activates.
The platform currently has effectively one real operator behind it, and
requiring per-tenant setup for a shared CRM integration is unnecessary
friction — one tenant already has its tools registered per-tenant, which is
why "the tools aren't getting called" for any *other* tenant: there's no
platform-level fallback for the auth token, only for the base URL.

Goal: any tenant with no tenant-specific CRM configuration gets the full
built-in tool catalog, authenticated with **one shared platform-level
token**, with zero per-tenant setup required. Tenant-specific overrides
(their own registered tools, or their own `crm:*` secrets) continue to work
exactly as today and take precedence — this is additive, not a replacement.

## Current mechanism (unchanged, for context)

`_load_crm_tools_uncached()` in `src/bootstrap.py` (~line 247-321):
1. If the tenant has rows in `chat_tools`, those are used exclusively —
   tenant-specific tools always win.
2. Otherwise, falls back to the platform catalog (`src/chatbot/catalog.py`'s
   `ALL_TOOLS`, 18 tools) with:
   - `base_url` = tenant's `crm:base_url` secret, else `PLATFORM_CRM_BASE_URL`
     env var, else nothing (returns no tools).
   - `api_token` = tenant's `crm:api_token` secret **only** — no platform
     fallback today. This is the gap.
   - `auth_type` = tenant's `crm:auth_type` secret, else hardcoded `"api_key"`
     — also no platform fallback today.

## Change

In the same fallback block (`src/bootstrap.py`), extend `api_token` and
`auth_type` resolution to mirror the existing `base_url` pattern:

```python
api_token = sr.get("crm:api_token") or os.environ.get("PLATFORM_CRM_API_TOKEN")
auth_type = sr.get("crm:auth_type") or os.environ.get("PLATFORM_CRM_AUTH_TYPE") or "api_key"
```

No change to precedence order, no change to the tenant-specific
(`chat_tools` rows) path, no change to `ALL_TOOLS` itself.

## Config declaration (consistency with the recent secrets cleanup)

`PLATFORM_CRM_BASE_URL` was already an undeclared env var (read via raw
`os.environ.get`, invisible to `Secrets` auditing) — same class of gap the
prior secrets-realignment work just fixed for provider keys. Declare all
three in `src/config.py`'s `Secrets` class as `Optional[str] = None`,
declare-only (adapters/bootstrap keep reading raw env, unchanged behavior),
matching the existing convention:

```python
PLATFORM_CRM_BASE_URL: Optional[str] = None
PLATFORM_CRM_API_TOKEN: Optional[str] = None
PLATFORM_CRM_AUTH_TYPE: Optional[str] = None
```

Add corresponding commented documentation lines to `.env.example`.

## Testing

Extend `tests/unit` coverage for `_load_crm_tools_uncached` (find existing
tests via `grep -rln "_load_crm_tools_uncached\|crm:api_token\|crm:base_url" tests/unit/` —
likely in a bootstrap or CRM-tools test file):
- Tenant with no `chat_tools` rows, no `crm:*` secrets, `PLATFORM_CRM_BASE_URL`
  + `PLATFORM_CRM_API_TOKEN` set → returns all 18 catalog tools, each `execs`
  entry's `token` equal to the platform token.
- Tenant with its own `crm:api_token` set (platform token also set) → tenant's
  own token wins (existing precedence, must not regress).
- Tenant with its own `chat_tools` rows registered (platform token set) →
  tenant-specific tools returned unchanged, platform fallback never consulted
  (existing precedence, must not regress).
- No `PLATFORM_CRM_API_TOKEN` and no tenant secret → `token` is `None` (today's
  existing behavior when nothing is configured at all — unchanged, not a new
  failure mode).

## Deployment / data step (not code, explicitly out of this implementation)

The one tenant that already has its own `chat_tools` rows will keep using
them (unchanged precedence) even after this ships — it will NOT automatically
start using the shared platform token until its existing per-tenant
registrations are cleared via the existing `DELETE /chat/tools/{name}`
endpoint (tenant-authed) or `crm:*` secrets are unset. This is a manual
follow-up step for the user/ops, not part of this code change.

## Out of scope

- Any new registration/admin endpoint — none needed, existing endpoints cover
  every path (`DELETE /chat/tools/{name}` for clearing a tenant's overrides).
- Per-tenant auth with shared tool *definitions* (the option considered and
  not chosen) — not needed given the single-shared-token decision.
- A DB-backed platform secrets table — env vars are consistent with how every
  other platform-level credential in this codebase works.

## Correction (same day)

The shared `PLATFORM_CRM_API_TOKEN` fallback described above was removed the
same day it shipped, after the product owner (who has direct knowledge of the
CRM's actual auth model — not something inferrable from this codebase alone)
flagged it: this platform's downstream CRM authorizes access **by the API
token itself**, not by a request parameter such as `operator_id`/`user_id`.
That means a single shared platform-level token would let *every* tenant's
chat sessions make CRM calls authorized as whichever one tenant that token
actually belongs to — a real cross-tenant data access issue, not just an
inconvenience.

`PLATFORM_CRM_BASE_URL` and `PLATFORM_CRM_AUTH_TYPE` remain shared/platform-level
fallbacks — those carry no tenant-isolation implication (they only answer
"which URL" and "which auth scheme", never "authorized as whom"). Only the
`api_token` fallback was removed.

Net effect: `resolve_crm_tools()` now resolves `api_token` solely from the
tenant's own `crm:api_token` secret (`sr.get("crm:api_token")`), with no
environment-variable fallback. Every tenant using the platform tool catalog
must have its own genuine `crm:api_token` secret configured (existing
backoffice CRM tab → `PATCH /tenants/{id}`) — there is no shared-token
fallback for auth, and there must never be one again. A tenant with no
`crm:api_token` secret still gets the full catalog back (base_url + auth_type
still resolve), but every tool's `token` is `None` until that tenant
configures its own secret.

## Addendum (2026-07-23): the live CRM needs both headers together

Live 401s against `apistage.betstudio.io` were traced to a second gap: the
downstream CRM requires **both** an `X-API-Key` header **and** an
`Authorization: Bearer <token>` header on every call — confirmed by the
product owner from the CRM's own curl example, and confirmed empirically
live (a request with only `Authorization` still got 401; both headers must
be present together). The existing `auth_type`/`token` mechanism only ever
produces one header (`Authorization: Bearer` for `bearer`, or `X-API-Key` for
the old `api_key` mode) — never both at once.

Fix: added a second, independent, additive per-tenant secret
`crm:x_api_key`. When configured, it is **always** sent as
`X-API-Key: <value>`, unconditionally, alongside whatever the existing
`token`/`auth_type` logic already produces (`Authorization: Bearer` in the
live case). This is purely additive — the existing single-header
`token`/`auth_type` mechanism is unchanged. Encrypted at rest exactly like
`crm:api_token` (same `tenant_secrets` table, same Fernet
`crypto.encrypt`/`crypto.decrypt`).

Edge case (handled deliberately, not silently): if a tool's
`auth_type == "api_key"` (the old single-token api_key mode, which sets
`X-API-Key` from `token`) *and* the new `x_api_key` is also configured, the
dedicated `x_api_key` field wins — it's the more specific, newer mechanism.
See `src/chatbot/tool_executor.py::execute_crm_tool`.
