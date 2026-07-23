# CRM as a first-class entity — design

**Date:** 2026-07-23
**Status:** Approved

## Purpose

Today, everything related to the downstream operator CRM (the player/wallet/
transactions backend behind `apistage.betstudio.io`) is modeled as either a
per-tenant field (`tenant_secrets`' `crm:base_url`/`crm:auth_type`, or the
plain `events_webhook_url` string) or a single global env var
(`PLATFORM_CRM_BASE_URL`), with the tool catalog itself hardcoded as one
Python dict (`src/chatbot/catalog.py`'s `ALL_TOOLS`) shared by the whole
platform.

This repeatedly caused confusion and real bugs this session: the
"platform-level" fallback wasn't actually platform-level in the sense that
matters — it's CRM-level (specific to BetStudio's API), and re-typing
CRM config per-tenant (including a `events_webhook_url` that has to be
hand-assembled with the operator_id baked into the path) produces exactly
the class of copy/paste and drift errors this session spent hours tracing.

**Goal:** introduce a `Crm` entity as a first-class concept. CRM-level
config (base URL, tool catalog, webhook URL template, default auth header
style) is defined once per CRM and inherited by every tenant registered
against it. Genuinely tenant/operator-specific values (the auth
token/x-api-key, the operator_id itself) remain per-tenant, as they are
today — this is additive/reorganizing, not a security model change.

## Scope boundaries (explicit)

- **"CRM" here means the operator/player-data backend** (BetStudio,
  `apistage.betstudio.io` — wallet, transactions, bonuses, profile tools).
  It is unrelated to and does not touch the Chatwoot integration (a
  separate support-inbox product with its own `chatwoot:*` tenant secrets)
  — that code path is untouched by this design.
- `crm:api_token` (bearer token) and `crm:x_api_key` (the second auth
  header) **stay exactly as they are today** — per-tenant encrypted
  secrets, resolved with no change to their current logic. Not part of
  this migration.
- `operator_id` **stays tenant-level**, exactly as today
  (`pipeline_config.crm.operator_id`, plain, non-secret).

## Data model

Two new tables (Alembic migration):

```python
class Crm(Base):
    __tablename__ = "crms"
    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # slug, e.g. "betstudio"
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    events_webhook_url_template: Mapped[Optional[str]] = mapped_column(String(500))
    # e.g. "https://bostage.betstudio.io/webhooks/crm/softphone-events/{operator_id}"
    # {operator_id} substituted from the tenant's own operator_id at send time.
    auth_type: Mapped[str] = mapped_column(String(20), default="api_key")  # api_key|bearer
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())


class CrmTool(Base):
    __tablename__ = "crm_tools"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    crm_id: Mapped[str] = mapped_column(String(50), ForeignKey("crms.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    endpoint: Mapped[str] = mapped_column(String(500), nullable=False)  # relative path, joined with crm.base_url
    method: Mapped[str] = mapped_column(String(10), default="GET")
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)  # same shape as today's ALL_TOOLS[name]["parameters"]
    __table_args__ = (UniqueConstraint("crm_id", "name", name="uq_crm_tools_crm_name"),)
```

`Tenant` (`src/models/tenant.py`) gains:
```python
crm_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("crms.id", ondelete="SET NULL"), nullable=True)
```
Nullable — a tenant not linked to any CRM behaves exactly as a from-scratch
tenant does today: no catalog tools unless it registers its own via
`chat_tools`.

## Resolution logic changes

`resolve_crm_tools()` (`src/bootstrap.py`) keeps its 3-tier precedence,
redefining tier 2:

1. **Tenant-registered `chat_tools` rows** — completely unchanged, still
   wins outright. No behavior change to this branch at all.
2. **Else, if `tenant.crm_id` is set:** load that `Crm` row + its
   `crm_tools` rows (one query, same batching discipline as the existing
   tenant-rows branch). Build `execs` using `crm.base_url` (was:
   `PLATFORM_CRM_BASE_URL` env / `crm:base_url` secret),
   `crm.auth_type` (was: `PLATFORM_CRM_AUTH_TYPE` env / `crm:auth_type`
   secret — **auth_type becomes CRM-level only, no tenant override**,
   since it's a property of that CRM's API design), the tenant's own
   `crm:api_token`/`crm:x_api_key` secrets (**unchanged** resolution —
   still per-tenant, still required for auth to actually work), and
   `operatorid` header from the tenant's own `operator_id` (**unchanged**
   — still computed the same way, `tenant.settings.crm.operator_id or
   tenant.id`).
3. **Else:** `"none"` — no tools, same as today.

**Separately**, tenant-event webhook delivery (`src/main.py`'s
`_notify_tenant_event`, `src/api/chat_webhooks.py`'s `send_bo_webhook`)
changes its URL resolution: instead of reading a fully literal per-tenant
`events_webhook_url` string, it resolves `tenant.crm.events_webhook_url_template`
and substitutes `{operator_id}` with the tenant's own `operator_id`. The
per-tenant `events_webhook_url` field is kept as an explicit **override**
(if set, it wins over the CRM template verbatim) — an escape hatch for a
tenant that genuinely needs a different URL shape than its CRM's default,
same "specific things can still be tenant-level" principle used
throughout this design.

`PLATFORM_CRM_BASE_URL`/`PLATFORM_CRM_AUTH_TYPE` env vars and their
`Secrets` declarations become dead once every tenant is migrated — removed
in this same work, matching the project's standing zero-tolerance for
dead config (see the secrets-realignment work from earlier this week).
Any pre-existing per-tenant `crm:base_url`/`crm:auth_type` `tenant_secrets`
rows (if a tenant had one set before this migration) become vestigial —
no longer read by `resolve_crm_tools()` once tier 2 sources these from
`Crm` instead. Leave the rows in place (harmless, matches how this
project has handled other superseded-but-harmless fields) rather than
writing a destructive cleanup step into this migration.

`GET /chat/tools/resolved`'s `source` field is relabeled: `"platform_fallback"`
→ `"crm_catalog"` (the old name is now inaccurate — this isn't a platform-wide
fallback, it's this tenant's specific CRM's catalog). Response should
additionally surface which CRM (`crm_id`) served the tools, for the
"active tools" backoffice panel to display.

## Migration (auto-seed, zero manual re-entry)

One Alembic migration, run in order:
1. `CREATE TABLE crms`, `CREATE TABLE crm_tools`, `ALTER TABLE tenants ADD COLUMN crm_id`.
2. Insert one `Crm` row: `id='betstudio'`, `name='BetStudio'`,
   `base_url=<current PLATFORM_CRM_BASE_URL env value at migration time>`,
   `auth_type=<current PLATFORM_CRM_AUTH_TYPE env value, or 'api_key' default>`,
   `events_webhook_url_template=<a template derived from any currently-set
   tenant events_webhook_url, with the trailing operator_id path segment
   replaced by {operator_id} — see "Migration detail" below>`.
3. Insert one `CrmTool` row per entry in today's `ALL_TOOLS` dict
   (`src/chatbot/catalog.py`), `crm_id='betstudio'` — a straight 1:1 copy
   of `name`/`description`/`default_path`→`endpoint`/`method`/`parameters`.
4. `UPDATE tenants SET crm_id='betstudio' WHERE crm_id IS NULL` — every
   existing tenant (including `stage`) gets linked, preserving today's
   working behavior with zero manual steps.

**Migration detail — deriving the webhook template:** the migration script
must read each already-migrated tenant's current literal
`events_webhook_url` (e.g. `stage`'s
`https://bostage.betstudio.io/webhooks/crm/softphone-events/ab858a8c-...`),
strip the trailing UUID segment (which matches that tenant's own
`operator_id`), and use the resulting prefix + `{operator_id}` as the
`Crm.events_webhook_url_template`. If tenants' existing URLs disagree on
the template shape, use the most common one and leave the others as
explicit per-tenant `events_webhook_url` overrides (per the override
mechanism above) rather than forcing a mismatch.

## Admin UI (simple, per approved scope)

- Backoffice tenant edit page (`static/backoffice.html`'s CRM tab): the
  existing Base URL / Auth type fields **move off** this tab (now
  CRM-level, not tenant-editable here) — replaced with a **CRM dropdown**
  (fetched from `GET /api/v1/crms`, showing `name`). API token, X-API-Key,
  Operator ID fields **stay exactly as they are** (still tenant-level).
- New minimal API, admin-gated (`require_admin`, same as every other
  platform-admin route):
  - `GET /api/v1/crms` — list (id, name, base_url — for the dropdown).
  - `POST /api/v1/crms` — create (name, base_url, events_webhook_url_template,
    auth_type, and a `tools` list in the request body — each item shaped
    like today's `ALL_TOOLS` entries).
  - `GET /api/v1/crms/{id}` — detail including its full tool list.
  - `PATCH /api/v1/crms/{id}` — update CRM-level fields and/or replace the
    tool list wholesale (simplest correct semantics for a JSON-editor UI —
    no per-tool PATCH endpoint needed at this admin-UI depth).
- New minimal backoffice page (or a modal/section reachable from the
  tenant list) to create/edit a CRM: name/base_url/webhook
  template/auth_type as plain inputs, tool list as one JSON textarea
  (pre-filled with the current tools on edit) — matches the approved
  "simple: dropdown + JSON editor" scope, not full per-tool CRUD forms.

## Testing

- `resolve_crm_tools()`: tenant-linked-to-CRM case returns that CRM's
  tools with `crm.base_url`/`crm.auth_type` and the tenant's own
  `api_token`/`x_api_key`/`operatorid`; tenant-registered `chat_tools`
  rows still take full precedence over a linked CRM (regression guard);
  tenant with no CRM link and no `chat_tools` rows still returns `"none"`.
- Webhook delivery: `events_webhook_url_template` + `operator_id`
  substitution produces the exact expected URL; a tenant with an explicit
  `events_webhook_url` override still uses that literal value instead of
  the template.
- Migration: seeds exactly one `Crm` row with the 18 tools, links every
  pre-existing tenant, and — this is the safety-critical assertion for
  this session's work — the `stage` tenant's CRM tool calls and webhook
  delivery produce byte-identical requests before and after migration
  (same `base_url`, same tool endpoints, same webhook URL).
- `GET /api/v1/crms` CRUD: create/list/get/update round-trip; admin-auth
  enforced (non-admin token rejected).
- Backoffice: CRM dropdown populates and saves `crm_id` on the tenant;
  existing tenant-level fields (API token, X-API-Key, Operator ID)
  continue to save/read exactly as before.

## Out of scope (explicit)

- Any change to `crm:api_token`/`crm:x_api_key`/`operator_id` resolution
  or storage — untouched.
- The Chatwoot integration — entirely separate, untouched.
- Per-tool CRUD UI for a CRM's catalog — JSON editor only, per approved scope.
- Supporting a tenant linked to more than one CRM simultaneously — one CRM
  per tenant (nullable FK), matching every requirement gathered this
  session; a tenant needing two distinct CRMs is a future decision, not
  designed here.
