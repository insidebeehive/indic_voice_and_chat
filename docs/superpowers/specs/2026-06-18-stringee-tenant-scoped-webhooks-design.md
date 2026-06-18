# Stringee tenant-scoped webhooks + correct conversation attribution

**Date:** 2026-06-18
**Status:** approved

## Problem

Stringee calls for the `stage` tenant were recorded under `dev` (or not found
under `stage`):

- **Softphone**: the answer webhook resolves the tenant from the URL slug
  (`/stringee/softphone-answer/{slug}`). Both tenants' calls hit the `dev`-slug
  URL, so every softphone row landed on `dev`.
- **Outbound voicebot**: the callout's answer URL is `…/stringee/answer` (no
  slug), and that route resolves the tenant by the **dialed/caller number**.
  `dev` and `stage` share number `918204268005`, which is registered to `dev`,
  so the answering bridge always ran with `dev`'s config.

Attribution must follow **who placed the call / owns the Stringee project**, not
the shared number.

## Decision

Keep a **single shared app domain** (`WEBHOOK_BASE_URL`); the only per-tenant
element is the **slug in the URL path**. No revert of the #94 `webhook_base_url`
removal.

## Changes

### 1. Outbound voicebot — slug in the answer URL
- `dev_console.dev_place_call` and `calls.call_lead` build the Stringee callout
  `answer_url` as `{WEBHOOK_BASE_URL}/stringee/answer/{tenant.slug}` — the tenant
  that called our place-call/campaign API.
- New route `GET|POST /stringee/answer/{tenant_slug}` resolves the tenant via
  `tenant_from_slug` and builds the bridge with that tenant's config.
- The existing number-based `/stringee/answer` stays as the **inbound fallback**.
- The conversation row is already created under the placing tenant (PR #96); this
  makes the live answering bridge agree with it.

### 2. Softphone — already slug-routed; surface the URLs
- No logic change: `/stringee/softphone-answer/{slug}` already attributes by slug.
- `TenantSummary` returns the per-tenant Stringee webhook URLs (computed from the
  platform base + slug):
  - `stringee_softphone_answer_url` = `{base}/stringee/softphone-answer/<slug>`
  - `stringee_answer_url` = `{base}/stringee/answer/<slug>`
- The backoffice Telephony pane shows them read-only (copy into each tenant's
  Stringee project) so softphone rows land on the correct tenant.

### 3. No `webhook_base_url` revert
Shared domain; per-tenant = slug only.

## Out of scope
- Inbound calls (still resolve by dialed number → number owner).
- Twilio/Exotel answer URLs (this is Stringee-specific).

## Testing
- Outbound place-call (dev-console) + campaign build `…/stringee/answer/<slug>`.
- New `/stringee/answer/{slug}` resolves the tenant by slug and builds the bridge.
- `TenantSummary` returns the two webhook URLs for a tenant.
