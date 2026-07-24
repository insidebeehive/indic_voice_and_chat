# Incorporating the frontend UI/navigation KB — design

**Date:** 2026-07-24
**Status:** Approved

## Purpose

The user added `data/kb/frontend KB/` — 22 markdown files documenting frontend
UI/navigation behavior, derived from the frontend monorepo, meant to
complement the existing backend-derived KB at `data/kb/global/` (business
logic). This design covers how to fold that content into the KB system that
was just built out (CRM-scoped shared docs + tenant-scoped docs).

## What's in the folder

- `global/` (11 files) — UI/navigation behavior verified identical across all
  frontend layout packages (auth flows, KYC, wallet, casino, sports, matka,
  bonuses, profile, responsible gaming, troubleshooting). Same topic list as
  `data/kb/global/`, different angle (UI/navigation vs. business logic).
- `layouts/layout-N.md` (10 files: `layout-1` … `layout-9`, `layout-sports`) —
  UI deltas specific to one frontend package (e.g. layout-1 is a casino-first
  full `/login`+`/register` layout; layout-2/3/4 are matka, phone-first
  single-page onboarding). These **contradict each other** — a bot must never
  see more than one layout's deltas for a given tenant.
- `operator-to-layout.md` — a mechanical mapping (derived from
  `apps/*/package.json` dependencies) from real operator app names (e.g.
  `jupiter-app`, `khelomama`, `mgm91`) to a layout number, with `operator_id`
  UUIDs where known. `jupiter-app`'s UUID
  (`ab858a8c-7ad4-47d2-a0b7-05ee93f8f134`) matches the existing `stage`
  tenant's `operator_id` — confirming layout-1 is `stage`'s layout.
- `README.md` — explains the resolution flow (operator → layout via the
  mapping → layout doc for deltas → fall back to `global/`).

## Scope decision

No "layout" concept exists anywhere in this codebase today (confirmed:
`Tenant`, `TenantSettings`, `TenantCRMConfig` have no such field). Building
one — a first-class `Layout`/`LayoutKBDocument` entity mirroring the
just-shipped `Crm`/`CrmKBDocument` pattern, with a `Tenant.layout_id` FK and
a third merge axis in every KB retrieval path — was considered and
explicitly rejected for now: most of `operator-to-layout.md`'s ~50 operators
don't correspond to any real `Tenant` row yet (only `stage` does), and this
content is static/derived (from a one-time monorepo scan), not something
requiring a live admin-editable CRUD surface. Building that entity now would
be speculative generality for an axis whose real usage pattern is still
mostly future-facing.

**Decision: lightweight, zero-new-entity approach.**
- The `global/` UI docs are genuinely CRM-wide (true for every tenant on a
  CRM regardless of layout) — they ride the **existing** CRM-level bundled
  auto-seed (`_seed_crm_kb`, `src/main.py`), which already recursively sweeps
  `data/kb/global/`. No code change needed there beyond relocating files.
- The `layouts/layout-N.md` deltas are ingested **per-tenant**, once, as a
  normal tenant-scoped `KBDocument` — via the existing tenant `POST
  /knowledge/ingest` endpoint, using the existing `scripts/ingest_kb.py`
  bulk-ingest tool (extended with a `--file` option for single-file use).
  This is a manual/scripted step at tenant-onboarding time, not an automatic
  runtime resolution — explicitly accepted by the user as the standing
  process: *"whenever new operator is registered we refer the mapping
  document to find out which layout it uses and dump the related kb in
  tenant kb."*
- `operator-to-layout.md` and `README.md` are **never ingested** into any
  bot-facing KB — they're internal reference material for whoever performs
  onboarding, not content a customer-facing bot should ever surface.

## File layout changes

```
data/kb/global/                      (existing, unchanged, auto-seeded per-CRM)
  01-account-registration-login.md
  ...
  frontend-ui/                       (NEW — same recursive sweep already picks this up)
    ui-01-login-register.md
    ui-02-kyc-verification.md
    ui-03-wallet-deposits.md
    ui-04-withdrawals.md
    ui-05-casino.md
    ui-06-sports.md
    ui-07-matka-lottery.md
    ui-08-bonuses.md
    ui-09-profile-settings.md
    ui-10-responsible-gaming.md
    ui-11-technical-troubleshooting.md

data/kb/layouts/                     (renamed from "data/kb/frontend KB" — NOT auto-seeded)
  layout-1.md ... layout-9.md, layout-sports.md
  operator-to-layout.md
  README.md
```

The `ui-` filename prefix is required, not cosmetic: `_seed_crm_kb`'s
`doc_id = f"crm_kb_{crm_id}_{f.stem}"` uses only the file's stem, not its
full relative path — `data/kb/global/` already has a `10-responsible-gaming.md`,
identical to one of the new files' original name. Without the prefix, the
second file ingested would silently collide/overwrite the first's
`CrmKBDocument` row (same `doc_id`). The prefix makes every stem
tree-wide unique without touching the seeding function itself. Verified doc_id
length stays under `CrmKBDocument.id`'s `String(50)` column for every
renamed file against today's only CRM id (`betstudio`) — longest case
`crm_kb_betstudio_ui-11-technical-troubleshooting` is 48 characters.

`data/kb/layouts/` sits outside `data/kb/global/`, so `_seed_crm_kb`'s sweep
(which only walks `data/kb/global/`) never touches it automatically — this
is what keeps the contradictory layout deltas from ever being bundled
platform-wide, with no new "skip this directory" logic required.

## Tooling change

`scripts/ingest_kb.py` gains a `--file` option (one or more explicit paths),
usable instead of or alongside the existing `--dir` (bulk) mode — the same
script now serves both the original "bulk-load the global bundle" use case
and the new "ingest one tenant's specific layout doc" use case. No new
script.

## Documentation change

Add a subsection to `docs/chatbot.md`'s Knowledge Base section (the
already-established home for KB documentation — no separate onboarding doc
exists in this repo) making the standing process explicit and discoverable:
when a new tenant/operator is registered, look up its layout in
`data/kb/layouts/operator-to-layout.md`, then run `scripts/ingest_kb.py
--file data/kb/layouts/layout-N.md --base-url ... --token <that tenant's
token>` against that tenant.

## Immediate action (not part of this plan's tasks — a manual follow-up)

`stage` (~ `jupiter-app`, confirmed via matching `operator_id`) maps to
layout-1. Once this plan's file moves are committed, ingest
`data/kb/layouts/layout-1.md` into `stage`'s KB using the extended
`scripts/ingest_kb.py --file`, run by whoever has `stage`'s live base URL and
tenant bearer token in hand — not executed automatically by this plan, since
it requires a real running deployment and a real tenant secret.

## Out of scope (explicit)

- Any new DB entity, migration, or KB-retrieval code change — this is a
  content-organization + tooling change only, riding entirely on
  already-shipped CRM-level and tenant-level KB infrastructure.
- Automatic layout resolution at runtime (e.g. inferring a tenant's layout
  from its `operator_id` and merging a layout retriever into every KB query)
  — deferred; revisit only if/when enough real tenants exist across enough
  distinct layouts that manual per-tenant ingestion becomes a genuine
  bottleneck.
- Ingesting `operator-to-layout.md` or either `README.md` into any bot KB.
