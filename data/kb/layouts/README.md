# Frontend UI/Navigation Knowledge Base

Complements the backend-derived `kb/global/` (business logic, UI-agnostic) with UI and navigation answers derived from the frontend monorepo. The frontend is white-labeled at the **layout package** level (`packages/layout-1` … `layout-9`, `layout-sports`); each layout serves many operators via its `app-configs/` directory. UI variation is therefore documented **per layout, not per operator**.

## Resolution flow

When answering a UI/navigation question for a specific operator:

1. Look up the operator in `operator-to-layout.md` → find its layout package.
2. Check `layouts/layout-N.md` for that layout's deltas (menu structure, screen flow, features shown/hidden).
3. Anything not overridden there falls back to the shared baseline in `global/`.
4. Anything requiring live operator config (amounts, limits, brand assets, payment methods enabled) is out of scope here — defer to the backend KB's `operator-specific-queries.md`.

## Contents

- `global/` — navigation/UI behavior verified identical across layouts (numbered topic files, same topic list as the backend KB).
- `layouts/layout-N.md` — one file per layout package, documenting only deltas from the global baseline, with the `appConfigs/` fields that drive each difference.
- `operator-to-layout.md` — mechanical operator → layout lookup, generated from `apps/*/package.json` layout dependencies.

## Provenance

Derived from actual frontend code: layout package `pages/`, `layout/`, `components/`, `constant/app-configs/`, and shared `ui-toolkit` components. Statements inferred rather than read directly from code are flagged `[INFERRED]` in place. No operator-specific values (amounts, limits, brand names) appear in global or layout docs.
