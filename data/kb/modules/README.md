# Product Module Knowledge Base

Per-tenant **opt-in** vertical/product content, ingested via `POST /api/v1/knowledge/ingest-layout`
(same endpoint used for frontend layouts — see `../layouts/README.md`). A tenant opts into a
module only if it actually offers that vertical: not every operator runs casino, sports betting,
and matka/lottery, so this content is never auto-seeded the way a CRM's bundled KB pack
(`../packs/<pack-name>/`) is for a CRM that has opted into one.

## Resolution flow

For a product/vertical support question:

1. Check whether the tenant has ingested the relevant module (casino / sports / matka) into its
   own KB. If not, the tenant doesn't offer that vertical — say so rather than answering from a
   module the tenant never opted into.
2. If ingested, the module's backend doc plus its UI-help counterpart (see below) together cover
   both "how the feature works" and "where to find it in the app."
3. Anything vertical-agnostic (account, KYC, wallet, deposits, withdrawals, bonuses, responsible
   gaming, security, technical help) lives in a CRM's bundled KB pack (`../packs/<pack-name>/`)
   and is already available to every tenant under a CRM that has opted into that pack — don't
   duplicate it here.

## Contents

Each vertical is a **pair** of files — a backend/business-logic doc and its frontend/UI-navigation
counterpart — ingested together as two separate KB documents (never merged into one):

| Vertical | Backend doc | UI doc |
| --- | --- | --- |
| Casino | `06-casino-games.md` | `ui-05-casino.md` |
| Sports betting | `07-sports-betting.md` | `ui-06-sports.md` |
| Matka/lottery | `08-matka-lottery-games.md` | `ui-07-matka-lottery.md` |

Voicebot doc-priority matching (`_VOICE_KB_PRIORITY` in `src/rag/context_builder.py`) keys off
the exact filename, so renaming any of these files would break that.

## Relationship to the other KB directories

- `../layouts/` — frontend **layout** deltas (per white-label package), vertical-agnostic.
- `../packs/<pack-name>/` — a CRM's bundled KB pack, e.g. `../packs/betting-default/`:
  vertical-agnostic baseline content (account, KYC, wallet, deposits, withdrawals, bonuses,
  responsible gaming, security, technical help), auto-seeded (`crm_kb_documents`) into every CRM
  that has opted into that pack via its `bundled_kb_pack` column.
- `modules/` (this directory) — vertical/product content, opt-in per tenant (`kb_documents`),
  ingested via the `casino` / `sports` / `matka` keys on `POST /api/v1/knowledge/ingest-layout`.

## Provenance

Derived from backend business logic and the frontend monorepo's layout packages. No
operator-specific values (amounts, limits, brand names) appear here — those stay in each
tenant's own KB, ingested separately per tenant.
