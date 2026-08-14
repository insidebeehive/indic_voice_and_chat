# Product Module Knowledge Base

Per-tenant **opt-in** vertical/product content, ingested via `POST /api/v1/knowledge/ingest-layout`
(same endpoint used for frontend layouts — see `../layouts/README.md`). A tenant opts into a
module only if it actually offers that vertical: not every operator runs casino, sports betting,
and matka/lottery, so this content is never force-seeded the way `../global/` is.

## Resolution flow

For a product/vertical support question:

1. Check whether the tenant has ingested the relevant module (casino / sports / matka) into its
   own KB. If not, the tenant doesn't offer that vertical — say so rather than answering from a
   module the tenant never opted into.
2. If ingested, the module's backend doc plus its UI-help counterpart (see below) together cover
   both "how the feature works" and "where to find it in the app."
3. Anything vertical-agnostic (account, KYC, wallet, deposits, withdrawals, bonuses, responsible
   gaming, security, technical help) lives in `../global/` and is already available to every
   tenant — don't duplicate it here.

## Contents

Each vertical is a **pair** of files — a backend/business-logic doc and its frontend/UI-navigation
counterpart — ingested together as two separate KB documents (never merged into one):

| Vertical | Backend doc | UI doc |
| --- | --- | --- |
| Casino | `06-casino-games.md` | `ui-05-casino.md` |
| Sports betting | `07-sports-betting.md` | `ui-06-sports.md` |
| Matka/lottery | `08-matka-lottery-games.md` | `ui-07-matka-lottery.md` |

These filenames are unchanged from when they lived under `../global/` and `../global/frontend-ui/`
— voicebot doc-priority matching (`_VOICE_KB_PRIORITY` in `src/rag/context_builder.py`) keys off
the exact filename, so renaming them would break that.

## Relationship to the other KB directories

- `../layouts/` — frontend **layout** deltas (per white-label package), vertical-agnostic.
- `../global/` — vertical-agnostic baseline content, force-seeded into every CRM
  (`crm_kb_documents`). Now that casino/sports/matka have moved here, `global/` covers only
  account, KYC, wallet, deposits, withdrawals, bonuses, responsible gaming, security, and
  technical help.
- `modules/` (this directory) — vertical/product content, opt-in per tenant (`kb_documents`),
  ingested via the `casino` / `sports` / `matka` keys on `POST /api/v1/knowledge/ingest-layout`.

## Provenance

Same provenance as the docs previously under `global/`: derived from backend business logic and
the frontend monorepo's layout packages. No operator-specific values (amounts, limits, brand
names) appear here — those stay in each tenant's own KB, ingested separately per tenant.
