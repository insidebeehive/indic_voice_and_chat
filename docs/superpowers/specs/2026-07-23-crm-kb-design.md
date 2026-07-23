# Knowledge Base moves from platform-level to CRM-level — design

**Date:** 2026-07-23
**Status:** Approved

## Purpose

Today the Knowledge Base (RAG) system has exactly two tiers: **tenant** docs
(`KBDocument`, `tenant_id` FK, tenant-scoped) and **platform** docs
(`PlatformKBDocument`, no FK at all — genuinely global, shared by literally
every tenant on the whole install). This is the same shape the CRM-tools
system had before the CRM-entity design fixed it
(`docs/superpowers/specs/2026-07-23-crm-entity-design.md`): a "platform"
concept that's actually CRM-specific (product/policy knowledge belongs to
a CRM, not to every tenant on the platform regardless of which downstream
operator they belong to), forced into a single global bucket because there
was no CRM entity to scope it to.

Now that `Crm` exists, this design retires the fully-global KB tier and
replaces it with a CRM-scoped one, using the same resolution pattern
already proven for tool catalogs.

## Scope boundaries (explicit)

- This is a straight **replacement**, not an additive third tier: there is
  no more "shared by every tenant on the platform" bucket after this ships.
  A tenant's available KB docs are its own tenant docs plus its linked
  CRM's docs — nothing wider.
- **Tenant-level KB (`KBDocument`) is completely untouched** — same table,
  same fields, same ingestion/query endpoints, same behavior.
- **Retrieval stays always-mixed**, not a tenant-wins-outright fallback
  like tools. Today's `_active_retrievers` searches platform + tenant
  together and merges by score; after this change it searches CRM + tenant
  together and merges by score, unchanged in kind. This is a deliberate
  difference from the tools precedent: a tenant plausibly wants both its
  own specific docs *and* the shared CRM/company knowledge available at
  once, unlike a tool implementation which is either the tenant's own or
  the platform default, not both.
- **pgvector only.** The live system uses pgvector
  (`config/default.yaml:45`); FAISS is an available-but-undeployed
  alternate provider. FAISS's existing platform/tenant split
  (`data/faiss/platform/`) is left as-is — no CRM tier for FAISS in this
  work. Revisit only if FAISS becomes a live provider.
- Embedder choice (`GeminiEmbedder`, 384-dim) is unaffected — not
  tenant/CRM-scoped today, stays that way.
- A tenant not linked to any CRM (`crm_id is None`) gets tenant-docs-only
  results — same graceful degradation as `resolve_crm_tools()`, never an
  error.

## Data model

**Rename** `PlatformKBDocument` (`src/models/benchmark.py`) →
`CrmKBDocument`, relocated to `src/models/crm.py` next to `Crm`/`CrmTool`
since it's now CRM-owned:

```python
class CrmKBDocument(Base):
    __tablename__ = "crm_kb_documents"  # renamed from platform_kb_documents
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    crm_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("crms.id", ondelete="CASCADE"),
        nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    extra_data: Mapped[dict] = mapped_column(JSON, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now())
```

`crm_id` is **required, not nullable** — every `CrmKBDocument` belongs to
exactly one CRM, matching `CrmTool`'s shape. No permanent NULL/unassigned
state to reason about.

`voicebot.knowledge_chunks` (pgvector, `src/providers/vector_store/pgvector_store.py`)
gains a nullable `crm_id` column alongside the existing `tenant_id` column.
Exactly one of the two is set per row — enforced at the application layer
(the vector-store config dict passes either a `tenant_id` or a `crm_id`,
never both), same as how `tenant_id IS NULL` already means "not a tenant
chunk" today. No DB-level CHECK constraint — consistent with how this
project already handles this exact pattern for `tenant_id`.

## Migration

One Alembic migration:

1. Rename table `platform_kb_documents` → `crm_kb_documents`.
2. Add `crm_id` column (nullable at the DB level during the migration
   steps, set NOT NULL only after backfill — the usual add-then-backfill-
   then-constrain sequence for a new required column on a possibly
   non-empty table).
3. Backfill: if **exactly one** `Crm` row exists at migration time, set
   every existing `crm_kb_documents.crm_id` to that CRM's id (mirrors the
   auto-seed approach used by the CRM-entity migration — today that's
   `betstudio`). If **more than one** `Crm` row exists when this runs,
   the migration **fails loudly** (raises, does not guess) — silently
   picking one CRM for pre-existing global docs would be a real, silent
   data-attribution error, worse than requiring a human to resolve it
   explicitly. Zero `Crm` rows is unreachable in practice (the CRM-entity
   migration always seeds one), but the same fail-loud behavior applies
   if it somehow happens.
4. Add `crm_id` to `voicebot.knowledge_chunks`, backfill existing
   `tenant_id IS NULL` rows the same way (same single-CRM assumption,
   same fail-loud-on-ambiguity rule).
5. `ALTER TABLE crm_kb_documents ALTER COLUMN crm_id SET NOT NULL` — this
   table's `crm_id` is always required (see Data Model above). The
   vector-store `knowledge_chunks.crm_id` column stays nullable at the DB
   level, since tenant-scoped chunks legitimately have no `crm_id` — no
   NOT NULL step for that column.

"Fails loudly" means: the migration function raises an unhandled exception
(e.g. `RuntimeError` with a clear message naming the ambiguous CRM ids),
which aborts `alembic upgrade head` with a non-zero exit and a visible
traceback — not a caught-and-logged warning that lets the upgrade proceed
with a guessed value.

## Resolution logic changes

**`src/bootstrap.py`**: `build_platform_retriever()` (a single process-wide
retriever built once at startup) is replaced by a small per-CRM cache —
same shape as `_PerTenantRegistry` (`src/auth/registry.py:123`), keyed by
`crm_id: str` instead of `TenantContext`, lazily building and caching one
`HybridRetriever` per CRM the first time it's needed. This is required
because — unlike the old single global bucket — there can now be more
than one CRM, each needing its own retriever pointed at its own
`crm_id`-scoped chunk partition.

**`src/api/knowledge.py`**: `_active_retrievers(tenant)` resolves the
tenant's `crm_id` (`tenant.settings.crm_id`, same field `resolve_crm_tools`
already reads) and looks up that CRM's retriever from the new per-CRM
cache — `None` if the tenant has no `crm_id`. Returns `[crm_retriever,
tenant_retriever]` (skipping `None`s), same list-and-merge shape as today,
just re-scoped. `_build_kb_context` (the voicebot's one-shot KB dump in
`src/bootstrap.py`) resolves the same way.

`GET /knowledge/stats` (mixes both scopes today) reports CRM-scoped counts
instead of platform-wide counts, keyed by the tenant's linked CRM.

## API changes

The four admin `platform-*` endpoints in `src/api/knowledge.py` move to
become CRM sub-resources on the existing CRM router (`src/api/crms.py`),
addressed by CRM instead of being a flat unscoped resource. Same
admin-only gating (`Depends(require_admin)`) as every other CRM admin
route:

- `POST /api/v1/crms/{crm_id}/kb/ingest` — multipart upload (was
  `POST /knowledge/platform-ingest`).
- `GET /api/v1/crms/{crm_id}/kb/documents` — list (was
  `GET /knowledge/platform-documents`).
- `DELETE /api/v1/crms/{crm_id}/kb/documents/{doc_id}` — remove (was
  `DELETE /knowledge/platform-documents/{id}`).
- `GET /api/v1/crms/{crm_id}/kb/documents/{doc_id}/download` — download
  (was `GET /knowledge/platform-documents/{id}/download`).

All four 404 on an unknown `crm_id`, matching `src/api/crms.py`'s existing
`_detail()` 404 pattern. Tenant-facing KB endpoints (`/knowledge/ingest`,
`/knowledge/documents`, `/knowledge/query`, `/knowledge/stats`) are
unchanged in path and behavior — only their internal resolution of "which
non-tenant docs to also search" changes, per the section above.

## Admin UI

- **CRM management page** (`static/backoffice.html`, where CRM tools are
  already managed): gains a "Knowledge Base" section for the selected
  CRM — upload, list, delete — mirroring the tenant KB tab's existing
  list/upload pattern, wired to the new `/crms/{crm_id}/kb/*` endpoints.
- **Tenant's own Knowledge Base tab**: keeps its tenant-upload section
  completely unchanged. The old "Platform" upload section is replaced
  with a **read-only** "Active CRM docs (from linked CRM)" list — same
  read-only-mirror pattern the Chat tab already uses for "Active tools"
  (list only; to add/remove a CRM's docs, go to the CRM management page).

## Testing

- `CrmKBDocument` model + migration: table rename, `crm_id` backfill to
  the single existing CRM, NOT NULL constraint applied after backfill;
  a migration-time test asserting the fail-loud path when more than one
  `Crm` row exists (constructed directly against a test DB, not the real
  migration history).
- `pgvector_store.py`: `crm_id`-scoped chunk write/read round-trip,
  parallel to the existing `tenant_id`-scoped tests; confirms a
  `tenant_id` row and a `crm_id` row don't cross-contaminate a search.
- Per-CRM retriever cache: same lazy-build-and-cache behavior as
  `_PerTenantRegistry` (build once, reuse on second call for the same
  `crm_id`, independent instances for different `crm_id`s).
- `_active_retrievers`/`_build_kb_context`: tenant linked to a CRM gets
  both retrievers searched and merged; tenant with no `crm_id` gets
  tenant-only results, no error; regression guard that tenant-only
  results are unaffected by CRM linkage (i.e. linking/unlinking a CRM
  never changes a tenant's own docs).
- Moved admin endpoints: admin-gated (401/403 without `require_admin`),
  404 on an unknown `crm_id`, and a full ingest→list→delete round-trip
  against a real in-memory SQLite session (no mocks), matching this
  codebase's existing test conventions throughout.
- Backoffice: CRM management page's new KB section round-trips upload/
  list/delete; tenant KB tab's read-only CRM-docs list renders correctly
  and its existing tenant-upload section is provably unaffected.

## Out of scope (explicit)

- FAISS gaining a CRM tier — untouched, still platform/tenant-split,
  since it isn't deployed anywhere live.
- Any change to embedder selection/config — unaffected.
- A tenant-facing (non-admin) way to browse a CRM's KB docs beyond the
  read-only list already specified above — no new tenant-facing endpoint.
- Supporting a tenant linked to more than one CRM — same one-CRM-per-tenant
  constraint as the CRM-entity design (`docs/superpowers/specs/2026-07-23-crm-entity-design.md`), not revisited here.
- Migrating/backfilling `voicebot.knowledge_chunks` rows that belong to a
  *deleted* `KBDocument`/`PlatformKBDocument` (orphaned chunks, if any
  exist) — out of scope; this migration only touches rows reachable from
  a live document row.

## Implemented (2026-07-23)

Built per the plan at `docs/superpowers/plans/2026-07-23-crm-kb.md` (9 tasks,
all complete, reviewed clean). `CrmKBDocument` (`src/models/crm.py`) replaces
`PlatformKBDocument`; the per-CRM retriever cache replaces
`build_platform_retriever()`; the CRM-scoped admin API
(`POST/GET/DELETE /api/v1/crms/{crm_id}/kb/*`) replaces the old flat
`/knowledge/platform-*` routes; the backoffice CRM page gained a KB section
and the tenant KB tab gained a read-only "Active CRM docs" list.
`docs/chatbot.md`'s "Knowledge base" section documents the shipped mechanism.

The migration (`alembic/versions/0010_crm_kb_documents.py`) has been reviewed
and committed but has **not** yet been applied to any real database — unlike
the CRM-entity migration (`0009`) above, there is no live-Neon-DB
verification for this plan yet. Task 2's report confirms it was checked
structurally only (migration logic + a direct-against-a-test-DB fail-loud
assertion), not run against `stage`'s actual Neon database. Applying it and
confirming `GET /api/v1/knowledge/stats` / `GET
/api/v1/crms/{crm_id}/kb/documents` against real data remains a pending
manual step (see the plan's "Final verification" section).

What HAS been verified: the full backend is built, and the bare
`.venv/bin/python -m pytest tests/unit -q` run (no flags needed) is fully
green at `24 failed, 1100 passed, 1 skipped, 22 errors` — exactly reconciled
against the pre-plan baseline (`24 failed, 1093 passed, 22 errors` at commit
`1dd988d`, before this plan's Task 1): only new tests were added across all
9 tasks, zero regressions, zero new failures.
