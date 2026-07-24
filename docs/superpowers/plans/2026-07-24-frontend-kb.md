# Frontend UI/Navigation KB Incorporation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the user-added `data/kb/frontend KB/` content into the existing KB system — CRM-wide UI docs ride the existing auto-seed, layout-specific deltas become a per-tenant, scripted onboarding step — with zero new DB entities or retrieval code changes.

**Architecture:** Pure content relocation (into the existing `data/kb/global/` auto-seeded tree, and a new non-seeded `data/kb/layouts/` reference tree) plus one CLI tool extension (`scripts/ingest_kb.py` gains `--file`) plus one documentation addition. No application code in `src/` changes.

**Tech Stack:** Plain file moves, Python argparse/pathlib (`scripts/ingest_kb.py`), Markdown docs.

## Global Constraints

- This is a content-organization + tooling change only — **no new DB entity, migration, or KB-retrieval code change** (per the design spec's explicit "Out of scope").
- `data/kb/frontend KB/global/*.md` (11 files) is currently **untracked** in git (confirmed via `git ls-files` — zero hits) — use plain `mv`/`cp` + `git add`, not `git mv`, when relocating it.
- Every relocated `global/` file MUST be renamed with a `ui-` prefix — this is required, not cosmetic: `_seed_crm_kb`'s `doc_id = f"crm_kb_{crm_id}_{f.stem}"` (`src/main.py`) uses only the file's stem, and the existing `data/kb/global/10-responsible-gaming.md` already has an identical stem to one of the new files. Without the prefix, ingesting the second file would silently collide with/overwrite the first's `CrmKBDocument` row.
- `data/kb/layouts/` (the layout-delta docs + `operator-to-layout.md` + `README.md`) must live **outside** `data/kb/global/`, so `_seed_crm_kb`'s recursive sweep (which only walks `data/kb/global/`) never touches it — this is what keeps contradictory layout deltas from being bundled platform-wide.
- `operator-to-layout.md` and any `README.md` are reference material only — **never** ingested into any bot-facing KB.
- Known pre-existing test-failure baseline (unrelated to this work, confirmed identical across the whole session): `test_chat_routes.py::test_claim_session_and_agent_ws`, `test_prompts.py::test_chatbot_prompt_has_scope_guardrails`, plus the other pre-existing failures/errors documented in `CLAUDE.md` and every prior plan this session. Test command: `.venv/bin/python -m pytest tests/unit -q`.

---

### Task 1: Relocate the frontend KB content

**Files:**
- Move: `data/kb/frontend KB/global/*.md` (11 files) → `data/kb/global/frontend-ui/` (renamed with a `ui-` prefix)
- Move: `data/kb/frontend KB/layouts/*.md` (10 files) + `data/kb/frontend KB/operator-to-layout.md` + `data/kb/frontend KB/README.md` → `data/kb/layouts/` (flat, no rename)
- Delete: the now-empty `data/kb/frontend KB/` directory (including its untracked `.DS_Store`)

**Interfaces:**
- Produces: `data/kb/global/frontend-ui/ui-01-login-register.md` … `ui-11-technical-troubleshooting.md` (auto-seeded into every CRM by the existing `_seed_crm_kb`, `src/main.py` — no code change, since it already does `Path("data/kb/global").rglob("*")`). `data/kb/layouts/layout-1.md` … `layout-9.md`, `layout-sports.md`, `operator-to-layout.md`, `README.md` (NOT auto-seeded — outside `data/kb/global/`).

- [ ] **Step 1: Create the destination directories and move the `global/` files with the required rename**

```bash
mkdir -p "data/kb/global/frontend-ui"
mv "data/kb/frontend KB/global/01-login-register.md"              "data/kb/global/frontend-ui/ui-01-login-register.md"
mv "data/kb/frontend KB/global/02-kyc-verification.md"             "data/kb/global/frontend-ui/ui-02-kyc-verification.md"
mv "data/kb/frontend KB/global/03-wallet-deposits.md"               "data/kb/global/frontend-ui/ui-03-wallet-deposits.md"
mv "data/kb/frontend KB/global/04-withdrawals.md"                   "data/kb/global/frontend-ui/ui-04-withdrawals.md"
mv "data/kb/frontend KB/global/05-casino.md"                        "data/kb/global/frontend-ui/ui-05-casino.md"
mv "data/kb/frontend KB/global/06-sports.md"                        "data/kb/global/frontend-ui/ui-06-sports.md"
mv "data/kb/frontend KB/global/07-matka-lottery.md"                 "data/kb/global/frontend-ui/ui-07-matka-lottery.md"
mv "data/kb/frontend KB/global/08-bonuses.md"                       "data/kb/global/frontend-ui/ui-08-bonuses.md"
mv "data/kb/frontend KB/global/09-profile-settings.md"              "data/kb/global/frontend-ui/ui-09-profile-settings.md"
mv "data/kb/frontend KB/global/10-responsible-gaming.md"            "data/kb/global/frontend-ui/ui-10-responsible-gaming.md"
mv "data/kb/frontend KB/global/11-technical-troubleshooting.md"     "data/kb/global/frontend-ui/ui-11-technical-troubleshooting.md"
```

Expected: `ls "data/kb/frontend KB/global"` now shows only `.DS_Store` (or is fully empty if that file doesn't exist on this machine); `ls data/kb/global/frontend-ui` shows exactly 11 files, all prefixed `ui-`.

- [ ] **Step 2: Move the layout files + reference docs into `data/kb/layouts/` (flat, no rename)**

```bash
mkdir -p "data/kb/layouts"
mv "data/kb/frontend KB/layouts/"*.md "data/kb/layouts/"
mv "data/kb/frontend KB/operator-to-layout.md" "data/kb/layouts/operator-to-layout.md"
mv "data/kb/frontend KB/README.md" "data/kb/layouts/README.md"
```

Expected: `ls data/kb/layouts` shows exactly 13 files (`layout-1.md` … `layout-9.md`, `layout-sports.md`, `operator-to-layout.md`, `README.md`).

- [ ] **Step 3: Remove the now-empty source directory**

```bash
rm -rf "data/kb/frontend KB"
```

Expected: `ls data/kb` no longer shows `frontend KB`; it now shows (among the pre-existing entries) both `global` and `layouts`.

- [ ] **Step 4: Verify no filename collision was introduced and the doc_id-length claim holds**

```bash
# Confirm every stem under data/kb/global/ (recursive) is unique — a duplicate
# here means _seed_crm_kb would silently collide two CrmKBDocument rows.
find data/kb/global -name "*.md" -exec basename {} \; | sort | uniq -d
```
Expected: **no output** (empty = no duplicate stems anywhere in the tree).

```bash
python3 -c "
longest = max((f for f in __import__('pathlib').Path('data/kb/global').rglob('*.md')), key=lambda p: len(p.stem))
doc_id = f'crm_kb_betstudio_{longest.stem}'
print(doc_id, len(doc_id))
assert len(doc_id) <= 50, 'doc_id exceeds CrmKBDocument.id String(50) column width'
print('OK, fits under String(50)')
"
```
Expected: prints the longest resulting `doc_id` and its length, followed by `OK, fits under String(50)`.

- [ ] **Step 5: Confirm the working tree change and commit**

```bash
git add data/kb/global/frontend-ui data/kb/layouts
git status --short data/kb/
```
Expected: shows `data/kb/frontend KB/` gone, `data/kb/global/frontend-ui/` (11 new files, staged `A`), and `data/kb/layouts/` (13 new files, staged `A`).

```bash
git commit -m "feat(kb): relocate the frontend UI/navigation KB — global docs auto-seed per-CRM, layout deltas stay reference-only"
```

---

### Task 2: Extend `scripts/ingest_kb.py` with a `--file` option

**Files:**
- Modify: `scripts/ingest_kb.py`
- Test: `tests/unit/test_ingest_kb_script.py` (new)

**Interfaces:**
- Consumes: nothing from Task 1 (this task is independent — it doesn't read the relocated files, just adds a CLI capability).
- Produces: `scripts.ingest_kb._collect_files(dir_arg: str | None, file_args: list[str] | None, exts: set[str]) -> list[pathlib.Path]` — a pure function extracted from `main()`'s existing file-collection logic, now also accepting explicit `--file` paths. `main()`'s `--dir` argument becomes optional; `--file` (new, `nargs="+"`) is optional; at least one of the two is required (exit code 2 with a clear stderr message if neither is given).

- [ ] **Step 1: Write the failing test**

```python
"""tests/unit/test_ingest_kb_script.py"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ingest_kb_script",
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "ingest_kb.py",
)
ingest_kb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest_kb)

_EXTS = {".md", ".txt"}


def test_collect_files_dir_only_sweeps_recursively(tmp_path: pathlib.Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "sub" / "b.md").write_text("b")
    (tmp_path / "ignored.pdf").write_text("x")

    files = ingest_kb._collect_files(str(tmp_path), None, _EXTS)

    assert {f.name for f in files} == {"a.md", "b.md"}


def test_collect_files_file_only_ignores_extension_filter(tmp_path: pathlib.Path) -> None:
    weird = tmp_path / "notes.weird"
    weird.write_text("x")

    files = ingest_kb._collect_files(None, [str(weird)], _EXTS)

    assert files == [weird]


def test_collect_files_missing_explicit_file_raises(tmp_path: pathlib.Path) -> None:
    missing = str(tmp_path / "does-not-exist.md")

    with pytest.raises(FileNotFoundError):
        ingest_kb._collect_files(None, [missing], _EXTS)


def test_collect_files_dir_and_file_combined_deduplicates(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.md").write_text("a")
    dupe_path = str(tmp_path / "a.md")  # already swept by --dir

    files = ingest_kb._collect_files(str(tmp_path), [dupe_path], _EXTS)

    assert len(files) == 1
    assert files[0].name == "a.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_ingest_kb_script.py -v`
Expected: FAIL — `AttributeError: module 'ingest_kb_script' has no attribute '_collect_files'`

- [ ] **Step 3: Extract `_collect_files` and update `main()`**

Replace the whole file's body from the `_DEFAULT_EXTS` line through the end of `main()`'s file-collection block (i.e. everything up to and including `if not files: ...` in the current file) with:

```python
_DEFAULT_EXTS = ".md,.txt,.pdf,.docx,.csv"


def _collect_files(
    dir_arg: str | None, file_args: list[str] | None, exts: set[str],
) -> list[pathlib.Path]:
    """Resolve the final file list from --dir (recursive, extension-filtered
    sweep) and/or --file (explicit paths — ingested regardless of the
    extension filter, since naming one directly is an intentional choice).
    De-duplicates by resolved path (a --file path may already be covered by
    an overlapping --dir sweep)."""
    files: list[pathlib.Path] = []
    if dir_arg:
        root = pathlib.Path(dir_arg)
        files.extend(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
        )
    if file_args:
        for f in file_args:
            p = pathlib.Path(f)
            if not p.is_file():
                raise FileNotFoundError(f"--file path not found: {f}")
            files.append(p)
    seen: set[pathlib.Path] = set()
    unique: list[pathlib.Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return sorted(unique)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", help="directory of docs to ingest (recursive, extension-filtered)")
    ap.add_argument(
        "--file", nargs="+",
        help="one or more explicit file paths to ingest (bypasses the --ext filter)",
    )
    ap.add_argument("--base-url", required=True, help="e.g. https://voicebot.biznexis.in")
    ap.add_argument("--token", required=True, help="tenant bearer token (vox_...)")
    ap.add_argument("--ext", default=_DEFAULT_EXTS, help="comma-separated extensions (applies to --dir only)")
    args = ap.parse_args()

    if not args.dir and not args.file:
        print("error: at least one of --dir or --file is required", file=sys.stderr)
        return 2

    exts = {e if e.startswith(".") else f".{e}" for e in args.ext.split(",")}
    base = args.base_url.rstrip("/")
    try:
        files = _collect_files(args.dir, args.file, exts)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not files:
        print(f"no files with {sorted(exts)} under {args.dir}", file=sys.stderr)
        return 1
```

Everything after this block (the `ok = 0` line through the end of `main()` and the `if __name__ == "__main__":` guard) is unchanged — it already just iterates `files`, uploads each, and reports.

Also update the module docstring's example section to show both use cases (append after the existing example, before the `Note:` line):

```python
"""Bulk-ingest a directory of knowledge-base docs into a tenant's KB via the API.

Posts every supported file under ``--dir`` to ``POST /api/v1/knowledge/ingest``
with the tenant's bearer token. Use it to load the global KB (the chatbot then
answers general queries from it via RAG).

Example (bulk):
  python scripts/ingest_kb.py \
    --dir /path/to/kb/global \
    --base-url https://voicebot.biznexis.in \
    --token vox_xxxxx

Example (single file — e.g. onboarding a tenant's layout-specific KB doc,
see data/kb/layouts/operator-to-layout.md):
  python scripts/ingest_kb.py \
    --file data/kb/layouts/layout-1.md \
    --base-url https://voicebot.biznexis.in \
    --token vox_xxxxx

Note: the deployed app must have the Gemini-embeddings build (PR #132) live, or
ingest returns 500 (no embedder).
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_ingest_kb_script.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Confirm the existing bulk-`--dir` behavior still works end-to-end (no live server needed — just argument wiring)**

Run: `.venv/bin/python scripts/ingest_kb.py --base-url http://x --token t`
Expected: prints `error: at least one of --dir or --file is required` to stderr and exits with code 2 (confirms the new required-one-of-two check fires correctly without needing a live server).

- [ ] **Step 6: Run the broader suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: same known baseline as every prior plan this session, plus this task's 4 new passing tests — zero new failures.

- [ ] **Step 7: Commit**

```bash
git add scripts/ingest_kb.py tests/unit/test_ingest_kb_script.py
git commit -m "feat(kb): add --file option to ingest_kb.py for single-doc tenant ingestion"
```

---

### Task 3: Document the standing onboarding process

**Files:**
- Modify: `docs/chatbot.md`

**Interfaces:**
- Consumes: `data/kb/layouts/operator-to-layout.md` (Task 1), `scripts/ingest_kb.py --file` (Task 2) — referenced by path/example only, no code dependency.

- [ ] **Step 1: Add a subsection to the Knowledge Base section**

In `docs/chatbot.md`, the "Knowledge base:" section currently ends with this paragraph (the last bullet under that heading, right before the next `## ` heading):

```
  voice path.
```

Append a new bullet immediately after it, still under the same "Knowledge base:" list:

```markdown
- **Frontend UI/navigation KB:** `data/kb/global/frontend-ui/` (11 files,
  `ui-`-prefixed to keep every stem under `data/kb/global/` unique — see the
  design spec) documents UI/navigation behavior common to every layout; it
  rides the same CRM-wide auto-seed as the backend docs above, no extra
  step needed. `data/kb/layouts/layout-N.md` (one per frontend package —
  `layout-1` … `layout-9`, `layout-sports`) documents UI **deltas** specific
  to one layout — these are NOT auto-seeded (they'd contradict each other
  across tenants on different layouts) and must be ingested per-tenant.
  **Standing process — do this whenever a new tenant/operator is
  registered:** look up its layout in
  `data/kb/layouts/operator-to-layout.md` (a mechanical operator → layout
  mapping, reference-only — never ingest this file or `data/kb/layouts/README.md`
  into any bot KB), then run:
  ```bash
  python scripts/ingest_kb.py \
    --file data/kb/layouts/layout-N.md \
    --base-url <that tenant's base URL> \
    --token <that tenant's bearer token>
  ```
  This lands the doc as a normal tenant-scoped `KBDocument` — no new entity,
  no per-layout admin UI; see
  `docs/superpowers/specs/2026-07-24-frontend-kb-design.md` for the full
  rationale.
```

- [ ] **Step 2: Commit**

```bash
git add docs/chatbot.md
git commit -m "docs(kb): document the per-tenant layout-KB onboarding process"
```

## Final verification (after all tasks)

```bash
.venv/bin/python -m pytest tests/unit -q
```
Expected: known baseline only (see Global Constraints) — zero new failures across all 3 tasks combined.

```bash
find "data/kb/frontend KB" 2>&1
```
Expected: `find: data/kb/frontend KB: No such file or directory` (confirms full relocation, nothing left behind).

```bash
find data/kb/global -name "*.md" -exec basename {} \; | sort | uniq -d
```
Expected: no output (no duplicate stems anywhere in the auto-seeded tree).

## Manual follow-up (not a task in this plan — requires a live deployment + real tenant secret)

Ingest `data/kb/layouts/layout-1.md` into the `stage` tenant's KB (confirmed match via `operator_id`
`ab858a8c-7ad4-47d2-a0b7-05ee93f8f134` = `jupiter-app` = layout-1 in `operator-to-layout.md`):

```bash
python scripts/ingest_kb.py \
  --file data/kb/layouts/layout-1.md \
  --base-url <stage's live base URL> \
  --token <stage's tenant bearer token>
```
