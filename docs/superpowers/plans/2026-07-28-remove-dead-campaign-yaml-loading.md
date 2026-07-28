# Remove Dead Campaign-YAML-File Loading Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the campaign-YAML-file loading code (`load_campaign`, `active_campaign_slug`, `seed_campaigns_if_empty`, and their two boot-time call sites in `src/main.py`) — dead since `config/campaigns/*.yaml` was deleted a month ago in commit `e3b6a60`, which left the file-reading code behind even though it's permanently a no-op now.

**Architecture:** Pure deletion, no behavior change (the deleted code has done nothing but log a "not found, using demo/skipping" warning on every boot for the last month). `parse_campaign_data`/`parse_campaign_yaml`/`LoadedCampaign` in `campaign_loader.py` are untouched — they're the live DB-backed campaign parsing logic (a campaign's config is stored as a YAML string in the `campaigns.config_yaml` DB column, unrelated to the deleted file-based path).

**Tech Stack:** Python 3, pytest.

## Global Constraints

- Branch is `stage` — re-verify with `git rev-parse --abbrev-ref HEAD` immediately before committing (this session had an incident where a commit landed on `main` because the working tree had silently drifted off `stage` earlier). Do not create a new branch.
- Run `.venv/bin/python -m pytest tests/unit -q` after the change. Baseline immediately before this task on `stage`: 17 failed, 1156 passed, 1 skipped, 0 errors. Expect: passed count drops by 7 (1156→1149, since the two deleted test files' 7 tests were all passing before deletion), failed count unchanged at 17 (all pre-existing/unrelated), zero new failures.
- No Alembic migration — no DB schema touched.
- Do not touch: `parse_campaign_data`, `parse_campaign_yaml`, `LoadedCampaign` (live), `seed_if_empty`/`sync_telephony_from_yaml` in `src/auth/seed.py` (different, still-live tenant/telephony seeding — do not confuse with `seed_campaigns_if_empty`), `config/tenants/*.yaml`/`config/default.yaml`/`config/provider_costs.yaml` (unrelated, live config), `tests/unit/test_campaign_resolver.py` (already covers the kept functions — run it as part of verification, don't edit it).

---

### Task 1: Delete the dead campaign-YAML-file loading path

**Files:**
- Modify: `src/dialogue/campaign_loader.py`
- Modify: `src/main.py`
- Modify: `src/auth/seed.py`
- Delete: `tests/unit/test_campaign_loader.py`
- Delete: `tests/unit/test_campaign_seed.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the only task).
- Produces: `src.dialogue.campaign_loader` no longer exports `active_campaign_slug`, `DEFAULT_CAMPAIGN_SLUG`, `DEFAULT_CAMPAIGNS_DIR`, or `load_campaign`. `src.auth.seed` no longer exports `seed_campaigns_if_empty`. `parse_campaign_data`, `parse_campaign_yaml`, `LoadedCampaign` keep their exact current signatures — nothing downstream of those three needs to change.

- [ ] **Step 1: Confirm you're on the right branch**

```bash
git rev-parse --abbrev-ref HEAD
```
Expected: `stage`. If it prints anything else, stop and report back — do not proceed or switch branches yourself.

- [ ] **Step 2: Rewrite `src/dialogue/campaign_loader.py`**

Current full file:

```python
"""Campaign-upfront script loading.

Reads one campaign YAML (selected by VOX_CAMPAIGN) into a script + slot schema
at startup. The active campaign drives every call this process handles. The
loader is campaign-agnostic — it only parses whatever the YAML declares.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema

log = logging.getLogger(__name__)

DEFAULT_CAMPAIGN_SLUG = "bharat_matka"
DEFAULT_CAMPAIGNS_DIR = Path("config/campaigns")


@dataclass
class LoadedCampaign:
    script: VoiceBotScript
    slots: SlotSchema


def active_campaign_slug() -> str:
    """The campaign slug to load at startup (env VOX_CAMPAIGN, default bharat_matka)."""
    return os.environ.get("VOX_CAMPAIGN", DEFAULT_CAMPAIGN_SLUG)


def parse_campaign_data(data: dict) -> LoadedCampaign:
    """Parse a campaign dict (with or without the top-level ``campaign:`` wrapper)
    into a script + slot schema. Shared by the YAML file loader and the DB-backed
    per-tenant resolver, so both interpret a campaign identically."""
    camp = data.get("campaign", data)  # tolerate with/without the wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    return LoadedCampaign(
        VoiceBotScript.from_campaign_yaml(merged),
        SlotSchema.from_campaign_yaml(camp.get("slots") or {}),
    )


def parse_campaign_yaml(text: str) -> LoadedCampaign:
    """Parse a campaign YAML string (e.g. a DB ``campaigns.config_yaml``)."""
    return parse_campaign_data(yaml.safe_load(text) or {})


def load_campaign(
    slug: str, campaigns_dir: Path = DEFAULT_CAMPAIGNS_DIR
) -> LoadedCampaign:
    """Load ``config/campaigns/<slug>.yaml`` into a script + slot schema.

    Missing/unreadable file -> warn and fall back to the demo script with an
    empty slot schema, so the app still boots.
    """
    path = campaigns_dir / f"{slug}.yaml"
    if not path.exists():
        from src.bootstrap import DEFAULT_DEMO_SCRIPT  # lazy: avoid import cost/cycle

        log.warning("campaign file not found: %s; using demo script", path)
        return LoadedCampaign(DEFAULT_DEMO_SCRIPT, SlotSchema())

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return parse_campaign_data(data)
```

Replace the entire file with:

```python
"""Campaign parsing.

A campaign's config is stored as a YAML string (a tenant's
``campaigns.config_yaml`` DB column); these functions turn that string into a
script + slot schema. Shared by every campaign consumer so they all interpret
a campaign identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema


@dataclass
class LoadedCampaign:
    script: VoiceBotScript
    slots: SlotSchema


def parse_campaign_data(data: dict) -> LoadedCampaign:
    """Parse a campaign dict (with or without the top-level ``campaign:`` wrapper)
    into a script + slot schema."""
    camp = data.get("campaign", data)  # tolerate with/without the wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    return LoadedCampaign(
        VoiceBotScript.from_campaign_yaml(merged),
        SlotSchema.from_campaign_yaml(camp.get("slots") or {}),
    )


def parse_campaign_yaml(text: str) -> LoadedCampaign:
    """Parse a campaign YAML string (e.g. a DB ``campaigns.config_yaml``)."""
    return parse_campaign_data(yaml.safe_load(text) or {})
```

(`logging`, `os`, `Path` imports are dropped along with `active_campaign_slug`/`DEFAULT_CAMPAIGN_SLUG`/`DEFAULT_CAMPAIGNS_DIR`/`load_campaign`, since nothing remaining in the file uses them.)

- [ ] **Step 3: Remove the two dead call sites in `src/main.py`**

Run `grep -n "seed_campaigns_if_empty\|campaign_loader\|campaign = load_campaign\|campaign loaded" src/main.py` first to get exact current line numbers (they may have shifted slightly since this plan was written).

3a. In the `from src.auth.seed import ...` line, remove `seed_campaigns_if_empty` from the import list. It currently reads:

```python
from src.auth.seed import seed_campaigns_if_empty, seed_if_empty, seed_provider_costs, sync_telephony_from_yaml
```

Change to:

```python
from src.auth.seed import seed_if_empty, seed_provider_costs, sync_telephony_from_yaml
```

3b. Delete this whole import line entirely (nothing else in `main.py` uses anything from `campaign_loader` after this task):

```python
from src.dialogue.campaign_loader import active_campaign_slug, load_campaign
```

3c. Delete these three lines entirely:

```python
    seeded_campaigns = await seed_campaigns_if_empty(sessionmaker)
    if seeded_campaigns:
        log.info("seeded default campaigns from VOX_CAMPAIGN", extra={"count": seeded_campaigns})
```

3d. Delete the `campaign = load_campaign(...)` line and the `log.info("campaign loaded", ...)` call that follows it — currently:

```python
    campaign = load_campaign(active_campaign_slug())
    log.info(
        "campaign loaded",
        extra={"slug": active_campaign_slug(), "agent": campaign.script.agent_name,
               "slots": list(campaign.slots.specs.keys())},
    )
    # Per-tenant campaign resolution: EVERY bridge (telephony + dev console)
```

Delete only the `campaign = load_campaign(...)` line and the 4-line `log.info(...)` call (5 lines total). Keep the `# Per-tenant campaign resolution: EVERY bridge (telephony + dev console)` comment and everything after it unchanged — that comment describes the real, live `DbCampaignResolver`-based resolution that happens right after, which this task does not touch.

- [ ] **Step 4: Remove `seed_campaigns_if_empty` from `src/auth/seed.py`**

Read the current file to find the exact boundaries of the function (starts at `async def seed_campaigns_if_empty(sessionmaker, campaigns_dir=None) -> int:`, ends right before the next top-level `def`/`async def`). Its current body:

```python
async def seed_campaigns_if_empty(sessionmaker, campaigns_dir=None) -> int:
    """Give every tenant a DB campaign migrated from the global ``VOX_CAMPAIGN``
    file, when they have none. Campaigns then diverge per-tenant via the
    ``/campaigns`` API. Idempotent (skips a tenant that already has a campaign).

    This backs the no-global-fallback resolution: every tenant needs a row, so
    the live call never falls back to a shared/global script.
    """
    from src.dialogue.campaign_loader import DEFAULT_CAMPAIGNS_DIR, active_campaign_slug

    base = campaigns_dir or DEFAULT_CAMPAIGNS_DIR
    slug = active_campaign_slug()
    path = base / f"{slug}.yaml"
    if not path.exists():
        log.warning("campaign seed: %s not found; skipping", path)
        return 0
    raw = path.read_text()
    data = yaml.safe_load(raw) or {}
    camp = data.get("campaign", data)
    name = camp.get("name") or (camp.get("agent") or {}).get("company") or slug

    seeded = 0
    async with sessionmaker() as session:
        tenant_ids = (await session.execute(select(Tenant.id))).scalars().all()
        for tid in tenant_ids:
            existing = (await session.execute(
                select(Campaign.id).where(Campaign.tenant_id == tid).limit(1))).first()
            if existing is not None:
                continue
            session.add(Campaign(
                id=f"camp_{tid}_default", tenant_id=tid, name=str(name),
                status="active", config_yaml=raw))
            seeded += 1
        await session.commit()
    if seeded:
        log.info("seeded default campaigns", extra={"count": seeded})
    return seeded
```

Delete this entire function.

Then check whether `import yaml` (near the top of `seed.py`) and the `Campaign` import (`from src.models.campaign import Campaign`) are still used anywhere else in the file with `grep -n "yaml\.\|Campaign(" src/auth/seed.py`. If `seed_campaigns_if_empty` was the only user of either, remove that import line too; if any other function in `seed.py` still references `yaml.safe_load` or constructs `Campaign(...)`, leave the corresponding import in place. Do the same check for the `Tenant` import if it's only used inside this function (`grep -n "Tenant\b" src/auth/seed.py`) — leave it if other functions use it too.

- [ ] **Step 5: Delete the two obsolete test files**

```bash
git rm tests/unit/test_campaign_loader.py
git rm tests/unit/test_campaign_seed.py
```

- [ ] **Step 6: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 17 failed, 1149 passed, 1 skipped, 0 errors (passed count drops from 1156 to 1149 — exactly the 7 tests deleted in Step 5 — failed count stays at 17, all pre-existing/unrelated). Also run `.venv/bin/python -m pytest tests/unit/test_campaign_resolver.py -v` specifically and confirm every test in it still passes (it covers the kept `parse_campaign_data`/`parse_campaign_yaml` functions and must be unaffected by this change).

Also run `grep -rn "load_campaign\|active_campaign_slug\|seed_campaigns_if_empty\|DEFAULT_CAMPAIGNS_DIR\|DEFAULT_CAMPAIGN_SLUG" src/ tests/` and confirm zero matches anywhere in the codebase.

- [ ] **Step 7: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # must print "stage" — stop if it doesn't
git add src/dialogue/campaign_loader.py src/main.py src/auth/seed.py tests/unit/test_campaign_loader.py tests/unit/test_campaign_seed.py
git status --short
```

Confirm the status output shows exactly: modified `src/dialogue/campaign_loader.py`, `src/main.py`, `src/auth/seed.py`; deleted `tests/unit/test_campaign_loader.py`, `tests/unit/test_campaign_seed.py`. No unrelated files. Then commit:

```bash
git commit -m "$(cat <<'EOF'
remove(campaign): delete the dead campaign-YAML-file loading path

config/campaigns/*.yaml was deleted a month ago (DB is sole source of
truth for campaigns), but the code that read those files was never
cleaned up — load_campaign(), active_campaign_slug(), and
seed_campaigns_if_empty() have been permanently hitting their
file-not-found fallback on every boot since. Delete them; the live
DB-backed path (parse_campaign_data/parse_campaign_yaml +
DbCampaignResolver) is untouched.
EOF
)"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — 17 failed, 1149 passed, 1 skipped, 0 errors (down from 1156 passed; failed count unchanged).
- `.venv/bin/python -m pytest tests/unit/test_campaign_resolver.py -v` — all pass, unaffected.
- `grep -rn "load_campaign\|active_campaign_slug\|seed_campaigns_if_empty\|DEFAULT_CAMPAIGNS_DIR\|DEFAULT_CAMPAIGN_SLUG" src/ tests/` — zero matches.
- `git status --short` clean except the pre-existing, unrelated untracked `docs/voice-recording-scripts*.md` files — do not touch those.
