# Remove Telnyx and Infobip Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely remove the Telnyx and Infobip telephony providers — both of which are currently broken (uninstantiable, missing an abstract-method implementation) — from the codebase, config, env files, and docs.

**Architecture:** Delete the two adapter files and their tests outright, remove their two registrations from the provider factory (`src/providers/__init__.py`), strip their entries from the two `.env` files and the cost catalog, and remove their dedicated docs. No interface changes, no other provider is touched.

**Tech Stack:** Python 3, pytest.

## Global Constraints

- Branch is `stage` (confirmed clean, checked out). Do NOT create a new branch — direct commits are this project's normal workflow.
- Run `.venv/bin/python -m pytest tests/unit -q` after the task. Baseline immediately before this change: 24 failed, 1155 passed, 1 skipped, 22 errors. Expect the failed+error count to drop by roughly 21 (both `test_telnyx_adapter.py` and `test_infobip_adapter.py` are deleted outright), with zero new failures anywhere else. Do not trust these exact numbers as gospel — re-measure fresh.
- No Alembic migration — this task touches no DB schema.
- `.env` contains real, live secrets (it's gitignored, never committed) — when editing it, remove lines by key name only; never copy actual secret values into any commit message, code comment, or this plan's own report file.
- Do not touch: `docs/livekit-sip-integration-plan.md`, `docs/sip-didlogic-integration-plan.md` (incidental, forward-looking mentions of `telnyx.py` as a sizing/pattern reference for unrelated future providers — out of scope), `docs/superpowers/plans/2026-07-23-crm-entity.md`, `docs/superpowers/plans/2026-07-23-crm-kb.md` (historical, already-implemented plan docs), `src/interfaces/telephony.py` and its Twilio/Exotel/Stringee implementations, and any already-seeded `provider_costs` DB rows for telnyx/infobip (harmless orphaned catalog entries — no migration, no direct DB access).

---

### Task 1: Delete the Telnyx and Infobip providers and every reference to them

**Files:**
- Delete: `src/providers/telephony/telnyx.py`
- Delete: `src/providers/telephony/infobip.py`
- Delete: `tests/unit/test_telnyx_adapter.py`
- Delete: `tests/unit/test_infobip_adapter.py`
- Delete: `docs/telnyx-trial-setup.md`
- Delete: `docs/infobip-trial-setup.md`
- Modify: `src/providers/__init__.py`
- Modify: `config/provider_costs.yaml`
- Modify: `.env` (gitignored, not committed — edit anyway, it's a local ops file)
- Modify: `.env.example`
- Modify: `docs/README.md`
- Modify: `tests/unit/test_dev_console.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the only task).
- Produces: `TELEPHONY_PROVIDERS` in `src/providers/__init__.py` shrinks from 5 entries (`twilio`, `exotel`, `stringee`, `infobip`, `telnyx`) to 3 (`twilio`, `exotel`, `stringee`). No other file in the codebase constructs `InfobipAdapter`/`TelnyxAdapter` directly or imports from `src.providers.telephony.telnyx`/`src.providers.telephony.infobip` — confirmed via a full-repo grep before this plan was written, so no other production code needs updating.

- [ ] **Step 1: Delete the two adapter files and their dedicated tests**

```bash
git rm src/providers/telephony/telnyx.py
git rm src/providers/telephony/infobip.py
git rm tests/unit/test_telnyx_adapter.py
git rm tests/unit/test_infobip_adapter.py
```

- [ ] **Step 2: Remove Telnyx/Infobip from the provider factory**

In `src/providers/__init__.py`, remove these two import lines (currently lines 27 and 29 — re-check with `grep -n "telnyx\|infobip" src/providers/__init__.py` since line numbers may have shifted by the time you run this):

```python
from src.providers.telephony.infobip import InfobipAdapter
```
and
```python
from src.providers.telephony.telnyx import TelnyxAdapter
```

Then change the `TELEPHONY_PROVIDERS` dict from:

```python
TELEPHONY_PROVIDERS: dict[str, type[ITelephonyProvider]] = {
    "twilio": TwilioAdapter,
    "exotel": ExotelAdapter,
    "stringee": StringeeAdapter,
    "infobip": InfobipAdapter,
    "telnyx": TelnyxAdapter,
}
```

to:

```python
TELEPHONY_PROVIDERS: dict[str, type[ITelephonyProvider]] = {
    "twilio": TwilioAdapter,
    "exotel": ExotelAdapter,
    "stringee": StringeeAdapter,
}
```

Everything else in the file (the `STT_PROVIDERS`/`LLM_PROVIDERS`/`TTS_PROVIDERS`/`VECTOR_STORE_PROVIDERS` dicts, `_lookup`, `get_telephony_provider`, `__all__`) is unchanged.

- [ ] **Step 3: Remove the two telephony-cost lines from the cost catalog**

In `config/provider_costs.yaml`, the `telephony:` section currently reads:

```yaml
telephony:
  twilio: 0.014
  exotel: 0.007
  stringee: 0.010
  infobip: 0.012
  telnyx: 0.0035
```

Change it to:

```yaml
telephony:
  twilio: 0.014
  exotel: 0.007
  stringee: 0.010
```

(This only affects which rows get seeded into the `provider_costs` DB table on next boot for a fresh/empty table — `seed_provider_costs` in `src/auth/seed.py` only inserts missing rows, so any rows already seeded for infobip/telnyx on an existing DB are left alone; that's fine, do not attempt to delete them.)

- [ ] **Step 4: Remove Infobip/Telnyx entries from `.env.example`**

In `.env.example`, the `# Telephony` section currently reads:

```
# Telephony
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
EXOTEL_API_KEY=
EXOTEL_API_TOKEN=
EXOTEL_ACCOUNT_SID=
# Infobip — per-account base URL (e.g. https://abc123.api.infobip.com),
# API key, and Calls Application id from the Infobip console.
INFOBIP_API_KEY=
INFOBIP_BASE_URL=
INFOBIP_APPLICATION_ID=
# Telnyx — bearer-token auth + Voice API Application id (Mission Control
# Portal → Voice → Programmable Voice → Applications).
TELNYX_API_KEY=
TELNYX_CONNECTION_ID=
# Stringee — REST API Key SID + Secret (Stringee dashboard → Project →
# API Key). Used to mint short-lived JWT X-STRINGEE-AUTH tokens.
STRINGEE_API_KEY_SID=
STRINGEE_API_KEY_SECRET=
```

Remove the two Infobip lines, their two comment lines, the two Telnyx lines, and their one comment line, leaving:

```
# Telephony
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
EXOTEL_API_KEY=
EXOTEL_API_TOKEN=
EXOTEL_ACCOUNT_SID=
# Stringee — REST API Key SID + Secret (Stringee dashboard → Project →
# API Key). Used to mint short-lived JWT X-STRINGEE-AUTH tokens.
STRINGEE_API_KEY_SID=
STRINGEE_API_KEY_SECRET=
```

- [ ] **Step 5: Remove Infobip/Telnyx entries from the real `.env`**

`.env` is gitignored (never committed) but still needs the same cleanup locally. Re-run `grep -n "INFOBIP\|TELNYX" .env` first to confirm the exact current line numbers before editing (this is a live ops file that may have drifted). You should find two groups:

1. In the `# Telephony` section: two comment lines introducing Infobip, then `INFOBIP_API_KEY=`, `INFOBIP_BASE_URL=`, `INFOBIP_APPLICATION_ID=`; then one comment line introducing Telnyx, then `TELNYX_API_KEY=<value>`, `TELNYX_CONNECTION_ID=<value>`. Delete all of these lines (both comment lines and both key blocks). Do not print, log, or copy the actual `TELNYX_API_KEY`/`TELNYX_CONNECTION_ID` values anywhere (not into a commit message, not into your task report) — just delete the lines.
2. Further down, under `# --- Per-tenant aliases for the dev tenant ---`: `TENANT_DEV_INFOBIP_KEY=${INFOBIP_API_KEY}`, `TENANT_DEV_INFOBIP_BASE_URL=${INFOBIP_BASE_URL}`, `TENANT_DEV_INFOBIP_APP_ID=${INFOBIP_APPLICATION_ID}`, `TENANT_DEV_TELNYX_KEY=${TELNYX_API_KEY}`, `TENANT_DEV_TELNYX_CONN=${TELNYX_CONNECTION_ID}`. Delete all five lines.

Leave every other line in `.env` untouched (including the Twilio/Exotel/Stringee blocks and their `TENANT_DEV_*` aliases, `DATABASE_URL`, `REDIS_URL`, etc.).

Note in your task report that any OTHER deployed environment's own `.env` (Stage, Production) needs this same manual cleanup — `.env` is never shared via git, so this edit only affects this local checkout.

- [ ] **Step 6: Remove the two doc links from `docs/README.md`**

In `docs/README.md`, the `## Setup & testing` section currently reads:

```markdown
## Setup & testing
- [Live testing](live-testing.md) — placing real calls, ngrok setup
- [Multi-tenant plan](multi-tenant-plan.md)
- [Stringee streaming](stringee-streaming.md)
- [Infobip trial setup](infobip-trial-setup.md)
- [Telnyx trial setup](telnyx-trial-setup.md)
```

Remove the last two lines, leaving:

```markdown
## Setup & testing
- [Live testing](live-testing.md) — placing real calls, ngrok setup
- [Multi-tenant plan](multi-tenant-plan.md)
- [Stringee streaming](stringee-streaming.md)
```

- [ ] **Step 7: Delete the two dedicated trial-setup docs**

```bash
git rm docs/telnyx-trial-setup.md
git rm docs/infobip-trial-setup.md
```

- [ ] **Step 8: Update the dev-console test that used "telnyx" as an example unsupported provider**

In `tests/unit/test_dev_console.py`, the test `test_place_call_rejects_unsupported_provider` currently reads:

```python
def test_place_call_rejects_unsupported_provider():
    _register_dev_tenant()
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "telnyx", "to_number": "+919999999999"})
        assert resp.status_code == 400
        assert "telnyx" in resp.json()["detail"]
    finally:
        set_tenant_resolver(None)
```

`telnyx` was already absent from the dev-console's supported place-call providers before this change (only `twilio`/`exotel`/`stringee` are supported there), so this test's behavior is unaffected either way — but since Telnyx is being deleted from the codebase entirely, replace the example with a clearly-fictional provider name so a future reader doesn't wonder whether Telnyx used to be supported:

```python
def test_place_call_rejects_unsupported_provider():
    _register_dev_tenant()
    try:
        resp = _client().post("/dev/place-call", json={
            "provider": "not_a_real_provider", "to_number": "+919999999999"})
        assert resp.status_code == 400
        assert "not_a_real_provider" in resp.json()["detail"]
    finally:
        set_tenant_resolver(None)
```

- [ ] **Step 9: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: no collection errors for `test_telnyx_adapter.py`/`test_infobip_adapter.py` (they no longer exist), `test_place_call_rejects_unsupported_provider` still passes, and the total failed+error count drops from the pre-task baseline (24 failed, 22 errors) by roughly 21, with no new failures introduced anywhere else. Also run `grep -rn "telnyx\|infobip" src/ tests/ config/ .env .env.example docs/README.md` (case-sensitive is fine here since real identifiers are lowercase) and confirm zero remaining matches across all of those paths (the two out-of-scope integration-plan docs and the two historical CRM plan docs are expected to still contain incidental mentions — that's fine, they're explicitly out of scope).

- [ ] **Step 10: Commit**

```bash
git add -A -- src/providers/__init__.py config/provider_costs.yaml .env.example docs/README.md tests/unit/test_dev_console.py
git add -A -- src/providers/telephony/telnyx.py src/providers/telephony/infobip.py tests/unit/test_telnyx_adapter.py tests/unit/test_infobip_adapter.py docs/telnyx-trial-setup.md docs/infobip-trial-setup.md
git status --short
```

Confirm the status output shows exactly the expected deletions (`D`) and modifications (`M`) — no unrelated files. `.env` is gitignored and will not appear in `git status`; it was still edited on disk per Step 5. Then commit:

```bash
git commit -m "$(cat <<'EOF'
remove(telephony): delete the Telnyx and Infobip providers

Both were already broken — ITelephonyProvider.redirect_to_stream was added
as an abstract method and neither adapter implemented it, so constructing
either raised TypeError. Rather than fix them, remove them: adapters,
tests, factory registrations, cost-catalog entries, env vars, and docs.
EOF
)"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — failed+error count drops by ~21 from the pre-task baseline (24 failed, 22 errors), no new failures.
- `grep -rn "telnyx\|infobip" src/ tests/ config/ .env .env.example docs/README.md` returns nothing.
- `git log --oneline -1` shows the new commit; `git status --short` is clean (aside from the pre-existing untracked `docs/voice-recording-scripts*.md` files, which are unrelated to this task and must not be touched).
