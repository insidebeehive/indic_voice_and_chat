# Backoffice campaign-script editor (structured, view + edit)

**Date:** 2026-06-18
**Status:** approved

## Why

The voicebot script lives in `campaigns.config_yaml` (a YAML blob in a DB column,
read live per call by `DbCampaignResolver`). Editing it today means hand-editing
YAML in the DB. Admins need a structured editor in the backoffice.

## Scope

View + edit **existing** campaigns (no create/delete). Structured form fields
(not raw YAML).

## Backend (`src/api/tenants.py`)

- `GET /api/v1/tenants/{tenant_id}/campaigns` (admin) → `{campaigns: [{id, name,
  status, script}]}` where `script` is parsed from each `config_yaml` via the
  existing `VoiceBotScript.from_campaign_yaml` + slot-agnostic parse, exposed as
  the structured fields below. Cross-tenant guard: only that tenant's campaigns.
- `PUT /api/v1/tenants/{tenant_id}/campaigns/{campaign_id}` (admin) → body =
  structured fields. Server:
  1. Loads the campaign (404 / cross-tenant 404 guard).
  2. Parses existing `config_yaml` → dict; locates the `campaign:` wrapper.
  3. Overlays provided fields into the `agent:` / `script:` sub-blocks and the
     top-level `name` — **preserving every key it doesn't model** (`slots`, `id`,
     `status`, etc.).
  4. Re-serializes to YAML and **validates** via `parse_campaign_yaml` (400 on
     failure).
  5. Saves `config_yaml` (+ `name`) and commits. Read live per call → no restart.

### Structured fields (`CampaignScriptIn`, all optional → overlay only provided)
| Field | YAML location |
|---|---|
| `name` | `campaign.name` (top level) |
| `agent_name` | `campaign.agent.name` |
| `company` | `campaign.agent.company` |
| `role` | `campaign.agent.role` |
| `personality` | `campaign.agent.personality` |
| `language` | `campaign.agent.language` |
| `gender` | `campaign.agent.gender` |
| `greeting` | `campaign.script.greeting` |
| `objective` | `campaign.script.objective` |
| `closing` | `campaign.script.closing` (string) |
| `conversation_style` | `campaign.script.conversation_style` |
| `max_turns` | `campaign.script.max_turns` |
| `talking_points` | `campaign.script.talking_points` (list) |
| `dos` | `campaign.script.dos` (list) |
| `donts` | `campaign.script.donts` (list) |
| `knowledge` | `campaign.script.knowledge` (dict[str,str]) |

`closing` is edited as a single string; a dict-closing campaign is flattened to
its first value on read and written back as a string (documented limitation).

## Frontend (`static/backoffice.html`)

New **Campaign** tab in the tenant detail. Campaign dropdown → form populated
from GET: scalars as inputs, `greeting`/`objective` as textareas, the three lists
as one-item-per-line textareas, `knowledge` as editable key/value rows. **Save** →
PUT → ✓ or the 400 validation message.

## Testing (TDD)

- GET returns parsed structured fields for a seeded campaign.
- PUT updates `greeting`/`company`/`talking_points`, **preserves `slots`**, rejects
  unparseable YAML (400), and persists (re-GET reflects it).

## Out of scope
Create/delete campaigns; raw-YAML editing; slot-schema editing (preserved, not
edited).
