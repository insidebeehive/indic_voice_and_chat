# Per-tenant analytics breakdowns (campaign, call type, channel, provider)

**Date:** 2026-06-18 · **Status:** approved

## Scope
Enrich the existing per-tenant analytics (`GET /tenants/{id}/analytics`) — no
cross-tenant view. Purely additive.

## Backend (src/api/tenants.py — tenant_analytics)
Widen the query to also read `campaign_id`, `channel`, `telephony_provider`.
Add to `TenantAnalytics`:
- `by_campaign: dict[str,int]` — keyed by campaign **name** (campaign_id→name from
  the tenant's campaigns; null campaign_id → "none"; unknown id → the id).
- `by_channel: dict[str,int]` — voice / webconsole / softphone.
- `by_provider: dict[str,int]` — telephony_provider (null → "none").
(`by_agent_type` already exists = call type: voicebot vs human.)
Each dict totals to `total_calls` (using "none"/"unknown" buckets).

## Frontend (static/backoffice.html — loadAnalytics)
Add breakdown blocks: Campaign, Call type (agent), Channel, Provider — same
`key: count` row style as By status / By outcome.

## Testing
Seed conversations with mixed campaign/channel/provider/agent_type → assert each
breakdown's counts and that they sum to total_calls.
