# Chat Support System — Product Requirements Document

**Audience:** CRM Architecture team  
**Authors:** AI Platform team  
**Status:** Proposed — for review and discussion  
**Related docs:** `coordination-service-prd.md`, `coordination-service-interface.md`

---

## Overview

The Chat Support System (CSS) is the CRM team's full-stack product for managing customer support delivered over chat. It owns everything between the customer's message and a resolution: ticket creation, routing decisions, agent queues, live agent chat, and post-interaction analytics.

CSS is what prior documents called "CRM Backend" — but it also includes CRM's own internal surfaces: the agent console (where human agents handle live chats) and the admin/supervisor dashboard (routing rules, SLA configuration, reports). The customer-facing chat widget remains a separate CRM Frontend deployment.

CSS integrates with the Coordination Service (CS) for all AI-powered sessions. For direct-human sessions, CSS handles everything itself without involving CS.

---

## Goals

- Receive chat sessions from CRM Frontend and route them correctly (AI, human, or hybrid).
- Manage the full lifecycle of a support interaction: ticket creation → assignment → resolution → analytics.
- Give human agents a productive console to handle escalated and direct sessions.
- Give supervisors real-time visibility into queue, agents, and SLA health.
- Give managers structured reports to measure and improve support performance.
- Be fully configurable: routing rules, SLA policies, business hours, agent teams — all manageable without code changes.

## Non-Goals

- Customer-facing UI (chat widget) — owned by CRM Frontend.
- AI conversation intelligence — owned by AI Platform via CS.
- Voice channel adapters (telephony / WebRTC media stack) — owned by CS (future sprint). CSS signals voice calls by sending a `call_offer` frame; the adapter infrastructure is CS's concern.
- Billing or subscription management.
- A general-purpose CRM (contacts, deals, pipelines) — this is support-only.

---

## User Personas

| Persona | Role | Primary surface |
|---|---|---|
| **Customer** | End user seeking support | Chat widget (CRM Frontend) — not CSS directly |
| **Support Agent** | Handles live chat sessions with customers | Agent console (CSS web app) |
| **Support Supervisor** | Monitors queue, agents, and SLA in real time | Dashboard (CSS web app) |
| **Admin** | Configures routing, SLA, agents, teams, CS connection | Admin panel (CSS web app) |

---

## System Topology

```
Customer browser
        │  chat widget (CRM Frontend)
        │  POST /api/chat/start ──────────────────────────────► CSS
        │                                                         │
        │  (if AI/hybrid) ws_url from CS ◄────────────────────── │ ──► CS ──► AI Platform
        │  WS wss://cs.example.com/chat/ws/{session_id}           │
        │                                                         │
        │  (if human) ws_url from CSS ◄─────────────────────────  │
        │  WS wss://css.example.com/api/chat/ws/{ticket_id}       │
        ▼                                                         │
  Customer talks                                                  │
                                                        ◄─────── CS webhooks (session_started,
                                                                  escalation_requested,
                                                                  session_closed)
Agent browser
        │  CSS agent console
        │  WS /api/dashboard (supervisor)
        │  WS /api/chat/agent-ws/{id} → proxied to CS agent-ws (escalated)
        │  WS /api/chat/ws/{ticket_id} (direct human)
        ▼
       CSS
```

**Traffic rules:**
- CRM Frontend calls CSS for every chat session. CSS decides AI vs human.
- For AI sessions: CSS calls CS, which calls AI Platform. CRM Frontend WS connects to CS.
- For human sessions: CSS owns the WS directly. CS is not involved.
- CS calls CSS's webhook for all AI session lifecycle events.
- Human agents connect to CSS. For escalated (AI→human) sessions, CSS proxies the agent WS through CS.

---

## Module 1 — Ingestion and Session Orchestration

### 1.1 Chat start endpoint

CRM Frontend calls CSS when a customer opens the chat widget:

```
POST /api/chat/start
Authorization: Bearer {crm-frontend-css-token}
Content-Type: application/json

{
  "operator_id": "acme",
  "user_id":     "player-42",
  "user_name":   "Rahul",
  "language":    "hi",
  "metadata":    { "account_tier": "vip", "page": "/withdraw" }
}
```

CSS resolves the appropriate tenant token from `operator_id` internally — CRM Frontend never handles a tenant token. `user_name` is optional; if omitted, the AI greets generically.

CSS responds with everything CRM Frontend needs to open a chat:

```json
{
  "ticket_id": "TKT-9001",
  "operator_flag": "ai",
  "session_id": "cs_a1b2c3d4",
  "ws_url": "wss://cs.example.com/chat/ws/cs_a1b2c3d4",
  "greeting": "Hello Rahul, how can I help?"
}
```

For direct-human sessions (`operator_flag = "human"`):
```json
{
  "ticket_id": "TKT-9001",
  "operator_flag": "human",
  "session_id": null,
  "ws_url": "wss://css.example.com/api/chat/ws/TKT-9001",
  "greeting": "An agent will be with you shortly."
}
```
`greeting` is a configurable auto-reply set in admin. `ws_url` points to CSS's own WS — CS is not involved.

### 1.2 Operator flag decision

CSS evaluates routing rules (ordered by priority) to determine `operator_flag`. Each rule has a condition and a result:

| Condition type | Example |
|---|---|
| Business hours | `outside_hours → human` |
| Customer tier | `account_tier = "vip" → human` |
| Page / context | `page starts_with "/withdraw" → hybrid` |
| Queue depth | `queue_depth > 20 → human` (overflow to direct queue) |
| Catch-all | `→ ai` (default) |

If no rule matches, CSS uses the tenant's `default_operator_flag` from admin config.

### 1.3 Ticket creation

A ticket is created immediately on session start, before the customer sends any message:

- `status`: `ai_active` (for ai/hybrid) or `queued` (for human)
- `priority`: `medium` by default; rules may set it (e.g. VIP → `high`)
- `sla_response_due_at`, `sla_resolution_due_at`: computed from `SLAPolicy.first_response_minutes` and `SLAPolicy.resolution_minutes` for the priority level
- `customer_id` (mapped from `user_id`), `customer_name` (mapped from `user_name`), `language` from the request
- `metadata` stored for agent context

---

## Module 2 — Ticket Management

### 2.1 Ticket fields

| Field | Type | Notes |
|---|---|---|
| `id` | string | TKT-{nanoid} |
| `tenant_id` | string | |
| `status` | enum | see state machine |
| `priority` | enum | low / medium / high / urgent |
| `category` | string | from category library; set manually or by routing rule |
| `customer_id` | string | from CRM Frontend |
| `customer_name` | string | |
| `language` | string | ISO 639-1 |
| `assigned_agent_id` | string? | null until claimed |
| `team_id` | string? | null if unassigned to team |
| `sla_response_due_at` | datetime | first-response SLA deadline |
| `sla_resolution_due_at` | datetime | resolution SLA deadline |
| `sla_response_breached` | bool | set if first response missed |
| `sla_resolution_breached` | bool | set if resolution missed |
| `created_at` | datetime | |
| `first_response_at` | datetime? | set when agent first replies |
| `resolved_at` | datetime? | |
| `closed_at` | datetime? | |
| `abandoned` | bool | true if customer left before any agent replied |
| `summary` | text? | AI-generated at close (from session_closed webhook) |
| `tags` | string[] | |

### 2.2 Ticket status state machine

> **`hybrid` sessions:** A `hybrid` session starts identically to `ai` — ticket status is `ai_active` and the AI handles the conversation. The difference is intent: hybrid sessions are configured to escalate to a human agent whenever the AI decides to or the customer requests it. The state machine below is identical for both `ai` and `hybrid`.

```
new
 │
 ├──(operator_flag=ai/hybrid)────► ai_active ──(escalation_requested)──► queued
 │                                      │                                    │
 │                    (session_closed, AI resolved without escalation)        │
 │                                      ▼                                    │
 │                                   closed ◄── (24h timeout, any status)    │
 │                                         ◄── (abandonment — no agent claimed)
 │                                                                           │
 ├──(operator_flag=human)──────────► queued ◄────────────────────────────────┘
 │                                      │   ──(abandonment)──────────────────► closed
 │                               (agent claims)
 │                                      ▼
 │                                 in_progress ◄──── (transfer)
 │                                      │
 │                   ┌──────────────────┴──────────────────┐
 │                   ▼                                     ▼
 │                resolved                            closed
 │            (by agent action)              (session_closed webhook)
 │                   │
 │               (customer reopens by starting a new chat — linked to same ticket
 │                if within 24 hours; configurable reopen window, default 24 h)
 └───────────────────┘
```

**Terminal transitions by trigger:**

| Trigger | From | To |
|---|---|---|
| `session_closed` webhook, `mode_at_close=ai` | `ai_active` | `closed` |
| `session_closed` webhook, `mode_at_close=ai`, no agent claimed | `queued` | `closed` (abandoned) |
| `session_closed` webhook, `mode_at_close=human` | `in_progress` | `resolved` |
| Customer WS disconnect (direct-human, no agent ever replied) | `queued` | `closed` (abandoned) |
| Agent action: Resolve | `in_progress` | `resolved` |
| Agent action: Close | any | `closed` |
| 24h timeout (background job) | any non-terminal | `closed` |

### 2.3 Ticket messages

Every message in the conversation is stored:

| Field | Description |
|---|---|
| `role` | `customer`, `ai_agent`, `human_agent`, `system` |
| `content` | Text content |
| `media_url` | S3 key (if image/video/audio) |
| `media_mime` | MIME type |
| `ts` | Timestamp |
| `is_internal` | `true` for agent notes (not visible to customer) |

For AI sessions, CSS stores messages from the `session_closed` transcript webhook. For human sessions, CSS stores messages in real time as they flow through its own WS.

### 2.4 Ticket actions (agent)

| Action | Precondition | Effect |
|---|---|---|
| Claim | Ticket in `queued` | Assigns to agent; status → `in_progress` |
| Add note | Any status | Inserts internal message (`is_internal=true`) |
| Change priority | Any status | Updates priority + recomputes SLA |
| Change category/tags | Any status | Updates metadata |
| Transfer | `in_progress` | Reassigns to another agent or team; status stays |
| Resolve | `in_progress` | Status → `resolved`; records `resolved_at` |
| Close | Any status | Status → `closed`; records `closed_at` |
| Reopen | `resolved` | Status → `in_progress`; SLA restarts |

---

## Module 3 — Queue Management

### 3.1 Queue entry

When a ticket enters `queued` status (either direct-human or escalation from AI), a `QueueEntry` is created:

```
QueueEntry:
  ticket_id, priority, enqueued_at, claimed_at (null), claimed_by (null)
```

Queue is ordered by: `priority DESC, enqueued_at ASC`.

### 3.2 Agent assignment modes (configurable per tenant)

| Mode | Behaviour |
|---|---|
| **Manual claim** | Agents see the queue and claim tickets themselves (default) |
| **Round-robin** | CSS auto-assigns to the next available online agent |
| **Least-busy** | CSS auto-assigns to the agent with fewest active tickets |
| **Skill-based** | CSS matches ticket category to agent skills (Phase 6+) |

### 3.3 Agent availability

Agents set their status: `online` / `busy` / `offline`.
- `online`: available for auto-assignment and manual claim
- `busy`: visible in queue but not auto-assigned
- `offline`: hidden from queue

CSS enforces a `max_concurrent_tickets` per agent (configurable, default 3). Agents at capacity are treated as `busy` for auto-assignment.

### 3.4 Business hours and overflow

Routing rules may exclude human routing outside business hours. When `outside_hours` is a condition:
- Direct-human requests are accepted and queued (agents pick up next business day)
- Or: an auto-reply is sent ("Our team is offline until 9am IST") and ticket is `queued`

When queue depth exceeds a configurable threshold:
- Customer receives an estimated wait time ("~15 min wait")
- Supervisor receives a queue-overflow alert

---

## Module 4 — Agent Console (Web)

### 4.1 Layout

Three-panel layout:
1. **Left sidebar:** Inbox (assigned tickets), Queue (claimable), Search
2. **Center:** Active chat — conversation thread, composer
3. **Right panel:** Customer context, ticket metadata, actions

### 4.2 Inbox and queue

- **Inbox:** Tickets assigned to the logged-in agent (status: `in_progress`)
- **Queue:** Tickets in `queued` status — shows customer name, wait time, priority; agent can claim
- Real-time updates via WebSocket: new escalation badge, unread message count

### 4.3 Chat pane

For **escalated AI sessions** (AI→human):
- Agent connects via `WS /api/chat/agent-ws/{ticket_id}`, which CSS proxies to CS agent-ws
- History shown on connect (from CS history frame)
- Customer messages arrive as `customer_message` frames
- Agent sends `{"type":"reply","text":"..."}` frames
- Media (voice/image/video) renders inline

For **direct human sessions**:
- Agent connects via `WS /api/chat/ws/{ticket_id}` (CSS's own WS)
- Symmetric: CSS relays messages between customer and agent
- Same frame protocol as AI sessions so agent console code is reused
- **Call customer** action: agent clicks "Call" → CSS sends a `call_offer{transport: webrtc}` frame to the customer widget; the agent console handles WebRTC signalling on its side. `pstn` transport is not available in direct-human sessions.

### 4.4 Customer context panel (right sidebar)

- Customer name, ID, account tier, language
- Past tickets (last 5, with status and date)
- Current ticket metadata: priority, category, SLA countdown
- Ticket action buttons: note, priority, transfer, resolve
- Canned responses: searchable list; click to insert into composer

### 4.5 Agent status

Toggle in the header: Online / Busy / Offline. CSS broadcasts status change to the supervisor dashboard in real time.

---

## Module 5 — Webhooks Receiver (from CS)

CSS registers one webhook URL with CS per tenant:
```
POST https://css.example.com/webhooks/cs?tenant={slug}
```

CSS verifies the `X-CS-Signature` HMAC header on every inbound request.

### Event: `session_started`

```json
{
  "event": "session_started",
  "session_id": "cs_a1b2c3d4",
  "customer": { "name": "Rahul", "id": "player-42" },
  "event_id": "evt_2a7c4b1d"
}
```

CSS action: deduplicate on `event_id`; confirm ticket exists (it was created at `POST /api/chat/start`); start SLA timer.

### Event: `escalation_requested`

```json
{
  "event": "escalation_requested",
  "session_id": "cs_a1b2c3d4",
  "reason": "Customer requested human support",
  "summary": "Customer is asking about a 3-day withdrawal delay.",
  "customer": { "name": "Rahul", "id": "player-42" },
  "claim_url": "/chat/sessions/cs_a1b2c3d4/claim",
  "agent_ws_url": "/chat/agent-ws/cs_a1b2c3d4",
  "event_id": "evt_9f3b1a2c"
}
```

> **Note:** CS does not forward a `bo_available` field. CSS manages its own agent availability via business-hours config and queue depth — do not depend on `bo_available` in this webhook.

CSS actions:
1. Deduplicate on `event_id` — if already processed, return 200 and stop.
2. Transition ticket status: `ai_active` → `queued` (verify ticket is in `ai_active` first; if already `queued`, idempotent no-op)
3. Add `QueueEntry` for this ticket
4. Store `claim_url` and `agent_ws_url` into the `chat_sessions` row for this ticket (CSS uses these when an agent claims)
5. Store escalation `summary` as an internal system message on the ticket
6. Notify online agents via WebSocket push (new queue entry)
7. CRM Frontend is already notified via the `escalation` frame CS forwarded over the customer relay WS — CSS does not re-signal the customer.

### Event: `session_closed`

```json
{
  "event": "session_closed",
  "session_id": "cs_a1b2c3d4",
  "mode_at_close": "human",
  "summary": "Customer asked about withdrawal; resolved by agent.",
  "event_id": "evt_4c8d2f1e",
  "transcript": [
    { "role": "customer",    "text": "Hello",         "ts": "..." },
    { "role": "ai_agent",   "text": "Hello Rahul…",  "ts": "..." },
    { "role": "human_agent","text": "I see the issue","ts": "..." }
  ]
}
```

CSS actions:
1. Deduplicate on `event_id`.
2. Store transcript as `TicketMessage` rows.
3. Store `summary` on ticket.
4. Transition ticket based on `mode_at_close`:
   - `ai` → `ai_active` → `closed` (AI resolved without a human agent)
   - `human` → `in_progress` → `resolved`
5. Record `resolved_at` (human) or `closed_at` (AI).
6. Compute and store resolution metrics for analytics.
7. Trigger CSAT survey (if configured).

---

## Module 6 — Real-Time Dashboard (Supervisor)

Supervisor connects to `WS /api/dashboard`. CSS pushes updates as events occur.

### Live counters (updated on every state change)

| Metric | What it counts |
|---|---|
| Active AI sessions | Tickets in `ai_active` |
| Active human sessions | Tickets in `in_progress` |
| Waiting in queue | `QueueEntry` records with `claimed_at = null` |
| Oldest wait | `enqueued_at` of the oldest unclaimed QueueEntry |
| Agents online | Agents with status `online` |
| Agents busy | Agents with status `busy` |
| SLA response at risk | Tickets where `sla_response_due_at < now + 10 min` and `first_response_at` is null |
| SLA response breached | Tickets where `sla_response_due_at < now` and `first_response_at` is null |
| SLA resolution at risk | Tickets where `sla_resolution_due_at < now + 10 min` and status not in (resolved, closed) |
| SLA resolution breached | Tickets where `sla_resolution_due_at < now` and status not in (resolved, closed) |

### Live feeds

- **Escalation feed:** last 10 escalations (customer name, reason, wait time, assigned agent)
- **Resolution feed:** last 10 resolved tickets (agent, time to resolve, outcome)

### Hourly volume chart

Bar chart: chats started per hour for today. Updated every 5 minutes.

### Alerts (pushed to supervisor WS)

- Queue depth exceeds configured threshold
- SLA breach detected
- No agents online

---

## Module 7 — Reporting and Analytics

All reports support period filters: Today / This Week / This Month / Custom date range. All reports exportable as CSV and PDF.

### Report: Volume

| Metric | Description |
|---|---|
| Total chats | All sessions started |
| AI-only resolved | Closed with `mode_at_close = ai` |
| Escalated to human | `escalation_requested` events |
| Human-only | Direct-human sessions |
| Abandoned | Customer disconnected before resolution |
| By hour / day | Volume over time (chart) |

### Report: Resolution

| Metric | Description |
|---|---|
| AI resolution rate | AI-only resolved / total chats |
| Escalation rate | Escalations / total chats |
| Avg messages before escalation | From session start to `escalation_requested` |
| Human resolution rate | Resolved by agent / human sessions |
| Reopened rate | Tickets reopened after resolution |

### Report: Performance

| Metric | Description |
|---|---|
| Avg first response time | `first_response_at - enqueued_at` (human sessions) |
| Avg handle time | `resolved_at - claimed_at` (human sessions) |
| Avg resolution time | `resolved_at - created_at` (all sessions) |
| Avg AI session duration | For AI-only sessions |

### Report: Agent

Per agent, for the selected period:

| Metric | Description |
|---|---|
| Chats handled | Tickets assigned and closed |
| Avg handle time | |
| Avg first response | Time from claim to first reply |
| Resolution rate | Resolved / handled |
| CSAT score | Average if collected |
| Online hours | Total hours in `online` status |

### Report: Queue

| Metric | Description |
|---|---|
| Avg wait time | `claimed_at - enqueued_at` |
| Peak queue depth | Max concurrent QueueEntries in period |
| Abandonment rate | Customers who left while waiting |
| Wait time distribution | Histogram: <1 min / 1–5 min / 5–15 min / >15 min |

### Report: SLA

| Metric | Description |
|---|---|
| First-response compliance rate | Tickets with `first_response_at ≤ sla_response_due_at` / total human sessions |
| Resolution compliance rate | Tickets with `resolved_at ≤ sla_resolution_due_at` / total sessions |
| First-response breaches by priority | Count of `sla_response_breached=true` by priority |
| Resolution breaches by priority | Count of `sla_resolution_breached=true` by priority |
| Avg first-response breach overrun | Avg (`first_response_at − sla_response_due_at`) for breached tickets |
| Avg resolution breach overrun | Avg (`resolved_at − sla_resolution_due_at`) for breached tickets |
| Breaches by agent | Which agents had the most SLA breaches (either type) |
| Breaches by team | Which teams had the most SLA breaches (either type) |

---

## Module 8 — Admin and Configuration

### CS Integration

```
GET  /api/admin/cs-config
PUT  /api/admin/cs-config

{
  "cs_base_url": "https://cs.example.com",
  "cs_token": "cs_vox_...",        // CSS's Bearer token for calling CS
  "webhook_secret": "..."          // CSS verifies CS webhooks with this
}
```

### Routing Rules

Rules are evaluated in priority order (highest first). First match wins.

```json
{
  "name": "VIP customers → human",
  "priority": 100,
  "condition": {
    "operator": "AND",
    "clauses": [
      { "field": "metadata.account_tier", "op": "eq", "value": "vip" }
    ]
  },
  "operator_flag": "human",
  "ticket_priority": "high"
}
```

Condition fields: `language`, `metadata.*`, `business_hours`, `queue_depth`, `customer_history.ticket_count`.

### Business Hours

```json
{
  "timezone": "Asia/Kolkata",
  "schedule": {
    "mon": { "open": "09:00", "close": "21:00" },
    "tue": { "open": "09:00", "close": "21:00" },
    "sat": { "open": "10:00", "close": "18:00" },
    "sun": null
  },
  "outside_hours_message": "Our team is offline. We'll respond next business day."
}
```

### SLA Policies

```json
[
  { "priority": "urgent", "first_response_minutes": 15,  "resolution_minutes": 120  },
  { "priority": "high",   "first_response_minutes": 60,  "resolution_minutes": 480  },
  { "priority": "medium", "first_response_minutes": 240, "resolution_minutes": 1440 },
  { "priority": "low",    "first_response_minutes": 480, "resolution_minutes": 2880 }
]
```

### Agent Management

Admins can:
- Invite agents by email (email invite with password-set link)
- Set role: `agent` / `supervisor` / `admin`
- Assign to teams
- Set `max_concurrent_tickets` per agent
- Deactivate agents

### Team Management

Teams have a name and a list of agents. Routing rules can target teams. Queue can be filtered by team in the agent console.

### Canned Responses

```json
{
  "title": "Withdrawal delay — standard",
  "category": "Payments",
  "content": "Thank you for reaching out. Withdrawals typically process within 3–5 business days. Your reference number is...",
  "shortcut": "/withdraw-delay"
}
```

Agents can search by title, category, or shortcut and insert into the composer.

---

## CS Integration Contract

### CSS → CS

| Call | Endpoint | When |
|---|---|---|
| Create AI session | `POST /chat/sessions` | `operator_flag = ai/hybrid` |
| Get session detail | `GET /chat/sessions/{id}` | Supervisor detail view |
| Claim session | `POST /chat/sessions/{id}/claim` | Agent claims escalated ticket |
| Agent WS | `WS /chat/agent-ws/{id}?token=` | Agent enters live chat on escalated ticket |

CSS never calls AI Platform directly.

### CS → CSS (webhooks)

CSS registers per tenant: `POST https://css.example.com/webhooks/cs?tenant={slug}`

| Event | CSS action |
|---|---|
| `session_started` | Confirm/create ticket; start SLA |
| `escalation_requested` | Enqueue ticket; alert agents |
| `session_closed` | Store transcript; resolve ticket; compute metrics |

### CSS → CRM Frontend

| Call | What it returns |
|---|---|
| `POST /api/chat/start` | `{ticket_id, operator_flag, session_id?, ws_url, greeting?}` |
| `WS /api/chat/ws/{ticket_id}` | Direct human chat WS (no CS) |

---

## Full API Surface

### Public (CRM Frontend → CSS)

```
POST   /api/chat/start                ← session handoff; returns {session_id, ws_url, greeting}
WS     /api/chat/ws/{ticket_id}       ← direct-human chat WS (no CS)
POST   /api/chat/call                 ← get ephemeral voice call URL (websocket transport)
POST   /api/chat/upload               ← multipart media upload (alternative to base64 WS)
```

### Webhook (CS → CSS)

```
POST   /webhooks/cs
```

### Agent console

```
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/me
POST   /api/me/status

GET    /api/tickets                    ?status=&priority=&assigned_to=&from=&to=
GET    /api/tickets/{id}
POST   /api/tickets/{id}/claim
POST   /api/tickets/{id}/notes
PATCH  /api/tickets/{id}              {priority, category, tags}
POST   /api/tickets/{id}/resolve
POST   /api/tickets/{id}/transfer     {agent_id? | team_id?}
POST   /api/tickets/{id}/reopen
POST   /api/tickets/{id}/close

GET    /api/queue                      (unclaimed QueueEntries, ordered)
GET    /api/agents                     (online/busy/offline counts + list)
WS     /api/chat/agent-ws/{ticket_id}  (proxied to CS for escalated; direct for human)
WS     /api/dashboard                  (supervisor real-time feed)

GET    /api/reports/volume             ?period=&from=&to=&format=json|csv|pdf
GET    /api/reports/resolution         same
GET    /api/reports/performance        same
GET    /api/reports/agents             same
GET    /api/reports/queue              same
GET    /api/reports/sla                same
```

### Admin panel

```
GET/PUT          /api/admin/cs-config
GET/POST         /api/admin/routing-rules
GET/PATCH/DELETE /api/admin/routing-rules/{id}
GET/POST         /api/admin/sla-policies
GET/PATCH        /api/admin/sla-policies/{id}
GET/PUT          /api/admin/business-hours
GET/POST/PATCH   /api/admin/agents/{id}
POST             /api/admin/agents/invite
GET/POST/PATCH   /api/admin/teams/{id}
GET/POST/PATCH   /api/admin/canned-responses/{id}
GET/POST/PATCH   /api/admin/categories
GET/PUT          /api/admin/notifications
```

---

## Data Model

### Core tables

**tenants** — id, name, slug, plan, created_at

**tickets** — id, tenant_id, status, priority, category, customer_id, customer_name, language, assigned_agent_id, team_id, sla_response_due_at, sla_resolution_due_at, sla_response_breached, sla_resolution_breached, created_at, first_response_at, resolved_at, closed_at, summary, tags[], metadata (jsonb), abandoned (bool)

> `sla_response_due_at` and `sla_resolution_due_at` are computed from `SLAPolicy.first_response_minutes` and `SLAPolicy.resolution_minutes` at ticket creation. Two separate deadline fields are required because the SLA report measures both dimensions independently.

**ticket_messages** — id, ticket_id, role, content, media_url, media_mime, ts, is_internal

**chat_sessions** — id, ticket_id, type (ai/human), cs_session_id, cs_ws_url, cs_claim_url, cs_agent_ws_url, status, started_at, ended_at

> `chat_sessions.cs_session_id` is the authoritative CS session reference for this ticket. The `escalation_requested` webhook handler stores `claim_url` and `agent_ws_url` into this row. There is no separate `ai_session_id` on the ticket — use `chat_sessions.cs_session_id` joined via `ticket_id`.

> **Abandonment detection:** CSS detects abandonment when a customer WS disconnects (direct-human sessions) or when CS delivers a `session_closed` webhook with no agent ever having claimed the ticket and `mode_at_close = ai`. Set `tickets.abandoned = true` and emit an `abandoned` `ticket_event` so the Volume and Queue reports can count it.

**queue_entries** — id, ticket_id, priority, enqueued_at, claimed_at, claimed_by_agent_id

**agents** — id, tenant_id, name, email, password_hash, role, status, max_concurrent_tickets, team_id, created_at, last_seen_at

**teams** — id, tenant_id, name, routing_weight

**routing_rules** — id, tenant_id, name, priority, condition (jsonb), operator_flag, ticket_priority, is_active

**sla_policies** — id, tenant_id, priority_level, first_response_minutes, resolution_minutes

**business_hours** — id, tenant_id, timezone, schedule (jsonb), outside_hours_message

**canned_responses** — id, tenant_id, title, category, content, shortcut

**cs_config** — id, tenant_id, cs_base_url, cs_token (encrypted), webhook_secret (encrypted)

### Analytics tables (write-optimized)

**ticket_events** — id, ticket_id, tenant_id, event_type, agent_id, occurred_at, metadata (jsonb)
(append-only; reports are generated by aggregating over this table)

**agent_status_log** — id, agent_id, status, changed_at
(used for "online hours" reporting)

---

## Tech Stack Recommendation

| Layer | Choice | Rationale |
|---|---|---|
| Backend API | Python 3.12 / FastAPI | Consistent with AI Platform and CS; async-native; WS support |
| Agent console + admin + dashboard | React 18 + TypeScript | Industry standard; native WS; component libraries (Ant Design / MUI) |
| Primary database | PostgreSQL 16 | Relational structure suits tickets, agents, SLA |
| Real-time / queue | Redis | Agent status, queue pub/sub, session map, dashboard WS fan-out |
| Background jobs | Celery + Redis | SLA timers, nightly report pre-computation, email alerts |
| Report storage | Materialized views + Redis cache | Nightly refresh for heavy aggregations; on-demand for current day |
| Media storage | S3-compatible | Ticket attachments (same bucket as AI Platform or separate) |
| Email | SendGrid / SES | Agent invite emails, SLA breach alerts |
| Deployment | Docker → Northflank | Consistent with AI Platform and CS |

---

## Implementation Phases

### Phase 1 — Foundation (Weeks 1–2)

**Deliverable:** Tickets are created from AI sessions; lifecycle events handled.

- Database schema: all tables above
- Auth: agent login (email + password + JWT), role middleware
- Tenant model and config
- `POST /api/chat/start`: operator flag evaluation, CS session creation, ticket creation
- `POST /webhooks/cs`: handle `session_started`, `escalation_requested`, `session_closed`
- Ticket CRUD APIs (no UI yet)
- Basic SLA timer (background job: mark `sla_response_breached` / `sla_resolution_breached` when overdue)

### Phase 2 — Agent Queue and Console (Weeks 3–4)

**Deliverable:** Agents can claim and handle escalated sessions end-to-end.

- Agent status API (`POST /me/status`)
- Queue API (`GET /api/queue`)
- Claim flow: `POST /tickets/{id}/claim` → CSS calls CS `/claim` → returns CS agent-ws URL
- `WS /api/chat/agent-ws/{ticket_id}`: proxies to CS agent-ws (for escalated sessions)
- Agent console React app: inbox, queue, chat pane (escalated sessions), ticket actions
- Real-time: WS push for new queue entries and customer messages

### Phase 3 — CRM Frontend Handoff and Direct Human WS (Week 5)

**Deliverable:** Full session handoff from customer click to agent or AI; direct human sessions work.

- Routing rule evaluation engine
- Business hours check
- `WS /api/chat/ws/{ticket_id}`: CSS's own WS for direct human sessions
- Agent console: add direct-human chat (same UI, different WS backend)
- CRM Frontend integration testing

### Phase 4 — Real-Time Dashboard (Week 6)

**Deliverable:** Supervisors have live visibility into queue, agents, and SLA.

- `WS /api/dashboard`: subscribe to live counter updates and alert events
- Dashboard React page: live counters, escalation feed, hourly chart, SLA alerts
- Supervisor role access control

### Phase 5 — Reporting (Weeks 7–8)

**Deliverable:** All six reports available; CSV and PDF export.

- `ticket_events` append pipeline wired into all state transitions
- Report computation queries (volume, resolution, performance, agent, queue, SLA)
- Materialized view refresh job (nightly for prior periods; on-demand for today)
- Redis caching layer for report endpoints
- CSV export (streaming)
- PDF export (WeasyPrint or headless Chrome)
- Report UI pages (React)

### Phase 6 — Admin Panel and Polish (Weeks 9–10)

**Deliverable:** System fully self-administrable without code changes.

- CS config panel
- Routing rules editor (condition builder UI)
- SLA policy editor
- Business hours editor
- Agent invite + management UI
- Team management UI
- Canned responses editor
- Category/tag library editor
- Notification settings (email alerts for SLA breach, queue overflow)
- Agent context panel: customer past tickets
- Canned response search and insert in composer

---

## Open Questions

1. **Customer identity verification** — Does CSS receive a signed JWT from CRM Frontend that proves `user_id`, or does CSS trust the user_id as passed? (Affects security posture and auth design.)

2. **CSAT collection** — Should CSS collect a post-chat satisfaction score from the customer? If yes: after which session types (AI-only / human)? Via what mechanism (star rating in the widget, SMS, email)?

3. **Omnichannel** — Is chat the only support channel in scope now? Should the data model be designed to accommodate future email or WhatsApp tickets in the same queue, or is chat-only acceptable?

4. **Direct-human WS ownership** — For `operator_flag = human`, CSS runs its own WebSocket. Should this eventually move to CS so CS owns all channel relay? Or is CSS's own WS acceptable long-term?

5. **Report pre-computation vs on-demand** — Nightly pre-compute (fast, slightly stale) or on-demand aggregation (always fresh, slower for large tenants)? Or a hybrid: pre-compute for prior periods, on-demand for today?

6. **Email and push notifications** — Email alerts for SLA breach and queue overflow are in scope. Are Slack webhooks or push notifications also needed?

7. **Multi-tenant deployment model** — Single CSS deployment serving all tenants (with tenant_id everywhere), or one deployment per tenant (simpler isolation, higher ops overhead)?

8. ~~**Reopen policy**~~ — *Resolved:* CSS links a new chat to the same ticket if started within a configurable window (default: 24 hours of resolution). This is reflected in the §2.2 state machine. Confirm the default window with the support team.

9. **Skill-based routing** — Is agent skill/specialisation (e.g. payments, technical, VIP) needed in Phase 1, or is it acceptable to add in a later sprint?

10. **Audit log** — Is a full audit trail of who changed what on each ticket required (compliance, legal)? If yes, this affects the Phase 1 schema.
