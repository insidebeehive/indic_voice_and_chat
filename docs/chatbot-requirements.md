# AI Chat-Support Platform — Requirements (draft, for later)

Status: **draft / not scheduled.** Prepared from a competitor admin demo
(`chat-demo` admin panel) plus what we can reuse from this repo. The key
differentiator vs the reference is **RAG** (the reference has none). When we pick
this up, run it through the brainstorming → spec flow before building.

## 1. Reference panel — feature inventory (what they have)

- **Engagement:** Tickets (number, status, priority, assignment, timestamps,
  filtering), live Chat (per-ticket thread, file uploads, **internal notes**,
  voice recording), **Call Requests** + **Call Logs** (queue, duration, status),
  **Customers** (active/inactive, ticket counts, contact history).
- **Automation:** Tags, Quick Replies, **Auto-Responses** (multi-language:
  English/Hindi/Bengali/Marathi/Telugu/Tamil…).
- **Administration:** Admin/agent management, Activity Logs, **System Monitoring**
  (DB / Redis / message-queue / WebSocket health), Settings.
- **Settings:** profile, password, notification prefs, **APK distribution**,
  **WhatsApp integration**.
- **Agent presence:** Online / Away / Busy.
- **Dashboard KPIs:** open / closed / urgent tickets, avg response time.
- **UX:** keyboard shortcuts (Ctrl+K/I/Enter), bulk import/export.
- **Gap:** no AI/RAG — auto-responses are static/rule-based.

## 2. What we already have to build on (this repo)

- **Multi-tenancy:** per-tenant config + encrypted secrets, resolver, backoffice
  admin (register/edit tenants, provider-scoped creds, analytics breakdowns).
- **Provider registry:** pluggable LLM (Gemini/Groq/Anthropic), STT, TTS — reuse
  for the chatbot LLM + the RAG answer model.
- **RAG scaffold:** `src/rag/*` (retriever, context_builder, BM25) + `faiss-cpu`
  vector store config per tenant — currently a scaffold, to be built out.
- **ChatBot scaffold:** `src/agents/chatbot.py` (untouched scaffold).
- **Outbound events webhooks** (signed) — reuse for CRM ticket/chat events.
- **Telephony + voicebot** — the reference's "Call Requests / Call Logs" maps
  directly onto our calls/conversations + voicebot (a real integration edge).
- **Analytics pattern** — per-tenant breakdowns (status/outcome/channel/…),
  directly reusable for ticket/chat dashboards.

## 3. Proposed requirements (ours)

### 3.1 Channels & widget
- Embeddable **web chat widget** (per-tenant key, themable), real-time over
  WebSocket (we already proxy WS on the same TLS endpoint).
- **WhatsApp** channel (we have a `whatsapp` provider slot; currently `fake`).
- Optional mobile app (the reference ships an APK) — defer.

### 3.2 Conversations & tickets
- Ticket model: status (open/pending/closed), priority, assignment, tags,
  timestamps; threaded messages; **internal notes** (agent-only); attachments.
- **Bot-first** handling with **escalation/handoff to a human agent** (presence:
  online/away/busy), and a clean bot↔human transcript.
- Customer directory: profile, history, linked tickets, linked **calls**.

### 3.3 AI + **RAG** (the differentiator — missing in the reference)
- **Per-tenant knowledge base:** ingest PDFs/DOCX/URLs/FAQ; chunk + embed into the
  tenant-namespaced vector store (faiss); re-index on update.
- **Retrieval-augmented answers:** retrieve top-k chunks → augment the LLM prompt
  → answer **with citations**; configurable confidence threshold.
- **Fallbacks:** low-confidence / no-hit → canned reply or human handoff (never
  hallucinate). Multi-language (match the customer's language, like the voicebot).
- **KB management UI** in the backoffice: upload/list/delete sources, see index
  status, test-query a question, preview retrieved chunks.
- Reuse the provider registry so the answer LLM + embeddings are per-tenant
  configurable.

### 3.4 Automation
- Tags, Quick Replies, and AI **auto-responses backed by RAG** (vs the reference's
  static rules) — suggested replies an agent can accept/edit.

### 3.5 Admin / ops
- Agent & admin management with roles; activity/audit logs.
- Settings: profile, notifications, channel config (WhatsApp), KB sources.
- System monitoring: DB / Redis / queue / WS health (we have `/health` degraded
  signals to build on).

### 3.6 Analytics
- Dashboard KPIs (open/closed/urgent, avg first-response + resolution time),
  **bot-deflection rate** (resolved by RAG without a human), CSAT, by-channel /
  by-tag / by-agent breakdowns — reuse the analytics breakdown pattern.

### 3.7 Integrations
- Signed **outbound webhooks** for ticket/message lifecycle (reuse the call-event
  pattern) → tenant CRM.
- **Voicebot/telephony tie-in:** "request a call" from chat → our outbound
  voicebot; surface call logs/outcomes against the customer/ticket.

## 4. Decisions to settle when we start (brainstorm inputs)
- Ticketing depth: full ticketing system vs lightweight conversation store?
- Embeddings provider + vector store at scale (faiss local vs hosted).
- Bot autonomy: suggest-only (agent-in-the-loop) vs fully autonomous with handoff.
- Reuse the existing backoffice app vs a dedicated agent console.
- WhatsApp provider (Infobip/Twilio/Meta Cloud API?).

## 5. Out of scope (for the first cut)
Mobile app/APK, billing, omnichannel beyond web + WhatsApp, full RBAC.
