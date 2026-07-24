# ChatBot module

Customer-facing inbound support agent: text + images, RAG-grounded, with
function-calling tools (knowledge search, CRM APIs, escalate, offer-call) and a
handoff to a browser voice call. One per tenant. The platform exposes APIs; the
CRM builds its own UI (a reference widget ships at `/chat-widget`).

It complements the VoiceBot: VoiceBot = outbound (we call leads); ChatBot =
inbound (customers chat with us).

## Architecture (reuses the existing platform)

| Concern | Reuses |
|---|---|
| Agent | `src/agents/chatbot.py` `ChatBotAgent` (text counterpart to `VoiceBotAgent`) |
| LLM | provider registry (`get_llm`); Gemini multimodal + function-calling (`src/providers/llm/gemini.py`) |
| RAG | `src/rag/*` (ingestion, `LocalEmbedder` multilingual MiniLM, `HybridRetriever` = FAISS + BM25, context builder + hallucination guard) |
| Vector store | per-tenant `FAISSAdapter` (index path `data/faiss/{tenant_id}/`) via the runtime registry |
| Sessions | Redis `SessionStore` (history/state) + Postgres (`chat_sessions`, `chat_messages`) |
| Tools | builtin `ToolSpec`s + per-tenant `chat_tools` rows; tokens encrypted in `tenant_secrets` |
| Voice handoff | the browser voice bridge (`make_browser_bridge_factory`) |
| Escalation | signed outbound webhooks (`tenant_events` / `emit_tenant_event`) |

Chat and a voice call for the same tenant share the **same** per-tenant
retriever instance, so a doc ingested via the knowledge API is retrievable by
that tenant's chatbot. Tenants are isolated by separate FAISS index paths.

## HTTP / WS API (`/api/v1`)

Sessions & conversation:
- `POST /chat/sessions` (tenant bearer) → `{session_id, greeting, ws_url}`. The
  `session_id` is the capability for the socket.
- `GET  /chat/sessions`, `GET /chat/sessions/{id}` (detail + messages)
- `WS   /chat/ws/{session_id}` — the conversation. The socket resolves the tenant
  from the session row (no creds over the WS). Client frames:
  `{type:message,text}`, `{type:image|video|audio, data|media_url, mime, text}`,
  `{type:end}`. Media frames take base64 `data` OR an https `media_url` (fetched
  server-side, SSRF-guarded, 1MB cap); `mime` is required with `data`, inferred
  from the response content-type with `media_url`; `text` is an optional caption
  (image/video only). `media_url` on a `type:message` frame is NOT honored —
  attachments must use a media frame type (or the REST upload below).
  Server frames: `typing`, `message` (text/sources/suggestions/action),
  `audio_ack`, `escalation`, `call_offer`, `ended` (with summary), `error`.
- `POST /chat/{session_id}/upload` — multipart image/video (alternative to base64).
- `POST /chat/message` (tenant bearer) — single-turn HTTP for async channels (WhatsApp).
- `GET  /chat/history/{session_id}` — Redis history.

Knowledge base:
- `POST /knowledge/ingest` (multipart: pdf/docx/txt/md/csv) → parse → chunk →
  embed → per-tenant store; row in `kb_documents`. Unchanged by the CRM-level
  KB work below — same table, same fields, same behavior.
- `GET/DELETE /knowledge/documents[/{id}]`, `POST /knowledge/query` (debug), `GET /knowledge/stats`.
- **CRM-level KB (shared docs, admin-managed):** a `Crm` can also hold its own
  shared knowledge base — `CrmKBDocument` rows, scoped by `crm_id` (required,
  not nullable) — managed on the CRM sub-resource:
  `POST/GET /api/v1/crms/{crm_id}/kb/ingest|documents`,
  `DELETE /api/v1/crms/{crm_id}/kb/documents/{id}`,
  `GET /api/v1/crms/{crm_id}/kb/documents/{id}/download`. Admin-only
  (`require_admin`), 404 on an unknown `crm_id` — same shape as the `Crm`
  CRUD API. These replace the old flat, fully-global `POST
  /knowledge/platform-ingest` / `GET|DELETE /knowledge/platform-documents[/{id}]`
  endpoints, which are gone.
- **Retrieval is always mixed, not tenant-wins-outright:** for every chat/voice
  turn a tenant's own `KBDocument` chunks AND its linked CRM's `CrmKBDocument`
  chunks are BOTH searched and the results merged by relevance score — this is
  a deliberate difference from the CRM-tools precedence below, where the
  tenant's own tools win outright over the CRM catalog. KB is additive
  (tenant docs *and* shared CRM/company docs together are useful at once);
  tools are substitutive (a tool implementation is either the tenant's own or
  the CRM default, not both). A tenant with no linked CRM (`crm_id is None`)
  simply gets tenant-docs-only results — never an error, same graceful
  degradation as `resolve_crm_tools()`. `GET /knowledge/stats` reflects both
  scopes for a CRM-linked tenant. Note: the *voicebot's* one-shot boot-time
  KB context (built once per call via `_build_kb_context`, used by every
  telephony/browser/S2S bridge factory) is CRM-scoped only — it never mixes
  in the tenant's own docs, a faithful carry-over of the prior platform-only
  behavior. The tenant+CRM merge described above applies to the chat agent's
  per-turn retrieval (`ChatBotAgent`, `/knowledge/query`), not this one-shot
  voice path.
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

CRM tools:
- `POST /chat/tools` — register endpoints the bot may call (`name`, `endpoint`,
  `method`, `auth_type`, `auth_token`, `parameters{name:{type,description,source}}`).
  The token is stored **encrypted** in `tenant_secrets`, never returned.
- `GET/DELETE /chat/tools[/{name}]`.
- `GET /chat/tools/resolved` — the tools this tenant will ACTUALLY get on its
  next chat turn, computed fresh (bypasses the CRM-tools cache): reflects the
  linked-`Crm`-catalog path too (`source: "crm_catalog"`), not just this
  tenant's own registered `chat_tools` rows (which is all `GET /chat/tools`
  sees).
- **Where the shared catalog lives:** a tenant that isn't running its own
  `chat_tools` gets its tools from the `Crm` entity it's linked to
  (`tenant.settings.crm_id` → `crms`/`crm_tools` DB rows) instead of a
  hardcoded dict + env var. A `Crm` row holds the shared `base_url`,
  `auth_type`, and `events_webhook_url_template`; `CrmTool` rows are its tool
  catalog (endpoint/method/parameters), joined onto `base_url` at resolve
  time. CRM entities are admin-managed via `GET/POST/PATCH
  /api/v1/crms[/{id}]` (no per-tenant auth — a platform-level admin
  resource), not through the tenant API.
- **Precedence is unchanged**: (1) the tenant's own `chat_tools` rows, if any,
  win outright — `source: "tenant"`; (2) otherwise, the linked `Crm`'s
  catalog is used — `source: "crm_catalog"` (renamed from the old
  `"platform_fallback"`, same mechanism, now DB-backed instead of a
  hardcoded catalog + `PLATFORM_CRM_*` env vars); (3) a tenant with neither
  gets `source: "none"`. Auth is still resolved per-tenant either way — a
  linked `Crm`'s tools always use the *tenant's own* `crm:api_token` /
  `crm:x_api_key` secrets and `operator_id`, never anything shared across
  tenants.
- Auth headers: `auth_type`/token produce a single header (`Authorization:
  Bearer <token>` or `X-API-Key: <token>`); for `crm_catalog` tools,
  `auth_type` comes from the linked `Crm` row while the token itself is
  still the tenant's own `crm:api_token`. Independently of that, the tenant's
  own `crm:x_api_key` secret — when set — is **always** sent as `X-API-Key`,
  alongside whatever the `auth_type`/token mechanism already produces (the
  live CRM requires both headers together). Encrypted at rest exactly like
  `crm:api_token`.

Voice handoff:
- `POST /chat/{session_id}/call` → summarizes the chat, stashes context under a
  10-min Redis token, returns `{call_url, call_id}`.
- `WS /chat/voice?tenant=&handoff=` — always-on browser voice; the voicebot
  starts with the chat summary in its lead context.

## The agent turn (agentic loop)

`enable_tools=True` in the prod factory. Per turn the LLM may call:
`search_knowledge_base` (RAG), `escalate_to_human`, `offer_voice_call`, or any
registered CRM tool. Loop: generate (with tools, text mode — Gemini rejects
json+tools) → execute tool calls → feed results back → repeat → final text.
Sources come from the search results; the hallucination guard runs only when a
search happened (a greeting / CRM answer is legitimately ungrounded). Images are
passed to Gemini natively; video extracts key frames (PyAV) or degrades to a
text note.

## Startup wiring (`src/main.py` lifespan)

- `chat.set_chatbot_factory(make_chatbot_factory(registry, sessionmaker))`,
  `chat.set_chat_sessionmaker(...)`, `chat.set_chat_handoff_store(...)`
- `knowledge.set_retriever_factory(lambda t: registry.retrievers.get(t))`
- `set_browser_bridge_factory(make_browser_bridge_factory(..., handoff_store=...))`
  is wired **always** so `/chat/voice` works; the dev console's own routes stay
  behind `VOX_DEV_CONSOLE`.

Provider keys (`GEMINI_API_KEY`, etc.) are platform-level; only telephony +
CRM-tool tokens are per-tenant (encrypted).

## DB

`chat_sessions`, `chat_messages` (Alembic `0004`); `chat_tools` (`0005`);
`kb_documents` (`0001`); `crms`/`crm_tools` + `tenants.crm_id` (`0009`, the
shared CRM-catalog entity described above); `crm_kb_documents` +
`knowledge_chunks.crm_id` (`0010`, the shared CRM-level KB described above —
renamed/backfilled from the old fully-global `platform_kb_documents`
table added in `0006`). Run `alembic upgrade head`.

## End-to-end test (local)

1. Set `GEMINI_API_KEY`; `alembic upgrade head`; start the app.
2. `POST /api/v1/chat/sessions` with a tenant token → open `/chat-widget`,
   paste the token, Start chat.
3. `POST /api/v1/knowledge/ingest` a product doc → ask about it → grounded answer
   with sources.
4. Send an image in the widget → the agent describes it.
5. `POST /api/v1/chat/tools` a CRM endpoint → ask a question that needs it → the
   bot calls it and uses the result.
6. Ask for a human → `escalation` frame + signed `chat.escalated` webhook to the
   tenant's events URL.
7. `POST /api/v1/chat/{id}/call` → open the `call_url` → the voice agent greets
   with the chat context.

Tests: `tests/unit/test_chat_routes.py`, `test_chatbot_agent.py`,
`test_chatbot_tools.py`, `test_chat_tools_routes.py`, `test_chat_tool_executor.py`,
`test_chat_escalation.py`, `test_knowledge_routes.py`;
`tests/integration/test_chatbot_e2e.py`, `test_chatbot_multitenant.py`.
