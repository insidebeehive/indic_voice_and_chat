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
| RAG | `src/rag/*` (ingestion, `GeminiEmbedder` 384-dim multilingual embeddings, `HybridRetriever` = vector search + BM25, context builder + hallucination/no-grounding/unverified-data guards, plus an outbound PII guard (`apply_pii_guard`) that runs last, on every reply and suggested followup, redacting mobile numbers/emails/bank account numbers) |
| Vector store | per-tenant, config-selected (`vector_store.provider`): `pgvector` by default, or file-backed `faiss` as an alternative — via `src/providers/get_vector_store` and the runtime registry. The CRM-level shared KB (below) is **pgvector-only**: FAISS has no CRM-level tier by design, so a CRM with `vector_store.provider != pgvector` simply gets no CRM-shared KB. `LocalEmbedder` (`sentence-transformers`) also exists in `src/rag/embeddings.py` but isn't what's wired into the live retriever factory. |
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
  telephony/browser/S2S bridge factory) uses the same `[crm, tenant]` merge
  as chat — all 6 voice call sites (4 in `src/bootstrap.py`, 2 in
  `src/api/dev_console.py`) build it from both the CRM-wide retriever and
  the tenant-scoped retriever, so a tenant's own docs (including any
  product-module content it has opted into, see below) are included in
  voice calls too, not just chat's per-turn retrieval (`ChatBotAgent`,
  `/knowledge/query`).
- **Bundled KB packs (opt-in per CRM):** the backend docs above and a
  companion frontend/UI-navigation set live together under
  `data/kb/packs/<pack-name>/` (e.g. `data/kb/packs/betting-default/`, with
  the UI docs in its `frontend-ui/` subfolder, 8 files, `ui-`-prefixed to
  keep every stem under the pack directory unique — see the design spec).
  A `Crm` row's `bundled_kb_pack` column names which pack (if any) gets
  auto-seeded into that CRM's shared KB at boot (`_seed_crm_kb` in
  `src/main.py`); a CRM with `bundled_kb_pack` unset gets no bundled docs at
  all. Seeding is all-or-nothing per pack — a CRM that opts in gets both the
  backend and frontend-ui docs together, no extra step needed.
  `data/kb/layouts/layout-N.md` (one per frontend package —
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
- **Product-module KB (opt-in per tenant):** `data/kb/modules/` (6 ingestible
  docs — one backend doc plus its `ui-`-prefixed UI-help counterpart for each
  of casino, sports betting, and matka/lottery — plus a reference-only
  `README.md`) is a third tier, independent of the CRM-level bundled-pack
  opt-in above: it is **NOT** seeded automatically for any CRM, regardless
  of that CRM's `bundled_kb_pack` choice — not every operator offers casino,
  sports, and matka. A tenant opts in explicitly, either via
  `POST /api/v1/knowledge/ingest-layout` with the key
  `casino`, `sports`, or `matka`, or from the backoffice's **Ingest KB doc**
  dropdown under the **Product modules** optgroup. One key ingests that
  vertical's **pair** of files as two separate tenant-scoped `KBDocument` rows
  — never concatenated into one — because each file keeps its own filename,
  which is what `_VOICE_KB_PRIORITY` (`src/rag/context_builder.py`) matches
  doc priority on. Ids are deterministic (`layout_<key>_<file stem>`), so
  re-ingesting the same key updates the existing rows instead of duplicating
  them. The allow-list of ingestible keys lives in `_INGESTIBLE_DOCS`
  (`src/api/knowledge.py`) — that dict lookup *is* the path-traversal guard,
  so a request value must never be turned into a filesystem path any other
  way. `data/kb/modules/README.md` is reference-only and never ingestible.

CRM tools:
- `POST /chat/tools` — register endpoints the bot may call (`name`, `endpoint`,
  `method`, `auth_type`, `auth_token`, `parameters{name:{type,description,source}}`).
  The token is stored **encrypted** in `tenant_secrets`, never returned.
- `GET/DELETE /chat/tools[/{name}]`.
- `GET /chat/tools/resolved` — the FULL set of tools this tenant will
  ACTUALLY get on its next chat turn, computed fresh (bypasses the
  CRM-tools cache): builtin tools, CRM tools, and the deposit-verification
  tool, in that order. Each entry's `kind` field is `"builtin"`,
  `"crm"`, or `"deposit_verification"`. Builtin tools
  (`search_knowledge_base`, `escalate_to_human`) are
  unconditionally present on every tenant's every turn. CRM tools reflect
  the linked-`Crm`-catalog path too (`source: "crm_catalog"`), not just this
  tenant's own registered `chat_tools` rows (which is all `GET /chat/tools`
  sees) — `source`/`crm_id` on the response describe this CRM resolution
  only, not whether builtin or deposit-verification tools are present. The
  deposit-verification tool appears exactly when it would be registered by
  the chatbot factory: DV enabled, a `webhook_url` set, its signing secret
  resolvable, and a sessionmaker available; its entry reports
  `auth_type: "hmac"` and `token_configured` reflecting that signing secret,
  not a bearer/API-key token.
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

Hot issues (live incident notices):
- `PUT/GET /hot-issues`: a tenant's own set, authenticated with `current_tenant`. That means the tenant's own token, or an admin token plus `X-Tenant-Slug`.
- `PUT/GET /crms/{crm_id}/hot-issues`: a CRM-wide set, admin only (`require_admin`). Every tenant linked to that CRM sees it.
- **Replace semantics:** a PUT replaces the scope's whole set in one transaction, and `{"issues": []}` clears it. A missing tenant or CRM row returns 404. The partner-facing contract (fields, limits, writing guidance) is in `docs/crm-api-contract.md` under "Hot Issues".
- **Storage:** notices live in their own `hot_issues` table, not the KB. The KB is reached only through `search_knowledge_base`, and voice truncates it to a budget, so a notice stored there could go unseen.
- **How the bot sees them:** `src/chatbot/hot_issues.py` loads the union of a tenant's own and its CRM's active notices, tenant first. It renders them as one delimited, defanged block that carries its own usage lead.
  - **Chat:** the block goes into the per-turn tail on every tools-path turn, so a change applies mid-session.
  - **Voice:** the block is added to the prompt once, at call start. A notice cleared mid-call still reaches the call already in progress.
- **Freshness:** the loader caches each scope for 30 s, and a write clears its own scope's entry in the same process. With more than one worker or replica, a change takes up to 30 s to appear.
- **Expiry:** `expires_at` resets to 24 h whenever a notice is re-sent without one.
- **Failure behaviour:** the loader never fails a turn. A DB error or a load slower than 1 s gives an empty set, with a warning logged.
- **Grounding:** figures in an active notice count as grounded for the unverified-data guard in chat and the sentence guard in voice. A figure the bot has relayed stays grounded through the conversation's prior turns even after the notice is cleared.

Bot voice by gender:
- **Config:** `TenantTTSConfig.voices` (`{"female": <voice id>, "male": <voice id>}`) holds an optional per-gender voice on `pipeline.chat_voice.tts` and `pipeline.tts`. It is set through `PATCH /tenants/{id}` (`pipeline.tts.voices`, `pipeline.chat_voice.tts.voices`), and `POST /tenants` (`tts.voices`) for `pipeline.tts`.
  - **Merging:** a PATCH merges per gender. Sending `""` for a gender removes it, and sending all genders as `""` clears the pair.
  - **Provider switch:** switching provider drops the pair, the same as `voice_id`.
- **Read-back:** `tts.voices` (raw) and `chat_voice.effective_voices`.
- **Selection:** the session's `bot_gender` (from `POST /chat/sessions`) picks the voice through `resolve_gender_voice` (`src/config_tenant.py`). This covers both chat voice-note replies and the chat-to-voice handoff call. With no voice for that gender, the existing `voice_id` is used.
  - **Logging:** a WARNING is logged when a pair is configured but lacks the requested gender. When there is no pair at all, it is a debug event only.
- **Precedence:** an explicit `?voice=` on the handoff still wins.
- **Script gender:** the handoff call's script gender stays `bot_gender`, so voice and grammar agree once a pair is set.
- Voice-note replies pass the resolved `voice_id` to every TTS provider.

Voice handoff:
- `POST /chat/{session_id}/call` → summarizes the chat, stashes context under a
  10-min Redis token, returns `{call_url, call_id}`.
- `WS /chat/voice?tenant=&handoff=` — always-on browser voice; the voicebot
  starts with the chat summary in its lead context.

## The agent turn (agentic loop)

`enable_tools=True` in the prod factory. Per turn the LLM may call:
`search_knowledge_base` (RAG), `escalate_to_human`, or any
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
