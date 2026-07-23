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
  embed → per-tenant FAISS; row in `kb_documents`.
- `GET/DELETE /knowledge/documents[/{id}]`, `POST /knowledge/query` (debug), `GET /knowledge/stats`.

CRM tools:
- `POST /chat/tools` — register endpoints the bot may call (`name`, `endpoint`,
  `method`, `auth_type`, `auth_token`, `parameters{name:{type,description,source}}`).
  The token is stored **encrypted** in `tenant_secrets`, never returned.
- `GET/DELETE /chat/tools[/{name}]`.
- `GET /chat/tools/resolved` — the tools this tenant will ACTUALLY get on its
  next chat turn, computed fresh (bypasses the CRM-tools cache): reflects the
  `PLATFORM_CRM_*` platform-fallback path too, not just this tenant's own
  registered `chat_tools` rows (which is all `GET /chat/tools` sees).
- Auth headers: `auth_type`/token produce a single header (`Authorization:
  Bearer <token>` or `X-API-Key: <token>`). The platform-catalog CRM
  additionally supports an independent, additive `crm:x_api_key` tenant
  secret — when set it is **always** sent as `X-API-Key`, alongside whatever
  the `auth_type`/token mechanism already produces (the live CRM requires
  both headers together). Encrypted at rest exactly like `crm:api_token`.

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
`kb_documents` (`0001`). Run `alembic upgrade head`.

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
