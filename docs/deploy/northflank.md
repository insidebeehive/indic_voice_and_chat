# Deploying to Northflank

Two stages: **(1) dev-console deploy now** (live test URL, no addons), and **(2) production follow-up**
(Redis + Postgres + telephony). The image is built from the repo `Dockerfile`.

The Dockerfile copies `src/ config/ static/ alembic/`, installs the `voice` extra (so **SileroVAD** works,
not the rougher EnergyVAD fallback), and runs `uvicorn src.main:app --host 0.0.0.0 --port 8000`.

---

## Stage 1 — Dev console (no Redis/Postgres needed)

The app opens Redis/DB **lazily**, and `GET /health` returns **HTTP 200** even when they're down
(body says `"status":"degraded"`), so the service is healthy on Northflank without any addons. Session
persistence is degraded — fine for dev-console testing.

### 1. Project
Create a Northflank **Project**, region **Singapore** (closest available to your providers/callers).

### 2. Service (build + deploy)
Create a **Combined Service**:
- **Source:** GitHub repo `ManojpGit/indic_voice_and_chat`, branch `main`.
- **Build:** Dockerfile, path `/Dockerfile`.
- **Port:** `8000`, **public**, protocol **HTTP** (Northflank serves WebSockets over the same public
  HTTPS endpoint, so the dev console's `wss://…/api/v1/dev/voice` works).
- **Health check:** HTTP `GET /health` on port 8000 (returns 200).
- **Resources:** 0.5 vCPU / 1 GB is plenty for the dev console.

### 3. Secrets / env vars
Add these (a Northflank **Secret Group** linked to the service, or service env vars). Values are in your
local `.env`:

| Key | Value | Required |
|---|---|---|
| `VOX_DEV_CONSOLE` | `1` | yes — enables `/dev/voice` |
| `TENANT_DEV_GEMINI_KEY` | `AIza…` | yes — LLM |
| `TENANT_DEV_SARVAM_KEY` | `…` | yes — TTS |
| `TENANT_DEV_DEEPGRAM_KEY` | `…` | yes — streaming STT |
| `TENANT_DEV_GROQ_KEY` | `gsk_…` | yes — batch STT fallback |
| `TENANT_DEV_ANTHROPIC_KEY` | `sk-ant-…` | optional (only if you switch the LLM to Claude) |

Leave `REDIS_URL` / `DATABASE_URL` / `WEBHOOK_BASE_URL` **unset** for Stage 1 (defaults in
`config/default.yaml` keep boot happy; telephony isn't used here).

### 4. Deploy + test
Deploy, wait for build → healthy. Open `https://<service>.<project>.code.run/dev/voice` (Northflank
assigns the domain). **Use headphones.** Confirm: opening plays, you speak → it replies, outcome panel
on end.

### 5. Measure (the payoff of the latency metric)
Capture ~8–10 turns and compare the `"browser turn (stream)"` log lines (`endpoint_gap_ms`,
`tts_first_ms`, `total_ms`) against your local Dubai baseline — this tells you exactly how much Singapore
helped. (Northflank → service → Logs.)

---

## Stage 2 — Production (when ready)

### Redis (sessions)
Add a Northflank **Redis** addon; set `REDIS_URL` to the addon's `redis://…` connection string.

### Postgres (persistence)
Add a Northflank **PostgreSQL** addon. Set `DATABASE_URL` using the **asyncpg** driver — convert the
addon's URI from `postgresql://USER:PASS@HOST:PORT/DB` to
`postgresql+asyncpg://USER:PASS@HOST:PORT/DB` (the app uses the async driver).

Migrations run **automatically on container start** — the Docker `CMD` is
`alembic upgrade head && uvicorn …`, so every deploy applies pending migrations
before serving (fail-fast: a failed migration won't start a stale app). The DB
role only needs rights on an already-provisioned schema; `env.py` skips
`CREATE SCHEMA` when the schema already exists (first-time provisioning still
needs a role with database-level CREATE, or pre-create the schema once).

For a **multi-replica** service, drop the in-container step and run
`alembic upgrade head` as a single **pre-deploy Job** instead (Alembic has no
cross-process lock, so replicas shouldn't migrate concurrently).

### Telephony (real calls)
Set `WEBHOOK_BASE_URL=https://<service>.<project>.code.run/api/v1/telephony` and repoint your
Twilio/Exotel number's webhooks to that URL. Mumbai (not Singapore) would be materially better for the
live call audio path if it ever becomes available on your provider — see
`docs/latency-llm-stt-experiments.md` (Experiment 2).

#### Stringee (turn-based IVR)
If a tenant uses Stringee:
- Set `telephony.provider: stringee` in the tenant's config.
- Add env `STRINGEE_API_KEY_SID` and `STRINGEE_API_KEY_SECRET` (shared across tenants).
- In the Stringee dashboard, set the number's **Answer URL** to
  `https://<service>.<project>.code.run/api/v1/telephony/stringee/answer` and **Status URL** to
  `https://<service>.<project>.code.run/api/v1/telephony/stringee/status/<tenant_slug>`.
- Ensure `WEBHOOK_BASE_URL` is set (Stringee must be able to fetch hosted audio and post event webhooks).
- **Note:** Stringee turn-based IVR is half-duplex with higher latency than Twilio/Exotel streaming,
  and single-instance only (no horizontal scaling).

### Tenant API auth (optional)
If you expose the tenant REST API publicly, set `TENANT_DEV_API_TOKENS=<comma,separated,tokens>` and the
admin tokens per `src/main.py` (`TENANT_<SLUG>_API_TOKENS`).

---

## Notes / gotchas
- **WebSockets:** the dev console + telephony both use WS. Northflank HTTP services proxy WS on the same
  public TLS endpoint — no extra config; the client builds `wss://` from `location.host`.
- **`/health` is always 200** (degraded vs ok in the body) — don't use the body's `status` as the
  health gate; the HTTP code is the gate.
- **Image size:** base + `voice` extra (faiss-cpu, onnxruntime). No torch/sentence-transformers (the
  `embeddings` extra is omitted — RAG isn't needed for the voice agent).
- **Secrets hygiene:** never bake keys into the image; use Northflank secret groups. `.env` is gitignored.
