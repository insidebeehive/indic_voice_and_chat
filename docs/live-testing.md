# Live Testing — Real Provider APIs + End-to-End Voice Call

This doc covers two flows:

1. **Provider smoke tests** — cheap (~cents), no phone call. Verifies API keys + adapter wiring against real Groq / Gemini / Sarvam / Twilio endpoints.
2. **End-to-end voice call** — places a real outbound Twilio call to your phone, runs the full agent loop with live STT/LLM/TTS.

Everything is gated behind explicit opt-in (`VOX_LIVE_TESTS=1`, ngrok URL, env vars) so it can never run in CI by accident.

---

## Prerequisites

### Accounts + keys

| Provider | Get a key                                       | Used for                       |
| -------- | ----------------------------------------------- | ------------------------------ |
| Groq     | https://console.groq.com/keys                   | STT (Whisper-large-v3)         |
| Gemini   | https://aistudio.google.com/app/apikey          | LLM (gemini-2.0-flash)         |
| Sarvam   | https://dashboard.sarvam.ai                     | TTS (bulbul / meera voice)     |
| Twilio   | https://console.twilio.com + buy a phone number | Outbound calls + Media Streams |

### Tools

```bash
# Public HTTPS+WSS tunnel for Twilio to reach your local server
brew install ngrok          # or download from https://ngrok.com
```

### Environment variables

Set these in your shell or in a local `.env` (which is gitignored):

```bash
# Provider keys for the dev tenant — replace with your own values, never commit
export TENANT_DEV_GROQ_KEY="gsk_..."
export TENANT_DEV_GEMINI_KEY="AIza..."
export TENANT_DEV_SARVAM_KEY="..."
export TENANT_DEV_TWILIO_SID="AC..."
export TENANT_DEV_TWILIO_TOKEN="..."

# Optional: API tokens that grant access to the dev tenant via /api/v1/*
export TENANT_DEV_API_TOKENS="dev-token-1"
```

Never commit these to the repo — `.env` is in `.gitignore`.

Inbound telephony webhooks are signature-verified using the tenant's existing
provider credentials, not a separate webhook secret: Twilio reuses
`TENANT_DEV_TWILIO_TOKEN` above (the same Auth Token Twilio signs webhooks
with), and Exotel/Stringee verify against `webhook:exotel_basic_user`/
`webhook:exotel_basic_password` and `webhook:stringee_signing_secret` — DB-backed
per-tenant secrets minted via `POST /tenants/{id}/webhook-credentials/rotate`,
not environment variables.

---

## Flow 1 — Provider Smoke Tests

The cheap path. One real call per provider, no phone ringing.

### Run

```bash
VOX_LIVE_TESTS=1 .venv/bin/pytest tests/live/test_providers_live.py -m live -v -s
```

`-s` keeps stdout visible so you see latency prints like:

```
[groq STT] latency=820ms text='' language='hi'
[gemini LLM] latency=420ms text='{"ok": true, "lang": "hi"}'
[gemini stream] TTFT=180ms total=950ms chunks=14
[sarvam TTS] latency=1100ms audio_bytes=48000 duration_ms=1500 sr=16000
[twilio creds] latency=210ms account valid; recent calls=0
```

If any env var is missing, that test skips with a clear "missing live-test env vars: [...]" message — no false failures.

### What each test does

| Test                              | Real call                            | Cost     |
| --------------------------------- | ------------------------------------ | -------- |
| `test_groq_stt_smoke`             | Sends a 1s synthetic tone to Whisper | ~$0.001  |
| `test_gemini_llm_smoke`           | One short JSON-mode prompt           | ~$0.0001 |
| `test_gemini_llm_streaming_smoke` | One streaming prompt                 | ~$0.0001 |
| `test_sarvam_tts_smoke`           | Synthesize ~30 chars of Hindi        | ~$0.001  |
| `test_twilio_credential_probe`    | Lists recent calls (no call placed)  | Free     |

---

## Flow 2 — End-to-End Voice Call

The real thing: agent calls your phone, speaks the opening line, listens, responds.

### Step-by-step

#### 1. Update `config/tenants/dev.yaml`

Edit two fields with your real values (everything else is already configured):

```yaml
pipeline:
  telephony:
    from_number: "+1XXXXXXXXXX" # your Twilio number
    webhook_base_url: "https://abc.ngrok.app/api/v1/telephony" # set after ngrok starts

phone_numbers:
  - "+1XXXXXXXXXX" # same Twilio number
```

#### 2. Start ngrok

```bash
ngrok http 8000
```

Copy the `https://...ngrok.app` URL. Update `webhook_base_url` in the YAML to:

```yaml
webhook_base_url: "https://abc-def-123.ngrok.app/api/v1/telephony"
```

#### 3. Start the FastAPI server

```bash
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000
```

You should see startup logs:

```
INFO startup app=vox-agent version=1.0.0
INFO tenant registered slug=dev tokens_count=1
INFO tenant registered slug=example tokens_count=0
```

Sanity check:

```bash
curl http://localhost:8000/health | jq
# {"status":"ok","tenants":[{"slug":"dev",...}], ...}
```

#### 4. Place the call

```bash
.venv/bin/python scripts/place_test_call.py --to +91YOURPHONE
```

Expected output:

```
placing call to +91YOURPHONE via +1XXXXXXXXXX...
  webhook: https://abc.ngrok.app/api/v1/telephony/twilio/voice
call initiated: sid=CA<32hex> status=ringing
```

#### 5. Answer the phone

Your phone rings. When you answer, the flow is:

1. Twilio POSTs to `/twilio/voice` → server returns TwiML with `<Stream url="wss://.../stream?tenant=dev"/>`
2. Twilio opens the Media Streams websocket
3. Bridge factory builds a `VoiceBotAgent` with the dev tenant's Groq/Gemini/Sarvam stack
4. **You hear "Namaste! Main Priya bol rahi hoon..."** — that's TTS-generated audio
5. Speak a reply
6. STT → Gemini → TTS → you hear the agent's response
7. Conversation continues until you hang up or the LLM decides to close

#### 6. Inspect the call

Server logs show each stage:

```
INFO twilio voice webhook tenant=dev to=+91... from=+1...
INFO twilio stream started streamSid=MZ...
INFO bridge factory built call tenant=dev session_id=call_...
```

Twilio's console also shows the call duration + media stream stats.

---

## Cost ballpark

A 1-minute test call:

- Twilio (outbound to India): ~$0.05
- Groq Whisper: free tier covers this
- Gemini Flash: ~$0.001
- Sarvam TTS: ~$0.01

Total: **~$0.06 per call**.

---

## Troubleshooting

| Symptom                                                          | Likely cause                                                  | Fix                                                            |
| ---------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------- |
| `place_test_call.py` says "webhook_base_url is not set to HTTPS" | YAML still has the `CHANGE-ME` placeholder                    | Update `webhook_base_url` with your ngrok URL                  |
| Twilio rings but call drops immediately                          | Webhook returns 5xx                                           | Check server logs; usually a missing env var on the dev tenant |
| Agent rings but stays silent                                     | TTS failed                                                    | Check `TENANT_DEV_SARVAM_KEY`; smoke test the TTS adapter      |
| `MissingEnvError` on call start                                  | Tenant YAML references an env var that isn't set in the shell | Re-export the env var; restart uvicorn                         |
| Call connects but agent doesn't react to your speech             | STT not returning text                                        | Check `TENANT_DEV_GROQ_KEY`; run the STT smoke test            |
| ngrok tunnel keeps reconnecting                                  | Free ngrok tier has session limits                            | Reserve a static subdomain or use a paid tier                  |

---

## Tearing it down

When you're done:

1. Hang up the phone
2. Stop uvicorn (Ctrl-C)
3. Stop ngrok (Ctrl-C)
4. (Optional) revert the placeholder values in `config/tenants/dev.yaml` if you don't want your real numbers in git history

The dev tenant's API tokens (`TENANT_DEV_API_TOKENS`) only grant access to the dev tenant's data, but you may want to rotate them periodically anyway.
