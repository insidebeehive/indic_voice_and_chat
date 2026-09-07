# vox-agent

Vendor-agnostic agentic framework for multilingual VoiceBot (outbound marketing calls) and ChatBot (RAG-powered text assistant) systems.

See `PRD.md` for the full specification.

## Status

**See [`docs/PROJECT-STATUS.md`](docs/PROJECT-STATUS.md) for the authoritative, up-to-date status.**

Current focus is a **Hindi-only outbound VoiceBot**, tested through the browser dev console.

- **Working & validated:** the voice core (Deepgram streaming STT + Groq batch fallback, Gemini 2.5-flash LLM, Sarvam TTS); the **dev console** with **server-side barge-in**; Twilio/Exotel streaming telephony bridges; campaign orchestration logic; the ChatBot (RAG-grounded, agentic CRM tool-calling — see `docs/chatbot.md`).
- **Partial:** Stringee turn-based IVR (built, live-blocked on Stringee's side); Telnyx/Infobip (auth scaffold only); campaign→live-call wiring.
- **Not started:** telephony barge-in (dev console only so far); a real benchmarking harness (current code is a basic skeleton); code-switching / multilingual (single-language Hindi today).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env

docker compose up -d redis postgres
alembic upgrade head

uvicorn src.main:app --reload --port 8000
curl http://localhost:8000/health
```

## Tests

```bash
pytest tests/unit/ -v
```

All adapter tests are fully mocked — no live API access required.
