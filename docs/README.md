# Docs Index

## Start here
- **[Handover](HANDOVER.md)** — the current, authoritative status of the whole project
- **[Architecture](ARCHITECTURE.md)** — system-wide view: multi-tenant platform, VoiceBot + ChatBot, the shared knowledge base
- **[Voice architecture](VOICE-ARCHITECTURE.md)** — turn-by-turn zoom-in on the VoiceBot runtime (cascade/S2S internals)
- **[ChatBot module](chatbot.md)** — full ChatBot API/DB reference
- **[CRM API contract](crm-api-contract.md)** — the tool endpoints a CRM partner implements
- [Project status](PROJECT-STATUS.md) — historical snapshot from 2026-06-15; superseded by Handover above

## Decision records & experiments
- **[Latency / LLM / STT experiments](latency-llm-stt-experiments.md)** — decision log with raw readings and rationale: Claude-vs-Gemini LLM A/B (kept Gemini), cloud-deployment latency analysis (modest gain only), Deepgram streaming STT adoption (validation spike + live A/B), and the cross-cutting finding that LLM+TTS inference is the latency bottleneck. **Source material for the project report.**

## Plans & specs
- [Design specs](superpowers/specs/) — e.g. `2026-06-03-deepgram-streaming-stt-design.md`
- [Implementation plans](superpowers/plans/) — e.g. `2026-06-03-deepgram-streaming-stt.md`

## Setup & testing
- [Live testing](live-testing.md) — placing real calls, ngrok setup
- [Multi-tenant plan](multi-tenant-plan.md)
- [Stringee streaming](stringee-streaming.md)
