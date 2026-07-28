# Docs Index

## Decision records & experiments
- **[Latency / LLM / STT experiments](latency-llm-stt-experiments.md)** — decision log with raw readings and rationale: Claude-vs-Gemini LLM A/B (kept Gemini), cloud-deployment latency analysis (modest gain only), Deepgram streaming STT adoption (validation spike + live A/B), and the cross-cutting finding that LLM+TTS inference is the latency bottleneck. **Source material for the project report.**

## Plans & specs
- [Design specs](superpowers/specs/) — e.g. `2026-06-03-deepgram-streaming-stt-design.md`
- [Implementation plans](superpowers/plans/) — e.g. `2026-06-03-deepgram-streaming-stt.md`

## Setup & testing
- [Live testing](live-testing.md) — placing real calls, ngrok setup
- [Multi-tenant plan](multi-tenant-plan.md)
- [Stringee streaming](stringee-streaming.md)
