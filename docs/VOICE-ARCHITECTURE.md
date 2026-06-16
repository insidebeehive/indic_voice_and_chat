# Voice architecture (the conversation runtime)

How one **voice turn** flows: caller audio comes in over a telephony/browser
transport, a **bridge** adapts it, it runs through one of **two pipelines** —
① the STT→LLM→TTS **cascade** or ② **speech-to-speech** (Gemini Live) — and the
**dialogue manager** (agent + state machine + slots + prompts) governs the turn and
records the outcome. All providers sit behind small interfaces.

```mermaid
flowchart TB
  classDef io fill:#11213a,stroke:#3b5374,color:#dbe7ff;
  classDef brain fill:#1c1407,stroke:#7c5e1e,color:#fde68a;
  classDef impl fill:#0b1220,stroke:#334155,color:#93c5fd;

  CALLER(["Caller — PSTN phone / browser mic"]):::io

  subgraph IO["Audio transport · ITelephonyProvider places/hangs up the call"]
    direction LR
    TW["Twilio / Exotel<br/>Media Streams · μ-law/PCM 8k · WS"]:::io
    SIP["SIP trunk · DiDLogic<br/>RTP · PCMU 8k"]:::io
    STR["Stringee<br/>turn IVR · record→WAV"]:::io
    BWS["Browser console<br/>PCM16 16k · WS"]:::io
  end

  subgraph BRIDGE["Bridge layer — _BaseLiveBridge · audio_utils (resample, μ-law↔PCM, VAD framing)"]
    direction LR
    BL["live / S2S bridges<br/>TelephonyLiveBridge · SipMediaBridge · GeminiLiveBridge"]
    BC["cascade bridges<br/>Twilio/ExotelMediaBridge · BrowserVoiceBridge · StringeeIvrBridge"]
  end

  subgraph CASCADE["① Cascade pipeline — PipelineEngine"]
    direction LR
    VAD["VAD / endpointing<br/>SileroVAD · EnergyVAD"]:::impl
    STT["STT · ISTTProvider<br/>Deepgram (stream) · Sarvam · Groq"]:::impl
    LLMC["LLM · ILLMProvider<br/>Gemini · Groq · Anthropic<br/>JSON envelope: text + slots + action"]:::impl
    TTS["TTS · ITTSProvider<br/>Sarvam bulbul"]:::impl
    VAD --> STT --> LLMC --> TTS
  end

  subgraph S2S["② Speech-to-speech"]
    LIVE["GeminiLiveSession · IRealtimeSession<br/>audio ↔ audio · ~1.4s first word · native barge-in<br/>record_turn_signal tool → action + slots"]:::impl
  end

  subgraph DM["Dialogue management (the brain — shared by both modes)"]
    direction LR
    AGENT["VoiceBotAgent<br/>apply_signal()"]:::brain
    SM["AgentStateMachine<br/>State · Event"]:::brain
    SLOTS["SlotSchema<br/>slot-filling"]:::brain
    PROMPT["VoiceBotScript + prompts<br/>cascade & S2S system instructions"]:::brain
    OUT["Outcome analysis<br/>analyze_call"]:::brain
  end

  CALLER <--> IO
  IO <--> BRIDGE
  BC -->|"caller audio"| CASCADE
  BL <-->|"audio ↔"| S2S
  CASCADE -->|"user text + action/slots"| AGENT
  S2S -->|"transcript + tool action/slots"| AGENT
  AGENT --> SM
  AGENT --> SLOTS
  AGENT --> PROMPT
  AGENT --> OUT
  TTS -->|"reply audio"| BRIDGE
  S2S -->|"reply audio"| BRIDGE
  PROMPT -.->|"system instruction"| LLMC
  PROMPT -.->|"system instruction"| LIVE
```

## The turn loop
1. **In** — caller audio arrives over the transport (Twilio/Exotel Media Streams μ-law/PCM @ 8 kHz,
   SIP/DiDLogic RTP @ 8 kHz, Stringee recorded WAV, or the browser at 16 kHz). The **bridge**
   (`_BaseLiveBridge` family) normalizes/resamples it (`audio_utils`).
2. **Understand** —
   - **Cascade:** VAD/endpointing detects end-of-utterance → **STT** → **LLM** (returns a JSON
     envelope: `response_text` + `updated_slots` + `action`) → **TTS** synthesizes the reply.
   - **S2S:** audio streams straight into **GeminiLiveSession**, which speaks the reply directly and
     emits a `record_turn_signal` tool call carrying `action` + `updated_slots`.
3. **Decide** — either path feeds **`VoiceBotAgent.apply_signal()`**, which advances the
   **state machine**, merges **slots**, and uses the **script/prompts** to steer the next turn.
4. **Out** — reply audio goes back through the bridge to the caller.
5. **End** — on hangup/terminal state, **`analyze_call`** classifies the outcome (interested,
   callback, no_answer, …) from the transcript + slots.

## Notes
- **Two modes, one brain.** Cascade and S2S share the same dialogue manager (state machine, slots,
  prompts, outcome). They differ only in *how* speech↔intent is produced: serialized STT→LLM→TTS
  vs. end-to-end audio. `pipeline.mode` selects per tenant; cascade is the controllable default,
  S2S the low-latency path (~1.4 s vs ~3.2 s first word).
- **Interfaces decouple providers.** `ISTTProvider`/`IStreamingSTTProvider`, `ILLMProvider`,
  `ITTSProvider`, `IRealtimeSession`, `ITelephonyProvider` — each provider is a swappable adapter
  behind its interface (Sarvam, Groq, Deepgram, Gemini, Anthropic, Gemini Live; Twilio/Exotel/
  Stringee/SIP).
- **Audio formats.** Telephony is 8 kHz mono (μ-law on Twilio, PCM on Exotel, PCMU/RTP on SIP);
  the browser is 16 kHz; bridges resample to the model's rate and back.
- **Barge-in.** S2S has it natively; the browser cascade has server-side barge-in; telephony
  cascade barge-in is a pending fast-follow.
