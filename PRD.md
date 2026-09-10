# PRD: Vendor-Agnostic Agentic Framework for Multilingual VoiceBot & ChatBot

> **Project Code:** `vox-agent`
> **Version:** 1.0.0
> **Last Updated:** May 2026

---

## 1. Overview

A configurable, vendor-agnostic framework for building multilingual VoiceBot (outbound marketing calls) and ChatBot (RAG-powered text assistant) systems that integrate with any enterprise CRM.

### Core Principles

| Principle | Rule |
|-----------|------|
| Provider Agnosticism | Every external service behind an abstract interface. Swap via config, not code. |
| Agent Autonomy | VoiceBot and ChatBot are autonomous agents with perception-reasoning-action loops. |
| Streaming-First | Voice pipeline streams STT→LLM→TTS with overlapping stages. Not bolted on later. |
| Separation of Concerns | Conversation logic ≠ delivery logic ≠ infrastructure logic. |
| Integration-Ready | REST APIs + webhooks + event bus. Framework augments CRM, doesn't replace it. |

---

## 2. Tech Stack

```
Language:       Python 3.11+
Framework:      FastAPI (async)
Task Queue:     asyncio (native) + optional Celery for campaign batch jobs
Session State:  Redis 7+
Persistence:    PostgreSQL 16+
Vector Store:   FAISS (default), swappable to ChromaDB / Pinecone / Qdrant
Embedding:      sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VAD:            Silero VAD
Audio:          pydub / soundfile for format conversion
Config:         YAML (pydantic-settings for validation)
Testing:        pytest + pytest-asyncio
Containerization: Docker + docker-compose
```

---

## 3. Project Structure

```
vox-agent/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── config/
│   ├── default.yaml              # Default pipeline config
│   └── campaigns/
│       └── sample_campaign.yaml  # Sample campaign config
├── src/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Pydantic settings, YAML loader
│   │
│   ├── interfaces/               # Abstract interface contracts
│   │   ├── __init__.py
│   │   ├── stt.py                # ISTTProvider
│   │   ├── llm.py                # ILLMProvider
│   │   ├── tts.py                # ITTSProvider
│   │   ├── telephony.py          # ITelephonyProvider
│   │   └── vector_store.py       # IVectorStore
│   │
│   ├── providers/                # Concrete provider adapters
│   │   ├── __init__.py
│   │   ├── stt/
│   │   │   ├── __init__.py
│   │   │   ├── groq_whisper.py   # GroqSTTAdapter
│   │   │   └── sarvam.py         # SarvamSTTAdapter
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── gemini.py         # GeminiLLMAdapter
│   │   │   └── groq.py           # GroqLLMAdapter
│   │   ├── tts/
│   │   │   ├── __init__.py
│   │   │   ├── gemini.py         # GeminiTTSAdapter
│   │   │   ├── sarvam.py         # SarvamTTSAdapter
│   │   │   ├── google.py         # GoogleTTSAdapter
│   │   │   ├── deepgram.py       # DeepgramTTSAdapter
│   │   │   ├── styletts.py       # StyleTTSAdapter
│   │   │   └── ai4bharat.py      # AI4BharatTTSAdapter
│   │   ├── telephony/
│   │   │   ├── __init__.py
│   │   │   ├── twilio.py         # TwilioAdapter
│   │   │   ├── exotel.py         # ExotelAdapter
│   │   │   └── stringee.py       # StringeeAdapter
│   │   └── vector_store/
│   │       ├── __init__.py
│   │       ├── faiss_store.py    # FAISSAdapter
│   │       ├── chroma_store.py   # ChromaDBAdapter
│   │       ├── pinecone_store.py # PineconeAdapter
│   │       └── qdrant_store.py   # QdrantAdapter
│   │
│   ├── agents/                   # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py               # BaseAgent (shared logic)
│   │   ├── voicebot.py           # VoiceBotAgent
│   │   ├── chatbot.py            # ChatBotAgent
│   │   └── state_machine.py      # AgentStateMachine
│   │
│   ├── pipeline/                 # Streaming voice pipeline
│   │   ├── __init__.py
│   │   ├── engine.py             # PipelineEngine (STT→LLM→TTS chain)
│   │   ├── vad.py                # Silero VAD wrapper
│   │   ├── interruption.py       # Interruption handler
│   │   ├── audio_utils.py        # Format conversion, normalization
│   │   └── sentence_detector.py  # Language-aware sentence segmentation
│   │
│   ├── dialogue/                 # Dialogue management
│   │   ├── __init__.py
│   │   ├── context.py            # Conversation context manager (Redis)
│   │   ├── slots.py              # Slot schema + slot filling logic
│   │   ├── prompts.py            # System prompt builder
│   │   └── response_parser.py    # Parse structured LLM output
│   │
│   ├── rag/                      # RAG pipeline
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Document parsing + chunking
│   │   ├── embeddings.py         # Embedding generation
│   │   ├── retriever.py          # Dense / hybrid retrieval
│   │   └── context_builder.py    # Assemble RAG context for LLM
│   │
│   ├── campaign/                 # Campaign orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Campaign execution engine
│   │   ├── scheduler.py          # Call scheduling + retry logic
│   │   ├── dnd_filter.py         # DND compliance filter
│   │   └── models.py             # Campaign data models
│   │
│   ├── integration/              # External system integration
│   │   ├── __init__.py
│   │   ├── event_bus.py          # Internal event bus
│   │   ├── webhooks.py           # Webhook manager
│   │   └── crm_client.py         # CRM integration client
│   │
│   ├── api/                      # FastAPI routes
│   │   ├── __init__.py
│   │   ├── sessions.py           # Session management endpoints
│   │   ├── campaigns.py          # Campaign CRUD + execution
│   │   ├── config_routes.py      # Pipeline configuration
│   │   ├── conversations.py      # Conversation history
│   │   ├── knowledge.py          # Knowledge base admin (RAG)
│   │   ├── webhooks_routes.py    # Webhook registration
│   │   ├── benchmarks.py         # Benchmark data endpoints
│   │   ├── chat.py               # ChatBot WebSocket/HTTP
│   │   └── telephony_hooks.py    # Telephony provider webhooks
│   │
│   ├── models/                   # Database models
│   │   ├── __init__.py
│   │   ├── database.py           # SQLAlchemy engine + session
│   │   ├── conversation.py       # Conversation, Turn, Event models
│   │   ├── campaign.py           # Campaign, Lead, CallResult models
│   │   └── benchmark.py          # Benchmark metric models
│   │
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       ├── logging.py            # Structured JSON logging
│       ├── metrics.py            # Latency tracking helpers
│       └── language.py           # Language detection, script detection
│
├── tests/
│   ├── unit/
│   │   ├── test_stt_adapters.py
│   │   ├── test_llm_adapters.py
│   │   ├── test_tts_adapters.py
│   │   ├── test_pipeline.py
│   │   ├── test_rag.py
│   │   ├── test_dialogue.py
│   │   └── test_slots.py
│   └── integration/
│       ├── test_voice_pipeline_e2e.py
│       ├── test_chatbot_e2e.py
│       └── test_campaign_flow.py
│
├── scripts/
│   ├── seed_knowledge_base.py    # Seed sample docs into RAG
│   ├── run_benchmark.py          # Execute benchmark suite
│   └── export_results.py         # Export benchmark results to CSV
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── deployment.md
```

---

## 4. Interface Contracts

### 4.1 ISTTProvider (`src/interfaces/stt.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

@dataclass
class STTResult:
    text: str
    confidence: float            # 0.0 - 1.0
    language: Optional[str]      # Detected language code
    word_timestamps: Optional[list[dict]]  # [{word, start, end}, ...]
    raw_response: dict           # Provider-specific raw response

@dataclass
class STTConfig:
    language: Optional[str] = None    # Language hint (e.g., "hi", "en")
    model: Optional[str] = None       # Provider-specific model name
    sample_rate: int = 16000
    enable_timestamps: bool = False

class ISTTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        """Transcribe a complete audio segment."""
        ...

    @abstractmethod
    async def transcribe_stream(
        self, audio_stream: AsyncIterator[bytes], config: STTConfig
    ) -> AsyncIterator[STTResult]:
        """Stream transcription results as audio arrives."""
        ...

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        ...
```

### 4.2 ILLMProvider (`src/interfaces/llm.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

@dataclass
class LLMMessage:
    role: str        # "system" | "user" | "assistant"
    content: str

@dataclass
class LLMConfig:
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    response_format: Optional[str] = "json"  # "json" | "text"

@dataclass
class LLMResult:
    text: str
    finish_reason: str           # "stop" | "length" | "tool_call"
    usage: dict                  # {prompt_tokens, completion_tokens}
    raw_response: dict

class ILLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig
    ) -> LLMResult:
        """Generate a complete response."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...
```

### 4.3 ITTSProvider (`src/interfaces/tts.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

@dataclass
class TTSConfig:
    language: str = "hi-IN"
    voice_id: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0
    output_format: str = "pcm"   # "pcm" | "wav" | "mp3"
    sample_rate: int = 16000

@dataclass
class TTSResult:
    audio: bytes
    duration_ms: float
    sample_rate: int

class ITTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        """Synthesize complete text to audio."""
        ...

    @abstractmethod
    async def synthesize_stream(
        self, text_stream: AsyncIterator[str], config: TTSConfig
    ) -> AsyncIterator[bytes]:
        """Stream audio as text segments arrive."""
        ...

    @abstractmethod
    def get_available_voices(self, language: str) -> list[dict]:
        """Return available voices for a language."""
        ...
```

### 4.4 ITelephonyProvider (`src/interfaces/telephony.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Callable

@dataclass
class CallConfig:
    to_number: str
    from_number: str
    webhook_url: str             # For call events
    audio_format: str = "pcm"
    sample_rate: int = 8000
    timeout_seconds: int = 30    # Ring timeout

@dataclass
class CallSession:
    session_id: str
    status: str                  # "ringing" | "answered" | "busy" | "no_answer" | "failed"
    to_number: str
    from_number: str

class ITelephonyProvider(ABC):
    @abstractmethod
    async def initiate_call(self, config: CallConfig) -> CallSession:
        """Initiate an outbound call."""
        ...

    @abstractmethod
    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        """Receive audio from the call (caller's speech)."""
        ...

    @abstractmethod
    async def stream_audio_out(
        self, session_id: str, audio_stream: AsyncIterator[bytes]
    ) -> None:
        """Send audio to the call (agent's speech)."""
        ...

    @abstractmethod
    async def hangup(self, session_id: str) -> None:
        """End the call."""
        ...

    @abstractmethod
    async def transfer(self, session_id: str, to_number: str) -> None:
        """Transfer the call to another number (warm transfer)."""
        ...
```

### 4.5 IVectorStore (`src/interfaces/vector_store.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Document:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)  # {source, page, language, section}
    embedding: Optional[list[float]] = None

@dataclass
class SearchResult:
    document: Document
    score: float

@dataclass
class VectorStoreConfig:
    index_path: Optional[str] = None    # For file-based stores (FAISS)
    collection_name: str = "default"
    embedding_dim: int = 384            # MiniLM-L12 dimension

class IVectorStore(ABC):
    @abstractmethod
    async def index(self, documents: list[Document]) -> int:
        """Index documents. Returns count indexed."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> list[SearchResult]:
        """Search for similar documents."""
        ...

    @abstractmethod
    async def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return total document count."""
        ...
```

---

## 5. Configuration Schema

### 5.1 Default Pipeline Config (`config/default.yaml`)

```yaml
app:
  name: vox-agent
  version: 1.0.0
  debug: false
  log_level: INFO

server:
  host: 0.0.0.0
  port: 8000
  workers: 4

redis:
  url: redis://localhost:6379/0
  session_ttl_seconds: 1800        # 30 min

database:
  url: postgresql+asyncpg://vox:vox@localhost:5432/vox_agent

pipeline:
  stt:
    provider: sarvam               # sarvam | groq
    model: saaras:v2
    language: hi-IN
    confidence_threshold: 0.6
    fallback_provider: groq         # Cross-provider fallback

  llm:
    provider: groq                  # groq | gemini
    model: llama-3.1-70b-versatile
    temperature: 0.7
    max_tokens: 512
    response_format: json

  tts:
    provider: sarvam               # sarvam | gemini | google | deepgram | styletts | ai4bharat
    language: hi-IN
    voice_id: null                  # Provider default
    speed: 1.0

  telephony:
    provider: exotel               # twilio | exotel | stringee
    from_number: "+91XXXXXXXXXX"
    webhook_base_url: https://your-domain.com/api/v1/telephony

  vector_store:
    provider: faiss                # faiss | chroma | pinecone | qdrant
    index_path: ./data/faiss_index
    embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    embedding_dim: 384

voice_pipeline:
  vad:
    model: silero
    threshold: 0.5
    min_speech_duration_ms: 250
    min_silence_duration_ms: 600   # Endpointing silence
  silence:
    post_response_timeout_s: 5
    extended_timeout_s: 12
    max_call_duration_s: 420       # 7 minutes
  interruption:
    enabled: true
    detection_interval_ms: 20

rag:
  chunking:
    strategy: recursive            # fixed | semantic | recursive
    chunk_size: 500                # tokens
    chunk_overlap: 100             # tokens
  retrieval:
    strategy: hybrid               # dense | hybrid
    top_k: 5
    bm25_weight: 0.3
    dense_weight: 0.7
    similarity_threshold: 0.0      # see config/default.yaml for why 0.0

compliance:
  calling_hours:
    start: "10:00"                 # IST
    end: "19:00"                   # IST
  dnd_check_enabled: true
  ai_disclosure: true              # Disclose AI identity if asked
  max_retry_attempts: 3
  retry_interval_hours: 2
```

### 5.2 Campaign Config (`config/campaigns/sample_campaign.yaml`)

```yaml
campaign:
  id: camp_2026_plan_b
  name: "Plan B Launch Campaign"
  status: active                   # draft | active | paused | completed
  start_date: "2026-06-01"
  end_date: "2026-06-30"

  call_list_source: crm            # crm | csv
  call_list_csv: null              # Path if source is csv

  concurrency:
    max_concurrent_calls: 10
    calls_per_minute: 20

  pipeline_override:               # Override default pipeline for this campaign
    tts:
      voice_id: "hindi_female_1"
    llm:
      temperature: 0.6

  script:
    agent_name: "Priya"
    agent_role: "Customer Engagement Specialist"
    company_name: "[Your Company]"
    language_default: "hi"

    opening: >
      Namaste {lead_name} ji, main {agent_name} bol rahi hoon
      {company_name} se. Aapka ek minute ho sakta hai?
      Humne aapke liye ek special offer tayyar kiya hai.

    talking_points:
      - "Plan B mein 500GB data milta hai unlimited calls ke saath"
      - "Monthly sirf Rs. 699 aur annual plan mein Rs. 599 per month"
      - "Abhi limited time offer chal raha hai - first 3 months free"

    qualifying_questions:
      - "Aap abhi kaunsa plan use kar rahe hain?"
      - "Kya data usage aapke liye important hai?"
      - "Budget ke hisaab se kaunsa range comfortable hoga?"

    objection_responses:
      busy: "Bilkul, aapka time important hai. Kya main aapko kal {time} pe call kar sakti hoon?"
      not_interested: "Main samajh sakti hoon. Kya main jaana sakti hoon ki aap already koi plan use kar rahe hain?"
      how_got_number: "Aapne humari website pe enquiry ki thi. Agar aap nahi chahte ki hum call karein, main abhi aapka number remove kar deti hoon."
      send_whatsapp: "Zaroor! Main abhi bhej deti hoon. Kya aap mujhe apna WhatsApp number confirm kar sakte hain?"
      is_ai: "Main ek AI assistant hoon {company_name} ki taraf se. Agar aap chahein toh main aapko humari team se connect kar sakti hoon."

    closing:
      positive: "Bahut accha! Main aapke liye {action} kar deti hoon. Dhanyavaad {lead_name} ji!"
      negative: "Koi baat nahi. Agar future mein kuch chahiye toh hum hamesha available hain. Dhanyavaad!"

  slots:
    lead_name:        { type: string,   required: true,  source: crm }
    interest_level:   { type: enum,     required: true,  values: [hot, warm, cold, not_interested] }
    current_provider: { type: string,   required: false }
    budget_range:     { type: string,   required: false }
    callback_time:    { type: datetime, required: false }
    whatsapp_number:  { type: phone,    required: false }
    decision_timeline:{ type: string,   required: false }
    objection_reason: { type: string,   required: false }
    call_disposition: { type: enum,     required: true,
      values: [interested_callback, interested_transfer, not_interested,
               busy_retry, dnd_requested, wrong_number, voicemail] }
```

---

## 6. Database Schema

### 6.1 PostgreSQL Tables

```sql
-- Campaigns
CREATE TABLE campaigns (
    id              VARCHAR(50) PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    status          VARCHAR(20) DEFAULT 'draft',
    config_yaml     TEXT NOT NULL,
    total_leads     INTEGER DEFAULT 0,
    calls_attempted INTEGER DEFAULT 0,
    calls_answered  INTEGER DEFAULT 0,
    leads_qualified INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- Leads
CREATE TABLE leads (
    id              VARCHAR(50) PRIMARY KEY,
    campaign_id     VARCHAR(50) REFERENCES campaigns(id),
    phone_number    VARCHAR(20) NOT NULL,
    name            VARCHAR(255),
    language_pref   VARCHAR(10),
    crm_lead_id     VARCHAR(100),
    metadata        JSONB DEFAULT '{}',
    status          VARCHAR(20) DEFAULT 'pending',  -- pending | called | retry | completed | dnd
    retry_count     INTEGER DEFAULT 0,
    next_retry_at   TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Conversations
CREATE TABLE conversations (
    id              VARCHAR(50) PRIMARY KEY,
    campaign_id     VARCHAR(50) REFERENCES campaigns(id),
    lead_id         VARCHAR(50) REFERENCES leads(id),
    agent_type      VARCHAR(20) NOT NULL,            -- voicebot | chatbot
    channel         VARCHAR(20) NOT NULL,            -- phone | webchat | whatsapp
    status          VARCHAR(20) NOT NULL,            -- active | completed | escalated | dropped
    disposition     VARCHAR(30),
    interest_level  VARCHAR(20),
    slots_data      JSONB DEFAULT '{}',
    pipeline_config JSONB NOT NULL,                  -- {stt, llm, tts, telephony} used
    duration_ms     INTEGER,
    total_turns     INTEGER DEFAULT 0,
    started_at      TIMESTAMP DEFAULT NOW(),
    ended_at        TIMESTAMP
);

-- Turns
CREATE TABLE turns (
    id                  SERIAL PRIMARY KEY,
    conversation_id     VARCHAR(50) REFERENCES conversations(id),
    turn_number         INTEGER NOT NULL,
    role                VARCHAR(10) NOT NULL,         -- user | agent
    content             TEXT NOT NULL,
    language            VARCHAR(10),
    stt_confidence      FLOAT,
    stt_latency_ms      INTEGER,
    llm_ttft_ms         INTEGER,                      -- Time to first token
    llm_total_ms        INTEGER,
    tts_first_chunk_ms  INTEGER,
    tts_total_ms        INTEGER,
    total_latency_ms    INTEGER,                      -- End-to-end
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Events
CREATE TABLE events (
    id              SERIAL PRIMARY KEY,
    conversation_id VARCHAR(50) REFERENCES conversations(id),
    event_type      VARCHAR(50) NOT NULL,             -- call.started, intent.detected, agent.escalated, etc.
    payload         JSONB NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Benchmark runs
CREATE TABLE benchmark_runs (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255),
    description     TEXT,
    pipeline_config JSONB NOT NULL,
    language        VARCHAR(10) NOT NULL,
    dataset         VARCHAR(100) NOT NULL,
    results         JSONB NOT NULL,                   -- {wer, cer, latency_avg, latency_p95, mos, ...}
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Knowledge base documents
CREATE TABLE kb_documents (
    id              VARCHAR(50) PRIMARY KEY,
    filename        VARCHAR(255) NOT NULL,
    source_type     VARCHAR(50),
    language        VARCHAR(10),
    chunk_count     INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{}',
    ingested_at     TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_leads_campaign ON leads(campaign_id, status);
CREATE INDEX idx_conversations_campaign ON conversations(campaign_id);
CREATE INDEX idx_turns_conversation ON turns(conversation_id);
CREATE INDEX idx_events_conversation ON events(conversation_id);
CREATE INDEX idx_events_type ON events(event_type);
```

### 6.2 Redis Key Schema

```
# Session state (TTL: 30 min)
session:{session_id}:state         → JSON: {agent_state, pipeline_config, lead_data}
session:{session_id}:history       → JSON: [{role, content, timestamp, metadata}, ...]
session:{session_id}:slots         → JSON: {slot_name: value, ...}

# Campaign state
campaign:{campaign_id}:active      → SET of active session_ids
campaign:{campaign_id}:stats       → HASH: {attempted, answered, qualified, ...}

# Rate limiting
ratelimit:telephony:{provider}     → Counter with TTL
ratelimit:stt:{provider}           → Counter with TTL
ratelimit:llm:{provider}           → Counter with TTL
```

---

## 7. API Endpoints

### 7.1 Campaign Management

```
POST   /api/v1/campaigns                    Create campaign
GET    /api/v1/campaigns                    List campaigns
GET    /api/v1/campaigns/{id}               Get campaign details + stats
PUT    /api/v1/campaigns/{id}               Update campaign config
POST   /api/v1/campaigns/{id}/start         Start campaign execution
POST   /api/v1/campaigns/{id}/pause         Pause campaign
POST   /api/v1/campaigns/{id}/resume        Resume campaign
POST   /api/v1/campaigns/{id}/leads         Upload lead list (CSV)
GET    /api/v1/campaigns/{id}/leads         List leads with status
GET    /api/v1/campaigns/{id}/stats         Real-time campaign stats
```

### 7.2 Session Management

```
GET    /api/v1/sessions                     List active sessions
GET    /api/v1/sessions/{id}                Get session state
DELETE /api/v1/sessions/{id}                Force terminate session
```

### 7.3 ChatBot

```
WS     /api/v1/chat/ws                      WebSocket for real-time chat
POST   /api/v1/chat/message                 HTTP message (async channels)
GET    /api/v1/chat/history/{session_id}     Get chat history
```

### 7.4 Telephony Webhooks (provider-specific)

```
POST   /api/v1/telephony/twilio/voice       Twilio voice webhook
POST   /api/v1/telephony/twilio/status       Twilio status callback
WS     /api/v1/telephony/twilio/stream       Twilio Media Streams WebSocket
POST   /api/v1/telephony/exotel/voice        Exotel voice webhook
WS     /api/v1/telephony/exotel/stream       Exotel streaming WebSocket
POST   /api/v1/telephony/stringee/voice      Stringee voice webhook
WS     /api/v1/telephony/stringee/stream     Stringee streaming WebSocket
```

### 7.5 Knowledge Base (RAG)

```
POST   /api/v1/knowledge/ingest             Upload + ingest documents
GET    /api/v1/knowledge/documents           List ingested documents
DELETE /api/v1/knowledge/documents/{id}      Remove document from index
POST   /api/v1/knowledge/query              Test retrieval (debug)
GET    /api/v1/knowledge/stats              Index statistics
```

### 7.6 Configuration

```
GET    /api/v1/config/pipeline              Get current pipeline config
PUT    /api/v1/config/pipeline              Update pipeline config
GET    /api/v1/config/providers             List available providers + capabilities
```

### 7.7 Conversations & Analytics

```
GET    /api/v1/conversations                List conversations (filterable)
GET    /api/v1/conversations/{id}           Full conversation with turns
GET    /api/v1/benchmarks/latency           Per-provider latency stats
GET    /api/v1/benchmarks/accuracy          STT accuracy metrics
```

### 7.8 Webhooks

```
POST   /api/v1/webhooks                     Register webhook
GET    /api/v1/webhooks                     List registered webhooks
DELETE /api/v1/webhooks/{id}                Unregister webhook
```

---

## 8. Agent State Machine

```
                    ┌─────────┐
                    │  IDLE   │
                    └────┬────┘
                         │ call connected
                         ▼
              ┌──────────────────────┐
              │     LISTENING        │◄──────────────┐
              └──────────┬───────────┘               │
                         │ utterance complete          │ interruption detected
                         ▼                             │
              ┌──────────────────────┐               │
              │    PROCESSING        │               │
              │ (STT→LLM reasoning)  │               │
              └──────────┬───────────┘               │
                         │ response ready              │
                         ▼                             │
              ┌──────────────────────┐               │
              │    RESPONDING        │───────────────┘
              │ (TTS→Telephony)      │
              └──────────┬───────────┘
                         │
              ┌──────────┴───────────┐
              │                      │
         normal flow            escalation trigger
              │                      │
              ▼                      ▼
     ┌────────────────┐    ┌──────────────────┐
     │ back to         │    │   ESCALATING     │
     │ LISTENING       │    │ (transfer/callback)│
     └────────────────┘    └────────┬─────────┘
                                     │
                                     ▼
                            ┌──────────────┐
                            │    ENDED     │
                            │ (persist data)│
                            └──────────────┘
```

**Transitions:**

| From | Event | To |
|------|-------|----|
| IDLE | Call connected / chat started | LISTENING |
| LISTENING | Utterance complete (VAD endpoint) | PROCESSING |
| PROCESSING | LLM response ready | RESPONDING |
| RESPONDING | Response delivery complete | LISTENING |
| RESPONDING | Interruption detected (barge-in) | LISTENING |
| RESPONDING | Escalation action from LLM | ESCALATING |
| LISTENING | Silence timeout | RESPONDING (prompt) |
| LISTENING | Extended silence | ENDED |
| ESCALATING | Transfer complete / callback scheduled | ENDED |
| Any | Max duration reached | ENDED |
| Any | Hangup detected | ENDED |

---

## 9. Event Bus Events

```python
# Event types emitted by the framework
EVENTS = {
    # Call lifecycle
    "call.initiated":     {"campaign_id", "lead_id", "phone_number"},
    "call.answered":      {"session_id", "campaign_id", "lead_id"},
    "call.completed":     {"session_id", "disposition", "duration_ms"},
    "call.failed":        {"lead_id", "reason"},  # busy, no_answer, invalid

    # Conversation
    "turn.completed":     {"session_id", "role", "content", "latency_ms"},
    "intent.detected":    {"session_id", "intent", "confidence"},
    "slot.filled":        {"session_id", "slot_name", "slot_value"},

    # Lead management
    "lead.qualified":     {"session_id", "lead_id", "interest_level", "slots"},
    "lead.scored":        {"lead_id", "score", "source"},  # voicebot | chatbot

    # Escalation
    "agent.escalated":    {"session_id", "reason", "context_summary"},

    # System
    "provider.error":     {"provider", "error", "session_id"},
    "provider.timeout":   {"provider", "latency_ms", "session_id"},
}
```

---

## 10. Development Phases

### Phase 1: Foundation (Week 3–4)
```
□ Project scaffolding (cookiecutter, pyproject.toml, docker-compose)
□ Config loader (pydantic-settings + YAML)
□ Database models + migrations (alembic)
□ Redis connection + session manager
□ Interface contracts (all 5 interfaces)
□ Provider factory (config → adapter instance)
□ FastAPI app skeleton with health check
□ Structured logging setup
```

### Phase 2: Provider Adapters (Week 4–5)
```
□ STT: GroqSTTAdapter (Whisper)
□ STT: SarvamSTTAdapter
□ LLM: GroqLLMAdapter (streaming)
□ LLM: GeminiLLMAdapter (streaming)
□ TTS: SarvamTTSAdapter
□ TTS: At least 2 more (Google, AI4Bharat or Deepgram)
□ Telephony: TwilioAdapter or ExotelAdapter (start with one)
□ VectorStore: FAISSAdapter
□ Unit tests for each adapter
```

### Phase 3: Voice Pipeline (Week 5–7)
```
□ Silero VAD wrapper
□ Audio format normalization (pcm/wav/mulaw conversion)
□ Sentence boundary detector (Hindi + English)
□ Pipeline engine (async STT→LLM→TTS chain)
□ Interruption handler
□ Silence/timeout handler
□ Dialogue context manager (Redis)
□ System prompt builder
□ Structured LLM response parser
□ Slot filling logic
□ Agent state machine
□ VoiceBotAgent class (ties it all together)
□ Telephony webhook handlers
□ End-to-end voice call test
```

### Phase 4: ChatBot + RAG (Week 8–9)
```
□ Document parser (PDF, DOCX, Markdown)
□ Chunking strategies (recursive, fixed, semantic)
□ Embedding generation pipeline
□ FAISS indexing + search
□ BM25 index for hybrid retrieval
□ Retriever (dense + hybrid)
□ RAG context builder
□ ChatBotAgent class
□ WebSocket chat endpoint
□ HTTP chat endpoint (WhatsApp integration)
□ Knowledge base admin API
□ Hallucination guard (confidence + threshold)
□ End-to-end ChatBot test
```

### Phase 5: Campaign Orchestration (Week 9–10)
```
□ Campaign CRUD API
□ Lead list import (CSV + CRM API)
□ DND filter
□ Call scheduler (time windows, rate limiting)
□ Campaign executor (concurrent call pool)
□ Voicemail detection
□ Retry logic
□ Post-call CRM update
□ Event bus + webhook delivery
□ VoiceBot→ChatBot handoff (WhatsApp follow-up)
□ Campaign stats API
```

### Phase 6: Benchmarking (Week 11–14)
```
□ Benchmark runner script
□ STT benchmark: WER/CER per provider per language
□ TTS benchmark: MOS scoring setup
□ Latency benchmark: per-stage timing per provider combination
□ Code-switching benchmark: WER on mixed speech
□ RAG benchmark: precision, recall, faithfulness (RAGAS)
□ End-to-end task completion benchmark
□ Statistical analysis (ANOVA)
□ Results export to CSV/charts
□ Provider recommendation matrix
```

### Phase 7: Documentation + Polish (Week 15–16)
```
□ API documentation (OpenAPI/Swagger)
□ Architecture diagrams (draw.io / mermaid)
□ Deployment guide
□ Final report chapters 6, 7, 8
□ Presentation slides (20 min)
□ Demo preparation
□ Viva preparation (10 min Q&A)
```

---

## 11. Environment Setup

### 11.1 Quick Start

```bash
# Clone and setup
git clone <repo-url> vox-agent
cd vox-agent
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy env file and fill in API keys
cp .env.example .env

# Start infrastructure
docker-compose up -d redis postgres

# Run migrations
alembic upgrade head

# Seed sample knowledge base (optional)
python scripts/seed_knowledge_base.py

# Start the server
uvicorn src.main:app --reload --port 8000
```

### 11.2 Required Environment Variables (`.env`)

```bash
# STT
GROQ_API_KEY=gsk_xxxxxxxxxxxx
SARVAM_API_KEY=xxxxxxxxxxxx

# LLM
GEMINI_API_KEY=xxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxx          # Shared with STT

# TTS
SARVAM_API_KEY=xxxxxxxxxxxx            # Shared with STT
GOOGLE_TTS_CREDENTIALS_PATH=./creds/google-tts.json
DEEPGRAM_API_KEY=xxxxxxxxxxxx

# Telephony
TWILIO_ACCOUNT_SID=ACxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxx
EXOTEL_API_KEY=xxxxxxxxxxxx
EXOTEL_API_TOKEN=xxxxxxxxxxxx

# Vector Store (only if using managed)
PINECONE_API_KEY=xxxxxxxxxxxx
QDRANT_URL=http://localhost:6333

# Database
DATABASE_URL=postgresql+asyncpg://vox:vox@localhost:5432/vox_agent
REDIS_URL=redis://localhost:6379/0

# Server
WEBHOOK_BASE_URL=https://your-domain.com
SECRET_KEY=your-secret-key-for-jwt
```

### 11.3 Docker Compose

```yaml
version: "3.8"
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: vox_agent
      POSTGRES_USER: vox
      POSTGRES_PASSWORD: vox
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data

  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [redis, postgres]
    volumes:
      - ./config:/app/config
      - ./data:/app/data

volumes:
  pgdata:
```

---

## 12. Key Implementation Notes

### 12.1 Provider Factory Pattern

```python
# src/providers/__init__.py
from src.interfaces.stt import ISTTProvider
from src.providers.stt.groq_whisper import GroqSTTAdapter
from src.providers.stt.sarvam import SarvamSTTAdapter

STT_PROVIDERS: dict[str, type[ISTTProvider]] = {
    "groq": GroqSTTAdapter,
    "sarvam": SarvamSTTAdapter,
}

def get_stt_provider(config: dict) -> ISTTProvider:
    provider_name = config["provider"]
    provider_class = STT_PROVIDERS[provider_name]
    return provider_class(config)

# Same pattern for LLM, TTS, Telephony, VectorStore
```

### 12.2 Structured LLM Output (VoiceBot)

```python
# Expected JSON from LLM for VoiceBot
VOICEBOT_RESPONSE_SCHEMA = {
    "response_text": str,           # What to say
    "language": str,                # "hi" | "en" | "mr" etc.
    "conversation_phase": str,      # opening | pitch | qualification | objection | closing
    "updated_slots": dict,          # {slot_name: value}
    "action": str,                  # continue | clarify | transfer | schedule_callback | send_info | close_positive | close_negative | end
    "action_reason": str,
    "sentiment": str,               # positive | neutral | negative | frustrated
    "internal_notes": str,
}
```

### 12.3 Structured LLM Output (ChatBot)

```python
# Expected JSON from LLM for ChatBot
CHATBOT_RESPONSE_SCHEMA = {
    "response_text": str,
    "language": str,
    "sources_used": list[str],      # ["doc_name:page"]
    "confidence": str,              # high | medium | low
    "action": str,                  # none | schedule_callback | send_info | create_ticket | escalate
    "suggested_followups": list[str],
}
```

### 12.4 Critical Path: First Call Working

The fastest path to a working demo:

```
1. Config loader + .env
2. One STT adapter (Sarvam or Groq)
3. One LLM adapter (Groq — fastest)
4. One TTS adapter (Sarvam or Google)
5. One Telephony adapter (Twilio — best docs)
6. Pipeline engine (basic, no streaming initially)
7. Basic dialogue (hardcoded system prompt, no slots)
8. FastAPI + Twilio webhook handler
→ You now have a working outbound call demo
```

Then iterate: add streaming, slots, interruption, other providers, RAG, campaign orchestration.

---

## 13. Testing Strategy

```
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                    # Unit tests (no API keys needed)
pytest tests/integration/             # Integration tests (need API keys)
pytest tests/unit/test_pipeline.py -v # Specific test file

# Run with coverage
pytest --cov=src --cov-report=html
```

**Unit tests** mock all provider APIs. Test logic, not API connectivity.
**Integration tests** use real APIs. Run manually, not in CI.

---

*End of PRD. Start building from Section 12.4 — get a first call working, then iterate.*
