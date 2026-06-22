"""Application configuration.

Two-layer model:
1. ``config/default.yaml`` provides non-secret defaults (provider names,
   thresholds, timeouts, model IDs).
2. ``.env`` / environment variables provide secrets (API keys, DB URL) and
   per-environment overrides.

``load_settings()`` is the single entry point. Call it once at startup; cache
the result with ``@lru_cache`` to make it cheap to inject anywhere.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_db_url(url: str) -> str:
    """Make a hosted Postgres URL usable by the async (asyncpg) engine.

    Managed providers (Northflank, Heroku, …) hand out libpq-style URLs like
    ``postgresql://…?sslmode=require``. The async engine needs the
    ``postgresql+asyncpg`` driver, and asyncpg doesn't accept libpq's
    ``sslmode`` query arg — it wants ``ssl``. Normalize both. SQLite and
    already-qualified URLs pass through untouched.
    """
    from urllib.parse import urlencode, urlsplit, urlunsplit

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    if "+asyncpg" in url.split("://", 1)[0] and "sslmode=" in url:
        parts = urlsplit(url)
        params = [(k, v) for k, v in
                  (p.split("=", 1) for p in parts.query.split("&") if p)]
        out, ssl_val = [], None
        for k, v in params:
            if k == "sslmode":
                ssl_val = v
            else:
                out.append((k, v))
        if ssl_val and ssl_val != "disable":
            out.append(("ssl", ssl_val))
        url = urlunsplit(parts._replace(query=urlencode(out)))
    return url


# --- Sub-configs (mirror PRD §5.1) ----------------------------------------


class AppConfig(BaseModel):
    name: str = "vox-agent"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 1800


class DatabaseConfig(BaseModel):
    url: str = "postgresql+asyncpg://vox:vox@localhost:5432/vox_agent"
    # All our tables live under this schema inside whatever database the URL
    # points at (so we never need a dedicated database). Ignored on SQLite
    # (tests), which has no schemas. Override with VOX_DB_SCHEMA.
    # NB: named ``db_schema`` (not ``schema``) to avoid shadowing
    # ``pydantic.BaseModel.schema``.
    db_schema: str = "voicebot"

    @field_validator("url")
    @classmethod
    def _normalize_url(cls, v: str) -> str:
        return normalize_db_url(v)


class STTConfig(BaseModel):
    provider: str
    model: Optional[str] = None
    language: str = "hi-IN"
    confidence_threshold: float = 0.6
    fallback_provider: Optional[str] = None


class LLMConfig(BaseModel):
    provider: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    response_format: str = "json"


class TTSConfig(BaseModel):
    provider: str
    language: str = "hi-IN"
    voice_id: Optional[str] = None
    speed: float = 1.0


class TelephonyConfig(BaseModel):
    provider: str
    from_number: str
    webhook_base_url: str


class VectorStoreConfig(BaseModel):
    provider: str
    index_path: Optional[str] = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384


class PipelineConfig(BaseModel):
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    telephony: TelephonyConfig
    vector_store: VectorStoreConfig


class VADConfig(BaseModel):
    model: str = "silero"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 600


class SilenceConfig(BaseModel):
    post_response_timeout_s: int = 5
    extended_timeout_s: int = 12
    max_call_duration_s: int = 420


class InterruptionConfig(BaseModel):
    enabled: bool = True
    detection_interval_ms: int = 20


class VoicePipelineConfig(BaseModel):
    vad: VADConfig = Field(default_factory=VADConfig)
    silence: SilenceConfig = Field(default_factory=SilenceConfig)
    interruption: InterruptionConfig = Field(default_factory=InterruptionConfig)


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 100


class RetrievalConfig(BaseModel):
    strategy: str = "hybrid"
    top_k: int = 5
    reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    similarity_threshold: float = 0.4


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class CallingHours(BaseModel):
    start: str = "10:00"
    end: str = "19:00"


class ComplianceConfig(BaseModel):
    calling_hours: CallingHours = Field(default_factory=CallingHours)
    dnd_check_enabled: bool = True
    ai_disclosure: bool = True
    max_retry_attempts: int = 3
    retry_interval_hours: int = 2


class MediaStorageConfig(BaseModel):
    endpoint_url: Optional[str] = None  # omit for AWS S3; set for R2/B2/MinIO
    access_key: str = ""
    secret_key: str = ""
    bucket: str = "chat-media"
    region: str = "auto"
    signed_url_ttl_seconds: int = 3600


# --- Top-level settings ---------------------------------------------------


class Secrets(BaseSettings):
    """Secrets and per-env overrides sourced from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Provider keys
    SARVAM_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    DEEPGRAM_API_KEY: Optional[str] = None
    GOOGLE_TTS_CREDENTIALS_PATH: Optional[str] = None
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    EXOTEL_API_KEY: Optional[str] = None
    EXOTEL_API_TOKEN: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    QDRANT_URL: Optional[str] = None

    # Infra overrides
    DATABASE_URL: Optional[str] = None
    VOX_DB_SCHEMA: Optional[str] = None
    REDIS_URL: Optional[str] = None

    # Media storage (S3-compatible)
    MEDIA_STORAGE_ENDPOINT_URL: Optional[str] = None
    MEDIA_STORAGE_ACCESS_KEY: Optional[str] = None
    MEDIA_STORAGE_SECRET_KEY: Optional[str] = None
    MEDIA_STORAGE_BUCKET: Optional[str] = None
    MEDIA_STORAGE_REGION: Optional[str] = None

    # Misc
    WEBHOOK_BASE_URL: Optional[str] = None
    SECRET_KEY: str = "change-me-in-prod"
    VOX_CONFIG_PATH: str = "config/default.yaml"


class Settings(BaseModel):
    """Merged settings: YAML defaults overlaid with env-derived secrets."""

    app: AppConfig
    server: ServerConfig
    redis: RedisConfig
    database: DatabaseConfig
    pipeline: PipelineConfig
    voice_pipeline: VoicePipelineConfig
    rag: RAGConfig
    compliance: ComplianceConfig
    media_storage: Optional[MediaStorageConfig] = None

    secrets: Secrets


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping")
    return data


def _apply_env_overrides(yaml_data: dict[str, Any], secrets: Secrets) -> dict[str, Any]:
    """Apply env-derived overrides to the YAML config dict in place."""
    if secrets.DATABASE_URL:
        yaml_data.setdefault("database", {})["url"] = secrets.DATABASE_URL
    if secrets.VOX_DB_SCHEMA:
        yaml_data.setdefault("database", {})["db_schema"] = secrets.VOX_DB_SCHEMA
    if secrets.REDIS_URL:
        yaml_data.setdefault("redis", {})["url"] = secrets.REDIS_URL
    if secrets.WEBHOOK_BASE_URL:
        yaml_data.setdefault("pipeline", {}).setdefault("telephony", {})[
            "webhook_base_url"
        ] = secrets.WEBHOOK_BASE_URL
    if secrets.MEDIA_STORAGE_ACCESS_KEY:
        yaml_data.setdefault("media_storage", {})["access_key"] = secrets.MEDIA_STORAGE_ACCESS_KEY
    if secrets.MEDIA_STORAGE_SECRET_KEY:
        yaml_data.setdefault("media_storage", {})["secret_key"] = secrets.MEDIA_STORAGE_SECRET_KEY
    if secrets.MEDIA_STORAGE_BUCKET:
        yaml_data.setdefault("media_storage", {})["bucket"] = secrets.MEDIA_STORAGE_BUCKET
    if secrets.MEDIA_STORAGE_ENDPOINT_URL:
        yaml_data.setdefault("media_storage", {})["endpoint_url"] = secrets.MEDIA_STORAGE_ENDPOINT_URL
    if secrets.MEDIA_STORAGE_REGION:
        yaml_data.setdefault("media_storage", {})["region"] = secrets.MEDIA_STORAGE_REGION
    return yaml_data


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load YAML defaults + env secrets into a validated Settings object."""
    secrets = Secrets()
    path = Path(config_path or os.environ.get("VOX_CONFIG_PATH") or secrets.VOX_CONFIG_PATH)
    yaml_data = _load_yaml(path)
    yaml_data = _apply_env_overrides(yaml_data, secrets)
    return Settings(**yaml_data, secrets=secrets)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton accessor — use this in FastAPI dependencies."""
    return load_settings()


def reset_settings_cache() -> None:
    """Test helper: clear the cached settings."""
    get_settings.cache_clear()
