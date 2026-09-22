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

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.redact import redact_url

# NB: this module's own load path (load_settings()/get_settings()) runs
# BEFORE src.utils.logging.configure_logging() does -- src/main.py's lifespan
# calls get_settings() (line ~524) one line above its configure_logging()
# call, because the log level configure_logging needs is itself a field on
# the Settings this loads. debug_event() is a no-op until the root logger is
# past its default (unconfigured) level, so every debug_event added in this
# module is dead code on that call path specifically -- it will not appear in
# a real boot's logs no matter what VOX_LOG_LEVEL is set to. It still fires
# under pytest (which sets the root level directly via --log-level, ahead of
# and independently of configure_logging) and on any later get_settings()
# call made after configure_logging has already run once (e.g.
# reset_settings_cache() + get_settings() from a test or a future reload
# path). Documented rather than "fixed": resequencing config/logging startup
# is a real change with its own risk, not something to slip in here.
log = logging.getLogger(__name__)


# libpq/psql-only query params some managed Postgres providers (Neon, …)
# append to connection strings by default. asyncpg.connect() has no such
# parameters and raises TypeError("unexpected keyword argument") on them —
# SQLAlchemy's asyncpg dialect passes every URL query param through as a
# connect() kwarg verbatim.
_LIBPQ_ONLY_PARAMS = frozenset({"channel_binding"})


def strip_libpq_only_query_params(url: str) -> str:
    """Drop ``_LIBPQ_ONLY_PARAMS`` keys from a URL's query string, without the
    scheme rewrite or ``sslmode`` translation ``normalize_db_url`` does.

    For raw-DSN consumers (e.g. ``asyncpg.create_pool(dsn)`` in the pgvector
    store) that don't go through SQLAlchemy — asyncpg's own DSN parser already
    understands ``sslmode`` natively, but any OTHER unrecognized query key
    (``channel_binding`` included) falls through into ``server_settings`` and
    Postgres rejects it as an unrecognized configuration parameter.
    """
    from urllib.parse import urlencode, urlsplit, urlunsplit

    if not any(p in url for p in _LIBPQ_ONLY_PARAMS):
        return url
    parts = urlsplit(url)
    params = [(k, v) for k, v in
              (p.split("=", 1) for p in parts.query.split("&") if p)]
    out = [(k, v) for k, v in params if k not in _LIBPQ_ONLY_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(out)))


def normalize_db_url(url: str) -> str:
    """Make a hosted Postgres URL usable by the async (asyncpg) engine.

    Managed providers (Northflank, Heroku, Neon, …) hand out libpq-style URLs
    like ``postgresql://…?sslmode=require&channel_binding=require``. The async
    engine needs the ``postgresql+asyncpg`` driver, and asyncpg doesn't accept
    libpq's ``sslmode`` query arg (it wants ``ssl``) or ``channel_binding`` at
    all. Normalize/strip both. SQLite and already-qualified URLs pass through
    untouched.
    """
    from urllib.parse import urlencode, urlsplit, urlunsplit

    from src.utils.logging import debug_event

    original = url
    scheme_rewritten = False
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
        scheme_rewritten = True
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    sslmode_translated = None
    channel_binding_stripped = False
    if "+asyncpg" in url.split("://", 1)[0] and ("sslmode=" in url or any(
            p in url for p in _LIBPQ_ONLY_PARAMS)):
        parts = urlsplit(url)
        params = [(k, v) for k, v in
                  (p.split("=", 1) for p in parts.query.split("&") if p)]
        out, ssl_val = [], None
        for k, v in params:
            if k == "sslmode":
                ssl_val = v
            elif k in _LIBPQ_ONLY_PARAMS:
                channel_binding_stripped = True
                continue
            else:
                out.append((k, v))
        if ssl_val and ssl_val != "disable":
            out.append(("ssl", ssl_val))
        sslmode_translated = ssl_val
        url = urlunsplit(parts._replace(query=urlencode(out)))
    # redact_url, never the raw url: this is the DB URL, which routinely
    # embeds a password in userinfo -- see the module-level ordering note
    # above for why this event won't reach a real boot's logs anyway.
    debug_event(
        log, "config db_url_normalize decision",
        original_url=redact_url(original), normalized_url=redact_url(url),
        scheme_rewritten=scheme_rewritten, sslmode_translated=sslmode_translated,
        channel_binding_stripped=channel_binding_stripped,
    )
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
    # SQLAlchemy/asyncpg's library defaults (pool_size=5, max_overflow=10 = 15
    # total) are too small for concurrent chat session bursts — new sessions
    # started under load queue for a free connection and can hang. Ignored on
    # SQLite (tests), which has no connection pool.
    pool_size: int = 20
    max_overflow: int = 30

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


class RetrievalSettings(BaseModel):
    """YAML-sourced retrieval knobs (``rag.retrieval`` in config/default.yaml).

    Bridged into the runtime ``src.rag.retriever.RetrievalConfig`` dataclass
    via ``retrieval_config_from_settings`` — this class is never passed to
    ``HybridRetriever`` directly.

    The rrf/similarity_threshold check below duplicates
    ``src.rag.retriever.validate_retrieval_config`` on purpose: it makes bad
    YAML fail at ``load_settings()``/startup instead of being swallowed by
    ``build_crm_retriever``'s ``except Exception`` (src/bootstrap.py) into a
    silent "no KB".
    """

    strategy: str = "hybrid"
    top_k: int = 5
    bm25_weight: float = 0.3
    dense_weight: float = 0.7
    # Read only under strategy: rrf -- see config/default.yaml's comment for
    # the full rationale (RRF fuses by rank, not score).
    rrf_k: int = Field(default=60, ge=1)
    # Must match config/default.yaml's pinned 0.0 (see the comment there for
    # why): this default is now live wherever a config omits the key, so a
    # nonzero default here would silently reintroduce the empty-retrieval
    # failure mode the YAML pin exists to avoid.
    similarity_threshold: float = 0.0

    @model_validator(mode="after")
    def _rrf_forbids_nonzero_threshold(self) -> "RetrievalSettings":
        if self.strategy == "rrf" and self.similarity_threshold != 0.0:
            raise ValueError(
                "rag.retrieval: strategy='rrf' does not support a nonzero "
                f"similarity_threshold (got {self.similarity_threshold!r}). "
                "RRF scores are reciprocal-rank sums on a much smaller scale "
                "than the 0-1 min-max scale 'hybrid' produces, so a "
                "hybrid-tuned floor would silently empty every result set. "
                "Set rag.retrieval.similarity_threshold to 0.0 for RRF and "
                "bound the result set with top_k -- see "
                "src.rag.retriever.validate_retrieval_config, this "
                "validation's runtime twin."
            )
        return self


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)


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
    ELEVENLABS_API_KEY: Optional[str] = None
    # Declared here for auditability only — adapters resolve these via os.environ.get(...) directly; declaring does not change adapter behavior.
    ANTHROPIC_API_KEY: Optional[str] = None
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    GOOGLE_TTS_API_KEY: Optional[str] = None
    VLLM_BASE_URL: Optional[str] = None
    VLLM_API_KEY: Optional[str] = None
    INDICF5_TTS_URL: Optional[str] = None
    EXOTEL_ACCOUNT_SID: Optional[str] = None
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    EXOTEL_API_KEY: Optional[str] = None
    EXOTEL_API_TOKEN: Optional[str] = None

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
    EVENTS_WEBHOOK_SECRET: Optional[str] = None  # platform-level HMAC signing key for outbound webhooks
    VOX_CONFIG_PATH: str = "config/default.yaml"
    VOX_LOG_LEVEL: Optional[str] = None

    # Grafana Cloud Loki log push (Phase 1 observability). Both unset ->
    # configure_logging() no-ops and behavior is unchanged from today.
    GRAFANA_LOKI_PUSH_URL: Optional[str] = None  # e.g. https://logs-prod-xxx.grafana.net/loki/api/v1/push
    GRAFANA_LOKI_PUSH_AUTH: Optional[str] = None  # "user:api_key" for HTTP basic auth

    # Grafana Cloud Prometheus metrics push (Phase 2 observability, TurnMetric
    # aggregation). Unset -> aggregate_and_push_turn_metrics() no-ops and
    # behavior is unchanged from today. See
    # src/observability/turn_metrics_push.py for the exact push protocol
    # (classic Pushgateway, not remote-write) and why.
    GRAFANA_PROMETHEUS_PUSH_URL: Optional[str] = None  # e.g. https://prometheus-prod-xxx.grafana.net/api/prom/push
    GRAFANA_PROMETHEUS_PUSH_AUTH: Optional[str] = None  # "user:api_key" for HTTP basic auth
    # How often the TurnMetric aggregation job runs. Floored well above 0:
    # each run does a full aggregation query over the rolling window (see
    # src/observability/turn_metrics_push.py), on the same process serving
    # live calls -- 0 or a negative value would turn src/main.py's background
    # loop into a hot loop hammering that query with no sleep in between.
    # 5s is a practical floor -- there's no legitimate reason to aggregate
    # dashboard metrics more often than that, and it still fails fast at
    # config-load time on an obvious misconfiguration (e.g. "0").
    METRICS_PUSH_INTERVAL_S: float = Field(default=60.0, ge=5.0)

    # Retention window for chat_turn_metrics/chat_tool_metrics (turn-metrics
    # plan, Phase 3, §11.2). Chat runs ~90x voice's TurnMetric volume (~4.7k
    # parent rows/month + ~12-15k child rows/month, vs. voice's 747 LIFETIME
    # rows) -- unlike TurnMetric, which is deliberately left unbounded because
    # it will never grow enough to matter, unbounded growth here is not
    # acceptable, hence src/main.py's periodic prune loop. Floored at 1 day so
    # a misconfiguration (e.g. "0") can't turn that loop into something that
    # deletes same-day rows on every run.
    CHAT_METRICS_RETENTION_DAYS: float = Field(default=90.0, ge=1.0)


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


_VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


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
    if secrets.VOX_LOG_LEVEL:
        normalized = secrets.VOX_LOG_LEVEL.strip().upper()
        if normalized in _VALID_LOG_LEVELS:
            yaml_data.setdefault("app", {})["log_level"] = normalized
        else:
            # print, not log.warning: this runs during config loading, which
            # happens at import time before configure_logging() is ever called
            # (see src/main.py's lifespan) -- a logger call here would hit an
            # unconfigured root logger. Must not raise: configure_logging() is
            # invoked from FastAPI's lifespan, so an uncaught exception during
            # config loading would crash-loop the whole service on every
            # restart if a bad value were ever set.
            print(
                f"WARNING: invalid VOX_LOG_LEVEL={secrets.VOX_LOG_LEVEL!r}, "
                f"expected one of {sorted(_VALID_LOG_LEVELS)} -- ignoring, "
                f"using configured default"
            )
            # No debug_event for the invalid-value branch above: it is the
            # exact value that decides whether DEBUG logging is even on, so a
            # debug_event here could only ever fire on some LATER settings
            # load, never the one that needed it. See the module-level
            # ordering note near the top of this file.
    from src.utils.logging import debug_event

    debug_event(
        log, "config env_override applied",
        database_url_overridden=bool(secrets.DATABASE_URL),
        db_schema_overridden=secrets.VOX_DB_SCHEMA,
        redis_url_overridden=bool(secrets.REDIS_URL),
        webhook_base_url_overridden=secrets.WEBHOOK_BASE_URL,
        media_storage_access_key_overridden=bool(secrets.MEDIA_STORAGE_ACCESS_KEY),
        media_storage_secret_key_overridden=bool(secrets.MEDIA_STORAGE_SECRET_KEY),
        media_storage_bucket_overridden=secrets.MEDIA_STORAGE_BUCKET,
        media_storage_endpoint_url_overridden=secrets.MEDIA_STORAGE_ENDPOINT_URL,
        media_storage_region_overridden=secrets.MEDIA_STORAGE_REGION,
        effective_log_level_override=yaml_data.get("app", {}).get("log_level"),
    )
    return yaml_data


def _ignored_yaml_keys(block: dict[str, Any], model_cls: type[BaseModel]) -> list[str]:
    """Keys in ``block`` that ``model_cls`` has no field for.

    Every sub-config here is a plain ``BaseModel`` with pydantic's default
    ``extra="ignore"`` -- an unrecognized key (a typo, or a key that used to
    be read and no longer is) is dropped with no error at any level. This is
    exactly how ``config/default.yaml``'s ``pipeline.tts.model: bulbul:v3``
    goes nowhere: ``TTSConfig`` has no ``model`` field. See ``load_settings``.
    """
    return sorted(set(block) - set(model_cls.model_fields))


# Load-time diagnostics, kept so they can be logged once logging exists.
#
# `load_settings` runs BEFORE `configure_logging` on a real boot and cannot not
# do: main.py's lifespan resolves settings precisely to learn the log level, and
# then configures logging with it. So every debug_event below evaluates against
# an unconfigured root logger and goes nowhere on a live server -- they fire
# only under pytest, which sets the root level itself, or on a later reload.
#
# That would have made the unknown-key sweep useless exactly where it matters:
# its whole job is to surface a dead config key like config/default.yaml's inert
# `tts.model`, and it would have been silent on every production boot. So the
# same payloads are stashed here and main.py emits them as one event
# immediately after configure_logging. Kept as plain dicts rather than replayed
# through debug_event with a computed name, which would defeat
# tests/unit/test_debug_event_call_sites.py's literal-name rule.
_PENDING_LOAD_DIAGNOSTICS: list[dict] = []


def drain_load_diagnostics() -> list[dict]:
    """Take the stashed load-time diagnostics, leaving none behind.

    Drains rather than copies so a later reload cannot re-report a previous
    load's findings as if they were current.
    """
    global _PENDING_LOAD_DIAGNOSTICS
    pending, _PENDING_LOAD_DIAGNOSTICS = _PENDING_LOAD_DIAGNOSTICS, []
    return pending


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load YAML defaults + env secrets into a validated Settings object."""
    from src.utils.logging import debug_event

    _PENDING_LOAD_DIAGNOSTICS.clear()

    secrets = Secrets()
    explicit_arg = config_path is not None
    env_path = os.environ.get("VOX_CONFIG_PATH")
    path = Path(config_path or env_path or secrets.VOX_CONFIG_PATH)
    debug_event(
        log, "config yaml_load decision",
        config_filename=str(path),
        source=("explicit_arg" if explicit_arg else "env_VOX_CONFIG_PATH" if env_path else "default"),
    )
    yaml_data = _load_yaml(path)
    debug_event(
        log, "config yaml_load result",
        config_filename=str(path), top_level_keys=sorted(yaml_data),
    )
    yaml_data = _apply_env_overrides(yaml_data, secrets)

    # Unknown-key sweep: top level plus the pipeline sub-blocks, which is
    # where config/default.yaml's own dead `tts.model` key lives. Cheap
    # (a handful of dict/set diffs) and runs once per process boot via
    # get_settings()'s @lru_cache, not per request -- no isEnabledFor guard
    # needed (see docs/debug-logging.md's cost-when-off section).
    top_level_ignored = sorted(set(yaml_data) - set(Settings.model_fields) - {"secrets"})
    pipeline_block = yaml_data.get("pipeline") or {}
    pipeline_nested_ignored = {
        name: unknown
        for name, cls in (
            ("stt", STTConfig), ("llm", LLMConfig), ("tts", TTSConfig),
            ("telephony", TelephonyConfig), ("vector_store", VectorStoreConfig),
        )
        if (unknown := _ignored_yaml_keys(pipeline_block.get(name) or {}, cls))
    }
    if top_level_ignored or pipeline_nested_ignored:
        unknown_keys = {
            "event": "config unknown_keys detected",
            "config_filename": str(path),
            "top_level_ignored": top_level_ignored,
            "pipeline_nested_ignored": pipeline_nested_ignored,
        }
        _PENDING_LOAD_DIAGNOSTICS.append(unknown_keys)
        debug_event(log, "config unknown_keys detected", **{
            k: v for k, v in unknown_keys.items() if k != "event"
        })

    settings = Settings(**yaml_data, secrets=secrets)
    _PENDING_LOAD_DIAGNOSTICS.append({
        "event": "config settings_load resolved",
        "config_filename": str(path),
        "database_url": redact_url(settings.database.url),
        "redis_url": redact_url(settings.redis.url),
        "effective_log_level": settings.app.log_level,
        "media_storage_configured": settings.media_storage is not None,
        "loki_push_configured": bool(secrets.GRAFANA_LOKI_PUSH_URL),
        "prometheus_push_configured": bool(secrets.GRAFANA_PROMETHEUS_PUSH_URL),
    })
    debug_event(
        log, "config settings_load resolved",
        config_filename=str(path),
        database_url=redact_url(settings.database.url),
        redis_url=redact_url(settings.redis.url),
        effective_log_level=settings.app.log_level,
        has_sarvam_key=bool(secrets.SARVAM_API_KEY),
        has_groq_key=bool(secrets.GROQ_API_KEY),
        has_gemini_key=bool(secrets.GEMINI_API_KEY),
        has_deepgram_key=bool(secrets.DEEPGRAM_API_KEY),
        has_elevenlabs_key=bool(secrets.ELEVENLABS_API_KEY),
        has_anthropic_key=bool(secrets.ANTHROPIC_API_KEY),
        has_azure_speech_key=bool(secrets.AZURE_SPEECH_KEY),
        has_google_tts_key=bool(secrets.GOOGLE_TTS_API_KEY),
        has_vllm_api_key=bool(secrets.VLLM_API_KEY),
        has_indicf5_tts_url=bool(secrets.INDICF5_TTS_URL),
        has_exotel_creds=bool(
            secrets.EXOTEL_ACCOUNT_SID and secrets.EXOTEL_API_KEY and secrets.EXOTEL_API_TOKEN
        ),
        has_twilio_creds=bool(secrets.TWILIO_ACCOUNT_SID and secrets.TWILIO_AUTH_TOKEN),
        media_storage_configured=settings.media_storage is not None,
        loki_push_configured=bool(secrets.GRAFANA_LOKI_PUSH_URL),
        prometheus_push_configured=bool(secrets.GRAFANA_PROMETHEUS_PUSH_URL),
    )
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton accessor — use this in FastAPI dependencies."""
    return load_settings()


def reset_settings_cache() -> None:
    """Test helper: clear the cached settings."""
    get_settings.cache_clear()
