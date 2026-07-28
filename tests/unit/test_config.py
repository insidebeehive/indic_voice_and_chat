from __future__ import annotations

from pathlib import Path

from src.config import load_settings, reset_settings_cache


def test_loads_default_yaml() -> None:
    s = load_settings()
    assert s.app.name == "vox-agent"
    assert s.app.version == "1.0.0"
    assert s.pipeline.stt.provider == "sarvam"
    assert s.pipeline.llm.provider == "gemini"
    assert s.pipeline.tts.provider == "sarvam"
    assert s.pipeline.telephony.provider == "twilio"
    assert s.pipeline.vector_store.provider == "pgvector"
    assert s.pipeline.vector_store.embedding_dim == 384


def test_voice_pipeline_defaults_round_trip() -> None:
    s = load_settings()
    assert s.voice_pipeline.vad.model == "silero"
    assert s.voice_pipeline.silence.max_call_duration_s == 420
    assert s.voice_pipeline.interruption.enabled is True


def test_env_override_database_url(monkeypatch) -> None:
    reset_settings_cache()
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+asyncpg://x:y@example:5432/somedb"
    )
    s = load_settings()
    assert s.database.url == "postgresql+asyncpg://x:y@example:5432/somedb"


def test_media_storage_config_from_env(monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_ACCESS_KEY", "ak")
    monkeypatch.setenv("MEDIA_STORAGE_SECRET_KEY", "sk")
    monkeypatch.setenv("MEDIA_STORAGE_BUCKET", "mybucket")
    monkeypatch.setenv("MEDIA_STORAGE_ENDPOINT_URL", "https://r2.example.com")
    from src.config import reset_settings_cache, load_settings
    reset_settings_cache()
    s = load_settings("config/default.yaml")
    assert s.media_storage is not None
    assert s.media_storage.access_key == "ak"
    assert s.media_storage.bucket == "mybucket"
    reset_settings_cache()


def test_explicit_path_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
app: {name: custom-app, version: "9.9.9"}
server: {host: 127.0.0.1, port: 1234, workers: 1}
redis: {url: "redis://r:6379/0", session_ttl_seconds: 60}
database: {url: "postgresql+asyncpg://u:p@h:5432/db"}
pipeline:
  stt: {provider: sarvam}
  llm: {provider: groq}
  tts: {provider: sarvam}
  telephony: {provider: twilio, from_number: "+91", webhook_base_url: "https://x"}
  vector_store: {provider: faiss}
voice_pipeline: {}
rag: {}
compliance: {}
""",
        encoding="utf-8",
    )
    s = load_settings(str(custom))
    assert s.app.name == "custom-app"
    assert s.server.port == 1234
