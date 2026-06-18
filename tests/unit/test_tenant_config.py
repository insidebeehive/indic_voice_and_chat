from __future__ import annotations

from pathlib import Path

import pytest

from src.config_tenant import (
    MissingEnvError,
    TenantConfigError,
    TenantSettings,
    discover_tenant_slugs,
    load_all_tenants,
    load_tenant,
    merge_provider_config,
    validate_credentials,
)


@pytest.fixture
def tenant_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tenants"
    d.mkdir()
    return d


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_tenant_minimal(tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme
""")
    t = load_tenant("acme", tenant_dir)
    assert t.id == "t_acme"
    assert t.slug == "acme"
    assert t.status == "active"
    assert t.default_language == "hi"


def test_load_tenant_full(tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme Telecom
default_language: en
pipeline:
  stt: {provider: sarvam, api_key_env: ACME_SARVAM}
  llm: {provider: groq, api_key_env: ACME_GROQ}
phone_numbers: ["+918888888888", "+917777777777"]
""")
    t = load_tenant("acme", tenant_dir)
    assert t.default_language == "en"
    assert t.pipeline.stt.provider == "sarvam"
    assert t.pipeline.llm.api_key_env == "ACME_GROQ"
    assert t.phone_numbers == ["+918888888888", "+917777777777"]


def test_load_tenant_unknown_slug_raises(tenant_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tenant("missing", tenant_dir)


def test_load_tenant_rejects_non_mapping(tenant_dir: Path) -> None:
    _write(tenant_dir / "bad.yaml", "- just a list\n- not a mapping\n")
    with pytest.raises(ValueError):
        load_tenant("bad", tenant_dir)


def test_load_tenant_validation_error_wraps(tenant_dir: Path) -> None:
    _write(tenant_dir / "bad.yaml", """
id: t_x
slug: ""
name: X
""")
    with pytest.raises(ValueError):
        load_tenant("bad", tenant_dir)


def test_discover_slugs_returns_sorted(tenant_dir: Path) -> None:
    for slug in ("globex", "acme", "stark"):
        _write(tenant_dir / f"{slug}.yaml", f"id: t_{slug}\nslug: {slug}\nname: {slug}\n")
    assert discover_tenant_slugs(tenant_dir) == ["acme", "globex", "stark"]


def test_discover_slugs_empty_dir(tenant_dir: Path) -> None:
    assert discover_tenant_slugs(tenant_dir) == []


def test_discover_slugs_missing_dir(tmp_path: Path) -> None:
    assert discover_tenant_slugs(tmp_path / "does-not-exist") == []


def test_load_all_tenants(tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", "id: t_acme\nslug: acme\nname: Acme\n")
    _write(tenant_dir / "globex.yaml", "id: t_globex\nslug: globex\nname: Globex\n")
    all_t = load_all_tenants(tenant_dir)
    assert set(all_t.keys()) == {"acme", "globex"}
    assert all_t["acme"].id == "t_acme"


def test_secret_resolution_success(monkeypatch, tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme
pipeline:
  stt: {provider: sarvam, api_key_env: ACME_SARVAM}
""")
    monkeypatch.setenv("ACME_SARVAM", "real-key-value")
    t = load_tenant("acme", tenant_dir)
    assert t.secret(t.pipeline.stt.api_key_env) == "real-key-value"


def test_secret_resolution_missing_env_raises(monkeypatch, tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme
pipeline:
  stt: {api_key_env: NEVER_SET}
""")
    monkeypatch.delenv("NEVER_SET", raising=False)
    t = load_tenant("acme", tenant_dir)
    with pytest.raises(MissingEnvError, match="NEVER_SET"):
        t.secret(t.pipeline.stt.api_key_env)


def test_secret_returns_none_when_env_name_is_none() -> None:
    t = TenantSettings(id="t_x", slug="x", name="X")
    assert t.secret(None) is None


def test_platform_webhook_base_url_reads_settings(monkeypatch) -> None:
    """The telephony webhook base is platform-level (WEBHOOK_BASE_URL →
    settings.pipeline.telephony.webhook_base_url), not per-tenant — the inbound
    callback is always our own app, common to every tenant."""
    from types import SimpleNamespace

    import src.config as cfg
    from src.config_tenant import platform_webhook_base_url

    fake = SimpleNamespace(pipeline=SimpleNamespace(telephony=SimpleNamespace(
        webhook_base_url="https://platform.example/api/v1/telephony")))
    monkeypatch.setattr(cfg, "get_settings", lambda: fake)
    assert platform_webhook_base_url() == "https://platform.example/api/v1/telephony"


def test_merge_provider_config_overrides_only_set_fields() -> None:
    from src.config_tenant import TenantSTTConfig

    tenant = TenantSTTConfig(provider="sarvam", api_key_env="X")
    global_layer = {"provider": "default", "language": "hi-IN", "model": "saaras:v2"}
    merged = merge_provider_config(tenant, global_layer, api_key="resolved-key")
    assert merged["provider"] == "sarvam"
    assert merged["language"] == "hi-IN"
    assert merged["model"] == "saaras:v2"
    assert merged["api_key"] == "resolved-key"
    # ``api_key_env`` is metadata, not provider config
    assert "api_key_env" not in merged


def test_tenant_dir_override_via_env(monkeypatch, tenant_dir: Path) -> None:
    _write(tenant_dir / "x.yaml", "id: t_x\nslug: x\nname: X\n")
    monkeypatch.setenv("VOX_TENANT_DIR", str(tenant_dir))
    assert discover_tenant_slugs() == ["x"]


def test_example_yaml_file_loads() -> None:
    """Sanity-check the shipped example.yaml is structurally valid."""
    t = load_tenant("example", tenant_dir=Path("config/tenants"))
    assert t.slug == "example"
    assert t.pipeline.stt.provider == "sarvam"


# --- Credential validation ---------------------------------------------


def test_validate_credentials_passes_on_no_providers_configured() -> None:
    """A tenant that doesn't declare any providers is fine — nothing to validate."""
    t = TenantSettings(id="t1", slug="t1", name="T1")
    validate_credentials(t)


def test_validate_credentials_passes_when_all_keys_declared() -> None:
    from src.config_tenant import (
        TenantLLMConfig, TenantPipelineConfig, TenantSTTConfig,
        TenantTTSConfig, TenantTelephonyConfig,
    )
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            stt=TenantSTTConfig(provider="sarvam", api_key_env="K1"),
            llm=TenantLLMConfig(provider="groq", api_key_env="K2"),
            tts=TenantTTSConfig(provider="sarvam", api_key_env="K1"),
            telephony=TenantTelephonyConfig(
                provider="twilio", account_sid_env="SID", auth_token_env="TOK",
            ),
        ),
    )
    validate_credentials(t)


def test_validate_credentials_raises_when_stt_provider_lacks_key() -> None:
    from src.config_tenant import TenantPipelineConfig, TenantSTTConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(stt=TenantSTTConfig(provider="sarvam")),
    )
    with pytest.raises(TenantConfigError, match="pipeline.stt.api_key_env"):
        validate_credentials(t)


def test_validate_credentials_raises_when_llm_provider_lacks_key() -> None:
    from src.config_tenant import TenantLLMConfig, TenantPipelineConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(llm=TenantLLMConfig(provider="groq")),
    )
    with pytest.raises(TenantConfigError, match="pipeline.llm.api_key_env"):
        validate_credentials(t)


def test_validate_credentials_raises_when_telephony_missing_sid() -> None:
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider="twilio", auth_token_env="TOK"),
        ),
    )
    with pytest.raises(TenantConfigError, match="account_sid_env"):
        validate_credentials(t)


def test_validate_credentials_raises_when_telephony_missing_token() -> None:
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider="twilio", account_sid_env="SID"),
        ),
    )
    with pytest.raises(TenantConfigError, match="auth_token_env"):
        validate_credentials(t)


def test_validate_credentials_collects_all_gaps_in_one_error() -> None:
    """One error message should list every missing field — admins fix in one round-trip."""
    from src.config_tenant import (
        TenantLLMConfig, TenantPipelineConfig, TenantSTTConfig,
        TenantTTSConfig, TenantTelephonyConfig,
    )
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            stt=TenantSTTConfig(provider="sarvam"),       # missing api_key_env
            llm=TenantLLMConfig(provider="groq"),         # missing api_key_env
            tts=TenantTTSConfig(provider="sarvam"),       # missing api_key_env
            telephony=TenantTelephonyConfig(provider="twilio"),  # missing sid + token
        ),
    )
    with pytest.raises(TenantConfigError) as ei:
        validate_credentials(t)
    msg = str(ei.value)
    # All five gaps should appear in the one error.
    assert "pipeline.stt.api_key_env" in msg
    assert "pipeline.llm.api_key_env" in msg
    assert "pipeline.tts.api_key_env" in msg
    assert "account_sid_env" in msg
    assert "auth_token_env" in msg


def test_validate_credentials_includes_source_in_error() -> None:
    """When called via load_tenant, the source path appears in the message
    so admins can find the offending YAML."""
    from src.config_tenant import TenantPipelineConfig, TenantSTTConfig
    t = TenantSettings(
        id="t1", slug="acme", name="T1",
        pipeline=TenantPipelineConfig(stt=TenantSTTConfig(provider="sarvam")),
    )
    with pytest.raises(TenantConfigError, match="config/tenants/acme.yaml"):
        validate_credentials(t, source="config/tenants/acme.yaml")


def test_load_tenant_rejects_provider_without_key(tenant_dir: Path) -> None:
    """End-to-end: load_tenant fails on bootstrap, not on first request."""
    _write(tenant_dir / "broken.yaml", """
id: t_broken
slug: broken
name: Broken
pipeline:
  stt: {provider: sarvam}
""")
    with pytest.raises(TenantConfigError, match="api_key_env"):
        load_tenant("broken", tenant_dir)


def test_load_tenant_accepts_unset_provider(tenant_dir: Path) -> None:
    """If provider is omitted, no validation is needed for that layer."""
    _write(tenant_dir / "chat_only.yaml", """
id: t_chat
slug: chat_only
name: Chat-only tenant
pipeline:
  llm: {provider: groq, api_key_env: GROQ_KEY}
""")
    t = load_tenant("chat_only", tenant_dir)
    assert t.pipeline.llm.provider == "groq"
    assert t.pipeline.stt.provider is None
    assert t.pipeline.telephony.provider is None


def test_validate_credentials_raises_when_mode_s2s_without_realtime() -> None:
    from src.config_tenant import TenantPipelineConfig
    t = TenantSettings(id="t1", slug="t1", name="T1",
                       pipeline=TenantPipelineConfig(mode="s2s"))
    with pytest.raises(TenantConfigError, match="pipeline.realtime"):
        validate_credentials(t)


def test_validate_credentials_raises_when_realtime_provider_lacks_key() -> None:
    from src.config_tenant import TenantPipelineConfig, TenantRealtimeConfig
    t = TenantSettings(id="t1", slug="t1", name="T1",
                       pipeline=TenantPipelineConfig(
                           realtime=TenantRealtimeConfig(provider="gemini_live")))
    with pytest.raises(TenantConfigError, match="pipeline.realtime.api_key_env"):
        validate_credentials(t)


def test_validate_credentials_passes_s2s_with_realtime() -> None:
    from src.config_tenant import TenantPipelineConfig, TenantRealtimeConfig
    t = TenantSettings(id="t1", slug="t1", name="T1",
                       pipeline=TenantPipelineConfig(
                           mode="s2s",
                           realtime=TenantRealtimeConfig(provider="gemini_live", api_key_env="GK")))
    validate_credentials(t)
