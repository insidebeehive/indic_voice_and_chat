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


def _all_logged_text(caplog) -> str:
    """Every field of every captured record, flattened to one grep-able
    string -- catches a leak in a `debug_event`/`extra=` field, not just the
    message. Mirrors tests/unit/test_provider_debug_logging.py's helper of
    the same name."""
    chunks = []
    for r in caplog.records:
        chunks.append(r.getMessage())
        for k, v in vars(r).items():
            chunks.append(f"{k}={v!r}")
    return "\n".join(chunks)


def test_secret_resolution_debug_log_never_carries_the_value(monkeypatch, caplog) -> None:
    """TenantSettings.secret() resolves telephony/webhook/LiveKit credentials
    (docs/debug-logging.md: "This package is where credentials LIVE") and its
    DEBUG boundary event ("tenant_config secret resolved") must carry a
    fingerprint + length, never the value itself -- the one absolute
    exception to "DEBUG logs full values".

    This is the regression test for the credential-leak demonstration run
    while writing this pass: temporarily adding `value=value` to that
    debug_event call (src/config_tenant.py's TenantSettings.secret) makes
    this test fail immediately, because the raw secret then appears in the
    captured record text below. No test in this file (or test_tenant_auth.py
    / test_tenant_registry.py / test_tenants_routes.py) exercised
    `secret()` at DEBUG before this one -- test_secret_resolution_success
    above asserts the RETURN value, never what gets logged.
    """
    caplog.set_level("DEBUG")
    monkeypatch.setenv("ACME_CANARY_SECRET", "s3cr3t-leak-canary-abcdef123456")
    t = TenantSettings(id="t_acme", slug="acme", name="Acme")
    resolved = t.secret("ACME_CANARY_SECRET")
    assert resolved == "s3cr3t-leak-canary-abcdef123456"
    logged = _all_logged_text(caplog)
    assert "s3cr3t-leak-canary-abcdef123456" not in logged
    # The event still needs to be USEFUL, not just safe -- fingerprint,
    # length, and the env var NAME (a reference, not a secret) should be
    # present so an operator can confirm the right value resolved.
    assert "ACME_CANARY_SECRET" in logged
    assert "value_len" in logged and "31" in logged


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


def test_validate_credentials_passes_when_stt_provider_has_no_key() -> None:
    """STT always resolves its key from the platform master env var, so a
    tenant-level api_key_env is optional — omitting it is not an error."""
    from src.config_tenant import TenantPipelineConfig, TenantSTTConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(stt=TenantSTTConfig(provider="sarvam")),
    )
    validate_credentials(t)


def test_validate_credentials_passes_when_llm_provider_has_no_key() -> None:
    """LLM always resolves its key from the platform master env var, so a
    tenant-level api_key_env is optional — omitting it is not an error."""
    from src.config_tenant import TenantLLMConfig, TenantPipelineConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(llm=TenantLLMConfig(provider="groq")),
    )
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
    """One error message should list every missing field — admins fix in one
    round-trip. STT/LLM/TTS api_key_env is never required (platform master
    key is always used), so only telephony gaps should appear here."""
    from src.config_tenant import (
        TenantLLMConfig, TenantPipelineConfig, TenantSTTConfig,
        TenantTTSConfig, TenantTelephonyConfig,
    )
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            stt=TenantSTTConfig(provider="sarvam"),       # no api_key_env needed
            llm=TenantLLMConfig(provider="groq"),         # no api_key_env needed
            tts=TenantTTSConfig(provider="sarvam"),       # no api_key_env needed
            telephony=TenantTelephonyConfig(provider="twilio"),  # missing sid + token
        ),
    )
    with pytest.raises(TenantConfigError) as ei:
        validate_credentials(t)
    msg = str(ei.value)
    # Only the telephony gaps should appear in the one error.
    assert "account_sid_env" in msg
    assert "auth_token_env" in msg


def test_validate_credentials_includes_source_in_error() -> None:
    """When called via load_tenant, the source path appears in the message
    so admins can find the offending YAML. Telephony is used here since it's
    the layer where a missing env-var reference still raises."""
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="acme", name="T1",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider="twilio"),
        ),
    )
    with pytest.raises(TenantConfigError, match="config/tenants/acme.yaml"):
        validate_credentials(t, source="config/tenants/acme.yaml")


def test_load_tenant_rejects_telephony_provider_without_creds(tenant_dir: Path) -> None:
    """End-to-end: load_tenant fails on bootstrap, not on first request.

    Telephony is the layer where a missing credential env-var reference is
    still a real gap (STT/LLM/TTS always use the platform master key, so
    those are no longer validated here — see test_load_tenant_accepts_*)."""
    _write(tenant_dir / "broken.yaml", """
id: t_broken
slug: broken
name: Broken
pipeline:
  telephony: {provider: twilio}
""")
    with pytest.raises(TenantConfigError, match="account_sid_env"):
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


def test_validate_credentials_passes_when_realtime_provider_has_no_key() -> None:
    """Realtime (s2s) always resolves its key from the platform master env
    var, so a tenant-level api_key_env is optional — omitting it is not an
    error."""
    from src.config_tenant import TenantPipelineConfig, TenantRealtimeConfig
    t = TenantSettings(id="t1", slug="t1", name="T1",
                       pipeline=TenantPipelineConfig(
                           realtime=TenantRealtimeConfig(provider="gemini_live")))
    validate_credentials(t)


def test_validate_credentials_passes_s2s_with_realtime() -> None:
    from src.config_tenant import TenantPipelineConfig, TenantRealtimeConfig
    t = TenantSettings(id="t1", slug="t1", name="T1",
                       pipeline=TenantPipelineConfig(
                           mode="s2s",
                           realtime=TenantRealtimeConfig(provider="gemini_live", api_key_env="GK")))
    validate_credentials(t)


def test_validate_credentials_passes_webconsole_without_creds() -> None:
    """webconsole is a browser transport, not a real telephony account —
    there is no adapter and no per-tenant creds to fall back from, so it is
    exempt from the account_sid_env/auth_token_env requirement."""
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider="webconsole"),
        ),
    )
    validate_credentials(t)


def test_validate_credentials_passes_webconsole_mixed_case() -> None:
    """The provider string is never normalized on write (tenants.py stores it
    verbatim from the request body), and every other consumer of
    telephony.provider lowercases before comparing. The exemption check must
    do the same, or a tenant registered as "WebConsole" would be refused
    outbound dialing by calls.py yet still be hard-required to declare
    telephony creds here."""
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider="WebConsole"),
        ),
    )
    validate_credentials(t)


def test_validate_credentials_passes_webconsole_s2s_with_chat_voice() -> None:
    """The real-world shape that motivated the exemption: a webconsole tenant
    running s2s realtime with its own chat_voice TTS override, and no
    telephony credentials at all — none are needed for webconsole."""
    from src.config_tenant import (
        ChatVoiceConfig, TenantPipelineConfig, TenantRealtimeConfig,
        TenantTTSConfig, TenantTelephonyConfig,
    )
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            mode="s2s",
            telephony=TenantTelephonyConfig(provider="webconsole"),
            realtime=TenantRealtimeConfig(provider="gemini_live"),
            chat_voice=ChatVoiceConfig(enabled=True, tts=TenantTTSConfig(provider="google")),
        ),
    )
    validate_credentials(t)


def test_validate_credentials_raises_when_webconsole_chat_voice_has_no_tts() -> None:
    """The exemption is narrow: it only skips the telephony credential check.
    A webconsole tenant with chat_voice enabled and no resolvable TTS
    anywhere still raises on the chat_voice rule."""
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider="webconsole"),
            chat_voice=ChatVoiceConfig(enabled=True),
        ),
    )
    with pytest.raises(TenantConfigError, match="pipeline.chat_voice.tts") as exc:
        validate_credentials(t)
    assert "account_sid_env" not in str(exc.value)
    assert "auth_token_env" not in str(exc.value)


def test_validate_credentials_raises_when_webconsole_s2s_without_realtime() -> None:
    """The exemption is narrow: it only skips the telephony credential check.
    A webconsole tenant in s2s mode with no realtime.provider still raises on
    the realtime rule."""
    from src.config_tenant import TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            mode="s2s",
            telephony=TenantTelephonyConfig(provider="webconsole"),
        ),
    )
    with pytest.raises(TenantConfigError, match="pipeline.realtime") as exc:
        validate_credentials(t)
    assert "account_sid_env" not in str(exc.value)
    assert "auth_token_env" not in str(exc.value)


# --- Chat voice replies --------------------------------------------------


def test_chat_voice_replies_disabled_by_default() -> None:
    t = TenantSettings(id="t1", slug="t1", name="T1")
    assert t.pipeline.chat_voice.enabled is False
    assert t.pipeline.chat_voice.tts.provider is None


def test_resolve_chat_tts_prefers_chat_voice_block() -> None:
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTTSConfig, resolve_chat_tts_config
    pipeline = TenantPipelineConfig(
        tts=TenantTTSConfig(provider="sarvam"),
        chat_voice=ChatVoiceConfig(enabled=True, tts=TenantTTSConfig(provider="google")),
    )
    result = resolve_chat_tts_config(pipeline)
    assert result is pipeline.chat_voice.tts
    assert result.provider == "google"


def test_resolve_chat_tts_falls_back_to_pipeline_tts() -> None:
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTTSConfig, resolve_chat_tts_config
    pipeline = TenantPipelineConfig(
        tts=TenantTTSConfig(provider="sarvam"),
        chat_voice=ChatVoiceConfig(enabled=True),  # no chat-specific tts
    )
    result = resolve_chat_tts_config(pipeline)
    assert result is pipeline.tts


def test_resolve_chat_tts_ignores_chat_block_without_provider() -> None:
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTTSConfig, resolve_chat_tts_config
    pipeline = TenantPipelineConfig(
        tts=TenantTTSConfig(provider="sarvam"),
        chat_voice=ChatVoiceConfig(enabled=True, tts=TenantTTSConfig(voice_id="anushka")),
    )
    result = resolve_chat_tts_config(pipeline)
    assert result is pipeline.tts


def test_resolve_chat_tts_returns_none_when_neither_has_a_provider() -> None:
    """The motivating case: a pure s2s tenant (calls handled entirely by
    pipeline.realtime) has no pipeline.tts at all, and no chat-specific
    override either — there is nothing to resolve chat audio from."""
    from src.config_tenant import TenantPipelineConfig, TenantRealtimeConfig, resolve_chat_tts_config
    pipeline = TenantPipelineConfig(
        mode="s2s", realtime=TenantRealtimeConfig(provider="gemini_live"),
    )
    assert resolve_chat_tts_config(pipeline) is None


def test_resolve_chat_tts_is_independent_of_enabled() -> None:
    """resolve_chat_tts_config deliberately ignores `enabled` — that gate is
    checked separately by the registry (`get_chat_tts`) and the validator
    (`validate_credentials`)."""
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTTSConfig, resolve_chat_tts_config
    pipeline = TenantPipelineConfig(
        chat_voice=ChatVoiceConfig(enabled=False, tts=TenantTTSConfig(provider="google")),
    )
    result = resolve_chat_tts_config(pipeline)
    assert result is pipeline.chat_voice.tts


def test_validate_credentials_raises_when_chat_voice_enabled_without_any_tts() -> None:
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantRealtimeConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            mode="s2s", realtime=TenantRealtimeConfig(provider="gemini_live"),
            chat_voice=ChatVoiceConfig(enabled=True),
        ),
    )
    with pytest.raises(TenantConfigError, match="chat_voice") as ei:
        validate_credentials(t)
    assert "pipeline.chat_voice.tts" in str(ei.value)


def test_validate_credentials_passes_when_chat_voice_enabled_with_pipeline_tts_only() -> None:
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTTSConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            tts=TenantTTSConfig(provider="sarvam"),
            chat_voice=ChatVoiceConfig(enabled=True),
        ),
    )
    validate_credentials(t)


def test_validate_credentials_passes_when_chat_voice_disabled_and_no_tts_anywhere() -> None:
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantRealtimeConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            mode="s2s", realtime=TenantRealtimeConfig(provider="gemini_live"),
            chat_voice=ChatVoiceConfig(enabled=False),
        ),
    )
    validate_credentials(t)


def test_validate_credentials_collects_chat_voice_gap_alongside_telephony_gaps() -> None:
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTelephonyConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            telephony=TenantTelephonyConfig(provider="twilio"),  # missing sid + token
            chat_voice=ChatVoiceConfig(enabled=True),  # nothing resolvable
        ),
    )
    with pytest.raises(TenantConfigError) as ei:
        validate_credentials(t)
    msg = str(ei.value)
    assert "account_sid_env" in msg
    assert "auth_token_env" in msg
    assert "chat_voice" in msg


def test_load_tenant_accepts_chat_voice_block(tenant_dir: Path) -> None:
    _write(tenant_dir / "chatvoice.yaml", """
id: t_chatvoice
slug: chatvoice
name: Chat Voice Tenant
pipeline:
  chat_voice:
    enabled: true
    tts: {provider: sarvam, voice_id: priya}
""")
    t = load_tenant("chatvoice", tenant_dir)
    assert t.pipeline.chat_voice.enabled is True
    assert t.pipeline.chat_voice.tts.voice_id == "priya"


def test_chat_voice_survives_pipeline_config_round_trip() -> None:
    """This is the exact mechanism src/auth/seed.py -> src/auth/db_resolver.py
    use to round-trip tenant config through the DB's JSON column, which is
    why chat_voice lives under `pipeline` rather than somewhere that isn't
    part of that round-trip."""
    from src.config_tenant import ChatVoiceConfig, TenantPipelineConfig, TenantTTSConfig
    t = TenantSettings(
        id="t1", slug="t1", name="T1",
        pipeline=TenantPipelineConfig(
            chat_voice=ChatVoiceConfig(enabled=True, tts=TenantTTSConfig(provider="google")),
        ),
    )
    pc = t.pipeline.model_dump()
    rebuilt = TenantPipelineConfig(**pc)
    assert rebuilt.chat_voice.enabled is True
    assert rebuilt.chat_voice.tts.provider == "google"
