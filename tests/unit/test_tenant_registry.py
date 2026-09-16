from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.auth.context import TenantContext
from src.auth.registry import (
    TenantProviders,
    TenantRuntimeRegistry,
    _PerTenantRegistry,
    make_per_tenant_registry,
)
from src.config_tenant import (
    ChatVoiceConfig,
    TenantLLMConfig,
    TenantPipelineConfig,
    TenantRealtimeConfig,
    TenantSTTConfig,
    TenantSettings,
    TenantTTSConfig,
    TenantTelephonyConfig,
)


def _tenant(slug: str, *, stt_key_env: str = "K1", llm_key_env: str = "K2",
            twilio_sid: str = "SID", twilio_tok: str = "TOK") -> TenantContext:
    s = TenantSettings(
        id=f"t_{slug}", slug=slug, name=slug.title(),
        pipeline=TenantPipelineConfig(
            stt=TenantSTTConfig(provider="sarvam", api_key_env=stt_key_env),
            llm=TenantLLMConfig(provider="groq", api_key_env=llm_key_env),
            tts=TenantTTSConfig(provider="sarvam", voice_id="meera", api_key_env=stt_key_env),
            telephony=TenantTelephonyConfig(
                provider="twilio",
                account_sid_env=twilio_sid,
                auth_token_env=twilio_tok,
            ),
        ),
    )
    return TenantContext(settings=s)


def _chat_tenant(
    *, enabled: bool, chat_tts: TenantTTSConfig | None = None,
    pipeline_tts: TenantTTSConfig | None = None, mode: str = "layered",
    realtime: TenantRealtimeConfig | None = None,
) -> TenantContext:
    s = TenantSettings(
        id="t_chat", slug="chat", name="Chat",
        pipeline=TenantPipelineConfig(
            mode=mode,
            realtime=realtime,
            tts=pipeline_tts or TenantTTSConfig(),
            chat_voice=ChatVoiceConfig(enabled=enabled, tts=chat_tts or TenantTTSConfig()),
        ),
    )
    return TenantContext(settings=s)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("K1", "acme-sarvam-key")
    monkeypatch.setenv("K2", "acme-groq-key")
    monkeypatch.setenv("SID", "ACacme")
    monkeypatch.setenv("TOK", "tok-acme")
    monkeypatch.setenv("K3", "globex-sarvam-key")
    monkeypatch.setenv("K4", "globex-groq-key")
    monkeypatch.setenv("SID2", "ACglobex")
    monkeypatch.setenv("TOK2", "tok-globex")
    yield


def _captured_factory():
    """Returns a factory plus the list of config dicts it has been called with."""
    calls: list[dict[str, Any]] = []

    def factory(cfg: dict[str, Any]) -> Any:
        calls.append(dict(cfg))
        return MagicMock(name=f"client-{len(calls)}", config=cfg)

    return factory, calls


def _providers(tmp_path: Path) -> tuple[TenantProviders, dict[str, list]]:
    stt, stt_calls = _captured_factory()
    llm, llm_calls = _captured_factory()
    tts, tts_calls = _captured_factory()
    tele, tele_calls = _captured_factory()
    vec, vec_calls = _captured_factory()
    providers = TenantProviders(
        global_defaults={
            "stt": {"language": "hi-IN", "model": "saaras:v2"},
            "llm": {"temperature": 0.7, "max_tokens": 512},
            "tts": {"language": "hi-IN", "speed": 1.0},
            "telephony": {"from_number": "+9100", "webhook_base_url": "https://x"},
            "vector_store": {"embedding_dim": 384},
        },
        stt_factory=stt,
        llm_factory=llm,
        tts_factory=tts,
        telephony_factory=tele,
        vector_store_factory=vec,
        base_vector_path=tmp_path / "faiss",
    )
    return providers, {"stt": stt_calls, "llm": llm_calls, "tts": tts_calls,
                       "telephony": tele_calls, "vector_store": vec_calls}


# --- TenantProviders ----------------------------------------------------


def test_get_stt_does_not_inject_tenant_api_key(tmp_path, env) -> None:
    # STT/LLM/TTS keys are PLATFORM-level: the registry must NOT resolve a
    # per-tenant api_key — the adapter reads its platform env var (SARVAM_API_KEY
    # etc.) itself. So no "api_key" is injected into the provider config.
    providers, calls = _providers(tmp_path)
    t = _tenant("acme")
    providers.get_stt(t)
    assert "api_key" not in calls["stt"][0]
    assert calls["stt"][0]["provider"] == "sarvam"
    # global defaults survived
    assert calls["stt"][0]["model"] == "saaras:v2"
    assert calls["stt"][0]["language"] == "hi-IN"


def test_non_telephony_never_reads_tenant_secret(tmp_path, env) -> None:
    """Even when the tenant config references an UNSET api_key_env, building an
    stt/llm/tts client must NOT raise — the per-tenant key lookup is gone, so the
    placeholder/missing var is irrelevant (this is the dev TENANT_DEV_GEMINI_KEY
    bug class)."""
    providers, calls = _providers(tmp_path)
    t = _tenant("acme", stt_key_env="DEFINITELY_UNSET", llm_key_env="ALSO_UNSET")
    providers.get_stt(t)   # must not raise MissingEnvError
    providers.get_llm(t)
    assert "api_key" not in calls["stt"][0]
    assert "api_key" not in calls["llm"][0]


def test_telephony_resolves_both_secrets(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _tenant("acme")
    providers.get_telephony(t)
    cfg = calls["telephony"][0]
    assert cfg["account_sid"] == "ACacme"
    assert cfg["auth_token"] == "tok-acme"
    assert cfg["from_number"] == "+9100"  # from global


def test_vector_store_index_path_is_per_tenant(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _tenant("acme")
    providers.get_vector_store(t)
    path = Path(calls["vector_store"][0]["index_path"])
    assert "t_acme" in str(path)
    # Directory was created
    assert path.parent.exists()


def test_providers_cache_per_tenant_layer(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _tenant("acme")
    a = providers.get_llm(t)
    b = providers.get_llm(t)
    assert a is b
    assert len(calls["llm"]) == 1


def test_providers_separate_instances_per_tenant(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    acme = _tenant("acme")
    globex = _tenant("globex", stt_key_env="K3", llm_key_env="K4",
                     twilio_sid="SID2", twilio_tok="TOK2")
    providers.get_llm(acme)
    providers.get_llm(globex)
    assert len(calls["llm"]) == 2          # a separate cached instance per tenant
    # Keys are platform-level now, so none are injected per tenant.
    assert "api_key" not in calls["llm"][0]
    assert "api_key" not in calls["llm"][1]


def test_provider_evict_drops_cached_clients(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _tenant("acme")
    providers.get_llm(t)
    providers.evict("t_acme")
    providers.get_llm(t)  # rebuilds
    assert len(calls["llm"]) == 2


# --- _PerTenantRegistry -------------------------------------------------


def test_per_tenant_registry_caches_one_per_tenant(env, tmp_path) -> None:
    factory = MagicMock(side_effect=lambda t: {"slug": t.slug, "n": 1})
    reg = make_per_tenant_registry(factory)
    acme = _tenant("acme")
    globex = _tenant("globex", stt_key_env="K3", llm_key_env="K4",
                     twilio_sid="SID2", twilio_tok="TOK2")
    a1 = reg.get(acme)
    a2 = reg.get(acme)
    g = reg.get(globex)
    assert a1 is a2          # cached
    assert g is not a1       # distinct tenant
    assert factory.call_count == 2


def test_per_tenant_registry_evict() -> None:
    factory = MagicMock(side_effect=lambda t: object())
    reg = make_per_tenant_registry(factory)
    t = _tenant("acme")
    reg.get(t)
    assert reg.has("t_acme") is True
    reg.evict("t_acme")
    assert reg.has("t_acme") is False


def test_runtime_registry_evict_clears_everywhere(tmp_path, env) -> None:
    providers, _ = _providers(tmp_path)
    retrievers = make_per_tenant_registry(lambda t: object())
    runtime = TenantRuntimeRegistry(
        providers=providers,
        retrievers=retrievers,
        dnd=make_per_tenant_registry(lambda t: object()),
        schedulers=make_per_tenant_registry(lambda t: object()),
        webhooks=make_per_tenant_registry(lambda t: object()),
        chat_channels=make_per_tenant_registry(lambda t: object()),
        session_stores=make_per_tenant_registry(lambda t: object()),
        crms=make_per_tenant_registry(lambda t: object()),
    )
    acme = _tenant("acme")
    runtime.providers.get_llm(acme)
    runtime.retrievers.get(acme)
    runtime.evict_tenant("t_acme")
    assert not runtime.retrievers.has("t_acme")


# --- chat TTS ------------------------------------------------------------


def test_get_chat_tts_returns_none_when_voice_replies_disabled(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(enabled=False, pipeline_tts=TenantTTSConfig(provider="sarvam", voice_id="meera"))
    assert providers.get_chat_tts(t) is None
    assert calls["tts"] == []
    assert (t.id, "chat_tts") not in providers._cache


def test_get_chat_tts_uses_chat_voice_tts_over_pipeline_tts(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(
        enabled=True,
        chat_tts=TenantTTSConfig(provider="google", voice_id="en-IN-Wavenet-D"),
        pipeline_tts=TenantTTSConfig(provider="sarvam", voice_id="meera"),
    )
    result = providers.get_chat_tts(t)
    assert result is not None
    assert calls["tts"][0]["provider"] == "google"
    assert calls["tts"][0]["voice_id"] == "en-IN-Wavenet-D"
    assert calls["tts"][0]["language"] == "hi-IN"  # global default survived
    assert "api_key" not in calls["tts"][0]


def test_get_chat_tts_falls_back_to_pipeline_tts_when_chat_block_empty(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(
        enabled=True, chat_tts=None,
        pipeline_tts=TenantTTSConfig(provider="sarvam", voice_id="meera"),
    )
    providers.get_chat_tts(t)
    assert calls["tts"][0]["provider"] == "sarvam"
    assert calls["tts"][0]["voice_id"] == "meera"
    assert len(calls["tts"]) == 1


def test_get_chat_tts_ignores_chat_block_without_provider(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(
        enabled=True,
        chat_tts=TenantTTSConfig(voice_id="anushka"),  # no provider — doesn't count
        pipeline_tts=TenantTTSConfig(provider="sarvam", voice_id="meera"),
    )
    providers.get_chat_tts(t)
    assert calls["tts"][0]["provider"] == "sarvam"
    assert calls["tts"][0]["voice_id"] == "meera"  # not "anushka"


def test_s2s_tenant_with_no_chat_tts_never_hits_platform_default(tmp_path, env) -> None:
    """Regression guard: an s2s tenant with no TTS config of its own must NOT
    silently fall through to `global_defaults["tts"]` and get billed for chat
    voice replies on the platform's own provider/key. The factory must never
    even be invoked."""
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(
        enabled=True, mode="s2s", realtime=TenantRealtimeConfig(provider="gemini_live"),
        pipeline_tts=TenantTTSConfig(),  # all unset
    )
    assert providers.get_chat_tts(t) is None
    assert calls["tts"] == []
    assert providers._cache == {}


def test_s2s_tenant_with_chat_tts_gets_its_own_provider(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(
        enabled=True, mode="s2s", realtime=TenantRealtimeConfig(provider="gemini_live"),
        chat_tts=TenantTTSConfig(provider="elevenlabs", voice_id="rachel"),
        pipeline_tts=TenantTTSConfig(),
    )
    result = providers.get_chat_tts(t)
    assert result is not None
    assert calls["tts"][0]["provider"] == "elevenlabs"


def test_chat_tts_cached_under_a_key_distinct_from_call_tts(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(enabled=True, pipeline_tts=TenantTTSConfig(provider="sarvam", voice_id="meera"))
    call_client_1 = providers.get_tts(t)
    chat_client_1 = providers.get_chat_tts(t)
    call_client_2 = providers.get_tts(t)
    chat_client_2 = providers.get_chat_tts(t)
    assert len(calls["tts"]) == 2  # two builds total, caching works both ways
    assert call_client_1 is call_client_2
    assert chat_client_1 is chat_client_2
    assert call_client_1 is not chat_client_1
    assert (t.id, "tts") in providers._cache
    assert (t.id, "chat_tts") in providers._cache


def test_evict_drops_chat_tts_cache(tmp_path, env) -> None:
    providers, calls = _providers(tmp_path)
    t = _chat_tenant(enabled=True, pipeline_tts=TenantTTSConfig(provider="sarvam", voice_id="meera"))
    providers.get_chat_tts(t)
    providers.evict(t.id)
    providers.get_chat_tts(t)
    assert len(calls["tts"]) == 2
