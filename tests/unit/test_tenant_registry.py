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
    TenantLLMConfig,
    TenantPipelineConfig,
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
