"""`apply_platform_pipeline_override` (src/config_tenant.py) -- the overlay
used both at every boot (src/main.py's lifespan) and live
(src/api/platform.py's PUT/DELETE) to apply an admin-set platform pipeline
default on top of config/default.yaml's own layer dict. Mirrors the
cross-provider guard already covered for the per-tenant case by
test_merge_provider_config_drops_default_model_and_voice_on_tts_provider_switch
in test_tenant_config.py.
"""

from __future__ import annotations

from src.config_tenant import apply_platform_pipeline_override


def test_same_provider_model_change_keeps_other_fields() -> None:
    """Same provider as the yaml default -- the cross-provider guard doesn't
    apply, so unrelated yaml fields (e.g. `language`) survive untouched while
    `model` takes the new value."""
    yaml_layer = {"provider": "sarvam", "language": "hi-IN", "model": "saaras:v2"}
    out = apply_platform_pipeline_override("stt", yaml_layer, "sarvam", "saaras:v3")
    assert out["provider"] == "sarvam"
    assert out["language"] == "hi-IN"
    assert out["model"] == "saaras:v3"


def test_different_provider_drops_stt_model_before_setting_new_provider() -> None:
    yaml_layer = {"provider": "sarvam", "language": "hi-IN", "model": "saaras:v2"}
    out = apply_platform_pipeline_override("stt", yaml_layer, "groq", None)
    assert out["provider"] == "groq"
    assert "model" not in out
    assert out["language"] == "hi-IN"  # not provider-specific, survives


def test_different_provider_drops_llm_model_before_setting_new_provider() -> None:
    yaml_layer = {"provider": "gemini", "model": "gemini-3.8-flash", "temperature": 0.7}
    out = apply_platform_pipeline_override("llm", yaml_layer, "groq", "llama-3.3-70b-versatile")
    assert out["provider"] == "groq"
    assert out["model"] == "llama-3.3-70b-versatile"
    assert out["temperature"] == 0.7


def test_different_provider_drops_tts_model_and_voice_id() -> None:
    """TTS's provider-specific set is {"model", "voice_id"} -- both must be
    dropped on a provider switch, not just model."""
    yaml_layer = {"provider": "sarvam", "model": "bulbul:v3", "voice_id": "anushka",
                  "language": "hi-IN"}
    out = apply_platform_pipeline_override("tts", yaml_layer, "elevenlabs", None)
    assert out["provider"] == "elevenlabs"
    assert "model" not in out
    assert "voice_id" not in out
    assert out["language"] == "hi-IN"


def test_tts_gains_a_model_key_it_never_had_in_the_yaml_dict() -> None:
    """TTSConfig (src/config.py) has no `model` field at all, so
    config/default.yaml's own tts dict never carries one -- an admin override
    that sets a model must still add the key."""
    yaml_layer = {"provider": "sarvam", "voice_id": "anushka", "language": "hi-IN"}
    assert "model" not in yaml_layer
    out = apply_platform_pipeline_override("tts", yaml_layer, "sarvam", "bulbul:v3")
    assert out["model"] == "bulbul:v3"


def test_model_none_clears_an_existing_model_key() -> None:
    yaml_layer = {"provider": "sarvam", "model": "saaras:v2"}
    out = apply_platform_pipeline_override("stt", yaml_layer, "sarvam", None)
    assert "model" not in out


def test_does_not_mutate_the_input_dict() -> None:
    """Repeated calls must always start from the SAME original yaml dict
    (src/config_tenant.py's own docstring for this function) -- so this must
    never mutate its `yaml_layer` argument in place."""
    yaml_layer = {"provider": "sarvam", "model": "saaras:v2"}
    original = dict(yaml_layer)
    apply_platform_pipeline_override("stt", yaml_layer, "groq", "whisper-large-v3")
    assert yaml_layer == original
