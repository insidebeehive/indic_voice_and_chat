from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.api.dev_console import make_browser_bridge_factory
from src.bootstrap import make_bridge_factory, make_exotel_bridge_factory
from src.dialogue.slots import SlotSchema


def test_all_factories_accept_slots_param_defaulting_empty() -> None:
    for fn in (make_bridge_factory, make_exotel_bridge_factory, make_browser_bridge_factory):
        params = inspect.signature(fn).parameters
        assert "slots" in params, f"{fn.__name__} missing slots param"
        default = params["slots"].default
        assert isinstance(default, SlotSchema) and default.specs == {}, fn.__name__


def _providers() -> SimpleNamespace:
    return SimpleNamespace(
        get_stt=lambda t: Mock(), get_llm=lambda t: Mock(), get_tts=lambda t: Mock(),
    )


def _tenant() -> SimpleNamespace:
    pipeline = SimpleNamespace(
        stt=SimpleNamespace(language="hi-IN"),
        llm=SimpleNamespace(temperature=0.5, max_tokens=256, response_format="json"),
        tts=SimpleNamespace(language="hi-IN", voice_id=None),
    )
    return SimpleNamespace(slug="dev", id="t1", settings=SimpleNamespace(pipeline=pipeline))


async def test_browser_factory_passes_slots_into_agent() -> None:
    slots = SlotSchema.from_campaign_yaml({"foo": {"type": "string"}})
    factory = make_browser_bridge_factory(_providers(), slots=slots)
    bridge = await factory(websocket=object(), tenant=_tenant())
    # No resolver → the closure campaign's schema reaches the agent, not an empty one.
    assert bridge._agent.slots.schema is slots


async def test_browser_factory_threads_lead_name_from_query() -> None:
    ws = SimpleNamespace(query_params={"lead_name": "Raju"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=_tenant())
    # The page-supplied lead name reaches the agent session (for the opening + prompt).
    assert bridge._agent.session.lead_data.get("lead_name") == "Raju"


def test_telephony_factories_are_async_for_per_call_campaign() -> None:
    # The telephony factories must be async so they can resolve the per-tenant
    # campaign per call (await resolver.resolve) like the dev-console ones.
    from src.bootstrap import (
        make_bridge_factory, make_exotel_bridge_factory, make_stringee_bridge_factory,
    )
    for fn in (make_bridge_factory, make_exotel_bridge_factory, make_stringee_bridge_factory):
        factory = fn(_providers())
        assert inspect.iscoroutinefunction(factory), fn.__name__


async def test_s2s_factory_tolerates_missing_tenant_realtime_key() -> None:
    """s2s bridge build must NOT raise when the tenant's ``realtime.api_key_env``
    is unset — the key is passed as ``None`` so ``GeminiLiveSession.connect`` can
    fall back to the platform ``GEMINI_API_KEY``/``GOOGLE_API_KEY``. Resolving it
    with the *raising* ``tenant.secret()`` instead crashes the bridge on connect
    → the Twilio WS dies → the call drops instantly with no audio."""
    from src.api import dev_call_control
    from src.api.telephony_live_bridge import TelephonyLiveBridge
    from src.config_tenant import MissingEnvError

    realtime = SimpleNamespace(
        model="gemini-x-live", voice="Aoede", allowed_voices=["Aoede"],
        language_code="hi-IN", api_key_env="TENANT_DEV_GEMINI_KEY")
    pipeline = SimpleNamespace(
        stt=SimpleNamespace(language="hi-IN"),
        llm=SimpleNamespace(temperature=0.5, max_tokens=256, response_format="json"),
        tts=SimpleNamespace(language="hi-IN", voice_id=None),
        realtime=realtime,
    )

    def _raises(env_var):  # the real TenantContext.secret() raises on a missing key
        raise MissingEnvError(f"{env_var!r} not set")

    tenant = SimpleNamespace(
        slug="dev", id="t1",
        settings=SimpleNamespace(pipeline=pipeline, timezone="Asia/Kolkata"),
        secret=_raises,
        secret_optional=lambda env_var: None,
    )

    dev_call_control.set_override("dev", mode="s2s", voice="Aoede", lead_name="")
    try:
        factory = make_bridge_factory(_providers())
        bridge = await factory(websocket=object(), tenant=tenant)
    finally:
        dev_call_control.pop_override("dev")
    assert isinstance(bridge, TelephonyLiveBridge)


async def test_browser_factory_loads_chat_handoff(fake_redis) -> None:
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about Plan B", "customer_id": "cust1"}))
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=_tenant())
    ld = bridge._agent.session.lead_data
    assert ld["name"] == "Raju"
    assert ld["chat_summary"] == "asked about Plan B"
    assert ld["customer_id"] == "cust1"


async def test_browser_factory_raises_when_llm_override_fails(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    ws = SimpleNamespace(query_params={"llm": "vllm"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="VLLM_BASE_URL"):
        await factory(websocket=ws, tenant=_tenant())


async def test_browser_factory_raises_when_tts_override_fails(monkeypatch) -> None:
    monkeypatch.delenv("INDICF5_TTS_URL", raising=False)
    ws = SimpleNamespace(query_params={"tts": "indicf5"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="INDICF5_TTS_URL"):
        await factory(websocket=ws, tenant=_tenant())


async def test_browser_factory_raises_when_batch_stt_override_fails(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ws = SimpleNamespace(query_params={"stt": "groq"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        await factory(websocket=ws, tenant=_tenant())


async def test_browser_factory_raises_when_streaming_stt_override_fails(monkeypatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    ws = SimpleNamespace(query_params={"stt": "deepgram"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        await factory(websocket=ws, tenant=_tenant())


async def test_browser_factory_resolves_campaign_per_call() -> None:
    from src.dialogue.campaign_loader import LoadedCampaign
    from src.dialogue.prompts import VoiceBotScript

    resolved = LoadedCampaign(
        VoiceBotScript.from_campaign_yaml({"name": "FromDB", "company": "Acme"}),
        SlotSchema.from_campaign_yaml({"db_slot": {"type": "string"}}))
    seen = {}

    class _Resolver:
        async def resolve(self, tenant_id, campaign_id=None):
            seen["args"] = (tenant_id, campaign_id)
            return resolved

    ws = SimpleNamespace(query_params={"campaign": "camp_9"})
    factory = make_browser_bridge_factory(_providers(), campaign_resolver=_Resolver())
    bridge = await factory(websocket=ws, tenant=_tenant())
    # The agent uses the DB-resolved campaign, and the ?campaign= id was passed through.
    assert seen["args"] == ("t1", "camp_9")
    assert bridge._agent.slots.schema is resolved.slots
    assert bridge._agent._script is resolved.script
