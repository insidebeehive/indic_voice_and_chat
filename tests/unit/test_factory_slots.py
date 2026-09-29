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
        get_chat_call_tts=lambda t: None,
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


async def test_browser_factory_skips_campaign_resolution_for_chat_handoff(fake_redis) -> None:
    """A chat->voice handoff call has no campaign context, and the resolved
    script/slots get replaced with a support-mode one anyway once the handoff
    blob loads — so campaign resolution must be skipped entirely for these
    calls. Regression guard: a tenant with no active campaign (resolver raises
    CampaignNotConfigured) must NOT have every chat->voice handoff rejected."""
    import json

    from src.dialogue.campaign_resolver import CampaignNotConfigured

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about Plan B", "customer_id": "cust1"}))

    class _RaisingResolver:
        async def resolve(self, tenant_id, campaign_id=None):
            raise CampaignNotConfigured(f"tenant {tenant_id} has no campaign")

    factory = make_browser_bridge_factory(
        _providers(),
        campaign_resolver=_RaisingResolver(),
        handoff_store=SimpleNamespace(redis=fake_redis),
    )
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=_tenant())  # must not raise
    ld = bridge._agent.session.lead_data
    assert ld["name"] == "Raju"
    assert ld["chat_summary"] == "asked about Plan B"


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


async def test_bridge_factory_merges_crm_and_tenant_kb_tiers() -> None:
    """Voice call sites must merge BOTH KB tiers into kb_context — the CRM-wide
    retriever AND the tenant's own opt-in retriever — mirroring chat's
    ``_active_retrievers()`` (src/api/knowledge.py). Regression guard: before
    this fix, ``registry`` was hardcoded to ``None`` at every voice call site,
    so tenant-tier-only content (e.g. casino/sports/matka once moved out of
    the CRM-wide tier) never reached voice calls."""
    from src.interfaces.vector_store import Document

    class _FakeRetriever:
        def __init__(self, docs) -> None:
            self._docs = docs

        def list_all(self, max_chunks: int = 200):
            return self._docs

    crm_doc = Document(
        id="crm-doc", content="CRM-wide FAQ content.", metadata={"filename": "00-crm-faq.md"})
    tenant_doc = Document(
        id="tenant-doc", content="Tenant-only casino content.",
        metadata={"filename": "06-casino-games.md"})

    class _FakeCrmRetrievers:
        def get(self, crm_id):
            return _FakeRetriever([crm_doc])

    fake_registry = SimpleNamespace(
        retrievers=SimpleNamespace(get=lambda tenant: _FakeRetriever([tenant_doc])))

    tenant = _tenant()
    tenant.settings.crm_id = "crm_x"

    factory = make_bridge_factory(
        _providers(), crm_retrievers=_FakeCrmRetrievers(), registry=fake_registry)
    bridge = await factory(websocket=object(), tenant=tenant)

    kb_context = bridge._agent._kb_context
    assert "crm-faq" in kb_context
    assert "casino-games" in kb_context


async def test_bridge_factory_kb_context_survives_cold_bm25(tmp_faiss_index) -> None:
    """Guards the await-plumbing through ``_build_kb_context``/``make_bridge_factory``:
    a tenant retriever built fresh in THIS process (cold in-memory BM25, no
    ``.index()`` call ever made on it — simulating a different worker than the
    one that served the tenant's ingest call) must still surface its
    persistently-stored KB content in the agent's kb_context, via the real
    ``HybridRetriever`` + ``FAISSAdapter`` (not a fake)."""
    from src.interfaces.vector_store import Document
    from src.providers.vector_store.faiss_store import FAISSAdapter
    from src.rag.embeddings import HashEmbedder
    from src.rag.retriever import HybridRetriever

    warm = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    await warm.index([
        Document(
            id="layout_casino::chunk-0",
            content="Casino games include slots and live dealer.",
            metadata={"filename": "06-casino-games.md", "section": 0},
        )
    ])

    cold = HybridRetriever(
        embedder=HashEmbedder(dim=64),
        vector_store=FAISSAdapter({"embedding_dim": 64, "index_path": tmp_faiss_index}),
    )
    assert cold.list_all() == []  # pins the bug's precondition

    fake_registry = SimpleNamespace(retrievers=SimpleNamespace(get=lambda tenant: cold))

    factory = make_bridge_factory(_providers(), registry=fake_registry)
    bridge = await factory(websocket=object(), tenant=_tenant())

    assert "Casino games include slots" in bridge._agent._kb_context


async def test_browser_factory_handoff_load_failure_logs_fingerprint_not_raw_token(fake_redis, caplog) -> None:
    """Fix 3: the handoff-context load-failure path
    (``make_browser_bridge_factory``'s ``except Exception`` around the Redis
    lookup) must log ``token_fingerprint(handoff_token)``, never the raw
    token -- WARNING is on in normal running, so a raw value logged here
    ships to Loki. Mirrors tests/unit/test_auth_secret_logging.py's shape:
    forces the failure with a malformed JSON blob (not a mock) so this is a
    real run through the actual except-branch, not an assertion about intent.
    """
    import logging

    from src.auth.audit import token_fingerprint

    token = "tok-canary-raw-9f3b2c1d"
    await fake_redis.set(f"chat_handoff:{token}", "not-valid-json{")
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": token})

    with caplog.at_level(logging.WARNING):
        await factory(websocket=ws, tenant=_tenant())  # must not raise

    logged = "\n".join(r.getMessage() + " " + repr(r.__dict__) for r in caplog.records)
    assert token not in logged, "raw handoff token was logged"
    assert token_fingerprint(token) in logged, "the token's fingerprint should be logged instead"


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


async def test_handoff_call_uses_chat_call_tts_when_resolved() -> None:
    """A chat->voice handoff call must speak in the chat voice-reply TTS
    (``providers.get_chat_call_tts``), not the call cascade's ``get_tts``,
    when the registry resolves one. Also proves tts_language/tts_voice come
    from ``resolve_chat_tts_config`` (the real function — not mocked), i.e.
    from ``pipeline.chat_voice.tts``, not ``pipeline.tts``."""
    chat_sentinel = Mock()
    providers = _providers()
    providers.get_chat_call_tts = lambda t: chat_sentinel
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    tenant = _tenant()
    tenant.settings.pipeline.tts.provider = None
    tenant.settings.pipeline.chat_voice = SimpleNamespace(
        enabled=False,  # must be ignored entirely for handoff calls
        tts=SimpleNamespace(provider="elevenlabs", model=None, language="en-IN", voice_id="chat-voice-1"),
    )
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=tenant)
    assert bridge._agent._engine._tts is chat_sentinel
    assert bridge._agent._engine._config.tts.language == "en-IN"
    assert bridge._agent._engine._config.tts.voice_id == "chat-voice-1"


async def test_handoff_call_falls_back_to_pipeline_tts_when_chat_call_tts_none() -> None:
    """When get_chat_call_tts resolves nothing (e.g. an s2s-only tenant with a
    layered handoff), the handoff call keeps today's fallback: pipeline.tts's
    client, language and voice_id."""
    pipeline_sentinel = Mock()
    providers = _providers()
    providers.get_tts = lambda t: pipeline_sentinel
    providers.get_chat_call_tts = lambda t: None
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    tenant = _tenant()
    tenant.settings.pipeline.tts = SimpleNamespace(language="hi-IN", voice_id="pipeline-voice")
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=tenant)
    assert bridge._agent._engine._tts is pipeline_sentinel
    assert bridge._agent._engine._config.tts.language == "hi-IN"
    assert bridge._agent._engine._config.tts.voice_id == "pipeline-voice"


async def test_non_handoff_call_never_calls_get_chat_call_tts() -> None:
    """Non-handoff (dev console) calls must be provably untouched by this
    change: get_chat_call_tts must not even be invoked."""
    providers = _providers()
    providers.get_chat_call_tts = Mock(side_effect=AssertionError(
        "get_chat_call_tts must not be called for a non-handoff call"))
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=object(), tenant=_tenant())
    assert bridge._agent._engine._tts is not None
    providers.get_chat_call_tts.assert_not_called()


async def test_handoff_call_with_tts_override_ignores_chat_call_tts() -> None:
    """An explicit ?tts= override on a handoff call still wins via the normal
    override path (get_tts_provider) — get_chat_call_tts must not be
    consulted at all in that branch."""
    providers = _providers()
    providers.get_chat_call_tts = Mock(side_effect=AssertionError(
        "get_chat_call_tts must not be called when ?tts= overrides"))
    ws = SimpleNamespace(query_params={"handoff": "tok1", "tts": "sarvam"})
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=_tenant())
    from src.providers.tts.sarvam import SarvamTTSAdapter
    assert isinstance(bridge._agent._engine._tts, SarvamTTSAdapter)
    providers.get_chat_call_tts.assert_not_called()


async def test_non_handoff_call_does_not_emit_handoff_tts_source_debug_event(caplog) -> None:
    """Non-handoff (dev console) calls must not emit the
    "dev_console browser_bridge handoff_tts_source" debug event at all — all
    three tts-selection branches only fire it `if is_handoff_call`. Checked
    directly against the log records (not just the get_chat_call_tts spy in
    test_non_handoff_call_never_calls_get_chat_call_tts above), so a future
    branch that emits the event unconditionally would be caught even if it
    didn't also call get_chat_call_tts."""
    import logging

    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with caplog.at_level(logging.DEBUG, logger="src.api.dev_console"):
        await factory(websocket=object(), tenant=_tenant())
    assert not any(
        r.getMessage() == "dev_console browser_bridge handoff_tts_source"
        for r in caplog.records
    )


async def test_handoff_call_voice_override_wins_over_chat_call_tts_resolution() -> None:
    """A dev-console ?voice= override on a handoff call that resolves via
    get_chat_call_tts must still win over resolve_chat_tts_config's own
    voice_id — the override roster check runs against `tts` (== the chat
    call TTS client here) regardless of which branch selected it."""
    class _FakeChatTTS:
        def get_available_voices(self, language):
            return [{"voice_id": "chat-voice-1"}, {"voice_id": "override-voice"}]

    fake_tts = _FakeChatTTS()
    providers = _providers()
    providers.get_chat_call_tts = lambda t: fake_tts
    ws = SimpleNamespace(query_params={"handoff": "tok1", "voice": "override-voice"})
    tenant = _tenant()
    tenant.settings.pipeline.tts.provider = None  # pipeline.tts.provider unset -- forces chat_voice.tts path
    tenant.settings.pipeline.chat_voice = SimpleNamespace(
        enabled=False,  # must be ignored entirely for handoff calls
        tts=SimpleNamespace(provider="elevenlabs", model=None, language="en-IN", voice_id="chat-voice-1"),
    )
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=tenant)
    assert bridge._agent._engine._tts is fake_tts
    assert bridge._agent._engine._config.tts.voice_id == "override-voice"


async def test_handoff_call_missing_chat_tts_language_falls_back_to_pipeline_language() -> None:
    """Fix 1: the language cascade for a handoff call resolved via
    get_chat_call_tts must be resolved.language or pipeline.tts.language or
    "hi-IN" -- NOT resolved.language or "hi-IN" (which skips the pipeline
    cascade's language entirely). Exercises resolve_chat_tts_config for real
    (chat_voice.tts declares a provider but no language, so it resolves with
    language=None), proving the code path in dev_console.py actually falls
    through to tenant.settings.pipeline.tts.language, not just a mock's
    canned return value."""
    chat_sentinel = Mock()
    providers = _providers()
    providers.get_chat_call_tts = lambda t: chat_sentinel
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    tenant = _tenant()
    tenant.settings.pipeline.tts.provider = None
    tenant.settings.pipeline.tts.language = "ta-IN"
    tenant.settings.pipeline.chat_voice = SimpleNamespace(
        enabled=False,
        tts=SimpleNamespace(provider="elevenlabs", model=None, language=None, voice_id="chat-voice-1"),
    )
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=tenant)
    assert bridge._agent._engine._tts is chat_sentinel
    assert bridge._agent._engine._config.tts.language == "ta-IN"
