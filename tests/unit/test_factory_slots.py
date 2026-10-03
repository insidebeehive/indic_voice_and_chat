from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.api.dev_console import make_browser_bridge_factory
from src.bootstrap import make_bridge_factory, make_exotel_bridge_factory
from src.defaults import DEFAULT_DEMO_SCRIPT
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
    return SimpleNamespace(
        slug="dev", id="t1",
        settings=SimpleNamespace(
            pipeline=pipeline, name="Acme", default_language="hi", prompt_pack="generic"),
    )


async def test_browser_factory_passes_slots_into_agent() -> None:
    slots = SlotSchema.from_campaign_yaml({"foo": {"type": "string"}})
    factory = make_browser_bridge_factory(_providers(), slots=slots)
    bridge = await factory(websocket=object(), tenant=_tenant())
    # No resolver → the closure campaign's schema reaches the agent, not an empty one.
    assert bridge._agent.slots.schema is slots


async def test_browser_factory_threads_lead_name_from_query() -> None:
    """lead_name is a dev-console override (Fix 2: /chat/voice must not honour
    it), so this test exercises the admin-gated allow_overrides=True path."""
    ws = SimpleNamespace(query_params={"lead_name": "Raju"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)
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


async def test_cascade_pipeline_config_preserves_temperature_zero() -> None:
    """B6: `temperature or 0.5` used to silently replace an intentional
    temperature=0.0 (deterministic output) with 0.5, since 0.0 is falsy.
    make_bridge_factory's cascade (non-s2s) pipeline_cfg must use an explicit
    is-None check for temperature. max_tokens keeps its `or 256` fallback
    (unlike temperature, 0 is not a valid max_tokens for groq/gemini -- an
    explicit 0 there is a misconfiguration, not a deliberate choice)."""
    from src.api import dev_call_control

    pipeline = SimpleNamespace(
        stt=SimpleNamespace(language="hi-IN"),
        llm=SimpleNamespace(temperature=0.0, max_tokens=0, response_format="json"),
        tts=SimpleNamespace(language="hi-IN", voice_id=None),
        mode="layered",
    )
    tenant = SimpleNamespace(
        slug="dev", id="t1",
        settings=SimpleNamespace(pipeline=pipeline, timezone="Asia/Kolkata"),
        secret=lambda env: "k", secret_optional=lambda env: None,
    )

    dev_call_control.pop_override("dev")  # ensure no stale override from another test
    factory = make_bridge_factory(_providers())
    bridge = await factory(websocket=object(), tenant=tenant)

    cfg = bridge._agent._engine._config
    assert cfg.llm.temperature == 0.0
    assert cfg.llm.max_tokens == 256  # 0 falls back, unlike temperature


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


async def test_handoff_call_script_built_from_tenant_not_demo_script(fake_redis) -> None:
    """Build item 2 (P1/B2): the handoff support script's identity must come
    from the TENANT, never DEFAULT_DEMO_SCRIPT (company='Vox Demo',
    agent_name='Priya') -- previously `cur_script` was still the closure's
    DEFAULT_DEMO_SCRIPT because campaign resolution (which would normally
    replace it) is skipped for handoff calls."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing", "customer_id": "cust1"}))
    tenant = _tenant()
    tenant.settings.name = "Acme Telecom"
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    script = bridge._agent._script
    assert script.company_name == "Acme Telecom"
    assert script.company_name != DEFAULT_DEMO_SCRIPT.company_name
    assert script.agent_name != DEFAULT_DEMO_SCRIPT.agent_name  # never "Priya"


async def test_handoff_call_agent_gender_derived_from_chat_call_voice(fake_redis) -> None:
    """F6: the handoff script must carry the gender of the voice the call
    actually speaks in, so Hindi verb forms match -- previously no gender was
    ever set on a handoff script (the ?voice=/?gender= derivation block only
    runs when allow_overrides lets sel_voice/caller_name_override be set,
    which /chat/voice never does)."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing"}))
    tenant = _tenant()
    tenant.settings.pipeline.tts.voice_id = "aditya"  # male, per voice_catalog.py
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    assert bridge._agent._script.gender == "male"


async def test_handoff_call_generic_pack_scope_directive_has_no_betting_words(fake_redis) -> None:
    """Build item 3 (B9): the handoff scope directive must be built from the
    tenant's prompt pack (TIER1_GENERAL), not a hard-coded betting-vertical
    topic list -- a generic-pack tenant's directive must carry none of the
    gambling vocabulary. Banned-word list mirrors
    test_prompts.py::test_chatbot_prompt_generic_pack_has_no_gambling_vocabulary."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing"}))
    tenant = _tenant()  # prompt_pack="generic" by default
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    directives = bridge._agent._extra_directives or []
    text = " ".join(directives).lower()
    for banned in (
        "kyc", "deposit", "withdraw", "self-exclu", "casino",
        "matka", "bonus", "responsible gaming", "betting works",
    ):
        assert banned not in text, f"found banned term {banned!r} in generic handoff directive"


async def test_handoff_call_betting_pack_scope_directive_keeps_betting_topics(fake_redis) -> None:
    """Complements the generic-pack test above: a betting-pack tenant's
    directive still names its own vertical's topics (via TIER1_GENERAL), so
    the pack-reuse didn't just silently drop all topic detail."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing"}))
    tenant = _tenant()
    tenant.settings.prompt_pack = "betting"
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    directives = bridge._agent._extra_directives or []
    text = " ".join(directives).lower()
    assert "kyc" in text
    assert "responsible gaming" in text


async def test_handoff_scope_directive_exact_wording_generic(fake_redis) -> None:
    """Round 2 item 1: the operator-approved sentence, verbatim, for the
    generic pack -- reads from SUPPORT_TOPICS, not the TIER1_GENERAL splice
    (which referenced "1." and "DATA RULE above", neither of which exist in
    the voice prompt)."""
    import json

    from src.dialogue.packs import generic as generic_pack

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing"}))
    tenant = _tenant()
    tenant.settings.name = "Acme"
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    directives = bridge._agent._extra_directives or []
    text = " ".join(directives)
    expected = (
        "SCOPE — SUPPORT CALL: You are customer support for Acme, "
        f"NOT a sales agent. Help only with Acme topics: {generic_pack.SUPPORT_TOPICS}. "
    )
    assert expected in text
    assert "DATA RULE" not in text
    assert "1. GENERAL" not in text


async def test_handoff_scope_directive_exact_wording_betting(fake_redis) -> None:
    """Same sentence, betting pack -- SUPPORT_TOPICS swaps in the betting
    vertical's topic list."""
    import json

    from src.dialogue.packs import betting as betting_pack

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing"}))
    tenant = _tenant()
    tenant.settings.name = "Acme"
    tenant.settings.prompt_pack = "betting"
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    directives = bridge._agent._extra_directives or []
    text = " ".join(directives)
    expected = (
        "SCOPE — SUPPORT CALL: You are customer support for Acme, "
        f"NOT a sales agent. Help only with Acme topics: {betting_pack.SUPPORT_TOPICS}. "
    )
    assert expected in text
    assert "DATA RULE" not in text
    assert "1. GENERAL" not in text


async def test_handoff_call_bot_name_and_gender_from_crm_reach_script(fake_redis) -> None:
    """Round 2 item 2: CRM-supplied bot_name/bot_gender (carried in the
    handoff blob written by chat.py's request_call / in-chat call-offer path)
    must reach the handoff script's identity, taking priority over the
    call's TTS-voice-derived gender."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing",
        "bot_name": "Meera", "bot_gender": "male",
    }))
    tenant = _tenant()
    tenant.settings.name = "Acme"
    tenant.settings.pipeline.tts.voice_id = "priya"  # female, per voice_catalog.py -- must lose to bot_gender
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    script = bridge._agent._script
    assert script.agent_name == "Meera"
    assert script.agent_role == "customer support agent"
    assert script.gender == "male"


async def test_handoff_call_no_bot_name_falls_back_to_neutral_identity(fake_redis) -> None:
    """No CRM bot_name -> the neutral nameless fallback identity (see the
    dev_console.py comment on this block and the report for the rendered
    sentence, pending a small prompts.py change for the fully-natural
    no-name case)."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing"}))
    tenant = _tenant()
    tenant.settings.name = "Acme"
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    script = bridge._agent._script
    assert script.agent_name == "the customer-support agent"
    assert script.agent_role == ""
    from src.dialogue.prompts import build_voicebot_system_prompt
    from src.dialogue.slots import SlotSchema
    prompt = build_voicebot_system_prompt(script, SlotSchema())
    assert "You are the customer-support agent at Acme. You are on a phone call with a customer." in prompt
    assert ", a  at" not in prompt and "with a lead" not in prompt


async def test_handoff_call_gender_chain_falls_back_to_female(fake_redis) -> None:
    """No bot_gender and no derivable TTS-voice gender -> "female", matching
    the chat prompt's own "You are female" default."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing"}))
    tenant = _tenant()
    tenant.settings.pipeline.tts.voice_id = None  # nothing to derive a gender from
    factory = make_browser_bridge_factory(
        _providers(), handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    assert bridge._agent._script.gender == "female"


async def test_browser_factory_raises_when_llm_override_fails(monkeypatch) -> None:
    """?llm= is a dev-console override -- only reached with allow_overrides=True."""
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    ws = SimpleNamespace(query_params={"llm": "vllm"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="VLLM_BASE_URL"):
        await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)


async def test_browser_factory_raises_when_tts_override_fails(monkeypatch) -> None:
    monkeypatch.delenv("INDICF5_TTS_URL", raising=False)
    ws = SimpleNamespace(query_params={"tts": "indicf5"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="INDICF5_TTS_URL"):
        await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)


async def test_browser_factory_raises_when_batch_stt_override_fails(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    ws = SimpleNamespace(query_params={"stt": "groq"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)


async def test_browser_factory_raises_when_streaming_stt_override_fails(monkeypatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    ws = SimpleNamespace(query_params={"stt": "deepgram"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)


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


# --- Hot issues (src/chatbot/hot_issues.py) threading into the cascade agent


async def test_bridge_factory_threads_hot_issues_into_cascade_agent(monkeypatch) -> None:
    """make_bridge_factory (Twilio cascade) must fetch the voice=True variant
    next to its _build_kb_context call and thread the rendered block into
    VoiceBotAgent(hot_issues=...)."""

    async def _fake_get_active_hot_issues(tenant_id, crm_id=None, *, voice=False):
        assert voice is True  # voice call sites must request the voice variant
        return SimpleNamespace(block="SENTINEL_HOT_ISSUES_BLOCK", keys=("t:pg-delay",))

    monkeypatch.setattr(
        "src.chatbot.hot_issues.get_active_hot_issues", _fake_get_active_hot_issues)

    factory = make_bridge_factory(_providers())
    bridge = await factory(websocket=object(), tenant=_tenant())

    assert bridge._agent._hot_issues == "SENTINEL_HOT_ISSUES_BLOCK"


async def test_exotel_factory_threads_hot_issues_into_cascade_agent(monkeypatch) -> None:
    async def _fake_get_active_hot_issues(tenant_id, crm_id=None, *, voice=False):
        assert voice is True
        return SimpleNamespace(block="SENTINEL_HOT_ISSUES_BLOCK", keys=("t:pg-delay",))

    monkeypatch.setattr(
        "src.chatbot.hot_issues.get_active_hot_issues", _fake_get_active_hot_issues)

    factory = make_exotel_bridge_factory(_providers())
    bridge = await factory(websocket=object(), tenant=_tenant())

    assert bridge._agent._hot_issues == "SENTINEL_HOT_ISSUES_BLOCK"


async def test_browser_factory_threads_hot_issues_into_cascade_agent(monkeypatch) -> None:
    async def _fake_get_active_hot_issues(tenant_id, crm_id=None, *, voice=False):
        assert voice is True
        return SimpleNamespace(block="SENTINEL_HOT_ISSUES_BLOCK", keys=("t:pg-delay",))

    monkeypatch.setattr(
        "src.chatbot.hot_issues.get_active_hot_issues", _fake_get_active_hot_issues)

    factory = make_browser_bridge_factory(_providers())
    bridge = await factory(websocket=object(), tenant=_tenant())

    assert bridge._agent._hot_issues == "SENTINEL_HOT_ISSUES_BLOCK"


async def test_bridge_factory_hot_issues_defaults_to_none_when_snapshot_empty() -> None:
    """The conftest autouse stub (empty snapshot) is the default test
    environment -- confirms the `hot.block or None` plumbing doesn't thread
    an empty string into VoiceBotAgent (which would make `if self._hot_issues`
    checks downstream behave differently than a genuine None)."""
    factory = make_bridge_factory(_providers())
    bridge = await factory(websocket=object(), tenant=_tenant())

    assert bridge._agent._hot_issues is None


async def test_stringee_factory_threads_hot_issues_into_cascade_agent(monkeypatch) -> None:
    """make_stringee_bridge_factory has no s2s branch at all -- it only ever
    builds the cascade VoiceBotAgent -- so this mirrors the Twilio/Exotel/
    browser cascade tests above, just via the IVR (call_id/base_url/fetch)
    calling convention instead of a websocket."""
    from src.bootstrap import make_stringee_bridge_factory

    async def _fake_get_active_hot_issues(tenant_id, crm_id=None, *, voice=False):
        assert voice is True
        return SimpleNamespace(block="SENTINEL_HOT_ISSUES_BLOCK", keys=("t:pg-delay",))

    monkeypatch.setattr(
        "src.chatbot.hot_issues.get_active_hot_issues", _fake_get_active_hot_issues)

    factory = make_stringee_bridge_factory(providers=_providers())

    async def _fetch(url):
        return b""

    bridge = await factory(
        call_id="c-9", tenant=_tenant(),
        base_url="https://h/api/v1/telephony/stringee", fetch=_fetch)

    assert bridge._agent._hot_issues == "SENTINEL_HOT_ISSUES_BLOCK"


def _s2s_tenant() -> SimpleNamespace:
    """A tenant configured for s2s mode -- carries both the cascade-style
    pipeline.tts.voice_id (make_bridge_factory/make_exotel_bridge_factory
    compute tts_voice_id BEFORE branching on mode) and the realtime config
    the s2s branch itself needs."""
    rt = SimpleNamespace(model="gemini-x-live", voice="Aoede", allowed_voices=["Aoede"],
                         language_code="hi-IN", api_key_env="K")
    pipeline = SimpleNamespace(
        stt=SimpleNamespace(language="hi-IN"),
        llm=SimpleNamespace(temperature=0.5, max_tokens=256, response_format="json"),
        tts=SimpleNamespace(language="hi-IN", voice_id=None),
        mode="s2s", realtime=rt,
    )
    return SimpleNamespace(
        slug="dev", id="t1",
        settings=SimpleNamespace(
            pipeline=pipeline, name="Acme", default_language="hi", prompt_pack="generic",
            timezone="Asia/Kolkata"),
        secret=lambda env: "fake-key", secret_optional=lambda env: "fake-key",
    )


async def test_twilio_factory_s2s_mode_threads_hot_issues_into_system_instruction(
    monkeypatch,
) -> None:
    """make_bridge_factory's s2s branch (mode == "s2s") must fetch the
    voice=True snapshot and thread hot.block all the way into the
    TelephonyLiveBridge's RealtimeConfig.system_instruction via
    _build_s2s_telephony_bridge."""
    async def _fake_get_active_hot_issues(tenant_id, crm_id=None, *, voice=False):
        assert voice is True
        return SimpleNamespace(block="SENTINEL_HOT_ISSUES_BLOCK", keys=("t:pg-delay",))

    monkeypatch.setattr(
        "src.chatbot.hot_issues.get_active_hot_issues", _fake_get_active_hot_issues)

    factory = make_bridge_factory(_providers())
    bridge = await factory(websocket=object(), tenant=_s2s_tenant())

    assert "SENTINEL_HOT_ISSUES_BLOCK" in bridge._config.system_instruction


async def test_exotel_factory_s2s_mode_threads_hot_issues_into_system_instruction(
    monkeypatch,
) -> None:
    async def _fake_get_active_hot_issues(tenant_id, crm_id=None, *, voice=False):
        assert voice is True
        return SimpleNamespace(block="SENTINEL_HOT_ISSUES_BLOCK", keys=("t:pg-delay",))

    monkeypatch.setattr(
        "src.chatbot.hot_issues.get_active_hot_issues", _fake_get_active_hot_issues)

    factory = make_exotel_bridge_factory(_providers())
    bridge = await factory(websocket=object(), tenant=_s2s_tenant())

    assert "SENTINEL_HOT_ISSUES_BLOCK" in bridge._config.system_instruction


async def test_dev_console_live_bridge_factory_threads_hot_issues_into_s2s_only(monkeypatch) -> None:
    """make_live_bridge_factory (dev_console S2S) must fetch (voice=True) and
    pass hot_issues into the direct build_s2s_system_instruction call, but
    NOT into the VoiceBotAgent it also constructs -- that agent's own prompt
    is never spoken for S2S (see the plan's dev_console S2S-agent note:
    S2S has no pre-TTS guard and the live session speaks from the
    RealtimeConfig.system_instruction instead)."""
    from src.api.dev_console import make_live_bridge_factory

    async def _fake_get_active_hot_issues(tenant_id, crm_id=None, *, voice=False):
        assert voice is True
        return SimpleNamespace(block="SENTINEL_HOT_ISSUES_BLOCK", keys=("t:pg-delay",))

    monkeypatch.setattr(
        "src.chatbot.hot_issues.get_active_hot_issues", _fake_get_active_hot_issues)

    rt = SimpleNamespace(provider="gemini_live", model="m", voice="Aoede", language_code="hi-IN")
    tenant = SimpleNamespace(
        id="t1", slug="dev",
        settings=SimpleNamespace(pipeline=SimpleNamespace(realtime=rt), timezone="Asia/Kolkata"),
    )
    providers = SimpleNamespace(
        get_stt=lambda t: None, get_llm=lambda t: object(), get_tts=lambda t: object())
    ws = SimpleNamespace(query_params={})

    factory = make_live_bridge_factory(providers)
    bridge = await factory(ws, tenant)

    assert "SENTINEL_HOT_ISSUES_BLOCK" in bridge._config.system_instruction
    assert bridge._agent._hot_issues is None  # S2S agent's own prompt is unused


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
    bridge = await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)
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
    bridge = await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)
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
    bridge = await factory(websocket=ws, tenant=tenant, allow_overrides=True)
    assert bridge._agent._engine._tts is fake_tts
    assert bridge._agent._engine._config.tts.voice_id == "override-voice"


async def test_handoff_call_bot_gender_picks_matching_chat_voice() -> None:
    """A handoff call with a CRM-supplied bot_gender (handoff_ctx, passed by
    /chat/voice -- see chat.py's _claim_chat_handoff) picks the matching
    voice from the resolved chat TTS config's `voices` pair
    (resolve_gender_voice, src/config_tenant.py), not just whatever
    `voice_id` it already had. The script's gender agrees too."""
    chat_sentinel = Mock()
    providers = _providers()
    providers.get_chat_call_tts = lambda t: chat_sentinel
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    tenant = _tenant()
    tenant.settings.pipeline.tts.provider = None
    tenant.settings.pipeline.chat_voice = SimpleNamespace(
        enabled=False,  # must be ignored entirely for handoff calls
        tts=SimpleNamespace(
            provider="elevenlabs", model=None, language="en-IN", voice_id="chat-voice-1",
            voices={"male": "chat-voice-male", "female": "chat-voice-female"}),
    )
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=tenant, handoff_ctx={"bot_gender": "male"})
    assert bridge._agent._engine._config.tts.voice_id == "chat-voice-male"
    assert bridge._agent._script.gender == "male"


async def test_dev_voice_handoff_lookup_feeds_gender_match_before_tts_selection(fake_redis) -> None:
    """Regression test for the ordering bug this round fixes: the dev
    console's own ?handoff= Redis lookup (only reached by /dev/voice, which
    never receives an already-resolved handoff_ctx -- /chat/voice always
    does, see test_handoff_call_bot_gender_picks_matching_chat_voice above)
    used to run AFTER TTS source selection, so bot_gender from that blob was
    always None at match time and a configured pair never got used on
    /dev/voice. Moving the lookup above TTS selection in
    make_browser_bridge_factory fixes this."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Raju", "chat_summary": "asked about billing",
        "bot_gender": "male",
    }))
    chat_sentinel = Mock()
    providers = _providers()
    providers.get_chat_call_tts = lambda t: chat_sentinel
    tenant = _tenant()
    tenant.settings.pipeline.tts.provider = None
    tenant.settings.pipeline.chat_voice = SimpleNamespace(
        enabled=False,  # must be ignored entirely for handoff calls
        tts=SimpleNamespace(
            provider="elevenlabs", model=None, language="en-IN", voice_id="chat-voice-1",
            voices={"male": "chat-voice-male", "female": "chat-voice-female"}),
    )
    factory = make_browser_bridge_factory(
        providers, handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(websocket=ws, tenant=tenant)
    assert bridge._agent._engine._config.tts.voice_id == "chat-voice-male"
    assert bridge._agent._script.gender == "male"


async def test_handoff_ctx_supplied_by_caller_skips_redis_lookup(fake_redis) -> None:
    """/chat/voice hands this factory an ALREADY resolved handoff_ctx
    (chat.py's _claim_chat_handoff) -- even when a handoff_store IS also
    wired and Redis has a blob under the same token, the caller-supplied
    handoff_ctx must win untouched, not get overwritten by a fresh lookup.
    Conflicting bot_gender values on each side make a silent overwrite
    observable."""
    import json

    await fake_redis.set("chat_handoff:tok1", json.dumps({
        "customer_name": "Redis Name", "bot_gender": "female",
    }))
    providers = _providers()
    tenant = _tenant()
    factory = make_browser_bridge_factory(
        providers, handoff_store=SimpleNamespace(redis=fake_redis))
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    bridge = await factory(
        websocket=ws, tenant=tenant,
        handoff_ctx={"customer_name": "Caller Name", "bot_gender": "male"})
    assert bridge._agent._script.gender == "male"
    assert bridge._agent.session.lead_data.get("name") == "Caller Name"


async def test_handoff_call_gender_fallback_warns_when_not_configured(caplog) -> None:
    """A handoff call requesting a gender the tenant hasn't configured a
    voice for (here, via the pipeline.tts fallback branch) falls back to the
    plain `voice_id` and logs a warning -- resolve_gender_voice's documented
    no-match contract, not a hard failure."""
    import logging

    pipeline_sentinel = Mock()
    providers = _providers()
    providers.get_tts = lambda t: pipeline_sentinel
    providers.get_chat_call_tts = lambda t: None
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    tenant = _tenant()
    tenant.settings.pipeline.tts = SimpleNamespace(
        language="hi-IN", voice_id="pipeline-voice", voices={"female": "pipeline-voice-f"})
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    with caplog.at_level(logging.WARNING, logger="src.api.dev_console"):
        bridge = await factory(websocket=ws, tenant=tenant, handoff_ctx={"bot_gender": "male"})
    assert bridge._agent._engine._config.tts.voice_id == "pipeline-voice"
    assert any(r.levelno == logging.WARNING and "gender" in r.getMessage().lower()
               for r in caplog.records)


async def test_handoff_call_gender_requested_with_no_pair_at_all_does_not_warn(caplog) -> None:
    """A tenant with NO female/male pair configured at all is the ordinary
    case, not a misconfiguration (see the test above, which has a pair
    missing just one gender and DOES warn) -- this logs a debug_event
    instead of a WARNING."""
    import logging

    pipeline_sentinel = Mock()
    providers = _providers()
    providers.get_tts = lambda t: pipeline_sentinel
    providers.get_chat_call_tts = lambda t: None
    ws = SimpleNamespace(query_params={"handoff": "tok1"})
    tenant = _tenant()
    tenant.settings.pipeline.tts = SimpleNamespace(
        language="hi-IN", voice_id="pipeline-voice", voices=None)
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    with caplog.at_level(logging.DEBUG, logger="src.api.dev_console"):
        bridge = await factory(websocket=ws, tenant=tenant, handoff_ctx={"bot_gender": "male"})
    assert bridge._agent._engine._config.tts.voice_id == "pipeline-voice"
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    assert any(
        r.getMessage() == "dev_console browser_bridge gender_voice_unconfigured"
        for r in caplog.records
    )


async def test_handoff_call_voice_override_wins_over_bot_gender() -> None:
    """An explicit ?voice= override still wins even when bot_gender would
    otherwise pick a different, gender-matched voice -- precedence is
    unchanged by gender-matching."""
    class _FakeChatTTS:
        def get_available_voices(self, language):
            return [{"voice_id": "chat-voice-male"}, {"voice_id": "override-voice"}]

    fake_tts = _FakeChatTTS()
    providers = _providers()
    providers.get_chat_call_tts = lambda t: fake_tts
    ws = SimpleNamespace(query_params={"handoff": "tok1", "voice": "override-voice"})
    tenant = _tenant()
    tenant.settings.pipeline.tts.provider = None
    tenant.settings.pipeline.chat_voice = SimpleNamespace(
        enabled=False,
        tts=SimpleNamespace(
            provider="elevenlabs", model=None, language="en-IN", voice_id="chat-voice-1",
            voices={"male": "chat-voice-male"}),
    )
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(
        websocket=ws, tenant=tenant, allow_overrides=True, handoff_ctx={"bot_gender": "male"})
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


# --- Fix 2: /chat/voice must not accept provider/voice overrides ----------
#
# The public, always-on /chat/voice route (src/api/chat.py's chat_voice_ws)
# calls run_browser_voice(..., allow_overrides=False) -- the default -- so it
# reaches this SAME factory as /dev/voice but with overrides switched off.
# Only ?tenant (resolved before this factory ever runs) and ?handoff stay
# honoured; every dev-console override param below must be ignored.


def test_browser_factory_allow_overrides_defaults_to_false() -> None:
    """Any NEW caller of the factory that forgets to pass allow_overrides
    must be safe by default -- pins the parameter's default, not just its
    current callers' behaviour."""
    import inspect

    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    assert inspect.signature(factory).parameters["allow_overrides"].default is False


async def test_browser_factory_ignores_stt_llm_tts_overrides_by_default() -> None:
    """?stt=/?llm=/?tts= must NOT build a client from the raw query param when
    allow_overrides is left at its default (False, the /chat/voice posture) --
    the tenant's own configured providers are used instead."""
    stt_sentinel, llm_sentinel, tts_sentinel = Mock(), Mock(), Mock()
    providers = _providers()
    providers.get_stt = lambda t: stt_sentinel
    providers.get_llm = lambda t: llm_sentinel
    providers.get_tts = lambda t: tts_sentinel
    # These overrides would otherwise raise (missing keys/URLs) or build a
    # different adapter entirely -- if any of them were honoured, this factory
    # call would either construct a different client than the sentinels below
    # or raise, so a passing assertion here is a real behavioural check, not
    # just an unreached-code guard.
    ws = SimpleNamespace(query_params={"stt": "deepgram", "llm": "vllm", "tts": "elevenlabs"})
    factory = make_browser_bridge_factory(providers, slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=_tenant())  # allow_overrides defaults to False
    assert bridge._agent._engine._tts is tts_sentinel
    assert bridge._llm is llm_sentinel


async def test_browser_factory_ignores_voice_caller_name_gender_lead_overrides_by_default() -> None:
    ws = SimpleNamespace(query_params={
        "voice": "some-other-voice", "caller_name": "Not The Agent",
        "gender": "female", "lead_name": "Attacker Name", "lead_gender": "male",
    })
    tenant = _tenant()
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=tenant)
    # tts_voice stayed the tenant's configured (None) voice -- never validated
    # against a roster or applied, because sel_voice never left "".
    assert bridge._agent._engine._config.tts.voice_id is None
    assert "lead_name" not in bridge._agent.session.lead_data
    assert "lead_gender" not in bridge._agent.session.lead_data
    # caller_name/gender feed VoiceBotScript replacements -- unreached, so the
    # script the agent was built with is the untouched closure default.
    assert bridge._agent._script is DEFAULT_DEMO_SCRIPT


async def test_browser_factory_ignores_campaign_override_by_default() -> None:
    from src.dialogue.campaign_loader import LoadedCampaign
    from src.dialogue.prompts import VoiceBotScript

    seen = {}

    class _Resolver:
        async def resolve(self, tenant_id, campaign_id=None):
            seen["campaign_id"] = campaign_id
            return LoadedCampaign(
                VoiceBotScript.from_campaign_yaml({"name": "FromDB", "company": "Acme"}),
                SlotSchema.from_campaign_yaml({"db_slot": {"type": "string"}}))

    ws = SimpleNamespace(query_params={"campaign": "camp_9"})
    factory = make_browser_bridge_factory(_providers(), campaign_resolver=_Resolver())
    await factory(websocket=ws, tenant=_tenant())  # allow_overrides defaults to False
    # The resolver still runs (a non-handoff call always resolves SOME
    # campaign) but never sees the query string's campaign id.
    assert seen["campaign_id"] is None


async def test_browser_factory_logs_ignored_override_param_names_not_values(caplog) -> None:
    """debug_event must record which override params were present, never what
    a caller tried to set them to."""
    import logging

    ws = SimpleNamespace(query_params={"tts": "elevenlabs", "voice": "some-voice-id"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    with caplog.at_level(logging.DEBUG, logger="src.api.dev_console"):
        await factory(websocket=ws, tenant=_tenant())
    matches = [r for r in caplog.records
               if r.getMessage() == "dev_console browser_bridge overrides_ignored"]
    assert len(matches) == 1
    record = matches[0]
    assert sorted(record.ignored_params) == ["tts", "voice"]
    assert record.reason == "allow_overrides_false"
    logged = repr(record.__dict__)
    assert "elevenlabs" not in logged
    assert "some-voice-id" not in logged


async def test_browser_factory_honours_stt_llm_tts_overrides_when_allowed() -> None:
    """The admin-gated /dev/voice posture (allow_overrides=True) is unchanged:
    overrides still build the requested provider, not the tenant's own."""
    from src.providers.tts.sarvam import SarvamTTSAdapter

    ws = SimpleNamespace(query_params={"tts": "sarvam"})
    factory = make_browser_bridge_factory(_providers(), slots=SlotSchema())
    bridge = await factory(websocket=ws, tenant=_tenant(), allow_overrides=True)
    assert isinstance(bridge._agent._engine._tts, SarvamTTSAdapter)
