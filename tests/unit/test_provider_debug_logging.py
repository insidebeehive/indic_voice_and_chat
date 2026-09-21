"""DEBUG-boundary logging for src/providers/ (docs/debug-logging.md, pass 2).

Every test here does two things: asserts the DEBUG line carries the
discriminating fields an operator would need (request/response bodies,
model/voice used, counts), and asserts no credential (API key, bearer token,
basic-auth secret, subscription key) ever reaches a log record — the one
hard rule for this package, since every adapter here holds a provider key.

The credential-safety assertions were verified by mutation while writing this
pass: temporarily changing an adapter's debug_event call to also log
`headers=...` (or the raw auth tuple) makes the corresponding test below fail
immediately, because the secret value then appears in the captured record
text. For ElevenLabs that mutation is kept below as its own test (patching a
local copy of the call, never the shipped adapter) so the proof stays
checked in; for Azure, Twilio and Exotel it was applied to the adapter by
hand, confirmed to fail the test, and reverted -- a real credential leak must
never be checked in, even disabled, so those three aren't repeated as
permanent tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import respx
from httpx import Response

from src.interfaces.llm import LLMConfig, LLMMessage
from src.interfaces.stt import STTConfig
from src.interfaces.telephony import CallConfig
from src.interfaces.tts import TTSConfig
from src.providers.llm.gemini import GeminiLLMAdapter
from src.providers.stt.deepgram import DeepgramSTTAdapter, DeepgramStreamSession
from src.providers.telephony.exotel import EXOTEL_BASE_URL, ExotelAdapter
from src.providers.telephony.softphone import mint_browser_credentials
from src.providers.telephony.twilio import TwilioAdapter
from src.providers.tts.azure import AzureTTSAdapter
from src.providers.tts.elevenlabs import ELEVENLABS_BASE_URL, ElevenLabsTTSAdapter
from src.providers.vector_store.faiss_store import FAISSAdapter


def _all_logged_text(caplog) -> str:
    """Every field of every captured record, flattened to one grep-able
    string -- catches a leak in an `extra=` field, not just the message."""
    chunks = []
    for r in caplog.records:
        chunks.append(r.getMessage())
        for k, v in vars(r).items():
            chunks.append(f"{k}={v!r}")
    return "\n".join(chunks)


# --- LLM: Gemini (llm/gemini.py) -------------------------------------------


def _gemini_response(text: str) -> SimpleNamespace:
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
        finish_reason=SimpleNamespace(name="STOP"),
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
    )


@pytest.mark.asyncio
async def test_gemini_generate_logs_full_request_and_response(caplog) -> None:
    """The boundary this pass exists for: a Gemini prompt and its reply,
    logged whole, with the fields that let an operator confirm what was
    actually sent and what came back -- not just that a call happened."""
    fake_generate = AsyncMock(return_value=_gemini_response("hello there"))
    client = SimpleNamespace(aio=SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_generate),
    ))
    adapter = GeminiLLMAdapter({"client": client, "model": "gemini-3.5-flash"})
    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="What is the capital of France?"),
    ]
    with caplog.at_level("DEBUG", logger="src.providers.llm.gemini"):
        result = await adapter.generate(messages, LLMConfig(response_format="text"))

    assert result.text == "hello there"
    events = {r.event for r in caplog.records if hasattr(r, "event")}
    assert "gemini generate request" in events
    assert "gemini generate response" in events

    req = next(r for r in caplog.records if getattr(r, "event", None) == "gemini generate request")
    assert req.model == "gemini-3.5-flash"
    assert req.system == "You are a helpful assistant."
    assert any(
        part.get("text") == "What is the capital of France?"
        for c in req.contents for part in c["parts"]
    )

    resp = next(r for r in caplog.records if getattr(r, "event", None) == "gemini generate response")
    assert resp.text == "hello there"
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0}


@pytest.mark.asyncio
async def test_gemini_generate_stream_debug_off_costs_no_accumulation(caplog) -> None:
    """`generate_stream` only builds its response-accumulator when DEBUG is
    enabled (see the comment in gemini.py) -- confirm the response event is
    genuinely absent at INFO, and that the stream still yields the right
    text either way (no control-flow change from the instrumentation)."""
    async def _stream(**kwargs):
        async def _gen():
            yield SimpleNamespace(text="foo", candidates=[])
            yield SimpleNamespace(text="bar", candidates=[])
        return _gen()

    client = SimpleNamespace(aio=SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=AsyncMock(side_effect=_stream)),
    ))
    adapter = GeminiLLMAdapter({"client": client})
    messages = [LLMMessage(role="user", content="hi")]

    with caplog.at_level("INFO", logger="src.providers.llm.gemini"):
        chunks = [c async for c in adapter.generate_stream(messages, LLMConfig())]
    assert chunks == ["foo", "bar"]
    assert not any(getattr(r, "event", "") == "gemini generate_stream response"
                    for r in caplog.records)

    caplog.clear()
    with caplog.at_level("DEBUG", logger="src.providers.llm.gemini"):
        chunks = [c async for c in adapter.generate_stream(messages, LLMConfig())]
    assert chunks == ["foo", "bar"]
    resp = next(r for r in caplog.records
                if getattr(r, "event", None) == "gemini generate_stream response")
    assert resp.text == "foobar"
    assert resp.chunks == 2


# --- TTS: ElevenLabs (tts/elevenlabs.py) -----------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_tts_logs_body_and_never_leaks_api_key(caplog) -> None:
    adapter = ElevenLabsTTSAdapter({"api_key": "el-super-secret-key"})
    voice_id = adapter._default_voice_id
    respx.post(f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}").mock(
        return_value=Response(200, content=b"\x00\x01\x02\x03" * 100)
    )
    with caplog.at_level("DEBUG", logger="src.providers.tts.elevenlabs"):
        result = await adapter.synthesize("namaste, kaise hain aap?", TTSConfig())

    assert result.audio
    logged = _all_logged_text(caplog)
    assert "namaste" in logged            # the customer text: DEBUG logs it whole
    assert "el-super-secret-key" not in logged

    req = next(r for r in caplog.records if getattr(r, "event", None) == "elevenlabs tts request")
    assert req.body["text"] == "namaste, kaise hain aap?"
    resp = next(r for r in caplog.records if getattr(r, "event", None) == "elevenlabs tts response")
    assert resp.audio_bytes == 400


@pytest.mark.asyncio
@respx.mock
async def test_elevenlabs_tts_mutation_would_have_caught_a_header_leak(caplog, monkeypatch) -> None:
    """Demonstrates the mutation described in the module docstring inline
    (rather than only by hand): patch the adapter to log its real request
    headers -- including the API key -- and show the credential-safety
    assertion above would fail against that code."""
    import src.providers.tts.elevenlabs as el_module

    adapter = ElevenLabsTTSAdapter({"api_key": "el-super-secret-key"})
    voice_id = adapter._default_voice_id
    respx.post(f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}").mock(
        return_value=Response(200, content=b"\x00\x01")
    )

    real_debug_event = el_module.debug_event

    def _leaky_debug_event(logger, event, **values):
        if event == "elevenlabs tts request":
            values = dict(values, headers=adapter._headers())  # the bug under test
        return real_debug_event(logger, event, **values)

    monkeypatch.setattr(el_module, "debug_event", _leaky_debug_event)

    with caplog.at_level("DEBUG", logger="src.providers.tts.elevenlabs"):
        await adapter.synthesize("hi", TTSConfig())

    logged = _all_logged_text(caplog)
    assert "el-super-secret-key" in logged  # proves the mutation actually leaks
    # i.e. the real (unpatched) adapter tested above is what keeps this from
    # ever being true in production.


# --- TTS: Azure (tts/azure.py) ---------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_azure_tts_logs_ssml_and_never_leaks_subscription_key(caplog) -> None:
    adapter = AzureTTSAdapter({"api_key": "azure-sub-key-xyz", "region": "eastus"})
    respx.post("https://eastus.tts.speech.microsoft.com/cognitiveservices/v1").mock(
        return_value=Response(200, content=b"\x00\x00" * 800)
    )
    with caplog.at_level("DEBUG", logger="src.providers.tts.azure"):
        result = await adapter.synthesize("aapka din shubh ho", TTSConfig(language="hi-IN"))

    assert result.audio
    logged = _all_logged_text(caplog)
    assert "aapka din shubh ho" in logged
    assert "azure-sub-key-xyz" not in logged

    req = next(r for r in caplog.records if getattr(r, "event", None) == "azure tts request")
    assert "aapka din shubh ho" in req.ssml


# --- STT: Deepgram (stt/deepgram.py) ---------------------------------------


@pytest.mark.asyncio
async def test_deepgram_open_stream_logs_url_and_header_keys_never_token(caplog) -> None:
    fake_ws = SimpleNamespace()

    async def _connector(url, headers):
        return fake_ws

    adapter = DeepgramSTTAdapter({"api_key": "dg-token-abc123", "connector": _connector})
    with caplog.at_level("DEBUG", logger="src.providers.stt.deepgram"):
        session = await adapter.open_stream(STTConfig(sample_rate=16000, language="hi"))
    session._tasks = []  # never started (start_tasks path not under test); nothing to cancel

    logged = _all_logged_text(caplog)
    assert "dg-token-abc123" not in logged

    req = next(r for r in caplog.records if getattr(r, "event", None) == "deepgram stream opening")
    assert req.header_keys == ["Authorization"]   # names only
    assert "api.deepgram.com" in req.url


def test_deepgram_unparseable_frame_is_logged_not_silently_dropped(caplog) -> None:
    """The skip category: a malformed frame used to vanish with zero trace."""
    session = DeepgramStreamSession(ws=SimpleNamespace(), start_tasks=False)
    with caplog.at_level("DEBUG", logger="src.providers.stt.deepgram"):
        ev = session._handle_raw("{not valid json")
    assert ev is None
    rec = next(r for r in caplog.records if getattr(r, "event", None) == "deepgram frame unparseable")
    assert rec.raw == "{not valid json"


# --- Telephony: Twilio (telephony/twilio.py) -------------------------------


@pytest.mark.asyncio
async def test_twilio_initiate_call_logs_request_response_never_leaks_auth_token(caplog) -> None:
    fake_call = SimpleNamespace(sid="CA123", status="queued")
    fake_client = Mock()
    fake_client.calls.create = Mock(return_value=fake_call)
    adapter = TwilioAdapter({
        "account_sid": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "auth_token": "twilio-auth-token-secret",
        "client": fake_client,
    })
    with caplog.at_level("DEBUG", logger="src.providers.telephony.twilio"):
        session = await adapter.initiate_call(CallConfig(
            to_number="+919999999999", from_number="+911111111111",
            webhook_url="https://example.com/hook", timeout_seconds=30,
        ))

    assert session.session_id == "CA123"
    logged = _all_logged_text(caplog)
    assert "twilio-auth-token-secret" not in logged
    req = next(r for r in caplog.records
               if getattr(r, "event", None) == "twilio initiate_call request")
    assert req.to == "+919999999999"
    resp = next(r for r in caplog.records
                if getattr(r, "event", None) == "twilio initiate_call response")
    assert resp.session_id == "CA123"


# --- Telephony: Exotel (telephony/exotel.py) -------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_exotel_initiate_call_logs_status_body_never_leaks_basic_auth(caplog) -> None:
    adapter = ExotelAdapter({
        "api_key": "exotel-key", "api_token": "exotel-token-secret",
        "account_sid": "acc123",
    })
    respx.post(f"{EXOTEL_BASE_URL}/v1/Accounts/acc123/Calls/connect").mock(
        return_value=Response(200, json={"Call": {"Sid": "CX1", "Status": "queued"}})
    )
    with caplog.at_level("DEBUG", logger="src.providers.telephony.exotel"):
        session = await adapter.initiate_call(CallConfig(
            to_number="+919999999999", from_number="+911111111111",
            webhook_url="https://example.com/hook",
        ))

    assert session.session_id == "CX1"
    logged = _all_logged_text(caplog)
    assert "exotel-token-secret" not in logged
    assert "exotel-key" not in logged

    resp = next(r for r in caplog.records
                if getattr(r, "event", None) == "exotel initiate_call response")
    assert resp.status == 200
    assert "CX1" in resp.body


# --- Vector store: FAISS (vector_store/faiss_store.py) ---------------------


@pytest.mark.asyncio
async def test_faiss_search_logs_discriminating_fields(tmp_path, caplog) -> None:
    from src.interfaces.vector_store import Document

    adapter = FAISSAdapter({"embedding_dim": 3, "index_path": str(tmp_path / "idx")})
    await adapter.index([
        Document(id="d1", content="hello", metadata={}, embedding=[1.0, 0.0, 0.0]),
        Document(id="d2", content="world", metadata={}, embedding=[0.0, 1.0, 0.0]),
    ])
    with caplog.at_level("DEBUG", logger="src.providers.vector_store.faiss_store"):
        results = await adapter.search([1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    rec = next(r for r in caplog.records if getattr(r, "event", None) == "faiss search")
    assert rec.top_k == 1
    assert rec.returned == 1
    assert "d1" in rec.result_ids


# --- Softphone credential minting (telephony/softphone.py) -----------------


def test_softphone_mint_never_logs_the_minted_token(caplog) -> None:
    from src.auth.context import TenantContext
    from src.config_tenant import TenantPipelineConfig, TenantSettings, TenantTelephonyConfig

    tenant = TenantContext(
        settings=TenantSettings(
            id="t1", slug="acme", name="Acme",
            pipeline=TenantPipelineConfig(telephony=TenantTelephonyConfig(
                provider="stringee", account_sid_env="S_SID", auth_token_env="S_SECRET")),
        ),
        secrets_resolved={"S_SID": "stringee-key-id", "S_SECRET": "stringee-shared-secret"},
    )
    with caplog.at_level("DEBUG", logger="src.providers.telephony.softphone"):
        creds = mint_browser_credentials(tenant, "agent-7", ttl_seconds=600)

    logged = "\n".join(f"{k}={v!r}" for r in caplog.records for k, v in vars(r).items())
    assert creds.token not in logged
    assert "stringee-shared-secret" not in logged
    rec = next(r for r in caplog.records if getattr(r, "event", None) == "softphone minted")
    assert rec.provider == "stringee"
    assert rec.identity == "agent-7"
