"""Twilio softphone webhooks: dial TwiML logs a manual call; the recording
callback transcribes → analyzes → persists the SAME outcome shape as AI calls."""

from __future__ import annotations

import audioop
import io
import json
import wave

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.api import telephony_hooks
from src.api.deps import get_db_session
from src.auth import register_tenant_for_test
from src.auth.middleware import set_tenant_resolver
from src.campaign.models import LeadCallOutcome
from src.config_tenant import (
    TenantPipelineConfig,
    TenantSettings,
    TenantSTTConfig,
    TenantTelephonyConfig,
)
from src.interfaces.llm import LLMResult
from src.interfaces.stt import STTConfig, STTResult
from src.models.conversation import Conversation
from src.models.database import Base
from src.models.tenant import Tenant

HEADERS: dict[str, str] = {}


def _stereo_wav(rate: int = 8000) -> bytes:
    left = b"\x10\x00" * 80
    right = b"\x20\x00" * 80
    stereo = audioop.add(
        audioop.tostereo(left, 2, 1.0, 0.0),
        audioop.tostereo(right, 2, 0.0, 1.0), 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(stereo)
    return buf.getvalue()


class _FakeSTT:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        self.calls += 1
        return STTResult(text=f"channel {self.calls} speech", confidence=1.0)


class _FakeLLM:
    async def generate(self, messages, config) -> LLMResult:
        return LLMResult(
            text='{"outcome": "interested", "summary": "Lead is keen.",'
                 ' "notes": "Wants the link.", "callback_datetime": null,'
                 ' "callback_phrase": null}',
            finish_reason="stop")


class _FakeProviders:
    def __init__(self) -> None:
        self.stt = _FakeSTT()
        self.llm = _FakeLLM()

    def get_stt(self, tenant):  # noqa: ANN001
        return self.stt

    def get_llm(self, tenant):  # noqa: ANN001
        return self.llm


@pytest_asyncio.fixture
async def ctx(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add(Tenant(id="t1", slug="acme", name="Acme"))
        await s.commit()

    async def _session_override():
        async with maker() as session:
            yield session

    set_tenant_resolver(None)
    register_tenant_for_test(
        TenantSettings(
            id="t1", slug="acme", name="Acme",
            pipeline=TenantPipelineConfig(
                stt=TenantSTTConfig(provider="groq", language="hi-IN"),
                telephony=TenantTelephonyConfig(
                    provider="twilio", from_number="+15550001111"),
            ),
        ),
        plaintext_tokens=["tok"],
    )
    telephony_hooks.set_softphone_providers(_FakeProviders())
    monkeypatch.setattr(
        telephony_hooks, "_download_twilio_recording",
        lambda url, sid, tok: _async_wav())
    monkeypatch.setattr(
        telephony_hooks, "_download_stringee_recording",
        lambda url, tenant: _async_wav())

    app = FastAPI()
    app.include_router(telephony_hooks.router, prefix="/api/v1")
    app.dependency_overrides[get_db_session] = _session_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, maker
    telephony_hooks.set_softphone_providers(None)
    set_tenant_resolver(None)
    await engine.dispose()


async def _async_wav() -> bytes:
    return _stereo_wav()


async def test_softphone_twiml_logs_manual_call_and_dials(ctx) -> None:
    client, maker = ctx
    resp = await client.post(
        "/api/v1/telephony/twilio/softphone-twiml/acme",
        data={"To": "+918618795697", "From": "client:agent-3", "CallSid": "CA-1"})
    assert resp.status_code == 200
    xml = resp.text
    assert "record-from-answer-dual" in xml
    assert "+918618795697" in xml
    assert 'callerId="+15550001111"' in xml
    assert "softphone-recording/acme" in xml

    async with maker() as s:
        row = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "CA-1")
        )).scalar_one()
    assert row.agent_type == "human"
    assert row.channel == "softphone"
    assert row.mode == "layered"
    assert row.tts_provider is None
    assert row.telephony_provider == "twilio"


async def test_twilio_softphone_caller_id_from_provider_number(ctx) -> None:
    client, maker = ctx
    # Symmetric to the Stringee path: the Twilio caller-ID is the tenant's number
    # registered for provider "twilio" in tenant_phone_numbers, not from_number.
    from src.models.tenant import TenantPhoneNumber
    async with maker() as s:
        s.add(TenantPhoneNumber(
            phone_number="+15557654321", tenant_id="t1", provider="twilio"))
        await s.commit()

    resp = await client.post(
        "/api/v1/telephony/twilio/softphone-twiml/acme",
        data={"To": "+918618795697", "From": "client:a", "CallSid": "CA-PN"})
    assert resp.status_code == 200
    assert 'callerId="+15557654321"' in resp.text


async def test_recording_callback_finalizes_same_outcome(ctx) -> None:
    client, maker = ctx
    # First log the manual call (as the dial TwiML would).
    await client.post(
        "/api/v1/telephony/twilio/softphone-twiml/acme",
        data={"To": "+918618795697", "From": "client:a", "CallSid": "CA-2"})

    resp = await client.post(
        "/api/v1/telephony/twilio/softphone-recording/acme",
        data={"CallSid": "CA-2", "RecordingUrl": "https://api.twilio.com/REC/abc",
              "RecordingDuration": "37"})
    assert resp.status_code == 200

    async with maker() as s:
        row = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "CA-2")
        )).scalar_one()
    # Identical outcome shape to an AI call: enum value, summary, notes, status, duration.
    assert row.outcome == LeadCallOutcome.INTERESTED.value
    assert row.summary == "Lead is keen."
    assert "Wants the link." in row.notes
    assert "https://api.twilio.com/REC/abc" in row.notes
    assert row.status == "ended"
    assert row.duration_ms == 37_000


async def test_stringee_answer_logs_manual_call_and_connects(ctx) -> None:
    client, maker = ctx
    resp = await client.post(
        "/api/v1/telephony/stringee/softphone-answer/acme",
        json={"call_id": "ST-1", "from": "agent-7", "to": "+918618795697"})
    assert resp.status_code == 200
    scco = resp.json()
    assert scco[0]["action"] == "connect"
    assert scco[0]["to"]["number"] == "+918618795697"
    assert scco[0]["to"]["type"] == "external"
    assert scco[0]["from"]["number"] == "+15550001111"
    # Outbound caller-ID must be a Stringee-provisioned number → type "external";
    # "internal" is for app users and makes Stringee reject the connect SCCO.
    assert scco[0]["from"]["type"] == "external"
    assert scco[0]["record"] == {"format": "wav", "channel": "two"}
    assert "softphone-recording/acme" in scco[0]["eventUrl"]

    async with maker() as s:
        row = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "ST-1")
        )).scalar_one()
    assert row.agent_type == "human"
    assert row.channel == "softphone"
    assert row.tts_provider is None


async def test_stringee_answer_caller_id_from_provider_number(ctx) -> None:
    client, maker = ctx
    # The tenant owns a Stringee number registered in tenant_phone_numbers; the
    # connect SCCO `from` must be THAT number (Stringee rejects a `from` it
    # doesn't own), not the generic telephony from_number.
    from src.models.tenant import TenantPhoneNumber
    async with maker() as s:
        s.add(TenantPhoneNumber(
            phone_number="918204268005", tenant_id="t1", provider="stringee"))
        await s.commit()

    resp = await client.post(
        "/api/v1/telephony/stringee/softphone-answer/acme",
        json={"call_id": "ST-PN", "from": "agent", "to": "+918618795697"})
    assert resp.status_code == 200
    scco = resp.json()
    assert scco[0]["from"]["number"] == "918204268005"
    assert scco[0]["from"]["type"] == "external"


async def test_stringee_answer_reads_destination_from_custom(ctx) -> None:
    client, maker = ctx
    # Real Stringee app-to-phone behaviour (from the live debugger): the answer
    # URL is fetched via GET with `to` EMPTY (fromInternal=true); the dialed lead
    # number arrives in customData, delivered as the `custom` query param (a JSON
    # string). The handler must still resolve the destination + connect.
    resp = await client.get(
        "/api/v1/telephony/stringee/softphone-answer/acme",
        params={"from": "dev", "to": "", "fromInternal": "true", "userId": "dev",
                "callId": "ST-CUSTOM-1", "custom": json.dumps({"to": "+918618795697"})})
    assert resp.status_code == 200
    scco = resp.json()
    assert scco[0]["action"] == "connect"
    assert scco[0]["to"]["number"] == "+918618795697"
    assert scco[0]["from"]["number"] == "+15550001111"

    async with maker() as s:
        row = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "ST-CUSTOM-1")
        )).scalar_one()
    assert row.agent_type == "human"
    assert row.channel == "softphone"


async def test_stringee_recording_finalizes_same_outcome(ctx) -> None:
    client, maker = ctx
    await client.post(
        "/api/v1/telephony/stringee/softphone-answer/acme",
        json={"call_id": "ST-2", "from": "agent-1", "to": "+918618795697"})

    resp = await client.post(
        "/api/v1/telephony/stringee/softphone-recording/acme",
        json={"call_id": "ST-2", "recordUrl": "https://rec.stringee/abc.wav",
              "duration": "52"})
    assert resp.status_code == 200

    async with maker() as s:
        row = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "ST-2")
        )).scalar_one()
    assert row.outcome == LeadCallOutcome.INTERESTED.value
    assert row.summary == "Lead is keen."
    assert "https://rec.stringee/abc.wav" in row.notes
    assert row.status == "ended"
    assert row.duration_ms == 52_000


async def test_stringee_recording_ignores_non_recording_event(ctx) -> None:
    client, _ = ctx
    # A call-progress event with no recording URL must just be acked (200), no 500.
    resp = await client.post(
        "/api/v1/telephony/stringee/softphone-recording/acme",
        json={"call_id": "ST-3", "event": "RINGING"})
    assert resp.status_code == 200
