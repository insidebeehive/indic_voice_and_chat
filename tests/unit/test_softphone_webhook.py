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

    async def transcribe_audio(self, audio, mime_type="audio/mpeg") -> str:
        # Stringee softphone recordings are transcribed by the multimodal LLM.
        return "agent: namaste\nlead: haan bataiye"


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
    telephony_hooks.set_softphone_sessionmaker(maker)  # background manual-call logging
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
    telephony_hooks.set_softphone_sessionmaker(None)
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
    assert "api.twilio.com" not in (row.notes or "")   # recording URL not echoed back
    assert row.status == "ended"
    assert row.duration_ms == 37_000


async def test_stringee_answer_logs_manual_call_and_connects(ctx) -> None:
    client, maker = ctx
    resp = await client.post(
        "/api/v1/telephony/stringee/softphone-answer/acme",
        json={"call_id": "ST-1", "userId": "agent-7", "from": "agent-7",
              "to": "+918618795697"})
    assert resp.status_code == 200
    scco = resp.json()
    rec = next(a for a in scco if a["action"] == "record")
    conn = next(a for a in scco if a["action"] == "connect")
    # Matches the known-working Stringee SCCO: record action (mp3) BEFORE connect.
    assert scco.index(rec) < scco.index(conn)
    assert rec["format"] == "mp3"
    assert rec["recordStereo"] is False and rec["record_type"] == 1
    assert "softphone-recording/acme" in rec["eventUrl"]
    # connect: from = the caller-ID number (internal), bare (no "+"); to = lead.
    assert conn["from"] == {
        "number": "15550001111", "alias": "15550001111", "type": "internal"}
    assert conn["to"]["number"] == "+918618795697"
    assert conn["to"]["type"] == "external"
    assert conn["peerToPeerCall"] is False

    async with maker() as s:
        row = (await s.execute(
            select(Conversation).where(Conversation.provider_call_sid == "ST-1")
        )).scalar_one()
    assert row.agent_type == "human"
    assert row.channel == "softphone"
    assert row.tts_provider is None


async def test_stringee_answer_from_uses_request_caller_id(ctx) -> None:
    client, _ = ctx
    # The browser places the call FROM the caller-ID number, so it arrives as the
    # request's `from`; the SCCO connect `from` must echo that same number (keeps
    # makeCall `from` == SCCO `from`, as in Stringee's working call).
    resp = await client.post(
        "/api/v1/telephony/stringee/softphone-answer/acme",
        json={"call_id": "ST-CID", "from": "918204267969", "to": "918618795697"})
    assert resp.status_code == 200
    conn = next(a for a in resp.json() if a["action"] == "connect")
    assert conn["from"]["number"] == "918204267969"


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
    conn = next(a for a in scco if a["action"] == "connect")
    assert conn["to"]["number"] == "+918618795697"
    assert conn["from"]["number"] == "15550001111"  # caller-ID number, bare

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
    assert "rec.stringee" not in (row.notes or "")     # recording URL not echoed back
    assert row.status == "ended"
    assert row.duration_ms == 52_000


async def test_stringee_recording_ignores_non_recording_event(ctx) -> None:
    client, _ = ctx
    # A call-progress event with no recording URL must just be acked (200), no 500.
    resp = await client.post(
        "/api/v1/telephony/stringee/softphone-recording/acme",
        json={"call_id": "ST-3", "event": "RINGING"})
    assert resp.status_code == 200
