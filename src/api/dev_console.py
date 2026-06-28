# src/api/dev_console.py
"""Dev-only browser voice console (gated by VOX_DEV_CONSOLE=1).

Serves a self-contained page at ``GET /dev/voice`` and runs a
``BrowserVoiceBridge`` at ``WS /api/v1/dev/voice``. Reuses the tenant's
provider stack exactly like the telephony bridges; intended for local
dialogue-management iteration with no telephony cost.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import asyncio

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api.answer_paths import ANSWER_PATHS
from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.api.gemini_live_bridge import RECORD_TURN_SIGNAL, GeminiLiveBridge
from src.auth.context import TenantContext
from src.auth.registry import TenantProviders
from src.api.call_store import insert_call
from src.bootstrap import DEFAULT_DEMO_SCRIPT
from src.config_tenant import platform_webhook_base_url
from src.models.database import get_sessionmaker
from src.dialogue.prompts import VoiceBotScript, build_s2s_system_instruction
from src.dialogue.slots import SlotSchema
from src.interfaces.realtime import RealtimeConfig
from src.providers.realtime.gemini_live import GeminiLiveSession
from src.interfaces.llm import LLMConfig
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.pipeline.vad import EnergyVAD, SileroVAD
from src.providers import get_streaming_stt_provider, get_telephony_provider

log = logging.getLogger(__name__)

_STATIC = Path(__file__).resolve().parents[2] / "static"

# WS path lives under the /api/v1 router; the page route is top-level.
ws_router = APIRouter(prefix="/dev", tags=["dev-console"])   # mounted under /api/v1
dev_router = APIRouter(tags=["dev-console"])                  # mounted at app root

# Factory: (websocket, tenant) -> BrowserVoiceBridge. Set during lifespan.
BrowserBridgeFactory = Callable[[WebSocket, TenantContext], BrowserVoiceBridge]
_browser_bridge_factory: Optional[BrowserBridgeFactory] = None


def dev_console_enabled() -> bool:
    return os.environ.get("VOX_DEV_CONSOLE", "") == "1"


def set_browser_bridge_factory(factory: Optional[BrowserBridgeFactory]) -> None:
    global _browser_bridge_factory
    _browser_bridge_factory = factory


def get_browser_bridge_factory() -> Optional["BrowserBridgeFactory"]:
    """The browser-voice bridge factory (used by the always-on chat→voice
    handoff route, which lives outside the VOX_DEV_CONSOLE gate)."""
    return _browser_bridge_factory


async def run_browser_voice(websocket: WebSocket, tenant: TenantContext) -> None:
    """Build + run a browser voice session for an already-accepted websocket.
    Shared by the dev console (/dev/voice) and the chat→voice handoff
    (/chat/voice)."""
    if _browser_bridge_factory is None:
        await websocket.close(code=1011, reason="browser bridge factory unset")
        return
    try:
        bridge = await _browser_bridge_factory(websocket, tenant)
    except Exception as e:  # noqa: BLE001 - e.g. tenant has no campaign configured
        log.warning("browser voice bridge build failed: %s", e)
        await websocket.close(code=1011, reason="no campaign configured for tenant")
        return
    try:
        await _run_billed_session(tenant, bridge, mode="layered")
    except WebSocketDisconnect:
        log.info("browser voice client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("browser voice bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


# Factory: (websocket, tenant) -> GeminiLiveBridge (the S2S path). Set during lifespan.
LiveBridgeFactory = Callable[[WebSocket, TenantContext], GeminiLiveBridge]
_live_bridge_factory: Optional[LiveBridgeFactory] = None


def set_live_bridge_factory(factory: Optional[LiveBridgeFactory]) -> None:
    global _live_bridge_factory
    _live_bridge_factory = factory


@dev_router.get("/dev/voice")
async def dev_voice_page() -> FileResponse:
    return FileResponse(_STATIC / "dev_console.html", media_type="text/html")


@dev_router.get("/dev/voices")
async def dev_voices(request: Request, tenant: str = "dev") -> dict:
    """Voice options for the console's Voice dropdown, per mode — config-driven so
    the list matches the active stack instead of a hardcoded one:

    - ``layered`` (cascade): the configured TTS provider's voice roster.
    - ``s2s`` (Gemini Live): full catalog of 30 voices with metadata (gender, style).
    """
    from src.auth.middleware import tenant_from_slug
    from src.providers.voice_catalog import list_voices

    try:
        tctx = await tenant_from_slug(tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")
    p = tctx.settings.pipeline

    # S2S: always show the full 30-voice catalog so devs can try any voice.
    # The configured voice (or first catalog voice) is the default selection.
    rt = p.realtime
    all_s2s = list_voices("gemini_live")
    s2s_voices = [v["voice_id"] for v in all_s2s]
    s2s_default = (rt.voice if rt else "") or (s2s_voices[0] if s2s_voices else "")

    # Layered: the configured TTS provider's roster (default = pipeline.tts.voice_id).
    layered_default = p.tts.voice_id or ""
    layered_voices: list[str] = []
    providers = getattr(request.app.state, "providers", None)
    if providers is not None:
        try:
            tts = providers.get_tts(tctx)
            roster = tts.get_available_voices(p.tts.language or "hi-IN")
            layered_voices = [v.get("voice_id") for v in roster if v.get("voice_id")]
        except Exception:  # noqa: BLE001 - never block the page on a provider hiccup
            log.warning("dev voices: could not list TTS voices", exc_info=True)
    if layered_default and layered_default not in layered_voices:
        layered_voices = [layered_default, *layered_voices]
    if not layered_default and layered_voices:
        layered_default = layered_voices[0]

    return {
        "layered": {"voices": layered_voices, "default": layered_default},
        "s2s": {"voices": s2s_voices, "default": s2s_default, "catalog": all_s2s},
    }


_TTS_META: dict[str, dict] = {
    "sarvam": {
        "label": "Sarvam AI",
        "languages": ["hi-IN", "en-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN",
                      "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"],
    },
    "gemini": {
        "label": "Gemini TTS",
        "languages": ["hi-IN", "en-IN", "bn-IN", "gu-IN", "ta-IN", "te-IN",
                      "mr-IN", "kn-IN", "ml-IN", "pa-IN", "en-US", "40+ more"],
    },
    "google": {
        "label": "Google Neural2 (Cloud TTS API required)",
        "languages": ["hi-IN", "en-IN"],
    },
    "azure": {
        "label": "Azure Neural",
        "languages": ["hi-IN", "en-IN", "mr-IN", "ta-IN", "te-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN"],
    },
}


@dev_router.get("/dev/providers")
async def dev_providers() -> dict:
    """STT/LLM/TTS provider lists for the layered-mode selectors."""
    from src.providers import LLM_PROVIDERS, STREAMING_STT_PROVIDERS, STT_PROVIDERS, TTS_PROVIDERS

    _labels: dict[str, dict[str, str]] = {
        "stt": {"sarvam": "Sarvam AI", "groq": "Groq Whisper", "gemini": "Gemini Flash"},
        "stt_streaming": {"deepgram": "Deepgram (streaming)"},
        "llm": {"gemini": "Gemini Flash", "groq": "Groq Llama-3", "anthropic": "Claude", "claude": "Claude"},
    }
    seen_cls: set = set()
    llm_opts = []
    for k in sorted(LLM_PROVIDERS):
        cls = LLM_PROVIDERS[k]
        if cls in seen_cls:
            continue
        seen_cls.add(cls)
        llm_opts.append({"id": k, "label": _labels["llm"].get(k, k)})

    def _opts(registry, label_map):
        return [{"id": k, "label": label_map.get(k, k)} for k in sorted(registry)]

    tts_opts = []
    for k in sorted(TTS_PROVIDERS):
        meta = _TTS_META.get(k, {})
        tts_opts.append({
            "id": k,
            "label": meta.get("label", k),
            "languages": meta.get("languages", []),
        })

    return {
        "stt": _opts(STT_PROVIDERS, _labels["stt"]) + _opts(STREAMING_STT_PROVIDERS, _labels["stt_streaming"]),
        "llm": llm_opts,
        "tts": tts_opts,
    }


@dev_router.get("/dev/tts-voices")
async def dev_tts_voices(provider: str = "sarvam", language: str = "hi-IN") -> dict:
    """Voice roster for a specific TTS provider (for the cascading voice dropdown)."""
    from src.providers import TTS_PROVIDERS

    if provider not in TTS_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"unknown TTS provider '{provider}'")
    adapter = TTS_PROVIDERS[provider]({})          # constructors no longer raise on missing key
    voices = adapter.get_available_voices(language)
    return {"provider": provider, "language": language, "voices": voices}


@dev_router.get("/dev/campaigns")
async def dev_campaigns(tenant: str = "dev") -> dict:
    """The tenant's campaigns, for the console's campaign selector. The selected
    id is passed to the voice WS as ``?campaign=<id>`` and drives the agent's
    script + slots for that call (per-tenant, no global fallback)."""
    from sqlalchemy import select

    from src.auth.middleware import tenant_from_slug
    from src.models.campaign import Campaign
    from src.models.database import get_sessionmaker

    try:
        tctx = await tenant_from_slug(tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")
    async with get_sessionmaker()() as s:
        rows = (await s.execute(
            select(Campaign).where(Campaign.tenant_id == tctx.id)
            .order_by(Campaign.status.desc(), Campaign.created_at.desc())
        )).scalars().all()
    return {"campaigns": [{"id": c.id, "name": c.name, "status": c.status} for c in rows]}


# --- Telephony control panel: place an outbound call + poll its status --------
#
# WebConsole runs in-browser (the WS routes above). The Telephony dropdown picks
# the provider; Twilio/Exotel place a real outbound call that runs the agent over
# the phone, and the console polls the in-memory call monitor the bridge writes
# to (keyed by the provider Call SID) for lifecycle (calling -> answered ->
# ended) + outcome. Mode/Voice are threaded via a one-shot override the bridge
# factory consumes. The selected provider's adapter is built on demand (creds
# resolve from the provider's env vars); the caller-ID comes from
# pipeline.telephony.outbound_from[provider]. Needs a publicly reachable host.

# Providers the dev console can place an outbound call with. The answer-webhook
# path per provider is the shared ANSWER_PATHS map. Twilio/Exotel run the
# media-stream bridge (S2S or cascade, per the Mode override); Stringee is a
# turn-based IVR with its own /stringee/answer.
_ANSWER_PATH = ANSWER_PATHS
_PLACE_CALL_PROVIDERS = tuple(_ANSWER_PATH)
# Stringee is IVR-only — Mode/Voice (S2S vs cascade) don't apply to it.
_STREAM_PROVIDERS = ("twilio", "exotel")


class PlaceCallRequest(BaseModel):
    provider: str               # "twilio" | "exotel"
    to_number: str
    mode: str = "s2s"           # "s2s" | "layered" — drives the placed call
    voice: str = ""             # S2S voice; "" -> tenant default
    lead_name: str = ""
    tenant: str = "dev"


@dev_router.post("/dev/place-call")
async def dev_place_call(req: PlaceCallRequest) -> dict:
    from src.auth.middleware import tenant_from_slug
    from src.interfaces.telephony import CallConfig

    from src.api import dev_call_control

    try:
        tenant = await tenant_from_slug(req.tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")

    tel = tenant.settings.pipeline.telephony
    provider = req.provider.strip().lower()
    if provider not in _PLACE_CALL_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider '{provider}' can't be placed from here; use {list(_PLACE_CALL_PROVIDERS)}")
    webhook_base = platform_webhook_base_url()   # platform-level; not per-tenant
    if not webhook_base:
        raise HTTPException(
            status_code=400, detail="platform WEBHOOK_BASE_URL is not set")

    # The dropdown drives the provider — build *its* adapter (creds resolve from the
    # provider's env vars) and dial from *its* configured caller-ID, independent of
    # the tenant's default/inbound provider.
    from_number = (tel.outbound_from or {}).get(provider)
    if not from_number and (tel.provider or "").lower() == provider:
        from_number = tel.from_number          # default block's number, if it matches
    if not from_number:
        # DB config may be from an old seed that lacked outbound_from / provider.
        # Fall back to the YAML (always present in the Docker image) as source of truth.
        try:
            from src.config_tenant import load_tenant as _load_tenant
            _yaml = _load_tenant(req.tenant)
            _yaml_tel = _yaml.pipeline.telephony
            from_number = (_yaml_tel.outbound_from or {}).get(provider)
            if not from_number and (_yaml_tel.provider or "").lower() == provider:
                from_number = _yaml_tel.from_number
        except Exception:
            pass
    if not from_number:
        raise HTTPException(
            status_code=400,
            detail=(f"no caller-ID configured for '{provider}'. Set "
                    f"pipeline.telephony.outbound_from.{provider} in config/tenants/{req.tenant}.yaml."))

    # Build the adapter with the SELECTED provider's per-tenant creds (the dropdown
    # provider may differ from the tenant's default) — no platform-env fallback.
    # Stringee's server adapter reads api_key_sid/api_key_secret (stored as the
    # account_sid/auth_token), the others account_sid/auth_token.
    def _cred(name):
        try:
            return tenant.secret(name) if name else None
        except Exception:  # noqa: BLE001 - missing env → let the adapter decide
            return None

    pcreds = tel.creds_for(provider)
    acct, auth = _cred(pcreds.account_sid_env), _cred(pcreds.auth_token_env)
    # Stringee: the callout needs a non-null userId, or it goes out as a
    # phone->phone external call and the Answer URL/SCCO never runs (silent bot).
    uid = _cred(pcreds.user_id_env)
    try:
        adapter = get_telephony_provider({
            "provider": provider,
            "account_sid": acct, "auth_token": auth,
            "api_key_sid": acct, "api_key_secret": auth,
            "user_id": uid,
            "base_url": tel.stringee_base_url,   # regional Stringee REST host, if set
        })
    except Exception as e:  # noqa: BLE001 - e.g. missing per-tenant credentials
        raise HTTPException(status_code=400, detail=f"telephony adapter for '{provider}' unavailable: {e}")

    # Thread the console's Mode/Voice/lead to the media-stream bridge factory.
    # Stringee ignores it (turn-based IVR), so don't leave a stale override.
    if provider in _STREAM_PROVIDERS:
        dev_call_control.set_override(
            tenant.slug, mode=req.mode, voice=req.voice.strip(), lead_name=req.lead_name.strip())
    # Scope the answer URL to the placing tenant's slug for ALL providers: this is
    # an outbound call WE place, so the tenant is known — the answer webhook resolves
    # by slug and the bridge is built with THIS tenant's config, instead of reverse-
    # resolving our own caller-ID (which would require the number to be registered in
    # tenant_phone_numbers). Every place-call provider has a slug-scoped answer route.
    answer_path = f"{_ANSWER_PATH[provider]}/{tenant.slug}"
    cfg = CallConfig(
        to_number=req.to_number.strip(),
        from_number=from_number,
        webhook_url=f"{webhook_base.rstrip('/')}/{answer_path}",
    )
    try:
        # Hard 20-second cap so Northflank's 30-second proxy timeout is never
        # reached — ensures a JSON error always makes it back to the browser.
        session = await asyncio.wait_for(adapter.initiate_call(cfg), timeout=20.0)
    except asyncio.TimeoutError:
        if provider in _STREAM_PROVIDERS:
            dev_call_control.pop_override(tenant.slug)
        log.error("dev place-call timed out", extra={"tenant": tenant.slug, "provider": provider})
        raise HTTPException(status_code=502, detail=f"call timed out after 20s — check {provider} API reachability from Northflank")
    except Exception as e:  # noqa: BLE001 - don't leave a stale override on failure
        if provider in _STREAM_PROVIDERS:
            dev_call_control.pop_override(tenant.slug)
        log.exception("dev place-call failed", extra={"tenant": tenant.slug, "provider": provider})
        raise HTTPException(status_code=502, detail=f"call failed: {e}")

    dev_call_control.monitor.set_status(session.session_id, "calling")
    # Register the conversation for the tenant that PLACED the call (not derived
    # from the number) — a real outbound voicebot/voice call, mirroring the
    # campaign path so the outcome persists + it shows in analytics. Best-effort:
    # a DB hiccup must not fail the (already-placed) call.
    call_id = f"call_{uuid.uuid4().hex[:16]}"
    try:
        sm = get_sessionmaker()
        async with sm() as db:
            await insert_call(
                db, call_id=call_id, tenant=tenant,
                provider_call_sid=session.session_id, channel="voice",
                mode=req.mode, voice=req.voice.strip() or None)
    except Exception:  # noqa: BLE001 — the call is already placed; recording is best-effort
        log.exception("dev place-call: failed to record conversation",
                      extra={"tenant": tenant.slug, "sid": session.session_id})
    log.info("dev console placed call", extra={
        "tenant": tenant.slug, "provider": provider, "call_sid": session.session_id})
    # Same shape as the campaign path's CallLeadResponse.
    return {"call_id": call_id, "status": "in_progress",
            "provider_call_sid": session.session_id}


@dev_router.post("/dev/reanalyze")
async def dev_reanalyze(call_id: str, request: Request, tenant: str = "dev") -> dict:
    """Re-run outcome analysis for a finished webconsole call.

    Used by the dev console Reanalyze button — no Bearer auth required (gated
    by VOX_DEV_CONSOLE). Reads stored turns from the DB, re-runs analyze_call,
    updates the conversation row, and returns the new outcome + summary.
    """
    from src.auth.middleware import tenant_from_slug
    from src.analysis.call_outcome import analyze_call
    from src.interfaces.llm import LLMMessage
    from src.models.conversation import Conversation, Turn
    from src.models.database import get_sessionmaker

    try:
        tctx = await tenant_from_slug(tenant)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=f"unknown tenant: {e}")

    providers = getattr(request.app.state, "providers", None)
    if providers is None:
        raise HTTPException(status_code=503, detail="provider registry not available")

    from sqlalchemy import select
    sm = get_sessionmaker()
    async with sm() as db:
        row = (await db.execute(
            select(Conversation).where(
                Conversation.id == call_id,
                Conversation.tenant_id == tctx.id,
            )
        )).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="conversation not found")

        turn_rows = (await db.execute(
            select(Turn).where(Turn.conversation_id == call_id).order_by(Turn.turn_number)
        )).scalars().all()
        if not turn_rows:
            raise HTTPException(status_code=422, detail="no transcript stored — cannot reanalyze")

        transcript = [LLMMessage(role=t.role, content=t.content) for t in turn_rows]
        slots = row.slots_data or {}
        from datetime import UTC, datetime as _dt
        try:
            llm = providers.get_llm(tctx)
            analysis = await analyze_call(
                transcript=transcript, slots=slots, telephony_status=None,
                final_action=None,
                tenant_timezone=getattr(tctx.settings, "timezone", "Asia/Kolkata"),
                now=_dt.now(UTC), llm=llm,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("dev reanalyze failed for %s", call_id)
            raise HTTPException(status_code=502, detail=f"analysis failed: {e}")

        row.outcome = analysis.outcome.value
        row.summary = analysis.summary
        row.notes = analysis.notes
        if analysis.callback_datetime:
            row.callback_at = analysis.callback_datetime.replace(tzinfo=None)
        await db.commit()

    log.info("dev reanalyzed call", extra={"call_id": call_id, "outcome": row.outcome})
    cb = analysis.callback_datetime
    return {
        "type": "outcome",
        "outcome": row.outcome,
        "summary": row.summary,
        "notes": row.notes,
        "source": analysis.analysis_source,
        "callback_datetime": cb.isoformat() if cb else None,
        "callback_phrase": analysis.callback_phrase,
    }


@dev_router.get("/dev/call-status/{call_sid}")
async def dev_call_status(call_sid: str) -> dict:
    from src.api import dev_call_control

    item = dev_call_control.monitor.get(call_sid)
    if item is None:
        return {"status": "unknown", "outcome": None}
    return item


def _parse_iso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


async def _run_billed_session(tenant, bridge, *, mode: str) -> None:
    """Run a browser-console bridge as a recorded + billed conversation.

    Inserts an in_progress conversation row (channel='webconsole') keyed by a
    fresh call_id, runs the bridge, then finalizes the row (status/outcome +
    derived duration + platform cost). ``mode`` is the actual path used
    ('s2s' for the live console, 'layered' for the cascade console) so the cost
    is billed correctly even when it differs from the tenant default. Telephony
    is excluded from the cost (the browser path uses no telephony). Failures here
    never break the call.
    """
    from src.api.call_store import insert_call, record_outcome, save_turns
    from src.models.database import get_sessionmaker

    sm = get_sessionmaker()
    call_id = f"call_{uuid.uuid4().hex[:16]}"
    try:
        async with sm() as s:
            await insert_call(s, call_id=call_id, tenant=tenant,
                              provider_call_sid=call_id, channel="webconsole", mode=mode)
    except Exception:  # noqa: BLE001
        log.exception("webconsole: failed to start call record")
    # Push call_id to the browser before the session starts so the dev console
    # can offer a Reanalyze button once the call ends.
    try:
        await bridge._send_json({"type": "session", "call_id": call_id})
    except Exception:  # noqa: BLE001
        pass
    try:
        await bridge.run()
    finally:
        payload = getattr(bridge, "_outcome_payload", None) or {}
        try:
            async with sm() as s:
                await record_outcome(
                    s, call_id, status="ended",
                    outcome=payload.get("outcome"), summary=payload.get("summary"),
                    notes=payload.get("notes"),
                    callback_at=_parse_iso(payload.get("callback_datetime")))
        except Exception:  # noqa: BLE001
            log.exception("webconsole: failed to finalize call record")
        # Persist the in-memory transcript so it can be re-analyzed later.
        try:
            agent = getattr(bridge, "_agent", None)
            turns = list(getattr(getattr(agent, "session", None), "turns", []))
            if turns:
                async with sm() as s:
                    await save_turns(s, conversation_id=call_id, turns=turns)
        except Exception:  # noqa: BLE001
            log.exception("webconsole: failed to save turns")


@ws_router.websocket("/voice")
async def dev_voice_ws(websocket: WebSocket) -> None:
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(
            websocket.query_params.get("tenant", "dev")
        )
    except Exception as e:  # noqa: BLE001
        log.warning("dev console tenant resolution failed: %s", e)
        await websocket.close(code=1008, reason="unknown tenant")
        return
    await run_browser_voice(websocket, tenant)


@ws_router.websocket("/voice-live")
async def dev_voice_live_ws(websocket: WebSocket) -> None:
    """Speech-to-speech (Gemini Live) path. Same client; different bridge."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    if _live_bridge_factory is None:
        await websocket.close(code=1011, reason="live bridge factory unset")
        return
    try:
        tenant = await tenant_from_slug(websocket.query_params.get("tenant", "dev"))
    except Exception as e:  # noqa: BLE001
        log.warning("dev console (s2s) tenant resolution failed: %s", e)
        await websocket.close(code=1008, reason="unknown tenant")
        return
    try:
        bridge = await _live_bridge_factory(websocket, tenant)
    except Exception as e:  # noqa: BLE001 - e.g. tenant has no realtime config
        log.warning("dev console (s2s) bridge build failed: %s", e)
        await websocket.close(code=1011, reason="s2s not configured for tenant")
        return
    try:
        await _run_billed_session(tenant, bridge, mode="s2s")
    except WebSocketDisconnect:
        log.info("dev console (s2s) client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("dev console (s2s) bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _build_browser_vad():
    """VAD for the dev console: prefer Silero (robust speech/noise discrimination
    so turns end cleanly on speakers); fall back to EnergyVAD if onnxruntime or
    the model isn't available. Silero needs 32 ms / 512-sample frames at 16 kHz.
    """
    try:
        vad = SileroVAD(sample_rate=16000, frame_ms=32, threshold=0.5)
        vad._ensure_model()  # load now so a failure falls back here, not mid-call
        log.info("dev console using SileroVAD")
        return vad
    except Exception as e:  # noqa: BLE001
        log.warning("SileroVAD unavailable (%s); using EnergyVAD", e)
        return EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0)


def _build_stream_provider(tenant: TenantContext):
    """Build a streaming-STT provider from pipeline.stt_streaming, or None.

    Returns None when no streaming config is present (batch behaviour) or when
    the provider can't be constructed (e.g. missing key) — the bridge then
    falls back to batch Groq, so this never blocks a call.
    """
    cfg = getattr(tenant.settings.pipeline, "stt_streaming", None)
    if cfg is None or not getattr(cfg, "provider", None):
        return None
    try:
        merged = {
            "provider": cfg.provider,
            "model": cfg.model,
            "language": cfg.language,
            "endpointing": cfg.endpointing,
            "utterance_end_ms": cfg.utterance_end_ms,
            # Platform-level key: the adapter reads DEEPGRAM_API_KEY from env.
        }
        return get_streaming_stt_provider(merged)
    except Exception as e:  # noqa: BLE001 - never block a call on streaming setup
        log.warning("streaming STT provider unavailable (%s); using batch", e)
        return None


def make_browser_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    campaign_resolver=None,
    handoff_store=None,
    platform_retriever=None,
) -> BrowserBridgeFactory:
    """Build a BrowserVoiceBridge per connection, wired to the tenant stack.

    Mirrors ``src.bootstrap.make_bridge_factory`` but returns a browser bridge.
    When a ``campaign_resolver`` is supplied, the agent's script + slots are
    resolved per call from the tenant's DB campaign (``?campaign=<id>``, else the
    tenant's active campaign, else the YAML fallback) instead of the global
    closure script/slots.
    """

    async def factory(websocket: WebSocket, tenant: TenantContext) -> BrowserVoiceBridge:
        import uuid

        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            qp0 = getattr(websocket, "query_params", {}) or {}
            lc = await campaign_resolver.resolve(tenant.id, qp0.get("campaign") or None)
            cur_script, cur_slots = lc.script, lc.slots

        from src.providers import (
            LLM_PROVIDERS, STREAMING_STT_PROVIDERS, STT_PROVIDERS, TTS_PROVIDERS,
            get_llm_provider, get_stt_provider, get_streaming_stt_provider, get_tts_provider,
        )

        query_params = getattr(websocket, "query_params", {}) or {}
        stt_sel = (query_params.get("stt") or "").strip().lower()
        llm_sel = (query_params.get("llm") or "").strip().lower()
        tts_sel = (query_params.get("tts") or "").strip().lower()

        # STT override — deepgram is streaming; sarvam/groq are batch.
        _stream_override = None
        if stt_sel in STREAMING_STT_PROVIDERS:
            stt = providers.get_stt(tenant)
            try:
                _stream_override = get_streaming_stt_provider({"provider": stt_sel})
            except Exception as e:  # noqa: BLE001 - missing key etc.
                log.warning("dev console: streaming STT override '%s' failed (%s); using default", stt_sel, e)
                _stream_override = _build_stream_provider(tenant)
        elif stt_sel in STT_PROVIDERS:
            try:
                stt = get_stt_provider({"provider": stt_sel})
            except Exception as e:  # noqa: BLE001
                log.warning("dev console: STT override '%s' failed (%s); using default", stt_sel, e)
                stt = providers.get_stt(tenant)
            _stream_override = None   # batch selected — disable streaming path
        else:
            stt = providers.get_stt(tenant)
            _stream_override = _build_stream_provider(tenant)

        try:
            llm = get_llm_provider({"provider": llm_sel}) if llm_sel in LLM_PROVIDERS else providers.get_llm(tenant)
        except Exception as e:  # noqa: BLE001
            log.warning("dev console: LLM override '%s' failed (%s); using default", llm_sel, e)
            llm = providers.get_llm(tenant)
        try:
            tts = get_tts_provider({"provider": tts_sel}) if tts_sel in TTS_PROVIDERS else providers.get_tts(tenant)
        except Exception as e:  # noqa: BLE001
            log.warning("dev console: TTS override '%s' failed (%s); using default", tts_sel, e)
            tts = providers.get_tts(tenant)

        tts_language = tenant.settings.pipeline.tts.language or "hi-IN"
        # Voice: ?voice= overrides the configured default (validated against the
        # TTS provider's roster), so the console's Voice dropdown applies in
        # layered mode just like it does for S2S.
        tts_voice = tenant.settings.pipeline.tts.voice_id
        sel_voice = (query_params.get("voice") or "").strip()
        if sel_voice:
            try:
                roster = {v.get("voice_id") for v in tts.get_available_voices(tts_language)}
            except Exception:  # noqa: BLE001
                roster = set()
            if sel_voice in roster:
                tts_voice = sel_voice
        pipeline_cfg = PipelineConfig(
            stt=STTConfig(language=tenant.settings.pipeline.stt.language or "hi-IN"),
            llm=LLMConfig(
                temperature=tenant.settings.pipeline.llm.temperature or 0.5,
                max_tokens=tenant.settings.pipeline.llm.max_tokens or 256,
                response_format=tenant.settings.pipeline.llm.response_format or "json",
            ),
            tts=TTSConfig(
                language=tts_language,
                voice_id=tts_voice,
                sample_rate=16000,
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)
        session_id = f"web_{uuid.uuid4().hex[:12]}"
        lead_name = (query_params.get("lead_name") or "").strip()
        lead_data = {"lead_name": lead_name, "name": lead_name} if lead_name else {}
        # Chat→voice handoff: a ?handoff=<token> resolves a short-lived Redis blob
        # (chat summary + customer context) so the voice agent continues the chat.
        handoff_token = (query_params.get("handoff") or "").strip()
        handoff_ctx: dict | None = None
        if handoff_token and handoff_store is not None:
            try:
                import json as _json
                raw = await handoff_store.redis.get(f"chat_handoff:{handoff_token}")
                if raw:
                    handoff_ctx = _json.loads(raw)
                    name = (handoff_ctx.get("customer_name") or "").strip()
                    if name:
                        lead_data["name"] = name
                        lead_data.setdefault("lead_name", name)
                    if handoff_ctx.get("chat_summary"):
                        lead_data["chat_summary"] = handoff_ctx["chat_summary"]
                    if handoff_ctx.get("customer_id"):
                        lead_data["customer_id"] = handoff_ctx["customer_id"]
            except Exception:  # noqa: BLE001 — a bad handoff blob must not block the call
                log.warning("chat handoff context load failed", extra={"token": handoff_token})

        # When a valid handoff is present, replace the campaign script with a
        # support-mode script. Campaign objective/opening/slots are irrelevant here.
        extra_directives: list[str] | None = None
        if handoff_ctx is not None:
            lang = (handoff_ctx.get("language") or cur_script.language_default or "hi")
            cur_script = VoiceBotScript(
                agent_name=cur_script.agent_name,
                agent_role="Customer Support",
                company_name=cur_script.company_name,
                language_default=lang,
            )
            cur_slots = SlotSchema()
            chat_summary = lead_data.get("chat_summary", "")
            if chat_summary:
                extra_directives = [
                    "CONTEXT — CHAT HANDOFF: The customer just switched from a support "
                    "chat conversation to this voice call. Summary of that chat:\n"
                    f"{chat_summary}\n\n"
                    "Continue helping them from where the chat left off. "
                    "Do NOT run a sales script. Do NOT ask them to repeat what they already "
                    "told you in the chat. Greet them briefly (e.g. 'I can hear you now, '  "
                    "'how can I help?') and pick up the conversation."
                ]
            else:
                extra_directives = [
                    "CONTEXT — CHAT HANDOFF: The customer switched from a support chat to "
                    "this voice call. Greet them briefly and ask how you can help."
                ]

        from src.bootstrap import _build_kb_context  # noqa: PLC0415

        kb_ctx = _build_kb_context(platform_retriever, None) or None
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id, lead_data=lead_data),
            state_machine=AgentStateMachine(),
            slot_schema=cur_slots,
            script=cur_script,
            engine=engine,
            store=None,
            extra_directives=extra_directives,
            kb_context=kb_ctx,
        )
        log.info("dev console built call", extra={"tenant": tenant.slug, "session_id": session_id})
        return BrowserVoiceBridge(
            websocket=websocket,
            agent=agent,
            vad=_build_browser_vad(),
            config=BrowserBridgeConfig(),
            stream_provider=_stream_override,
            llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
        )

    return factory


def make_live_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
    *,
    campaign_resolver=None,
    platform_retriever=None,
) -> LiveBridgeFactory:
    """Build a GeminiLiveBridge (S2S) per connection from pipeline.realtime.

    With a ``campaign_resolver``, the agent's script + slots come per call from
    the tenant's DB campaign (``?campaign=<id>`` → active → YAML fallback)."""

    async def factory(websocket: WebSocket, tenant: TenantContext) -> GeminiLiveBridge:
        import uuid

        rt = getattr(tenant.settings.pipeline, "realtime", None)
        if rt is None or not getattr(rt, "provider", None):
            raise RuntimeError("tenant has no pipeline.realtime config for S2S")

        cur_script, cur_slots = script, slots
        if campaign_resolver is not None:
            qp0 = getattr(websocket, "query_params", {}) or {}
            lc = await campaign_resolver.resolve(tenant.id, qp0.get("campaign") or None)
            cur_script, cur_slots = lc.script, lc.slots

        llm = providers.get_llm(tenant)
        # The agent is the same; only the bridge differs. The engine is required by
        # the constructor (the Live path doesn't synthesize via it).
        engine = PipelineEngine(
            providers.get_stt(tenant), llm, providers.get_tts(tenant),
            PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(sample_rate=16000)),
        )
        qp = getattr(websocket, "query_params", {}) or {}
        lead_name = (qp.get("lead_name") or "").strip()
        lead_data = {"lead_name": lead_name, "name": lead_name} if lead_name else {}
        session_id = f"live_{uuid.uuid4().hex[:12]}"
        from src.bootstrap import _build_kb_context  # noqa: PLC0415

        kb_ctx = _build_kb_context(platform_retriever, None) or None
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id, lead_data=lead_data),
            state_machine=AgentStateMachine(), slot_schema=cur_slots, script=cur_script,
            engine=engine, store=None, kb_context=kb_ctx,
        )

        # Voice: ?voice= overrides the config default. In the dev console any voice
        # from the full catalog is allowed (not just the tenant's allowed_voices).
        from src.providers.voice_catalog import list_voices as _lv
        _catalog_voices = {v["voice_id"] for v in _lv("gemini_live")}
        voice = (qp.get("voice") or "").strip() or rt.voice
        if voice and voice not in _catalog_voices:
            voice = rt.voice
        # Platform-level key: connect() reads GEMINI_API_KEY / GOOGLE_API_KEY.
        key = None
        config = RealtimeConfig(
            model=rt.model, voice=voice, language_code=rt.language_code,
            system_instruction=build_s2s_system_instruction(
                cur_script, cur_slots, lead_data, kb_context=kb_ctx),
            tools=[RECORD_TURN_SIGNAL],
        )

        async def connect(cfg: RealtimeConfig):
            return await GeminiLiveSession.connect(cfg, api_key=key)

        log.info("dev console built S2S call", extra={
            "tenant": tenant.slug, "session_id": session_id, "voice": voice, "model": rt.model})
        return GeminiLiveBridge(
            websocket=websocket, agent=agent, config=config, connect_session=connect,
            llm=llm, tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"))

    return factory
