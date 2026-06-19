"""Stringee telephony adapter.

Stringee is a Vietnam-based CPaaS with carrier interconnects across
Southeast Asia and India. Their authentication model is different from
Twilio/Exotel: instead of HTTP Basic Auth with an SID + token, Stringee
issues short-lived JWT access tokens signed with your API key + secret.

REST surface implemented:
- ``initiate_call``  POST /v1/call2/callout (or /v1/call/click2call) — outbound dial
- ``hangup``         POST /v1/call2/{sid}/hangup
- ``transfer``       not natively supported by their public REST API;
                     would need their SDK or an SCC (Stringee Call Control)
                     script. Implemented as a documented NotImplementedError.

Audio Streaming:
Stringee's audio streaming runs through their proprietary client SDKs
(JS / Android / iOS / Flutter), not a documented server-side WebSocket
protocol like Twilio Media Streams. Wiring our agent into Stringee
audio therefore requires a different pattern than the Twilio/Exotel
bridges. We stub ``stream_audio_in/out`` with a clear error message so
the gap is loud and isolated. See the docstrings in
``StringeeAdapter.stream_audio_in`` for the recommended path forward.

Endpoint reference: https://developer.stringee.com/docs/api-reference
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.interfaces.telephony import (
    CallConfig,
    CallSession,
    ITelephonyProvider,
)

log = logging.getLogger(__name__)

STRINGEE_BASE_URL = "https://api.stringee.com"


def mint_server_token(api_key_sid: str, api_key_secret: str, *, ttl_seconds: int = 3600) -> str:
    """Mint a Stringee server REST JWT (``rest_api: true``, ``cty`` header).

    Standalone version of ``StringeeAdapter._make_access_token`` for callers that
    only need a token (e.g. authenticating a recording download) without building
    a full adapter. Same header/claim shape the REST API requires.
    """
    import jwt  # PyJWT

    now = int(time.time())
    payload = {
        "jti": f"{api_key_sid}-{now}",
        "iss": api_key_sid,
        "exp": now + ttl_seconds,
        "rest_api": True,
    }
    token = jwt.encode(
        payload, api_key_secret, algorithm="HS256", headers={"cty": "stringee-api;v=1"})
    return token.decode("ascii") if isinstance(token, bytes) else token


def _bare_number(number: str) -> str:
    """Stringee requires bare digits — no leading '+' (E.164 with '+' is rejected
    as r:10 FROM/TO_NUMBER_INVALID_FORMAT). Strip a leading '+' and surrounding
    whitespace so callers/config may keep the '+E.164' form."""
    return (number or "").strip().lstrip("+")

# Stringee call event → our vocabulary.
_STATUS_MAP = {
    "STARTING":  "ringing",
    "RINGING":   "ringing",
    "ANSWERED":  "answered",
    "ENDED":     "answered",
    "BUSY":      "busy",
    "NO_ANSWER": "no_answer",
    "FAILED":    "failed",
}


class StringeeAdapter(ITelephonyProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        # Per-tenant creds only — NO platform-env fallback. The caller resolves the
        # tenant's Stringee keys (active_creds) and passes them here; a missing key
        # must fail loudly, not silently dial on the platform's Stringee account.
        api_key_sid = config.get("api_key_sid")
        api_key_secret = config.get("api_key_secret")
        if not (api_key_sid and api_key_secret):
            raise ValueError(
                "StringeeAdapter requires the tenant's api_key_sid + api_key_secret "
                "(configure the tenant's Stringee keys; the platform env is no longer "
                "a fallback)"
            )
        self._api_key_sid = api_key_sid
        self._api_key_secret = api_key_secret
        # Stringee keys are region-scoped: a key issued in (e.g.) asia-2 is
        # rejected by the global api.stringee.com with "keySid invalid". Allow
        # the regional API base to come from config or the STRINGEE_BASE_URL env
        # var (e.g. https://asia-2.api.stringee.com).
        self._base_url = (
            config.get("base_url")
            or os.environ.get("STRINGEE_BASE_URL")
            or STRINGEE_BASE_URL
        ).rstrip("/")
        self._timeout = config.get("timeout", 30.0)
        # ``userId`` is included in the callout body when set, but be aware:
        # a SERVER REST callout (/v1/call2/callout) is treated by Stringee as an
        # EXTERNAL phone->phone call regardless of this body field — it does NOT
        # become an app-user->phone call and Stringee does not run our SCCO for it
        # (confirmed live: outbound voicebot calls show UserID=null / phone->phone,
        # TTS=0). Only a CLIENT-SDK access token carries a userId that makes a call
        # app-originated, which is why the browser softphone works but server-
        # originated outbound AI does not. Use Twilio/Exotel for outbound AI;
        # reserve Stringee for the browser softphone + inbound. Kept here in case a
        # future Stringee feature/config honors it. Per-tenant via config["user_id"]
        # (STRINGEE_USER_ID env is a legacy fallback only).
        self._user_id: str | None = config.get("user_id") or os.environ.get("STRINGEE_USER_ID")
        # Tests can inject a pre-built bearer; otherwise we mint a fresh JWT.
        self._token_override: str | None = config.get("access_token")

    def _make_access_token(self, ttl_seconds: int = 3600) -> str:
        """Mint a Stringee access JWT signed with HS256 (api_key_secret).

        Stringee requires the JWT *header* to carry ``cty: stringee-api;v=1``
        in addition to the standard ``alg``/``typ``. Without it the REST API
        rejects the token with HTTP 403 ``{"r": 5, "message": "keySid
        invalid"}`` even when the keySid and signature are correct.
        """
        import jwt  # PyJWT — already pulled in transitively by Twilio SDK

        now = int(time.time())
        # The server REST callout requires ``rest_api: true`` (else r:45). NOTE: a
        # server callout is treated as an EXTERNAL phone->phone call — a body
        # ``userId`` does NOT make it app-originated and Stringee does not run our
        # SCCO for it (confirmed live: outbound voicebot calls = UserID null /
        # phone->phone / TTS 0). Outbound AI should go via Twilio/Exotel.
        payload = {
            "jti": f"{self._api_key_sid}-{now}",
            "iss": self._api_key_sid,
            "exp": now + ttl_seconds,
            "rest_api": True,
        }
        return jwt.encode(
            payload,
            self._api_key_secret,
            algorithm="HS256",
            headers={"cty": "stringee-api;v=1"},
        )

    def _headers(self) -> dict[str, str]:
        token = self._token_override or self._make_access_token()
        return {
            "X-STRINGEE-AUTH": token,
            "Content-Type": "application/json",
        }

    async def initiate_call(self, config: CallConfig) -> CallSession:
        """Place an outbound call via the Stringee REST API.

        Stringee's ``callout`` endpoint takes a JSON ``answer_url`` that
        returns an SCC (Stringee Call Control) script when the destination
        picks up — equivalent to TwiML.
        """
        # BARE digits (Stringee rejects '+E.164' as r:10).
        from_number = _bare_number(config.from_number)
        to_number = _bare_number(config.to_number)
        # ``from`` = the project DID, ``type: internal`` (the callout ACCEPTS this
        # r:0 and the phone rings). Using the userId as the from.number is rejected
        # r:4 FROM_NUMBER_NOT_FOUND, so the DID it is; the originating app user goes
        # in the top-level ``userId`` below instead.
        body: dict[str, Any] = {
            "from": {"type": "internal", "number": from_number, "alias": from_number},
            "to": [{"type": "external", "number": to_number, "alias": to_number}],
            "answer_url": config.webhook_url,
        }
        # Include ``userId`` when set, but note Stringee ignores it for a server
        # callout (stays external phone->phone; our SCCO does not run). Outbound AI
        # voicebot should use Twilio/Exotel; Stringee is for the browser softphone
        # (app-originated via a client token) + inbound.
        if self._user_id:
            body["userId"] = self._user_id
        # Log WHICH project (keySid=iss) + region (base) is authenticating, so a
        # FROM_NUMBER_NOT_BELONG_YOUR_PROJECT (r:15) is easy to diagnose: the
        # number is fine, the keys/region just don't own it.
        log.info("stringee callout", extra={
            "iss": self._api_key_sid, "base": self._base_url,
            "from": from_number, "to": to_number})
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/call2/callout",
                headers=self._headers(),
                json=body,
            )
            # Surface the response BODY on HTTP errors (e.g. a 403 carries
            # Stringee's reason — invalid user, permission, region) instead of the
            # bare httpx "403 Forbidden". ``raise_for_status`` hides the body.
            if resp.status_code >= 400:
                log.warning("stringee callout HTTP error",
                            extra={"status": resp.status_code, "body": resp.text})
                raise RuntimeError(
                    f"Stringee callout HTTP {resp.status_code}: {resp.text}")
            payload = resp.json()
        # Stringee returns HTTP 200 even on logical errors — the real outcome is in
        # ``r`` (0 = accepted). Surface a non-zero ``r`` instead of silently
        # returning an empty call id, so the caller (and logs) see WHY it failed
        # (e.g. invalid FROM user, number format, plan limits).
        r_code = payload.get("r")
        log.info("stringee callout response", extra={"r": r_code, "response": payload})
        if r_code not in (0, None):
            raise RuntimeError(
                f"Stringee callout rejected (r={r_code}): "
                f"{payload.get('message') or payload.get('msg') or payload}")
        call_id = payload.get("call_id") or payload.get("callId") or ""
        raw_status = (payload.get("status") or "STARTING").upper()
        return CallSession(
            session_id=str(call_id),
            status=_STATUS_MAP.get(raw_status, raw_status.lower()),
            to_number=config.to_number,
            from_number=config.from_number,
        )

    async def hangup(self, session_id: str) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/call2/{session_id}/hangup",
                headers=self._headers(),
            )
            if resp.status_code >= 500:
                resp.raise_for_status()

    async def transfer(self, session_id: str, to_number: str) -> None:
        """Stringee doesn't expose a direct REST transfer — requires an
        SCC script update or client-side SDK action. Surface this gap
        explicitly rather than silently no-op."""
        raise NotImplementedError(
            "Stringee transfer is not supported via the public REST API. "
            "Implement via an SCC script update or the client SDK; see "
            "https://developer.stringee.com/docs/voice-api/call-control"
        )

    # --- Media Streams stubs --------------------------------------------
    # Stringee does not expose a server-side bidirectional media WebSocket
    # in the Twilio Media Streams / Exotel Voicebot Streaming style. Every
    # supported real-time audio path requires either their client SDK
    # (JS / Android / iOS / Flutter) acting as a participant, or an SCC
    # script orchestrating a conference into which our agent joins as
    # another participant. The integration shape therefore lives outside
    # this adapter; we surface the gap loudly here.
    #
    # Recommended integration path (when wired):
    #
    #   1. Provision a "bot" Stringee user account for this tenant.
    #   2. On call answer, the SCC ``answer_url`` script creates a
    #      ``conference`` and dials our bot user into it
    #      (`<connect><conference id="..."/></connect>`).
    #   3. A separate Node/Python service runs the Stringee JS/Web SDK
    #      headlessly under the bot identity, exposing local audio in/out
    #      via a WebRTC data channel or a local WS bridge.
    #   4. That bridge forwards PCM frames to our agent over the network
    #      using the same shape ExotelMediaBridge already consumes.
    #
    # Until that bridge is built, calling these methods is a programmer
    # error — the adapter raises a NotImplementedError that names the
    # missing piece so the caller can either pick a different telephony
    # provider or build the conference bridge.
    #
    # See ``docs/stringee-streaming.md`` for the long-form recipe.

    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "Stringee server-side audio streaming is not supported; route this "
            "call through a Stringee conference + bot-user client SDK bridge. "
            "See docs/stringee-streaming.md for the integration recipe."
        )
        if False:  # pragma: no cover - keep the generator type honest
            yield b""

    async def stream_audio_out(
        self,
        session_id: str,
        audio_stream: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError(
            "Stringee server-side audio streaming is not supported; route this "
            "call through a Stringee conference + bot-user client SDK bridge. "
            "See docs/stringee-streaming.md for the integration recipe."
        )
