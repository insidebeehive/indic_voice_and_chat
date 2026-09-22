"""Deepgram streaming STT adapter (live websocket).

Implements IStreamingSTTProvider. One DeepgramStreamSession owns a single
live websocket to Deepgram's /v1/listen endpoint, feeds it PCM16 audio, and
emits STTStreamEvents (interim / final / endpoint). Message-parse and
accumulation live in the pure ``_handle_raw`` method for testability; the
async receiver loop just calls it and enqueues non-None events. A keepalive
task sends ``{"type":"KeepAlive"}`` so the socket survives agent-speech gaps.

Uses the ``websockets`` library directly (bundled with deepgram-sdk) for full
control over framing/keepalive and easy faking in tests; the deepgram-sdk
callback client is not used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
from urllib.parse import urlencode

from src.interfaces.stt import (
    ISTTStreamSession,
    IStreamingSTTProvider,
    STTConfig,
    STTStreamEvent,
)
from src.utils.logging import debug_event

DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-2"
DEFAULT_LANGUAGE = "hi"

Connector = Callable[[str, dict[str, str]], Awaitable[Any]]

log = logging.getLogger(__name__)


class DeepgramStreamSession(ISTTStreamSession):
    def __init__(
        self,
        ws: Any,
        *,
        keepalive_interval: float = 5.0,
        start_tasks: bool = True,
    ) -> None:
        self._ws = ws
        self._keepalive_interval = keepalive_interval
        self._queue: asyncio.Queue[Optional[STTStreamEvent]] = asyncio.Queue()
        self._acc: list[str] = []
        self._endpointed = False
        self._closed = False
        self._tasks: list[asyncio.Task] = []
        if start_tasks:
            self._tasks.append(asyncio.create_task(self._receiver()))
            self._tasks.append(asyncio.create_task(self._keepalive()))

    def _handle_raw(self, raw: str) -> Optional[STTStreamEvent]:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            # A malformed/non-JSON frame from Deepgram is dropped entirely —
            # the caller never learns a frame was even received. Logging the
            # raw frame (not just "it failed") is what makes this diagnosable
            # instead of indistinguishable from a dropped connection.
            debug_event(log, "deepgram frame unparseable", raw=raw)
            return None
        mtype = msg.get("type")
        if mtype == "UtteranceEnd":
            if self._endpointed:
                return None
            text = " ".join(self._acc).strip()
            self._reset_utterance()
            if not text:
                return None
            return STTStreamEvent(type="endpoint", text=text)
        if mtype != "Results":
            return None
        alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
        transcript = (alt.get("transcript") or "").strip()
        is_final = bool(msg.get("is_final"))
        speech_final = bool(msg.get("speech_final"))
        conf = float(alt.get("confidence", 1.0) or 1.0)
        if speech_final:
            if transcript:
                self._acc.append(transcript)
            text = " ".join(self._acc).strip()
            self._reset_utterance(endpointed=True)
            if not text:
                return None
            return STTStreamEvent(type="endpoint", text=text, confidence=conf)
        if not transcript:
            return None
        if self._endpointed:
            # New speech after the previous endpoint: a fresh utterance has
            # begun, so clear the stale flag — otherwise this utterance's
            # UtteranceEnd backup would be suppressed (it would never endpoint
            # if Deepgram doesn't emit speech_final, e.g. after an audio gap
            # following a barge-in).
            self._endpointed = False
        if is_final:
            self._acc.append(transcript)
            return STTStreamEvent(type="final", text=transcript, confidence=conf)
        return STTStreamEvent(type="interim", text=transcript, confidence=conf)

    def _reset_utterance(self, endpointed: bool = False) -> None:
        self._acc = []
        self._endpointed = endpointed

    async def _receiver(self) -> None:
        reason = "stream ended"
        try:
            async for raw in self._ws:
                ev = self._handle_raw(raw if isinstance(raw, str) else raw.decode("utf-8"))
                if ev is not None:
                    await self._queue.put(ev)
        except Exception as e:  # noqa: BLE001 - surface as end-of-stream
            reason = f"{type(e).__name__}: {e}"
        finally:
            # Only noteworthy if Deepgram dropped us; a normal aclose() sets _closed.
            if not self._closed:
                log.warning("deepgram stream ended unexpectedly", extra={"reason": reason})
            await self._queue.put(None)

    async def _keepalive(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._keepalive_interval)
                if self._closed:
                    return
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
        except Exception:  # noqa: BLE001
            pass

    async def send(self, pcm16: bytes) -> None:
        if self._closed:
            return
        await self._ws.send(pcm16)

    async def events(self) -> AsyncIterator[STTStreamEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.send(json.dumps({"type": "CloseStream"}))
        except Exception:  # noqa: BLE001
            pass
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            await self._ws.close()
        except Exception:  # noqa: BLE001
            pass


class DeepgramSTTAdapter(IStreamingSTTProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._api_key = config.get("api_key") or os.environ.get("DEEPGRAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DeepgramSTTAdapter requires an API key (config 'api_key' or "
                "DEEPGRAM_API_KEY env var)"
            )
        self._model = config.get("model") or DEFAULT_MODEL
        self._language = config.get("language") or DEFAULT_LANGUAGE
        self._endpointing = int(config.get("endpointing") or 300)
        self._utterance_end_ms = int(config.get("utterance_end_ms") or 1000)
        self._keepalive_interval = float(config.get("keepalive_interval") or 5.0)
        self._connector: Connector = config.get("connector") or _default_connector

    def _build_url(self, config: STTConfig) -> str:
        params = {
            "encoding": "linear16",
            "sample_rate": config.sample_rate or 16000,
            "channels": 1,
            "model": self._model,
            "language": config.language or self._language,
            "smart_format": "true",
            "interim_results": "true",
            "endpointing": self._endpointing,
            "utterance_end_ms": self._utterance_end_ms,
        }
        return f"{DEEPGRAM_WS_URL}?{urlencode(params)}"

    async def open_stream(self, config: STTConfig) -> ISTTStreamSession:
        url = self._build_url(config)
        headers = {"Authorization": f"Token {self._api_key}"}
        # `url` carries the resolved model/language/sample_rate as query
        # params (no secret — the key rides in the Authorization header
        # below), so it's safe to log whole. `header_keys` only, never the
        # header values, same convention as src/chatbot/tool_executor.py.
        debug_event(log, "deepgram stream opening", url=url,
                    header_keys=list(headers.keys()))
        ws = await self._connector(url, headers)
        return DeepgramStreamSession(
            ws, keepalive_interval=self._keepalive_interval, start_tasks=True
        )


async def _default_connector(url: str, headers: dict[str, str]) -> Any:
    try:
        import websockets  # bundled with deepgram-sdk
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "DeepgramSTTAdapter requires 'websockets' (install deepgram-sdk)."
        ) from e
    # A more tolerant ping_timeout (default 20s) avoids the client closing the
    # socket with 1011 "keepalive ping timeout" during quiet stretches or when
    # the event loop is briefly busy with LLM/TTS work. The bridge also reopens
    # on any drop, so this just reduces how often that has to happen.
    try:
        return await websockets.connect(
            url, additional_headers=headers, ping_timeout=60, close_timeout=5
        )
    except TypeError:  # pragma: no cover - older websockets
        return await websockets.connect(
            url, extra_headers=headers, ping_timeout=60, close_timeout=5
        )
