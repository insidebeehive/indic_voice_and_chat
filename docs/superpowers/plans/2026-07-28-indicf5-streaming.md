# IndicF5 Streaming Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `IndicF5TTSAdapter.synthesize()` to consume the new streaming endpoint the IndicF5 team is implementing on their RunPod server, per the contract we specified to them: chunked HTTP over a regular POST, raw PCM16 mono chunks (no per-chunk WAV framing), end-of-stream on connection close. This removes the server-side "buffer the whole file before responding" latency that was likely most of the ~15s `tts_total_ms` measured earlier this session.

**Architecture:** Replace the current single blocking `client.post()` + `resp.content` call with `client.stream("POST", ...)` + `resp.aiter_bytes(chunk_size=4096)` — the exact pattern already proven in this codebase by `ElevenLabsTTSAdapter.synthesize_stream`. Accumulate chunks into one buffer and return the same `TTSResult` as today, so the external contract to the engine is completely unchanged — no engine code touched.

**Tech Stack:** Python 3, httpx, pytest, pytest-asyncio, respx.

## Global Constraints

- Branch is `stage` — re-verify with `git rev-parse --abbrev-ref HEAD` immediately before committing (this session had two separate incidents where a commit landed on `main` because the working tree had silently drifted off `stage`). Do not create a new branch.
- **The exact new endpoint path is UNCONFIRMED.** The IndicF5 team is implementing a new, separate streaming endpoint (the old non-streaming `POST /tts` stays as-is), but the exact path hasn't been shared yet — it will be verified live against the real pod once it's up, together with them, on Stage. This plan uses a best guess (`/tts/stream`) isolated into a single named constant so correcting it later is a one-line diff, not a redesign. The commit message MUST say this plainly.
- Do NOT add a fallback to the old non-streaming `/tts` endpoint if the new one fails — keep this simple (full replace). A dual-path fallback would be premature complexity for a transitional state neither side has tested yet.
- Run `.venv/bin/python -m pytest tests/unit -q` after the task. Baseline immediately before this task: 2 failed (both known pre-existing, unrelated — `test_chat_routes.py::test_claim_session_and_agent_ws`, `test_prompts.py::test_chatbot_prompt_has_scope_guardrails`), 1165 passed, 1 skipped, 0 errors. Do not touch either of those two known failures.
- No Alembic migration — this is a provider-adapter code change only.

---

### Task 1: Stream IndicF5's synthesize() from the new endpoint

**Files:**
- Modify: `src/providers/tts/indicf5.py`
- Test: `tests/unit/test_tts_adapters.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the only task). Uses the existing `ITTSProvider`/`TTSConfig`/`TTSResult` interface (`src/interfaces/tts.py`, unchanged) and `resample_pcm16`/`normalize_for_tts` (unchanged).
- Produces: `IndicF5TTSAdapter.synthesize()` keeps its exact existing signature (`async def synthesize(self, text: str, config: TTSConfig) -> TTSResult`) — no caller anywhere in the codebase needs to change.

- [ ] **Step 1: Confirm you're on the right branch**

```bash
git rev-parse --abbrev-ref HEAD
```
Expected: `stage`. If it prints anything else, stop and report back — do not proceed or switch branches yourself.

- [ ] **Step 2: Write the failing tests**

In `tests/unit/test_tts_adapters.py`:

2a. Delete the now-unused `_wav()` helper (currently lines 28-34) — after this task, no test constructs a WAV-wrapped body anymore (the new streaming endpoint returns raw PCM, no header):
```python
def _wav(pcm: bytes, rate: int) -> bytes:
    """Minimal RIFF/WAVE wrapper (16-bit mono) around raw PCM."""
    import struct
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(pcm)) + pcm)
```
Delete this entire function.

2b. Add `_STREAM_PATH` to the import line for the module under test:
```python
from src.providers.tts.indicf5 import IndicF5TTSAdapter
```
Change to:
```python
from src.providers.tts.indicf5 import IndicF5TTSAdapter, _STREAM_PATH
```

2c. Replace these 5 existing tests (currently spanning roughly lines 165-256, the ones that call `indicf5.synthesize(...)`):

```python
@pytest.mark.asyncio
@respx.mock
async def test_indicf5_synthesize_resamples_24k_to_requested_rate(indicf5: IndicF5TTSAdapter) -> None:
    # IndicF5 renders at its model rate (24kHz) regardless of the request,
    # but the pipeline/bridges are wired for the REQUESTED rate and don't
    # honor TTSResult.sample_rate — unresampled 24k plays at ~2/3 speed.
    # The adapter must strip the WAV header AND resample to the request.
    pcm = b"\x01\x02" * 2400  # 4800 bytes => 0.1s @ 24kHz mono 16-bit
    route = respx.post(f"{_INDICF5_URL}/tts").mock(
        return_value=Response(200, content=_wav(pcm, 24000)))
    result = await indicf5.synthesize("नमस्कार", TTSConfig(language="mr-IN", sample_rate=16000))
    assert result.sample_rate == 16000
    # 0.1s of audio at 16kHz mono 16-bit = 3200 bytes (duration preserved).
    assert len(result.audio) == pytest.approx(3200, abs=8)
    assert result.duration_ms == pytest.approx(100.0, rel=0.05)
    body = route.calls[0].request.read()
    import json as _json
    payload = _json.loads(body)
    assert payload["lang"] == "mr"  # BCP-47 mr-IN mapped to the server's bare code


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_no_resample_when_rates_match(indicf5: IndicF5TTSAdapter) -> None:
    pcm = b"\x01\x02" * 1600  # 3200 bytes => 0.1s @ 16kHz
    respx.post(f"{_INDICF5_URL}/tts").mock(
        return_value=Response(200, content=_wav(pcm, 16000)))
    result = await indicf5.synthesize("hi", TTSConfig(language="hi-IN", sample_rate=16000))
    assert result.audio == pcm  # byte-identical, no needless conversion
    assert result.sample_rate == 16000


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_retries_5xx_then_succeeds(indicf5: IndicF5TTSAdapter) -> None:
    pcm = b"\x00\x01" * 100
    route = respx.post(f"{_INDICF5_URL}/tts")
    # 16k here so no resample obscures the byte-identity check — this test is
    # about retry behavior; resampling has its own tests above.
    route.side_effect = [Response(503), Response(200, content=_wav(pcm, 16000))]
    result = await indicf5.synthesize("Namaste", TTSConfig(language="hi-IN"))
    assert result.audio == pcm
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_does_not_retry_4xx(indicf5: IndicF5TTSAdapter) -> None:
    route = respx.post(f"{_INDICF5_URL}/tts").mock(return_value=Response(422))
    with pytest.raises(httpx.HTTPStatusError):
        await indicf5.synthesize("hi", TTSConfig(language="hi-IN"))
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_passes_extra_pronunciations_to_normalize(
    indicf5: IndicF5TTSAdapter, monkeypatch,
) -> None:
    captured = {}

    def _fake_normalize(text, language=None, extra=None):
        captured["extra"] = extra
        return text

    monkeypatch.setattr("src.providers.tts.indicf5.normalize_for_tts", _fake_normalize)
    pcm = b"\x01\x02" * 100
    respx.post(f"{_INDICF5_URL}/tts").mock(return_value=Response(200, content=_wav(pcm, 16000)))

    config = TTSConfig(language="hi-IN", extra_pronunciations={"XYZ": "एक्स वाय ज़ेड"})
    await indicf5.synthesize("hello XYZ", config)

    assert captured["extra"] == {"XYZ": "एक्स वाय ज़ेड"}
```

with these 6 (5 rewritten for the new raw-PCM streaming endpoint, plus 1 new test proving multi-chunk reassembly):

```python
@pytest.mark.asyncio
@respx.mock
async def test_indicf5_synthesize_resamples_24k_to_requested_rate(indicf5: IndicF5TTSAdapter) -> None:
    # IndicF5 always renders at its fixed native rate (24kHz) regardless of
    # the request, but the pipeline/bridges are wired for the REQUESTED rate
    # and don't honor TTSResult.sample_rate — unresampled 24k plays at ~2/3
    # speed. The adapter must resample to the request. Raw PCM, no WAV header.
    pcm = b"\x01\x02" * 2400  # 4800 bytes => 0.1s @ 24kHz mono 16-bit
    route = respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(
        return_value=Response(200, content=pcm))
    result = await indicf5.synthesize("नमस्कार", TTSConfig(language="mr-IN", sample_rate=16000))
    assert result.sample_rate == 16000
    # 0.1s of audio at 16kHz mono 16-bit = 3200 bytes (duration preserved).
    assert len(result.audio) == pytest.approx(3200, abs=8)
    assert result.duration_ms == pytest.approx(100.0, rel=0.05)
    body = route.calls[0].request.read()
    import json as _json
    payload = _json.loads(body)
    assert payload["lang"] == "mr"  # BCP-47 mr-IN mapped to the server's bare code


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_no_resample_when_rates_match(indicf5: IndicF5TTSAdapter) -> None:
    pcm = b"\x01\x02" * 1600  # 3200 bytes @ 24kHz (IndicF5's fixed native rate)
    respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(
        return_value=Response(200, content=pcm))
    result = await indicf5.synthesize("hi", TTSConfig(language="hi-IN", sample_rate=24000))
    assert result.audio == pcm  # byte-identical, no needless conversion
    assert result.sample_rate == 24000


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_retries_5xx_then_succeeds(indicf5: IndicF5TTSAdapter) -> None:
    pcm = b"\x00\x01" * 100
    route = respx.post(f"{_INDICF5_URL}{_STREAM_PATH}")
    # 24k (native) here so no resample obscures the byte-identity check —
    # this test is about retry behavior; resampling has its own test above.
    route.side_effect = [Response(503), Response(200, content=pcm)]
    result = await indicf5.synthesize("Namaste", TTSConfig(language="hi-IN", sample_rate=24000))
    assert result.audio == pcm
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_does_not_retry_4xx(indicf5: IndicF5TTSAdapter) -> None:
    route = respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(return_value=Response(422))
    with pytest.raises(httpx.HTTPStatusError):
        await indicf5.synthesize("hi", TTSConfig(language="hi-IN"))
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_passes_extra_pronunciations_to_normalize(
    indicf5: IndicF5TTSAdapter, monkeypatch,
) -> None:
    captured = {}

    def _fake_normalize(text, language=None, extra=None):
        captured["extra"] = extra
        return text

    monkeypatch.setattr("src.providers.tts.indicf5.normalize_for_tts", _fake_normalize)
    pcm = b"\x01\x02" * 100
    respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(return_value=Response(200, content=pcm))

    config = TTSConfig(language="hi-IN", extra_pronunciations={"XYZ": "एक्स वाय ज़ेड"})
    await indicf5.synthesize("hello XYZ", config)

    assert captured["extra"] == {"XYZ": "एक्स वाय ज़ेड"}


@pytest.mark.asyncio
@respx.mock
async def test_indicf5_reassembles_multiple_stream_chunks(indicf5: IndicF5TTSAdapter) -> None:
    # Content larger than the adapter's 4096-byte chunk_size forces httpx's
    # aiter_bytes to split delivery into multiple chunks — this proves the
    # adapter accumulates ALL chunks (not just the first) into the final
    # audio, rather than silently truncating a multi-chunk stream.
    pcm = bytes(range(256)) * 50  # 12800 bytes, spans multiple 4096-byte chunks
    respx.post(f"{_INDICF5_URL}{_STREAM_PATH}").mock(
        return_value=Response(200, content=pcm))
    result = await indicf5.synthesize("hi", TTSConfig(language="hi-IN", sample_rate=24000))
    assert result.audio == pcm  # exact reassembly, no truncation/corruption
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_tts_adapters.py -v -k indicf5`
Expected: all 6 new/rewritten `synthesize`-calling tests FAIL (the current adapter still posts to `/tts`, not `{_STREAM_PATH}`, and respx's `@respx.mock` raises an assertion error for any unmocked route the code actually hits). `test_indicf5_voices_cover_indic_languages`, `test_indicf5_constructor_requires_url`, `test_indicf5_registered_in_provider_registry` continue to pass unaffected (they don't call `synthesize`).

- [ ] **Step 4: Implement the streaming adapter**

In `src/providers/tts/indicf5.py`, replace the module docstring:

```python
"""IndicF5 TTS adapter — self-hosted fine-tuned voice server (RunPod).

The server exposes ``POST {base_url}/tts`` with ``{"text", "lang", "speed"}``
and returns WAV bytes. There is no native streaming and no voice parameter
(the fine-tune IS the voice), so ``synthesize_stream`` synthesizes per text
segment like the Sarvam adapter, and ``get_available_voices`` reports the one
fine-tuned voice for every IndicF5 language.

Config: ``base_url`` in the provider config, or the platform-level
``INDICF5_TTS_URL`` env var (e.g. ``https://<pod-id>-8000.proxy.runpod.net``).
"""
```

with:

```python
"""IndicF5 TTS adapter — self-hosted fine-tuned voice server (RunPod).

``synthesize`` streams from ``POST {base_url}{_STREAM_PATH}`` with
``{"text", "lang", "speed"}`` — chunked HTTP, raw PCM16 mono chunks (no
per-chunk WAV framing), end-of-stream on connection close. The exact stream
path is UNCONFIRMED pending live verification against the real pod — see
``_STREAM_PATH`` below. There is no voice parameter (the fine-tune IS the
voice), so ``synthesize_stream`` synthesizes per text segment like the
Sarvam adapter, and ``get_available_voices`` reports the one fine-tuned
voice for every IndicF5 language.

Config: ``base_url`` in the provider config, or the platform-level
``INDICF5_TTS_URL`` env var (e.g. ``https://<pod-id>-8000.proxy.runpod.net``).
"""
```

Remove the now-unused import (confirmed via grep: nothing else in this file uses it after this change):
```python
from src.providers.tts.sarvam import _extract_pcm
```
Delete this line entirely.

Add two new module-level constants, placed after the existing `_TTS_ATTEMPTS = 2  # initial try + 1 retry` line:
```python
_TTS_ATTEMPTS = 2  # initial try + 1 retry

# UNCONFIRMED — best guess pending live verification against the real pod.
# If IndicF5's actual path differs, this is the only line that needs to change.
_STREAM_PATH = "/tts/stream"

# IndicF5 always renders at this fixed rate regardless of what's requested.
# The old WAV-wrapped /tts response let us read this from the header; the new
# raw-PCM streaming response has no header, so it's hardcoded here instead.
_NATIVE_SAMPLE_RATE = 24000
```

Replace the entire `synthesize` method body (currently everything from `async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:` through its final `return TTSResult(...)`) with:

```python
    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        # Same normalization as Sarvam: speak currency amounts and rewrite
        # words TTS mispronounces, scoped by language/script.
        text = normalize_for_tts(text, config.language, extra=config.extra_pronunciations)
        body = {
            "text": text,
            "lang": _server_lang(config.language),
            "speed": config.speed,
        }
        timeout = httpx.Timeout(self._timeout, connect=min(self._timeout, 5.0))
        chunks: list[bytes] | None = None
        last_exc: Exception | None = None
        for attempt in range(_TTS_ATTEMPTS):
            try:
                collected: list[bytes] = []
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", f"{self._base_url}{_STREAM_PATH}", json=body,
                    ) as resp:
                        resp.raise_for_status()
                        async for chunk in resp.aiter_bytes(chunk_size=4096):
                            if chunk:
                                collected.append(chunk)
                chunks = collected
                break
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                log.warning("indicf5 tts stream transient error (attempt %d/%d): %s",
                            attempt + 1, _TTS_ATTEMPTS, e)
            except httpx.HTTPStatusError as e:
                # Retry only transient 5xx; surface 4xx (bad request) at once.
                # A mid-stream failure after some chunks were already read also
                # lands here (or above) and discards `collected` entirely —
                # there is no partial-audio recovery, the whole request retries.
                if e.response.status_code >= 500 and attempt + 1 < _TTS_ATTEMPTS:
                    last_exc = e
                    log.warning("indicf5 tts stream %s (attempt %d/%d); retrying",
                                e.response.status_code, attempt + 1, _TTS_ATTEMPTS)
                    continue
                raise
        if chunks is None:
            raise last_exc  # type: ignore[misc]  # set whenever the loop didn't break
        audio_bytes = b"".join(chunks)
        if not audio_bytes:
            raise RuntimeError("IndicF5 TTS returned empty audio")

        # Raw PCM16 mono, no WAV header on this endpoint — IndicF5 always
        # renders at its fixed native rate (24kHz), same as the old /tts path,
        # just no longer readable from a header since there isn't one anymore.
        actual_rate = _NATIVE_SAMPLE_RATE
        # The pipeline sends TTS audio to the sink as-is and the bridges are
        # wired for the REQUESTED rate (TTSConfig(sample_rate=16000) →
        # 16k→8k telephony conversion happens there) — TTSResult.sample_rate
        # is informational, not honored. 24kHz passed through unresampled
        # would play at ~2/3 speed, so convert here, like Sarvam's API does
        # natively when asked for a speech_sample_rate.
        if actual_rate != config.sample_rate:
            audio_bytes, _ = resample_pcm16(audio_bytes, actual_rate, config.sample_rate)
        duration_ms = (len(audio_bytes) / max(config.sample_rate * 2, 1)) * 1000.0
        return TTSResult(
            audio=audio_bytes,
            duration_ms=duration_ms,
            sample_rate=config.sample_rate,
        )
```

Do NOT change `synthesize_stream` or `get_available_voices` — both are untouched (`synthesize_stream` calls `self.synthesize(...)` per segment, so it automatically benefits from the new streaming path).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_tts_adapters.py -v`
Expected: all tests in this file pass, including all 6 IndicF5 `synthesize`-related tests and the 3 unaffected ones, plus every Sarvam test in the same file (unaffected by this change).

- [ ] **Step 6: Run the full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit -q`
Expected: 2 failed (the same two known pre-existing, unrelated failures), 1 skipped, 0 errors, passed count up by 1 from baseline (one net new test added).

- [ ] **Step 7: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # must print "stage" — stop if it doesn't
git add src/providers/tts/indicf5.py tests/unit/test_tts_adapters.py
git status --short
```

Confirm the status output shows exactly these 2 modified files, no unrelated changes. Then commit:

```bash
git commit -m "$(cat <<'EOF'
feat(tts): stream IndicF5 synthesis from the new pod endpoint

IndicF5's server buffered the whole response before replying (measured
~15s tts_total_ms in this session's benchmarking round). The IndicF5 team
is implementing real chunked-HTTP streaming server-side per the contract
we specified (raw PCM16 chunks, no per-chunk WAV framing, EOF on
connection close) — this updates our adapter to consume it the same way
ElevenLabsTTSAdapter.synthesize_stream already does (client.stream() +
aiter_bytes()), accumulating chunks into the same TTSResult the engine
already expects, so no engine code changes.

IMPORTANT: the exact endpoint path (_STREAM_PATH = "/tts/stream") is an
UNCONFIRMED best guess — the IndicF5 team hasn't shared the real path yet
and it needs live verification against the actual pod on Stage once it's
up. If wrong, only that one constant needs to change.
EOF
)"
```

---

## Verification

- `.venv/bin/python -m pytest tests/unit -q` — 2 failed (both known pre-existing), 0 errors, passed count up by 1 from baseline.
- `.venv/bin/python -m pytest tests/unit/test_tts_adapters.py -v` — all pass.
- `git diff --stat` (before committing) touches only `src/providers/tts/indicf5.py` and `tests/unit/test_tts_adapters.py`.
- Manually confirm `_STREAM_PATH` is the ONLY place the new endpoint path string appears in the adapter (so correcting it later, once IndicF5 shares the real path, is a true one-line change).
- **Do not consider this integration verified** — it has not been tested against the real pod. The next step, once the pod is up, is a live end-to-end call on Stage to confirm the actual path/format match what was guessed here.
