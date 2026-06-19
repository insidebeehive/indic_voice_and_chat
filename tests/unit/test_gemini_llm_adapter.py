from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.providers.llm.gemini import GeminiLLMAdapter


def _response(text: str, finish_reason: str = "STOP",
              prompt_tokens: int = 10, completion_tokens: int = 5) -> SimpleNamespace:
    """Build a fake response shape mirroring google.genai's GenerateContentResponse."""
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
        finish_reason=SimpleNamespace(name=finish_reason),
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
        ),
    )


def _make_client(*, generate_return: Any = None,
                 stream_chunks: list[Any] | None = None) -> SimpleNamespace:
    generate = AsyncMock(return_value=generate_return) if generate_return else AsyncMock()

    if stream_chunks is not None:
        async def _gen():
            for c in stream_chunks:
                yield c

        async def _start_stream(**kwargs):
            return _gen()

        stream = AsyncMock(side_effect=_start_stream)
    else:
        stream = AsyncMock()

    models = SimpleNamespace(
        generate_content=generate,
        generate_content_stream=stream,
    )
    return SimpleNamespace(aio=SimpleNamespace(models=models))


# --- generate() ---------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_text_and_usage() -> None:
    client = _make_client(generate_return=_response('{"x": 1}'))
    adapter = GeminiLLMAdapter({"client": client, "model": "gemini-2.0-flash"})

    result = await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(model="gemini-2.0-flash", temperature=0.4, max_tokens=128),
    )
    assert result.text == '{"x": 1}'
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    call_kwargs = client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"
    assert call_kwargs["config"]["temperature"] == 0.4
    assert call_kwargs["config"]["max_output_tokens"] == 128


@pytest.mark.asyncio
async def test_generate_maps_system_role_into_system_instruction() -> None:
    client = _make_client(generate_return=_response("ok"))
    adapter = GeminiLLMAdapter({"client": client})

    await adapter.generate(
        [
            LLMMessage(role="system", content="be terse"),
            LLMMessage(role="system", content="reply in Hindi"),
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content="namaste"),
            LLMMessage(role="user", content="status?"),
        ],
        LLMConfig(),
    )
    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["config"]["system_instruction"] == "be terse\n\nreply in Hindi"
    contents = kwargs["contents"]
    # System messages live in system_instruction, not in contents
    assert all(c["role"] in ("user", "model") for c in contents)
    # Assistant -> model
    assert any(c["role"] == "model" for c in contents)
    # Order preserved
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_generate_json_format_sets_mime_type() -> None:
    client = _make_client(generate_return=_response("{}"))
    adapter = GeminiLLMAdapter({"client": client})
    await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(response_format="json"),
    )
    cfg = client.aio.models.generate_content.await_args.kwargs["config"]
    assert cfg["response_mime_type"] == "application/json"


@pytest.mark.asyncio
async def test_generate_text_format_omits_mime_type() -> None:
    client = _make_client(generate_return=_response("plain"))
    adapter = GeminiLLMAdapter({"client": client})
    await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig(response_format="text"))
    cfg = client.aio.models.generate_content.await_args.kwargs["config"]
    assert "response_mime_type" not in cfg


@pytest.mark.asyncio
async def test_generate_finish_reason_max_tokens() -> None:
    client = _make_client(generate_return=_response("trunc", finish_reason="MAX_TOKENS"))
    adapter = GeminiLLMAdapter({"client": client})
    result = await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    assert result.finish_reason == "length"


@pytest.mark.asyncio
async def test_generate_finish_reason_safety_blocked() -> None:
    client = _make_client(generate_return=_response("", finish_reason="SAFETY"))
    adapter = GeminiLLMAdapter({"client": client})
    result = await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    assert result.finish_reason == "blocked"


# --- generate_stream() --------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stream_yields_text_per_chunk() -> None:
    client = _make_client(stream_chunks=[
        _response("Hel"),
        _response("lo, "),
        _response("world"),
    ])
    adapter = GeminiLLMAdapter({"client": client})

    tokens: list[str] = []
    async for t in adapter.generate_stream(
        [LLMMessage(role="user", content="hi")], LLMConfig(),
    ):
        tokens.append(t)
    assert tokens == ["Hel", "lo, ", "world"]


@pytest.mark.asyncio
async def test_generate_stream_skips_empty_chunks() -> None:
    empty = SimpleNamespace(text="", candidates=[])
    client = _make_client(stream_chunks=[empty, _response("hello"), empty])
    adapter = GeminiLLMAdapter({"client": client})
    tokens = []
    async for t in adapter.generate_stream([LLMMessage(role="user", content="hi")], LLMConfig()):
        tokens.append(t)
    assert tokens == ["hello"]


# --- Transient 5xx retry ------------------------------------------------


class _FakeAPIError(Exception):
    """Mimics google.genai's APIError: carries an HTTP ``code``."""

    def __init__(self, code: int) -> None:
        super().__init__(f"{code} transient")
        self.code = code


def _flaky_client(*, fail_times: int, code: int,
                  stream_chunks: list[Any] | None = None,
                  generate_return: Any = None):
    """Client that raises ``code`` for the first ``fail_times`` calls, then succeeds."""
    calls = {"stream": 0, "generate": 0}

    async def _gen():
        for c in stream_chunks or []:
            yield c

    async def _start_stream(**kwargs):
        if calls["stream"] < fail_times:
            calls["stream"] += 1
            raise _FakeAPIError(code)
        calls["stream"] += 1
        return _gen()

    async def _do_generate(**kwargs):
        if calls["generate"] < fail_times:
            calls["generate"] += 1
            raise _FakeAPIError(code)
        calls["generate"] += 1
        return generate_return

    models = SimpleNamespace(
        generate_content=AsyncMock(side_effect=_do_generate),
        generate_content_stream=AsyncMock(side_effect=_start_stream),
    )
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return client, calls


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Keep retry tests fast — skip the real backoff delay."""
    import src.providers.llm.gemini as gemini_mod
    monkeypatch.setattr(gemini_mod.asyncio, "sleep", AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 500, 503])
async def test_generate_stream_retries_transient_then_succeeds(code) -> None:
    client, calls = _flaky_client(
        fail_times=1, code=code, stream_chunks=[_response("ok")],
    )
    adapter = GeminiLLMAdapter({"client": client})
    tokens = [t async for t in adapter.generate_stream(
        [LLMMessage(role="user", content="hi")], LLMConfig(),
    )]
    assert tokens == ["ok"]
    assert calls["stream"] == 2  # one failure + one success


@pytest.mark.asyncio
async def test_generate_stream_raises_after_exhausting_retries() -> None:
    client, calls = _flaky_client(
        fail_times=99, code=500, stream_chunks=[_response("never")],
    )
    adapter = GeminiLLMAdapter({"client": client})
    with pytest.raises(_FakeAPIError):
        async for _ in adapter.generate_stream(
            [LLMMessage(role="user", content="hi")], LLMConfig(),
        ):
            pass
    assert calls["stream"] == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_generate_stream_does_not_retry_non_retriable() -> None:
    client, calls = _flaky_client(
        fail_times=99, code=400, stream_chunks=[_response("x")],
    )
    adapter = GeminiLLMAdapter({"client": client})
    with pytest.raises(_FakeAPIError):
        async for _ in adapter.generate_stream(
            [LLMMessage(role="user", content="hi")], LLMConfig(),
        ):
            pass
    assert calls["stream"] == 1  # 400 is not retried


@pytest.mark.asyncio
async def test_generate_retries_transient_then_succeeds() -> None:
    client, calls = _flaky_client(
        fail_times=1, code=500, generate_return=_response('{"ok": 1}'),
    )
    adapter = GeminiLLMAdapter({"client": client})
    result = await adapter.generate(
        [LLMMessage(role="user", content="hi")], LLMConfig(),
    )
    assert result.text == '{"ok": 1}'
    assert calls["generate"] == 2


# --- Construction ------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GeminiLLMAdapter({})


# --- _extract_text fallback ---------------------------------------------


def test_extract_text_falls_back_to_candidate_parts() -> None:
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[
                    SimpleNamespace(text="a"),
                    SimpleNamespace(text="b"),
                ]),
                finish_reason=SimpleNamespace(name="STOP"),
            )
        ],
    )
    assert GeminiLLMAdapter._extract_text(response) == "ab"


def test_extract_text_empty_when_no_candidates() -> None:
    assert GeminiLLMAdapter._extract_text(SimpleNamespace(text=None, candidates=[])) == ""


@pytest.mark.asyncio
async def test_transcribe_audio_returns_text_and_sends_inline_audio() -> None:
    client = _make_client(generate_return=_response("नमस्ते, transcript"))
    adapter = GeminiLLMAdapter({"client": client})
    out = await adapter.transcribe_audio(b"\xff\xe3audio-bytes", "audio/mpeg")
    assert out == "नमस्ते, transcript"
    parts = client.aio.models.generate_content.call_args.kwargs["contents"][0]["parts"]
    assert any(p.get("inline_data", {}).get("mime_type") == "audio/mpeg"
               and p["inline_data"]["data"] == b"\xff\xe3audio-bytes" for p in parts)


@pytest.mark.asyncio
async def test_transcribe_audio_returns_empty_on_failure() -> None:
    from unittest.mock import AsyncMock
    client = _make_client()
    client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("boom"))
    adapter = GeminiLLMAdapter({"client": client})
    assert await adapter.transcribe_audio(b"x", "audio/mpeg") == ""


# --- Multimodal content (Phase 0) --------------------------------------


@pytest.mark.asyncio
async def test_generate_multimodal_content_emits_inline_data_parts() -> None:
    from src.interfaces.llm import ContentPart
    client = _make_client(generate_return=_response("ok"))
    adapter = GeminiLLMAdapter({"client": client})
    msg = LLMMessage(role="user", content=[
        ContentPart(type="text", text="what is this?"),
        ContentPart(type="image", inline_data={"mime_type": "image/jpeg", "data": b"img"}),
    ])
    await adapter.generate([msg], LLMConfig(response_format="text"))
    contents = client.aio.models.generate_content.await_args.kwargs["contents"]
    parts = contents[0]["parts"]
    assert parts[0] == {"text": "what is this?"}
    assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert parts[1]["inline_data"]["data"] == b"img"


# --- Tool / function calling (Phase 0) ---------------------------------


def _tool_call_response(name: str, args: dict, finish_reason: str = "STOP") -> SimpleNamespace:
    part = SimpleNamespace(text=None, function_call=SimpleNamespace(name=name, args=args))
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part]),
        finish_reason=SimpleNamespace(name=finish_reason),
    )
    return SimpleNamespace(
        text=None, candidates=[candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )


@pytest.mark.asyncio
async def test_generate_with_tools_passes_function_declarations() -> None:
    from src.interfaces.llm import ToolSpec
    client = _make_client(generate_return=_response("ok"))
    adapter = GeminiLLMAdapter({"client": client})
    tools = [ToolSpec(
        name="search_kb", description="search",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})]
    await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(response_format="text", tools=tools))
    cfg = client.aio.models.generate_content.await_args.kwargs["config"]
    decls = cfg["tools"][0]["function_declarations"]
    assert decls[0]["name"] == "search_kb"
    assert decls[0]["parameters"]["properties"]["q"]["type"] == "string"


@pytest.mark.asyncio
async def test_generate_extracts_tool_calls_from_response() -> None:
    from src.interfaces.llm import ToolSpec
    client = _make_client(generate_return=_tool_call_response("search_kb", {"q": "refund"}))
    adapter = GeminiLLMAdapter({"client": client})
    result = await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(response_format="text", tools=[ToolSpec("search_kb", "s", {"type": "object"})]))
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_kb"
    assert result.tool_calls[0].arguments == {"q": "refund"}


@pytest.mark.asyncio
async def test_generate_with_tools_omits_json_mime() -> None:
    # Gemini rejects response_mime_type=application/json together with tools.
    from src.interfaces.llm import ToolSpec
    client = _make_client(generate_return=_response("ok"))
    adapter = GeminiLLMAdapter({"client": client})
    await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(response_format="json", tools=[ToolSpec("t", "d", {"type": "object"})]))
    cfg = client.aio.models.generate_content.await_args.kwargs["config"]
    assert "response_mime_type" not in cfg
    assert "tools" in cfg


@pytest.mark.asyncio
async def test_tool_result_message_becomes_function_response() -> None:
    from src.interfaces.llm import ToolCall
    client = _make_client(generate_return=_response("ok"))
    adapter = GeminiLLMAdapter({"client": client})
    msgs = [
        LLMMessage(role="user", content="status?"),
        LLMMessage(role="assistant", content="",
                   tool_calls=[ToolCall(id="c1", name="search_kb", arguments={"q": "x"})]),
        LLMMessage(role="tool", name="search_kb", tool_call_id="c1", content='{"results": []}'),
    ]
    await adapter.generate(msgs, LLMConfig(response_format="text"))
    contents = client.aio.models.generate_content.await_args.kwargs["contents"]
    model_parts = [c for c in contents if c["role"] == "model"]
    assert model_parts and "function_call" in model_parts[0]["parts"][0]
    fr = [p for c in contents for p in c["parts"] if "function_response" in p]
    assert fr and fr[0]["function_response"]["name"] == "search_kb"


@pytest.mark.asyncio
async def test_generate_no_tools_is_unchanged_regression() -> None:
    # The str-content + no-tools path must be byte-identical to before Phase 0.
    client = _make_client(generate_return=_response("ok"))
    adapter = GeminiLLMAdapter({"client": client})
    await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert "tools" not in kwargs["config"]
