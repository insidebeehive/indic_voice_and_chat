"""The Pipeline tab must show only the layered- or s2s-mode section relevant
to a tenant's current mode, not both stacked at once, and the chat-voice
heading must reflect that its TTS is also used for a call started from chat.

Mirrors test_backoffice_events_webhook.py's style: reads static/backoffice.html
as a plain text fixture and does substring/slice assertions — no browser/JS
execution.
"""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "static" / "backoffice.html").read_text(encoding="utf-8")


def test_chat_voice_heading_mentions_call() -> None:
    assert "Chat voice replies/call" in HTML
    # Catches an incomplete rename (old heading left in place alongside/instead
    # of the new one).
    assert "Chat voice replies</h2>" not in HTML


def test_mode_pane_containers_exist() -> None:
    assert 'id="pane_pipe_layered"' in HTML
    assert 'id="pane_pipe_s2s"' in HTML


def test_mode_select_wired_to_onchange_handler() -> None:
    start = HTML.index('<select id="p_mode"')
    # The select's opening tag is short — a small window after it is enough
    # to catch the onchange attribute without slicing in unrelated markup.
    tag = HTML[start:start + 200]
    assert 'onchange="onPipeModeChange()"' in tag


def test_on_pipe_mode_change_defined_and_toggles_both_panes() -> None:
    assert "function onPipeModeChange()" in HTML
    start = HTML.index("function onPipeModeChange()")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert "pane_pipe_layered" in fn
    assert "pane_pipe_s2s" in fn


def test_on_pipe_mode_change_uses_hidden_property_not_style_display() -> None:
    """The toggle must use the ``.hidden`` DOM property (both the "show both"
    fallback branch and the effective-mode branch), never ``style.display`` --
    a regression to display-flipping would silently break layout/CSS that
    assumes `[hidden]` (see this repo's artifact/page reset convention) and
    is exactly the kind of change that could slip in while still "working"
    visually in a quick manual check."""
    start = HTML.index("function onPipeModeChange()")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert ".hidden = " in fn
    assert "style.display" not in fn


def test_load_pipeline_calls_on_pipe_mode_change() -> None:
    start = HTML.index("async function loadPipeline")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert "onPipeModeChange()" in fn
    # Ordering, not just presence: the substring "onPipeModeChange()" also
    # occurs earlier in this same function body as part of the
    # onchange="onPipeModeChange()" HTML attribute inside the template
    # literal (the #p_mode <select>) -- that occurrence proves nothing about
    # *initial* pane visibility. rindex finds the LAST occurrence, which is
    # the bare trailing call made right after pane_pipe's innerHTML is set;
    # confirm it really is a bare call (ends in ";", not inside an
    # onchange="..." attribute, which would end in "\"") so a future
    # unrelated later occurrence of the substring can't fool this.
    innerhtml_idx = fn.index('$("pane_pipe").innerHTML =')
    call_idx = fn.rindex("onPipeModeChange()")
    assert call_idx > innerhtml_idx, (
        "onPipeModeChange() must be invoked AFTER pane_pipe's innerHTML is "
        "assigned, not just referenced inside the onchange attribute markup"
    )
    tail = fn[call_idx + len("onPipeModeChange()"):call_idx + len("onPipeModeChange()") + 1]
    assert tail == ";", f"expected a bare trailing call ending in ';', found {tail!r}"


def _loadpipeline_fn() -> str:
    start = HTML.index("async function loadPipeline")
    return HTML[start:HTML.index("\n}\n", start)]


def _pane_content(fn: str, div_id: str) -> str:
    """The raw SOURCE-TEXT slice bounded by a pane's opening
    ``<div id="...">`` and its matching ``</div>``. Both panes' source text
    only ever calls out to pipeLayerHtml()/voicePickerHtml()/etc. -- helper
    functions defined *elsewhere* in the file -- rather than pasting their
    template bodies inline, so there is exactly one literal ``</div>`` between
    each pane's open tag and its own close in this file's source (verified by
    reading the file: neither pane's inline template contains a nested
    ``<div>``). The first ``</div>`` after the open tag is therefore the
    pane's own closer, not some unrelated inner one.
    """
    open_tag = f'<div id="{div_id}">'
    start = fn.index(open_tag)
    close = fn.index("</div>", start)
    return fn[start:close]


def test_mode_select_and_chat_voice_section_sit_outside_both_panes() -> None:
    """The #p_mode dropdown and the chatVoiceHtml() call must be rendered
    OUTSIDE pane_pipe_layered/pane_pipe_s2s -- they apply regardless of mode
    (chat voice notes exist independent of the call pipeline's mode, and the
    mode selector obviously can't live inside the section it controls).
    A regression that nested either inside one pane would make it disappear
    when the other pane is shown -- toggling panes would silently also hide
    the mode selector or the chat-voice section, which the existing tests
    (presence-only, or scoped to one function) would not catch."""
    fn = _loadpipeline_fn()
    layered = _pane_content(fn, "pane_pipe_layered")
    s2s = _pane_content(fn, "pane_pipe_s2s")
    for pane_name, pane in (("pane_pipe_layered", layered), ("pane_pipe_s2s", s2s)):
        assert 'id="p_mode"' not in pane, f'#p_mode select leaked into {pane_name}'
        assert "chatVoiceHtml(" not in pane, f'chatVoiceHtml() call leaked into {pane_name}'


def test_chat_voice_hint_mentions_chat_started_call_fallback() -> None:
    """chatVoiceHtml()'s hint must explain that this same TTS also backs a
    voice call started from a chat session, and where it falls back to when
    unset. Without this, an operator has no way to know why changing "Chat
    voice replies/call" settings also changes what that call sounds like. It
    must not claim the bot offers the call -- the bot never does."""
    start = HTML.index("function chatVoiceHtml(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert "voice call started from a chat session" in fn
    assert "that call falls back to the call TTS above" in fn
    assert "bot offers" not in fn
