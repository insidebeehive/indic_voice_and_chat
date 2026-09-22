"""The reference widget must actually render a voice-note reply's audio.

`tests/unit/test_chat_audio_reply_tts.py` covers the server half thoroughly:
synthesis, upload, persistence, and the `audio_url`/`audio_mime` fields on the
outbound `message` frame. Every one of those tests passed while a customer who
sent a voice note heard nothing, because `static/chat_widget.html`'s
`ws.onmessage` handled `type === "message"` by rendering `m.text` and dropping
the audio fields on the floor. The whole pipeline was green and the feature was
dead at the last hop.

That is a direct consequence of the wire design in `_send_reply`
(`src/api/chat.py`): audio rides as OPTIONAL fields on the existing `message`
frame precisely so an unaware client keeps working. The cost of that choice is
that an unaware client fails silently -- there is no frame type to not-handle,
no error, nothing in the server logs. So the consumer needs its own test.

These are string assertions against the shipped HTML rather than a DOM test:
the repo has no JS test runner, and adding one for a single handler is not
worth it. They are scoped to the `message` branch specifically, so they cannot
be satisfied by the `audio_ack` or `history` branches -- both of which already
rendered audio correctly, and neither of which fires for an agent's reply.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WIDGET = Path(__file__).resolve().parents[2] / "static" / "chat_widget.html"


@pytest.fixture(scope="module")
def widget_source() -> str:
    assert WIDGET.is_file(), f"reference widget missing at {WIDGET}"
    return WIDGET.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def message_branch(widget_source: str) -> str:
    """Just the `if (m.type === "message") { ... }` body of ws.onmessage.

    Isolating it is the point of this file: `audio_url` appearing *somewhere*
    in a 600-line widget proves nothing about the agent-reply path.
    """
    start = widget_source.index('if (m.type === "message") {')
    end = widget_source.index('} else if (m.type === "call_offer")', start)
    branch = widget_source[start:end]
    # Guard the guard: if the handler is ever restructured so this slice stops
    # being the reply-rendering code, the assertions below would pass or fail
    # for reasons unrelated to audio.
    assert 'add("agent", m.text)' in branch, (
        "the sliced branch no longer renders the agent's reply text; the "
        "extraction bounds in this fixture need updating"
    )
    return branch


def test_message_branch_renders_reply_audio(message_branch: str) -> None:
    """The agent-reply branch must consume `audio_url`, not just `text`."""
    assert "m.audio_url" in message_branch, (
        "chat_widget.html renders an agent reply's text but ignores the "
        "audio_url field _send_reply sends on the same frame -- a customer who "
        "sends a voice note gets a silent text answer"
    )
    assert "addAudioBubble(" in message_branch, (
        "audio_url is referenced but no playable element is created for it"
    )


def test_reply_audio_url_carries_session_id(message_branch: str) -> None:
    """`GET /chat/media/{id}` authorizes by bearer token OR `?session_id=`.

    The widget holds no bearer token, so omitting the query parameter makes the
    media fetch 401 and the player render as a broken control -- which looks
    identical to no audio having been sent at all.
    """
    audio_line = next(
        (ln for ln in message_branch.splitlines() if "m.audio_url" in ln and "addAudioBubble" in ln),
        None,
    )
    assert audio_line is not None, "expected one line that plays m.audio_url"
    assert re.search(r'session_id=["\s+]*\+?\s*sessionId', audio_line), (
        f"reply audio URL is used without ?session_id=; the media endpoint "
        f"will reject it with 401. Line: {audio_line.strip()}"
    )


def test_add_audio_bubble_helper_exists(widget_source: str) -> None:
    """`addAudioBubble` is declared after ws.onmessage and relied on by it.

    That works only because function declarations hoist within the widget's
    single IIFE. If it is ever converted to a `const` arrow function it would
    still pass the branch assertions above and then throw a TDZ
    ReferenceError at runtime on the first voice reply.
    """
    assert re.search(r"\bfunction\s+addAudioBubble\s*\(", widget_source), (
        "addAudioBubble must stay a hoisted function declaration -- the "
        "message handler above calls it before its definition point"
    )
