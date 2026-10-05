"""The Pipeline tab must show only the layered- or s2s-mode section relevant
to a tenant's current mode, not both stacked at once, and the chat-voice
heading must reflect that its TTS is also used for a call started from chat.

Mostly mirrors test_backoffice_events_webhook.py's style: reads
static/backoffice.html as a plain text fixture and does substring/slice
assertions -- no browser/JS execution. The Female/Male voice picker tests
near the bottom of this file are the exception: they run the relevant JS (via
node) against a stubbed DOM and inspect the PATCH body savePipeline()
produces.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

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


# --- Male voice picker -------------------------------------------------
# TenantTTSConfig.male_voice_id (src/config_tenant.py), surfaced on the
# pipeline tab so an operator can set the voice the CRM's bot_gender picks
# for a male bot. The existing cv_tts picker is relabelled "Female".

def _chat_voice_html_fn() -> str:
    start = HTML.index("function chatVoiceHtml(")
    return HTML[start:HTML.index("\n}\n", start)]


def test_female_label_shown_for_existing_voice_picker() -> None:
    """The existing cv_tts picker (voice_id, unchanged plumbing) is now
    labelled "Female", not "Chat TTS"."""
    fn = _chat_voice_html_fn()
    assert 'voicePickerHtml("cv_tts", "Female"' in fn


def test_male_picker_rendered_with_shared_picker_machinery() -> None:
    """The Male picker must be wired through the shared catalog-backed
    picker machinery (voicePickerHtml), not a hand-rolled select -- that's
    what gives it a roster, a custom-entry fallback, and a "current" hint
    for free."""
    fn = _chat_voice_html_fn()
    assert 'voicePickerHtml("cv_tts_male", "Male"' in fn
    assert "cv.effective_male_voice_id" in fn


def test_bot_gender_hint_present() -> None:
    fn = _chat_voice_html_fn()
    assert "bot_gender picks Male voice for male bots" in fn


def test_refresh_voice_options_cascades_cv_tts_to_male_picker() -> None:
    """refreshVoiceOptions("cv_tts") must also re-query the Male picker --
    it shares cv_tts's provider/language inputs, so anything that re-queries
    the Female picker (a provider pick via onPipeProviderChange('cv_tts'), or
    the Chat TTS language input's onchange="refreshVoiceOptions('cv_tts')")
    must not leave Male showing a stale roster from the previous
    provider/language."""
    start = HTML.index("async function refreshVoiceOptions(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert 'refreshVoiceOptions("cv_tts_male")' in fn
    assert 'kind === "cv_tts"' in fn

    # The existing onchange attributes (unchanged by this feature) are what
    # actually trigger the cascade above -- confirm they're still exactly
    # these, not renamed to some other wrapper function.
    assert 'id="p_cv_tts_provider" onchange="onPipeProviderChange(\'cv_tts\')"' in HTML
    assert 'id="p_cv_tts_language" placeholder="blank = leave unchanged" onchange="refreshVoiceOptions(\'cv_tts\')"' in HTML


def test_load_pipeline_populates_male_picker() -> None:
    """loadPipeline() must refresh the Male picker on initial render, same as
    it already does for the Female (cv_tts) picker -- otherwise it'd sit
    empty until the operator manually touches the Chat TTS provider/language
    fields. It does so by calling refreshVoiceOptions("cv_tts"), which itself
    cascades to cv_tts_male (see the cascade test above) -- loadPipeline's
    own source need not repeat that literal string."""
    fn = _loadpipeline_fn()
    assert 'refreshVoiceOptions("cv_tts")' in fn


# --- Behavioural: actually run savePipeline()/resolveVoiceQuery() in node --
# Extracts the backoffice page's inline <script> (minus its top-level
# bootstrap calls -- loadCreds()/loadTenants()/loadKB(null, null), which hit
# browser-only APIs this harness doesn't stub and whose side effects nothing
# below depends on: every function/variable they'd need is declared earlier
# in the same script, which IS kept) and evaluates it in node with minimal
# stubs for the browser globals it touches at load time (document/window/
# localStorage/fetch/navigator). apiSend/loadTenants/loadPipeline are then
# redeclared AFTER the extracted script in the same scope -- a later
# `function`/`async function` declaration of the same name overwrites the
# earlier one once both are hoisted, so savePipeline()'s calls to them resolve
# to these test doubles instead of hitting the real network.

NODE = shutil.which("node")
# Scoped to just the node-backed tests below (not a module-level `pytestmark`)
# -- the string-matching tests above this point don't need node and must keep
# running even where it isn't installed.
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _extract_backoffice_script() -> str:
    start = HTML.index("<script>") + len("<script>")
    end = HTML.index("</script>", start)
    script = HTML[start:end]
    boot_marker = "loadCreds(); loadTenants();"
    return script[: script.index(boot_marker)]


_SCRIPT = _extract_backoffice_script()

_HARNESS_TEMPLATE = """
'use strict';
const sent = [];
const _elements = new Map();
// A real <select> clears/re-derives .value as a side effect of assigning
// .innerHTML (the previously-selected <option> no longer exists once the
// options are replaced, so the browser falls back to whichever option -- if
// any -- is now first/selected). This stub element is a plain object: its
// .value is untouched by writing .innerHTML, so it has no equivalent of that
// reset. Code whose correctness depends on keepValue actually being cleared
// by an innerHTML rewrite (as opposed to the explicit `select.value =
// keepValue` restores refreshVoiceOptions() does afterwards) cannot be
// exercised against this stub -- it would pass here even if that reset
// logic were deleted from the real browser's behavior entirely.
function _makeElement(id) {
  return { id, value: "", textContent: "", innerHTML: "", disabled: false, hidden: false };
}
const document = {
  getElementById(id) {
    if (!_elements.has(id)) _elements.set(id, _makeElement(id));
    return _elements.get(id);
  },
  querySelectorAll() { return []; },
  querySelector() { return null; },
  createElement() { return _makeElement("_tmp"); },
};
const window = { location: { origin: "http://test" } };
const localStorage = { getItem() { return null; }, setItem() {} };
function fetch() { return Promise.resolve({ ok: true, status: 200, json: async () => ({}) }); }
const navigator = { clipboard: { writeText: async () => {} } };

__SCRIPT__

async function apiSend(method, path, body) {
  sent.push({ method, path, body });
  return { ok: true, status: 200, json: {} };
}
async function loadTenants() {}
function loadPipeline() {}

let RESULT = null;

(async () => {
__SETUP__
  console.log("@@RESULT@@" + JSON.stringify(__CAPTURE__));
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
"""


def _run_node(setup_js: str, capture: str = "sent"):
    """Runs `setup_js` (plain statements, no `let`/`const` redeclaring
    anything the extracted script already declares -- reassign instead, e.g.
    `PIPE_CURRENT.tts = {...};`) after the extracted backoffice script in
    the stubbed node environment above, and returns the parsed JSON the
    harness prints: the list of {method, path, body} apiSend() was called
    with (capture="sent"), or whatever `setup_js` assigned to `RESULT`
    (capture="result")."""
    harness = (
        _HARNESS_TEMPLATE.replace("__SCRIPT__", _SCRIPT)
        .replace("__SETUP__", setup_js)
        .replace("__CAPTURE__", "RESULT" if capture == "result" else "sent")
    )
    proc = subprocess.run([NODE, "-e", harness], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"node harness failed:\n{proc.stderr}"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@") :])
    raise AssertionError(f"no result marker in node output:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}")


def _save(setup_js: str) -> dict:
    """Runs savePipeline("t1") with the given setup (DOM field values), and
    returns the single PATCH body apiSend() was called with. Fails if
    savePipeline sent zero or more than one request."""
    sent = _run_node(
        f"""
  {setup_js}
  await savePipeline("t1");
"""
    )
    assert len(sent) == 1, f"expected exactly one apiSend call, got {sent!r}"
    call = sent[0]
    assert call["method"] == "PATCH"
    assert call["path"] == "/api/v1/tenants/t1"
    return call["body"]


def _set(field_id: str, value: str) -> str:
    return f'$("{field_id}").value = {json.dumps(value)};'


@needs_node
def test_save_pipeline_male_voice_only_sets_male_voice_id() -> None:
    """(a) picking a Male voice produces chat_voice.tts.male_voice_id."""
    body = _save(_set("p_cv_tts_male_voice_select", "male-1"))
    assert body == {"pipeline": {"chat_voice": {"tts": {"male_voice_id": "male-1"}}}}


@needs_node
def test_save_pipeline_female_voice_only_sets_voice_id() -> None:
    """(b) picking a Female voice is just the existing cv_tts voice_id plumbing."""
    body = _save(_set("p_cv_tts_voice_select", "fem-1"))
    assert body == {"pipeline": {"chat_voice": {"tts": {"voice_id": "fem-1"}}}}


@needs_node
def test_save_pipeline_provider_female_and_male_in_one_save() -> None:
    """(c) picking a Chat TTS provider, Female and Male in one save."""
    body = _save(
        _set("p_cv_tts_provider", "elevenlabs")
        + _set("p_cv_tts_voice_select", "fem-3")
        + _set("p_cv_tts_male_voice_select", "male-3"),
    )
    assert body == {
        "pipeline": {"chat_voice": {"tts": {"provider": "elevenlabs", "voice_id": "fem-3", "male_voice_id": "male-3"}}}
    }


@needs_node
def test_save_pipeline_no_voice_touched_omits_both_fields() -> None:
    body = _save(_set("p_stt_language", "hi-IN"))
    assert body == {"pipeline": {"stt": {"language": "hi-IN"}}}


@needs_node
def test_resolve_voice_query_male_kind_uses_cv_tts_provider_and_language() -> None:
    """The Male picker's roster query always uses the Chat TTS layer's own
    provider/language (p_cv_tts_provider/p_cv_tts_language, PIPE_CURRENT.
    cv_tts) -- a static mapping, not a dynamic layer choice."""
    result = _run_node(
        """
  PIPE_CURRENT.cv_tts = {provider: "cv-tts-provider", language: "en-IN"};
  const before = resolveVoiceQuery("cv_tts_male");
  $("p_cv_tts_provider").value = "elevenlabs";
  $("p_cv_tts_language").value = "hi-IN";
  const after = resolveVoiceQuery("cv_tts_male");
  RESULT = [before, after];
""",
        capture="result",
    )
    before, after = result
    assert before == {"provider": "cv-tts-provider", "language": "en-IN"}
    assert after == {"provider": "elevenlabs", "language": "hi-IN"}


@needs_node
def test_female_and_male_pickers_list_only_their_own_gender() -> None:
    """The Female picker (cv_tts) lists only female voices and the Male picker
    (cv_tts_male) only male ones; a voice with no gender is in neither. The
    call TTS picker is unfiltered."""
    result = _run_node(
        """
  SELECTED = "t1";
  PIPE_CURRENT.cv_tts = {provider: "sarvam", language: "hi-IN"};
  PIPE_CURRENT.tts = {provider: "sarvam", language: "hi-IN"};
  VOICE_ROSTER_CACHE["sarvam|hi-IN"] = [
    {voice_id: "priya", gender: "female"},
    {voice_id: "aditya", gender: "male"},
    {voice_id: "mystery", gender: ""},
  ];
  await refreshVoiceOptions("cv_tts");
  await refreshVoiceOptions("cv_tts_male");
  await refreshVoiceOptions("tts");
  RESULT = {
    female: $("p_cv_tts_voice_select").innerHTML,
    male: $("p_cv_tts_male_voice_select").innerHTML,
    call: $("p_tts_voice_select").innerHTML,
  };
""",
        capture="result",
    )
    assert 'value="priya"' in result["female"]
    assert 'value="aditya"' not in result["female"]
    assert 'value="mystery"' not in result["female"]
    assert 'value="aditya"' in result["male"]
    assert 'value="priya"' not in result["male"]
    assert 'value="mystery"' not in result["male"]
    for v in ("priya", "aditya", "mystery"):
        assert f'value="{v}"' in result["call"]
