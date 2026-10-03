"""The Pipeline tab must show only the layered- or s2s-mode section relevant
to a tenant's current mode, not both stacked at once, and the chat-voice
heading must reflect that its TTS is also used for a call started from chat.

Mostly mirrors test_backoffice_events_webhook.py's style: reads
static/backoffice.html as a plain text fixture and does substring/slice
assertions -- no browser/JS execution. The gender-voice-picker save/layer-
choice logic near the bottom of this file is the exception: those tests
actually run the relevant JS (via node) against a stubbed DOM and inspect the
PATCH body produced, since a string-matching assertion on savePipeline()'s
source can't tell a correct layer-choice from a subtly wrong one.
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


# --- Gender voice pickers (female/male) ------------------------------------
# TenantTTSConfig.voices (src/config_tenant.py), surfaced on the pipeline tab
# so an operator can set the pair bot_gender picks between for chat
# voice-note replies and calls started from chat.

def _chat_voice_html_fn() -> str:
    start = HTML.index("function chatVoiceHtml(")
    return HTML[start:HTML.index("\n}\n", start)]


def test_gender_picker_kinds_rendered_in_chat_voice_html() -> None:
    """Both gender pickers must be wired through the shared catalog-backed
    picker machinery (voicePickerHtml), not hand-rolled selects -- that's
    what gives them a roster, a custom-entry fallback, and a "current" hint
    for free."""
    fn = _chat_voice_html_fn()
    assert 'voicePickerHtml("cv_tts_female"' in fn
    assert 'voicePickerHtml("cv_tts_male"' in fn


def test_gender_pickers_seeded_from_target_layers_current_voices() -> None:
    """The gender pickers' "current" hint must come from whichever layer a
    save would actually target right now (chatVoicesTargetsChatLayer's own
    "no live pick yet" reduction, `cv.source === "own"`) -- chat_voice.
    effective_voices (ChatVoiceInfo, src/api/tenants.py) when that's true,
    else the `ttsVoices` param (pipeline.tts's own raw stored pair, passed in
    by loadPipeline). Unconditionally seeding from effective_voices (the
    pre-fix behaviour) is wrong for a CASCADE tenant -- effective_voices is
    still populated there (resolved via the pipeline.tts fallback) but a
    save's `voices` pair in that state targets pipeline.tts, not
    chat_voice.tts, so the "current" shown next to a chat_voice.tts-framed
    picker would describe the wrong layer."""
    fn = _chat_voice_html_fn()
    assert "cv.effective_voices" in fn
    assert "ttsVoices" in fn
    assert "genderCurrentVoices.female" in fn
    assert "genderCurrentVoices.male" in fn
    # Not just referenced -- actually gated on source "own" (the no-live-pick
    # render-time reduction of chatVoicesTargetsChatLayer), not used bare.
    assert 'cv.source === "own"' in fn


def test_gender_remove_option_exists() -> None:
    """A gender picker must offer an explicit "remove this gender's voice"
    choice distinct from "leave unchanged" -- without it there is no way to
    clear a previously-set gender override from this UI, only overwrite it."""
    assert "— remove this gender's voice —" in HTML
    assert '"__remove__"' in HTML


def test_is_gender_voice_kind_covers_both_genders() -> None:
    start = HTML.index("function isGenderVoiceKind(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert '"cv_tts_female"' in fn
    assert '"cv_tts_male"' in fn


def test_gender_voice_field_value_distinguishes_remove_from_leave_unchanged() -> None:
    """genderVoiceFieldValue() is the function savePipeline() must use for
    the gender pickers instead of voiceFieldValue() -- it has to tell apart
    "leave unchanged" (omit the gender) from "remove" (send ""), which a
    plain voiceFieldValue() (two-state: omit or a value) cannot."""
    start = HTML.index("function genderVoiceFieldValue(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert "__remove__" in fn
    assert "return null" in fn


def test_gender_picker_field_ids_share_cv_tts_provider_and_language() -> None:
    """voicePickerFieldIds()/resolveVoiceQuery() must resolve the gender
    kinds' provider/language off the Chat TTS layer's own inputs
    (p_cv_tts_provider/p_cv_tts_language) and PIPE_CURRENT.cv_tts, not a
    separate per-gender pair of inputs that doesn't exist -- see
    voicePickerBaseKind."""
    start = HTML.index("function voicePickerBaseKind(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert '"cv_tts_female"' in fn
    assert '"cv_tts_male"' in fn
    assert '"cv_tts"' in fn

    ids_start = HTML.index("function voicePickerFieldIds(")
    ids_fn = HTML[ids_start:HTML.index("\n}\n", ids_start)]
    assert "voicePickerBaseKind(kind)" in ids_fn

    query_start = HTML.index("function resolveVoiceQuery(")
    query_fn = HTML[query_start:HTML.index("\n}\n", query_start)]
    assert "voicePickerBaseKind(kind)" in query_fn


def test_refresh_voice_options_cascades_cv_tts_to_gender_pickers() -> None:
    """refreshVoiceOptions("cv_tts") must also re-query both gender pickers
    -- they share the same provider/language inputs (voicePickerBaseKind), so
    anything that re-queries the base picker (a provider pick via
    onPipeProviderChange('cv_tts'), or the Chat TTS language input's
    onchange="refreshVoiceOptions('cv_tts')") must not leave the gender
    pickers showing a stale roster from the previous provider/language."""
    start = HTML.index("async function refreshVoiceOptions(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert 'refreshVoiceOptions("cv_tts_female")' in fn
    assert 'refreshVoiceOptions("cv_tts_male")' in fn
    assert 'kind === "cv_tts"' in fn

    # The existing onchange attributes (unchanged by this feature) are what
    # actually trigger the cascade above -- confirm they're still exactly
    # these, not renamed to some other wrapper function.
    assert 'id="p_cv_tts_provider" onchange="onPipeProviderChange(\'cv_tts\')"' in HTML
    assert 'id="p_cv_tts_language" placeholder="blank = leave unchanged" onchange="refreshVoiceOptions(\'cv_tts\')"' in HTML


def test_on_pipe_provider_change_updates_chat_voices_layer_hint() -> None:
    """A Chat TTS provider pick can flip which layer a gender `voices` pair
    saves to (chatVoicesTargetsChatLayer) -- onPipeProviderChange('cv_tts')
    must refresh the hint, not just the voice rosters."""
    start = HTML.index("function onPipeProviderChange(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert "updateChatVoicesLayerHint()" in fn


def test_load_pipeline_populates_gender_pickers() -> None:
    """loadPipeline() must refresh the gender pickers on initial render, same
    as it already does for the base cv_tts picker -- otherwise they'd sit
    empty until the operator manually touches the Chat TTS provider/language
    fields. It does so by calling refreshVoiceOptions("cv_tts"), which itself
    cascades to cv_tts_female/cv_tts_male (see the cascade test above) --
    loadPipeline's own source need not repeat those literal strings."""
    fn = _loadpipeline_fn()
    assert 'refreshVoiceOptions("cv_tts")' in fn
    assert "updateChatVoicesLayerHint()" in fn


# --- Layer-choice logic: pipeline.chat_voice.tts vs pipeline.tts -----------
# resolve_chat_tts_config (src/config_tenant.py) only reads
# pipeline.chat_voice.tts when it declares its OWN provider; a `voices` pair
# saved there while it has none is silently never used at runtime. The UI
# must route the pair to pipeline.tts in that case instead.
#
# The structural (string-matching) versions of this used to assert that
# savePipeline()'s source text merely *referenced* chatVoicesTargetsChatLayer/
# cvTts.voices/tts.voices -- which proves the code path exists, not that it
# produces the right PATCH body for any given tenant state. Replaced below by
# tests that actually RUN savePipeline() (via node) and inspect the body it
# sends apiSend().

def test_chat_voices_layer_hint_element_and_updater_exist() -> None:
    """A hint element under the gender pickers must say which layer the pair
    will be saved to (or why it can't be saved at all), and a JS function
    must keep it in sync with the call/Chat TTS provider picks (not just
    render a static string at load time that goes stale the moment the
    operator picks a provider)."""
    assert 'id="p_cv_voices_layer_hint"' in HTML
    start = HTML.index("function updateChatVoicesLayerHint(")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert "chatVoicesTargetsChatLayer" in fn
    assert "chatVoicesGenderPickersEnabled" in fn
    assert "p_cv_voices_layer_hint" in fn


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
    `PIPE_CHAT_VOICE_INFO = {...};`) after the extracted backoffice script in
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


def _save(cv: dict, setup_js: str) -> dict:
    """Runs savePipeline("t1") with PIPE_CHAT_VOICE_INFO set to `cv` and the
    given extra setup (DOM field values), and returns the single PATCH body
    apiSend() was called with. Fails if savePipeline sent zero or more than
    one request."""
    sent = _run_node(
        f"""
  PIPE_CHAT_VOICE_INFO = {json.dumps(cv)};
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
def test_save_pipeline_own_tenant_both_genders_targets_chat_voice_tts() -> None:
    """(a) own tenant, both genders -> chat_voice.tts.voices."""
    body = _save(
        {"source": "own", "effective_provider": "elevenlabs"},
        _set("p_cv_tts_female_voice_select", "fem-1") + _set("p_cv_tts_male_voice_select", "male-1"),
    )
    assert body == {"pipeline": {"chat_voice": {"tts": {"voices": {"female": "fem-1", "male": "male-1"}}}}}


@needs_node
def test_save_pipeline_cascade_tenant_male_only_targets_pipeline_tts() -> None:
    """(b) cascade tenant, male only -> tts.voices."""
    body = _save({"source": "cascade", "effective_provider": "sarvam"}, _set("p_cv_tts_male_voice_select", "male-2"))
    assert body == {"pipeline": {"tts": {"voices": {"male": "male-2"}}}}


@needs_node
def test_save_pipeline_remove_female_on_own_sends_empty_string() -> None:
    """(c) remove female on own -> {"female": ""} on chat_voice.tts."""
    body = _save({"source": "own", "effective_provider": "elevenlabs"}, _set("p_cv_tts_female_voice_select", "__remove__"))
    assert body == {"pipeline": {"chat_voice": {"tts": {"voices": {"female": ""}}}}}


@needs_node
def test_save_pipeline_cascade_plus_chat_provider_pick_targets_chat_voice_tts() -> None:
    """(d) cascade tenant + Chat TTS provider + both genders -> chat_voice.tts
    with that provider and the voices."""
    body = _save(
        {"source": "cascade", "effective_provider": "sarvam"},
        _set("p_cv_tts_provider", "elevenlabs")
        + _set("p_cv_tts_female_voice_select", "fem-3")
        + _set("p_cv_tts_male_voice_select", "male-3"),
    )
    assert body == {
        "pipeline": {"chat_voice": {"tts": {"provider": "elevenlabs", "voices": {"female": "fem-3", "male": "male-3"}}}}
    }


@needs_node
def test_save_pipeline_source_none_no_providers_sends_no_voices() -> None:
    """(e) source "none", no providers -> no `voices` anywhere in the body,
    even if a gender select somehow still carries a stray value (the
    pickers should be disabled in this state, but savePipeline() must not
    rely on that alone)."""
    body = _save(
        {"source": "none"},
        _set("p_mode", "s2s") + _set("p_cv_tts_female_voice_select", "stray-pick"),
    )
    assert body == {"pipeline": {"mode": "s2s"}}
    assert "voices" not in json.dumps(body)


@needs_node
def test_save_pipeline_source_none_plus_call_tts_provider_targets_pipeline_tts() -> None:
    """(f) source "none" + call TTS provider picked + female -> tts with
    provider and voices."""
    body = _save(
        {"source": "none"},
        _set("p_tts_provider", "sarvam") + _set("p_cv_tts_female_voice_select", "f-voice-3"),
    )
    assert body == {"pipeline": {"tts": {"provider": "sarvam", "voices": {"female": "f-voice-3"}}}}


@needs_node
def test_save_pipeline_no_voices_touched_omits_voices_key() -> None:
    """(g) a save that touches no voices -> no `voices` key, on either layer,
    regardless of source."""
    body = _save({"source": "cascade", "effective_provider": "sarvam"}, _set("p_stt_language", "hi-IN"))
    assert body == {"pipeline": {"stt": {"language": "hi-IN"}}}
    assert "voices" not in json.dumps(body)


@needs_node
def test_resolve_voice_query_gender_kind_follows_dynamic_base_layer() -> None:
    """(h) the dynamic roster base: for a cascade tenant,
    resolveVoiceQuery("cv_tts_female") returns the call TTS provider/
    language. Picking a Chat TTS provider flips it to the chat layer."""
    before, after = _run_node(
        """
  PIPE_CHAT_VOICE_INFO = {"source": "cascade"};
  PIPE_CURRENT.tts = {provider: "sarvam-tts-provider", language: "hi-IN", voices: null};
  PIPE_CURRENT.cv_tts = {provider: "", language: "en-IN"};
  const before = resolveVoiceQuery("cv_tts_female");
  $("p_cv_tts_provider").value = "elevenlabs-chat-provider";
  const after = resolveVoiceQuery("cv_tts_female");
  RESULT = [before, after];
""",
        capture="result",
    )
    assert before == {"provider": "sarvam-tts-provider", "language": "hi-IN"}
    assert after == {"provider": "elevenlabs-chat-provider", "language": "en-IN"}


# --- updateChatVoicesLayerHint(): gender pickers' live hint/disabled state --

@needs_node
def test_update_chat_voices_layer_hint_cascade_new_provider_hides_current() -> None:
    """(a) a cascade tenant whose pipeline.tts already has a `voices` pair,
    and whose (resolved-via-fallback) chat_voice.effective_voices happens to
    carry a stale value of its own -- picking a NEW Chat TTS provider flips
    the save target to chat_voice.tts, which has nothing of ITS OWN yet (it's
    about to be created by this save, not read back from). The female
    picker's current-hint must show no current voice ("--"), not `pf`
    (pipeline.tts's pair) and not the stale effective_voices value either."""
    result = _run_node(
        """
  PIPE_CHAT_VOICE_INFO = {"source": "cascade", "effective_voices": {"female": "stale-effective"}};
  PIPE_CURRENT.tts = {provider: "", language: "", voices: {female: "pf"}};
  $("p_cv_tts_provider").value = "elevenlabs";
  updateChatVoicesLayerHint();
  RESULT = $("p_cv_tts_female_voice_current_hint").textContent;
""",
        capture="result",
    )
    assert result == "(current: —)"


@needs_node
def test_update_chat_voices_layer_hint_own_tenant_shows_effective_voices() -> None:
    """(b) an "own" tenant's hint must show chat_voice.effective_voices."""
    result = _run_node(
        """
  PIPE_CHAT_VOICE_INFO = {"source": "own", "effective_voices": {"female": "ef", "male": "em"}};
  updateChatVoicesLayerHint();
  RESULT = {
    female: $("p_cv_tts_female_voice_current_hint").textContent,
    male: $("p_cv_tts_male_voice_current_hint").textContent,
  };
""",
        capture="result",
    )
    assert result == {"female": "(current: ef)", "male": "(current: em)"}


@needs_node
def test_update_chat_voices_layer_hint_cascade_no_provider_shows_pipeline_tts() -> None:
    """(c) a cascade tenant with no Chat TTS provider picked targets
    pipeline.tts -- the hint must show pipeline.tts's own stored `voices`
    pair (`pf`)."""
    result = _run_node(
        """
  PIPE_CHAT_VOICE_INFO = {"source": "cascade"};
  PIPE_CURRENT.tts = {provider: "", language: "", voices: {female: "pf"}};
  updateChatVoicesLayerHint();
  RESULT = $("p_cv_tts_female_voice_current_hint").textContent;
""",
        capture="result",
    )
    assert result == "(current: pf)"


@needs_node
def test_update_chat_voices_layer_hint_source_none_disables_then_enables() -> None:
    """(d) source "none" with no provider picked disables both gender
    selects (nothing resolves at runtime to save the pair onto); picking a
    call TTS provider and re-running enables them."""
    result = _run_node(
        """
  PIPE_CHAT_VOICE_INFO = {"source": "none"};
  updateChatVoicesLayerHint();
  const beforeFemale = $("p_cv_tts_female_voice_select").disabled;
  const beforeMale = $("p_cv_tts_male_voice_select").disabled;
  $("p_tts_provider").value = "sarvam";
  updateChatVoicesLayerHint();
  const afterFemale = $("p_cv_tts_female_voice_select").disabled;
  const afterMale = $("p_cv_tts_male_voice_select").disabled;
  RESULT = {beforeFemale, beforeMale, afterFemale, afterMale};
""",
        capture="result",
    )
    assert result == {"beforeFemale": True, "beforeMale": True, "afterFemale": False, "afterMale": False}
