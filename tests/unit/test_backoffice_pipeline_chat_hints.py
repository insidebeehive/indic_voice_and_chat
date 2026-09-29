"""Backoffice visibility hints added by the tenant-registration/backoffice
audit (B3/V1/V2/V3):

- V1: the Pipeline tab's LLM layer says chat replies never use it (chat
  always runs on the platform LLM -- src/auth/registry.py get_platform_llm).
- V2: the Chat tab shows chat_support.support_hours read-only, with a clear
  hint when it's empty (empty == "always available" handoffs, 24x7).
- V3: the provider-source badge ("(default)" / "platform default") next to
  each pipeline layer -- this one already existed before the audit; these
  tests just pin that it still does.

Same style as test_backoffice_events_webhook.py: read the static HTML as
text and assert on the substrings/functions the page actually contains.
"""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "static" / "backoffice.html").read_text(encoding="utf-8")


def test_llm_layer_has_chat_platform_model_hint() -> None:
    """V1: chat ignores pipeline.llm entirely -- the Pipeline tab's LLM
    layer must say so next to the picker, not leave an operator to assume
    changing it affects chat replies too."""
    assert 'Used for voice calls only — chat replies use the platform model.' in HTML
    start = HTML.index('pipeLayerHtml("llm", "LLM", llm')
    end = HTML.index(")}", start)
    assert "Used for voice calls only" in HTML[start:end]


def test_provider_source_badge_shown_for_every_layer() -> None:
    """V3: _layer()'s provider_source/model_source (src/api/tenants.py)
    must actually reach the page next to each layer -- the tenant list
    row (lay()) and the Pipeline tab editor hint (layHint(), via
    pipeLayerHtml) both read it."""
    assert "l.provider_source" in HTML
    assert "l.model_source" in HTML
    for kind, label in (("stt", "STT"), ("llm", "LLM"), ("tts", "TTS (call)"), ("realtime", "Realtime")):
        assert f'pipeLayerHtml("{kind}", "{label}"' in HTML
    # Every pipeLayerHtml call renders layHint(current) inside its "(current: ...)" label.
    assert "(current: ${layHint(current)})" in HTML


def test_support_hours_shown_read_only_with_empty_hint() -> None:
    """V2: chat_support.support_hours has no PATCH route, so the Chat tab
    only displays it (read-only) -- and must say plainly what an empty
    value means, since empty is "always available", not "unset/off"."""
    assert "function supportHoursHtml(cs)" in HTML
    start = HTML.index("function supportHoursHtml(cs)")
    end = HTML.index("\n}\n", start)
    fn = HTML[start:end]
    assert "No support hours set" in fn
    assert "24" in fn  # "24×7" -- always-available framing
    # Rendered into the Chat tab (loadChat's pane_chat template), fed from
    # GET .../chat-config's chat_support block.
    assert "supportHoursHtml(cfg.chat_support)" in HTML
    assert 'Support hours <small class="hint">(back-office handoff availability — read-only)</small>' in HTML
