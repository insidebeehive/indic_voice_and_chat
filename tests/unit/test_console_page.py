"""The /console (tenant) and /admin (admin) pages are served (no infra)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@asynccontextmanager
async def _no_lifespan(app):
    yield


def _app():
    from src import main as main_module

    app = FastAPI(lifespan=_no_lifespan)
    app.add_api_route("/console", main_module.api_console, methods=["GET"])
    app.add_api_route("/admin", main_module.admin_console, methods=["GET"])
    app.add_api_route("/admin/tenants", main_module.backoffice, methods=["GET"])
    return app


@pytest.mark.asyncio
async def test_tenant_console_served() -> None:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/console")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "Vox Tenant Console" in body
    # Tenant flows present; admin-only register/models absent.
    assert "/api/v1/campaigns" in body
    assert "/api/v1/calls/" in body
    assert "/api/v1/voices" in body
    assert "/api/v1/tenants" not in body
    assert 'href="/admin"' in body          # cross-link to the admin page


@pytest.mark.asyncio
async def test_admin_console_served() -> None:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin")
    assert resp.status_code == 200
    body = resp.text
    assert "Vox Admin Console" in body
    # Admin flows: register tenant + model catalog + provider costs.
    assert "/api/v1/tenants" in body
    assert "/api/v1/models" in body
    assert "/api/v1/providers/" in body
    assert 'href="/console"' in body        # cross-link to the tenant page


@pytest.mark.asyncio
async def test_backoffice_served() -> None:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    assert resp.status_code == 200
    body = resp.text
    assert "Vox Backoffice" in body
    assert "/api/v1/tenants" in body          # tenant list
    assert "/analytics" in body and "/billing" in body


@pytest.mark.asyncio
async def test_backoffice_voice_pickers_replace_free_text() -> None:
    """The three voice fields (chat TTS, call TTS, realtime) must be
    catalog-backed <select> pickers, not free-text inputs an operator has to
    know an exact (possibly deprecated) voice id for — see the voice_catalog
    drift the pipeline editor used to be blind to."""
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    body = resp.text

    # The old free-text voice_id/voice inputs are gone.
    assert 'id="p_cv_tts_voice_id"' not in body
    assert 'id="p_tts_voice_id"' not in body
    assert 'id="p_realtime_voice"' not in body

    # Each layer wires up the shared picker (ids are built from `kind` at
    # render time via voicePickerHtml/voicePickerFieldIds — see p_${kind}_*
    # in static/backoffice.html — so the served source names the kind, not
    # the literal element id).
    for kind in ("cv_tts", "tts", "realtime"):
        assert f'voicePickerHtml("{kind}"' in body

    # Shared fetch/render logic lives once, not copy-pasted per picker.
    assert body.count("function refreshVoiceOptions(") == 1
    assert body.count("function voicePickerHtml(") == 1
    assert "custom / other" in body   # custom-entry escape hatch for cloned/unlisted voices

    # It queries the real voice catalog endpoint.
    assert "/api/v1/voices?provider=" in body

    # Each of the three language inputs re-triggers the fetch on change —
    # sarvam/azure/google/indicf5 rosters genuinely differ per language, so a
    # missing handler on any one of these silently freezes that picker's
    # roster to whatever language it first loaded with.
    assert 'onchange="refreshVoiceOptions(\'tts\')"' in body
    assert 'onchange="refreshVoiceOptions(\'cv_tts\')"' in body
    assert 'onchange="refreshVoiceOptions(\'realtime\')"' in body

    # savePipeline must send the field name each layer's config actually
    # uses: `voice_id` for tts/chat-voice TTS (TenantTTSConfig), `voice` for
    # realtime (TenantRealtimeConfig) — mixing these up would silently no-op
    # the PATCH for whichever layer got the wrong name.
    assert "tts.voice_id = ttsVoice" in body
    assert "cvTts.voice_id = cvVoice" in body
    assert "realtime.voice = realtimeVoice" in body


@pytest.mark.asyncio
async def test_backoffice_hidden_attribute_is_defended_at_stylesheet_level() -> None:
    """The UA stylesheet's ``[hidden]{display:none}`` loses to any
    author-origin ``display`` rule — and this page has one (``label { ...
    display:flex... }``) that applies to the custom-voice-id <label>, which
    also carries `hidden`. Without an author-origin ``[hidden]`` override,
    that label renders regardless of the `hidden` attribute/`.hidden`
    property, and the custom voice id field is always visible next to the
    picker's "— leave unchanged —" select — see voiceFieldValue, which
    silently discards it in that state.

    This only checks the CSS rule is served; it cannot execute JS/CSSOM to
    confirm the label is actually invisible in a real browser — that remains
    unverified by any test in this suite.
    """
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    body = resp.text

    # The actual rule (not just a mention of `[hidden]` in a comment, which
    # this file's own explanatory comment about the UA stylesheet contains).
    # Must win over `label { ... display:flex ... }` and any future
    # author-origin display rule, not just the ones that exist today, hence
    # !important rather than mere specificity.
    assert "[hidden] { display: none !important; }" in body

    # The label this bug was found on still carries `hidden` (i.e. the fix is
    # the stylesheet rule, not switching this element away from `hidden`).
    assert 'id="p_${kind}_voice_custom_wrap" hidden' in body


@pytest.mark.asyncio
async def test_backoffice_dropped_selection_warning_is_present_in_source() -> None:
    """refreshVoiceOptions must not silently revert a non-empty prior voice
    pick to the sentinel when the new provider/language roster doesn't
    contain it (e.g. switching a tenant's TTS language to one with a
    disjoint roster) — the operator must be told, by name, which voice was
    dropped, and the message must not be suppressed just because the new
    roster is non-empty.

    This only confirms the warning branch and its wording are present in the
    served JS source; it cannot execute the script to confirm the message
    actually appears (and the sentinel actually gets selected) at runtime —
    that remains unverified by any test in this suite. In particular it does
    NOT catch branch *ordering*: reordering the message branches so the
    success case clears the message first reintroduces the original bug
    verbatim while every assertion here still passes (verified by mutation).
    Ordering is the thing that broke; source-matching cannot see it.
    """
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    body = resp.text

    assert "keepValueSelectable" in body
    assert "previously selected voice" in body
    # The warning must be computed from whether the prior pick survived into
    # the new roster, not just "roster is empty" — the Azure hi-IN → mr-IN
    # case the bug was reported against has a non-empty but disjoint roster.
    assert "!keepValueSelectable" in body
