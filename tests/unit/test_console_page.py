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
async def test_backoffice_chat_analytics_shows_tokens_cost_and_cache_coverage() -> None:
    """Token/cost reporting (loadAnalytics' chatHtml block) must be present
    and must never let the cache-hit rate render as a bare percentage.

    NOTE: this is a served-HTML string match, not a browser — it cannot
    execute the JS or fetch /chat-analytics, so it cannot confirm that a real
    zero-coverage tenant actually renders "no cache data yet" at runtime.
    What it CAN and does confirm is that the *source* still contains the
    zero-coverage branch, in the right shape, so a code change that deletes
    or short-circuits that branch is caught here even though no test in this
    suite drives a browser against the page."""
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    body = resp.text

    # Totals + per-model breakdown are wired up.
    assert "ca.total_input_tokens" in body
    assert "ca.total_output_tokens" in body
    assert "ca.total_cost" in body
    assert "ca.by_model" in body
    assert "escT(m.llm_provider)" in body and "escT(m.llm_model)" in body

    # Cost is explicitly labeled as a local estimate, not provider billing.
    assert "provider_costs" in body and "not the provider" in body.lower()

    # The cache-hit rate must be a ternary gated on cache_metrics_turns > 0,
    # with pct1(...) only in the true branch and an explicit "no data"
    # fallback in the false branch -- a bare `pct1(ca.cache_hit_rate_pct)`
    # rendered unconditionally would fail this regex (it requires the
    # `> 0 ? ... pct1 ... : ... no cache data yet` shape).
    import re
    assert re.search(
        r"ca\.cache_metrics_turns\s*>\s*0\s*\?\s*`\$\{pct1\(ca\.cache_hit_rate_pct\)\}.*?"
        r":\s*`<span[^`]*no cache data yet</span>`",
        body, re.S,
    ), "cache-hit rate must be gated on cache_metrics_turns, with an explicit no-data fallback"


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


@pytest.mark.asyncio
async def test_backoffice_deposit_verification_tab_wired_up() -> None:
    """The Deposit verification tab/pane must follow the same pattern as
    every other tab (tab_x/pane_x pair, key present in showTab's list, a
    load call from selectTenant) — before this there was no UI for the
    feature anywhere (`grep deposit_verification static/` returned nothing),
    which is why an operator trying to enable it found nothing to save.

    This only confirms the wiring exists in the served markup/source; it
    cannot execute the JS to confirm clicking the tab actually shows the
    pane at runtime — that remains unverified by any test in this suite.
    """
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    body = resp.text

    assert '<div class="tab" id="tab_dv" onclick="showTab(\'dv\')">' in body
    assert '<div id="pane_dv" hidden></div>' in body
    # showTab's key list must include "dv", or clicking every OTHER tab would
    # leave pane_dv's `hidden` untouched — it would never hide once shown, and
    # its tab would never highlight `.active`.
    assert '["an", "pipe", "crm", "camp", "chat", "kb", "bill", "tel", "dv"]' in body
    # selectTenant must actually populate the pane when a tenant is picked —
    # a wired-up tab with nothing feeding it renders forever empty.
    assert "loadDepositVerification(id);" in body
    assert "function loadDepositVerification(id)" in body


@pytest.mark.asyncio
async def test_backoffice_deposit_verification_edit_form_partial_override() -> None:
    """Every edit control must default to a "leave unchanged" sentinel
    (matching the Pipeline tab's discipline) so an untouched field never
    overwrites its stored value, and the secret field must be write-only:
    type=password, never pre-filled from a prior value, and never echoed
    back into the page after a save.

    This only confirms the controls and their defaults/attributes exist in
    the served source; it cannot execute the JS to confirm a real save
    actually omits untouched fields from the PATCH body at runtime, nor that
    the secret input truly never receives a value — that remains unverified
    by any test in this suite.
    """
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    body = resp.text

    # enabled: leave-unchanged / enabled / disabled tri-state select.
    assert '<select id="dv_enabled">' in body
    assert '<option value="">— leave unchanged —</option>\n          <option value="true">enabled</option>' in body

    # webhook_url: plain text, blank = leave unchanged, no value= attribute
    # (never pre-filled from the stored URL — matching the Pipeline tab's
    # inputs, which show the current value only as a hint, not as a prefill).
    assert 'id="dv_webhook_url"' in body
    assert 'id="dv_webhook_url" style="width:22rem" placeholder="blank = leave unchanged" /' in body

    # webhook_secret: write-only. type=password, no value attribute anywhere
    # near it, and the raw string "webhook_secret" never appears as something
    # being read back out of a GET response (only ever written from the form).
    # Assert the COMPLETE tag literally, not just a prefix ending before
    # where a value= attribute would go — a prefix match (or the narrower
    # 'value="${t.deposit_verification' check that used to stand here) still
    # passes if a value= attribute is appended, however it's spelled or
    # wrapped (e.g. value="${escAttr(...)}"). Proven by mutation: appending
    # ` value="${escAttr(t.x)}"` to this tag in the source fails this exact
    # match, whereas both of the old assertions kept passing against it.
    assert '<input id="dv_webhook_secret" type="password"' in body
    assert ('<input id="dv_webhook_secret" type="password" style="width:18rem" '
            'placeholder="blank = leave unchanged" /></label>') in body

    # contract: leave-unchanged / multipart_verdict / json_ticket_relay — the
    # two contracts are independent and incompatible, and a mismatch is not
    # caught when you save: it surfaces later as a rejected inbound callback
    # logged with reason="wrong_contract" (src/api/deposit_verification.py:307).
    assert '<select id="dv_contract">' in body
    assert '<option value="multipart_verdict">multipart_verdict</option>' in body
    assert '<option value="json_ticket_relay">json_ticket_relay</option>' in body
    assert "a mismatch is not caught when you save" in body

    # timeout_minutes / screenshot_url_ttl_seconds: numeric, both documented
    # as > 0 (DepositVerificationConfig's gt=0 constraints).
    assert 'id="dv_timeout_minutes" type="number"' in body
    assert 'id="dv_ttl_seconds" type="number"' in body

    # mobile_metadata_keys: comma-separated, scoped to json_ticket_relay only.
    assert 'id="dv_mobile_keys"' in body
    assert "json_ticket_relay</code> only" in body

    # Save wiring + failure handling matches savePipeline's pattern: PATCH the
    # tenant with a deposit_verification block, render errors via
    # formatApiError in a warnbox, and leave the form populated on failure.
    assert "function saveDepositVerification(id)" in body
    assert 'apiSend("PATCH", `/api/v1/tenants/${id}`, { deposit_verification: dv })' in body
    assert "formatApiError(r.json)" in body

    # Partial-override discipline, asserted against the actual JS guards
    # rather than just the controls' HTML defaults above (which say nothing
    # about what saveDepositVerification does with a value once read). Each
    # field must stay behind its own `if (...)` truthiness check so a blank
    # or untouched control is never sent in the PATCH body — dropping the
    # `enabled` guard in particular would mean a blank/leave-unchanged
    # webhook_secret control sends `webhook_secret: ""` on every save,
    # silently overwriting a working live secret. Proven by mutation: making
    # any one of these three unconditional (dropping its `if (...) ` prefix)
    # in the source fails exactly that assertion while the other two and
    # every assertion above still pass.
    assert 'if (enabled) dv.enabled = enabled === "true";' in body
    assert 'if (webhookUrl) dv.webhook_url = webhookUrl;' in body
    assert 'if (webhookSecret) dv.webhook_secret = webhookSecret;' in body


@pytest.mark.asyncio
async def test_backoffice_deposit_verification_enabled_but_inert_warning() -> None:
    """The single most valuable thing this UI must do: make "enabled but
    inert" impossible to miss. `enabled: true` with no webhook_url and/or no
    resolvable secret must render a prominent warning naming exactly which
    requirement is missing and stating the tool is not registered — an
    `enabled: true` tenant must never look like the feature is working.

    This only confirms the warning branch and its wording exist in the
    served JS source; it cannot execute the script to confirm the warning
    actually renders (versus the "active"/"off" branches) for a given
    tenant's real field values at runtime — that remains unverified by any
    test in this suite.
    """
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/admin/tenants")
    body = resp.text

    assert "function dvStateHtml(t)" in body
    # Three distinct states, not a single generic on/off render.
    assert "is <b>off</b> for this tenant" in body
    assert "is <b>on and active</b>" in body
    assert "Enabled but INERT" in body
    # Names the exact missing requirement(s) rather than a generic "broken".
    assert '"a webhook URL"' in body
    assert '"a resolvable webhook secret"' in body
    # States the consequence in plain terms — not just "misconfigured".
    assert "the tool is <b>not registered</b> and this feature does <b>nothing</b>" in body
