"""The backoffice events-webhook form must use the fields the API actually has.

It used to live on the Telephony tab and read ``telephony_events_webhook_url``
/ ``telephony_events_webhook_secret_set``, which the tenant list never returns
(so it always showed "not set"), and sent the URL inside ``telephony``, where
PATCH /tenants ignores it. The request shape it must match is pinned by
test_tenants_routes.py::test_update_tenant_stores_events_webhook_secret.
"""
from __future__ import annotations

from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "static" / "backoffice.html").read_text(encoding="utf-8")


def test_events_webhook_form_reads_real_tenant_fields() -> None:
    assert "telephony_events_webhook" not in HTML
    assert "tInfo.events_webhook_url" in HTML
    assert "tInfo.events_webhook_secret_set" in HTML


def test_events_webhook_form_sends_url_top_level_and_secret_in_keys() -> None:
    start = HTML.index("async function saveEventsWebhook")
    fn = HTML[start:HTML.index("\n}\n", start)]
    assert "body.events_webhook_url = url" in fn
    assert "body.telephony = { keys: { events_webhook_secret: secret } }" in fn
    assert "`/api/v1/tenants/${id}`, body)" in fn
