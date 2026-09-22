"""Route-level tests for /api/v1/webhooks/* endpoints.

Every route is admin-gated at the router level (``dependencies=[Depends(require_admin)]``
in src/api/webhooks_routes.py — Fix 2 of the 2026-09 webhook-auth remediation:
these routes have no tenant scoping at all (``WebhookManager.register()`` is
process-global), so ``require_admin`` is the only coherent gate). Every call
below carries ``ADMIN_HEADERS``; the rejection tests assert the gate holds
with no such header.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import webhooks_routes
from src.auth.middleware import set_admin_tokens
from src.integration.webhooks import WebhookManager

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest.fixture
def app():
    manager = WebhookManager()
    webhooks_routes.set_webhook_manager(manager)
    set_admin_tokens(["admin-token"])
    a = FastAPI()
    a.include_router(webhooks_routes.router)
    yield a
    webhooks_routes.set_webhook_manager(None)
    set_admin_tokens([])


def test_register_webhook(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/webhooks", headers=ADMIN_HEADERS, json={
        "url": "https://example.com/webhook",
        "event_filters": ["call.*"],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://example.com/webhook"
    assert body["event_filters"] == ["call.*"]
    assert body["id"].startswith("wh_")


def test_list_webhooks(app: FastAPI) -> None:
    client = TestClient(app)
    client.post("/webhooks", headers=ADMIN_HEADERS, json={"url": "https://a.example", "event_filters": ["*"]})
    client.post("/webhooks", headers=ADMIN_HEADERS, json={"url": "https://b.example", "event_filters": ["call.completed"]})
    resp = client.get("/webhooks", headers=ADMIN_HEADERS)
    body = resp.json()
    assert body["total"] == 2


def test_delete_webhook(app: FastAPI) -> None:
    client = TestClient(app)
    reg = client.post("/webhooks", headers=ADMIN_HEADERS, json={"url": "https://x"}).json()
    resp = client.delete(f"/webhooks/{reg['id']}", headers=ADMIN_HEADERS)
    assert resp.status_code == 200
    after = client.get("/webhooks", headers=ADMIN_HEADERS).json()
    assert after["total"] == 0


def test_delete_unknown_404(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.delete("/webhooks/nope", headers=ADMIN_HEADERS)
    assert resp.status_code == 404


def test_routes_503_when_manager_unset() -> None:
    webhooks_routes.set_webhook_manager(None)
    set_admin_tokens(["admin-token"])
    a = FastAPI()
    a.include_router(webhooks_routes.router)
    client = TestClient(a)
    try:
        assert client.get("/webhooks", headers=ADMIN_HEADERS).status_code == 503
    finally:
        set_admin_tokens([])


# --- Fix 2: admin gate -------------------------------------------------


def test_register_webhook_rejects_without_admin_token(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post("/webhooks", json={"url": "https://example.com/webhook"})
    assert resp.status_code == 401


def test_list_webhooks_rejects_without_admin_token(app: FastAPI) -> None:
    client = TestClient(app)
    assert client.get("/webhooks").status_code == 401


def test_delete_webhook_rejects_without_admin_token(app: FastAPI) -> None:
    client = TestClient(app)
    reg = client.post("/webhooks", headers=ADMIN_HEADERS, json={"url": "https://x"}).json()
    resp = client.delete(f"/webhooks/{reg['id']}")
    assert resp.status_code == 401


def test_register_webhook_rejects_wrong_admin_token(app: FastAPI) -> None:
    client = TestClient(app)
    resp = client.post(
        "/webhooks", headers={"Authorization": "Bearer not-the-admin-token"},
        json={"url": "https://example.com/webhook"},
    )
    assert resp.status_code == 403
