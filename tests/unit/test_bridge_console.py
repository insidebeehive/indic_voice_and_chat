# tests/unit/test_bridge_console.py
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.bridge_console import page_router, router
from src.auth.middleware import set_admin_tokens

ADMIN_HEADERS = {"Authorization": "Bearer admin-token"}


@pytest.fixture(autouse=True)
def _admin_token():
    set_admin_tokens(["admin-token"])
    yield
    set_admin_tokens([])


def _page_client():
    app = FastAPI()
    app.include_router(page_router)
    return TestClient(app)


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_bridge_page_served_with_no_token():
    """The HTML shell carries no data — open, unauthenticated on page_router."""
    resp = _page_client().get("/dev/bridge")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


_ADMIN_GATED_ROUTES = [
    ("GET", "/dev/bridge/tenants"),
    ("POST", "/dev/bridge/place-call"),
]

_PLACE_CALL_BODY = {"provider": "twilio", "to_number": "+919999999999", "tenant": "dev"}


def _issue(client, method, path, headers=None):
    if method == "POST":
        return client.post(path, json=_PLACE_CALL_BODY, headers=headers)
    return client.get(path, headers=headers)


@pytest.mark.parametrize("method,path", _ADMIN_GATED_ROUTES)
def test_admin_gated_routes_401_without_token(method, path):
    resp = _issue(_client(), method, path)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", _ADMIN_GATED_ROUTES)
def test_admin_gated_routes_403_with_wrong_token(method, path):
    resp = _issue(_client(), method, path, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 403
