"""The explainer proxy: it must forward, and it must fail honestly."""
from __future__ import annotations

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pl_predictor.api import explain as explain_mod


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(explain_mod.router)
    return TestClient(app)


def test_forwards_to_the_services_own_route(monkeypatch, client):
    """The site calls /api/explain/pl/<id>; the service serves /explain/pl/<id>."""
    seen = {}

    def fake_get(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return _resp(200, {"headline": "h", "sections": [], "source": "template", "pick_timing": "pre_kickoff"})

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)
    body = client.get("/api/explain/pl/12345")
    assert body.status_code == 200
    assert body.json()["headline"] == "h"
    assert seen["url"].endswith("/explain/pl/12345")


def test_forwards_a_path_id_whole(monkeypatch, client):
    """An F1-style id is season-round-session, but ids may carry slashes; the
    path must arrive intact rather than truncated at the first segment."""
    seen = {}

    def fake_get(url, timeout=None):
        seen["url"] = url
        return _resp(200, {"headline": "h"})

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)
    client.get("/api/explain/f1/2026-12-race")
    assert seen["url"].endswith("/explain/f1/2026-12-race")


def test_never_returns_a_key(monkeypatch, client):
    """The service holds the OpenRouter key; this route must not echo settings
    that could carry it, and its own error text is a fixed string."""
    monkeypatch.setattr(explain_mod.requests, "get", lambda url, timeout=None: _resp(500, {"detail": "boom"}))
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 502
    assert "key" not in body.text.lower()
    assert body.json()["detail"] == "The summary service is not available."


def test_an_unreachable_explainer_is_a_502_not_a_crash(monkeypatch, client):
    def boom(url, timeout=None):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(explain_mod.requests, "get", boom)
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 502
    assert body.json()["detail"] == "The summary service is not available."


def test_a_timeout_is_a_502_too(monkeypatch, client):
    def slow(url, timeout=None):
        raise requests.Timeout("too slow")

    monkeypatch.setattr(explain_mod.requests, "get", slow)
    assert client.get("/api/explain/pl/1").status_code == 502


def test_no_summary_is_a_404_the_site_can_swallow(monkeypatch, client):
    monkeypatch.setattr(explain_mod.requests, "get", lambda url, timeout=None: _resp(404, {"detail": "nope"}))
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 404
    assert body.json()["detail"] == "No summary for this fixture."


def test_the_timeout_is_configurable(monkeypatch, client):
    monkeypatch.setattr(explain_mod, "EXPLAINER_TIMEOUT_S", 7.5)
    seen = {}
    monkeypatch.setattr(
        explain_mod.requests, "get", lambda url, timeout=None: (seen.update(t=timeout), _resp(200, {}))[1]
    )
    client.get("/api/explain/pl/1")
    assert seen["t"] == 7.5


def _resp(status: int, body: dict):
    class R:
        status_code = status
        ok = status < 400

        def json(self):
            return body

    return R()
