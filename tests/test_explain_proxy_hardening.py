"""The explainer proxy's error contract, including the cases it must not leak.

The proxy turns an upstream failure into one fixed 502 so an upstream body —
which could carry key material or an internal path — never reaches the browser.
These tests pin every status the upstream can produce, and the ones a rewrite
must keep pinned.
"""
from __future__ import annotations

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pl_predictor.api import explain as explain_mod

SECRET = "sk-or-v1-SECRET"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(explain_mod.router)
    return TestClient(app)


def _resp(status: int, body: dict):
    class R:
        status_code = status
        ok = 200 <= status < 300
        text = str(body)

        def json(self):
            if status == 204:
                raise ValueError("no body")
            return body

    return R()


def _not_json():
    class R:
        status_code = 200
        ok = True
        text = "<html>not json</html>"

        def json(self):
            raise ValueError("Expecting value")

    return R()


def test_an_upstream_500_body_is_never_echoed(monkeypatch, client):
    monkeypatch.setattr(explain_mod.requests, "get", lambda url, **kw: _resp(500, {"detail": SECRET}))
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 502
    assert SECRET not in body.text


def test_a_3xx_is_a_failure_not_a_forwarded_body(monkeypatch, client):
    """A 3xx falls outside `ok`, so an unguarded `return response.json()` would
    hand the browser whatever the redirect body said."""
    monkeypatch.setattr(explain_mod.requests, "get", lambda url, **kw: _resp(302, {"detail": SECRET}))
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 502
    assert SECRET not in body.text


def test_redirects_are_not_followed(monkeypatch, client):
    """Following a redirect would let a misconfigured EXPLAINER_URL post this
    service's request at another host entirely."""
    seen = {}

    def fake_get(url, **kwargs):
        seen["kwargs"] = kwargs
        return _resp(200, {"headline": "h"})

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)
    client.get("/api/explain/pl/1")
    assert seen["kwargs"].get("allow_redirects") is False


def test_a_non_json_upstream_200_is_a_502_not_a_500(monkeypatch, client):
    """response.json() outside the try turns a bad upstream body into a bare
    500, which the site's error state cannot tell from a crash."""
    monkeypatch.setattr(explain_mod.requests, "get", lambda url, **kw: _not_json())
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 502
    assert body.json()["detail"] == "The summary service is not available."


def test_the_sport_segment_cannot_walk_out_of_the_route(monkeypatch, client):
    """`sport` is interpolated into the upstream path, so a value like
    `../admin` would rewrite the request. Reject it before any request goes out."""
    seen = {}
    monkeypatch.setattr(
        explain_mod.requests, "get", lambda url, **kw: (seen.update(url=url), _resp(200, {}))[1]
    )
    res = client.get("/api/explain/..%2Fadmin/1")
    assert res.status_code == 502
    assert "url" not in seen


def test_an_id_cannot_walk_out_of_the_explain_route(monkeypatch, client):
    seen = {}
    monkeypatch.setattr(
        explain_mod.requests, "get", lambda url, **kw: (seen.update(url=url), _resp(200, {}))[1]
    )
    res = client.get("/api/explain/pl/..%2F..%2Fstatus")
    assert res.status_code == 502
    assert "url" not in seen


def test_an_unreachable_explainer_is_a_502(monkeypatch, client):
    def boom(url, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(explain_mod.requests, "get", boom)
    assert client.get("/api/explain/pl/1").status_code == 502


def test_the_timeout_is_below_the_browsers(monkeypatch):
    """The browser gives up at 20 s for this site. A sync proxy route holds a
    worker thread for its whole duration, so it must not wait longer than the
    client it serves — otherwise the slowest tier is the one nobody waits for.
    """
    assert explain_mod.EXPLAINER_TIMEOUT_S < 20
