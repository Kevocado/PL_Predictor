"""The /api/explain/{sport}/{id}/context proxy: same safety rules as the summary route."""
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pl_predictor.api import explain as explain_module

SECRET = "sk-or-v1-SECRET"
FIXED = {"detail": "The summary service is not available."}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.text = SECRET
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(explain_module.router)
    return TestClient(app, raise_server_exceptions=False)


def _capture(monkeypatch, response):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(explain_module.requests, "get", fake_get)
    return calls


def test_context_is_not_swallowed_by_the_catch_all(client, monkeypatch):
    """The catch-all `{explainer_id:path}` would match `<id>/context` and forward
    it to the SUMMARY route; the context route must win."""
    body = {"matchups": [], "form_rows": [], "player_context": {}}
    calls = _capture(monkeypatch, FakeResponse(200, body))
    res = client.get("/api/explain/pl/12345/context")
    assert res.status_code == 200
    assert res.json() == body
    assert [u for u, _ in calls] == [f"{explain_module.EXPLAINER_URL}/explain/pl/12345/context"]
    # and the summary route is still the summary route
    calls.clear()
    client.get("/api/explain/pl/12345")
    assert calls[0][0] == f"{explain_module.EXPLAINER_URL}/explain/pl/12345"


def test_timeout_constant_and_no_redirects(client, monkeypatch):
    calls = _capture(monkeypatch, FakeResponse(200, {}))
    client.get("/api/explain/pl/12345/context")
    kwargs = calls[0][1]
    assert kwargs["timeout"] == explain_module.EXPLAINER_TIMEOUT_S
    assert kwargs["allow_redirects"] is False


def test_segments_are_quoted(client, monkeypatch):
    calls = _capture(monkeypatch, FakeResponse(200, {}))
    client.get("/api/explain/pl/a%20b/c/context")
    assert calls[0][0].endswith("/explain/pl/a%20b%2Fc/context")


@pytest.mark.parametrize("status", [301, 404, 500, 502])
def test_non_2xx_is_the_fixed_502_without_the_upstream_body(client, monkeypatch, status):
    _capture(monkeypatch, FakeResponse(status, {"detail": SECRET}))
    res = client.get("/api/explain/pl/12345/context")
    assert res.status_code == 502
    assert res.json() == FIXED
    assert SECRET not in res.text


@pytest.mark.parametrize("exc", [requests.ConnectionError("x " + SECRET), requests.Timeout("t")])
def test_a_raised_error_is_the_fixed_502(client, monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(explain_module.requests, "get", boom)
    res = client.get("/api/explain/pl/12345/context")
    assert res.status_code == 502
    assert res.json() == FIXED


@pytest.mark.parametrize("payload", [["a"], "s", 3, None, requests.JSONDecodeError("x", "y", 0), ValueError("bad")])
def test_a_2xx_that_is_not_a_json_object_is_the_fixed_502(client, monkeypatch, payload):
    _capture(monkeypatch, FakeResponse(200, payload))
    res = client.get("/api/explain/pl/12345/context")
    assert res.status_code == 502
    assert res.json() == FIXED


@pytest.mark.parametrize("sport,explainer_id", [
    ("pl", "secrets/../x"),
    ("pl", "../x"),
    ("pl", "/etc/passwd"),
    ("pl", ""),
    ("pl/x", "12345"),
    ("..", "12345"),
])
def test_refused_inputs_never_reach_the_explainer(monkeypatch, sport, explainer_id):
    calls = _capture(monkeypatch, FakeResponse(200, {}))
    with pytest.raises(Exception) as excinfo:
        explain_module.explain_context(sport, explainer_id)
    assert getattr(excinfo.value, "status_code", None) == 502
    assert excinfo.value.detail == FIXED["detail"]
    assert calls == []


def test_walk_outs_through_http_never_reach_the_explainer(client, monkeypatch):
    calls = _capture(monkeypatch, FakeResponse(200, {}))
    for path in ("/api/explain/pl/..%2Fx/context", "/api/explain/pl/%2Fetc%2Fpasswd/context",
                 "/api/explain/pl//context"):
        assert client.get(path).status_code in (404, 502)
    assert calls == []
