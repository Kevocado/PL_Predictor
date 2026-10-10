"""The explainer proxy's error contract, including the cases it must not leak.

The proxy turns an upstream failure into one fixed 502 so an upstream body —
which could carry key material or an internal path — never reaches the browser.
These tests pin every status the upstream can produce, and the ones a rewrite
must keep pinned.
"""
from __future__ import annotations

import importlib
import json

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pl_predictor.api import explain as explain_mod

SECRET = "sk-or-v1-SECRET"

GAME_ID = "777"

UNAVAILABLE_LITERAL = "The summary service is not available."

UPSTREAM_BODY = {
    "verdict": "ARSENAL by 2.1 goals, against a line of ARSENAL -1.5.",
    "factors": [{"key": "spread", "text": "It rates ARSENAL 2.1 goals better."}],
}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(explain_mod.router)
    return TestClient(app)


def _resp(status: int, body: dict):
    """Build a real requests.Response with the given status and JSON body."""
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode()
    return response


def _not_json():
    """Build a real requests.Response with 200 status but non-JSON body."""
    response = requests.Response()
    response.status_code = 200
    response._content = b"<html>not json</html>"
    return response


def test_an_upstream_500_body_is_never_echoed(monkeypatch, client):
    monkeypatch.setattr(explain_mod.requests, "get", lambda url, **kw: _resp(500, {"detail": SECRET}))
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 502
    assert SECRET not in body.text
    assert body.json()["detail"] == UNAVAILABLE_LITERAL


def test_a_3xx_is_a_failure_not_a_forwarded_body(monkeypatch, client):
    """A 3xx falls outside `ok`, so an unguarded `return response.json()` would
    hand the browser whatever the redirect body said."""
    monkeypatch.setattr(explain_mod.requests, "get", lambda url, **kw: _resp(302, {"detail": SECRET}))
    body = client.get("/api/explain/pl/1")
    assert body.status_code == 502
    assert SECRET not in body.text
    assert body.json()["detail"] == UNAVAILABLE_LITERAL


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
    assert body.json()["detail"] == UNAVAILABLE_LITERAL


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


def test_the_timeout_is_below_the_browsers():
    """The browser gives up at 20 s for this site. A sync proxy route holds a
    worker thread for its whole duration, so it must not wait longer than the
    client it serves — otherwise the slowest tier is the one nobody waits for.
    """
    assert explain_mod.EXPLAINER_TIMEOUT_S < 20


# --- comprehensive error status coverage (mirrors NBA) ---

#: Every non-2xx the proxy must turn into a 502. **The 3xx range is here for a
#: reason:** with only 400-and-up, `< 300` -> `<= 300` and `< 300` -> `< 400` were
#: both silent, and a 3xx carrying a JSON body would be handed to the browser as a
#: summary -- contradicting both the module docstring and this file's headline
#: claim. "The window is a range" is proved by testing below 200; where it ENDS is a
#: separate question, and only a 3xx asks it.
UPSTREAM_ERROR_STATUSES = [300, 301, 302, 303, 304, 307, 308, 399,
                          400, 401, 403, 404, 422, 429, 500, 502, 503]


@pytest.mark.parametrize("status", UPSTREAM_ERROR_STATUSES)
def test_an_upstream_error_status_becomes_a_502_carrying_no_upstream_body(monkeypatch, status):
    secret = "sk-or-v1-should-never-reach-a-browser"

    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = f'{{"detail": "{secret}", "path": "/srv/internal/{secret}"}}'.encode()
        return response  # returned, NOT raised

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)

    app = FastAPI()
    app.include_router(explain_mod.router)
    with TestClient(app) as c:
        res = c.get(f"/api/explain/pl/{GAME_ID}")

    assert res.status_code == 502, f"[upstream {status}] {res.text[:200]!r}"
    assert secret not in res.text, f"[upstream {status}] an upstream body was echoed"
    assert "/srv/internal" not in res.text, f"[upstream {status}] an upstream path was echoed"
    assert res.json() == {"detail": UNAVAILABLE_LITERAL}, res.text[:200]


@pytest.mark.parametrize("status", UPSTREAM_ERROR_STATUSES)
def test_a_raised_upstream_error_becomes_a_502_carrying_nothing(monkeypatch, status):
    """The other half, and it is a different branch. Real `requests` attaches the
    response to the exception, so the message is a candidate for leaking."""
    secret = "sk-or-v1-should-never-reach-a-browser"

    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = f'{{"detail": "{secret}"}}'.encode()
        raise requests.HTTPError(f"upstream {status}", response=response)

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)

    app = FastAPI()
    app.include_router(explain_mod.router)
    with TestClient(app) as c:
        res = c.get(f"/api/explain/pl/{GAME_ID}")

    assert res.status_code == 502, f"[upstream {status}] {res.text[:200]!r}"
    assert secret not in res.text, f"[upstream {status}] an upstream body was echoed"
    assert res.json() == {"detail": UNAVAILABLE_LITERAL}, res.text[:200]


def test_a_200_that_is_not_json_is_a_502(monkeypatch):
    """Something else is answering on the explainer's port, or it is
    misconfigured. Either way the site should show its retry state rather than a
    bare 500 it cannot interpret."""

    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b"<html>not json</html>"
        return response

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)

    app = FastAPI()
    app.include_router(explain_mod.router)
    with TestClient(app) as c:
        res = c.get(f"/api/explain/pl/{GAME_ID}")

    assert res.status_code == 502, res.text[:200]
    assert "not json" not in res.text


@pytest.mark.parametrize("status", [201, 202, 204, 299])
def test_any_2xx_is_forwarded_not_just_200(monkeypatch, status):
    """The window is a RANGE, and only 200 and non-2xx were exercised, so
    narrowing `200 <= status < 300` to `status == 200` left every test green.

    A 204 is worth calling out: it is a 2xx with no body, so `response.json()`
    raises and it takes the 502 path -- which is correct, because there is no
    summary to forward.
    """
    def fake_get(url, **kwargs):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(UPSTREAM_BODY).encode() if status != 204 else b""
        return response

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)

    app = FastAPI()
    app.include_router(explain_mod.router)
    with TestClient(app) as c:
        res = c.get(f"/api/explain/pl/{GAME_ID}")

    if status == 204:
        assert res.status_code == 502, res.text[:200]
    else:
        assert res.status_code == 200, f"[{status}] {res.text[:200]!r}"
        assert res.json() == UPSTREAM_BODY


def test_a_dead_explainer_is_a_502_not_a_500(monkeypatch):
    def fake_get(url, **kwargs):
        raise requests.ConnectionError("explainer is down")

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)

    app = FastAPI()
    app.include_router(explain_mod.router)
    with TestClient(app) as c:
        res = c.get(f"/api/explain/pl/{GAME_ID}")

    assert res.status_code == 502, res.text[:200]
    assert "explainer is down" not in res.text, "an internal reason leaked to the browser"


# --- the route is not a forwarder to anything on that host -----------------


#: Cases that must never reach the explainer. The list is built to defeat three
#: plausible weakenings of the guard:
#:
#: * dropping the `sport` half -- only `..%2Fadmin/x` puts a `..` into `sport`
#:   (it decodes to `sport=".."`, `explainer_id="admin/x"`), and it is the ONLY
#:   case that reaches that half;
#: * `in` -> `startswith` -- `secrets/../x` has the `..` INTERIOR to the segment,
#:   so a prefix check satisfies every other case;
#: * an absolute-path id -- `nba//etc/passwd` contains no `..` at all, so a
#:   walk-out check alone forwards it.
WALK_OUT_PATHS = [
    ("/api/explain/pl/../../secrets", False),
    ("/api/explain/../admin/x", False),
    ("/api/explain/..%2Fadmin/x", True),
    ("/api/explain/pl/%2e%2e/%2e%2e/etc/passwd", True),
    ("/api/explain/pl/secrets%2F..%2Fx", True),
    ("/api/explain/pl/secrets/../x", False),
    ("/api/explain/pl//etc/passwd", True),
    ("/api/explain/pl/%2Fetc%2Fpasswd", True),
]


@pytest.mark.parametrize(("path", "must_refuse"), WALK_OUT_PATHS,
                         ids=[f"{p}|refuse={r}" for p, r in WALK_OUT_PATHS])
def test_a_path_that_walks_out_never_reaches_the_explainer(monkeypatch, path, must_refuse):
    sent: list[str] = []

    def fake_get(url, **kwargs):
        sent.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"verdict": "leaked"}'
        return response

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)

    app = FastAPI()
    app.include_router(explain_mod.router)
    with TestClient(app) as c:
        res = c.get(path)

    if must_refuse:
        assert not sent, (
            f"[{path}] reached the wire but should have been refused before any "
            f"request: {sent}"
        )
        assert res.json() == {"detail": UNAVAILABLE_LITERAL}, (
            f"[{path}] the refusal did not use the fixed message: {res.text[:200]!r}"
        )
    for url in sent:
        assert url.startswith(f"{explain_mod.EXPLAINER_URL}/explain/"), (
            f"[{path}] a request escaped the explainer's route: {url}"
        )
        assert ".." not in url, f"[{path}] a traversal reached the wire: {url}"
        tail = url[len(f"{explain_mod.EXPLAINER_URL}/explain/"):]
        assert tail.startswith("pl/"), f"[{path}] wrong sport segment: {url}"
        assert not tail[len("pl/"):].startswith("/"), (
            f"[{path}] an absolute-path id was forwarded: {url}"
        )


def test_a_2xx_that_is_not_an_object_is_a_502(monkeypatch):
    """`response.json()` can return a list, a string, a number or `null`.

    Checking here makes it the same 502 as every other unusable answer.
    """
    for payload in (b"[1, 2, 3]", b'"just a string"', b"null", b"7"):
        def fake_get(url, _payload=payload, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response._content = _payload
            return response

        monkeypatch.setattr(explain_mod.requests, "get", fake_get)
        app = FastAPI()
        app.include_router(explain_mod.router)
        with TestClient(app) as c:
            res = c.get(f"/api/explain/pl/{GAME_ID}")
        assert res.status_code == 502, f"[{payload!r}] {res.status_code}: {res.text[:200]!r}"
        assert payload.decode() not in res.text or payload == b"null", res.text[:200]


def test_a_slash_in_the_sport_is_refused(monkeypatch):
    """The one character that changes the URL's STRUCTURE rather than its content.

    `quote`'s default is `safe='/'`, so `quote(sport)` would leave a slash intact
    and the sport segment would become two segments.
    """
    sent: list[str] = []

    def fake_get(url, **kwargs):
        sent.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(UPSTREAM_BODY).encode()
        return response

    monkeypatch.setattr(explain_mod.requests, "get", fake_get)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        explain_mod.explain("pl/x", GAME_ID)
    assert caught.value.status_code == 502
    assert not sent, f"a request was sent for a refused sport: {sent}"


def test_an_empty_id_is_refused():
    """An empty id is not one relative segment."""
    from fastapi import HTTPException
    sent: list[str] = []

    def fake_get(url, **kwargs):
        sent.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(UPSTREAM_BODY).encode()
        return response

    m = pytest.MonkeyPatch()
    m.setattr(explain_mod.requests, "get", fake_get)
    try:
        with pytest.raises(HTTPException) as caught:
            explain_mod.explain("pl", "")
        assert caught.value.status_code == 502
        assert not sent, f"a request was sent for an empty id: {sent}"
    finally:
        m.undo()


def test_the_timeout_helper_survives_a_non_string():
    """`_positive_float` must handle non-strings without taking the app down."""
    assert explain_mod._positive_float(None, "X") == explain_mod._DEFAULT_TIMEOUT_S
    assert explain_mod._positive_float(float("nan"), "X") == explain_mod._DEFAULT_TIMEOUT_S
    assert explain_mod._positive_float(float("inf"), "X") == explain_mod._DEFAULT_TIMEOUT_S
    assert explain_mod._positive_float("abc", "X") == explain_mod._DEFAULT_TIMEOUT_S


def test_an_unparseable_timeout_does_not_take_the_app_down(monkeypatch):
    """`float(os.getenv(...))` at import raises on a typo, so it must fall back."""
    for bad in ("", "abc", "0", "-1", "600", "1e400"):
        monkeypatch.setenv("EXPLAINER_TIMEOUT_S", bad)
        try:
            reloaded = importlib.reload(explain_mod)
            assert reloaded.EXPLAINER_TIMEOUT_S == explain_mod._DEFAULT_TIMEOUT_S, (
                f"EXPLAINER_TIMEOUT_S={bad!r} resolved to "
                f"{reloaded.EXPLAINER_TIMEOUT_S}, expected the default "
                f"{explain_mod._DEFAULT_TIMEOUT_S}"
            )
        finally:
            monkeypatch.delenv("EXPLAINER_TIMEOUT_S", raising=False)
            importlib.reload(explain_mod)


def test_the_explainer_url_is_read_from_the_environment(monkeypatch):
    """Read at *import*, not per request."""
    monkeypatch.setenv("EXPLAINER_URL", "http://elsewhere.test:9999/")
    reloaded = importlib.reload(explain_mod)
    try:
        assert reloaded.EXPLAINER_URL == "http://elsewhere.test:9999", (
            f"EXPLAINER_URL={reloaded.EXPLAINER_URL!r}; trailing slash should "
            f"be stripped"
        )
    finally:
        monkeypatch.delenv("EXPLAINER_URL", raising=False)
        importlib.reload(explain_mod)